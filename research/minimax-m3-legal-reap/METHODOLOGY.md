# How the MiniMax-M3 Legal REAP 0.22 Artifact Was Produced

This document describes the engineering method behind the artifact at a public-safe
level: how a ~428B-parameter model was observed and pruned on hardware far too small
to hold it resident, how long-context native sparse attention was handled, and how the
pruned checkpoint was serialized without ever loading the model. Hostnames, IP
addresses, ports, local file paths, and private data contents are intentionally
omitted; hardware is described by class only.

## The core constraint

MiniMax-M3 is a native multimodal MoE with ~428B total parameters (~23B active),
60 decoder layers (0–2 dense, 3–59 MoE with 128 routed + 1 shared expert, top-4
sigmoid routing), MiniMax Sparse Attention (MSA), and a 1M-token context. In bf16 the
checkpoint is on the order of 0.8 TB on disk. No single node in the working fleet — a
pair of **NVIDIA GB10 unified-memory systems (DGX Spark class), ~128 GB unified memory
each** — can hold it resident. Two facts follow: the model must be processed
**one decoder block at a time** (a "conveyor"), and quantizing to fit is both
unnecessary (streaming fits at ~15 GB per block) and unsafe for this measurement
(post-training quantization flips a meaningful fraction of top-k expert selections,
which is exactly the quantity REAP measures). The whole pipeline therefore runs in
bf16.

## Stage 1 — Observation on a two-node streaming conveyor

The REAP saliency signal is gathered by running the legal calibration corpus through
the model and recording, per expert, the router-gated activation statistics. Because
the model cannot be resident, this uses a **streaming conveyor observer**: each decoder
block is materialized on the accelerator on demand (empty-weights construction plus
per-block weight load, ~14.5 GB peak), run over the batch, and released; boundary
activations are cached and replayed to the next block. A built-in shadow check asserts
that the streamed path is numerically identical to a resident forward, and regression
tests pin conveyor-equals-sequential.

Work was split across the two nodes by **disjoint calibration components** (each node
owns a disjoint set of role/length-bucket cells). This needs no new sharding code: the
per-cell observations merge through the existing observation-manifest machinery and are
byte-structurally identical to a single-node run, so the downstream prune gates cannot
tell the difference. The calibration spans 13 length-bucketed components (975 rows),
with context tiers up to 16,384 tokens.

### Handling MiniMax Sparse Attention at long context

MSA is not dense attention. On the MoE layers a learned indexer selects, per query, a
sparse set of key/value blocks (top-16 blocks of 128 tokens). Two subtleties had to be
resolved to keep the observation faithful to how the base model actually computes:

1. **Short vs long context.** Top-16 × 128-token blocks cover every causally eligible
   block only up to ~2,048 tokens. At or below that threshold, sparse selection is
   equivalent to dense causal attention and a dense math kernel is provably identical
   (pinned by a CPU regression that runs the real indexer/attention classes). Above
   2,048 tokens the sparse selection genuinely discards blocks, so the long-context
   route must run the **native sparse** attention path — the same computation the
   reference model performs. The observer resolves this per component from the context
   tier and refuses any conflicting override.

2. **The long-context memory cliff (and its fix).** At 16,384 tokens the first sparse
   layer builds an explicit additive attention mask. A naive `[1, heads, S, S]`
   materialization at 64 heads and S = 16,384 is on the order of tens of GB of
   transient memory — fatal on a unified-memory box, where an overrun is an
   uncatchable OS-level out-of-memory event rather than a recoverable allocator error.
   The fix is a **head-chunked native-sparse attention implementation**: it rebuilds
   the *exact* native sparse mask by calling the modeling's own block-mask builder
   (never a reimplementation of the math), then loops over small chunks of attention
   heads, computing scaled-dot-product attention per chunk and concatenating. Heads are
   independent, so each chunk's output is bit-identical to the full-head kernel, while
   the transient is capped at `chunk × S²` instead of `heads × S²`. A further
   refinement builds the mask **per index-head group** lazily (the indexer's four
   groups select different blocks and are not collapsible), holding peak mask memory to
   about 1 GB. Equivalence to the full-head path is pinned by tests and re-attested on
   the actual accelerator. The already-completed shorter context tiers are unaffected
   and were not recomputed.

Defense-in-depth accompanied the fix: a per-process CUDA memory cap that converts a
runaway allocation into a catchable error with a traceback, phase/memory telemetry to
localize any blow-up, and cgroup memory limits so a failure stays contained to the
worker instead of taking down the node. Both nodes ran under boot-persistent
supervisors with watchdogs and periodic artifact sync-back; resume is checkpointed at
block granularity with a content-based fingerprint, so an interrupted run resumes
without recomputation and refuses to mix incompatible configurations.

## Stage 2 — Model-free cut-plan computation

The decision of *which* experts to drop is a pure function of the observation tensors
plus the schedule knobs, so it is computed with **no model resident**, reusing the
reference pruner's own selection primitives (imported, not reimplemented) so the result
is bit-for-bit the cut a resident prune would have produced:

1. Merge per-role observations with a role-weighted recombination emphasizing legal
   drafting.
2. Compute a **per-layer adaptive** retained-expert schedule at overall ratio 0.22,
   under capability floors (no under-covered layer, no low-confidence carrier cut
   below a safe count). Dense layers 0–2 are read from the config and skipped — never
   hard-coded.
3. Per MoE layer, take the lowest-saliency experts (`topk`, largest=False) down to the
   layer's retained count (89–111 of 128), optionally preserving designated
   super-experts.

The result is a cut plan with a stable, order-independent **cut-plan hash** over the
per-layer retained-expert sets, which is written into the checkpoint's provenance.

## Stage 3 — Streaming split-to-split serialization

The reference checkpoint saver gathers a full live state dict from a resident, hooked
model — it fundamentally needs the model loaded. This artifact instead uses a
**streaming split-to-split serializer** that rewrites the on-disk shards directly and
never loads the model. It works because the on-disk MiniMax-M3 checkpoint stores each
expert as independent split tensors (the "original" MiniMax layout); the modeling code
fuses them at load and un-fuses them at save. By reading and writing the split layout
directly, the serializer never materializes the packed representation and touches only
about one shard buffer plus one tensor of peak memory.

Per tensor, the transform is:

- **drop** the vision tower, multimodal projector, and patch-merge stack (text-only
  output);
- **rename** the VL text-stack keys to the text-only layout (the same remap the
  text-only load performs);
- **retained expert:** copy byte-identical, renumbering the expert id to its dense
  position in the sorted retained set;
- **dropped expert:** omit;
- **router weight and routing-bias rows:** `index_select` the retained rows (REAP's
  router "renorm" is a row slice — no numeric renormalization);
- **everything else** (dense MLP, shared expert, attention including the MSA indexer,
  layernorms, embeddings, final norm, LM head): copy byte-identical.

Shards are written greedily to a target size, then finalized into a contiguous
`model-XXXXX-of-NNNNN` set with a matching safetensors index. The serializer fails loud
on any decoder-layer tensor with an unrecognized prefix, any expert tensor on a
supposedly-dense layer, any router tensor for an unplanned layer, or a duplicate output
key — it never silently drops a decoder tensor. Alongside the shards it writes a
provenance sidecar and an expert-count census.

See [`STREAMING-SERIALIZE-PSEUDOCODE.md`](STREAMING-SERIALIZE-PSEUDOCODE.md) for the
sanitized pseudocode.

## Why the source itself is not published here

The production implementation of the observer, cut-plan, and serializer imports
internal pruning-engine primitives and profile machinery, so it is not cleanly
extractable into a self-contained public module. Rather than publish a fragment that
would not build, this bundle documents the method and ships pseudocode for the
model-free serializer — the component with the most reuse value and the fewest internal
dependencies.
