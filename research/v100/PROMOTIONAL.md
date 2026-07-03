# Promotional Material

## Short Positioning

PLI Labs publishes practical owned-hardware research for local legal AI. The
V100 work shows how to evaluate older enterprise GPUs without pretending that
model-load success is production readiness.

## Longer Positioning

Many legal teams can repurpose owned hardware before they can justify premium
cloud inference or new data-center accelerators. V100-class systems are not
modern, but they remain useful when the model stack respects their limits.

PLI Labs summarizes public lessons by hardware class, backend family, model
class, context tier, and rounded performance band. Private topology, recovery
steps, launch details, and raw failure logs stay internal.

## Public Claims

- V100-class hardware can still run useful local AI workloads.
- Backend validation matters more than headline VRAM.
- Loading a huge model is not proof that inference works.
- Public numbers should separate measured results from inferred or planned
  experiments.
- Legal workloads need validation on realistic long-form prompts, not only
  toy completions.

## Suggested Post Copy

PLI Labs published public-safe V100 research notes for owned-hardware legal AI.
The short version: older enterprise GPUs are still useful, but only when the
backend, quantization path, context tier, and workload are tested together.

## Links

- https://proprietarylegal.com
- https://proprietarylegal.ai

## July 2026 optimization sprint (copy block)

One session on owned V100 hardware, every number measured with full
provenance: 36.7–86.6x faster time-to-first-token for repeated legal system
prompts, a 2x hidden KV-cache cliff removed with one build flag, a
previously-unservable 27B hybrid model brought up on a single 32GB card via
a published patch, 2.56–4.5x decode from CUDA graphs on hardware the
ecosystem had written off (a 122B MoE taken from a measured 39.7 t/s stock
baseline to 67.6 t/s, via the first known tensor-parallel-4 graph capture
on sm_70), and a ~10x per-GPU lane win from model-level analysis — plus the
negative results that keep a production fleet honest: the driver-wedge ban,
and the finding that our own tuned build regresses single-card dense
prefill by 35% (recipes are shape-dependent; we publish that too). Full
tables: research/v100/serving-optimization-2026-07/.
