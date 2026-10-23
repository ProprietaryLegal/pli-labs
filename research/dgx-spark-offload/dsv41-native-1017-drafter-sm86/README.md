# Native DeepSeek-V4.1-Flash-1017 drafter on RTX 3090 (SM86): official MXFP4/FP8 weights, custom kernels, NVLink TP2

**Source code and results, not model weights.** One of the separately usable pieces from PLI Labs' DeepSeek-V4.1-Flash
deployment on two DGX Spark (GB10) nodes with an x86 offload host. Overview:
[DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).

**Status (2026-10-23), stated plainly:**
- **It runs, and on the real NVLink pair its fastest arm drafts 2–12 % faster per step** than the EXL3-quantized drafter we
  serve today. Measured acceptance is the same within noise.
- **It is NOT yet claimed to be source-faithful.** Our strict fidelity gate currently **fails** for the native arms. The
  attention computation they inherited from our worker does its softmax in one pass, while DeepSeek's official kernel
  works in online 64-key blocks with different rounding points.
- **A fix is in progress.** It may change the numbers below, including the latency.
- **Not serving in production.**

Hugging Face entry: https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86

## What it is

DeepSeek-V4.1-Flash-1017 ships its own drafter: the multi-token-prediction stages `mtp.0–2`, a Markov head and a
confidence head (2,401 tensors). It stores the routed experts in MXFP4 and the dense matrices in FP8 E4M3 with E8M0 32×32
block scales. It shares the target's BF16 embedding and head.

Our production drafter runs a re-quantized EXL3 copy of it on an RTX 3090
([remote drafter entry](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090)). This entry runs **the
official weight values themselves** on two NVLinked RTX 3090s. SM86 has no FP8 or FP4 tensor cores, so the formats are
decoded inside custom kernels:
- the activation quantize-dequantize the model was trained with is kept before every FP8/FP4 linear;
- routers and biases stay FP32;
- BF16 rounding happens at the official points;
- in the official linears, only the FP32 summation order differs. Attention is the exception, as described in the status.

**Arms** (selectable; all use the deployed K6 draft head unless named):

| arm | experts | dense | notes |
|---|---|---|---|
| `native-packed` | official MXFP4 bytes, decoded in the GEMM | official FP8 bytes | smallest memory |
| `native-expanded` | decoded once to BF16 (exact) | decoded to BF16 (exact) | largest memory (~16 GiB/card at load) |
| `native-hybrid-kx` | register-fed MXFP4 kernels (`kernels/mxfp4`, `kernels/grouped`) | BF16 | first fast arm |
| **`native-hybrid-kx2`** | fused MXFP4 MoE kernel (`kernels/fusedmoe`), per-shape variant table | FP8 kernel (`kernels/fp8dense`) only where measured faster (M5/M10/M15, main_proj M8); BF16 elsewhere; paired BF16 shared-expert projection at T20 | exact fused epilogues; fastest |

## Numerical evidence (kx2, complete two-rank emulation, stages 0–2, T5/T10/T15/T20)

| check | result | gate |
|---|---|---|
| focused component and integration suite | 171 passed | all pass |
| CUDA graph vs eager, B1–B4 × n_out 4/5 | 8 / 8 exact IDs and confidence; both ranks agree | exact |
| expert projections vs our first native implementation | 12 BF16 bit differences in 2,073,600; worst 99.9942 %; all 12 match FP64 rounding on our side | ≥ 99.99 % |
| shared and post-TP epilogues vs the unfused operations | 48 / 48 bit-exact, signed zero included | exact |
| routed expert output vs FP64 | worst relative norm 3.39e-8 | ≤ 1e-6 |
| selected FP8 dense vs our first packed-FP8 implementation | worst 99.9072 % BF16 agreement | ≥ 99.9 % |
| FP32 dense partials vs FP64 | worst relative norm 7.71e-8 | ≤ 1e-6 |

These gates are normwise; a pointwise 1e-6 claim would be false near cancellation, and we don't make it. Aggregate
receipts: `code/offload/dspark/native/tools/n2/evidence/`.

## Latency

**On the real NVLink pair** (2026-10-23; drafter step p50, µs; k = 5; 200 samples per case after 0.5 s settling; both
cards capped at 325 W):

| arm | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| deployed EXL3 drafter, TP2 (K2), n_out 5 | 2,028 | 2,512 | 2,896 | 3,722 |
| native-hybrid-kx, n_out 5 | 2,246 | 2,804 | 3,151 | 3,737 |
| **native-hybrid-kx2, n_out 5** | **1,782 (−12 %)** | **2,457 (−2 %)** | **2,821 (−3 %)** | **3,447 (−7 %)** |
| deployed EXL3 drafter, TP2, n_out 4 | 2,000 | 2,443 | 2,820 | 3,280 |
| native-hybrid-kx2, n_out 4 | 1,825 (−8.7 %) | 2,442 (−0.1 %) | 2,812 (−0.3 %) | 3,099 (−5.5 %) |

**Per card, emulated TP2 on one 3090** (median of three interleaved settled runs, µs, n_out 5):

| rank | arm | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|
| 0 | deployed EXL3 (K2) | 1,734.7 | 2,225.2 | 2,503.7 | 3,298.8 |
| 0 | native-hybrid-kx2 | 1,650.7 (−4.8 %) | 2,070.5 (−7.0 %) | 2,347.0 (−6.3 %) | 3,076.1 (−6.8 %) |
| 1 | deployed EXL3 (K2) | 1,815.6 | 2,317.8 | 2,658.2 | 3,365.3 |
| 1 | native-hybrid-kx2 | 1,753.0 (−3.5 %) | 2,184.8 (−5.7 %) | 2,380.4 (−10.5 %) | 3,176.0 (−5.6 %) |

**What changed between the two native arms:** FP8 dense where it measured faster, and exact fused epilogues. They saved
about 300 µs of dense work and about 140 µs of epilogue work per rank at B1. Tuning attempts that did not help:
- register tile and split sweeps;
- an alternative expert-kernel mode at T15, which was slower (250 µs against 216 µs);
- FP8 for the 128-row main projection: 563 µs against 268 µs in BF16.

## Acceptance (realized, on live captured rounds)

453 captured production rounds (542 live blocks), replaying each arm's drafts against the target's committed tokens:

| drafter | accepted per round |
|---|---|
| the engine's own drafter during capture | 1.810 |
| deployed EXL3 drafter (K2), replayed | 1.801 |
| native-hybrid-kx | 1.823 |
| native-hybrid-kx2 | 1.807 |

Equal within noise (standard errors ~0.02–0.03). The native arms' drafts differ block-by-block from the EXL3 drafter's
(they are different weight representations), but they are accepted at the same rate. This is not an acceptance gain.

## Memory (per card, real pair)

| arm | load time | load peak (allocated) | serving (nvidia-smi) |
|---|---|---|---|
| deployed EXL3 drafter (K2) | 61 s | 4,604 / 4,598 MiB | 5,214 / 5,004 MiB |
| native-hybrid-kx | 332 s | 6,833 / 6,827 MiB | 7,584 / 7,540 MiB |
| **native-hybrid-kx2** | **330 s** | **7,348 / 7,344 MiB** | **8,090 / 7,926 MiB** |

Load time is dominated by native admission and prepacking. A warm file cache brings it to ~160–170 s per rank in
single-card runs.

## Fidelity status (why "source-faithful" is not claimed yet)

Our strict check replays captured rounds and compares every internal state against an FP32 reference built from the
official source. That covers the KV window, each stage's attention and FFN outputs, the head's hidden state and logits,
routing, drafts and confidence. It classifies every disagreement.

**Result:** the native arms currently **fail its entry gates**, both before and after the first fix attempt:
- the first stage's attention error is hundreds of times the summation-order band;
- KL is 3–5× that band;
- before the fix, head self-check flags also fired.

**Cause found so far:** our worker's attention, inherited from earlier work, computes the softmax in a single pass, with
the attention sink inside the maximum and one BF16 probability rounding. DeepSeek's official kernel processes online
64-key blocks, rounds probabilities to BF16 per block, and adds the sink after the loop. On a synthetic test with
identical inputs, 38,799 of 81,920 BF16 outputs differ (relative norm 0.0030).

**Fix in progress.** A lane has implemented the official online-64 attention and FP32 head and Markov arithmetic. The
first corrected pilot removes the state flags, but the gate still fails, and the investigation continues. Until the gate
passes, treat these arms as **fast and acceptance-equal, but not yet source-faithful**.

## When to use it, and when not

- **Use it** to study or serve the model maker's own drafter values on consumer Ampere cards, or as a reference for
  decoding MXFP4/FP8 on SM86 without tensor-core FP8.
- **Don't use it** as a drop-in production drafter yet: the fidelity gate is open, the serving integration is still being
  qualified, and it needs about 8 GB per card against about 5 GB for the EXL3 drafter.

## Requirements and use

CPU suite in this published copy: 11 passed, 17 skipped (GPU and weight-shard tests skip without them).


1. **Hardware:** two RTX 3090s with an NV4 bridge, peer access enabled, about 8 GiB free per card.
2. **Weights:** the official shards from `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
   - Get the draft shards 44–46, the embed shard 2 and the head shard 43 with `code/offload/dspark/native/fetch_official.sh`
     and `fetch_official_shared.sh`. Set the download directory inside each script first. Both pin a revision and check sha256.
   - Every tensor's format, bytes and sha256 are listed in `NATIVE_MANIFEST.json`.
3. **Benchmark:** from `code/`, run `PYTHONPATH=. python -m offload.dspark.native.tools.native_bench <arm> pair <out.json> 0 200`.
   This layout keeps the `offload/` package level, because the code imports `offload.dspark.*`.
4. **Parity:** `native_parity`, with `native_graphcheck` for graph vs eager. Design and official semantics:
   `code/offload/dspark/native/NATIVE.md`.
5. **Kernels:** `code/offload/dspark/kernels/{fusedmoe,fp8dense,mxfp4,grouped}/`, each with its README, oracle,
   qualification script and tests.
6. **Two-card plumbing:** the worker package from the
   [two-card entry](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) is included under
   `code/offload/dspark/worker/`.

## Related entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
  [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)
- **Separate:** [remote 3090 drafter + KV](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) ·
  [two-card NVLink drafter](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) ·
  [async hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) ·
  [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) ·
  [ofl transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) ·
  [bench + interference tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

## Credits and licence

- Model: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT); its
  official inference code is the arithmetic reference.
- The MXFP4 register-fed kernels adapt portions of [ro99/strata](https://github.com/ro99/strata) (Apache-2.0), with its
  licence and our changes in each kernel's `NOTICE`.
- EXL3 decoding for the K6 head and the control arm is from [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT).

Our code is Apache-2.0. Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
