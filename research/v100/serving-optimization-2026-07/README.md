# V100 Serving Optimization Sprint — Case Study (July 2026)

**Proprietary Legal Intelligence / PLI Labs**

One engineering session on a 15-GPU Tesla V100-SXM2-32GB fleet (sm_70, NVLink
board topology) serving a production legal-drafting workload. Every number
below is a real measurement from that session — 3-run medians with recorded
model file, binary git SHA, build flags, sampling parameters, and GPU set. No
extrapolations. Where a hypothesis failed, the negative result is published
with the same prominence as the wins.

Target models: Qwen 3.5/3.6 (122B-A10B MoE, 27B GDN hybrid) and Gemma 4
(31B dense, 26B-A4B MoE) — the model families that actually run the
practice's drafting, review, extraction, and reasoning lanes.

---

## Headline results

| Optimization | Before | After | Gain |
|---|---|---|---|
| Prefix caching, 6k-token system prompt (Gemma-4-31B) | 42.0 s prompt processing | 1.0–1.2 s | **36.7x** |
| Prefix caching, 12k-token system prompt | 93.1 s | 1.4–1.6 s | **61.7x** |
| Prefix caching, 6k prefix (Gemma-4-26B-A4B, full-fit single GPU) | 3.36 s | **39 ms** | **86.6x** |
| Mixed-type KV-cache prefill (missing FA kernels) | 79–94 t/s | 154–161 t/s | **~2x cliff removed** |
| Qwen3.6-27B (GDN hybrid) on one 32GB V100 | did not fit (OOM) | 27.5 t/s decode | **unservable → served** |
| vLLM decode via CUDA graphs (sm_70, 4B proof model) | 51.6 t/s eager | 132.8 t/s | **2.56x** |
| vLLM Qwen3.5-122B, tensor-parallel-4 + CUDA graphs | 14.8 t/s eager | **67.6 t/s** | **4.5x** |
| Gemma-4 lane throughput per GPU (lane redesign†) | 39.2 t/s on 4 GPUs (~10 t/s/GPU) | 101.3 t/s on 1 GPU | **~10x per GPU†** |
| Qwen3.5-122B-A10B vs stock Ollama baseline | **39.7 t/s (3-run measured)** | **61.8 t/s** llama.cpp / **67.6 t/s** vLLM | **1.56x / 1.70x** (prefill ~19x) |

All decode figures are single-stream generation tokens/sec at the stated
context unless noted. †The Gemma-lane figure compares different same-family
checkpoints (dense 31B vs 26B-A4B MoE) — a deployment redesign, not a
same-model speedup; throughput is measured on both sides, and the task-level
quality-parity evaluation is still in progress. Aggregate-throughput and prefill figures are called out
separately in each section. Honesty note: the tuned multi-GPU build recipe
is shape-dependent — on a single-card full-fit *dense* model it **regresses**
prefill by 35% (see the model matrix below); stock build + f16 KV wins there.

---

## 1. Prefix caching for repeated system prompts (36.7–61.7x)

Legal drafting harnesses re-send large role contracts and formatting spines
with every request: a 6,000–12,000-token stable prefix and a short varying
tail. llama.cpp's `--cache-reuse` makes the server reuse the KV cache for the
common prefix and reprocess only the tail.

**Measured (Gemma-4-31B dense, single V100-32GB, llama.cpp build 9860):**

| Prefix | Cold prompt processing | Warm (cache hit) | Tokens reprocessed | Speedup |
|---|---|---|---|---|
| ~6,045 tokens | 41,831 / 42,145 / 42,167 ms | 1,234 / 1,002 / 1,199 ms | 11 | **36.7x mean** |
| ~12,030 tokens | 93,155 / 93,082 ms | 1,409 / 1,607 ms | 11 | **61.7x mean** |

The benefit *grows* with prefix length — exactly the scaling a long-system-
prompt legal workload wants.

**The gotcha that makes this a consulting finding:** Gemma-4 is a
sliding-window-attention (SWA) model. With `--cache-reuse` alone the server
silently logs `cache_reuse is not supported by this context, it will be
disabled` and you get nothing. The flag pair that works is:

```
--cache-reuse 256 --swa-full
```

Even with no flags at all, the longest-common-prefix slot cache still
delivered 9.1x on append-only prompts — but the full flag pair is what
unlocks the order-of-magnitude wins above. The follow-up matrix across the
other production models (86.6x on the single-card MoE lane; why the flag
pair is a NO-OP on the hybrid 122B) is in "Context optimizations,
before/after" below.

## 2. KV-cache quantization kernels: removing a hidden 2x cliff

Quantized KV caches (q8_0, q5_1, q4_0) buy 2–2.9x more context per GB of
VRAM. On V100, llama.cpp's default build only instantiates FlashAttention
kernels for a subset of K/V type combinations. Any *mixed* combination falls
off a silent performance cliff.

**Measured (Gemma-4-31B, single V100, pp512, identical git SHA, only the
build flag differs):**

| K/V types | Default build | `GGML_CUDA_FA_ALL_QUANTS=ON` build |
|---|---|---|
| f16/f16 | 161.4 t/s | 161.5 t/s (parity) |
| q8_0/q8_0 | 161.0 t/s | 161.2 t/s (parity) |
| **f16/q8_0 (mixed)** | **79.4 t/s (−51%)** | **161.3 t/s (flat)** |
| **q8_0/f16 (mixed)** | **94.2 t/s (−42%)** | **161.4 t/s (flat)** |

With the full kernel set, all 16 K/V combinations hold 154–161 t/s.
Deployment rules that fall out of the data:

- q8_0/q8_0 = 2x KV compression at **zero** measured prefill cost.
- q5_1/q5_1 = ~2.9x compression for ~4% prefill cost.
- Build with `GGML_CUDA_FA_ALL_QUANTS=ON` before serving any non-f16 KV
  configuration on Volta.

**Context-depth behavior (q8_0 KV, KV depth 0 → 32,768 tokens):**

| KV depth | Prefill (pp512) | Generation (tg128) |
|---|---|---|
| 0 | 161.6 t/s | 4.65 t/s* |
| 4,096 | 153.4 t/s (−5%) | 3.88 t/s |
| 16,384 | 141.0 t/s (−13%) | 3.08 t/s |
| 32,768 | 126.1 t/s (−22%) | 2.13 t/s (−54%) |

*Generation absolutes in this table are partial-offload (the 31B does not
fully fit one 32GB card); the relative curve is the finding. Prefill
degrades gracefully; generation cost is where long context bites — which is
precisely what prefix caching (section 1) eliminates for warm requests.

## 3. Serving a GDN-hybrid Qwen3.6-27B on a single V100 (OOM → 27.8 t/s)

Qwen 3.5/3.6 hybrid models interleave GDN linear-attention layers with full
attention. Stock LMDeploy/TurboMind 0.14.0 cannot serve the AWQ 27B on a
32GB card: its weight builder resolves a mixed-format fused projection by
dequantizing *everything* to fp16 — two 0.94 MB fp16 matrices per layer drag
42 MB of W4 weights up to 161 MB fp16, across 48 layers.

**The patch** (published in `patches/deltanet_awq.patch`): instead of
dequantizing the W4 majority, requantize the two tiny fp16 matrices into the
AWQ format (RTN, group-128, MSE-optimal clip search). The checkpoint's
existing AWQ weights are committed **bit-exactly** (round-trip max error 0.0,
verified on real tensors).

| Metric | Stock 0.14.0 | Patched |
|---|---|---|
| Committed engine weights | 25.20 GiB (+ cache → OOM on 32GB) | **19.78 GiB** |
| Load on one V100-32GB | fails at layer 63/64 | **loads, 26.2 GiB steady** |
| Single-stream decode | — | **27.5 t/s** |
| Prefill (4k prompt) | — | **1,093 t/s** |
| Aggregate @8 streams | — | **114–185 t/s** (clock-dependent) |
| Output quality | — | coherent, reasoning chains intact; int4 requant of the gate/decay projections caused **no** measurable degradation |

## 4. CUDA graphs on sm_70 vLLM: 2.56x decode

Conventional wisdom holds CUDA-graph capture is unreliable on Volta serving
stacks; the production default was `--enforce-eager`. Measured on a
V100-tuned vLLM fork (v1.2.1 class, FLASH_ATTN_V100 paged-attention backend,
fp16), Qwen3-4B-AWQ, single V100:

| Mode | Decode (3x128, greedy) | Prefill (5,281 tokens) |
|---|---|---|
| Eager | 51.1 / 51.6 / 51.8 t/s | 3,836 t/s |
| CUDA graphs (FULL_AND_PIECEWISE) | **131.3 / 132.9 / 132.8 t/s** | 3,796 t/s |

Capture cost 14 s and 0.75 GiB once at startup; output was **bit-identical**
to eager. Decode improves 2.56x; prefill is unchanged (graphs remove launch
overhead, which only dominates at generation batch sizes).

**The production-scale result:** the same experiment at tensor-parallel-4 on
Qwen3.5-122B-A10B (the first known TP4 graph capture on sm_70 — 57 s, 0.21 GiB
per rank, no crash): eager 14.8 t/s → **graphs 67.0–67.6 t/s single-stream
(4.5x)**, 3,395 t/s prefill, 66.7 t/s aggregate @4 streams. That beats the
tuned llama.cpp recipe (61.8 t/s, 3-run verified) by ~9% single-stream and
6.7x on prefill — and it means the conventional "eager-only on Volta" policy
was itself the bottleneck. Follow-up validation extended the lane to a
**65,536-token context window** (TP4 graph capture at 64k intact; 27k- and
62k-token buried-fact recalls both correct; 663k tokens of KV headroom at
0.90 memory utilization), and a task-level quality gate cleared it for
production: correct on statute citation, arithmetic, strict-JSON extraction,
format compliance, and both long recalls, with one flagged suspect case
citation — the standard cite-check discipline for any LLM legal output. One honesty note: on this hybrid-MoE architecture,
graph output is coherent but *not* bit-identical to eager (it was bit-identical
on the dense proof model) — quality comparisons should run graphs-vs-graphs.

Getting to this number also surfaced three real build/packaging defects in
the fork (a torch-version gate that silently drops the entire generic op
library, a stale bundled attention module, and a device-visibility parser
that rejects UUID-form pinning). All three are documented with fixes and
upstreamed — see the "1CatAI vLLM fork: before/after our fixes" section
below. This is the kind of issue that quietly costs teams weeks.

## 5. Model-level optimization: MoE for the Gemma lane (~10x per GPU)

The biggest single win wasn't a kernel — it was recognizing that a
same-family MoE checkpoint dominates the dense deployment for
throughput-bound lanes. This is a lane redesign (different checkpoint),
not a same-model optimization — quality gates passed spot checks and the
full task-level evaluation is the promotion gate:

| Gemma-4 lane option | Hardware | Generation | Prefill | Per-GPU |
|---|---|---|---|---|
| 31B dense, validated best recipe | 4x V100 board | 39.2 t/s | 1,602 t/s | ~10 t/s |
| **26B-A4B MoE (4B active), full-fit** | **1x V100** | **101.3 t/s** | **1,846 t/s** | **101 t/s** |

At 16k tokens of KV depth the MoE still generates at 93.4 t/s (−8%,
graceful). Quality gate: correct facts and clean reasoning at temperature 0
through the model's thinking-channel template. Recommended as the speed
option for reviewer/drafter roles, with task-level evals as the promotion
gate.

## 6. Negative results (published on purpose)

- **GDN kernels are not the hybrid's bottleneck.** nsys profiling showed the
  entire GDN chain is 1.7% of the per-token budget. The "obvious" fusion
  project would have wasted weeks. Profile first.
- **GEMM tile-dispatch tuning had zero end-to-end effect** at decode batch
  sizes for the 27B hybrid: cold-heuristic vs tuned vs imported dispatch
  caches all measured 27.8 t/s within noise. What *did* move aggregate
  throughput was **thermals**: the same @8-stream workload measured 185 t/s
  at 59–65 °C and 114 t/s on a hot card stepping down clocks. On dense
  hardware fleets, cooling is a throughput feature.
- **Tensor-split mode with ≥8K prefill reproducibly wedges the Volta
  driver** (D-state, reboot-only recovery, confirmed twice). Long-context
  work belongs in layer-split mode on this class of hardware. Bans like this
  belong in the serving policy, not in tribal memory.

## Model-by-model before/after (verified matrix, 2026-07-03 rerun)

Every "before" cell below was re-measured on 2026-07-03 as an explicit
baseline run (3-run medians, thermal-guarded, raw llama-bench/Ollama JSON in
[`data/`](data/)); "after" cells are the tuned recipes, re-verified the same
day. Full provenance tables:
[`data/model_matrix_4gpu_board_2026-07-03.md`](data/model_matrix_4gpu_board_2026-07-03.md)
(4-GPU NVLink board) and
[`data/model_matrix_single_card_2026-07-03.md`](data/model_matrix_single_card_2026-07-03.md)
(single V100).

### Qwen3.5-122B-A10B MoE (flagship, 4-GPU board)

| Cell | Decode (tg128) | Prefill (pp512) | vs stock |
|---|---|---|---|
| BEFORE — stock Ollama, defaults (3-run: 39.36/39.73/39.90) | 39.7 t/s | 174.7 t/s | — |
| AFTER — llama.cpp tuned (plain build, layer split, FA, f16 KV) | **61.8 t/s** | 507.6 t/s | **1.56x / 2.9x** |
| AFTER — PLILabsV100-vLLM, TP4 + CUDA graphs (AWQ) | **67.6 t/s** | **3,395 t/s** | **1.70x / ~19x** |

An earlier draft of this page cited the Ollama baseline as "~38 t/s
(estimated)"; it is now a measured 39.7 t/s (3 runs within 1.4%), which
slightly *lowers* the claimed speedups. The 32k-depth verification row that
previously thermal-aborted has also landed: 57.2 t/s at 32,768 tokens of KV
depth (q8 KV).

### Gemma-4-31B dense, 4-GPU board

| Cell | Decode | Prefill | vs default |
|---|---|---|---|
| BEFORE — plain build, all-default flags | 20.5 t/s | 824 t/s | — |
| AFTER — short-context tensor recipe (≤4k prompts only) | **39.2 t/s** | **1,602 t/s** | **1.91x / 1.94x** |
| AFTER — long-context safe: layer split + draft model (speculative) | **31.3 t/s** | — | 1.53x |

Tensor-split is **banned ≥8k-token prefill** on this hardware class (driver
wedge, section 6); the long-context lane keeps 1.53x via a small same-family
draft model instead.

### Gemma-4-26B-A4B MoE, single GPU

| Cell | Decode | Prefill |
|---|---|---|
| BEFORE — stock build, default flags | 100.3 t/s | 1,625 t/s |
| AFTER — tuned build + flags | 101.3 t/s | **1,846 t/s** (+13.6%) |

Plainly: stock llama.cpp defaults are already near-optimal for this model —
modern defaults auto-enable FlashAttention, and decode is flat. The big win
here is not the tuning, it's the **lane redesign**: 101.3 t/s on ONE GPU
versus the dense 31B's 39.2 t/s on FOUR.

### Gemma-4-31B dense, single GPU (new cell — an honest negative)

| Cell | Decode | Prefill |
|---|---|---|
| BEFORE — stock build, default flags, f16 KV | **30.6 t/s** | **681.6 t/s** |
| "Tuned" campaign build (FORCE_MMQ), same flags | 30.4 t/s | 440.5 t/s (**−35%**) |

Our multi-GPU campaign recipe **regresses** single-card dense prefill by 35%.
The FORCE_MMQ build exists for tensor-split output correctness on multi-card
boards; single-card, the stock build is both correct and faster. Guidance:
the recipe is shape-dependent — stock build + f16 KV wins single-card dense.
(The same build gains +13.6% prefill on the single-card MoE above.)

### Qwen3.6-27B GDN hybrid, single GPU

| Cell | Decode | Prefill |
|---|---|---|
| BEFORE — stock TurboMind 0.14.0 | **OOM on 32 GB** (fails at layer 63/64) | — |
| AFTER — our published GDN/AWQ patch (section 3) | **27.5 t/s** | 1,093 t/s (4k prompt) |

0 → served: the only "infinite" speedup on this page, and the one that
required a code patch rather than a flag.

---

## Context optimizations, before/after

Legal workloads are long-context workloads: repeated multi-thousand-token
system prompts and drafting sessions that accumulate KV depth. Measured
2026-07-03 (raw JSON in `data/`):

### Prefix-cache time-to-first-token, ~6k-token stable prefix

| Model | Cold TTFT | Warm TTFT | Speedup | Mechanism |
|---|---|---|---|---|
| Gemma-4-31B dense (6k prefix) | 42.0 s | 1.1 s | **36.7x** | `--cache-reuse 256 --swa-full` |
| Gemma-4-31B dense (12k prefix) | 93.1 s | 1.5 s | **61.7x** | same |
| Gemma-4-26B-A4B MoE, full-fit single GPU | 3.36 s | **39 ms** | **86.6x** | built-in LCP slot cache alone |
| — same, `--cache-reuse 256 --swa-full` | 3.56 s | 38 ms | 94.7x | explicit reuse |
| Qwen3.5-122B MoE, 4-GPU board | 9.5 s | 1.08 s | **8.9x** | built-in LCP slot cache |

Two findings worth paying for:

- **The SWA gotcha applies to the A4B MoE too.** It is a sliding-window
  model like its dense sibling: `--cache-reuse` alone is silently disabled
  (`cache_reuse is not supported by this context`); `--swa-full` is
  required. For pure append-only prefixes, though, the built-in
  longest-common-prefix slot cache already achieves full reuse (warm
  requests reprocess only the ~11-token question tail), and `--swa-full`
  costs ~6% on *cold* prefill — reserve the explicit flag pair for
  edited/shifted prefixes.
- **`--cache-reuse` is a NO-OP on Qwen3.5-122B.** The server logs
  `cache_reuse is not supported by this context, it will be disabled` — the
  hybrid linear/full-attention architecture does not support KV shifting.
  The 8.9x warm-TTFT win (9.5 s → 1.08 s) comes entirely from the default
  LCP slot cache and requires a static-prefix-first prompt layout; warm
  requests still reprocess ~510 tokens (LCP match granularity is coarser on
  this hybrid than on the Gemma models).

### KV-cache quantization at depth: when the model fits, f16 wins

| Model @ KV depth | f16 KV decode | q8 KV decode | Delta |
|---|---|---|---|
| Qwen3.5-122B @16k | **54.2 t/s** | 52.8 t/s | −2.6% |
| Gemma-4-26B-A4B @8k | **94.9 t/s** | 82.9 t/s | −12.7% |
| Gemma-4-31B dense @8k (layer split) | **19.6 t/s** | 18.0 t/s | −8% |

q8 KV quantization is for **context-starved deployments only** — buy it when
you need the 2x KV capacity, not for speed. When you do serve quantized or
mixed KV types, the `GGML_CUDA_FA_ALL_QUANTS` build (section 2) is what
removes the hidden 2x mixed-KV prefill cliff (79.4 → 161.3 t/s).

---

## 1CatAI vLLM fork: before/after our fixes

The vLLM numbers above run on our rebuild of the community 1CatAI V100 vLLM
fork (v1.2.1 class). The as-shipped build was **unable to serve any model** —
three independent defects, all diagnosed with reproducible evidence and fixed
(upstreamed as [1CatAI/1Cat-vLLM PR #96](https://github.com/1CatAI/1Cat-vLLM/pull/96)):

1. **A torch-version gate silently skips the entire generic op library.**
   The build gates the `_C_stable_libtorch` extension on torch ≥ 2.10; built
   against torch 2.9.1, the wheel ships without `silu_and_mul`, `rms_norm`,
   `rotary_embedding`, the whole KV-cache-ops namespace — every model fails
   at initialization with an op-missing `AttributeError`.
2. **Stale bundled attention module.** The packaged flash-attention-v100 was
   v1.0.0 (4 ops); the current backend requires 16+ (paged decode, XQA,
   split-KV prefill). The strict backend correctly refused to run rather
   than silently fall back — but refused is refused.
3. **UUID-form `CUDA_VISIBLE_DEVICES` crashes the model-registry
   subprocess** (`int()` parse bug), breaking the standard
   pin-GPUs-by-UUID operational practice; numeric indices required.

| Stage | Result |
|---|---|
| BEFORE — fork as shipped | **cannot serve** (init failure, all models) |
| AFTER — our rebuild + fixes (PR #96) | serves, correct output, all sm_70 kernels active |

Then, on the fixed build, eager vs CUDA graphs (the section-4 experiment):

| Measurement | Eager | CUDA graphs | Delta |
|---|---|---|---|
| 1x V100, Qwen3-4B-AWQ decode | 51.5 t/s | **132.8 t/s** | **2.56x** |
| TP4, Qwen3.5-122B-A10B-AWQ decode | 14.8 t/s | **67.6 t/s** | **4.5x** |
| TP4 122B prefill (4.2k tokens) | 2,996 t/s | **3,395 t/s** | +13% |
| TP4 122B aggregate @4 streams | 50.4 t/s | **66.7 t/s** | +32% |

Greedy output on the dense proof model is **bit-identical** eager vs graphs;
the TP4 122B capture is, to our knowledge, the first working tensor-parallel-4
CUDA-graph capture on sm_70. The consulting point: the difference between "this
fork is broken, Volta is dead" and a 67.6 t/s 122B production lane was three
diagnosable build defects and the willingness to measure.

## Methodology

- Every benchmark row records model file, binary git SHA, build flags,
  sampling parameters, and GPU set. No result without verified identity.
- 3-run medians; llama-bench JSON output retained.
- Real-output validation gates before any recipe is declared (fast numbers
  with broken output are worse than slow numbers).
- Continuous thermal monitoring with fail-loud alerts and automatic abort
  thresholds during every GPU experiment.
- Negative results recorded with the same provenance as wins.

## Artifacts — download and run

**Patches**
- [`patches/deltanet_awq.patch`](patches/deltanet_awq.patch) — the LMDeploy/TurboMind
  GDN-hybrid AWQ builder fix (section 3). Apply to lmdeploy 0.14.0 site-packages.
- [`patches/deltanet_verify_cpu.py`](patches/deltanet_verify_cpu.py) — rerunnable
  CPU verifier: proves the checkpoint AWQ weights survive the patch bit-exactly.

**Scripts (ready to run)**
- [`scripts/build_llamacpp_sm70.sh`](scripts/build_llamacpp_sm70.sh) — the V100 build
  recipe incl. `GGML_CUDA_FA_ALL_QUANTS=ON` (removes the section-2 cliff) and the
  CMake 3.28 gotcha.
- [`scripts/serve_gemma4_31b_prefix_cache.sh`](scripts/serve_gemma4_31b_prefix_cache.sh)
  — the 36.7–61.7x prefix-cache serving recipe (`--cache-reuse 256 --swa-full`).
- [`scripts/serve_gemma4_26b_a4b.sh`](scripts/serve_gemma4_26b_a4b.sh) — the 101 t/s
  single-V100 Gemma-4 MoE lane.
- [`scripts/serve_qwen36_27b_turbomind.sh`](scripts/serve_qwen36_27b_turbomind.sh)
  — the patched single-V100 Qwen3.6-27B recipe with the memory floor documented.
- [`scripts/thermal_guarded_bench.sh`](scripts/thermal_guarded_bench.sh) —
  provenance-grade benching with a fail-loud thermal abort (how every number in
  this study was collected).

**Data (raw measurements, sanitized)**
- [`data/kv_quant_fa_all_quants_build.json`](data/kv_quant_fa_all_quants_build.json) /
  [`data/kv_quant_default_build.json`](data/kv_quant_default_build.json) — the full
  16-combo KV-type matrix behind section 2 (llama-bench JSON, 3-run).
- [`data/context_depth_curve_q8kv.json`](data/context_depth_curve_q8kv.json) — the
  0→32k KV-depth curve.
- [`data/gemma4_26b_a4b_fullfit_profile.json`](data/gemma4_26b_a4b_fullfit_profile.json)
  — the single-card MoE profile behind section 5.
- [`data/prefix_cache_reuse_raw.json`](data/prefix_cache_reuse_raw.json) +
  [`data/lc_cache_reuse_v100.md`](data/lc_cache_reuse_v100.md) +
  [`data/lc_cache_test.py`](data/lc_cache_test.py) — raw cold/warm timings, the
  writeup, and the reusable measurement harness behind section 1.
- [`data/model_matrix_4gpu_board_2026-07-03.md`](data/model_matrix_4gpu_board_2026-07-03.md) /
  [`data/model_matrix_single_card_2026-07-03.md`](data/model_matrix_single_card_2026-07-03.md)
  — the full provenance tables behind the verified model-by-model matrix
  (which cells are newly measured vs reused, builds, flags, peak temps).
- `data/ba_*.json` — the 2026-07-03 baseline-rerun raw artifacts: stock-Ollama
  122B baseline (3-run), plain-build default-flag baselines for both Gemma
  models (4-GPU and single-card), the single-card dense regression cells,
  KV-quant-at-depth A/Bs, and the prefix-cache TTFT A/Bs (per-arm cold/warm
  timings for the 86.6x/94.7x and 8.9x rows).

Thermal logs and the lane-by-lane engineering log remain in the private
campaign repository; everything above is published per the
[research/v100 privacy boundary](../README.md).

---

*PLI Labs runs owned-hardware inference research so legal teams can deploy
models they can inspect, measure, and afford. If your firm is serving LLMs
on datacenter GPUs — current or previous generation — this is the kind of
measured, provenance-backed optimization work we do.*

- Website: https://proprietarylegal.com
- Research: https://proprietarylegal.ai
