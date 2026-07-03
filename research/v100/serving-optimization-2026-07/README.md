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
| Mixed-type KV-cache prefill (missing FA kernels) | 79–94 t/s | 154–161 t/s | **~2x cliff removed** |
| Qwen3.6-27B (GDN hybrid) on one 32GB V100 | did not fit (OOM) | 27.8 t/s decode | **unservable → served** |
| vLLM decode via CUDA graphs (sm_70) | 51.6 t/s eager | 132.8 t/s | **2.56x** |
| Gemma-4 lane throughput per GPU (MoE swap) | 40 t/s on 4 GPUs (10 t/s/GPU) | 101.3 t/s on 1 GPU | **~10x per GPU** |
| Qwen3.5-122B-A10B vs stock Ollama baseline | 38 t/s | 58.6–61.7 t/s | **1.6x** |

All decode figures are single-stream generation tokens/sec at the stated
context unless noted. Aggregate-throughput and prefill figures are called out
separately in each section.

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
unlocks the order-of-magnitude wins above.

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

Getting to this number also surfaced three real build/packaging defects in
the fork (a torch-version gate that silently drops the entire generic op
library, a stale bundled attention module, and a device-visibility parser
that rejects UUID-form pinning). All three are documented with fixes — the
kind of issue that quietly costs teams weeks.

## 5. Model-level optimization: MoE for the Gemma lane (~10x per GPU)

The biggest single win wasn't a kernel — it was recognizing that a
same-family MoE checkpoint dominates the dense deployment for
throughput-bound lanes:

| Gemma-4 lane option | Hardware | Generation | Prefill | Per-GPU |
|---|---|---|---|---|
| 31B dense, validated best recipe | 4x V100 board | 40.0 t/s | 1,642 t/s | 10 t/s |
| **26B-A4B MoE (4B active), full-fit** | **1x V100** | **101.3 t/s** | **1,845 t/s** | **101 t/s** |

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

## Methodology

- Every benchmark row records model file, binary git SHA, build flags,
  sampling parameters, and GPU set. No result without verified identity.
- 3-run medians; llama-bench JSON output retained.
- Real-output validation gates before any recipe is declared (fast numbers
  with broken output are worse than slow numbers).
- Continuous thermal monitoring with fail-loud alerts and automatic abort
  thresholds during every GPU experiment.
- Negative results recorded with the same provenance as wins.

## Artifacts

- [`patches/deltanet_awq.patch`](patches/deltanet_awq.patch) — the
  LMDeploy/TurboMind GDN-hybrid AWQ builder fix (section 3).
- Full raw benchmark JSON, thermal logs, and lane-by-lane engineering log
  remain in the private campaign repository; sanitized conclusions are
  published here per the [research/v100 privacy boundary](../README.md).

---

*PLI Labs runs owned-hardware inference research so legal teams can deploy
models they can inspect, measure, and afford. If your firm is serving LLMs
on datacenter GPUs — current or previous generation — this is the kind of
measured, provenance-backed optimization work we do.*

- Website: https://proprietarylegal.com
- Research: https://proprietarylegal.ai
