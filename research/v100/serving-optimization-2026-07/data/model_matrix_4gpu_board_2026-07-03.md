# Before/After Model Matrix — Board B (4× V100-SXM2-32GB, NV2 quad), 2026-07-03

Lane: Board B UUIDs `GPU-54038fcd…`, `GPU-b4ac6471…`, `GPU-049de101…`, `GPU-6f7b0a42…`
(this-boot indices 2,3,4,5). Peak lane temp during the whole suite: **77C** (no thermal aborts).
Builds: all llama.cpp upstream SHA `fdb1db877` (plain `llama.cpp/build`, build-mmq
FORCE_MMQ+NO_PEER_COPY, build-faq +FA_ALL_QUANTS), CUDA 12.6.85. Raw artifacts: `kernel-bench/ba_*.json`
(new) and `kernel-bench/bB_*.json` (reused, verified 2026-07-03 04:15Z suite).

## Per-model before/after (pp512 / tg128, r=3 medians)

### Qwen3.5-122B-A10B MoE (Opus-Reasoning Q4_K_M GGUF, 74GB)
| Cell | Definition | pp t/s | tg t/s | Provenance |
|---|---|---|---|---|
| **BEFORE** | **stock Ollama** (defaults: fa off, KV f16, batch 512), spare-port instance :11500 pinned to Board B, 3× 128-tok generations, 88-tok prompt | 174.7 (prompt rate) | **39.73** (39.36/39.73/39.90) | ba_qwen122_before_ollama.json (NEW) |
| **AFTER (GGUF)** | llama.cpp plain fdb1db877, `-sm layer -fa 1 -ctk/-ctv f16 -b 2048 -ub 512` | 507.6 | **61.82** | bB_qwen122_plain_layer_f16.json (REUSED, r=3) |
| AFTER (alt) | build-mmq, layer + q8_0 KV (ctx-starved recipe) | 490.8 | 57.58 | bB_qwen122_mmq_layer_q8.json (REUSED) |
| AFTER (vLLM) | PLILabsV100-vLLM TP4 + CUDA graphs, AWQ | 3395 | 67.0–67.6 | 1cat_v121_validation.md (REUSED) |
| **Speedup** | best GGUF vs stock Ollama | **2.9×** pp | **1.56×** tg (1.70× with vLLM/AWQ) | |

### Gemma-4-31B dense (Q8_0 GGUF)
| Cell | Definition | pp t/s | tg t/s | Provenance |
|---|---|---|---|---|
| **BEFORE** | llama.cpp plain build, ALL-DEFAULT flags (`-ngl 99` only; fa=auto, layer split), r=3 | 824.3 | **20.52** | ba_gemma31_before_plain_default.json (NEW) |
| **AFTER (short-ctx)** | build-faq, `-sm tensor -fa 1 -ctk/-ctv f16 -b 2048 -ub 512` (≤4K prompts ONLY — tensor+≥8K prefill is BANNED) | 1601.6 | **39.21** | bB_gemma_tensor_f16.json (REUSED, r=3) |
| AFTER (long-ctx safe) | build-faq, `-sm layer -fa 1` f16 KV | 505.2 | 20.67 | bB_gemma_layer_f16.json (REUSED) |
| AFTER (long-ctx + draft) | layer + E2B spec-decode draft | — | 31.3 | specdecode_e2b_gemma31b.md (REUSED) |
| **Speedup** | tensor short-ctx vs default | **1.94×** pp | **1.91×** tg (1.53× long-ctx via draft) | |

Note: gemma BEFORE was NOT stock Ollama (model present in ollama as 12-GPU spread ≈20 t/s,
results.json 2026-07-02 — same ballpark as the 4-card default). BEFORE here = "plain build, default flags",
as defined in the work order.

## Context table — Qwen3.5-122B (flagship)

### tg128 at KV depth (build-faq, `-sm layer -fa 1`, r=2 medians) — NEW
| Depth | f16 KV | q8_0/q8_0 KV | delta |
|---|---|---|---|
| 0 (ref) | 61.82 (plain, r=3, reused) | 57.58 (mmq, reused) | −6.9% |
| 8192 | **58.36** | 52.64 (sd 2.7) | −9.8% |
| 16384 | **54.21** (sd 2.8) | 52.78 | −2.6% |
| 32768 (reused, q8/mmq) | — | 57.2 | bB_qwen122_pp32k_r1.json |

Verdict: **f16 KV wins at every measured depth at full fit** — q8_0 KV is for ctx-starved configs only.
The 122B is depth-resilient (−12% tg from 0→16k vs gemma-dense −54% @32k on card14).

### Prefix-cache TTFT (llama-server :8250, build-mmq recipe, ~6050-token system prefix + short question, 2 cycles each) — NEW
| Arm | cold prompt_ms | warm prompt_ms | warm prompt_n | speedup |
|---|---|---|---|---|
| baseline (no --cache-reuse) | 9734 / 9493 | 1090 / 1079 | 510 | **8.9×** |
| `--cache-reuse 256` | 9594 / 9366 | 1081 / 1073 | 510 | 8.9× (identical) |

⚠️ **`--cache-reuse` is a NO-OP for Qwen3.5-122B**: server logs
`cache_reuse is not supported by this context, it will be disabled` (hybrid linear/full-attention
model — KV shifting unsupported, same class of restriction as vLLM prefix-caching on this arch).
The 8.9× warm-TTFT win comes entirely from the default LCP slot cache (`cache_prompt: true`) and
requires static-prefix-first prompt layout. Warm requests reprocess ~510 tokens (not ~25 as on
gemma) — LCP match granularity is coarser on this hybrid; still 9.5s → 1.08s.

## Context table — Gemma-4-31B at depth (layer split — tensor banned ≥8K) — NEW
| Depth | f16 KV, default flags (plain) | q8_0 KV + FA (build-faq) |
|---|---|---|
| 8192 | **19.62** | 18.02 (−8%) |

Full-fit gemma also prefers f16 KV at depth (matches the 04:15Z suite: 20.67 → 19.20 q8 at depth 0).

## Cells reused vs newly measured
- NEW (this lane): 122B stock-Ollama BEFORE (3-run), gemma plain-default BEFORE (r=3), 122B KV-depth
  A/B d8192+d16384 f16-vs-q8 (r=2), 122B TTFT cache-reuse A/B (2 cycles × 2 arms), gemma d8192 KV A/B (r=2).
- REUSED (verified, not rerun): all AFTER cells (bB_* 04:15Z suite r=3), pp32k row (bB_qwen122_pp32k_r1),
  vLLM TP4 numbers, spec-decode gemma numbers, Ollama 12-GPU spread rows.
- SKIPPED: none. No thermal aborts (peak 77C).

## Anomalies / ops notes
1. **Orphaned ollama runner**: killing the spare `ollama serve` by port does NOT kill its `ollama runner`
   child — it reparents to init and holds ~28GB/GPU until keep_alive expiry. Kill the runner PID explicitly.
2. **049de101 (idx 4) warm-idle ≈61C** after a load: a cool-wait threshold of <60C never converges on this
   card — use <66C. It peaked 77C during this suite (83C abort line never approached).
3. Gemma default-flags pp512 (824) exceeds the tuned `-fa 1 -b 2048 -ub 512` layer rows (~505) — fa=auto
   + default batching is better for layer-split pp on this dense model; the tensor recipe still wins overall.
4. Ollama BEFORE ran fast (load 38s) because the 74GB blob was in page cache; timings are steady-state
   (3 runs within 1.4%).
