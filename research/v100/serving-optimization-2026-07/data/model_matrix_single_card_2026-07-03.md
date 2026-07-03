# Single-card BEFORE/AFTER model matrix — card14 (V100-SXM2-32GB), 2026-07-03

Lane: single-card before/after baseline suite. GPU: card14, UUID `GPU-c76e1399-0586-f2ca-58f9-3e83a7b78b72`,
pinned via `CUDA_VISIBLE_DEVICES` (this-boot index 14). Host: the V100 host.

Builds (same source SHA, provenance `kernel-bench/build-faq.provenance.txt`):
- **BEFORE ("stock")** = `llama.cpp/build` — llama.cpp 9860 (`fdb1db877`), plain upstream CUDA build
  (FORCE_MMQ=OFF, NO_PEER_COPY=OFF, FA_ALL_QUANTS=OFF), **all-default llama-bench flags except `-ngl 99`**
  (note: default `-fa` is **auto (-1)**, which does enable FlashAttention on CUDA — modern defaults are not FA-off).
- **AFTER ("tuned")** = `llama.cpp/build-faq` — same SHA, FORCE_MMQ=ON + NO_PEER_COPY=ON +
  FA_ALL_QUANTS=ON, explicit `-fa 1` (+ KV-quant flags where stated).

All rows: llama-bench `-o json`, raw JSONs + stderr (incl. peak temps) in `kernel-bench/ba_*_card14.json`.
Thermal guard: 5s poll, abort >=83C — never triggered (peak across suite: 72C).

## Model-by-model BEFORE/AFTER (single card, pp512 / tg128)

| Model | Cell | Build / flags | pp512 t/s | tg128 t/s | r | Provenance |
|---|---|---|---|---|---|---|
| **gemma-4-26B-A4B MoE** (UD-Q4_K_M 16.9G) | BEFORE (stock default) | plain build, `-ngl 99` only (fa auto, f16 KV) | 1624.9 ±11.0 | 100.28 ±0.15 | 3 | `ba_a4b_before_card14.json` (NEW, peak 51C) |
| | AFTER (tuned) | build-faq, `-ngl 99 -fa 1` (f16 KV) | **1845.5** | **101.31** | — | REUSED `a4b_card14_profile.json` (2026-07-03) |
| | delta | | **+13.6% pp** | +1.0% tg | | |
| **gemma-4-31B dense** (Q4_K_M requant 17.4G, single-card full fit) | BEFORE (stock default) | plain build, `-ngl 99` only (fa auto, f16 KV) | **681.6 ±1.9** | **30.63 ±0.06** | 3 | `ba_gemma31b_q4_before_card14.json` (NEW, peak 67C) |
| | AFTER (tuned, f16 KV) | build-faq, `-ngl 99 -fa 1` | 440.5 ±0.6 | 30.43 ±0.05 | 3 | `ba_gemma31b_q4_faq_f16_card14.json` (NEW) |
| | AFTER (tuned, q8 KV) | build-faq, `-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0` | 434.9 ±0.03 | 28.58 ±0.03 | 3 | `ba_gemma31b_q4_tuned_card14.json` (NEW, peak 72C) |
| **Qwen3.6-27B** | BEFORE | llama.cpp any | — (no GGUF on disk; not downloaded for this run) | — | | SKIPPED |
| | AFTER | TurboMind W4 AWQ + INT8KV (deltanet patch) | 1093 (4k prefill) | **27.5** (stock vLLM/TM = OOM pre-patch) | — | REUSED `turbomind_card14.md` Attempt 4 |

- ⚠️ **31B dense headline (honest):** on a single-card FULL-FIT dense Q4 model, the "tuned" campaign recipe is a
  **regression**: the FORCE_MMQ=ON build costs **-35% pp512** on dense Q4_K_M (682 → 440, isolated by the faq-f16
  middle cell — same flags, only the build differs beyond `-fa 1`), and q8 KV separately costs ~6% tg (30.4 → 28.6).
  Stock defaults (plain build, fa auto, f16 KV) are the best measured single-card config for this model. The campaign's
  FORCE_MMQ requirement exists for `-sm tensor` output CORRECTNESS on multi-card — irrelevant single-card, where the
  plain build is both correct and faster at prefill. q8 KV remains the right call when context capacity, not speed,
  is the constraint. (Contrast: the same build gains +13.6% pp on the A4B MoE — MMQ effect is shape-dependent.)
- 31B GGUF provenance: `gemma4-dense-Q4_K_M.requant.gguf`, sha256
  `fec09df7348be4c4b654d7f0f4f6646777bd76de06f8d11c5d155c970955af99`, produced this session by
  `build-faq/bin/llama-quantize --allow-requantize` from `gemma4-dense.gguf` (Q8_0) — Q8→Q4
  requant, 4.87 BPW; **quality slightly below a from-f16 Q4_K_M** (speed numbers unaffected). Real-generation
  sanity: PASS — coherent greedy output ("The capital of France is Paris"), real-gen 30.9 t/s ≈ bench 30.4.
- A4B before/after: default is already near-optimal for tg (fa auto engages FA); build-faq adds +13.6% prefill on
  this MoE. The published tuned recipe (101.3 tg / 1845 pp) is reused unchanged from `a4b_card14_profile.json`.

## Context-optimization BEFORE/AFTER (gemma-4-26B-A4B, single card)

### (a) Prefix-cache TTFT — llama-server :8260, tuned serve recipe, ~6000-token system prefix, n_predict=16, 2 cold/warm cycles each (mirrors `lc_cache_raw.json` methodology)

| Config | cold prompt_ms (mean, n=2) | warm prompt_ms (mean, n=2) | warm prompt_n | speedup |
|---|---|---|---|---|
| BEFORE: no `--cache-reuse` (built-in LCP slot cache only) | 3360.3 | 38.8 | 11 | **86.6x** |
| AFTER: `--cache-reuse 256 --swa-full` | 3557.8 | 37.6 | 11 | **94.7x** |

- **SWA check (recorded): A4B IS an SWA model** — with `--cache-reuse 256` alone the server logs
  `cache_reuse is not supported by this context, it will be disabled`; `--swa-full` is REQUIRED (same as dense gemma-4).
- Finding: for the pure append-only prefix pattern, the built-in longest-common-prefix slot cache already achieves
  full reuse (warm reprocesses only the 11-token question tail) — `--cache-reuse` adds nothing here (its value is
  edited/shifted prefixes) and `--swa-full` costs ~6% on COLD prefill (3558 vs 3360 ms) via the larger KV.
  Unlike the earlier `-ngl 36` partial-offload dense run (warm 1.0-1.6 s), full-fit warm TTFT is **~38 ms**.
- Raw: `ba_a4b_ttft_nocache_card14.json`, `ba_a4b_ttft_cachereuse_card14.json`; server logs
  `ba_srv_{nocache,cr,crswa}_8260.log`. Peak 62C.

### (b) KV quant at depth — llama-bench `-d 8192 -p 512 -n 128`, r=2

| Config | pp512 @8k t/s | tg128 @8k t/s |
|---|---|---|
| BEFORE: plain build, default (fa auto, f16 KV) | 1461.1 ±29.4 | **94.89 ±0.05** |
| AFTER: build-faq `-fa 1 -ctk q8_0 -ctv q8_0` | **1560.3 ±5.3** | 82.85 ±0.11 |

- q8 KV at 8k depth: +6.8% prefill (build-faq MMQ effect), **-12.7% tg** — confirms the full-fit-MoE-prefers-f16-KV
  finding from `a4b_card14_profile.json` (d=16384: f16 93.4 vs q8 79.8). Buy q8 KV only for context capacity.
- Raw: `ba_a4b_d8192_default_card14.json`, `ba_a4b_d8192_tuned_card14.json`. Peaks 54/58C.

## Reused vs newly measured
- REUSED: A4B tuned pp512/tg128 + 16k-depth rows (`a4b_card14_profile.json`); Qwen3.6-27B AFTER
  (`turbomind_card14.md` Attempt 4, 27.5 t/s, stock = OOM); dense-31B multi-card context (`kvab_card14_*.json`,
  `depthcurve_card14_faq_q8.json`, `lc_cache_raw.json` — partial-offload Q8, kept as-is, NOT single-card-full-fit comparable).
- NEW (this lane): A4B stock baseline; A4B d8192 KV A/B; A4B TTFT A/B (full-fit); 31B-Q4 single-card stock /
  faq-f16 / faq-q8 cells; Q4 requant itself.
- SKIPPED: Qwen3.6-27B llama.cpp cross-backend row (no GGUF on disk, not downloaded for this run); card13 (thermally dead, untouched).
