# LC-CACHE: llama.cpp prompt/prefix cache reuse — the test GPU (V100 32GB)

Date: 2026-07-02. Campaign: V100 optimization, "R2 top serving win" experiment.

## Config provenance
- Binary: `llama-server`
  - `--version`: `version: 9860 (fdb1db877)`, built with GNU 13.3.0 for Linux x86_64
  - sha256: `2fb81ad1a4a1702e183128c8813d004677bd732f9ab7abad0a22d89bd3d389ac`
- Model: `gemma4-dense.gguf` (32,635,670,112 bytes, mtime 2026-05-13)
- GPU: the test GPU, `GPU-c76e1399-0586-f2ca-58f9-3e83a7b78b72`, pinned via `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<uuid>`
- Flags: `-ngl 36 -c 16384 -b 2048 -ub 512 -fa 1 --cache-reuse 256 --swa-full --slots -np 1 --host 127.0.0.1 --port 8199`
- Workload: synthetic ~6000-token (and ~12000-token) repeated system prefix + small unique question tail, `/completion` with `n_predict=16`, `temperature=0`, `cache_prompt=true`. Each COLD trial uses a distinct prefix (guaranteed cache miss); the paired WARM request reuses the same prefix with a different question.

## CAVEATS
1. **Partial offload**: `-ngl 36` — model does NOT fully fit in 32GB, some layers run on CPU. Absolute prompt_ms is not representative of a fully-offloaded model; RELATIVE cold-vs-warm deltas are the valid signal.
2. **`--cache-reuse` requires `--swa-full` for gemma4**: with the specified flags alone the server logs `cache_reuse is not supported by this context, it will be disabled` (gemma4 uses sliding-window attention). Adding `--swa-full` enables it (at higher KV memory cost). NOTE: even WITHOUT `--swa-full`, plain longest-common-prefix slot caching still worked for this exact pattern (see baseline row) — `--cache-reuse` matters for shifted/edited prefixes, not the pure append-only legal-drafting harness pattern.

## Results

### Baseline (no --swa-full; cache_reuse disabled, plain slot prefix cache only), ~6k prefix
| label | prompt_n | prompt_ms | predicted_n | predicted_ms | speedup (prompt_ms) |
|---|---|---|---|---|---|
| baseline-cold | 5866 | 42544.3 | 1 | 0.0 | — |
| baseline-warm | 7 | 4682.6 | 2 | 2038.6 | 9.1x |

### With --swa-full + --cache-reuse 256, ~6k prefix (3 alternating cold/warm pairs, distinct prefixes+questions)
| label | prompt_n | prompt_ms | predicted_n | predicted_ms | speedup (prompt_ms) |
|---|---|---|---|---|---|
| p6k-cold-0 | 6050 | 41830.8 | 1 | 0.0 | — |
| p6k-warm-0 | 11 | 1234.3 | 16 | 3916.0 | 33.9x |
| p6k-cold-1 | 6042 | 42145.3 | 1 | 0.0 | — |
| p6k-warm-1 | 11 | 1002.3 | 16 | 3914.7 | 42.0x |
| p6k-cold-2 | 6042 | 42167.0 | 1 | 0.0 | — |
| p6k-warm-2 | 11 | 1199.0 | 16 | 3893.1 | 35.2x |

### ~12k prefix (2 pairs)
| label | prompt_n | prompt_ms | predicted_n | predicted_ms | speedup (prompt_ms) |
|---|---|---|---|---|---|
| p12k-cold-0 | 12033 | 93154.8 | 1 | 0.0 | — |
| p12k-warm-0 | 11 | 1408.9 | 16 | 5217.8 | 66.1x |
| p12k-cold-1 | 12028 | 93081.8 | 1 | 0.0 | — |
| p12k-warm-1 | 11 | 1607.2 | 16 | 5101.1 | 57.9x |

## Summary
- ~6k prefix: cold prompt_ms mean 42047.7 (n=3), warm mean 1145.2 (n=3) → **36.7x mean prefill speedup** (range 33.9-42.0x). Warm requests reprocessed only 11 tokens (the changed question tail).
- ~12k prefix: cold mean 93118.3 (n=2), warm mean 1508.0 (n=2) → **61.7x mean prefill speedup** (range 57.9-66.1x). Benefit scales with prefix length, as expected.
- End-to-end (wall) at 6k: ~42.9s cold vs ~5.1s warm for a 16-token answer; at 12k: ~94s vs ~6.7s.
- Anomaly note 1: cold requests consistently returned predicted_n=1 (model emitted EOS immediately after the raw repeated prefix); warm requests generated the full 16 tokens. Does not affect prompt_ms comparison.
- Anomaly note 2: warm prompt_ms is ~1.0-1.6s for only 11 tokens — dominated by CPU-side layers from partial offload (-ngl 36); on a fully-offloaded model warm TTFT would be far lower still.
- Practical takeaway for legal-drafting harnesses: the big-repeated-system-prompt pattern gets its win from llama-server's built-in longest-common-prefix slot cache (`cache_prompt`), which works even where `--cache-reuse` is disabled (gemma4 SWA without `--swa-full`, 9.1x baseline). `--cache-reuse 256` (+`--swa-full`) adds robustness when the prefix is edited/shifted rather than purely appended. Keep `-np` slots >= number of distinct prefix families to avoid cache eviction thrash.

## Thermal
Card14 peak 76C during sustained 12k prefill (poll every 15s, kill threshold 83C never approached; THERMAL_KILL count 0). Idle-return 65C immediately after server kill, GPU 0 MiB / no compute processes confirmed.

Raw JSON: `lc_cache_raw.json`. Harness: `lc_cache_test.py`. Server logs: `lc_cache_server_8199.log`, `lc_cache_server_8199_swafull.log`. Temps: `lc_cache_temps.log`.
