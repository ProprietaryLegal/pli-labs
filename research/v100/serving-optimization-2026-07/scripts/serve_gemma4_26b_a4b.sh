#!/usr/bin/env bash
# Gemma-4-26B-A4B MoE on a single V100-32GB: measured 101.3 t/s generation, 1845 t/s prefill.
set -euo pipefail
MODEL=${MODEL:?path to gemma-4-26B-A4B UD-Q4_K_M GGUF}
BIN=${BIN:-./llama.cpp/build-sm70/bin/llama-server}
exec "$BIN" -m "$MODEL" \
  -ngl 999 -fa 1 -b 2048 -ub 512 -c "${CTX:-16384}" \
  --jinja --reasoning-format none \
  --cache-reuse 256 --swa-full --cache-prompt \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}"
# f16 KV preferred at full fit (q8_0 costs ~12% tg on this model); MoE => layer split only.
