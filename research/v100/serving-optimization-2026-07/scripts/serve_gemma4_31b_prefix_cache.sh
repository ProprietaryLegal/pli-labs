#!/usr/bin/env bash
# Gemma-4-31B llama.cpp serving with the measured 36.7-61.7x prefix-cache recipe.
# CRITICAL for SWA models (Gemma-4): --cache-reuse is silently disabled without --swa-full.
set -euo pipefail
MODEL=${MODEL:?path to gemma-4 31B GGUF}
BIN=${BIN:-./llama.cpp/build-sm70/bin/llama-server}
exec "$BIN" -m "$MODEL" \
  -ngl 999 -fa 1 -sm layer -b 2048 -ub 512 -c "${CTX:-32768}" \
  --jinja --reasoning-format none \
  --cache-reuse 256 --swa-full --cache-prompt \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}"
# V100 hazard: never use -sm tensor with prompts >=8K tokens (reproducible driver wedge).
