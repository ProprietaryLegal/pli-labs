#!/usr/bin/env bash
# Qwen3.6-27B-AWQ (GDN hybrid) on ONE V100-32GB via LMDeploy/TurboMind 0.14.0
# REQUIRES patches/deltanet_awq.patch applied to lmdeploy (stock 0.14.0 OOMs — case study §3).
# Measured: 27.5 t/s single-stream, 1093 t/s prefill, 114-185 t/s @8 streams (clock-dependent).
set -euo pipefail
MODEL=${MODEL:?path to QuantTrio/Qwen3.6-27B-AWQ snapshot dir}
exec lmdeploy serve api_server "$MODEL" \
  --backend turbomind --model-format awq --tp 1 \
  --session-len 16384 --cache-max-entry-count 0.25 \
  --max-batch-size 16 --max-prefill-token-num 4096 \
  --quant-policy 8 --disable-vision-encoder \
  --server-port "${PORT:-8080}"
# cache-max-entry-count floor: >=0.25 (the GDN linear-state needs ~1.2GB; 0.10 fails loudly).
