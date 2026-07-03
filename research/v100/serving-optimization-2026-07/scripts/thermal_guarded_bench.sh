#!/usr/bin/env bash
# Provenance-grade llama-bench wrapper with a fail-loud thermal guard.
# Usage: GPU_UUID=GPU-xxxx ./thermal_guarded_bench.sh <llama-bench args...>
set -euo pipefail
GPU_UUID=${GPU_UUID:?set GPU_UUID to the UUID under test (pin by UUID, indices shift across boots)}
ABORT_C=${ABORT_C:-83}
BENCH=${BENCH:-./llama.cpp/build-sm70/bin/llama-bench}
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_UUID" "$BENCH" "$@" &
PID=$!
while kill -0 $PID 2>/dev/null; do
  T=$(nvidia-smi --query-gpu=uuid,temperature.gpu --format=csv,noheader | grep "${GPU_UUID#GPU-}" | cut -d',' -f2 | tr -d ' ')
  if [ "${T:-0}" -ge "$ABORT_C" ]; then echo "THERMAL ABORT at ${T}C" >&2; kill $PID; exit 98; fi
  sleep 10
done
wait $PID
