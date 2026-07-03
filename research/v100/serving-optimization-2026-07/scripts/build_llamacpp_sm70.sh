#!/usr/bin/env bash
# llama.cpp build recipe for Tesla V100 (sm_70) serving with quantized/mixed KV caches.
# Removes the ~2x mixed-KV prefill cliff measured on default builds (see case study §2).
set -euo pipefail
REPO=${1:-llama.cpp}
[ -d "$REPO" ] || git clone https://github.com/ggml-org/llama.cpp "$REPO"
cmake -B "$REPO/build-sm70" -S "$REPO" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DGGML_CUDA_FORCE_MMQ=ON \
  -DGGML_CUDA_NO_PEER_COPY=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DLLAMA_BUILD_WEBUI=OFF \
  -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
cmake --build "$REPO/build-sm70" --target llama-bench llama-cli llama-server -j "$(nproc)"
echo "Binaries in $REPO/build-sm70/bin"
# Note: do NOT pass -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY with CMake 3.28+:
# it breaks CUDA compiler detection (archive-finish bug in the try-compile).
