# LLM Inference On V100 - Public Guide Summary

This page summarizes a 15-page V100 inference guide prepared for a 9-GPU V100
cluster.

## Main Point

V100 inference is still practical, but only when the backend is selected around
Volta. The guide identified three generally viable classes: GGUF through
`llama.cpp`-style backends, Ollama for quick management, and carefully
configured vLLM for API serving.

## Hardware Constraints

- Use float16 paths. V100 does not support bf16 tensor-core inference.
- Do not assume FlashAttention-2.
- Do not assume native FP8 compute.
- Do not assume Marlin or modern AWQ/GPTQ fast paths.
- Use topology-aware GPU placement because NVLink islands are uneven.

## Backend Guidance

- GGUF backends are the broadest compatibility lane.
- vLLM can be useful for multi-user API serving, but V100 support is fragile and
  flag-sensitive.
- Ollama is useful for quick local testing and model management.
- Modern stacks that assume newer GPU features should be treated skeptically
  until tested on V100.

## Benchmark Guidance

The guide recommends capturing:

- topology and NVLink status;
- backend and model identity;
- prompt-processing tokens per second;
- generation tokens per second;
- time to first token;
- peak VRAM per GPU;
- GPU utilization and temperature;
- failure logs for skipped or crashed models.

## Deployment Guidance

The practical deployment model is a layered stack:

- one lane for flexible GGUF serving;
- one lane for OpenAI-compatible API serving;
- one orchestration layer for swapping models across GPU groups;
- one monitoring path for GPU state and service health.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
