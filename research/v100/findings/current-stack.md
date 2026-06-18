# Current V100 Stack Findings

## Hardware Shape

The research covers V100-class systems observed in two states:

- a 9 x V100-SXM2-32GB configuration used for the April 2026 Ollama and GGUF
  benchmark report;
- a larger V100 pool later observed with 13 V100-class 32GB cards and mixed
  NVLink islands.

The important topology lesson is stable across both states: do not treat
aggregate VRAM as one uniform memory pool. NVLink islands matter. A fast plan
keeps tensor-parallel communication inside a tightly connected island and uses
other methods for spanning weaker interconnects.

## Volta Rules

- V100 supports strong fp16 tensor-core compute but lacks bf16 tensor-core paths.
- V100 does not have native FP8 compute.
- V100 does not support the standard modern FlashAttention-2 path.
- Marlin and many GPTQ/AWQ fast paths target newer GPUs. A V100 stack needs a
  Volta-specific replacement or fallback.
- V100 can still be useful because LLM decode is often memory-bandwidth bound
  and because large owned VRAM pools can carry models that would otherwise be
  expensive to serve.

## Working Backend Classes

- GGUF through `llama.cpp`-style backends is the broadest compatibility lane.
- Ollama is useful for quick model management and was used for the April 2026
  benchmark report.
- vLLM is useful for API serving, but the V100 path is version-sensitive and
  often needs V100-specific forks or flags.
- LMDeploy/TurboMind-style W4 paths are worth testing for dense models where
  Volta support is explicit.

## Operational Finding

The strongest public rule is not tied to one model: V100 testing must be
proven on the actual hardware lane. Public numbers from A100/H100, or from a
different backend, are hypotheses until the exact V100 backend, model, context,
and prompt set are measured.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
