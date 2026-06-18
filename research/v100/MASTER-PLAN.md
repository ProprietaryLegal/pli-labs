# V100 Master Plan

## Thesis

V100 hardware remains useful for local AI when the software stack is
chosen around Volta's limits instead of around modern Ampere/Hopper assumptions.
The opportunity is not "V100 beats new GPUs." The opportunity is that a large
owned V100 pool can run strong open models privately, cheaply, and repeatably if
the backend, quantization, topology, and benchmark process are correct.

## Hard Constraints

- V100 is compute capability 7.0.
- Use float16 paths. Do not assume bf16, native FP8 compute, FlashAttention-2,
  FlashInfer, or Marlin kernels.
- Weight-only 4-bit can be fast only when the backend provides a real Volta
  kernel path. A generic 4-bit path can be slower than fp16.
- Tensor parallelism should stay inside an NVLink island. Cross-board tensor
  parallelism over PCIe is the main topology trap.
- Pipeline parallelism across boards can increase capacity, but it does not
  automatically improve single-request decode speed.
- First-request warmup and kernel compilation must be excluded from steady-state
  benchmark claims.

## Current Best Direction

Use three lanes:

- `llama.cpp` or compatible forks for GGUF flexibility, fallback loading, and
  broad model coverage.
- V100-focused vLLM forks for OpenAI-compatible serving, continuous batching, and
  Volta-specific AWQ/MoE work.
- LMDeploy/TurboMind-style lanes for dense W4 cases where the Volta path is
  validated.

## Highest-Value Experiments

1. Re-measure V100 board baselines with served model identity, deterministic
   prompts, and warmup discarded.
2. Test MTP speculative decoding on exact served models and the target prompt
   set.
3. Test `--enforce-eager` versus conservative CUDA graph settings, but treat the
   graph path as experimental until load, long-prefill, and output checks pass.
4. Confirm FP8-KV or quantized-KV behavior per backend. On V100, FP8 is a storage
   and fit question, not native FP8 compute.
5. Benchmark V100-specific fused MoE configuration where the backend exposes it.
6. Probe PP2 x TP4 versus TP8 plus expert parallel on a smaller model before
   trying a 200B-plus candidate.

## Public Validation Standard

Each public benchmark should publish:

- model family and quantization;
- backend and version family;
- GPU count and topology class;
- context length;
- prompt tokens and generated tokens;
- wall-clock generation throughput;
- whether the server loaded successfully;
- failure reason if it did not load;
- whether the number is measured, extrapolated, or only a recommendation.

## PLI Positioning

PLI's V100 work is infrastructure research for private, local AI. The
commercial point is practical: firms need systems they can inspect, run locally,
and evaluate against real workflows. The V100 fleet is a proving ground for
doing that without assuming cloud GPU availability.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
