# V100 Speedup Findings

## Best Bets

1. **MTP speculative decoding.** Worth testing where the model family includes
   compatible MTP heads. Treat public 1.7x to 2.2x claims as hypotheses until
   measured on the exact V100 lane and prompt set.
2. **V100-specific AWQ/MoE kernels.** The largest quantization win is using a
   real Volta fast path. Generic 4-bit fallback can underperform fp16.
3. **KV-cache compression.** Useful for context fit and sometimes decode speed,
   but backend-specific. Do not call it native FP8 compute on V100.
4. **Attention backend selection.** V100-specific attention backends can matter,
   but each model needs long-prefill and decode tests.
5. **CUDA graph experiments.** Potentially useful, but not safe to promote until
   the exact lane loads, runs long prompts, and keeps enough KV capacity.

## Kill List

Do not assume these are available on V100:

- bf16 tensor-core inference;
- native FP8 compute;
- FlashAttention-2 as used on newer GPUs;
- FlashInfer-first paths;
- Marlin/GPTQ-Marlin fast paths;
- generic expert-parallel performance across weak interconnects.

## Benchmark Discipline

Every speed claim should state whether it is:

- measured on this V100 lane;
- measured on another V100 system;
- extrapolated from newer GPUs;
- only a proposed experiment.

The research found several attractive speedup claims that are plausible but not
decision-grade until they are measured on the exact V100 stack.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
