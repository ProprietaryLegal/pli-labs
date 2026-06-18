# V100 Software Stack Findings

## Backend Matrix

| Backend class | V100 status | Best use |
|---|---|---|
| `llama.cpp` / compatible forks | Strong compatibility | GGUF serving, broad model coverage, fallback loading |
| Ollama | Strong compatibility through GGUF | Quick tests, model management, local demos |
| V100-focused vLLM forks | Useful but version-sensitive | OpenAI-compatible serving, continuous batching, selected AWQ/MoE lanes |
| LMDeploy/TurboMind | Promising for validated W4 dense lanes | Dense quantized serving on Volta |
| Stock modern vLLM | Fragile on V100 | Only with exact supported versions and flags |
| TensorRT-LLM, modern TGI, FlashInfer-first stacks | Poor fit | Generally not the first V100 choice |

## V100-Specific Settings

Common safe starting assumptions:

- use float16, not bf16;
- set model length conservatively until KV capacity is proven;
- benchmark after warmup;
- confirm which attention backend actually loaded;
- confirm whether a 4-bit kernel is a real V100 fast path or a slow fallback;
- capture `/v1/models` or equivalent served identity before believing a result.

## Quantization Lessons

4-bit is not automatically fast on V100. The fastest modern 4-bit kernels often
depend on GPU features that Volta does not have. A V100 fast path usually works
by dequantizing into fp16 tensor-core work or by using a Volta-specific kernel.

For GGUF, quantized KV cache can be a practical context-length tool. For vLLM,
FP8-KV can help fit context when the backend supports the compatible storage
format, but it should not be described as native FP8 compute on V100.

## Attention Lessons

The safest public statement is:

- upstream FlashAttention-2 is not the normal V100 path;
- V100-specific attention forks exist;
- fallback attention backends may work but can be slower or model-sensitive;
- long-prefill tests are mandatory before promoting an attention backend.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
