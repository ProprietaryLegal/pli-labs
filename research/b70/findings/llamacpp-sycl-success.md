# llama.cpp SYCL Four-Card Success

## Result

The B70 research achieved the base goal: a single 122B-class Qwen3.5 MoE GGUF
served across all four Intel Arc Pro B70 cards through llama.cpp SYCL
layer-split.

The measured model footprint was about 72 GB of GPU memory distributed across
the four 32 GB cards. That is the key capacity result: the model was not merely
loaded on CPU or on one card; it used the B70 island as a four-card serving
target.

## Baseline Smoke

The initial four-card success produced coherent output and showed:

- about 20 tok/s cold single-stream generation;
- about 38 tok/s warm single-stream generation;
- no host wedge or reboot;
- stable model-serving behavior through the OpenAI-compatible API shape.

## Promoted Legal Profile

Later legal-context testing promoted a larger batch/microbatch profile:

| Setting | Public value |
|---|---:|
| Context for always-on profile | 12,288 |
| Batch | 16,384 |
| Microbatch | 16,384 |
| Parallel requests | 1 |
| KV cache | f16 |
| Split mode | layer |
| Continuous batching | enabled |

The promotion decision came from legal-context prompts rather than toy prompts.
The same profile held at L0, L1, and L2 prompt sizes in the benchmark summary.

## Why It Worked

The reliable lane avoids collective libraries and tensor-parallel worker
barriers. That matters on B70 because the unstable research path involved
collective-heavy tensor-parallel serving. llama.cpp layer-split uses a simpler
sequential layer distribution, which made the 122B-class model serveable and
operationally safe.

## Limitation

This is not the final speed story. Layer-split does not create true tensor
parallelism. It is a proven capacity and reliability result that gives PLI a
large local legal lane while the tensor-parallel XPU path matures.

## Public Release Conclusion

The public headline should be precise:

Four Arc Pro B70 cards can run a 122B-class local legal AI model through
llama.cpp SYCL layer-split, with long-context benchmark proof. The result is
production-useful, but it is not the same claim as a four-card tensor-parallel
speed win.
