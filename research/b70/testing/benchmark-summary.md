# B70 Public Benchmark Summary

## Promoted 122B llama.cpp SYCL Profile

The promoted public profile is Dockerized llama.cpp `server-intel`, four-card
B70 SYCL layer-split, a 122B-class Qwen3.5 MoE GGUF, f16 K/V, continuous
batching, and batch/microbatch of 16,384.

| Legal tier | Prompt tokens | Output target | Context | Server prompt tok/s | Server decode tok/s | Client output tok/s | Total tok/s | Result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| L0 service-window | 9,274 | 1,024 | 12,288 | 613.64 | 35.24 | 23.15 | 232.80 | promoted |
| L0 live verify | 9,274 | 1,024 | 12,288 | 609.79 | 35.17 | 23.06 | 231.94 | verified |
| L1 service-window | 17,646 | 2,048 | 20,480 | 584.44 | 34.79 | 22.97 | 220.91 | promoted |
| L2 service-window | 35,034 | 2,048 | 40,960 | 482.60 | 30.80 | 14.71 | 266.34 | promoted |

## Conservative Fallbacks

| Profile | Scope | Public conclusion |
|---|---|---|
| batch/microbatch 8,192 | L0/L1/L2 proof | Conservative rollback; slightly slower than the promoted profile |
| batch/microbatch 4,096 | L0/L1/L2 proof | Older stable fallback; no longer preferred |
| batch/microbatch 32,768 | L0/L1 speed promise, L2 readiness failure | Interesting lower-context candidate, not promoted |

## Alternative Stack Results

| Lane | Model class | Scope | Public result | Conclusion |
|---|---|---|---|---|
| vLLM XPU one-card | 7B-class | Legal L0/L1 canary | low-30 tok/s client output on longer prompt shapes | useful smaller-model XPU lane |
| OpenVINO GenAI HETERO | 7B INT4 export | Legal L0 canary | 44.40 output tok/s with 256 output tokens | viable exported-model lane, not a 122B replacement |
| OpenVINO GenAI HETERO | 27B dense-derived INT4 export | Legal L0 canary | 24.76 output tok/s with 1,024 output tokens | promising research lane |
| source-built llama.cpp | 122B-class | L0/L1 comparator | viable, but did not beat Docker across promotion gates | research-only |

## Prompt Cache Finding

Exact repeated long-context prompts showed a major workflow win. In the L2
repeated-context test, the second cached request reused almost the entire
prompt and dropped from roughly 82 seconds to roughly 8 seconds. That is a
workflow optimization, not a raw model-speed claim.

The caveat is important: realistic follow-up prompts must preserve an identical
prefix to get the same benefit. Benchmark the prompt format before relying on
cache reuse.

## What Counts As Public Evidence

Numbers in this folder are public evidence only when they include:

- backend class;
- model class;
- GPU count;
- context length;
- prompt tokens;
- generated tokens;
- load/readiness status;
- throughput metrics for successful runs;
- failure reason for failed runs.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
