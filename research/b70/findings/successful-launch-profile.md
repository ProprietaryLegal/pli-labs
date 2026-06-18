# Successful B70-Class Launch Profile

This note publishes the useful part of the B70 research record: a successful
owned-hardware launch profile with measured legal-work throughput. It omits
hostnames, network addresses, local paths, service names, process output, and
failed launch recipes.

## Anonymous Machine Background

- Hardware class: four-card Intel Arc Pro B70 workstation.
- Accelerator memory: 32 GB per card, 128 GB aggregate VRAM.
- Runtime lane: Intel Level Zero / SYCL through a GGUF-compatible llama.cpp
  server build.
- Workload: long-context legal drafting and review prompts.
- Measurement status: measured on the owned research machine, not inferred from
  vendor numbers.

## Successful Large-MoE GGUF Stack

The promoted lane was a layer-split GGUF server path for a
122B-class mixture-of-experts model with about 10B active parameters per token.
The public artifact most closely aligned with this lane is the B70-friendly
MiniMax-M2.7 legal REAP GGUF release:

https://huggingface.co/ProprietaryLegal/minimax-m2.7-reap-172b-a10b-q4-k-m-gguf

Public launch template:

```bash
llama-server \
  --model <local-gguf-path> \
  -ngl 99 \
  -sm layer \
  -c 12288 \
  -b 16384 \
  -ub 16384 \
  -np 1 \
  -cb \
  -ctk f16 \
  -ctv f16 \
  -fa auto \
  --kv-offload \
  --defrag-thold 0.1 \
  --no-mmap
```

The template intentionally leaves out private path, host, port, and service
manager details. The settings above are the successful serving shape: full GPU
offload where supported, layer split across the B70 cards, 12K context for the
legal readiness lane, large prompt/micro batches, Flash Attention auto mode,
FP16 KV tensors, and KV offload.

## Measured Legal Throughput

| Tier | Prompt tokens | Output tokens | Prompt tok/s | Decode tok/s | Client output tok/s | Total tok/s | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| L0 legal canary | 9,274 | 1,024 | 609.79 | 35.17 | 23.06 | 232.07 | Passed |
| L0 service window | 9,274 | 1,024 | 613.64 | 35.24 | 23.15 | 232.80 | Passed |
| L1 medium legal context | 17,646 | 2,048 | 584.44 | 34.79 | 22.97 | 220.91 | Passed |
| L2 long legal context | 35,034 | 2,048 | 482.60 | 30.80 | 14.71 | 266.34 | Passed |

The practical finding is stable: the B70 layer-split lane remained usable for
real legal prompts, with decode near 35 tokens per second at the shorter and
medium legal tiers and near 31 tokens per second at the longer tier. Client
output throughput was lower on the longest prompt because long-context overhead
became visible to the caller.

## Smaller OpenVINO Research Lane

A separate OpenVINO GenAI HETERO lane succeeded on smaller exported INT4
models. It is useful for runtime research, not a quality-equivalent substitute
for the large-MoE GGUF lane.

| Model class | Prompt tokens | Output tokens | Output tok/s | Total tok/s | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 7B-class INT4 | 8,429 | 256 | 44.40 | 1,506.16 | Passed |
| 27B-class INT4 L0 | 9,274 | 1,024 | 24.76 | 248.96 | Passed |
| 27B-class INT4 L1 | 17,646 | 2,048 | 23.99 | 230.65 | Passed |

## Public Conclusion

B70-class owned hardware is a practical local legal AI lane when the runtime,
model format, quantization, and prompt tier are validated together. The
successful path is best described as a capacity and privacy lane, not as a
generic tensor-parallel benchmark claim.
