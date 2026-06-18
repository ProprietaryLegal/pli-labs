# Current B70 Stack Findings

## Hardware Shape

The public research covers a 4 x Intel Arc Pro B70 inference island.

Public hardware facts:

- Each card has 32 GB VRAM.
- The four-card island provides 128 GB aggregate VRAM.
- The card class exposes Intel XMX AI engines and uses Intel's Xe / Level Zero
  software path rather than CUDA.
- Official materials describe PCIe Gen 5 x16 native support and a 230 W Intel
  branded-card power point.

The important operational lesson is the same as with any multi-GPU local AI
system: aggregate VRAM is not the same thing as one uniform fast memory pool.
Backend selection determines whether four cards are a capacity lane, a speed
lane, or both.

## Promoted Reliability Lane

The measured promoted lane is Dockerized llama.cpp `server-intel` with SYCL
layer-split across all four B70s.

The production-safe profile from the research window used:

- 122B-class Qwen3.5 MoE GGUF;
- Q4_K_M weights;
- four visible B70 devices;
- layer split rather than row/tensor split;
- 12K default serving context for the always-on lane;
- f16 K/V cache;
- batch and microbatch of 16,384;
- continuous batching enabled;
- KV offload enabled;
- memory mapping disabled.

The service-window gates also validated larger legal-context shapes up to a
35K-token prompt plus 2K-token output target.

## What The Result Means

This proves that four B70s can carry a 122B-class local legal drafting and
review lane. It does not prove that the current lane is a tensor-parallel speed
record. llama.cpp layer-split is a capacity and reliability path: it avoids
collective-communication failures, but only one layer range computes at a time.

## Operational Rule

B70 benchmark promotion should require:

- endpoint readiness, not just process launch;
- a real completion, not just model load;
- long-context legal prompt evidence;
- restore proof after service-window tests;
- clear distinction between prompt throughput, decode throughput, and
  client-visible output throughput.

## Links

- https://proprietarylegal.ai
- https://proprietarylegal.com
