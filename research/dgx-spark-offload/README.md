# DeepSeek-V4.1-Flash on 2× DGX Spark + x86 host offload

**Combined entry: hardware, architecture, production configuration and all measured results.** This is source code and
documentation; it contains no model weights. The served model is the public
[DeepSeek-V4.1-Flash-1017 EXL3 2.75 bpw legal-weighted build](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark).

**What we got:**
- **62 % more KV cache.** It now holds four 300K-token requests at once, which the stock setup can't fit.
- **4.6–6.6 % faster time-to-first-token on long prompts.**
- **Long tasks finish 3.9 % sooner at 150K** than the stock two-Spark recipe.
- The cost is 3–9 % single-stream decode speed.

The model stays on the Sparks. An x86 host contributes RAM, two RDMA links and RTX 3090s.

## Separately usable optimizations

| entry | what it does | status |
|---|---|---|
| [dsv41-remote-engram-rdma](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) | Engram tables (2 × ~101.5 GB) served from host RAM over RDMA; exact local fallback | proven, production |
| [dsv41-remote-dspark-drafter-3090](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) | DSpark drafter on an RTX 3090 ("capacity" mode); freed Spark memory → KV | proven, production |
| [dsv41-dspark-drafter-tp2-nvlink](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) | the drafter split over an NVLinked 3090 pair | proven in-engine (−2.5 %/round), under review |
| [dsv41-async-hostflag-handoff](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) | GB10 async deadlock finding + CUDA-free proxy hand-off | proven on 3090; GB10 speed pending |
| [ofl-rdma-transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) | RDMA RC / TCP request-response transport into caller memory | proven, production |
| [dsv41-longctx-bench-tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools) | long-context suite, direct per-step timer, link-load and drill tools | proven |
| [dsv41-spark-offload-findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings) | round-trip analysis, deadlocks, interference, exactness methodology, what didn't help | combined findings |

GitHub mirror: [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload).

## Folders in this directory (each optimization is self-contained: README + code)

| folder | Hugging Face |
|---|---|
| [`dsv41-remote-engram-rdma/`](dsv41-remote-engram-rdma/) | [HF](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) |
| [`dsv41-remote-dspark-drafter-3090/`](dsv41-remote-dspark-drafter-3090/) | [HF](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) |
| [`dsv41-dspark-drafter-tp2-nvlink/`](dsv41-dspark-drafter-tp2-nvlink/) | [HF](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) |
| [`dsv41-async-hostflag-handoff/`](dsv41-async-hostflag-handoff/) | [HF](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) |
| [`ofl-rdma-transport/`](ofl-rdma-transport/) | [HF](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) |
| [`dsv41-longctx-bench-tools/`](dsv41-longctx-bench-tools/) | [HF](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools) |
| [`dsv41-spark-offload-findings/`](dsv41-spark-offload-findings/) | [HF](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings) |
| [`config/production-D.env.example`](config/production-D.env.example) | sanitized production settings |

Hugging Face collection: https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e
Combined entry on Hugging Face: https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload

## Hardware

| | |
|---|---|
| Serving nodes | 2 × NVIDIA DGX Spark (GB10 Grace Blackwell, 121 GiB unified memory each), tensor-parallel 2 over the ConnectX-7 200 Gb/s fabric |
| Offload host | x86, AMD EPYC 7302P (16 cores / 32 threads, one NUMA node), 503 GiB RAM |
| Host ↔ Spark links | one ConnectX-5 Ex (PCIe Gen4 x16) with one 40 Gb/s DAC to each Spark (40GBASE-CR4), MTU 9000, RoCE v2. Measured 39.2 Gb/s per link, ~78.4 Gb/s both. On the Spark side this port shares its PCIe uplink with one TP rail. |
| Drafter GPUs | 2 × RTX 3090 24 GB with an NV4 NVLink bridge, power-capped at 325 W each (one card in the production config, two in the TP2 variant) |

## Software

- Image `ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5`: vLLM `0.31.1.dev0+g6e517b15.dsv41gb10` with HeNryous patches
  0022–0050, from [HeNryous/deepseek-v41-1017-dgx-spark](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark) @ b1726644.
- Launcher: [sudoingX/dgx-spark-deepseek-v41](https://github.com/sudoingX/dgx-spark-deepseek-v41) @ cc59c36. Our copy also
  accepts extra docker arguments for the overlays.
- exllamav3 1.6.0 (EXL3).

## What runs where (production configuration "D-kv6")

| component | where | notes |
|---|---|---|
| target model, 40 layers, TP2 | both Sparks | unchanged weights, sampling, tokenizer, reasoning parser |
| KV cache | both Sparks | **6.0 GiB/rank = 1,604,926 tokens** (stock: 3.75 GiB = 989,460) |
| Engram tables (layers 1 and 14) | host RAM, 189 GiB | rows over RDMA, one link per Spark; local NVMe reader kept as exact fallback |
| DSpark drafter (3 layers + Markov head) | host RTX 3090 | rank 0 sends tap rows and broadcasts drafts to rank 1; capacity policy: no drafter on the Sparks |
| per-request context | max-model-len 327,680 | the model supports 1,048,576 (1M validation in progress) |

The production settings, sanitized, are in `config/production-D.env.example`.

## Results (two DGX Spark, T = 1, thinking at effort "high", 1,500-token budgets, cold cache unless noted, n = 2–3)

| | A: stock recipe | B: + remote Engram | D-kv6: + remote drafter + KV 6 GiB |
|---|---|---|---|
| KV pool (tokens) | 989,460 | 989,460 | **1,604,926** |
| short prompt, stream tok/s | 55.96 ± 1.65 | — | 51.07 ± 0.78 |
| 32K TTFT | 15.53 ± 0.12 s | 14.70 ± 0.07 s | 14.81 ± 0.54 s |
| 32K stream tok/s | 64.57 ± 0.80 | — | 60.82 ± 2.30 |
| 150K TTFT | 73.62 ± 0.11 s | 69.44 ± 0.04 s | 68.79 ± 0.14 s |
| 150K task time | 100.35 ± 0.55 s | 95.17 ± 1.61 s | 96.46 ± 1.11 s |
| 256K / 300K TTFT, stream tok/s | — | — | 118.7 s, 51.6 / 140.3 s, 52.2 |
| 32K × 2 / × 3 aggregate tok/s | 40.59 / 47.21 | — | 40.81 / 47.36 |
| 4 × 300K at once | does not fit | does not fit | **all 4 complete** |
| MemAvailable per Spark | 5–7 GiB | 7 GiB | 5–8 GiB |

**Isolated effects:**
- **Remote Engram (B):** −5.4 % (32K) and −5.7 % (150K) TTFT, decode level, byte-exact.
- **Remote drafter (C):** frees ~2.5 GiB of model load per rank and ~3–5 GiB of available memory; +7–9 % per decode round at
  one stream, ≤ 2 % under concurrency; acceptance unchanged.
- **Two-card drafter (E):** −2.5 % per round versus one card, in-engine.
- **Async hand-off:** GB10 measurement pending.

## Deploy and roll back

Each optimization has its own switch and its own fallback (see its entry):

| to remove | do this | what you get back |
|---|---|---|
| remote Engram | unset `VLLM_ENGRAM_REMOTE*` and the overlay env | the image's NVMe reader |
| remote drafter | unset `VLLM_DSPARK_REMOTE*`, remove the overlay mounts, KV back to 3.75 GiB | the image's local drafter |
| two-card drafter | swap the worker unit back | the one-card worker; the engine reconnects in ~5 s |

The serve script refuses a half-enabled drafter configuration, and refuses a raised KV pool while the drafter is local, since
that combination is untested.

## Limits and open items

- Single-stream decode is 3–9 % slower with synchronous remote rounds on GB10. The async hand-off and the two-card worker
  target this.
- In progress, not yet proven on GB10:
  - KV page packing (upstream HeNryous 0057/0060; offline allocator figures: 1,604,926 → 2,502,600 tokens at 6 GiB/rank);
  - 1M-token context;
  - async speed.
- One host is a shared dependency. If it fails, the Sparks keep serving: Engram falls back to NVMe, and decode runs without
  speculation.

## Credits and licence

DeepSeek ([DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT) · HeNryous (Spark
image and patch set; [quant](https://huggingface.co/HeNryous/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-DGX-Spark)) · sudoingX
(two-Spark recipe and launcher) · turboderp ([exllamav3](https://github.com/turboderp-org/exllamav3), MIT) ·
[vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).

Our code is Apache-2.0.

Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
