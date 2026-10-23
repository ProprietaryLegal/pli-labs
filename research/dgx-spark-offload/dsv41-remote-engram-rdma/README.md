# Remote Engram over RDMA: DeepSeek-V4.1-Flash's Engram tables served from an x86 host's RAM

**Source code, not model weights.** One of the separately usable optimizations from PLI Labs' DeepSeek-V4.1-Flash
deployment on two DGX Spark (GB10) nodes with an x86 offload host. Overview, hardware and combined results:
[DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
Methodology and findings: [dsv41-spark-offload-findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings).

**Status: proven and in production** (since 2026-10-22).

Hugging Face entry: https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma

## What it does

DeepSeek-V4.1-Flash-1017 has two Engram layers (1 and 14) whose lookup tables are ~101.5 GB each (shards
`model-00047/00048-of-00048.safetensors`; rows of 256 B FP8 plus 8 B ue8m0 scales; 24 hash columns, 12 per TP rank). The
DGX Spark recipe keeps them on each node's NVMe and reads rows with an O_DIRECT reader (`disk_offload`), so every 4,096-token
prefill chunk waits on random NVMe reads.

This package moves the row reads to a host with enough RAM:

- **Server** (`code/engram/server/engram_server.cpp`, x86): loads and sha256-verifies both tables into RAM (189 GiB
  resident, transparent huge pages) and answers row requests over RDMA RC (RoCE v2) or TCP. It listens only on the direct-link
  addresses you configure. Requests carry the table identity, and wrong identities or out-of-range rows are refused.
- **Spark side** (`code/engram/spark/`): an overlay of the image's `engram.py` plus a small C++ extension that replaces the
  disk reader's `enqueue_read` with a remote fetch. It uses the same spin-mode stream memory operations as the stock reader,
  and the proxy thread writes rows into the same staging buffers. **The local NVMe reader stays as an exact fallback**: a
  transport error or timeout drops that lookup to local disk.
- **Transport:** [ofl-rdma-transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) (included in `code/transport`).

## When to use it, and when not

Use it when your prompts are long, so prefill dominates, and a host with ≥ ~200 GB of free RAM has a direct RDMA link to
each Spark. Expect a prefill and time-to-first-token gain only; decode is unchanged.

Don't bother when prompts are short, when there's no spare-RAM host, or when your NVMe reads are already fast enough that
prefill isn't Engram-bound. It doesn't free GPU memory on the Sparks: the tables were never resident there.

## Requirements

- x86 host: ~190 GiB of RAM for the two tables, plus a ConnectX NIC with RoCE v2. Measured on a ConnectX-5 Ex at 40 Gb/s DAC
  to each Spark, MTU 9000.
- Spark image `ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5`. The overlay checks the image's `engram.py` and
  `engram_disk.cpp` hashes before it loads.
- Both Engram shards from `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, on the host.

## Measured effect (conditions below)

| context | baseline TTFT | remote-Engram TTFT | change |
|---|---|---|---|
| 32K cold | 15.53 ± 0.12 s | 14.70 ± 0.07 s | **−5.4 %** |
| 150K cold | 73.62 ± 0.11 s | 69.44 ± 0.04 s | **−5.7 %** |
| 150K full task (prefill + 1,500 tokens) | 100.35 ± 0.55 s | 95.17 ± 1.61 s | **−5.2 %** |

- **Decode per round:** level within the spread of three repeats (+0.4 to +3.3 % cold, n = 3). This is neither a
  demonstrated cost nor a gain.
- **Exactness:** every row was byte-identical to local disk. That's 839,040 lookup entries per rank, both layers, RDMA and
  TCP, 0 bad rows, and wrong-identity and out-of-range requests were refused. In-engine module outputs are bf16
  bit-identical to the image's disk path (tested on an RTX 3090).
- **Free-running outputs:** these stay within the engine's own A-vs-A nondeterminism floor (see findings).
- **Reliability:** 0 fallbacks, 0 transport errors and 0 reconnects across all runs. Remote lookups average 50–60 µs at
  decode batch sizes and ~205 µs over whole long-context suites.

**Conditions:**
- Two DGX Spark GB10 nodes, TP2.
- Model: [EXL3 2.75 bpw legal-weighted build](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark).
- Settings: max-model-len 327,680; thinking at effort "high"; T = 1, top_p = 1; cold prefix cache; prompt filler of public
  court opinions; repeats as shown.

## Deploy

1. **On the host:** build with `make -C code/engram/server`, then run the server as a service (an example unit is in
   `code/engram/server/`). It loads in ~70 s and prints table identities.
2. **On each Spark:** run `code/engram/spark/stage_overlay.sh <spark-a> <spark-b>` (copies and verifies the overlay), then
   `code/engram/spark/prebuild_jit.sh <spark-a> <spark-b>` with the serve stopped.
3. **Start the serve** with these container env vars:
   - `PYTHONPATH=<staged>/engram/spark/py`
   - `DSV41_ENGRAM_OVERLAY=<staged>/engram/spark/engram.py`
   - `VLLM_ENGRAM_REMOTE=rdma://<host-ip-a>:<port>@<spark-a-ip>,rdma://<host-ip-b>:<port>@<spark-b-ip>`
   - `VLLM_ENGRAM_REMOTE_TIMEOUT_MS=100`
   - `VLLM_ENGRAM_REMOTE_REQUIRED=0` (production; use 1 in experiments so that a fallback fails the run)
4. **Check it's working:** `docker logs <serve> | grep engram_remote` prints one line per table per minute, with lookups,
   rows, latency, fallbacks and transport errors.

Full details are in `code/engram/INTEGRATION.md`.

## Rollback

Unset `VLLM_ENGRAM_REMOTE` (and the overlay env), then restart the serve. The image's own disk reader takes over unchanged.
Stopping the host server while the serve is up makes every lookup fall back to local disk, which is slower but exact.

## Limits

- The prefill gain is ~5–6 % at 32K–150K; at short prompts it's within noise.
- RAM: 189 GiB resident on the host.
- One host serves both Sparks, each over its own link.
- The Spark's direct link shares its PCIe uplink with one of the TP rails. Measured interference: ~2 % on prefill with
  both links loaded to 35 Gb/s ([findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)).
- Tested with the v5 image only.

## Related entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
  [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)
- **Separate:** [remote 3090 drafter + KV](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) ·
  [two-card NVLink drafter](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) ·
  [async host-flag hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) ·
  [ofl transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) ·
  [bench + interference tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

## Credits and licence

The overlay is derived from the DeepSeek-V4.1 DGX Spark serving stack:
- [HeNryous/deepseek-v41-1017-dgx-spark](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark) (image v5, patches 0022–0050);
- the [sudoingX recipe](https://github.com/sudoingX/dgx-spark-deepseek-v41);
- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0).

Model: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).

Our code is Apache-2.0. Upstream files keep their own licences and headers.

Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
