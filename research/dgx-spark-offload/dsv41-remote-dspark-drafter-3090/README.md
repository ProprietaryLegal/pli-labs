# Remote DSpark drafter on an RTX 3090, with the freed Spark memory spent on KV cache

**Source code, not model weights.** One of the separately usable optimizations from PLI Labs' DeepSeek-V4.1-Flash deployment
on two DGX Spark (GB10) nodes with an x86 offload host. Overview and combined results:
[DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).

**Status: proven and in production** (configuration "D-kv6", since 2026-10-23).

Hugging Face entry: https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090

## What it does

DeepSeek-V4.1-Flash-1017 ships a 3-layer DSpark drafter (mtp.0–2: sliding window 128, block 5, Markov rank-256 head, 128
experts top-3). The DGX Spark recipe runs it on the Sparks, next to the 40-layer target. All-in (weights, CUDA graphs,
buffers, draft KV) it costs about 5 GiB of unified memory per Spark.

This package moves the drafter to an RTX 3090 on the x86 host:

- **Worker** (`code/dspark/worker/`): a standalone PyTorch drafter.
  - Bit-exact EXL3 decoding of the drafter's own K2 experts, K5 dense layers and K6 head.
  - CUDA graphs for 1–4 requests and every n_out, with a 128-row context window per slot.
  - Speaks the CONTRACT v1.8 wire protocol over the [ofl](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) RDMA
    transport.
  - Safety:
    - It refuses tensors not covered by its signed receipt.
    - It validates positions, generations and replies.
    - It exits for a systemd restart if its CUDA backend dies.
- **Spark overlay** (`code/dspark/spark/`): a `remote_dspark` speculator package, plus one guarded branch in vLLM's
  `spec_decode/__init__.py`. When `VLLM_DSPARK_REMOTE` is unset, the image path runs unchanged.
  - Rank 0 gathers the target's tap rows (layers 37/38/39) with one fused Triton kernel, sends them, and broadcasts the
    returned drafts and confidences to rank 1.
  - Replies are validated (token ids, confidence range, shapes, generations) before the engine consumes them. Failures give
    zero drafts on both ranks.
  - A bounded circuit breaker trips after 3 failures, cools down for 256 rounds, then probes.
  - The reconnect backoff is capped.
- **Policies:**
  - `shadow` compares remote against local drafts.
  - `resilience` falls back to the local drafter.
  - `capacity` removes the drafter from the Sparks entirely: no draft weights, Markov cache, draft graphs or draft KV layers.

**The memory win comes from `capacity`.** The deployment then raises the KV pool from 3.75 to 6.0 GiB per rank.

## When to use it, and when not

Use it when you need context room or concurrency more than single-stream speed: long legal documents, several long requests
at once, prompts near the 300K limit. You'll also need a host with a 3090-class GPU (~8.3 GiB of VRAM) and a low-latency
direct link.

Don't use it when single-request latency is all that matters. One-stream decode costs 3–9 % in synchronous mode (see below;
the [two-card worker](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) and
[async hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) recover part of it). Don't use it
without a reliable host, either: if the worker is down the serve keeps answering, but without speculation (~28 ms per token).

## Requirements

- RTX 3090 (24 GB). The worker uses 8.1–8.3 GiB including a 2 GiB Markov cache.
- The drafter tensors from the model checkpoint, copied to the host (sha256 receipt).
- The ofl transport on both sides, and the image `ghcr.io/henryous/deepseek-v41-1017-dgx-spark:v5`. The overlay checks the
  image's `spec_decode/__init__.py` hash.

## Measured effect

Two DGX Spark GB10, TP2. Model: the [EXL3 2.75 bpw legal-weighted build](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark).
T = 1, thinking at effort "high", 1,500-token budgets, cold prefix cache, n = 2–3 repeats. A = the stock recipe; D-kv6 =
remote Engram + this drafter (capacity) + KV 6 GiB/rank.

| | A (stock) | D-kv6 | change |
|---|---|---|---|
| KV pool | 989,460 tokens | **1,604,926 tokens** | **+62 %** |
| max. concurrency at 327,680 tokens/request | ~3.0× | **4.90×** | |
| 4 × 300K requests at once | does not fit | **all 4 completed** (TTFT 620–640 s, 16.9–23.4 tok/s each) | new capability |
| MemAvailable per Spark while serving | 5–7 GiB | 5–8 GiB | same headroom |
| short prompt, stream tok/s | 55.96 ± 1.65 | 51.07 ± 0.78 | −8.7 % |
| 32K, stream tok/s / TTFT | 64.57 / 15.53 s | 60.82 / 14.81 s | −5.8 % / −4.6 % |
| 150K, stream tok/s / TTFT / task | 56.10 / 73.62 s / 100.35 s | 54.23 / 68.79 s / 96.46 s | −3.3 % / −6.6 % / **−3.9 %** |
| 32K × 2 / × 3 aggregate | 40.59 / 47.21 tok/s | 40.81 / 47.36 tok/s | level |

- The TTFT and task gains include [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma).
- Drafter alone (C, capacity, KV unchanged): +7–9 % per decode round at one stream, ≤ 2 % under concurrency. Model load is
  2.5 GiB smaller per rank.
- Acceptance is unchanged: shadow parity vs the local drafter is 0.99 / 0.97 / 0.96 / 0.95 / 0.93 by position.
- Remote round trip: p50 3.76 ms, of which the worker is 3.44 ms. 0 failed rounds outside the drill.
- Prefix caching is unaffected: a 36,016-token repeat gets 35,968 cached tokens, the same as the stock drafter.
- **Failure drill:** the worker was stopped mid-request.
  - The request finished, with target-only steps at 27.7 ms instead of drafted steps at 39.6 ms.
  - It reconnected automatically once the worker was back, and the backoff cap is now 5 s.
  - The breaker never tripped.

## Deploy

1. **Host:** copy the drafter tensors with a receipt. Install the worker as a systemd unit, pinned to one card
   (`DSPARK_WORKER_GPU_UUID`, `--max-model-len` up to 1,048,576). It binds only to your direct-link addresses: edit
   `DIRECT_LINK_IPS`.
2. **Sparks:** stage the overlay with `code/dspark/spark/stage_overlay.sh` (sha-checked) and bind-mount it read-only.
   - `remote_dspark/` goes into vLLM's `spec_decode/`, and the overlay replaces that directory's `__init__.py`.
   - The ofl directory goes to `/opt/dsv41-ofl`.
3. **Serve env:**
   - `VLLM_DSPARK_REMOTE=o1rdma://<host-ip>:<port>`
   - `VLLM_DSPARK_REMOTE_BIND=<spark-a-ip>`
   - `VLLM_DSPARK_REMOTE_POLICY=capacity`
   - the drafter and Markov identities
   - `VLLM_DSPARK_REMOTE_ROUND_MS=<verify-only round costs>`, `VLLM_DSPARK_REMOTE_DRAFT_MS=<remote draft cost>`
   - `VLLM_DSPARK_REMOTE_TIMEOUT_MS=50`
   - `VLLM_DSPARK_REMOTE_BACKOFF_MAX_MS=5000`
   - then raise the KV pool (`--kv-cache-memory-bytes`, 6 GiB/rank here).
4. **Check it's working:** the log shows `remote DSpark (capacity policy) rank 0 -> …` and `GPU KV cache size`. Rank 0's stats
   JSON has rounds_ok, fallback_* and rtt.

The full procedure is in `code/dspark/INTEGRATION.md`; the wire contract is in `code/dspark/CONTRACT.md`.

## Rollback

Unset `VLLM_DSPARK_REMOTE*`, remove the overlay mounts, restore the KV size, and restart. The image's local drafter returns
unchanged. Only the drafter loses its remote path; remote Engram is independent.

## Limits

- Single-stream decode cost: −3 to −9 %, with synchronous rounds on GB10.
- One worker serves one engine (rank 0 talks, rank 1 receives the broadcast), up to 4 concurrent requests.
- Greedy drafts only. Draft top-k, watermarking, PP/DP/CP are refused at startup.
- Tested with image v5 and this checkpoint's drafter identity only.

## Related entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
  [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)
- **Separate:** [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) ·
  [two-card NVLink drafter](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) ·
  [native 1017 drafter on SM86](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86) ·
  [async host-flag hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) ·
  [ofl transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) ·
  [bench + interference tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

## Credits and licence

- The worker vendors parts of [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT) for EXL3 decoding
  (`code/dspark/worker/exl3/VENDORED.md`).
- The overlay modifies a [vLLM](https://github.com/vllm-project/vllm) file (Apache-2.0), as built by
  [HeNryous](https://github.com/HeNryous/deepseek-v41-1017-dgx-spark) and [sudoingX](https://github.com/sudoingX/dgx-spark-deepseek-v41).
- Model: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).

Our code is Apache-2.0.

Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
