# Two-card (TP2 over NVLink) DSpark drafter worker for DeepSeek-V4.1-Flash

**Source code, not model weights.** One of the separately usable optimizations from PLI Labs' DeepSeek-V4.1-Flash deployment
on two DGX Spark (GB10) nodes with an x86 offload host. It extends the
[remote 3090 drafter](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) from one card to an NVLinked
pair of RTX 3090s. Overview: [DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).

**Status: proven in-engine (−2.5 % per decode round, interleaved A/B); under independent review; not yet the production
default.**

Hugging Face entry: https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink

## What it does

One worker process drives both cards (`--tp 2`) and splits the drafter the way the Spark target splits itself:
- **Vocabulary-sharded** head and Markov, with an argmax exchange each Markov step.
- **Head-split** `wq_b` and attention, **group-split** `wo_a`, **row-parallel** `wo_b`.
- **Channel-split** routed experts (the EXL3 bank at tp_size 2), and a column/row-split shared expert.
- **Replicated:** the embedding, mHC, norms, `wq_a`, `wkv`, the KV ring, routing and confidence.

Collectives use a small custom CUDA extension (`code/dspark/worker/p2p/`): a one-shot push over NVLink with epoch flags,
double buffering, and a sticky error word on timeout. Partial sums are fp32, and a + b is commutative, so both ranks hold
identical bits. Each rank's CUDA graphs are captured separately and replayed in lockstep.

On the wire it is the same contract (CONTRACT v1.8) on the same port. The Spark side doesn't change: swap the worker unit
and the engine reconnects.

## When to use it, and when not

Use it when you already run the remote drafter, have an NVLinked pair free, and want the ~1 ms per round back.

Don't use it if you'd rather keep a spare GPU, are power-limited (the pair peaks at ~710 W at 325 W caps), or have no NVLink:
PCIe peer copies would erase the gain. If one card does the job, use one card.

## Requirements

- Two RTX 3090s with an NV4 bridge. Peer access must be enabled.
- About 5.2 and 4.5 GiB reserved on the two cards.
- The same drafter tensors and receipt as the one-card worker.

## Measured effect

**In-engine, on the production serve.** D-kv6, two DGX Spark GB10, the
[EXL3 2.75 bpw legal-weighted build](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark).
Only the worker was swapped; TP2 and one-card were interleaved; step time is a direct per-step timer (gaps between SSE
events), p50 per request:

| workload | TP2 step (ms) | one-card step (ms) | change |
|---|---|---|---|
| short × 3 | 38.5 / 38.0 / 38.4 | 38.5 / 40.0 / 39.4 | −2.5 % |
| 32K × 3 | 41.1 / 41.9 / 41.6 | 42.6 / 41.9 / 43.2 | −2.5 % |
| 32K, 3 concurrent | 73.3 / 73.6 / 74.2 | 75.6 / 75.3 / 76.0 | −2.6 % |
| 150K × 2 | 40.2 / 40.9 | 41.9 / 41.2 | −2.4 % |

6,142 drafted steps, worker p50 2.44 ms (one card ~3.23 ms), 0 TP communication errors.

**Worker in isolation** (CUDA-graph replay p50, both cards at 325 W):

| requests, n_out | one card | TP2 | change |
|---|---|---|---|
| 1, 5 | 2.94 ms | 1.94 ms | −34 % |
| 2, 5 | 3.67 ms | 2.42 ms | −34 % |
| 3, 4 | 4.10 ms | 2.67 ms | −35 % |
| 4, 4 | 4.89 ms | 3.17 ms | −35 % |

- Collectives are 3.6–4.7 % of the step.
- Full STEP over RDMA loopback: 3.14 → 2.34 ms at 1 request, 5.01 → 3.67 ms at 4 requests.

**Parity:**
- Real pair vs single-device emulation: bit-identical.
- Agreement with the engine's own drafter on live captures: 92.6–93.4 % of blocks, the same as the one-card worker. Every
  first mismatch is a numeric near-tie.
- One-card vs TP2: drafts agree 1.000 at positions 0 and 900,000. Confidence is identical.
- Power: pair peak 709.5 W, NVLink replay/recovery/CRC errors 0.

## Deploy

1. Stop the one-card unit, then `code/dspark/worker/run/start_tp2.sh`. It refuses if either card is busy, installs the
   `--tp 2` unit (`Conflicts=` the one-card unit, same port) and waits for the ready line.
2. Lower `VLLM_DSPARK_REMOTE_DRAFT_MS` by ~0.8 ms for the depth policy (it was calibrated on one card).

Design and numbers: `code/dspark/worker/TP2.md`.

## Rollback

Run `stop_tp2.sh` and start the one-card unit. The engine reconnects within ~5 s; meanwhile it decodes without speculation.

## Limits

- −2.5 % per round on its own: below a 5 % promotion bar, but consistent across 1–3 streams and 32K–150K.
- Insert is ~0.1 ms slower than one card (host overhead).
- Test-tool-only issue: building the one-card model on rank 1's card in the same process before TP2 causes an illegal
  address in the first collective. The worker never does this. The root cause is under review.

## Related entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
  [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)
- **Separate:** [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) ·
  [remote 3090 drafter + KV](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) ·
  [async host-flag hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) ·
  [ofl transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) ·
  [bench + interference tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

## Credits and licence

EXL3 decoding vendored from [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT). Our code is Apache-2.0.

Model: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (MIT).

Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
