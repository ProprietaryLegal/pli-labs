# Findings: offloading DeepSeek-V4.1-Flash from 2× DGX Spark to an x86 host — what costs what, what broke, what didn't help

**Combined findings entry.** Hardware, configuration and the full results table are in
[DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
Each optimization has its own entry, linked at the bottom. Labels used throughout: **measured** = our runs; **upstream** =
reported by the original authors; **estimate** = engineering arithmetic.

Hugging Face entry: https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings

## 1. The remote drafter's round cost: lost overlap, not the link

Synchronous remote drafting added **2.5–3.3 ms per decode round** at one stream (measured: +7.3 % short, +6.1 % 32K,
+8.5 % 150K; acceptance-normalized, n = 2–3). We broke the round down using paired timings from the source and stats.
Medians from different runs were never subtracted from each other: histogram percentiles are bin upper edges and don't add.

- **The worker on the 3090** takes p50 3.44 ms per round. The local drafter on GB10 takes p50 4.47 ms (CUDA events). So
  drafting itself is *faster* remotely.
- **The wire and the client**, outside the worker's own timer, add ~0.15 ms (shadow means: 4.118 − 3.973 ms). Bookkeeping
  outside gather and RPC adds another ~0.15 ms.
- **The cable is a tiny share.** A worst-case 6-row request is 184,416 bytes:
  - it serializes in 36.9 µs at 40 Gb/s and 14.8 µs at 100 Gb/s;
  - so a 100 Gb cable saves **≤ 0.022 ms per round**, about 0.05 % of a 42.7 ms round;
  - one request every 42.7 ms uses ~0.09 % of the link.
- **The remaining cost is lost overlap.** The local drafter queues GPU work and returns, so the engine's CPU prepares the next
  step while drafting runs. The synchronous remote round makes rank 0's CPU wait for target verification, then the network,
  then the worker, before the model runner can return. That serializes CPU work that used to overlap. The async design
  restores the overlap; its GB10 measurement is pending.
- **Other work on the host is not the cause.** We sampled the host passively during real worker steps:
  - The worker's connection thread spent 2.6 ms waiting for a CPU across 262 steps, about 0.010 ms per step.
  - The host was 94–96 % idle. IRQ deltas were negligible. PCIe traffic was ≤ 23 MB/s. There was no swap or reclaim.
  - The CPU governor was already `performance`.
  - The co-tenant GPU jobs, a TTS server on other cards, were not responsible.

## 2. GB10 async-round deadlocks

A design that parks the engine stream on `cuStreamWaitValue32` while a proxy thread calls CUDA deadlocks on GB10. Any
first-time module load blocks behind the parked stream, and that covers both lazily loaded kernels and Triton's explicit
`cuModuleLoadData`. `CUDA_MODULE_LOADING=EAGER` fixes the first case only. The fix is a proxy thread that makes no CUDA calls
at all, using host-mapped counters. Details and evidence:
[dsv41-async-hostflag-handoff](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff).

## 3. Link interference with the TP fabric

On the Spark, the direct port to the host shares its PCIe uplink with one TP rail. We ran paced bulk RDMA on both host links
during the benchmark (measured):

| load per link (achieved) | 32K TTFT | 150K TTFT (load covers 60 of the 69 s prefill) | decode step p50 |
|---|---|---|---|
| none (repeats) | 14.55 s | 68.88 s | 41.9–42.4 ms |
| 10 Gb/s (9.93) | 14.59 s (+0.3 %) | — | 42.8 ms |
| 35 Gb/s (34.7) | 14.91 s (+2.5 %) | 70.31 s (+2.1 %) | 42.6 ms |

Near-saturating bulk traffic costs ~2 % of prefill. Normal Engram plus drafter traffic is far below 10 Gb/s, so no interface
change is warranted. A second PCIe function on the Spark port failed queue-pair setup as configured; we didn't pursue it.

## 4. Failure drill

We stopped the drafter worker 25 s into a 6,000-token request and started it again 31 s later (measured):
- The request completed (4,758 tokens, finish "stop", no error).
- Steps went from 39.6 ms drafted (~2 tokens each) to 27.7 ms target-only (1 token each) during the outage.
- Drafting resumed automatically.
- Counters: 2,101 rounds not ready, 1 timeout, 5 connect failures, breaker never tripped.

Reconnection came 20 s after the worker was ready, because the backoff doubles up to 30 s. We cap it at 5 s in production.

## 5. Exactness and nondeterminism methodology

- **The engine isn't run-to-run deterministic at long context, even against itself.** With greedy decoding, thinking off and
  a fresh cache, the stock configuration diverges from itself at token 34 (8K) and token 4 (32K). Prompt top-5 logprob
  deltas reach 11.9. So free-running output comparisons can't certify an offload.
- **What we used instead:**
  - byte-exact comparisons at the boundary: 839,040 Engram lookup entries per rank against local disk, 0 bad rows;
  - in-module bit-identity checks;
  - drafter parity on captured live rounds: 0.99 / 0.97 / 0.96 / 0.95 / 0.93 by position, 0.909 for full blocks;
  - generated text compared only against the A-vs-A floor;
  - acceptance and cached-token checks.
- **Planned next:** teacher-forced and fixed-prefix comparisons with full-distribution KL.
- **The ms/round diagnostic is only a diagnostic.** `1000 × (1 + accepted/round) / tok/s` assumes compatible counters and one
  bonus token per round, and EOS, mixed scheduling and adaptive depth all break that. We added a **direct per-step timer**
  (gaps between SSE events) and use it for worker A/Bs. The A/Bs are interleaved on the same running server.
- **Matched suite:** the same prompts (public court opinions and filings), T = 1, top_p = 1, thinking at effort "high",
  priority 10, per-request cache salts, 1,500-token budgets plus one full task, finalists at 256K/300K and 4 × 300K.

## 6. What did not help, or not on its own

| idea | result | label |
|---|---|---|
| 100 Gb/s host links for the drafter | ≤ 0.022 ms per round | estimate from bytes on the wire |
| CPU pinning / IRQ isolation on the host | ~0 at one stream (runqueue budget 0.010 ms/step) | measured |
| shorter spin / event wait in the transport | +0.068 ms per round, saves one CPU core | measured |
| sending projected state instead of raw taps | saves ~25 µs of wire, adds ~47 MiB + projection work back on the Spark | estimate |
| remote drafter without spending the freed memory | +7–9 % per round at one stream, no benefit | measured |
| remote Engram for decode | level within noise; the gain is prefill only | measured |
| `CUDA_MODULE_LOADING=EAGER` for async safety | insufficient: Triton loads modules explicitly | measured (gdb) |

## 7. In progress (not proven on GB10)

- **KV page packing** (upstream HeNryous 0051→0060). Our offline run of the engine's own allocator code reproduces upstream's
  figures exactly and predicts 1,604,926 → 2,502,600 tokens at 6 GiB/rank (3,154,934 at a 1M cap). It is reviewed, and its
  GB10 run is next.
- **1M-token context.** The model supports 1,048,576 positions, and the drafter worker supports `--max-model-len 1048576`
  (one-card and two-card drafts agree at position 900,000). The GB10 needle tests are pending.
- **Async hand-off speed on GB10.**

## All entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · findings (this page)
- **Separate:** [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) ·
  [remote 3090 drafter + KV](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) ·
  [two-card NVLink drafter](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) ·
  [async host-flag hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) ·
  [ofl transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) ·
  [bench + interference tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

The `code/roundtrip/` folder holds the loopback microbenchmark and the passive contention sampler used for section 1.

Apache-2.0. Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
