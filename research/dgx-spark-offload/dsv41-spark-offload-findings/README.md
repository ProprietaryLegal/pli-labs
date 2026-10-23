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
  then the worker, before the model runner can return. That serializes CPU work that used to overlap.
- **Async rounds give most of it back on GB10** (measured 2026-10-23): −0.9 to −3.9 % per round with one drafter card,
  and −3.9 to −6.5 % with the two-card drafter, against synchronous rounds. Time to first token was at parity. Details in
  section 2.
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

### 2b. A proxy thread inside the engine process costs prefill time; a separate process doesn't (measured)

The first working async version ran the round in a thread of the engine process:
- decode was −3.3 to −4.5 % per round;
- time to first token was **+4 % at 32K and +5–8 % at 150K**;
- 31 rounds per benchmark set hit the 200 ms watchdog.

Two suspects were ruled out by their own runs:
- the interpreter switch interval: no change at Python's default 5 ms;
- `CUDA_MODULE_LOADING=EAGER`: synchronous + EAGER had normal time to first token. EAGER also makes every serve start
  take 1,456 s instead of 364 s.

Moving the round into a torch-free proxy process, sharing a `memfd` region registered with `cuMemHostRegister`, removed
the cost. Time to first token went to parity, with 0 watchdog fires. A spinning and a napping proxy measured the same, so
the cost was the in-process placement, not the spin.

### 2c. A review found silent draft corruption; it is fixed

An independent review of the process proxy found the following. If a proxy loses a round to the watchdog but keeps
running, it can write that round's rows into the shared staging the GPU is already refilling for the next round. The
result is **wrong drafts reported with status OK**: on the GPU, 12 of 12 wrong at every stall point tried.

Drafts are verified by the target, so outputs never changed. But wrong drafts silently waste speculation. The fix builds
headers and resync rows in private buffers and sends them as two parts, so the staging is only read. Result: 12 of 12
exact. The same hardening round added:
- start-failure fallback;
- heartbeat stall detection: 3 parked rounds instead of 64;
- bounded respawn: 10,853 drafts exact across three SIGKILLs.

A second reviewer re-ran everything on old and new code. Evidence:
[async entry](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff).

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

**Async proxy drills on GB10 (2026-10-23, two-card worker, live generations):**
- `kill -9` of the proxy: lost after about 1 s, respawned about 2 s later, 3 / 3 requests completed. The drafter worker
  kept its process and took a new session.
- `SIGSTOP`: lost by heartbeat (531 ms against the 400 ms limit), respawned about 2 s later, 2 / 2 completed, no stopped or
  zombie processes left.
- Ten idle minutes produced no false loss.
- Switching the drafter from two cards to one took 12 s.

Details: [async entry](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff).

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
| `CUDA_MODULE_LOADING=EAGER` against the async TTFT cost | not the cause; +18 min per serve start | measured |
| a shorter interpreter switch interval for the proxy thread | not the cause of the TTFT cost | measured |
| the two-card drafter alone, synchronous | −0.6 to −2.6 % per round: real but small without async | measured |

## 7. NVLink pair: peer buffers in PyTorch's expandable segments cause illegal addresses (measured)

The two-card drafter's collective buffers are written by the peer GPU over NVLink. Allocated from PyTorch's caching
allocator with `expandable_segments` on, old code crashed with Xid 31 on both cards; the same code passed with expandable
segments off. Giving the peer state its own `cudaMalloc`, outside the Torch allocator and outside graph capture, fixed it.

A second, independent hazard: PyTorch 2.10 keeps a class-wide default capture stream, so a capture on one card can leave a
stream that a later capture on the other card wrongly uses. The fix is to pass an owned stream at every capture site.

Details: [two-card entry](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink).

## 8. How many drafted tokens a perfect drafter would get accepted (exact, T = 1)

**Why it matters.** Speculative decoding speed depends on accepted drafts per round. At temperature 1, a greedy draft token
d is accepted with probability p_target(d), so a perfect greedy drafter accepts the target's top token with probability
max p.

**How we measured the ceiling exactly.**
1. We re-ran each committed sequence once through the served model with `prompt_logprobs: 20`. That returns the
   target's top-20 next-token logprobs at every position on the exact committed prefix, with one prefill per sequence and
   no engine change. The mass outside the top 20 was 0.11–0.19 %.
2. On the same rounds, we scored:
   - the perfect greedy drafter's expected accepted tokens per round;
   - a lower bound for our deployed drafter;
   - the KL divergence from the target to the drafter, at each draft position.

**Result** (529 rounds of four legal-domain requests, served model an EXL3 2.75 bpw quant of 1017, 5 drafts per round):

| | accepted per round |
|---|---|
| perfect greedy drafter | **3.151** |
| our deployed drafter (lower bound) | **1.831** |
| per request, perfect / deployed | 3.40 / 1.91 · 3.03 / 1.90 · 3.43 / 2.38 · 2.86 / 1.43 |

- The target's mean top-token probability is 0.83–0.84 at every draft position.
- KL(target ‖ drafter) grows with position: 0.65, 1.47, 1.98, 2.83 and 3.36 nats at draft positions 1–5.

**How to read it.**
- The headroom is large: 1.32 tokens per round.
- It sits mostly in the later draft positions, where the drafter guesses several tokens past the last target state it
  sees. Part of that gap is structural, since a drafter can't see future target states.
- Adapting the drafter to this target's own outputs is the experiment we are running next; it is not a result yet.

A cheaper bound we tried first: a "logit lens" on the target's next-to-last-layer input gives a valid floor (2.05). The
exact replay showed the true ceiling is far higher.

## 9. In progress (not proven on GB10)

- **The native 1017 drafter** ([entry](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86)) is fast on
  the real NVLink pair, with acceptance equal to our deployed drafter. Its strict fidelity gate still fails, traced so far
  to single-pass vs online-64 attention, and the fix is in progress.
- **Prefix storage across restarts.** Our overlay of upstream HeNryous 0056/0058/0077/0081 plus a host-RAM tier. Upstream
  reports a 75K-token prefix restored after a restart in 1.3–2.1 s against 33.8 s; that is their claim, not our
  measurement. Our code is CPU-tested and under independent review.
- **Prefill ordering for mixed long requests.** A CPU model predicts the first answer of four concurrent 300K requests at
  about 156 s instead of 620 s, with total times within 3 %. That is a model result, not measured.

- **KV page packing** (upstream HeNryous 0051→0060). Our offline run of the engine's own allocator code reproduces upstream's
  figures exactly and predicts 1,604,926 → 2,502,600 tokens at 6 GiB/rank (3,154,934 at a 1M cap). It is reviewed, and its
  GB10 run is next.
- **1M-token context.** The model supports 1,048,576 positions, and the drafter worker supports `--max-model-len 1048576`
  (one-card and two-card drafts agree at position 900,000). The GB10 needle tests are pending.

## All entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · findings (this page)
- **Separate:** [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) ·
  [remote 3090 drafter + KV](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) ·
  [two-card NVLink drafter](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) ·
  [native 1017 drafter on SM86](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86) ·
  [async host-flag hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) ·
  [ofl transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) ·
  [bench + interference tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

The `code/roundtrip/` folder holds the loopback microbenchmark and the passive contention sampler used for section 1.
`code/dspark/d1/` holds the exact-ceiling replay builder and scorer for section 8 (`exact_ceiling.py`), with
`results/exact_ceiling.json` (aggregates only). Prompts and tokens are not published.

Apache-2.0. Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
