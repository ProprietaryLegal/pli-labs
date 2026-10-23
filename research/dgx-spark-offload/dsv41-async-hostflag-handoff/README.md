# Async remote drafting on GB10: the deadlock finding, the CUDA-free hand-off, and the out-of-process proxy

**Source code and findings, not model weights.** One of the separately usable pieces from PLI Labs' DeepSeek-V4.1-Flash
deployment on two DGX Spark (GB10) nodes with an x86 offload host. Overview:
[DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).

**Status (updated 2026-10-23):**
- **Measured on GB10.** With the round in a separate proxy process, decode is −0.9 to −3.9 % per round with one drafter
  card, and **−3.9 to −6.5 % with the two-card drafter**, against synchronous rounds. Time to first token is at parity, with
  0 watchdog fires.
- **Hardened and independently reviewed.** Five defects found in review are fixed, each with a test that fails on the old
  code: late-proxy draft corruption, start failure, stall detection, respawn and shutdown. The reviewer re-ran every probe
  and cleared the runtime code for GB10.
- **Not yet the production default.** The production switch-over checks are running now.


Hugging Face entry: https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff

## The problem

With a remote drafter, synchronous rounds make the engine's CPU wait for the network, which loses the overlap the local
drafter had. The async design avoids that:
1. The engine stream parks on a device flag (`cuStreamWaitValue32`).
2. A proxy does the network round.
3. The flag is released after the reply lands.

On GB10 the first version deadlocked twice:

1. **Self-test, rank 0 stuck in `cuLaunchKernel`.** The main stream was parked on the flag, then the CPU launched a kernel
   for the first time (`torch.cuda._sleep`). Under CUDA's default **lazy module loading**, a first launch loads the kernel's
   module and waits for the device. The device can't go idle while the stream is parked, so it deadlocks. This reproduces
   on an RTX 3090 (`code/results/C/lazy-repro/repro.py`): the original self-test hangs, and passes in 0.02 s with
   `CUDA_MODULE_LOADING=EAGER` or with every kernel pre-launched.
2. **Warm-up, with `CUDA_MODULE_LOADING=EAGER`.** gdb shows:
   - the engine thread spinning in `cuModuleLoadData`, called from Triton's `loadBinary` (a first-use JIT kernel; eager
     loading does not cover explicit module loads);
   - the proxy thread blocked in `cuMemcpyHtoDAsync`, the reply copy that precedes the release.

   **Any CUDA call on the proxy side between the engine's wait and the release can deadlock against a module load**, so
   eager loading is not enough.

## The fix, step 1: a CUDA-free hand-off

The engine stream and the proxy meet through 32-bit counters in pinned, host-mapped memory:

- **Engine stream:** gather → D2H of the rows into pinned memory → **write `gathered = t`** (`cuStreamWriteValue32` on the
  host-mapped word) → **wait `released ≥ t`** (`cuStreamWaitValue32` on host memory) → copy the reply in.
- **Proxy:** spin on `gathered` in a small C library (`hostflag.c`: acquire/release atomics, no CUDA) → network round →
  write the reply into pinned memory → **store `released = t`**. It never calls CUDA or torch, and a test fails if it does.
- **Failures** (timeout, breaker, enqueue error, disconnect) still release the stream, again with no CUDA call. A watchdog
  releases any round older than its deadline (200 ms) and the round decodes target-only.
- **Startup self-test** exercises the mechanism with every kernel pre-launched. It refuses async, logging the reason, if
  the device can't use host pointers for stream memory operations.

On an RTX 3090 (mock worker) the old design deadlocked at the first fresh `cuModuleLoadData` during a parked round. The new
one completed 30 of 30 such rounds and a 10,000-round faulted soak with 19,082 drafts bit-exact.

## The fix, step 2: the round runs in its own process (GB10 finding)

**The in-process version cost time to first token on GB10.** It ran the hand-off proxy as a thread inside the engine
process. Measured against synchronous rounds on the same setup (2–3 repeats per workload):
- decode was −3.3 to −4.5 % per round against synchronous rounds;
- but time to first token was **+4 % at 32K and +5–8 % at 150K**;
- 31 rounds per benchmark set hit the watchdog.

We ruled out the two obvious suspects, each by its own run:
- the interpreter switch interval: no change at the default 5 ms;
- `CUDA_MODULE_LOADING=EAGER`: synchronous + EAGER had normal time to first token. EAGER also makes every serve start take
  1,456 s instead of 364 s.

**What fixed it: the proxy as a separate, torch-free process.**
- The two sides share one `memfd` region: hand-off words, staging rows and replies. It is registered with the GPU through
  `cuMemHostRegister` and works on GB10's coherent memory.
- The proxy process imports no torch. It is 32 MB resident, with all 20 cores allowed. It uses the same `hostflag.c`
  waits, with a short spin and then a nap.
- Measured on GB10 (window 4; direct per-step timer, p50 per request, 1,500-token budgets, T = 1):

| workload | sync, one card | async process, one card | async process, two cards (NVLink TP2) |
|---|---|---|---|
| short × 3, step (ms) | 39.6 / 40.1 / 39.4 | 38.3 / 38.1 / 38.3 (−3.5 %) | 37.1 / 37.4 / 36.9 (**−6.5 %**) |
| 32K × 3, step (ms) | 41.2 / 42.2 / 42.3 | 40.2 / 40.6 / 41.3 (−2.4 %) | 39.6 / 39.5 / 40.2 (**−4.5 %**) |
| 32K, 3 concurrent, step (ms) | 75.9 / 76.1 / 76.1 | 75.1 / 75.6 / 75.2 (−0.9 %) | 72.8 / 73.5 / 73.0 (**−3.9 %**) |
| 150K × 2, step (ms) | 41.5 / 42.2 | 40.5 / 39.9 (−3.9 %) | 39.6 / 39.8 (**−5.0 %**) |
| 32K TTFT | 14.56–14.59 s | 14.53–14.54 s | 14.49–14.63 s |
| 150K TTFT | 69.10–69.11 s | 69.24–69.74 s | 68.89–69.09 s |

The one-card async 32K TTFT excludes the first request after a fresh start (15.67 s, lazy module loading; see below).

**Counters over the set:**
- 0 watchdog fires and 0 preempted rounds, against 31 per set for the thread version;
- engine-side submit p50 140 µs, p99 362 µs;
- the first request after a fresh start pays lazy module loading (0.55 s short, about 1 s extra at 32K). A one-request
  warm-up at serve start absorbs it.

**Spin and nap are equal** (a spinning variant measured within noise of the napping one at every workload), so the cost
came from running the round inside the engine process, not from the spin. The nap variant frees a core.

## Hardening (found in review, fixed, re-reviewed)

| defect | what it was | fix | evidence (old code → new code) |
|---|---|---|---|
| late-proxy draft corruption | a proxy that lost a round to the watchdog could still write that round's rows into the shared staging the GPU was refilling for the next round: wrong drafts with status OK | headers and resync rows built in private buffers, sent as two parts; the staging is only read | GPU probe: 12 of 12 wrong drafts → 12 of 12 exact, at all three stall points |
| start failure | a proxy that could not start crashed the engine | synchronous rounds with the reason recorded (unless strict start); the shared region is freed on every refusal path | 0 fds / 0 mappings left on all four CUDA refusal paths |
| stall detection | a stopped proxy cost 64 parked rounds | heartbeat every ≤ 100 ms; lost when older than max(2 × deadline, timeout + 250 ms) = 400 ms, or after 3 fires with no round finished | 64 parked rounds (1.40 s) → 3 (0.07 s) at a 20 ms deadline; a slow but live proxy is never declared lost |
| respawn | a lost proxy left the engine target-only for the rest of the run | the GPU stream is drained (event query), the hand-off words reset, and a new proxy starts with a new worker session; at most 3 respawns, restored after an hour | GPU soak 6,000 rounds with a SIGKILL every 2,000: **10,853 drafts exact, 0 wrong**, 3 respawns |
| shutdown | losing a proxy blocked the engine thread (3 s in the test) | non-blocking kill and reap; the speculator closes its round at shutdown | worst submit < 0.3 s |

**Cost of the hardening:** engine-side submit +5.6 to +9.0 µs, about 0.02 % of a 38 ms decode step. Proxy round and
network time are unchanged within run-to-run noise.

**Independent review:**
- It re-ran each probe on the old and new code, plus the paths the test suite did not cover:
  - the process proxy sending over the RDMA transport's client: 0 exact / 12 wrong → 12 / 0;
  - 20,000 real resets against the live C watchdog: no fire;
  - 80 simulated container freezes: 0 false alarms;
  - two more kill soaks: 5,608 and 5,307 drafts exact, 0 wrong.
- CPU suite: 222 passed, 1 skipped (196 / 1 before).
- Its remaining notes concern the production runbook (GPU ownership after promotion, check ordering), not the runtime.

## When to use it, and when not

Use it when an inference engine parks a CUDA stream on a host- or device-side flag that something else releases: remote
drafters, remote KV or embedding fetches, RDMA proxies. It works on GB10 (coherent NVLink-C2C) and on discrete GPUs that
report `CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_MEM_OPS`.

Don't use it as a substitute for checking ordering on your platform: the CPU's release store must become visible before the
GPU observes the flag. Don't assume `CUDA_MODULE_LOADING=EAGER` makes a CUDA-calling proxy safe; it doesn't. Prefer the
process proxy over an in-process thread, because the thread version costs time to first token.

## Deploy / rollback

- **Deploy:**
  1. Build `libdsv41_hostflag.so` (`cc -O2 -shared -fPIC hostflag.c`) next to the transport library.
  2. Set `VLLM_DSPARK_REMOTE_ASYNC=1 VLLM_DSPARK_REMOTE_ASYNC_PROXY=process VLLM_DSPARK_REMOTE_ASYNC_SPIN_US=50
     VLLM_DSPARK_REMOTE_ASYNC_NAP_US=20 VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS=3 VLLM_DSPARK_REMOTE_ASYNC_RESPAWN_RESET_S=3600`.
     The remote drafter entry has the rest of the config.
  3. Leave `CUDA_MODULE_LOADING` at its default.
  4. Pass checks, in the rank-0 log: `remote DSpark async mode on (host-flag hand-off self-test passed; …)`, then
     `remote DSpark proxy process running`.
- **Rollback:** unset `VLLM_DSPARK_REMOTE_ASYNC` to get synchronous rounds.

Design notes: `code/dspark/spark/ORDERING.md` (word protocol, races, respawn) and `OVERLAP.md` (what the engine overlaps).

## Related entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
  [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)
- **Separate:** [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) ·
  [remote 3090 drafter + KV](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) ·
  [two-card NVLink drafter](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) ·
  [ofl transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport) ·
  [bench + interference tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

Apache-2.0. Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
