# GB10 async-round deadlocks (CUDA lazy loading vs a parked stream) and the host-flag hand-off that removes them

**Source code and a finding, not model weights.** One of the separately usable pieces from PLI Labs' DeepSeek-V4.1-Flash
deployment on two DGX Spark (GB10) nodes with an x86 offload host. Overview:
[DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).

**Status:**
- The deadlocks are reproduced and root-caused. The fix is proven deadlock-free on an RTX 3090.
- GB10 speed is **not yet measured**: the GB10 run is pending.
- The fix is under independent review.

Hugging Face entry: https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff

## The problem

With a remote drafter, synchronous rounds make the engine's CPU wait for the network, which loses the overlap the local
drafter had. The async design avoids that:
1. The engine stream parks on a device flag (`cuStreamWaitValue32`).
2. A proxy thread does the network round.
3. The flag is released after the reply lands.

On GB10 this deadlocked twice:

1. **Self-test, rank 0 stuck in `cuLaunchKernel`.** The main stream was parked on the flag, then the CPU launched a kernel
   for the first time (`torch.cuda._sleep`). Under CUDA's default **lazy module loading**, a first launch loads the kernel's
   module and waits for the device. The device can't go idle while the stream is parked, so it deadlocks. This reproduces
   on an RTX 3090 (`code/results/C/lazy-repro/repro.py`): the original self-test hangs, and passes in 0.02 s with
   `CUDA_MODULE_LOADING=EAGER` or with every kernel pre-launched.
2. **Warm-up, with `CUDA_MODULE_LOADING=EAGER`.** gdb shows:
   - the engine thread spinning in `cuModuleLoadData`, called from Triton's `loadBinary` (a first-use JIT kernel; eager
     loading does not cover explicit module loads);
   - the proxy thread blocked in `cuMemcpyHtoDAsync`, the reply copy that precedes the release.

   The release needs that copy, and the copy waits behind the module load. **Any CUDA call on the proxy thread between
   the engine's wait and the release can deadlock against a module load**, so eager loading is not enough.

## The fix: a CUDA-free proxy thread

The engine stream and the proxy thread meet through two 32-bit counters in pinned, host-mapped memory. The Engram disk
reader on these nodes already uses the same handshake.

- **Engine stream:** gather → D2H of the rows into pinned memory → **write `gathered = t`** (`cuStreamWriteValue32` on the
  host-mapped word) → **wait `released ≥ t`** (`cuStreamWaitValue32` on host memory) → copy the reply in.
- **Proxy thread:** spin on `gathered` in a small C library (`hostflag.c`: acquire/release atomics, no CUDA) → network round →
  write the reply into pinned memory → **store `released = t`**. It never calls CUDA or torch, and a test fails if it does.
- **Failures** (timeout, breaker, enqueue error, disconnect) still release the stream, again with no CUDA call.
- **Startup self-test:** exercises the mechanism with every kernel pre-launched. It refuses async, logging the reason, if the
  device can't use host pointers for stream memory operations.

## Evidence (RTX 3090, mock worker on localhost)

- **Deadlock regression:**
  - The old design deadlocked at the first fresh `cuModuleLoadData` while a round was parked; it was killed at 90 s.
  - The new design completed 30 of 30 such rounds, 89/89 drafts exact. Each load waited ~41 ms (the injected worker delay),
    then returned.
- **Soak:**
  - 10,000 rounds in 26 s, pipelined 3 deep, with 270 fresh module loads during parked rounds and 49 injected faults
    (delays past the timeout, dropped replies, worker restarts).
  - 19,082 drafts bit-exact; 9,319 failed-request rows correctly zero.
  - Proxy round p50 664 µs, against 861 µs for the old design under the same fault schedule.
- **Suites:** 169 CPU tests pass, and the drafter contract conformance suites are unchanged.

## When to use it, and when not

Use it when an inference engine parks a CUDA stream on a host- or device-side flag that another thread releases. That goes
for remote drafters, remote KV or embedding fetches, and RDMA proxies. On GB10 (coherent NVLink-C2C) and on discrete GPUs
that report `CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_MEM_OPS`.

Don't use it as a substitute for checking ordering on your platform: the CPU's release store must become visible before the
GPU observes the flag. Don't assume `CUDA_MODULE_LOADING=EAGER` alone makes a CUDA-calling proxy safe; it doesn't.

## Deploy / rollback

- **Deploy:** build `libdsv41_hostflag.so` (`cc -O2 -shared -fPIC hostflag.c`) next to the transport library, and set
  `VLLM_DSPARK_REMOTE_ASYNC=1` (see the remote drafter entry). The pass check is the log line
  `remote DSpark async mode on (host-flag hand-off self-test passed; …)`.
- **Rollback:** unset `VLLM_DSPARK_REMOTE_ASYNC` to get synchronous rounds.

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
