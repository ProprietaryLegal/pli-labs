# ofl: a small request/response transport over RDMA RC (RoCE v2) and TCP, with responses placed into caller memory

**Source code.** This is the transport under PLI Labs' DeepSeek-V4.1-Flash offloads:
[remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) and the
[remote drafter](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090). Overview:
[DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).

**Status: proven.** In production on both links since 2026-10-22.

Hugging Face entry: https://huggingface.co/ProprietaryLegal/ofl-rdma-transport

## What it does

One client talks to one server over a connection that carries requests `(op, generation, payload)` and responses
`(status, bytes)`. Each response is **placed directly into caller-owned destination ranges**: up to 16 ranges of registered
memory.

- **RDMA backend:**
  - A request is one SEND from a registered client slot into a pre-posted server receive slot.
  - The server RDMA-WRITEs the response bytes into the client's ranges, pipelined while the handler is still producing them,
    then sends a completion.
  - The client busy-polls its completion queue against a deadline.
  - A TCP bootstrap socket gives liveness: on EOF the server tears down the queue pair.
- **TCP backend:** the same API, used for debugging and as a fallback.
- **API:**
  - Several requests can be in flight (`submit`/`wait`), and requests can be zero-copy via `ofl_client_payload_buf`.
  - Timeouts are fatal per connection, so a late response can never land in reused memory.
  - Device, port and GID are discovered from the local IPv4 of the direct link, which keeps traffic off other fabrics.
- **Memory:** host memory only, with no GPUDirect. NVIDIA's DGX Spark porting guide says GPUDirect RDMA is not supported on
  GB10; use `cudaHostAlloc` + `ibv_reg_mr`. No relaxed ordering on any memory region: the header must land after the data.

Code: `ofl.h` (the API), `ofl.cpp` (both backends, ~1.1k lines, depends only on libibverbs + pthreads), `ofl.py` (ctypes),
`ofl_pingpong.cpp` (self-test + microbenchmark), `ofl_probe.c` (read-only RDMA discovery), `test_ofl.py` (24 checks, including
timeout, late data and reconnect), `API.md`.

## When to use it, and when not

Use it for small-to-medium request/response RPC between a GB10 node (or any host) and a peer over a direct RoCE link, when
the response must land in pre-registered buffers with microsecond-level overhead.

Don't use it as a general-purpose messaging layer. It is one client per connection, strictly request/response, with no
encryption: bind it to direct, private links only.

## Measured (ConnectX-7 on DGX Spark ↔ ConnectX-5 Ex on an x86 host, 40 Gb/s DAC, MTU 9000)

| | Spark A link | Spark B link |
|---|---|---|
| 64 B RDMA send, average | 1.38 µs | 1.38 µs |
| 64 MiB RDMA write, Spark → host | 39.23 Gb/s | 39.23 Gb/s |
| 64 MiB RDMA write, host → Spark | 39.21 Gb/s | 39.21 Gb/s |

- Both links at once: ~78.4 Gb/s aggregate.
- Application round trip (ofl RDMA, small payload): ~3–6 µs. TCP: ~55 µs.
- Drafter STEP (184 KB request, 24 B reply) on host loopback: p50 22.9 µs with a copy into the send area, 18.2 µs in place.
  Busy-poll vs event wait at a 42 ms cadence: 36 vs 104 µs.
- Engram in production: 37.9 M rows per table per rank over a long-context suite, mean ~205 µs per lookup batch, 0 transport
  errors.

## Build / use

`make` (needs `libibverbs-dev`), then see `API.md`. Python: `import ofl` with `OFL_LIB` pointing at `libofl.so`. Run
`ofl_probe` to list devices and GIDs.

## Related entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
  [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)
- **Separate:** [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) ·
  [remote 3090 drafter + KV](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) ·
  [two-card NVLink drafter](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) ·
  [async host-flag hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) ·
  [bench + interference tools](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

Apache-2.0. Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
