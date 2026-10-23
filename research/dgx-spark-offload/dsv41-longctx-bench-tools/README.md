# Long-context serving benchmark with a direct per-step timer, plus RDMA link-interference and failure-drill tools

**Source code.** These are the measurement tools behind every result in PLI Labs' DeepSeek-V4.1-Flash offload work. Overview:
[DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
Methodology: [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings).

**Status: proven.** Used for every A/B/C/D/E comparison.

Hugging Face entry: https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools

## What's in it

- **`bench/bench2.py`** (standard library only). It sends matched prompts to an OpenAI-compatible vLLM server.
  - **Prompts:** short legal tasks; long-context prompts of 32K–300K tokens built from public court opinions and RECAP filings
    (bring your own corpus: `opinions_cl.jsonl`, `filings_recap.jsonl`); synthetic records.
  - **Sampling:** fixed (T, top_p, reasoning effort, priority), per-request cache salts, cold or warm prefix cache.
  - **Concurrency:** 1–4 concurrent requests, plus a "mixed" case: a long prefill arriving while two streams decode.
  - **What it records:**
    - TTFT, first visible answer, reasoning and answer token split, per-stream and aggregate tok/s, task time.
    - Server-side speculation counters (accepted per position, rounds).
    - Node telemetry: MemAvailable, GPU temperature and clock, sampled over ssh.
  - **The direct per-step timer:** the arrival time of every content-bearing SSE event. vLLM streams one event per engine
    step, so the gaps are a client-side round timer: p10/p50/p90/max plus the raw times. This replaces the indirect
    `1000 × (1 + accepted/round) / tok/s` diagnostic, which assumes compatible counters and breaks with EOS, mixed scheduling
    and adaptive depth.
- **`bench/suite.sh`:** the screening suite.
  - Stages: short, c32 (1/2/3 + mixed), warm32, c150 (+ mixed), warm150.
  - Finalists: c256, c300, and **cap4** (4 × 300K concurrent, a capacity test).
- **`bench/compact.sh`:** a matched set for worker A/B on a running server.
- **`bench/report.py` / `summ.py`:** mean ± sd tables; failures never enter the averages.
- **`bench/drill_worker.sh`:** failure drill. It stops and restarts a remote worker in the middle of one long request and
  diffs the engine's remote-round counters.
- **`bench/link_load.sh` / `s1_load.py`:** paced RDMA bulk load on one or both direct links during a benchmark. It refuses to
  start unless the server is idle; `--during-benchmark` allows requests afterwards. It records the achieved Gb/s.
- **`bench/s1_*.py`:** link inventory and the latency/bandwidth matrix (64 B – 64 MiB, send/write/read, one and both links).
- **`roundtrip/`:**
  - `ofl_loopback.cpp`: copy vs in-place and busy-poll vs event-wait microbenchmarks.
  - `observe_x86host.sh`: passive CPU, IRQ and GPU contention sampling for a remote worker; no ptrace, no renice.

## Results produced with these tools

- **Interference:** ~2 % TTFT at 35 Gb/s per link (TTFT 14.91 vs 14.55 s at 32K; 70.31 vs 68.88 s at 150K), and +1–2 % per
  decode step. At 10 Gb/s it's within noise.
- **Drill:** the request completes; target-only steps at 27.7 ms vs drafted steps at 39.6 ms; automatic reconnect.
- **Two-card vs one-card worker:** −2.5 % per round, measured with the direct per-step timer.

## Related entries

- **Combined:** [full stack](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) ·
  [findings](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings)
- **Separate:** [remote Engram](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma) ·
  [remote 3090 drafter + KV](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) ·
  [two-card NVLink drafter](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) ·
  [native 1017 drafter on SM86](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86) ·
  [async host-flag hand-off](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) ·
  [ofl transport](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport)
- **GitHub:** [ProprietaryLegal/pli-labs › research/dgx-spark-offload](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload)

Apache-2.0. Published by PLI Labs (ProprietaryLegal) · https://proprietarylegal.ai
