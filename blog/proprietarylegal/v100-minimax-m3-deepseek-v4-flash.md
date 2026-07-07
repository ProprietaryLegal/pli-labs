# Running MiniMax-M3 and DeepSeek-V4-Flash on 2017 V100s — Measured Numbers, One Silent-Corruption Bug, and a Downloadable Stack

PLI Labs has two of the newest frontier-class open MoE models — MiniMax-M3
(~428B total / ~23B active) and DeepSeek-V4-Flash (284B total / ~13B active) —
serving on a single node of eight NVIDIA V100-SXM2-32GB GPUs: nine-year-old,
compute-capability-7.0 hardware that most current inference stacks no longer
target. We are not aware of a prior published DeepSeek-V4-Flash datapoint on
Volta. This post documents the working configuration, the measured throughput,
and a numeric-corruption bug we root-caused along the way that anyone serving
DeepSeek-family MLA models on older CUDA hardware should know about.

Why bother with V100s? Our program is private legal AI: models a law practice
can run entirely on-premises, on hardware that costs less used than one year of
per-seat SaaS. Eight 32GB V100s provide 256GB of pooled VRAM on one node. If
current MoE models can be made to run correctly there, frontier-adjacent
capability becomes available to small firms with strict confidentiality
requirements and modest budgets.

## The downloadable stack

Everything is public in our llama.cpp fork:
https://github.com/Mermiges/llama.cpp

- **`v100-unified`** — the serving branch: llama.cpp with MiniMax-M3 support
  (from upstream PR #24523) and DeepSeek-V4-Flash support (upstream PR #24162)
  merged into one sm_70-buildable tree, plus an opt-in decode-time MSA
  block-sparse attention path for M3 (`LLAMA_MINIMAX_M3_MSA=1`, disabled by
  default; requires a reconverted GGUF carrying the indexer tensors).
- **`ds4-volta-fix`** — the branch above plus a fail-loud guard for the
  Volta K-cache bug described below.

Build (CUDA 12.6 toolchain):

```sh
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-bench -j
```

Models are the unsloth dynamic GGUF quants: MiniMax-M3 `UD-Q2_K_XL` (133 GiB)
and DeepSeek-V4-Flash `UD-Q4_K_XL` (144 GiB), sha256-verified against the
Hugging Face LFS manifests before every run.

Launch shape that works (both models):

```sh
llama-server -m <model>.gguf -ngl 999 -sm layer -fa 1 \
  -b 1024 -ub 512 --cache-type-k f16 --cache-type-v f16 -c 8192..32768
```

Two hard rules on this hardware: **layer split only** (`-sm layer` — tensor
split with long prefills destabilized the driver in our testing), and **f16
KV-cache** (for DeepSeek-V4-Flash this is a correctness requirement, not a
preference — see below).

## Measured throughput

Eight V100-SXM2-32GB, two NVLink quad boards, llama-bench `-r 3`, power-capped
at 245 W per GPU (−17% below stock), build `87e767e5c`:

| model | quant | pp512 t/s | pp8192 t/s | tg128 t/s |
|---|---|---:|---:|---:|
| MiniMax-M3 | UD-Q2_K_XL, f16 KV | 280.1 ± 10.8 | 269.7 ± 7.5 | 31.2 ± 0.4 |
| MiniMax-M3 | UD-Q2_K_XL, q8_0 KV | 282.3 ± 17.0 | 239.7 ± 7.7 | 29.5 ± 0.1 |
| DeepSeek-V4-Flash | UD-Q4_K_XL, f16 KV | 228.3 ± 0.3 | 209.2 ± 1.1 | 12.2 ± 0.1 |

Working notes:

- **31 tokens/s of generation from a 428B-class model on 2017 GPUs** is an
  interactive-use speed. MiniMax-M3 also held coherent output through 16K, 32K,
  and 64K-token contexts at under 21 GB per card, and passed our legal quality
  screen (five substantive legal-drafting probes plus a needle-in-haystack
  retrieval at ~7,400 prompt tokens) 6/6.
- **f16 KV-cache beat q8_0 on both axes for M3** (+12% prefill at 8K, +6%
  generation). Quantized KV saves VRAM but pays a dequantization tax on every
  attention read; with 256GB of pooled VRAM the saving buys nothing at these
  context lengths.
- DeepSeek-V4-Flash's 12.2 t/s is usable for batch summarization and drafting
  lanes; its prefill (209 t/s at 8K) holds up well for document-ingest work.
- Scaling past 8 GPUs in one llama.cpp process currently aborts with
  `peer mapping resources exhausted` — CUDA limits peer mappings to 8 devices,
  and llama.cpp's VMM pool grants every device access to every pool. A
  `GGML_CUDA_NO_VMM=ON` build is under test as the workaround for 10- and
  12-card configurations; results will follow.

## The bug: quantized K-cache silently corrupts DeepSeek MLA output — UPDATE: on every backend, not just Volta

> **Update (later on 2026-07-07):** continued investigation disproved the sm_70 attribution below. The identical corruption reproduces on CPU-only inference; the true cause is llama.cpp's quantized-KV Hadamard rotation diverting DeepSeek-V4 off its sparse attention paths (graph-level, backend-independent). A verified fix and the full corrected analysis are published — see [ggml-org/llama.cpp#25382](https://github.com/ggml-org/llama.cpp/issues/25382) and [our research record](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100). The operational guidance below (f16 K-cache) remains correct on unpatched builds.


The first DeepSeek-V4-Flash runs loaded cleanly, computed at full speed on all
eight GPUs, returned HTTP 200s and a green `/health` — and emitted confident
gibberish (`" newcom wrt wrtêu newcom dialog大有lean…"`). No CUDA error, no
abort, nothing in the logs. Two independently downloaded, sha256-verified
quants garbled identically, so the weights were not the problem.

Isolating it took a controlled experiment rather than a guess:

1. **CPU control:** the same GGUF on the same build, CPU-only, was coherent —
   so the model graph, tokenizer, and weights were fine; the fault was in a
   CUDA kernel path.
2. **Architecture control:** MiniMax-M3, with conventional attention head
   geometry, was coherent on the same GPUs with a quantized KV-cache — so the
   fault was specific to something DeepSeek uniquely exercises.
3. **One-variable matrix:** on a single V100 with all attention on-GPU and the
   MoE experts kept on CPU, we toggled one knob at a time across FlashAttention
   on/off/auto and K/V cache types in {q8_0, f16}². Our first hypothesis
   (broken FlashAttention kernels) died here: a patch disabling FA changed
   nothing, and FA-on with f16 KV was perfectly coherent.

The decisive table: output is garbage **if and only if the K-cache is
quantized**, independent of FlashAttention mode and independent of the V-cache
type. DeepSeek's multi-head latent attention stores an unusually wide K vector
(head dimension 576 = 512 latent + 64 rotary); the q8_0 K-cache dequantization
path for that shape produces numerically wrong attention scores on
compute-capability-7.0 hardware. Normal head sizes are unaffected, which is
why M3 and every mainstream model serve fine with quantized KV on the same
cards.

The operational fix is configuration, and it costs almost nothing: run
`--cache-type-k f16`. DeepSeek's latent K-cache is small by design, so the
VRAM difference is negligible. We verified coherence at the full 8-GPU scale
after the change, including needle retrieval from multi-thousand-token
prompts.

Because this failure mode is the worst kind — a server that looks healthy
while producing garbage — the `ds4-volta-fix` branch adds a fail-loud guard:
on sm_70 devices, a DeepSeek-MLA model with a quantized K-cache now refuses to
start with an explicit error message instead of serving corrupted output. Two
candidate kernel-level fixes (fp32-staged dequantization and an fp32 KQ path)
did not resolve the underlying numeric defect in our testing, so the guard is
the honest ship until the root kernel cause is pinned; that work is ongoing.

## Method notes for evaluators

Every number above carries provenance: build SHA, sha256-verified model
artifacts, exact flags, and three-repetition means with deviations. Quality
screening treats an empty completion as a failure (early tooling here marked
empty responses as passes when a reasoning model spent its whole token budget
thinking — a silent-pass seam worth checking for in your own harnesses).
Infrastructure interruptions are excluded from quality denominators rather
than scored against the model.

Nothing in this post is legal advice, and neither model is offered as an
autonomous lawyer. These are research measurements toward lawyer-supervised,
on-premises legal AI. Questions and replication reports are welcome.

*PLI Labs — Proprietary Legal Intelligence, LLC*
