# DeepSeek-V4-Flash on Volta (sm_70) — Full Research Notes, 2026-07-07

Complete public record of PLI Labs' DeepSeek-V4-Flash bring-up on a single
8× V100-SXM2-32GB node: what worked, what silently corrupted, the controlled
experiments that pinned it, measured throughput, failed fixes (published as
negative results), and the downloadable stack. Companion blog post:
[`blog/proprietarylegal/v100-minimax-m3-deepseek-v4-flash.md`](../../blog/proprietarylegal/v100-minimax-m3-deepseek-v4-flash.md).

## 1. Stack and provenance

| item | value |
|---|---|
| Hardware | 8× Tesla V100-SXM2-32GB (two NVLink NV2 quad boards), CC 7.0, one node |
| Power | capped 245 W/GPU (−17% below stock) throughout — all numbers include this |
| Toolchain | CUDA 12.6, GCC 13.3, Release, `CMAKE_CUDA_ARCHITECTURES=70` |
| Serving code | https://github.com/Mermiges/llama.cpp — branch `v100-unified` (MiniMax-M3 PR-24523 + DeepSeek-V4 PR-24162 merged, binary `version 9644/9898`), branch `ds4-volta-fix` (adds the fail-loud guard, commit `22c3ea063`) |
| Model | unsloth `DeepSeek-V4-Flash-GGUF` `UD-Q4_K_XL` (144.44 GiB, 5 shards) — **all shards sha256-verified against the HF LFS manifest**; cross-checked with the older `Q4_K_M-XL` set (also byte-verified) |
| Architecture | deepseek4: 284B total / ~13B active MoE; MLA absorbed attention — K head width 576 (512 latent + 64 RoPE), V width 512, single KV head (MQA); V is a view of K in the 576 path |
| Launch shape | `-ngl 999 -sm layer -fa 1 -b 1024 -ub 512 -c 8192` |

## 2. Symptom

The model loads cleanly across 8 GPUs, computes at full speed, serves HTTP 200
and a green `/health` — and emits confident non-language:

```
/completion "The capital of France is" → " newcom wrt wrtêu newcom dialog大有lean…"
```

Identical garbage on two independently downloaded, byte-verified quants ⇒ the
weights are not the cause. No CUDA error, no Xid, no log line. This is the
worst failure class: silent numeric corruption behind a healthy-looking
endpoint.

## 3. Controlled isolation (the decisive experiments)

**Control 1 — CPU:** same GGUF, same build, `-ngl 0` → coherent (" Paris.").
Graph, tokenizer, weights all fine; fault is GPU-side.

**Control 2 — architecture:** MiniMax-M3 (conventional attention geometry) on
the same 8 GPUs, same build, WITH quantized q8_0 KV → coherent through
64K-token contexts. Fault is specific to something DeepSeek uniquely
exercises.

**One-variable matrix** (single V100, all 43 attention layers on GPU, experts
kept on CPU via `-ot 'exps=CPU'`, greedy decoding). First round varied
FlashAttention and KV type one at a time and killed our initial hypothesis
(broken FA kernels): an FA-disabling patch changed nothing (patched ==
unpatched at every configuration).

**K-vs-V pinpoint matrix** (`-fa 1`, single card):

| K cache | V cache | output |
|---|---|---|
| q8_0 | q8_0 | **GARBAGE** |
| q8_0 | f16  | **GARBAGE** |
| f16  | q8_0 | coherent |
| f16  | f16  | coherent |

**Finding:** corruption ⇔ **quantized K-cache**, independent of FA mode
(0/1/auto) and independent of the V-cache type. FlashAttention itself computes
the 576/512 MLA geometry correctly on Volta with an f16 K-cache.

Method lesson worth stealing: our first (wrong) root cause came from an A/B
that moved two knobs at once (`-fa` and KV type together). Never name a root
cause from a confounded A/B.

## 4. Operational fix (proven, zero code)

Serve with `--cache-type-k f16`. The MLA latent K-cache is small by design, so
the VRAM cost is negligible. Verified coherent at full 8-GPU scale, including
a legal-drafting quality screen and verbatim needle retrieval from a
multi-thousand-token prompt.

## 5. Fail-loud guard (published code)

Because the failure is silent, branch `ds4-volta-fix` (commit `22c3ea063`)
adds a guard: on CC 7.0 CUDA devices, a DeepSeek-MLA model (K width 576) with
a quantized K-cache now **refuses context creation** with

```
quantized K-cache is numerically broken on sm_70 for DeepSeek MLA; use --cache-type-k f16
```

instead of serving garbage. The compute capability is queried through the
backend registry's proc-address mechanism (no layering violation), non-Volta
and non-MLA paths are untouched, and a null-context crash in the server on
rejected configs was fixed in the same commit. Verified: f16-K cells coherent,
q8-K cells rejected, across the full 4-cell matrix.

## 6. Failed kernel fixes (negative results, published on purpose)

Two read-side kernel repairs were implemented, tested, and reverted — both
still garbled:

1. Staging the q8_0 K conversion through fp32→fp16 for the 576-wide path.
2. A Volta-only fp32 KQ multiply path in the FA tile kernel.

Since repairing the dequant **read** path does not help, the corruption likely
enters earlier. Open suspects, in ranked order: the q8_0 quantize-on-**write**
path for K rows written through a strided view; the V-is-a-view-of-K aliasing
arithmetic in the f16 staging buffers (consistent with all observed evidence);
and the vector-kernel q8_0 KQ dot for width 576. A dedicated max-effort lane
is on this; results will be published either as a fix or as an
upstream-issue-grade root-cause writeup.

(Also noted, unrelated pre-existing failure: `test-backend-ops -o
FLASH_ATTN_EXT` aborts on Volta in the mma-f16 hsk=320/hsv=256 case with a
CUDA invalid-argument — present with and without any of our changes.)

## 7. Measured throughput (8× V100, `-r 3` means ± σ)

| model | config | pp512 t/s | pp8192 t/s | tg128 t/s |
|---|---|---:|---:|---:|
| DeepSeek-V4-Flash UD-Q4_K_XL | f16 KV, FA on, layer split | 228.28 ± 0.31 | 209.24 ± 1.14 | 12.22 ± 0.12 |
| MiniMax-M3 UD-Q2_K_XL (comparison) | f16 KV | 280.07 ± 10.83 | 269.66 ± 7.54 | 31.23 ± 0.41 |
| MiniMax-M3 UD-Q2_K_XL | q8_0 KV | 282.25 ± 16.99 | 239.69 ± 7.68 | 29.53 ± 0.06 |

Guidance that fell out of the matrix: on 32GB-per-card V100 nodes, **f16
KV-cache is both the safe and the fast choice** — quantized KV lost on both
prefill (−11% at 8K) and generation (−5.4%) for M3, and is a correctness
hazard for DeepSeek MLA. Quantized KV remains a VRAM lever only for very long
contexts.

Scaling note: one llama.cpp process cannot currently use more than 8 of these
GPUs — CUDA's 8-peer-mapping limit aborts allocation (`peer mapping resources
exhausted`) because the VMM pool grants every device access to every pool.
10/12-GPU runs on a `GGML_CUDA_NO_VMM=ON` build are in progress and will be
added here.

## 8. Reproduction

```sh
git clone -b ds4-volta-fix https://github.com/Mermiges/llama.cpp
cmake -S llama.cpp -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-bench -j
# correct serving config (8 GPUs):
build/bin/llama-server -m DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf \
  -ngl 999 -sm layer -fa 1 -b 1024 -ub 512 -c 8192 \
  --cache-type-k f16 --cache-type-v f16
# bug repro (single GPU, experts on CPU) — the guard will refuse it; that refusal is the fix working:
build/bin/llama-server -m <same> -ngl 999 -ot 'exps=CPU' -c 4096 -fa 1 --cache-type-k q8_0
```

Quality screening note for evaluators: treat an **empty completion as a
failure**. Reasoning models can spend an entire small token budget thinking;
early tooling here marked those empty responses as passes until we made empty
output fail loudly.

*PLI Labs — Proprietary Legal Intelligence, LLC. Nothing here is legal advice;
these are research measurements toward lawyer-supervised, on-premises legal AI.*

## 9. Root cause, final (2026-07-07 second investigation) — the sm_70 attribution was WRONG

Section 6's "open suspects" resolved. A second lane cleared every GPU kernel
(new `test-backend-ops` cases: FLASH_ATTN_EXT 56/56 and quantize-on-write
SET_ROWS 2/2, CUDA-vs-CPU, at the exact 576/512 MLA shape with q8_0) and then
ran the control the first investigation had missed: **CPU-only inference with
a quantized K-cache — identical garbage.** The original "CPU coherent" control
had silently used the default f16 K-cache. The defect is backend-independent.

Mechanism (verified by discriminator): a quantized K/V cache enables the
Hadamard incoherence rotation (`attn_rot_k/v`); allocating it diverts
DeepSeek-V4 layers off their sparse CSA/HCA/lightning-indexer attention (which
asserts the rotation is absent) into `build_raw_attention`, which applies the
rotation with a plain `ggml_mul_mat` (not the block-reshaping
`ggml_mul_mat_aux`) and never un-rotates the MLA V-is-a-view-of-K output
before the `v_mla` up-projection. Disabling all rotations at runtime with
q8_0-K → coherent, on CPU and GPU.

Fix shipped (branch `ds4-volta-fix`, patches in `patches/`): disable the KV
rotation for the deepseek4 architecture; quantized K then flows through the
model's designed sparse attention as plain q8_0. Live 4-cell verification
(`evidence/ds4fix-live-matrix-20260707.md`): all four K/V combinations
coherent; f16 behavior byte-identical (rotation was never active there).
Upstream issue with minimal repro and three fix options:
https://github.com/ggml-org/llama.cpp/issues/25382. A rotation-aware rewrite
of the sparse paths (the proper long-term fix) is in progress.

Method lesson, twice earned in one day: a control is only a control if the
variable you care about is actually held. "CPU is coherent" meant "CPU with
f16-K is coherent" — the quantized-K CPU cell was never run, and its absence
cost two failed kernel-fix attempts aimed at hardware that was never broken.
