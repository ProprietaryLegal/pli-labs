# Restoring GPTQ on Tesla V100 in 1Cat-vLLM — Root Cause and Fix (research note, 2026-07-10)

> **Status: runtime-verified 2026-07-10** on a Tesla V100-SXM2-32GB (host gooch, TP1,
> fp16, dedicated sm_70 wheel from `fix/gptq-sm70`, torch 2.10.0+cu128, py3.12). GPTQ
> (`desc_act=false`) and compressed-tensors WNA16 (zero env flags) both serve with
> coherent greedy output and sane logprobs — see §7 for the full matrix and the two scope
> corrections (act-order and bias are rejected by the sm_70 host dispatch).

[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) is the vLLM fork that keeps modern
serving alive on Volta (sm_70): TurboMind-derived W4A16 kernels, a flash-attention-v100
port, and a Marlin port credited as marlin_v100. AWQ works out of the box. GPTQ does not:
issue #87 reports `CUDA error: no kernel image is available` inside `gptq_marlin_repack`
on V100 with the official wheel, and #94 reports
`Quantization scheme is not supported for the current GPU. Min capability: 75. Current capability: 70.`

We root-caused both. The punchline: **the Volta GPTQ kernels are already in the source
tree — complete, including act-order support. No CUDA needed writing.** GPTQ on V100 is
broken by a packaging constraint and one Python capability gate.

## 1. What is actually in the tree

- `csrc/quantization/marlin/sm70_gptq_marlin_repack.cu` — GPTQ→Marlin weight repack for
  sm_70, 4-bit and 8-bit, including `has_perm=true` (act-order / `desc_act` / `g_idx`)
  template variants — though note (runtime finding, §7): the GEMM host dispatch
  `TORCH_CHECK`-rejects act-order, so `desc_act=true` checkpoints still do not serve
- `sm70_marlin_u4b8_gemm.cu`, `sm70_marlin_u8b128_gemm.cu` — GEMMs for the GPTQ scalar
  types (unsigned 4-bit with symmetric bias 8; 8-bit with bias 128)
- `csrc/moe/marlin_moe_wna16/sm70_*.cu` — the MoE equivalents
- Python-side, `MarlinLinearKernel.get_min_capability() == 70` and
  `AutoGPTQConfig.get_min_capability() == 60` already

## 2. Root cause A — the op-name collision makes sm_70 Marlin unshippable

The sm_70 and sm_75+ Marlin translation units register the **same torch ops**
(`gptq_marlin_repack`, `awq_marlin_repack`, `marlin_gemm`, `moe_wna16_marlin_gemm`).
Duplicate `TORCH_LIBRARY_IMPL` registration is not allowed, so the build system makes
them mutually exclusive:

```cmake
if (MARLIN_SM70_ARCHS AND NOT MARLIN_OTHER_ARCHS)   # dedicated 7.0-only build
  ... build sm70 marlin, define ENABLE_SM70_MARLIN ...
elseif(MARLIN_SM70_ARCHS)
  message(STATUS "Skipping SM70 Marlin kernels in mixed Marlin arch build")
```

Consequences:

- A **mixed-arch** build (`TORCH_CUDA_ARCH_LIST='7.0 7.5 8.0 ...'`) silently drops sm_70
  Marlin while compiling everything else for sm_70. The wheel looks Volta-capable, passes
  every Python gate, then dies at weight-load time inside `gptq_marlin_repack` — the
  precise #87 signature (the reporter's cuobjdump showed sm_70 cubins for other kernels
  but only `EF_CUDA_SM80` for the repack kernel).
- The current official `docker/Dockerfile` pins
  `torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX'` — 7.0 is gone entirely.
  Either way, **no official wheel can serve Marlin-format quantization on V100.**

## 3. Root cause B — the gate asymmetry ("AWQ works, GPTQ doesn't")

| path | gate | out-of-the-box on sm_70? |
|---|---|---|
| AWQ (`quant_config.json`) | `VLLM_SM70_AWQ_TURBOMIND` defaults **True** → min-cap 70 | yes |
| GPTQ (`quantize_config.json`) | `AutoGPTQConfig` min-cap 60 → `MarlinLinearKernel` (70) | yes, *but only on a dedicated sm_70 build nobody ships* |
| compressed-tensors WNA16 (GPTQ- or AWQ-produced) | returns **75** unless `VLLM_SM70_COMPRESSED_TENSORS_TURBOMIND=1` or `VLLM_SM70_QUANT_BACKEND=marlin` | **no** — #94's exact error, even on a correct sm_70 build |

## 4. The fix

Branch [`fix/gptq-sm70`](https://github.com/Mermiges/1Cat-vLLM/tree/fix/gptq-sm70), two
commits on upstream `main` (`4e9fdbc8`):

1. **`CompressedTensorsWNA16.get_min_capability()`** additionally returns 70 when
   `torch.ops._C.sm70_marlin_available()` is true — an existing binding for the
   `ENABLE_SM70_MARLIN` compile flag (already used by the fork's fp4 utils). Sm_70 builds
   then load compressed-tensors WNA16 with no env vars; builds without the kernels keep
   the 75 gate. Fixes #94.
2. **Actionable failure instead of a cryptic one**: the sm_75+ repack host functions now
   `TORCH_CHECK(cc >= 7.5, "...rebuild with TORCH_CUDA_ARCH_LIST=7.0...")`, and the CMake
   skip message is a `WARNING` spelling out the V100 consequence. De-mystifies #87.

And the operative user fix — build the dedicated V100 wheel:

```bash
# py3.12, torch==2.10.0+cu128 (cu128 is the last torch CUDA flavor with sm_70)
export TORCH_CUDA_ARCH_LIST=7.0 FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0 MAX_JOBS=$(nproc)
pip wheel --no-build-isolation --no-deps -w dist .
```

## 5. Evidence (compile-level)

On a dedicated sm_70 wheel built from current main (CUDA 12.6, torch 2.10.0+cu128,
24-core EPYC, ~13 min):

- `cuobjdump --list-elf vllm/_C.abi3.so` → all cubins sm_70
- demangled symbols: `marlin::gptq_marlin_repack_kernel<256, {4,8}, {has_perm: false,true},
  {a8bit: false,true}, {64,128,256}>` — the act-order variants are compiled
- `sm70_marlin_u4b8_gemm` / `u8b128` / MoE marlin TUs present in `_C` / `_moe_C`
- import test (CPU): `vllm._custom_ops.sm70_marlin_available() == True`;
  `torch.ops._C.gptq_marlin_repack` resolves

## 6. What we did NOT do

- No runtime execution — no V100 was free. Kernel presence ≠ kernel correctness.
- No universal-wheel fix. The right upstream end-state is to end the either/or build
  (distinct op names or a host-side capability dispatch TU compiled for all archs) so one
  official wheel serves Volta and Turing+ together. Proposed to the maintainer; mechanical
  but needs dual-arch runtime testing.

## 7. Runtime verification results (2026-07-10, gooch, Tesla V100-SXM2-32GB, GPU 0)

Fresh py3.12 venv, `torch==2.10.0+cu128`, wheel built from `fix/gptq-sm70` (`054e2fd2`)
with `TORCH_CUDA_ARCH_LIST=7.0`; all runs TP1, `dtype=float16`, greedy, `logprobs=3`,
`enforce_eager=True` (see caveat 3 below).

| checkpoint | format | result |
|---|---|---|
| `TheBloke/Llama-2-7B-Chat-GPTQ` (4-bit g128, `desc_act=false`) | GPTQ | **PASS** — `Using MarlinLinearKernel for AutoGPTQLinearMethod`, repack runs (no kernel-image error), coherent greedy output ("The capital of France is" → "Paris, which is located in the northern part of the country."), chosen-token logprobs all finite in [-1.4, 0] |
| `nm-testing/Meta-Llama-3-8B-Instruct-W4A16-compressed-tensors-test` (4-bit g128) | compressed-tensors | **PASS with zero `VLLM_SM70_*` env vars** — the commit-1 auto-detect behavior; `Using MarlinLinearKernel for CompressedTensorsWNA16`; coherent, sane logprobs; ~68 output tok/s aggregate batch-4 (8B, eager, shared host) |
| `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` (`desc_act=true`) | GPTQ | clean rejection: `act_order is not supported for the SM70 Marlin path.` |
| `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` | GPTQ | clean rejection: `SM70 Marlin does not support bias. TODO: add epilogue bias fusion.` |

Scope corrections found at runtime (pre-existing sm_70 port limitations, orthogonal to the
patch — clean `TORCH_CHECK` errors, not crashes):

1. **Act-order does not serve.** The `has_perm=true` repack cubins exist, but
   `sm70_marlin_dispatch.cu` hard-rejects `has_act_order` before the GEMM. V100 GPTQ
   support = `desc_act=false` checkpoints only. (§1 corrected accordingly.)
2. **Bias epilogue unsupported** — Qwen-family checkpoints (QKV bias) cannot serve on the
   sm_70 Marlin path; bias-free families (Llama, Mistral) work.
3. **Default `VLLM_COMPILE` mode failed on this host** with a device-side assert in the
   rotary-embedding gather during the profile run; `enforce_eager` works. Independent of
   quantization; to be reproduced and filed separately.

Also worth recording: the first wheel from the patched branch was silently defective — an
interrupted compile left two 0-byte object files (`sm70_marlin_u4b8_gemm.cu.o`,
`sm70_marlin_u8b128_gemm.cu.o`) which the relink accepted, producing an `undefined symbol`
ImportError. Check `find build -name "*.o" -size 0` after any interrupted vLLM build.

## Credits

- **1CatAI / 1Cat-vLLM** for the entire sm_70 kernel port this fix merely unlocks —
  TurboMind (LMDeploy) lineage, flash-attention-v100, marlin_v100.
- vLLM project for the Marlin kernel family.
- PLI Labs V100 serving research (proprietarylegal.ai).

**Upstream PR:** https://github.com/1CatAI/1Cat-vLLM/pull/100 (opened 2026-07-10; comments with validation evidence on issues #87 and #94).
