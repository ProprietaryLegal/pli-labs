# Restoring GPTQ on Tesla V100 in 1Cat-vLLM — Root Cause and Fix (research note, 2026-07-10)

> **Status: compile-verified, runtime-pending.** All V100 nodes were occupied by a
> production run when this was written; kernel presence is proven by `cuobjdump`, not yet
> by execution. Do not promote to the blog until the §7 runtime checklist passes on real
> Volta hardware. Draft for the eventual post: "Restoring GPTQ on Tesla V100 in 1Cat-vLLM".

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
  template variants — the classic GPTQ-vs-AWQ format delta is handled
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

## 7. Runtime verification checklist (before promoting this note to the blog)

1. 1× V100-SXM2-32GB, `fix/gptq-sm70` wheel, torch 2.10.0+cu128, py3.12.
2. Small GPTQ-Int4 model with `desc_act=true` (exercises the perm path), TP1,
   `--dtype float16`: engine starts, repack runs, generation coherent.
3. compressed-tensors WNA16 checkpoint with **no** `VLLM_SM70_*` env vars (the commit-1
   behavior), same checks.
4. Logprob sanity vs the AWQ variant of the same model; decode t/s recorded.

## Credits

- **1CatAI / 1Cat-vLLM** for the entire sm_70 kernel port this fix merely unlocks —
  TurboMind (LMDeploy) lineage, flash-attention-v100, marlin_v100.
- vLLM project for the Marlin kernel family.
- PLI Labs V100 serving research (proprietarylegal.ai).
