"""CPU-only verification of the deltanet AWQ patch (no CUDA init).

1. Loads real tensors (layer 1 GDN) from the QuantTrio/Qwen3.6-27B-AWQ
   checkpoint on CPU.
2. Runs the patched pipeline: split_qkv -> requant_mixed -> fuse_gdn.
3. Verifies format, shapes, packability, TP=1/TP=2 fusion, and numeric
   round-trip (AWQ region must be bit-exact; b/a region RTN error stats).
4. Walks all safetensors headers and computes committed engine weight
   bytes BEFORE vs AFTER the patch.
"""
import json
import os
import struct
import sys
from collections import defaultdict
from glob import glob

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch  # noqa: E402

SNAP = glob('/mnt/nvme2/huggingface/hub/models--QuantTrio--Qwen3.6-27B-AWQ/snapshots/*')[0]

from lmdeploy.turbomind.linear import Linear  # noqa: E402
from lmdeploy.turbomind.weight_format import (AWQFormat, TrivialFormat,  # noqa: E402
                                              pack_u4_row)
from lmdeploy.turbomind.builders.deltanet import (split_qkv, requant_mixed,  # noqa: E402
                                                  fuse_gdn, _rtn_quant_to_awq)

awq_fmt = AWQFormat(block_in=128)
triv_fmt = TrivialFormat()

# ---------------------------------------------------------------------------
# Load layer-1 GDN tensors from safetensors (CPU)
# ---------------------------------------------------------------------------
from safetensors import safe_open  # noqa: E402

index = json.load(open(f'{SNAP}/model.safetensors.index.json'))['weight_map']

def load(key):
    with safe_open(f'{SNAP}/{index[key]}', framework='pt', device='cpu') as f:
        return f.get_tensor(key)

P = 'model.language_model.layers.1.linear_attn.'

def awq_linear(prefix):
    tensors = {
        'weight': awq_fmt.normalize(load(prefix + '.qweight'), 'weight'),
        'scales': awq_fmt.normalize(load(prefix + '.scales'), 'scales'),
        'zeros':  awq_fmt.normalize(load(prefix + '.qzeros'), 'zeros'),
    }
    return Linear(tensors=tensors, weight_format=awq_fmt)

def triv_linear(prefix):
    return Linear(tensors={'weight': triv_fmt.normalize(load(prefix + '.weight'), 'weight')},
                  weight_format=triv_fmt)

qkv = awq_linear(P + 'in_proj_qkv')
z   = awq_linear(P + 'in_proj_z')
b   = triv_linear(P + 'in_proj_b')
a   = triv_linear(P + 'in_proj_a')

print('in_proj_qkv weight', tuple(qkv.tensors['weight'].shape), qkv.tensors['weight'].dtype)
print('in_proj_z   weight', tuple(z.tensors['weight'].shape))
print('in_proj_b   weight', tuple(b.tensors['weight'].shape), b.tensors['weight'].dtype)
print('in_proj_a   weight', tuple(a.tensors['weight'].shape))

NUM_K_HEADS, NUM_V_HEADS = 16, 48

# ---------------------------------------------------------------------------
# Pipeline: split -> requant -> fuse (tp=1 and tp=2)
# ---------------------------------------------------------------------------
q_l, k_l, v_l = split_qkv(qkv, num_k_heads=NUM_K_HEADS, num_v_heads=NUM_V_HEADS)
outs = requant_mixed(q_l, k_l, v_l, z, b, a, data_type=None)
fmt_names = [type(o.weight_format).__name__ for o in outs]
assert all(n == 'AWQFormat' for n in fmt_names), fmt_names
print('\nrequant_mixed formats:', fmt_names)

for tp in (1, 2):
    fused = fuse_gdn(*outs, tp=tp)
    w, s, zz = fused.tensors['weight'], fused.tensors['scales'], fused.tensors['zeros']
    assert isinstance(fused.weight_format, AWQFormat)
    assert w.shape == (5120, 16480) and w.dtype == torch.uint8, w.shape
    assert s.shape == (40, 16480) and s.dtype == torch.float16
    assert zz.shape == (40, 16480) and zz.dtype == torch.float16
    # commit-time pack must succeed (out dim % 8, per-shard % 8)
    packed = fused.weight_format.pack(w, 'weight')
    assert packed.tensor.dtype == torch.int32 and packed.tensor.shape == (5120, 2060)
    per_shard = 16480 // tp
    assert per_shard % 8 == 0
    print(f'tp={tp}: fused in_proj_all OK  weight u4 {tuple(w.shape)} '
          f'packed int32 {tuple(packed.tensor.shape)}  per-shard out={per_shard}')

# ---------------------------------------------------------------------------
# Numeric round-trip
# ---------------------------------------------------------------------------
fused = fuse_gdn(*outs, tp=1)
deq = awq_fmt.dequant(fused.tensors, None)['weight'].float()

ref_qkvz = awq_fmt.dequant(qkv.tensors, None)['weight'].float()
ref_z = awq_fmt.dequant(z.tensors, None)['weight'].float()

# tp=1 layout: [Q | K | V | Z | B | A] with Q,K,V exactly the block split of qkv
qs, ks, vs = 16 * 128, 16 * 128, 48 * 128
err_qkv = (deq[:, :qs + ks + vs] - ref_qkvz).abs().max().item()
err_z = (deq[:, qs + ks + vs:qs + ks + vs + 6144] - ref_z).abs().max().item()
print(f'\nAWQ region round-trip: max|err| qkv={err_qkv:.3e}  z={err_z:.3e}  (must be 0.0)')
assert err_qkv == 0.0 and err_z == 0.0

for name, lin, off in (('b', b, 16384), ('a', a, 16432)):
    ref = lin.tensors['weight'].float()
    got = deq[:, off:off + 48]
    diff = (got - ref).abs()
    denom = ref.abs().clamp(min=1e-8)
    print(f'in_proj_{name} RTN int4 g128: max|err|={diff.max():.3e}  '
          f'mean|err|={diff.mean():.3e}  rms(w)={ref.pow(2).mean().sqrt():.3e}  '
          f'mean rel={(diff / denom).median():.4f}')

# also verify layer-0 (all-trivial) path is untouched
t0 = Linear(tensors={'weight': torch.randn(5120, 10240, dtype=torch.float16)}, weight_format=triv_fmt)
outs0 = requant_mixed(t0, None, data_type=None)
assert outs0[0] is t0 and outs0[1] is None
print('all-trivial group passes through unchanged (layer 0 OK)')

# ---------------------------------------------------------------------------
# Whole-model committed-weight accounting from safetensors headers
# ---------------------------------------------------------------------------
DT = {'F16': 2, 'BF16': 2, 'F32': 4, 'I32': 4, 'I64': 8, 'U8': 1}

shapes = {}
for f in glob(f'{SNAP}/model-*.safetensors'):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k != '__metadata__':
            shapes[k] = (v['dtype'], v['shape'])

def numel(sh):
    n = 1
    for d in sh:
        n *= d
    return n

def logical_kn(prefix):
    """(K, N) of a linear from checkpoint shapes (AWQ or fp16)."""
    if prefix + '.qweight' in shapes:
        _, s = shapes[prefix + '.qweight']       # [K, N/8] int32
        return s[0], s[1] * 8
    _, s = shapes[prefix + '.weight']            # [N, K] fp16 (HF layout)
    return s[1], s[0]

def awq_committed(K, N, g=128):
    return K * N // 2 + 2 * (K // g) * N * 2     # u4 weight + fp16 scales + fp16 zeros

def is_awq(prefix):
    return prefix + '.qweight' in shapes

groups = defaultdict(int)

def add(group, nbytes):
    groups[group] += nbytes

lin_layers, full_layers = [], []
for i in range(64):
    if f'model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight' in shapes or \
       f'model.language_model.layers.{i}.linear_attn.in_proj_qkv.qweight' in shapes:
        lin_layers.append(i)
    else:
        full_layers.append(i)

before = defaultdict(int)
after = defaultdict(int)

for i in range(64):
    lp = f'model.language_model.layers.{i}.'
    # ---- MLP (identical before/after) ----
    for m in ('gate_proj', 'up_proj', 'down_proj'):
        p = lp + 'mlp.' + m
        K, N = logical_kn(p)
        nb = awq_committed(K, N) if is_awq(p) else K * N * 2
        before['mlp'] += nb; after['mlp'] += nb
    # ---- norms ----
    for nrm in ('input_layernorm', 'post_attention_layernorm'):
        _, s = shapes[lp + nrm + '.weight']
        before['norms'] += numel(s) * 2; after['norms'] += numel(s) * 2
    if i in lin_layers:
        ap = lp + 'linear_attn.'
        # in_proj group
        comps = [(ap + 'in_proj_qkv'), (ap + 'in_proj_z'), (ap + 'in_proj_b'), (ap + 'in_proj_a')]
        kns = [logical_kn(p) for p in comps]
        K = kns[0][0]
        Ntot = sum(n for _, n in kns)
        fp16_bytes = K * Ntot * 2
        if any(is_awq(p) for p in comps):
            before['gdn_in_proj'] += fp16_bytes            # dequant_mixed blowup
            after['gdn_in_proj'] += awq_committed(K, Ntot)  # stays W4
        else:
            before['gdn_in_proj'] += fp16_bytes            # layer 0: native fp16
            after['gdn_in_proj'] += fp16_bytes
        # out_proj (unchanged either way)
        K2, N2 = logical_kn(ap + 'out_proj')
        nb = awq_committed(K2, N2) if is_awq(ap + 'out_proj') else K2 * N2 * 2
        before['gdn_out_proj'] += nb; after['gdn_out_proj'] += nb
        # scalars + conv1d + norm
        small = 0
        for t in ('A_log', 'dt_bias'):
            dt, s = shapes[ap + t]
            small += numel(s) * DT[dt]
        dt, s = shapes[ap + 'conv1d.weight']
        small += numel(s) * DT[dt]
        _, s = shapes[ap + 'norm.weight']
        small += numel(s) * 2
        before['gdn_small'] += small; after['gdn_small'] += small
    else:
        sp = lp + 'self_attn.'
        for m in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            K, N = logical_kn(sp + m)
            nb = awq_committed(K, N) if is_awq(sp + m) else K * N * 2
            before['full_attn'] += nb; after['full_attn'] += nb
        for nrm in ('q_norm', 'k_norm'):
            _, s = shapes[sp + nrm + '.weight']
            before['norms'] += numel(s) * 2; after['norms'] += numel(s) * 2

for k in ('model.language_model.embed_tokens.weight', 'lm_head.weight',
          'model.language_model.norm.weight'):
    _, s = shapes[k]
    before['embed_lmhead'] += numel(s) * 2; after['embed_lmhead'] += numel(s) * 2

vis = sum(numel(s) * DT[dt] for k, (dt, s) in shapes.items() if k.startswith('model.visual'))
mtp = sum(numel(s) * DT[dt] for k, (dt, s) in shapes.items() if k.startswith('mtp.'))
before['vision(fp16, loaded unless disabled)'] = vis
after['vision(fp16, loaded unless disabled)'] = vis

G = 1024 ** 3
print(f'\n{len(lin_layers)} linear-attn layers, {len(full_layers)} full-attn layers')
print(f'{"component":42s} {"BEFORE":>10s} {"AFTER":>10s}')
for k in sorted(set(before) | set(after)):
    print(f'{k:42s} {before[k]/G:9.3f}G {after[k]/G:9.3f}G')
tb, ta = sum(before.values()), sum(after.values())
print(f'{"TOTAL committed engine weights":42s} {tb/G:9.3f}G {ta/G:9.3f}G')
print(f'{"TOTAL w/o vision tower":42s} {(tb-vis)/G:9.3f}G {(ta-vis)/G:9.3f}G')
print(f'(mtp.* not loaded by TurboMind: {mtp/G:.3f}G ignored)')
print(f'\nsaving from patch: {(tb-ta)/G:.3f} GiB')
