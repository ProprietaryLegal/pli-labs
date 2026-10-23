"""p2p.cu collectives: protocol (epochs, parity reuse, chunking), combine bits, timeouts.

Single-device mode (one visible GPU): both "ranks" live on the same device and run on two streams (small grids, so
the two kernels of a collective are co-resident); this exercises the protocol and the arithmetic, not NVLink.
Two-device mode (two visible GPUs with peer access): the real pair through P2PPair.
Each round enqueues a long random sequence of collectives on both ranks without host syncs (so parity slots are
reused while the other rank may be behind) and checks every result against torch on the host side:
  ar_sum      out_r == fp32(x0 + x1) (bit-identical on both ranks), bf16 output == torch's .to(bfloat16)
  xchg_argmax both ranks write i0 if v0 >= v1 else i1 + V0 (ties -> rank 0)
  xchg_margin global argmax as above and top1 - top2 of the union
Run: python -m pytest -q worker/tests/test_p2p.py   (CUDA_VISIBLE_DEVICES = one or two UUIDs)
"""

from __future__ import annotations

import random

import pytest
import torch

from worker import p2p

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

MAXW = 32 * 5120


class SameDevicePair:
    """Two RankStates on one device; rank r launches on stream r."""

    def __init__(self, dev, timeout_ms=2000.0):
        self.devs = [dev, dev]
        self.st = [p2p.RankState(dev, MAXW) for _ in range(2)]
        self.max_words = MAXW
        self.vocab = 1 << 62
        self.timeout_ns = int(timeout_ms * 1e6)
        self.streams = [torch.cuda.Stream(dev), torch.cuda.Stream(dev)]
        p2p.preload(dev)
        warm_torch(dev)

    def peer(self, r):
        return self.st[1 - r].ptr

    ar_sum = p2p.P2PPair.ar_sum
    xchg_argmax = p2p.P2PPair.xchg_argmax
    xchg_margin = p2p.P2PPair.xchg_margin
    xchg_markov = p2p.P2PPair.xchg_markov


def warm_torch(dev):
    """Load the torch kernels the sequence launches while a collective may spin (lazy loading, p2p.preload)."""
    with torch.cuda.device(dev):
        torch.full((1, 5), -1, dtype=torch.int64, device=dev)
        torch.zeros(1, 5, dtype=torch.float32, device=dev)
        torch.empty(4, device=dev).copy_(torch.ones(4))
        torch.cuda._sleep(1)
        torch.cuda.synchronize(dev)


def make_pair():
    n = torch.cuda.device_count()
    if n >= 2:
        pr = p2p.P2PPair([torch.device("cuda", 0), torch.device("cuda", 1)], MAXW)
        pr.streams = [torch.cuda.Stream(torch.device("cuda", r)) for r in range(2)]
        for d in pr.devs:
            warm_torch(d)
        return pr
    return SameDevicePair(torch.device("cuda", 0))


def run_sequence(pr, seed: int, n_ops: int = 60):
    rnd = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    ops, outs = [], [[], []]
    for _ in range(n_ops):
        kind = rnd.choice(["ar32", "ar16", "argmax", "margin"])
        if kind in ("ar32", "ar16"):
            n = rnd.choice([5, 10, 15, 20, 1, 32]) * 5120
            nb = rnd.choice([1, 8, 16, 32, 64])
            xs = [torch.randn(n, generator=g) * 10 for _ in range(2)]
            ops.append((kind, n, nb, xs))
        else:
            B = rnd.randint(1, 4)
            vals = [torch.randn(B, generator=g).round(decimals=1) for _ in range(2)]   # ties happen
            idx = [torch.randint(0, 64000, (B,), generator=g) for _ in range(2)]
            top2 = [torch.stack((v, v - torch.rand(B, generator=g)), 1) for v in vals]
            ops.append((kind, B, rnd.randint(0, 4), (vals, idx, top2)))
    # stage inputs on the devices, then launch everything without host syncs (rank 1 delayed by a dummy kernel)
    dev_in = [[], []]
    for r in range(2):
        d = pr.devs[r]
        for kind, a, b, data in ops:
            if kind in ("ar32", "ar16"):
                dev_in[r].append(data[r].to(d))
            else:
                vals, idx, top2 = data
                dev_in[r].append((vals[r].float().to(d), idx[r].to(d), top2[r].float().contiguous().to(d)))
        torch.cuda.synchronize(d)
    for r in (1, 0):
        with torch.cuda.device(pr.devs[r]), torch.cuda.stream(pr.streams[r]):
            if r == 1:
                torch.cuda._sleep(2_000_000)                       # rank 1 starts ~1 ms late
            for (kind, a, b, data), x in zip(ops, dev_in[r]):
                d = pr.devs[r]
                if kind == "ar32":
                    outs[r].append(pr.ar_sum(r, x, torch.empty_like(x), b))
                elif kind == "ar16":
                    outs[r].append(pr.ar_sum(r, x, torch.empty(x.shape, dtype=torch.bfloat16, device=d), b))
                elif kind == "argmax":
                    v, i, _ = x
                    out = torch.full((a, 5), -1, dtype=torch.int64, device=d)
                    pr.xchg_argmax(r, v, i, 0 if r == 0 else 64000, out, b)
                    outs[r].append(out)
                else:
                    _, i, t2 = x
                    out = torch.full((a, 5), -1, dtype=torch.int64, device=d)
                    mg = torch.zeros(a, 5, dtype=torch.float32, device=d)
                    pr.xchg_margin(r, t2, i, 0 if r == 0 else 64000, out, mg, b)
                    outs[r].append((out, mg))
    for r in range(2):
        torch.cuda.synchronize(pr.devs[r])
    for k, (kind, a, b, data) in enumerate(ops):
        o0, o1 = outs[0][k], outs[1][k]
        if kind in ("ar32", "ar16"):
            ref = data[0] + data[1]
            if kind == "ar16":
                ref = ref.to(torch.bfloat16)
            assert torch.equal(o0.cpu(), ref), (k, kind)
            assert torch.equal(o1.cpu(), ref), (k, kind)
        else:
            vals, idx, top2 = data
            v0, v1 = top2[0][:, 0].float(), top2[1][:, 0].float()
            want = torch.where(v0 >= v1, idx[0], idx[1] + 64000)
            for o in (o0, o1):
                ids = (o[0] if kind == "margin" else o).cpu()
                assert torch.equal(ids[:, b], want), (k, kind, ids[:, b], want)
                assert (ids[:, [c for c in range(5) if c != b]] == -1).all()
            if kind == "margin":
                a1, a2, b1, b2 = top2[0][:, 0], top2[0][:, 1], top2[1][:, 0], top2[1][:, 1]
                second = torch.where(a1 >= b1, torch.maximum(a2, b1), torch.maximum(b2, a1))
                want_mg = torch.maximum(a1, b1) - second
                for o in (o0, o1):
                    assert torch.equal(o[1].cpu()[:, b], want_mg.float()), (k, o[1][:, b], want_mg)
    hdr = [[int(v) for v in s.hdr[:5].cpu()] for s in pr.st]
    assert hdr[0][2] == 0 and hdr[1][2] == 0, f"error words set: {hdr}"
    assert hdr[0][0] == hdr[1][0], f"epochs diverged: {hdr}"
    return hdr


def test_sequences():
    pr = make_pair()
    for seed in range(6):
        hdr = run_sequence(pr, seed)
    assert hdr[0][4] == 6 * 60


def test_timeout_sets_error_and_reset():
    if torch.cuda.device_count() >= 2:
        pr = p2p.P2PPair([torch.device("cuda", 0), torch.device("cuda", 1)], MAXW, timeout_ms=50.0)
    else:
        pr = SameDevicePair(torch.device("cuda", 0), timeout_ms=50.0)
    d = pr.devs[0]
    x = torch.randn(5 * 5120, device=d)
    with torch.cuda.device(d):
        pr.ar_sum(0, x, torch.empty_like(x), 8)                      # rank 1 never joins
    torch.cuda.synchronize(d)
    h0 = [int(v) for v in pr.st[0].hdr[:5].cpu()]
    h1 = [int(v) for v in pr.st[1].hdr[:5].cpu()]
    assert h0[2] & 1 and h0[3] == 1, h0                              # own timeout at epoch 1
    assert h1[2] & 2, h1                                             # mirrored into the peer
    # fail fast: in error, rank 0's next collectives skip the peer (no further timeout) and keep their own data
    import time
    with torch.cuda.device(d):
        v = torch.tensor([0.5], device=d)
        i = torch.tensor([123], device=d)
        out = torch.full((1, 5), -1, dtype=torch.int64, device=d)
        y = torch.empty_like(x)
        torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        pr.xchg_argmax(0, v, i, 0, out, 2)
        pr.ar_sum(0, x, y, 8)
        torch.cuda.synchronize(d)
        dt = time.perf_counter() - t0
    assert dt < 0.03, dt                                             # 2 collectives, no 50 ms wait
    assert int(out[0, 2]) == 123 and torch.equal(y, x), (out, (y - x).abs().max())
    for s in pr.st:
        with torch.cuda.device(s.dev):
            p2p.ext().reset(s.buf)
    torch.cuda.synchronize(d)
    assert all(int(v) == 0 for s in pr.st for v in s.hdr[:5].cpu())


def test_xchg_markov_matches_triton_finish():
    """The fused finish + exchange == tp2's Triton finish (_argmax_finish_vi_kernel) + the emulation's combine, with
    ties inside the partials (left tie-break) and between ranks (rank 0), cached and uncached previous tokens."""
    import triton

    from worker.tp2 import _argmax_finish_vi_kernel
    pr = make_pair()
    pr.vocab = 130000                    # covers the candidate ids (< 64000 + 60000) and the token table
    g = torch.Generator().manual_seed(3)
    for trial in range(40):
        B = 1 + trial % 4
        nc, nu = 32, 127
        V = 130000
        slot = torch.full((V,), -1, dtype=torch.int32)
        slot[torch.randperm(V, generator=g)[:V // 2]] = 7
        prev = torch.randint(0, V, (B, 3), generator=g)            # use a strided column, as heads() does
        ins = []
        for r in range(2):
            vals_c = torch.randint(0, 6, (B, nc), generator=g).float()     # many exact ties
            vals_u = torch.randint(0, 6, (B, nu), generator=g).float()
            idx_c = torch.randint(0, 60000, (B, nc), generator=g)
            idx_u = torch.randint(0, 60000, (B, nu), generator=g)
            ins.append((vals_c, idx_c, vals_u, idx_u))
        outs = []
        for r in range(2):
            d = pr.devs[r]
            with torch.cuda.device(d), torch.cuda.stream(pr.streams[r]):
                pv = prev.to(d)[:, 1]
                parts = tuple(t.to(d) for t in ins[r])
                out = torch.full((B, 5), -1, dtype=torch.int64, device=d)
                pr.xchg_markov(r, pv, slot.to(d), parts, 0 if r == 0 else 64000, out, trial % 5)
                outs.append(out)
        for r in range(2):
            torch.cuda.synchronize(pr.devs[r])
        # reference: Triton finish per rank, then the combine
        d = pr.devs[0]
        loc = []
        for r in range(2):
            vals_c, idx_c, vals_u, idx_u = (t.to(d) for t in ins[r])
            val = torch.empty(B, dtype=torch.float32, device=d)
            idx = torch.empty(B, dtype=torch.int64, device=d)
            pv = prev.to(d)[:, 1]
            _argmax_finish_vi_kernel[(B,)](pv, slot.to(d), vals_c, idx_c, vals_u, idx_u, val, idx, nc, nu,
                                           pv.stride(0), BLOCK=triton.next_power_of_2(max(nc, nu)), num_warps=4)
            loc.append((val.cpu(), idx.cpu() + (0 if r == 0 else 64000)))
        want = torch.where(loc[0][0] >= loc[1][0], loc[0][1], loc[1][1])
        for o in outs:
            assert torch.equal(o.cpu()[:, trial % 5], want), (trial, o[:, trial % 5], want)


def test_bad_received_id_is_refused():
    """A peer id outside [0, vocab) never reaches the output: the rank keeps its own candidate, error bit 2 is set."""
    pr = make_pair()
    pr.vocab = 1000
    vals = [torch.tensor([1.0]), torch.tensor([5.0])]               # rank 1 would win
    idxs = [torch.tensor([7]), torch.tensor([10 ** 12])]            # but its id (+ v_off) is out of range
    outs = []
    for r in range(2):
        d = pr.devs[r]
        with torch.cuda.device(d), torch.cuda.stream(pr.streams[r]):
            out = torch.full((1, 5), -1, dtype=torch.int64, device=d)
            pr.xchg_argmax(r, vals[r].to(d), idxs[r].to(d), 0 if r == 0 else 500, out, 0)
            outs.append(out)
    for r in range(2):
        torch.cuda.synchronize(pr.devs[r])
    assert int(outs[0][0, 0]) == 7                                   # rank 0 refused the bad id, kept its own
    assert int(pr.st[0].hdr[2].cpu()) & 4
