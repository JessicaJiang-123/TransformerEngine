# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Self-written verification for the migrated blockwise FP8 grouped GEMM triton kernels.

SUT: grouped_gemm_fp8_blockwise_triton_kernel (fprop trans_b=T / dgrad trans_b=F, variable-M)
     grouped_gemm_fp8_blockwise_variable_k_triton_kernel (wgrad, variable-K + segment_m).
Baseline = the already-verified dense direction-matched fake / full, built PER GROUP and
concatenated. Judged with cdiff(tri,fake) + adaptive signed-mean + SQNR.

KERNEL CONSTRAINT (verified below): the grouped kernels require the GEMM OUTPUT's last
dim (the N-tile) to be 128-aligned -- fprop output N, dgrad/wgrad output K. Non-aligned
output dim => fprop/dgrad silently wrong, wgrad compile-fails. The token dim M may be any
variable length (that's the MoE point). Production MoE dims (moe_intermediate, hidden) are
128-aligned so this never triggers; the limitation tests below guard that understanding.
"""
import pytest
import torch

from transformer_engine.pytorch.triton_kernels.blockwise_fp8_grouped_gemm import (
    grouped_gemm_fp8_blockwise_triton_kernel,
    grouped_gemm_fp8_blockwise_variable_k_triton_kernel,
)
from transformer_engine.pytorch.triton_kernels.blockwise_fp8_quantize import (
    quantize_fp8_blockwise,
    quantize_fp8_blockwise_weight,
    quantize_fp8_blockwise_segment_m,
)

DEV = "cuda"
E4M3 = torch.float8_e4m3fn


def _cdiff(x, y):
    x = x.double().flatten(); y = y.double().flatten()
    d = (x * x).sum() + (y * y).sum()
    return (1 - (2 * (x * y).sum()) / d).item() if d > 0 else 0.0


def _sqnr(r, t):
    r = r.double(); t = t.double()
    return (10 * torch.log10((r ** 2).sum() / (((r - t) ** 2).sum() + 1e-30))).item()


def _deq_row(t):
    if t.numel() == 0:
        return t.float()
    f, s = quantize_fp8_blockwise(t.contiguous(), E4M3, axis=1)
    return f.float() * s.repeat_interleave(128, 1)[:, :t.shape[1]]


def _deq_col(t):
    if t.numel() == 0:
        return t.float()
    f, s = quantize_fp8_blockwise(t.contiguous(), E4M3, axis=0)
    return f.float() * s.repeat_interleave(128, 0)[:t.shape[0], :]


def _deq_2d(t):
    f, s = quantize_fp8_blockwise_weight(t.contiguous(), E4M3)
    return f.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:t.shape[0], :t.shape[1]]


def _gen(shape, mode):
    if mode == "normal":
        return torch.randn(shape, device=DEV)
    t = torch.randn(shape, device=DEV) * 4.0
    m = torch.rand(shape, device=DEV) < 0.01
    t[m] *= 50
    return t


def _offs(gl):
    o = [0]
    for m in gl:
        o.append(o[-1] + m)
    return o


def _fprop(X, W, go, offs, G):
    a, s = quantize_fp8_blockwise(X, E4M3, axis=1)
    b, bs = quantize_fp8_blockwise_weight(W, E4M3)
    tri = grouped_gemm_fp8_blockwise_triton_kernel(a, b, s, bs, go, trans_b=True, out_dtype=torch.bfloat16)
    Mtot, N = X.shape[0], W.shape[1]
    fake = torch.zeros(Mtot, N, device=DEV); full = torch.zeros(Mtot, N, device=DEV)
    for g in range(G):
        s0, e0 = offs[g], offs[g + 1]
        if e0 > s0:
            fake[s0:e0] = _deq_row(X[s0:e0]) @ _deq_2d(W[g]).t()
            full[s0:e0] = X[s0:e0] @ W[g].t()
    return tri, fake, full


def _dgrad(dY, W, go, offs, G):
    b, bs = quantize_fp8_blockwise_weight(W, E4M3)
    g_, gs = quantize_fp8_blockwise(dY, E4M3, axis=1)
    tri = grouped_gemm_fp8_blockwise_triton_kernel(g_, b, gs, bs, go, trans_b=False, out_dtype=torch.bfloat16)
    Mtot, K = dY.shape[0], W.shape[2]
    fake = torch.zeros(Mtot, K, device=DEV); full = torch.zeros(Mtot, K, device=DEV)
    for g in range(G):
        s0, e0 = offs[g], offs[g + 1]
        if e0 > s0:
            fake[s0:e0] = _deq_row(dY[s0:e0]) @ _deq_2d(W[g])
            full[s0:e0] = dY[s0:e0] @ W[g]
    return tri, fake, full


def _wgrad(dY, X, glt, go, offs, G, N, K):
    gc, gcs, _, vko = quantize_fp8_blockwise_segment_m(dY, E4M3, 128, glt, go)
    ac, acs, _, _ = quantize_fp8_blockwise_segment_m(X, E4M3, 128, glt, go)
    tri = grouped_gemm_fp8_blockwise_variable_k_triton_kernel(gc, ac, gcs, acs, vko, out_dtype=torch.bfloat16)
    fake = torch.zeros(G, N, K, device=DEV); full = torch.zeros(G, N, K, device=DEV)
    for g in range(G):
        s0, e0 = offs[g], offs[g + 1]
        if e0 > s0:
            fake[g] = _deq_col(dY[s0:e0]).t() @ _deq_col(X[s0:e0])
            full[g] = dY[s0:e0].t() @ X[s0:e0]
    return tri, fake, full


def _check(tri, fake, full):
    e = tri.double() - full.double()
    sc = full.double().abs().mean().item() + 1e-30
    cd = _cdiff(tri, fake)
    signed = e.mean().item() / sc
    unsigned = e.abs().mean().item() / sc
    tol = max(1e-3, 6.0 * unsigned / (full.numel() ** 0.5))
    assert cd < 1e-3, f"cdiff={cd:.2e}"
    assert abs(signed) < tol, f"signed={signed:.2e} tol={tol:.2e}"
    assert _sqnr(full, tri) > 20.0, f"SQNR={_sqnr(full, tri):.1f}"


CONFIGS = {
    "even": ([128, 128, 128], None),
    "varlen": ([100, 200, 150], None),
    "tiny": ([100, 7, 200], None),
    "single": ([1, 256, 1], None),
    "empty": ([128, 0, 64], None),
    "B1": ([300], None),
    "B8": ([40, 88, 17, 150, 3, 77, 200, 9], None),
    "B24": ([31, 17, 64, 5, 128, 42, 9, 200, 3, 77, 15, 88,
             120, 1, 55, 33, 7, 99, 44, 150, 22, 66, 11, 180], None),
    "adv": ([130, 130], [1.0, 1e-3]),
}
# N,K both 128-aligned (production dims); M is arbitrary variable length.
SHAPEMODE = [(256, 512, "normal"), (384, 640, "normal"), (256, 512, "wide")]


@pytest.mark.parametrize("N,K,mode", SHAPEMODE, ids=[f"{n}x{k}-{m}" for (n, k, m) in SHAPEMODE])
@pytest.mark.parametrize("cfg", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize("layout", ["fprop", "dgrad", "wgrad"])
def test_grouped_gemm(layout, cfg, N, K, mode):
    torch.manual_seed(0)
    gl, scales = CONFIGS[cfg]
    G = len(gl); offs = _offs(gl); Mtot = offs[-1]
    go = torch.tensor(offs, dtype=torch.int64, device=DEV)
    glt = torch.tensor(gl, dtype=torch.int64, device=DEV)
    X = _gen((max(Mtot, 1), K), mode)[:Mtot].clone()
    dY = _gen((max(Mtot, 1), N), mode)[:Mtot].clone()
    W = _gen((G, N, K), mode)
    if scales:
        for g in range(G):
            X[offs[g]:offs[g + 1]] *= scales[g]
            dY[offs[g]:offs[g + 1]] *= scales[g]
    if layout == "fprop":
        _check(*_fprop(X, W, go, offs, G))
    elif layout == "dgrad":
        _check(*_dgrad(dY, W, go, offs, G))
    else:
        _check(*_wgrad(dY, X, glt, go, offs, G, N, K))


# --- limitation guards: output last-dim must be 128-aligned (production dims are) ---
@pytest.mark.parametrize("layout,N,K", [("fprop", 257, 512), ("dgrad", 256, 520)],
                         ids=["fprop_N_nonalign", "dgrad_K_nonalign"])
def test_alignment_limitation_numeric(layout, N, K):
    """Non-aligned OUTPUT dim => silently wrong. If this ever drops <1e-3 the kernel was
    fixed; update the constraint note. (M variable-length is fine; only output N-tile matters.)"""
    torch.manual_seed(0)
    gl = [100, 200]; G = 2; offs = _offs(gl)
    go = torch.tensor(offs, dtype=torch.int64, device=DEV)
    X = torch.randn(300, K, device=DEV); dY = torch.randn(300, N, device=DEV); W = torch.randn(G, N, K, device=DEV)
    tri, fake, _ = _fprop(X, W, go, offs, G) if layout == "fprop" else _dgrad(dY, W, go, offs, G)
    assert _cdiff(tri, fake) > 1e-3, "kernel now handles non-aligned output dim; update docs"


def test_wgrad_K_nonalign_compile_fails():
    """variable-k wgrad kernel fails to compile when K%128!=0. Production hidden dim is aligned."""
    torch.manual_seed(0)
    gl = [100, 200]; G = 2; offs = _offs(gl); N, K = 256, 520
    go = torch.tensor(offs, dtype=torch.int64, device=DEV)
    glt = torch.tensor(gl, dtype=torch.int64, device=DEV)
    X = torch.randn(300, K, device=DEV); dY = torch.randn(300, N, device=DEV)
    with pytest.raises(Exception):
        _wgrad(dY, X, glt, go, offs, G, N, K)
