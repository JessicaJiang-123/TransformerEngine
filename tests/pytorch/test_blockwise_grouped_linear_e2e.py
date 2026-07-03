# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""End-to-end (spec-level) verification of the blockwise FP8 grouped Linear.

Unlike the per-kernel tests (implementation level: triton vs a fake reference sharing
the same fp8 inputs), this drives the REAL autograd Function _GroupedLinearBlockwiseFP8
(fwd + bwd via torch.autograd) and compares out/dgrad/wgrad against BF16 full-precision
autograd. This validates that the whole MoE fp8 orchestration (which direction each
operand is quantized, operand reuse, segment_m padding, the three GEMM layouts) produces
gradients that TRACK full precision within the quantization floor, with NO systematic
signed bias (the slow-crash signature). Judged by SQNR + adaptive signed-mean.
"""
import pytest
import torch

from transformer_engine.pytorch.module.grouped_linear_blockwise import _GroupedLinearBlockwiseFP8

DEV = "cuda"


def _sqnr(r, t):
    r = r.double(); t = t.double()
    return (10 * torch.log10((r ** 2).sum() / (((r - t) ** 2).sum() + 1e-30))).item()


def _judge(name, g_fp8, g_ref):
    e = g_fp8.double() - g_ref.double()
    sc = g_ref.double().abs().mean().item() + 1e-30
    signed = e.mean().item() / sc
    unsigned = e.abs().mean().item() / sc
    tol = max(2e-3, 6.0 * unsigned / (g_ref.numel() ** 0.5))
    assert _sqnr(g_ref, g_fp8) > 20.0, f"{name} SQNR={_sqnr(g_ref, g_fp8):.1f}"
    assert abs(signed) < tol, f"{name} signed={signed:.2e} tol={tol:.2e}"


def _nta(m_splits):
    return (m_splits, False, None, True, False, None, None, None, None, None, None, None,
            False, False, False, torch.bfloat16, True, None, None, False, False, None, None, False)


def _gen(shape, mode):
    if mode == "normal":
        return torch.randn(shape, device=DEV, dtype=torch.bfloat16)
    t = torch.randn(shape, device=DEV) * 4.0
    m = torch.rand(shape, device=DEV) < 0.01
    t[m] *= 50
    return t.bfloat16()


CONFIGS = {
    "even": ([128, 128, 128], None),
    "varlen": ([100, 200, 150], None),
    "tiny": ([100, 7, 200], None),
    "single": ([1, 256, 1], None),
    "B8": ([40, 88, 17, 150, 3, 77, 200, 9], None),
    "adv": ([130, 130], [1.0, 1e-3]),
}


@pytest.mark.parametrize("mode", ["normal", "wide"])
@pytest.mark.parametrize("cfg", list(CONFIGS), ids=list(CONFIGS))
def test_grouped_linear_e2e(cfg, mode):
    torch.manual_seed(0)
    m_splits, scales = CONFIGS[cfg]
    K, N = 512, 256
    Mtot = sum(m_splits)
    offs = [0]
    for m in m_splits:
        offs.append(offs[-1] + m)

    X = _gen((Mtot, K), mode)
    Ws = [_gen((N, K), mode) for _ in m_splits]
    if scales:
        for g in range(len(m_splits)):
            X[offs[g]:offs[g + 1]] *= scales[g]
            Ws[g] *= scales[g]
    X = X.detach().requires_grad_(True)
    Ws = [w.detach().requires_grad_(True) for w in Ws]
    gout = torch.randn(Mtot, N, device=DEV, dtype=torch.bfloat16)

    # fp8 path: real autograd Function
    out = _GroupedLinearBlockwiseFP8.apply(X, _nta(m_splits), *Ws)
    out.backward(gout)
    dX_fp8 = X.grad.detach().clone()
    dW_fp8 = torch.cat([w.grad.detach().reshape(-1) for w in Ws])

    # bf16 full-precision autograd reference
    Xr = X.detach().clone().requires_grad_(True)
    Wr = [w.detach().clone().requires_grad_(True) for w in Ws]
    out_ref = torch.cat([Xr[offs[g]:offs[g + 1]] @ Wr[g].t() for g in range(len(m_splits))], 0)
    out_ref.backward(gout)
    dW_ref = torch.cat([w.grad.detach().reshape(-1) for w in Wr])

    _judge("out", out, out_ref)
    _judge("dgrad", dX_fp8, Xr.grad)
    _judge("wgrad", dW_fp8, dW_ref)
