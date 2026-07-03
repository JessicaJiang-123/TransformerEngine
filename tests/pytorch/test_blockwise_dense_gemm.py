# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Self-written verification for the migrated blockwise FP8 dense GEMM triton kernel.

The stock ``test_float8_blockwise_gemm_exact.py`` only exercises ``tex.generic_gemm``
(C++ hipblaslt), never the triton SUT. This test drives the triton kernel via
``gemm_blockwise`` and applies the Real/Fake/Baseline decomposition:

  * fake = dequant(quant A) @ dequant(quant B)  -- SAME fp8 inputs, high-precision
           accumulate; the quantization DIRECTION must match the GEMM's contraction
           axis (fprop/dgrad rowwise, wgrad colwise), else it's a false failure.
  * full = A @ B (bf16 truth).

Judged with cdiff(tri,fake) (kernel impl), SQNR(tri,full) (accuracy floor), and a
sample-size-adaptive signed-mean (systematic bias -> slow crash). Covers all three
layouts (NT/NN/TN) x non-aligned shapes x wide dynamic range (split-accumulator).
"""
import pytest
import torch

from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import Float8BlockQuantizer
from transformer_engine.pytorch.triton_kernels.blockwise_fp8_gemm import gemm_blockwise
from transformer_engine.pytorch.triton_kernels.blockwise_fp8_quantize import (
    quantize_fp8_blockwise,
    quantize_fp8_blockwise_weight,
)

DEV = "cuda"
E4M3 = torch.float8_e4m3fn


def _cdiff(x, y):
    x = x.double().flatten(); y = y.double().flatten()
    return (1 - (2 * (x * y).sum()) / ((x * x).sum() + (y * y).sum())).item()


def _sqnr(r, t):
    r = r.double(); t = t.double()
    return (10 * torch.log10((r ** 2).sum() / (((r - t) ** 2).sum() + 1e-30))).item()


def _quantize(t, dim):
    q = Float8BlockQuantizer(fp8_dtype=TE_DType[E4M3], rowwise=True, columnwise=True,
                             amax_epsilon=0.0, force_pow_2_scales=False, block_scaling_dim=dim)
    return q.update_quantized(t.contiguous(),
                              q.make_empty(tuple(t.shape), dtype=E4M3, device=DEV, requires_grad=False))


def _deq_row(t):  # 1x128 along K
    f, s = quantize_fp8_blockwise(t.contiguous(), E4M3, axis=1)
    return f.float() * s.repeat_interleave(128, 1)[:, :t.shape[1]]


def _deq_col(t):  # 1x128 along M
    f, s = quantize_fp8_blockwise(t.contiguous(), E4M3, axis=0)
    return f.float() * s.repeat_interleave(128, 0)[:t.shape[0], :]


def _deq_2d(t):   # 128x128 weight
    f, s = quantize_fp8_blockwise_weight(t.contiguous(), E4M3)
    return f.float() * s.repeat_interleave(128, 0).repeat_interleave(128, 1)[:t.shape[0], :t.shape[1]]


def _gen(shape, mode):
    if mode == "normal":
        return torch.randn(shape, device=DEV)
    t = torch.randn(shape, device=DEV) * 4.0          # wide dynamic range
    m = torch.rand(shape, device=DEV) < 0.01
    t[m] *= 50                                        # outlier injection (split-accumulator)
    return t


SHAPES = [(256, 512, 256), (257, 512, 256), (256, 512, 257), (256, 520, 256),
          (127, 384, 129), (1, 512, 256), (3, 511, 130)]
SHAPE_IDS = [f"{m}x{k}x{n}" for (m, k, n) in SHAPES]


@pytest.mark.parametrize("mode", ["normal", "wide"])
@pytest.mark.parametrize("M,K,N", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("layout", ["fprop", "dgrad", "wgrad"])
def test_dense_gemm(layout, M, K, N, mode):
    torch.manual_seed(0)
    if layout == "fprop":
        X, W = _gen((M, K), mode), _gen((N, K), mode)
        tri = gemm_blockwise(_quantize(W, 2), _quantize(X, 1), True, False, torch.bfloat16)
        fake = _deq_row(X) @ _deq_2d(W).t()
        full = X @ W.t()
    elif layout == "dgrad":
        dY, W = _gen((M, N), mode), _gen((N, K), mode)
        tri = gemm_blockwise(_quantize(W, 2), _quantize(dY, 1), False, False, torch.bfloat16)
        fake = _deq_row(dY) @ _deq_2d(W)
        full = dY @ W
    else:  # wgrad: both operands colwise (contraction M)
        dY, X = _gen((M, N), mode), _gen((M, K), mode)
        tri = gemm_blockwise(_quantize(X, 1), _quantize(dY, 1), False, True, torch.bfloat16)
        fake = _deq_col(dY).t() @ _deq_col(X)
        full = dY.t() @ X

    e = tri.double() - full.double()
    scale = full.double().abs().mean().item() + 1e-30
    signed = e.mean().item() / scale
    unsigned = e.abs().mean().item() / scale
    n = full.numel()
    signed_tol = max(1e-3, 6.0 * unsigned / (n ** 0.5))   # sample-size-adaptive

    cd = _cdiff(tri, fake)
    assert cd < 1e-3, f"cdiff(tri,fake)={cd:.2e} -> kernel impl bug"
    assert abs(signed) < signed_tol, f"signed_bias={signed:.2e} tol={signed_tol:.2e}"
    assert _sqnr(full, tri) > 20.0, f"SQNR={_sqnr(full, tri):.1f}dB too low"
