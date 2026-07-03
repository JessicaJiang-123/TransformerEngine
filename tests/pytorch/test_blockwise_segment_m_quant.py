# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Self-written verification for ``quantize_fp8_blockwise_segment_m``.

No existing TE / sglang / Primus-Turbo test covers segment_m (the MoE grouped-wgrad
colwise operand). Golden = per-group INDEPENDENT colwise (axis=0) 1x128 quantization
with per-segment padding to a multiple of 128, matching the kernel's documented
semantics. Judged with fp8-ULP tolerance on data and fp32-ULP tolerance on scale
(the 2-ULP scale slack absorbs triton's GPU rcp-based fp32 divide vs torch IEEE).
"""
import pytest
import torch

from transformer_engine.pytorch.triton_kernels.blockwise_fp8_quantize import (
    quantize_fp8_blockwise_segment_m,
)

DEV = "cuda"
BL = 128
QDT = torch.float8_e4m3fn  # gfx950: e4m3fn / FMAX=448


def _golden(x, gl, N, dtype):
    FMX = torch.finfo(dtype).max
    offs = [0]
    for L in gl:
        offs.append(offs[-1] + int(L))
    plens = [((int(L) + BL - 1) // BL) * BL for L in gl]
    pofs = [0]
    for pl in plens:
        pofs.append(pofs[-1] + pl)
    Mp = pofs[-1]
    q = torch.zeros(Mp, N, device=DEV, dtype=dtype)
    s = torch.zeros(Mp // BL, N, device=DEV, dtype=torch.float32)
    for g, L in enumerate(gl):
        L = int(L)
        if L == 0:
            continue
        xg = x[offs[g]:offs[g + 1]].float()
        ps = pofs[g]
        for b in range(plens[g] // BL):
            m0 = b * BL
            rows = xg[m0:min(m0 + BL, L)]
            amax = rows.abs().amax(0) if rows.shape[0] > 0 else torch.zeros(N, device=DEV)
            amax = torch.maximum(amax, torch.full_like(amax, 1e-4))
            scale = FMX / amax
            q[ps + m0:ps + m0 + rows.shape[0]] = (rows * scale).clamp(-FMX, FMX).to(dtype)
            s[ps // BL + b] = 1.0 / scale
    return q, s, Mp, plens, pofs


def _fp8_ulp(a, b):
    au = a.view(torch.uint8).int()
    bu = b.view(torch.uint8).int()
    sa, sb = au & 0x80, bu & 0x80
    ma, mb = au & 0x7F, bu & 0x7F
    dist = torch.where(sa == sb, (ma - mb).abs(), ma + mb)
    sign_err = int(((sa != sb) & (ma != 0) & (mb != 0)).sum())
    return sign_err, int(dist.max()), float((dist > 0).float().mean())


CASES = {
    "even_3x128": [128, 128, 128],
    "varlen_nonmult": [100, 200, 45],
    "empty_group": [128, 0, 64],
    "single_token": [1, 130, 1],
    "adversarial_1e-3": [130, 130],
    "all_zero_group": [128, 64],
}
SCALES = {"adversarial_1e-3": [1.0, 1e-3], "all_zero_group": [0.0, 1.0]}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("N", [256, 272], ids=["N256", "N272"])
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_segment_m_quant(case, N, dtype):
    gl = CASES[case]
    scales = SCALES.get(case)
    glt = torch.tensor([int(x) for x in gl], dtype=torch.int64, device=DEV)
    go = torch.zeros(len(gl) + 1, dtype=torch.int64, device=DEV)
    go[1:] = torch.cumsum(glt, 0)
    parts = [
        (torch.randn(int(L), N, device=DEV) * (scales[i] if scales else 1.0)).to(dtype)
        for i, L in enumerate(gl)
        if int(L) > 0
    ]
    x = torch.cat(parts, 0).contiguous()

    xf8, xs, vkl, vko = quantize_fp8_blockwise_segment_m(x, QDT, BL, glt, go)
    q, s, Mp, plens, pofs = _golden(x, gl, N, QDT)

    # 1) structure: padded group lengths / offsets
    assert vkl[:len(plens)].tolist() == plens
    assert int(vko[len(plens)]) == Mp

    # 2) padding rows must be exactly zero (else wgrad gets polluted)
    for g, L in enumerate(gl):
        L = int(L)
        if 0 < L and pofs[g] + L < pofs[g + 1]:
            assert xf8[pofs[g] + L:pofs[g + 1]].float().abs().max().item() == 0.0

    # 3) data: fp8 <=1 ULP, no sign error, <0.5% jitter
    se, ud, ur = _fp8_ulp(xf8[:Mp], q)
    assert se == 0 and ud <= 1 and ur < 0.005, f"data sign={se} ulp={ud} ratio={ur}"

    # 4) scale: fp32 <=2 ULP (GPU rcp divide slack)
    su = int(
        (xs[:Mp // BL].contiguous().view(torch.int32) - s.contiguous().view(torch.int32))
        .abs()
        .max()
    )
    assert su <= 2, f"scale fp32 ULP={su}"
