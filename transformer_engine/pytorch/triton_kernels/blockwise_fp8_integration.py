# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Glue between TE's Float8BlockQuantizer / Float8BlockwiseQTensor and the ported
# ROCm Triton blockwise-FP8 kernels. Only used when NVTE_USE_FP8_BLOCK_SCALING_TRITON
# is on (ROCm gfx950). Carries our own scale layout in the tensor's data fields;
# the matching GEMM consumer (cpp_extensions/gemm.py) reads the same fields back.

import torch
import transformer_engine_torch as tex

from .common import te_dtype_to_torch_dtype
from .blockwise_fp8_quantize import (
    quantize_fp8_blockwise,
    quantize_fp8_blockwise_dual,
    quantize_fp8_blockwise_weight,
)

BLOCK = 128


def _u8(fp8_tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret an FP8 tensor as uint8 (TE stores blockwise data as uint8)."""
    return fp8_tensor.view(torch.uint8)


def quantize_into(quantizer, src: torch.Tensor, dst) -> None:
    """Triton blockwise-quantize `src` into the Float8BlockwiseQTensor `dst` fields.

    Honors quantizer.block_scaling_dim (1 = 1x128 activation, 2 = 128x128 weight)
    and rowwise/columnwise usage. Both directions are always produced from the
    ORIGINAL high-precision `src` (no double-quantization).
    """
    dt = te_dtype_to_torch_dtype(quantizer.dtype)  # torch fp8 dtype (e4m3fn on gfx950)
    orig_shape = tuple(src.shape)
    x = src.reshape(-1, orig_shape[-1]).contiguous()  # [M, K]

    if quantizer.block_scaling_dim == 2:
        # 128x128 weight blocks.
        if quantizer.rowwise_usage:
            fp8, scale = quantize_fp8_blockwise_weight(x, dt, BLOCK)
            dst._rowwise_data = _u8(fp8).reshape(orig_shape)
            dst._rowwise_scale_inv = scale
        if quantizer.columnwise_usage:
            fp8c, scalec = quantize_fp8_blockwise_weight(x.t().contiguous(), dt, BLOCK)
            dst._columnwise_data = _u8(fp8c)
            dst._columnwise_scale_inv = scalec
    else:
        # 1x128 activation blocks.
        if quantizer.rowwise_usage and quantizer.columnwise_usage:
            row, srow, col, scol = quantize_fp8_blockwise_dual(x, dt, BLOCK)
            dst._rowwise_data = _u8(row).reshape(orig_shape)
            dst._rowwise_scale_inv = srow
            dst._columnwise_data = _u8(col).reshape(orig_shape)
            dst._columnwise_scale_inv = scol
        elif quantizer.rowwise_usage:
            row, srow = quantize_fp8_blockwise(x, dt, axis=1, block_size=BLOCK)
            dst._rowwise_data = _u8(row).reshape(orig_shape)
            dst._rowwise_scale_inv = srow
        else:
            col, scol = quantize_fp8_blockwise(x, dt, axis=0, block_size=BLOCK)
            dst._columnwise_data = _u8(col).reshape(orig_shape)
            dst._columnwise_scale_inv = scol

    dst._data_format = tex.Float8BlockScaleTensorFormat.GEMM_READY
    dst._fp8_dtype = quantizer.dtype
