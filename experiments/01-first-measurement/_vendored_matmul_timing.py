#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# https://www.apache.org/licenses/LICENSE-2.0
"""
The two functions in this file were copied verbatim from vllm-project/vllm's benchmarks/kernels/benchmark_w8a8_block_fp8.py:
https://github.com/vllm-project/vllm/blob/v0.27.1/benchmarks/kernels/benchmark_w8a8_block_fp8.py#L38-L115
https://github.com/vllm-project/vllm/blob/v0.27.1/benchmarks/kernels/benchmark_w8a8_block_fp8.py#L175-L199

They are needed by compare_default_vs_tuned.py so that matmul and timing use the same logic as during autotuning with benchmarks/kernels/benchmark_w8a8_block_fp8.py.
Regular python import (import ...) would not work since benchmarks/kernels/ isn't part of the installed vllm package.
"""

from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.triton_utils import triton


def w8a8_block_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    config: dict[str, Any],
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert triton.cdiv(N, block_n) == Bs.shape[0]
    assert triton.cdiv(K, block_k) == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    if A.dtype == torch.float8_e4m3fn:
        kernel = _w8a8_triton_block_scaled_mm
    else:
        raise RuntimeError("Currently, only support tune w8a8 block fp8 kernel.")

    kernel[grid](
        A, B, C, As, Bs, M, N, K, block_n, block_k,
        A.stride(-2), A.stride(-1),
        B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1),
        As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        **config,
    )

    return C


def benchmark_config(
    A, B, As, Bs, block_size, config, out_dtype=torch.float16, num_iters=10
):
    def run():
        w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)

    torch.accelerator.synchronize()
    # JIT complication & warmup
    for _ in range(5):
        run()
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        torch.accelerator.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg
