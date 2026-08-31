#!/usr/bin/env python3
"""
Default-vs-tuned kernel comparison for vLLM's dense W8A8 block-FP8 Triton GEMM
kernel: latency, and whether the two configs actually agree numerically.

Why this exists: neither of vLLM's own two relevant scripts do this.
`benchmarks/kernels/benchmark_w8a8_block_fp8.py` (the tuner) searches for the
fastest config and discards the timing once it picks a winner.
`benchmarks/kernels/benchmark_block_fp8_gemm.py` is designed to do a real
before/after comparison through vLLM's production code path, but is broken on
vllm/vllm-openai:v0.27.1.
"""

import json

import torch

from _vendored_matmul_timing import benchmark_config, w8a8_block_matmul
from vllm.utils.platform_utils import get_device_name_as_file_name

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}

# Same 18-value batch-size grid benchmark_w8a8_block_fp8.py's main() uses by default
BATCH_SIZES = [
    1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256,
    512, 1024, 1536, 2048, 3072, 4096,
]

# Qwen/Qwen3.8-27B-FP8, tensor_parallel_size=2, NVIDIA L40S dense W8A8 block-FP8 shapes
# Captured from the live predictor pod's "Using default..." warnings.
# Cross-checked against the model's config.json.
SHAPES = [
    (17408, 5120),
    (8192, 5120),
    (7168, 5120),
    (5120, 8704),
    (5120, 3072),
]
BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
NUM_ITERS = 50  # more than tune()'s internal 10, cheap since only ~180 calls total


def make_tensors(M, N, K, block_n, block_k):
    # Mirrors benchmark_w8a8_block_fp8.py's own tune() tensor construction.
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    factor = 1e-2

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * factor
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * factor
    return A, B, As, Bs


def main():
    device_name = get_device_name_as_file_name()

    for N, K in SHAPES:
        json_path = (
            f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
            f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
        )
        with open(json_path) as f:
            tuned_configs = {int(k): v for k, v in json.load(f).items()}

        print(f"\n### N={N}, K={K}, device={device_name}")
        print("| batch size (M) | default (us) | tuned (us) | speedup | max output diff |")
        print("|---:|---:|---:|---:|---:|")
        for M in BATCH_SIZES:
            A, B, As, Bs = make_tensors(M, N, K, BLOCK_N, BLOCK_K)
            default_us = benchmark_config(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], DEFAULT_CONFIG, OUT_DTYPE,
                num_iters=NUM_ITERS,
            )
            tuned_us = benchmark_config(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], tuned_configs[M], OUT_DTYPE,
                num_iters=NUM_ITERS,
            )
            speedup = (default_us - tuned_us) / default_us * 100

            # Sanity check: result should stay the same.
            out_default = w8a8_block_matmul(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], DEFAULT_CONFIG, OUT_DTYPE
            )
            out_tuned = w8a8_block_matmul(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], tuned_configs[M], OUT_DTYPE
            )
            max_diff = (out_default.float() - out_tuned.float()).abs().max().item()

            print(f"| {M} | {default_us:.2f} | {tuned_us:.2f} | {speedup:+.1f}% | {max_diff:.4f} |")


if __name__ == "__main__":
    main()
