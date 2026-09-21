#!/usr/bin/env python3
"""Isolated benchmark for ALL 5 W8A8 shapes at M=2048, default vs tuned,
to test whether GEMM "heaviness" (FLOPs) predicts the severity of the
real-vs-isolated inflation seen in real serving.
"""
import json
import torch

from _vendored_matmul_timing import benchmark_config
from vllm.utils.platform_utils import get_device_name_as_file_name

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
NUM_ITERS = 2000
M = 2048

SHAPES = [
    ("gate_up_proj", 17408, 5120),
    ("in_proj_qkvz", 8192, 5120),
    ("qkv_proj", 7168, 5120),
    ("down_proj", 5120, 8704),
    ("out_proj", 5120, 3072),
]

DEFAULT_CONFIG_TEMPLATE = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}


def make_tensors(M, N, K, block_n, block_k):
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
    print(f"device_name={device_name}")

    for role, N, K in SHAPES:
        json_path = (
            f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
            f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
        )
        with open(json_path) as f:
            tuned_configs = {int(k): v for k, v in json.load(f).items()}
        selected_config = tuned_configs[M]

        A, B, As, Bs = make_tensors(M, N, K, BLOCK_N, BLOCK_K)

        default_us_trials = []
        tuned_us_trials = []
        for _ in range(3):
            default_us = benchmark_config(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], DEFAULT_CONFIG_TEMPLATE, OUT_DTYPE,
                num_iters=NUM_ITERS,
            )
            tuned_us = benchmark_config(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], selected_config, OUT_DTYPE,
                num_iters=NUM_ITERS,
            )
            default_us_trials.append(default_us)
            tuned_us_trials.append(tuned_us)

        default_mean = sum(default_us_trials) / len(default_us_trials)
        tuned_mean = sum(tuned_us_trials) / len(tuned_us_trials)
        speedup = (default_mean - tuned_mean) / default_mean * 100
        print(f"{role} (N={N},K={K}): tuned_config={selected_config}")
        print(f"  default_trials={[f'{x:.1f}' for x in default_us_trials]} mean={default_mean:.1f}us")
        print(f"  tuned_trials={[f'{x:.1f}' for x in tuned_us_trials]} mean={tuned_mean:.1f}us")
        print(f"  isolated speedup: {speedup:+.1f}%")


if __name__ == "__main__":
    main()
