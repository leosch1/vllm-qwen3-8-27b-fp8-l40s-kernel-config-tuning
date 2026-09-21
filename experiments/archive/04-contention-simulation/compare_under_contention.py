#!/usr/bin/env python3
"""
Direct, causal test of the resource-contention hypothesis from the writeup:
does the tuned config's kernel-level advantage over default shrink (or
reverse) when the GPU is busy with other concurrent work, the way it
genuinely is during real serving (attention, NCCL, other layers' GEMM
calls)?

Runs the exact same default-vs-tuned comparison as
compare_default_vs_tuned.py, at the 18 tuned batch sizes, while
_noisy_neighbor.py runs as a separate background process continuously
saturating the GPU with unrelated matmuls. Compare this script's output
against compare_default_vs_tuned.py's (isolated) numbers for the same
batch sizes -- if contention is the real mechanism, the gap between the two
should be largest exactly where the tuned configs use the most SM
resources (highest num_warps/num_stages), i.e. around M=48-64 per the
writeup's own table.
"""

import json
import subprocess
import sys
import time

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
NUM_ITERS = 500

NEIGHBOR_MATMUL_SIZE = 4096  # M=N=K for the background load


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
    neighbor = subprocess.Popen(
        [sys.executable, "_noisy_neighbor.py", str(NEIGHBOR_MATMUL_SIZE)]
    )
    print(f"noisy neighbor started (pid {neighbor.pid}), warming up...")
    time.sleep(5)
    if neighbor.poll() is not None:
        raise RuntimeError("noisy neighbor process died before warmup finished")

    try:
        device_name = get_device_name_as_file_name()

        for N, K in SHAPES:
            json_path = (
                f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
                f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
            )
            with open(json_path) as f:
                tuned_configs = {int(k): v for k, v in json.load(f).items()}

            print(f"\n### N={N}, K={K}, device={device_name} (under contention)")
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

                out_default = w8a8_block_matmul(
                    A, B, As, Bs, [BLOCK_N, BLOCK_K], DEFAULT_CONFIG, OUT_DTYPE
                )
                out_tuned = w8a8_block_matmul(
                    A, B, As, Bs, [BLOCK_N, BLOCK_K], tuned_configs[M], OUT_DTYPE
                )
                max_diff = (out_default.float() - out_tuned.float()).abs().max().item()

                print(f"| {M} | {default_us:.2f} | {tuned_us:.2f} | {speedup:+.1f}% | {max_diff:.4f} |")
                if neighbor.poll() is not None:
                    raise RuntimeError("noisy neighbor process died mid-run")
    finally:
        neighbor.terminate()
        try:
            neighbor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            neighbor.kill()


if __name__ == "__main__":
    main()
