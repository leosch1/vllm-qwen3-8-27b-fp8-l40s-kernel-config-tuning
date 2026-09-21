"""
Isolated kernel benchmark: default vs. cache-aware config (17's round-2
autotuning-script output). Mirrors compare_default_vs_tuned.py exactly --
same DEFAULT_CONFIG, BATCH_SIZES, HELD_OUT_BATCH_SIZES, SHAPES, NUM_ITERS,
benchmark_config()/w8a8_block_matmul (vendored verbatim from vLLM's own
benchmark_w8a8_block_fp8.py) -- only TUNED_DIR points at the cache-aware
configs instead of this repo's original tuned-configs/.
"""

import json
from typing import Any

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.triton_utils import triton
from vllm.utils.platform_utils import get_device_name_as_file_name


def w8a8_block_matmul(A, B, As, Bs, block_size, config, output_dtype=torch.float16):
    block_n, block_k = block_size[0], block_size[1]
    M = A.numel() // A.shape[-1]
    N, K = B.shape
    C = A.new_empty(A.shape[:-1] + (N,), dtype=output_dtype)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    kernel = _w8a8_triton_block_scaled_mm
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


def benchmark_config(A, B, As, Bs, block_size, config, out_dtype=torch.float16, num_iters=10):
    def run():
        w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)

    torch.accelerator.synchronize()
    for _ in range(5):
        run()
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    latencies = []
    for _ in range(num_iters):
        torch.accelerator.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    # /10 matches _vendored_matmul_timing.py's own benchmark_config() exactly
    # (the documented upstream bug -- see experiments/README.md's "Known
    # caveat" section) -- kept for byte-for-byte methodology parity with 01;
    # a constant factor, cancels in the speedup ratio this script reports.
    return sum(latencies) / (num_iters * 10) * 1000


DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}

BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
HELD_OUT_BATCH_SIZES = [3, 6, 12, 20, 28, 40, 56, 80, 112, 160, 200, 384, 768, 1280, 1792, 2560, 3584]

SHAPES = [(17408, 5120), (8192, 5120), (7168, 5120), (5120, 8704), (5120, 3072)]
BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
GROUPFIX_DIR = "/tmp/groupfix-configs"
NUM_ITERS = 2000


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
    torch.cuda.init()
    device_name = get_device_name_as_file_name()

    for N, K in SHAPES:
        json_path = (
            f"{GROUPFIX_DIR}/N={N},K={K},device_name={device_name},"
            f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
        )
        with open(json_path) as f:
            groupfix_configs = {int(k): v for k, v in json.load(f).items()}
        anchors = list(groupfix_configs.keys())

        points = [(M, "anchor") for M in BATCH_SIZES]
        points += [(M, "held-out") for M in HELD_OUT_BATCH_SIZES]
        points.sort()

        print(f"\n### N={N}, K={K}, device={device_name}")
        print("| M | type | nearest anchor | default (us) | groupfix (us) | speedup | max output diff |")
        print("|---:|---|---:|---:|---:|---:|---:|")
        for M, ptype in points:
            nearest = M if ptype == "anchor" else min(anchors, key=lambda x: abs(x - M))
            selected_config = groupfix_configs[nearest]

            A, B, As, Bs = make_tensors(M, N, K, BLOCK_N, BLOCK_K)
            default_us = benchmark_config(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], DEFAULT_CONFIG, OUT_DTYPE, num_iters=NUM_ITERS,
            )
            groupfix_us = benchmark_config(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], selected_config, OUT_DTYPE, num_iters=NUM_ITERS,
            )
            speedup = (default_us - groupfix_us) / default_us * 100

            out_default = w8a8_block_matmul(A, B, As, Bs, [BLOCK_N, BLOCK_K], DEFAULT_CONFIG, OUT_DTYPE)
            out_groupfix = w8a8_block_matmul(A, B, As, Bs, [BLOCK_N, BLOCK_K], selected_config, OUT_DTYPE)
            max_diff = (out_default.float() - out_groupfix.float()).abs().max().item()

            print(
                f"| {M} | {ptype} | {nearest} | {default_us:.2f} | {groupfix_us:.2f} "
                f"| {speedup:+.1f}% | {max_diff:.4f} |"
            )


if __name__ == "__main__":
    main()
