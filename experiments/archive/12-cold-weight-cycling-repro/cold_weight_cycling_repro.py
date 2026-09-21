#!/usr/bin/env python3
"""Cold-weight-cycling reproduction (experiment 12).

Tests a sharper, more specific version of the cache-locality hypothesis from
09-isolated-vs-real-flops-inflation / 05's Tensor-Active/DRAM-Read finding:
real serving forces every `gate_up_proj` call to read a *different* layer's
weight matrix (64 distinct ~89MB tensors cycling through a 96MB L2 cache),
never the same one twice in a row -- unlike the isolated benchmark, which
calls the identical kernel on the identical `B` tensor thousands of times in
a row (mostly L2-resident after the first call).

This is a single-process, single-stream, single-GEMM-shape harness -- no
other kernel types interleaved, no multi-request scheduling, no CUDA graphs.
It isolates exactly one variable (does `B` change every call or not) that
`11-interleaved-cache-locality-repro`'s two designs did not test: both of
those varied *which other kernels* ran in between repeats of `gate_up_proj`,
but (as far as can be determined) kept calling `gate_up_proj` on the *same*
`B` tensor throughout -- meaning neither design actually forced the cold
DRAM refetch that real serving's 64-distinct-layers structure guarantees.

Not using `_vendored_matmul_timing.benchmark_config()` here (it also has the
documented /10 upstream bug, and more importantly its `run()` closure has no
way to vary the weight tensor per iteration) -- this reimplements the same
CUDA-event bracketing methodology directly, with a correct /num_iters
division.
"""
import json
import torch

from _vendored_matmul_timing import w8a8_block_matmul
from vllm.utils.platform_utils import get_device_name_as_file_name

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
NUM_ITERS = 2000
M = 2048
N, K = 17408, 5120  # gate_up_proj -- the one shape/M that regresses in real serving
NUM_WEIGHT_COPIES = 64  # matches the real model's actual layer count

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}


def make_A(m, k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    factor = 1e-2
    A_fp32 = (torch.rand(m, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    As = torch.rand(m, k_tiles, dtype=torch.float32, device="cuda") * factor
    return A, As


def make_B_copies(n_copies, n, k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    factor = 1e-2
    n_tiles = (n + BLOCK_N - 1) // BLOCK_N
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    copies = []
    for _ in range(n_copies):
        B_fp32 = (torch.rand(n, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
        B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * factor
        copies.append((B, Bs))
    return copies


def time_config(A, As, B_copies, config, num_iters):
    def run(i):
        B, Bs = B_copies[i % len(B_copies)]
        return w8a8_block_matmul(A, B, As, Bs, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)

    torch.accelerator.synchronize()
    for i in range(5):
        run(i)
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies = []
    for i in range(num_iters):
        torch.accelerator.synchronize()
        start_event.record()
        run(i)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg_us = sum(latencies) / num_iters * 1000  # ms -> us, correct division (no /10 bug)
    return avg_us


def main():
    device_name = get_device_name_as_file_name()
    print(f"device_name={device_name}")

    json_path = (
        f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
        f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )
    with open(json_path) as f:
        tuned_configs = {int(k): v for k, v in json.load(f).items()}
    tuned_config = tuned_configs[M]
    print(f"tuned_config={tuned_config}")

    print(f"Allocating A (fixed) + {NUM_WEIGHT_COPIES} distinct B copies "
          f"for gate_up_proj (N={N},K={K}), M={M}...")
    A, As = make_A(M, K)
    B_copies = make_B_copies(NUM_WEIGHT_COPIES, N, K)
    total_mb = NUM_WEIGHT_COPIES * N * K / 1e6
    print(f"Total weight-copy footprint: {total_mb:.1f} MB "
          f"(single copy: {N * K / 1e6:.1f} MB, L40S L2 cache: 96 MB)")

    print("\n=== Baseline: same B every call (matches isolated_all_shapes.py) ===")
    default_same = time_config(A, As, [B_copies[0]], DEFAULT_CONFIG, NUM_ITERS)
    tuned_same = time_config(A, As, [B_copies[0]], tuned_config, NUM_ITERS)
    print(f"  default: {default_same:.1f}us  tuned: {tuned_same:.1f}us  "
          f"isolated speedup: {(default_same - tuned_same) / default_same * 100:+.1f}%")

    print(f"\n=== Cold: cycling through {NUM_WEIGHT_COPIES} distinct B copies ===")
    default_cold = time_config(A, As, B_copies, DEFAULT_CONFIG, NUM_ITERS)
    tuned_cold = time_config(A, As, B_copies, tuned_config, NUM_ITERS)
    print(f"  default: {default_cold:.1f}us  tuned: {tuned_cold:.1f}us  "
          f"cold speedup: {(default_cold - tuned_cold) / default_cold * 100:+.1f}%")

    print("\n=== Summary: same-B -> cold-cycling shift, per config ===")
    print(f"  default: {default_same:.1f}us -> {default_cold:.1f}us  "
          f"({(default_cold / default_same - 1) * 100:+.1f}%)")
    print(f"  tuned:   {tuned_same:.1f}us -> {tuned_cold:.1f}us  "
          f"({(tuned_cold / tuned_same - 1) * 100:+.1f}%)")
    print("\nFor reference, real serving (from 05's profiling-results): "
          "default ~1181us (flat/faster than isolated), tuned ~2057us (+76%).")


if __name__ == "__main__":
    main()
