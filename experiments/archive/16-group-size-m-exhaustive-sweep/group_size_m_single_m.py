#!/usr/bin/env python3
"""GROUP_SIZE_M single-M comparison, no cross-M eviction (experiment 16, follow-up).

The full 18-M exhaustive sweep (`group_size_m_exhaustive.py`) round-robins
across ALL 72 (M, GROUP_SIZE_M) conditions -- meaning every 50-iteration
block for (M=1024, GROUP_SIZE_M=1) is immediately preceded by a block using
a DIFFERENT M's A/B tensors (different physical memory), so it starts cold
every single time and only partially warms up within its own 50 iterations.
The isolated ncu probe (deeply warmed by 10 dedicated prior launches of the
SAME config, no cross-M interruption) found the opposite ordering --
GROUP_SIZE_M=1 losing by ~11-12%, backed by consistent L2-hit-rate/DRAM-byte
evidence -- suggesting the exhaustive sweep's cross-M cold-start transients
may have biased its result.

This script tests that directly: round-robins ONLY across the 4
GROUP_SIZE_M candidates at a SINGLE fixed M=1024, using ONE A/B pair
allocated once and never touched by any other M's tensors. If this
reproduces the exhaustive sweep's ordering (GROUP_SIZE_M=1 wins), the
cross-M contamination theory is wrong. If it reproduces ncu's ordering
(grouped wins), the theory is confirmed.
"""
import json
import statistics

import torch

from _vendored_matmul_timing import w8a8_block_matmul
from vllm.utils.platform_utils import get_device_name_as_file_name

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
N, K = 17408, 5120  # gate_up_proj
M = 1024
GROUP_CANDIDATES = [1, 16, 32, 64]

ITERS_PER_BLOCK = 50
N_REPEATS = 32  # 50*32 = 1600 iterations/condition -- double the exhaustive sweep's sample


def make_A(m, k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    A_fp32 = (torch.rand(m, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    As = torch.rand(m, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    return A, As


def make_B(n, k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    n_tiles = (n + BLOCK_N - 1) // BLOCK_N
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    B_fp32 = (torch.rand(n, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    return B, Bs


def run_block(A, As, B, Bs, config, n_iters):
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    latencies = []
    for _ in range(n_iters):
        torch.accelerator.synchronize()
        start_event.record()
        w8a8_block_matmul(A, B, As, Bs, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event) * 1000)  # ms -> us
    return latencies


def main():
    device_name = get_device_name_as_file_name()
    json_path = (
        f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
        f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )
    with open(json_path) as f:
        tuned_configs = {int(k): v for k, v in json.load(f).items()}
    base_config = dict(tuned_configs[M])
    print(f"base_config (M={M}) = {base_config}", flush=True)

    # ONE A/B pair for the WHOLE script -- never touched by any other M.
    A, As = make_A(M, K)
    B, Bs = make_B(N, K)
    torch.accelerator.synchronize()
    print("tensors allocated (single M, no cross-M switching)", flush=True)

    configs = {g: {**base_config, "GROUP_SIZE_M": g} for g in GROUP_CANDIDATES}

    print("\n=== warmup ===", flush=True)
    for g, cfg in configs.items():
        run_block(A, As, B, Bs, cfg, 5)
    torch.accelerator.synchronize()

    per_block_means = {g: [] for g in GROUP_CANDIDATES}
    print(f"\n=== {N_REPEATS} repeats x {ITERS_PER_BLOCK} iters/block, "
          f"round-robin across ONLY the 4 GROUP_SIZE_M candidates ===", flush=True)
    for r in range(N_REPEATS):
        for g, cfg in configs.items():
            lat = run_block(A, As, B, Bs, cfg, ITERS_PER_BLOCK)
            per_block_means[g].append(sum(lat) / len(lat))
        print(f"  repeat {r + 1}/{N_REPEATS}: " +
              "  ".join(f"GM={g}={per_block_means[g][-1]:.2f}us" for g in GROUP_CANDIDATES),
              flush=True)

    print("\n=== Results: mean +/- stderr (us) ===")
    means = {}
    for g in GROUP_CANDIDATES:
        blocks = per_block_means[g]
        mean = statistics.mean(blocks)
        stderr = statistics.stdev(blocks) / (len(blocks) ** 0.5)
        means[g] = mean
        print(f"  GROUP_SIZE_M={g:>3}: {mean:8.2f} +/- {stderr:5.2f} us  (n={len(blocks)} blocks x {ITERS_PER_BLOCK} iters)")

    winner = min(means, key=means.get)
    print(f"\nWinner: GROUP_SIZE_M={winner}")
    print(f"GROUP_SIZE_M=1 vs winner: {(means[1] / means[winner] - 1) * 100:+.2f}%")
    print("\nFor reference:")
    print("  Exhaustive (cross-M round-robin) sweep found: GROUP_SIZE_M=1 wins by 5.7%")
    print("  Isolated ncu probe (deeply pre-warmed, no cross-M) found: GROUP_SIZE_M=1 LOSES by ~11-12%")


if __name__ == "__main__":
    main()
