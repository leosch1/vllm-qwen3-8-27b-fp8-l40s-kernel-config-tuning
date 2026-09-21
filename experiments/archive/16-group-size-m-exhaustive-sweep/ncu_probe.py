#!/usr/bin/env python3
"""ncu L2/DRAM probe: GROUP_SIZE_M candidates at M=1024, gate_up_proj.

Tests whether `GROUP_SIZE_M=1`'s isolated-timing win at M=1024 (563.05us vs
595.38us for GROUP_SIZE_M=64, a clean, non-noise 5.7% gap -- see
`group_size_m_exhaustive.py`'s results) comes from higher L2 hit rate (the
naive cache-locality prediction, which predicts GROUP_SIZE_M=16/32/64
should win, not 1) or from higher achieved DRAM bandwidth utilization via
greater memory-level parallelism (the alternative hypothesis: GROUP_SIZE_M=1
spreads a wave's concurrent blocks across ~136 distinct B-tiles at once --
maximal request diversity -- while grouped values concentrate on a handful
of reused tiles, which could under-fill the memory pipeline's outstanding-
request depth even though it moves fewer total bytes).

Same fixed-A/B-reused-across-conditions methodology as the timing sweep and
as the real tuning script -- one A, one B, generated once, reused for every
launch below.

Layout (0-indexed launch count, matching-kernel-name only):
  GROUP_SIZE_M=1:  launches 0-9 warmup, launch 10 = probe   (skip=10)
  GROUP_SIZE_M=16: launches 11-20 warmup, launch 21 = probe (skip=21)
  GROUP_SIZE_M=32: launches 22-31 warmup, launch 32 = probe (skip=32)
  GROUP_SIZE_M=64: launches 33-42 warmup, launch 43 = probe (skip=43)
"""
import json

import torch

from _vendored_matmul_timing import w8a8_block_matmul
from vllm.utils.platform_utils import get_device_name_as_file_name

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
N, K = 17408, 5120  # gate_up_proj
M = 1024
GROUP_CANDIDATES = [1, 16, 32, 64]
WARMUP_PER_CONDITION = 10


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

    A, As = make_A(M, K)
    B, Bs = make_B(N, K)
    torch.cuda.synchronize()

    launch_idx = 0
    for g in GROUP_CANDIDATES:
        cfg = dict(base_config)
        cfg["GROUP_SIZE_M"] = g
        for i in range(WARMUP_PER_CONDITION + 1):
            w8a8_block_matmul(A, B, As, Bs, [BLOCK_N, BLOCK_K], cfg, OUT_DTYPE)
            torch.cuda.synchronize()
            is_probe = i == WARMUP_PER_CONDITION
            print(f"launch {launch_idx}: GROUP_SIZE_M={g}{' <- PROBE' if is_probe else ''}", flush=True)
            launch_idx += 1


if __name__ == "__main__":
    main()
