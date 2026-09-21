#!/usr/bin/env python3
"""
Fully-randomized re-run of this experiment's original default-vs-tuned
comparison (compare_default_vs_tuned.py, at the repo root). Same 175-point
grid (5 shapes x 18 anchor + 17 held-out M), same fixed-tensors-per-point
design (one A/B/As/Bs tuple per (shape, M) point, generated exactly like
the original's make_tensors() -- NOT weight-cycling, this experiment never
used that), same DEFAULT_CONFIG, same tuned-configs/ lookup, same
NUM_ITERS=2000, same per-iteration torch.accelerator.synchronize()-bracketed
timing as vLLM's own benchmark_config().

The only change: instead of running the full 2000-iteration default block
then the full 2000-iteration tuned block sequentially at each point (the
original's design, and every comparison in this project until 19's Parts
8-11 found that design carries an unquantified, sometimes large bias), every
single launch needed across all 350 (shape, M, config) combinations x 2000
iterations = 700,000 total launches is generated as one flat list, shuffled
into a single random order, and executed strictly in that order -- bucketed
back by (shape, M, config) label only after all launches complete. See
19/README.md Part 11 for why this design is trusted over the sequential-
block one: null-point noise collapses ~7x and previously-invisible real
effects become unambiguous (>20 SD from the null mean).

This script does NOT touch or delete the original compare_default_vs_tuned.py
or its results -- it's a new, independent measurement for comparison.
"""

import json
import random
import statistics

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


def time_one(A, B, As, Bs, block_size, config, out_dtype, start_event, end_event):
    torch.accelerator.synchronize()
    start_event.record()
    w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event)


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

SHAPES = {
    "gate_up_proj": (17408, 5120),
    "in_proj_qkvz": (8192, 5120),
    "qkv_proj": (7168, 5120),
    "down_proj": (5120, 8704),
    "out_proj": (5120, 3072),
}
BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "/tmp/tuned-configs"
NUM_ITERS = 2000


def make_tensors(M, N, K, block_n, block_k):
    # Mirrors the original compare_default_vs_tuned.py's make_tensors() exactly.
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

    points = [(M, "anchor") for M in BATCH_SIZES] + [(M, "held-out") for M in HELD_OUT_BATCH_SIZES]
    points.sort()

    print("=== allocating fixed tensors for every (shape, M) point ===")
    shape_data = {}
    for shape, (N, K) in SHAPES.items():
        with open(f"{TUNED_DIR}/N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json") as f:
            tuned_cfgs = {int(k): v for k, v in json.load(f).items()}
        anchors = sorted(tuned_cfgs)

        tensors_by_m = {}
        nearest_by_m = {}
        for M, _ptype in points:
            nearest = min(anchors, key=lambda x: abs(x - M))
            nearest_by_m[M] = nearest
            tensors_by_m[M] = make_tensors(M, N, K, BLOCK_N, BLOCK_K)
        shape_data[shape] = {
            "N": N, "K": K, "tensors_by_m": tensors_by_m,
            "nearest_by_m": nearest_by_m, "tuned_cfgs": tuned_cfgs,
        }
        print(f"  {shape}: N={N}, K={K}, {len(tensors_by_m)} (A,B,As,Bs) tuples allocated")
    torch.accelerator.synchronize()
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("=== warmup: compiling every distinct kernel variant ===")
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    for shape, sd in shape_data.items():
        for M, _ptype in points:
            A, B, As, Bs = sd["tensors_by_m"][M]
            nearest = sd["nearest_by_m"][M]
            for cfg in (DEFAULT_CONFIG, sd["tuned_cfgs"][nearest]):
                time_one(A, B, As, Bs, [BLOCK_N, BLOCK_K], cfg, OUT_DTYPE, start_event, end_event)
    print("warmup done")

    print("=== building and shuffling the full launch sequence ===")
    work = []
    for shape in SHAPES:
        for M, _ptype in points:
            for config_name in ("default", "tuned"):
                work.extend([(shape, M, config_name)] * NUM_ITERS)
    random.shuffle(work)
    print(f"total launches: {len(work)}")

    print("=== executing fully-randomized sequence ===")
    results = {}
    for i, (shape, M, config_name) in enumerate(work):
        sd = shape_data[shape]
        A, B, As, Bs = sd["tensors_by_m"][M]
        nearest = sd["nearest_by_m"][M]
        cfg = DEFAULT_CONFIG if config_name == "default" else sd["tuned_cfgs"][nearest]
        lat = time_one(A, B, As, Bs, [BLOCK_N, BLOCK_K], cfg, OUT_DTYPE, start_event, end_event)
        results.setdefault((shape, M, config_name), []).append(lat)
        if (i + 1) % 100000 == 0:
            print(f"  {i + 1}/{len(work)} launches done")

    print()
    for shape in SHAPES:
        print(f"\n### {shape}, fully-randomized")
        print("| M | type | nearest anchor | default (us) | tuned (us) | speedup |")
        print("|---:|---|---:|---:|---:|---:|")
        for M, ptype in points:
            nearest = shape_data[shape]["nearest_by_m"][M]
            d_lat = results[(shape, M, "default")]
            t_lat = results[(shape, M, "tuned")]
            d_us = statistics.mean(l * 1000.0 / 10 for l in d_lat)
            t_us = statistics.mean(l * 1000.0 / 10 for l in t_lat)
            speedup = (d_us - t_us) / d_us * 100
            print(f"| {M} | {ptype} | {nearest} | {d_us:.2f} | {t_us:.2f} | {speedup:+.1f}% |")


if __name__ == "__main__":
    main()
