#!/usr/bin/env python3
"""
Default-vs-tuned comparison, PER-POINT randomized order: unlike the fully-
randomized rerun (compare_default_vs_tuned_fully_randomized.py), which
shuffles launches across ALL 5 shapes x 35 M-points x 2 configs together,
this randomizes ONLY within each (shape, M) point independently -- one
fixed (A,B,As,Bs) tuple per point (this experiment's original always-warm
design, so the weight tensor is retrievable from L2/already resident --
NOT the cache-cycling variant), 2000 "default" + 2000 "tuned" labels
shuffled together and executed strictly in that per-point-local order.

This isolates a narrower question than the fully-randomized rerun: does
merely randomizing WHICH CONFIG RUNS WHEN (holding the same weight tensor
fixed throughout, no other shapes/M-values interspersed at all) already
remove the bias the original sequential-block design showed -- or does
that bias require the broader cross-shape/cross-M diversity the fully-
randomized design provides? Comparing this experiment's result against
both the original (chart.html) and the fully-randomized one
(chart-fully-randomized.html) tells us which randomization axis is doing
the real work.

Same 175-point grid, same DEFAULT_CONFIG and tuned-configs/ lookup, same
NUM_ITERS=2000 per config per point, same /10 parity scaling as every
other script in this project.
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

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    for shape, (N, K) in SHAPES.items():
        with open(f"{TUNED_DIR}/N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json") as f:
            tuned_cfgs = {int(k): v for k, v in json.load(f).items()}
        anchors = sorted(tuned_cfgs)

        print(f"\n### {shape} (N={N}, K={K}), per-point randomized order")
        print("| M | type | nearest anchor | default (us) | tuned (us) | speedup |")
        print("|---:|---|---:|---:|---:|---:|")

        for M, ptype in points:
            nearest = min(anchors, key=lambda x: abs(x - M))
            cfg_tuned = tuned_cfgs[nearest]

            A, B, As, Bs = make_tensors(M, N, K, BLOCK_N, BLOCK_K)

            # warmup both configs (5 each), untimed
            for _ in range(5):
                time_one(A, B, As, Bs, [BLOCK_N, BLOCK_K], DEFAULT_CONFIG, OUT_DTYPE, start_event, end_event)
                time_one(A, B, As, Bs, [BLOCK_N, BLOCK_K], cfg_tuned, OUT_DTYPE, start_event, end_event)

            # per-point randomized order: 2000 "default" + 2000 "tuned" labels, shuffled together
            labels = ["default"] * NUM_ITERS + ["tuned"] * NUM_ITERS
            random.shuffle(labels)

            latencies = {"default": [], "tuned": []}
            for label in labels:
                cfg = DEFAULT_CONFIG if label == "default" else cfg_tuned
                lat = time_one(A, B, As, Bs, [BLOCK_N, BLOCK_K], cfg, OUT_DTYPE, start_event, end_event)
                latencies[label].append(lat)

            d_us = statistics.mean(l * 1000.0 / 10 for l in latencies["default"])
            t_us = statistics.mean(l * 1000.0 / 10 for l in latencies["tuned"])
            speedup = (d_us - t_us) / d_us * 100
            print(f"| {M} | {ptype} | {nearest} | {d_us:.2f} | {t_us:.2f} | {speedup:+.1f}% |")


if __name__ == "__main__":
    main()
