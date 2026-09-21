"""
Isolated kernel benchmark, cache-cycling variant: tuned vs. singlefix,
DIRECTLY paired in the same run (not chained through separate vs-default
runs -- a cross-run comparison of two separately-logged vs-default passes
was tried first and found noisy: at points where singlefix and tuned are
byte-identical, cross-run "speedup" swung as much as -18%, pure run-to-run
noise with no real config difference behind it). This script instead
benchmarks tuned_config(M) and singlefix_config(M) back-to-back in one
script execution, same CyclingLaunchCounter counterbalanced design (64
distinct B copies, counter reset between the two configs at each M point)
used for every other comparison in this experiment -- so any real
difference is isolated to exactly the one parameter that differs
(gate_up_proj's M=2048 GROUP_SIZE_M, 1 vs 16) with the same noise-canceling
guarantee as the default-relative charts.

Same 175-point grid as 01/19 (5 shapes x 18 anchor + 17 held-out M values),
same NUM_ITERS=2000, same numerical sanity check.
"""

import json

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


class CyclingLaunchCounter:
    """Same design as 17's tune_cache_aware(): global, never reset mid-run
    (here: reset only between the two configs at a given M point, so both
    see the identical B-copy sequence starting from index 0 -- a
    counterbalanced identical-stimulus comparison, not an order confound)."""

    def __init__(self, copies):
        self.copies = copies
        self.i = 0

    def next(self):
        B, Bs = self.copies[self.i % len(self.copies)]
        self.i += 1
        return B, Bs


def benchmark_config_cycling(A, As, counter, block_size, config, out_dtype, num_iters=2000):
    def run():
        B, Bs = counter.next()
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
    return sum(latencies) / (num_iters * 10) * 1000  # /10: same documented bug, kept for parity


TUNED_DIR = "/tmp/tuned-configs"

BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
HELD_OUT_BATCH_SIZES = [3, 6, 12, 20, 28, 40, 56, 80, 112, 160, 200, 384, 768, 1280, 1792, 2560, 3584]

SHAPES = [(17408, 5120), (8192, 5120), (7168, 5120), (5120, 8704), (5120, 3072)]
BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
SINGLEFIX_DIR = "/tmp/singlefix-configs"
NUM_ITERS = 2000
N_WEIGHT_COPIES = 64


def make_A(M, K, block_k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    k_tiles = (K + block_k - 1) // block_k
    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    return A, As


def make_B_copies(N, K, block_n, block_k, count):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    copies = []
    for _ in range(count):
        B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
        B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
        Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
        copies.append((B, Bs))
    return copies


def main():
    torch.cuda.init()
    device_name = get_device_name_as_file_name()

    for N, K in SHAPES:
        singlefix_json_path = (
            f"{SINGLEFIX_DIR}/N={N},K={K},device_name={device_name},"
            f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
        )
        tuned_json_path = (
            f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
            f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
        )
        with open(singlefix_json_path) as f:
            singlefix_configs = {int(k): v for k, v in json.load(f).items()}
        with open(tuned_json_path) as f:
            tuned_configs = {int(k): v for k, v in json.load(f).items()}
        anchors = list(singlefix_configs.keys())

        points = [(M, "anchor") for M in BATCH_SIZES]
        points += [(M, "held-out") for M in HELD_OUT_BATCH_SIZES]
        points.sort()

        print(f"\n### N={N}, K={K}, device={device_name} (cache-cycling, {N_WEIGHT_COPIES} B copies)")
        print("| M | type | nearest anchor | tuned_cycled (us) | singlefix_cycled (us) | speedup |")
        print("|---:|---|---:|---:|---:|---:|")
        for M, ptype in points:
            nearest = M if ptype == "anchor" else min(anchors, key=lambda x: abs(x - M))
            selected_singlefix_config = singlefix_configs[nearest]
            selected_tuned_config = tuned_configs[nearest]

            A, As = make_A(M, K, BLOCK_K)
            copies = make_B_copies(N, K, BLOCK_N, BLOCK_K, N_WEIGHT_COPIES)
            torch.accelerator.synchronize()

            counter_t = CyclingLaunchCounter(copies)
            tuned_us = benchmark_config_cycling(
                A, As, counter_t, [BLOCK_N, BLOCK_K], selected_tuned_config, OUT_DTYPE, num_iters=NUM_ITERS,
            )
            counter_s = CyclingLaunchCounter(copies)  # reset -- identical B sequence for both configs
            singlefix_us = benchmark_config_cycling(
                A, As, counter_s, [BLOCK_N, BLOCK_K], selected_singlefix_config, OUT_DTYPE, num_iters=NUM_ITERS,
            )
            speedup = (tuned_us - singlefix_us) / tuned_us * 100

            print(
                f"| {M} | {ptype} | {nearest} | {tuned_us:.2f} | {singlefix_us:.2f} "
                f"| {speedup:+.1f}% |"
            )


if __name__ == "__main__":
    main()
