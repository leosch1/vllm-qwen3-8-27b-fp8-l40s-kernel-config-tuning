"""
Isolated kernel benchmark, L2-FLUSH variant: default config vs. this
repo's committed tuned-configs/ (the output of the L2-flush-patched
tuner on leosch1/vllm@qwen3-8-27b-fp8-dense-tuning -- see
experiments/04-l2-flush-retune/), both timed under the SAME mechanism
that actually produced the winning config -- a direct L2 flush before
every timed launch. Using the same mechanism for tuning and validation
means this script asks the most literal possible question: under the
exact condition the tuner optimized for, does the config it picked
actually win?

Only one B/Bs pair is needed per point (unlike cache-cycling approaches
that need many copies to force eviction) -- the flush before every timed
launch guarantees a cold cache regardless of what ran immediately before.

Same 175-point grid as experiments/01-first-measurement/'s script (5
shapes x 18 anchor + 17 held-out M values), same NUM_ITERS=2000, same
numerical sanity check, same documented /10 timing-bug scaling for
parity.
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


def resolve_l2_bytes():
    device = torch.cuda.current_device()
    try:
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        if l2 and l2 > 0:
            print(f"resolved L2_cache_size from device properties: {l2/1e6:.1f}MB", flush=True)
            return l2
    except AttributeError:
        pass
    fallback = 128 * 1024 * 1024  # generous fallback if this PyTorch build doesn't expose L2_cache_size
    print(f"WARNING: L2_cache_size not exposed by this PyTorch build, "
          f"falling back to {fallback/1e6:.0f}MB", flush=True)
    return fallback


def benchmark_config_flush(A, B, As, Bs, flush_buf, block_size, config, out_dtype, num_iters=2000):
    def run():
        w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)

    torch.accelerator.synchronize()
    for _ in range(5):
        run()  # warmup: JIT compile only, doesn't need a cold cache
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    latencies = []
    for _ in range(num_iters):
        flush_buf.zero_()  # force eviction of A/B/C from L2 before this launch
        torch.accelerator.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    return sum(latencies) / (num_iters * 10) * 1000  # /10: same documented bug, kept for parity


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
L2FLUSH_DIR = "./tuned-configs"
NUM_ITERS = 2000


def make_A(M, K, block_k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    k_tiles = (K + block_k - 1) // block_k
    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    return A, As


def make_B(N, K, block_n, block_k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    return B, Bs


def main():
    torch.cuda.init()
    device_name = get_device_name_as_file_name()

    l2_bytes = resolve_l2_bytes()
    flush_buf = torch.empty(int(l2_bytes * 1.5) // 4, dtype=torch.int32, device="cuda")

    for N, K in SHAPES:
        json_path = (
            f"{L2FLUSH_DIR}/N={N},K={K},device_name={device_name},"
            f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
        )
        with open(json_path) as f:
            l2flush_configs = {int(k): v for k, v in json.load(f).items()}
        anchors = list(l2flush_configs.keys())

        points = [(M, "anchor") for M in BATCH_SIZES]
        points += [(M, "held-out") for M in HELD_OUT_BATCH_SIZES]
        points.sort()

        print(f"\n### N={N}, K={K}, device={device_name} (L2-flush pressure, direct flush per launch)")
        print("| M | type | nearest anchor | default_flushed (us) | l2flush_flushed (us) | speedup |")
        print("|---:|---|---:|---:|---:|---:|")
        for M, ptype in points:
            nearest = M if ptype == "anchor" else min(anchors, key=lambda x: abs(x - M))
            selected_config = l2flush_configs[nearest]

            A, As = make_A(M, K, BLOCK_K)
            B, Bs = make_B(N, K, BLOCK_N, BLOCK_K)  # ONE pair -- flush makes copies unnecessary
            torch.accelerator.synchronize()

            default_us = benchmark_config_flush(
                A, B, As, Bs, flush_buf, [BLOCK_N, BLOCK_K], DEFAULT_CONFIG, OUT_DTYPE, num_iters=NUM_ITERS,
            )
            l2flush_us = benchmark_config_flush(
                A, B, As, Bs, flush_buf, [BLOCK_N, BLOCK_K], selected_config, OUT_DTYPE, num_iters=NUM_ITERS,
            )
            speedup = (default_us - l2flush_us) / default_us * 100

            print(
                f"| {M} | {ptype} | {nearest} | {default_us:.2f} | {l2flush_us:.2f} "
                f"| {speedup:+.1f}% |"
            )


if __name__ == "__main__":
    main()
