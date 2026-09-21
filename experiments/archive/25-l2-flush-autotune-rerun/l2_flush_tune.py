#!/usr/bin/env python3
"""L2-flush rerun of vLLM's real W8A8 block-FP8 autotuning search
(experiment 24).

`17` fixed the real tuning script's blind spot (`tune()` reuses one fixed
`B` across all 1280 candidate configs, so a config's true cache-locality
cost is invisible to it) by cycling `B` through 64 independently-random
copies, one per real transformer layer. That works, but `N_WEIGHT_COPIES=64`
is arbitrary from a general standpoint -- it's sized to this model's layer
count, not to the actual physical constraint (the working set needs to
exceed the GPU's L2 capacity). This script targets that constraint
directly instead: flush L2 once, right before every *timed* kernel launch,
using a buffer sized from the device's own reported L2 capacity. No
per-model tuning, and only ONE `A`/`B` pair is needed -- not 64 -- since
the flush guarantees eviction regardless of how many distinct tensors
exist.

Everything else -- search space (`get_configs_compute_bound()`, verbatim),
per-candidate benchmarking methodology (5 warmup + 10 timed CUDA-event-
bracketed iterations, including the real script's own documented /10 timing
bug -- kept for exact methodology parity, a constant factor that cancels in
ranking), `OutOfResources` handling -- is copied unchanged from the real
script, same as `17`.

Scope: `gate_up_proj` (N=17408, K=5120) only, at M=1024 and M=2048 -- same
two points `17`'s stage-1 pilot used, for direct comparison. The full
5-shape x 18-M rerun is `trainjob-full.yaml`, not this standalone script.
"""

import json

import torch
from tqdm import tqdm

from _vendored_matmul_timing import w8a8_block_matmul
from vllm.triton_utils import triton
from vllm.utils.platform_utils import get_device_name_as_file_name

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
N, K = 17408, 5120  # gate_up_proj
M_VALUES = [1024, 2048]

# Generous, documented fallback in case this PyTorch build's device-
# properties binding doesn't expose an L2-size field at all (attribute name
# has been observed to vary across versions -- confirmed present as
# `L2_cache_size`, underscored, not `L2CacheSize`, on 2.13.0+cu130) --
# comfortably exceeds every current data-center GPU's L2 (H100/H200 ~50MB,
# L40S ~96MB, B200 ~126MB known as of this writing).
FALLBACK_L2_BYTES = 128 * 1024 * 1024


def resolve_l2_bytes():
    device = torch.cuda.current_device()
    try:
        l2 = torch.cuda.get_device_properties(device).L2_cache_size
        if l2 and l2 > 0:
            print(f"  resolved L2_cache_size from device properties: {l2/1e6:.1f}MB", flush=True)
            return l2
    except AttributeError:
        pass
    print(f"  WARNING: L2_cache_size not exposed by this PyTorch build, "
          f"falling back to {FALLBACK_L2_BYTES/1e6:.0f}MB", flush=True)
    return FALLBACK_L2_BYTES


def get_configs_compute_bound():
    # copied verbatim from benchmarks/kernels/benchmark_w8a8_block_fp8.py
    configs = []
    for num_stages in [2, 3, 4, 5]:
        for block_m in [16, 32, 64, 128, 256]:
            for block_k in [64, 128]:
                for block_n in [32, 64, 128, 256]:
                    for num_warps in [4, 8]:
                        for group_size in [1, 16, 32, 64]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                }
                            )
    return configs


def make_A(m, k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    factor = 1e-2
    A_fp32 = (torch.rand(m, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    As = torch.rand(m, k_tiles, dtype=torch.float32, device="cuda") * factor
    return A, As


def make_B(n, k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    factor = 1e-2
    n_tiles = (n + BLOCK_N - 1) // BLOCK_N
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    B_fp32 = (torch.rand(n, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * factor
    return B, Bs


def benchmark_config_flush(A, B, As, Bs, flush_buf, block_size, config, out_dtype, num_iters=10):
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
    # /10 matches the real script's own documented timing bug -- kept for
    # exact methodology parity; a constant factor, cancels in ranking.
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune_l2_flush(M, N, K, block_size, out_dtype, search_space, flush_buf):
    A, As = make_A(M, K)
    B, Bs = make_B(N, K)  # ONE pair -- the flush makes extra copies unnecessary
    torch.accelerator.synchronize()

    best_config = None
    best_time = float("inf")
    n_ok, n_skipped = 0, 0
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config_flush(
                A, B, As, Bs, flush_buf, block_size, config, out_dtype, num_iters=10
            )
        except triton.runtime.autotuner.OutOfResources:
            n_skipped += 1
            continue
        n_ok += 1
        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    assert best_config is not None
    print(f"  {n_ok} configs evaluated, {n_skipped} skipped (OutOfResources)", flush=True)
    return best_config, best_time


def main():
    device_name = get_device_name_as_file_name()
    json_path = (
        f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
        f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )
    with open(json_path) as f:
        original_tuned = {int(k): v for k, v in json.load(f).items()}

    l2_bytes = resolve_l2_bytes()
    flush_buf = torch.empty(int(l2_bytes * 1.5) // 4, dtype=torch.int32, device="cuda")

    search_space = get_configs_compute_bound()
    print(f"search space size: {len(search_space)}", flush=True)

    results = {}
    for M in M_VALUES:
        print(f"\n=== tuning M={M} (L2-flush, single A/B pair) ===", flush=True)
        best_config, best_time = tune_l2_flush(
            M, N, K, [BLOCK_N, BLOCK_K], OUT_DTYPE, search_space, flush_buf
        )
        results[M] = {
            "l2_flush_best_config": best_config,
            "l2_flush_best_time_us_buggy_div10": best_time,
            "original_always_warm_config": original_tuned[M],
        }
        print(f"M={M}:", flush=True)
        print(f"  L2-flush winner      : {best_config}  (avg={best_time:.2f}us)", flush=True)
        print(f"  original tuner picked: {original_tuned[M]}", flush=True)
        same_group = best_config["GROUP_SIZE_M"] == original_tuned[M]["GROUP_SIZE_M"]
        print(f"  GROUP_SIZE_M {'MATCHES' if same_group else 'DIFFERS'} "
              f"({best_config['GROUP_SIZE_M']} vs {original_tuned[M]['GROUP_SIZE_M']})",
              flush=True)

    with open("l2_flush_tune_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote l2_flush_tune_results.json", flush=True)


if __name__ == "__main__":
    main()
