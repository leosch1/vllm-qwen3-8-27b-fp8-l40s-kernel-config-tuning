#!/usr/bin/env python3
"""Cache-aware rerun of vLLM's real W8A8 block-FP8 autotuning search
(experiment 17).

Does the real tuning script (`benchmarks/kernels/benchmark_w8a8_block_fp8.py`)
pick a different `GROUP_SIZE_M` if its one confirmed blind spot (14, 16) is
fixed? `tune()` there generates ONE `A`/`B` pair per `(M,N,K)` and reuses it,
unmodified, across all 1280 candidate configs -- so a config's L2-locality
cost is invisible to it. Prior experiments only ever patched the tuner's
*output* JSON by hand (`groupfix`, in 15) or measured the cache-eviction
effect with a standalone script (12) -- never actually fixed `tune()` itself
and reran the real search. This script does that.

**The one change from the real script**: instead of one fixed, never-cleared
`B`, this pre-allocates N_WEIGHT_COPIES=64 independently-random copies
(matching the real model's 64 transformer layers, the same count 12 used) and
selects one via a counter that increments on *every individual kernel
launch* -- warmup and timed alike -- across the *entire* search for a given
`M`, not reset per-candidate-config. This matters: 12's own `ncu`
follow-up found a single isolated cache miss shows zero measurable cost for
either config -- only continuous, rapid, sustained cycling through many
distinct tensors does. A per-candidate reset (touch a few different copies,
then go back to copy 0 for the next config) would reproduce that "single
isolated miss" pattern, not the sustained-pressure pattern that's actually
been shown to matter. A global, never-reset counter keeps the cache under
continuous pressure for the whole run, the same way 12's design did.

`A` is created once and reused throughout, unmodified from the real script
-- 12 already established activations are much smaller (10.5MB vs `B`'s
89.1MB) and contribute far less L2 pressure, so holding it fixed isolates
the one variable that matters.

Everything else -- search space (`get_configs_compute_bound()`, verbatim),
per-candidate benchmarking methodology (5 warmup + 10 timed CUDA-event-
bracketed iterations, including the real script's own documented /10 timing
bug -- kept deliberately for exact methodology parity; it's a constant
factor that cancels in the ranking this script cares about), `OutOfResources`
handling -- is copied unchanged from the real script.

Scope: `gate_up_proj` (N=17408, K=5120) only, at M=1024 and M=2048 -- the two
values with the richest existing comparison data in this project (14's
`groupfix` swap and 16's `ncu`/single-M work both used one of these). A full
18-M rerun would be the natural next step but wasn't run here; flagged as
follow-up in the README, not silently assumed to generalize.
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
N_WEIGHT_COPIES = 64  # matches the real model's layer count, matches 12's design


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


class CyclingLaunchCounter:
    """Global, never-reset counter -- see module docstring for why this must
    not be reset per-candidate-config."""

    def __init__(self, copies):
        self.copies = copies
        self.i = 0

    def next(self):
        B, Bs = self.copies[self.i % len(self.copies)]
        self.i += 1
        return B, Bs


def benchmark_config_cycling(A, As, counter, block_size, config, out_dtype, num_iters=10):
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
    # /10 matches the real script's own documented timing bug -- kept for
    # exact methodology parity; a constant factor, cancels in ranking.
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune_cache_aware(M, N, K, block_size, out_dtype, search_space):
    A, As = make_A(M, K)
    print(f"  allocating {N_WEIGHT_COPIES} distinct B copies "
          f"({N_WEIGHT_COPIES * N * K / 1e9:.2f}GB total)...", flush=True)
    copies = [make_B(N, K) for _ in range(N_WEIGHT_COPIES)]
    torch.accelerator.synchronize()
    counter = CyclingLaunchCounter(copies)

    best_config = None
    best_time = float("inf")
    n_ok, n_skipped = 0, 0
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config_cycling(
                A, As, counter, block_size, config, out_dtype, num_iters=10
            )
        except triton.runtime.autotuner.OutOfResources:
            n_skipped += 1
            continue
        n_ok += 1
        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    assert best_config is not None
    print(f"  {n_ok} configs evaluated, {n_skipped} skipped (OutOfResources), "
          f"{counter.i} total kernel launches", flush=True)
    return best_config, best_time


def main():
    device_name = get_device_name_as_file_name()
    json_path = (
        f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
        f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )
    with open(json_path) as f:
        original_tuned = {int(k): v for k, v in json.load(f).items()}

    search_space = get_configs_compute_bound()
    print(f"search space size: {len(search_space)}", flush=True)

    results = {}
    for M in M_VALUES:
        print(f"\n=== tuning M={M} (cache-aware, {N_WEIGHT_COPIES}-copy cycling) ===",
              flush=True)
        best_config, best_time = tune_cache_aware(
            M, N, K, [BLOCK_N, BLOCK_K], OUT_DTYPE, search_space
        )
        results[M] = {
            "cache_aware_best_config": best_config,
            "cache_aware_best_time_us_buggy_div10": best_time,
            "original_always_warm_config": original_tuned[M],
        }
        print(f"M={M}:", flush=True)
        print(f"  cache-aware winner : {best_config}  (avg={best_time:.2f}us)",
              flush=True)
        print(f"  original tuner picked: {original_tuned[M]}", flush=True)
        same_group = best_config["GROUP_SIZE_M"] == original_tuned[M]["GROUP_SIZE_M"]
        print(f"  GROUP_SIZE_M {'MATCHES' if same_group else 'DIFFERS'} "
              f"({best_config['GROUP_SIZE_M']} vs {original_tuned[M]['GROUP_SIZE_M']})",
              flush=True)

    with open("cache_aware_tune_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nwrote cache_aware_tune_results.json", flush=True)


if __name__ == "__main__":
    main()
