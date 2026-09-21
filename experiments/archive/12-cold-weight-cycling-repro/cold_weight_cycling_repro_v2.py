#!/usr/bin/env python3
"""Cold-weight-cycling reproduction, v2 -- fixes three gaps found in the v1
run (see this experiment's README):

1. No order control: v1 ran default-same, tuned-same, default-cold,
   tuned-cold as one long block each, in that fixed sequence -- any
   monotonic clock-ramp/warm-up drift over the run would bias later
   segments toward looking faster, independent of the real manipulation.
   Fixed by splitting each condition's iteration budget into many small
   blocks and round-robining through all 4 conditions repeatedly, so any
   drift affects all 4 roughly equally instead of favoring later segments.
2. No per-iteration data retained: v1 only kept the final average, so a
   within-run trend (clock still ramping) couldn't be checked after the
   fact without a full rerun. Fixed by recording every iteration's latency,
   tagged by (condition, repeat_index), and printing a per-repeat-block
   breakdown so a trend is directly visible in the output.
3. v1's capture had no GPU_METRICS table at all -- best guess is total
   capture duration was too short (v1 processed 229,231 events vs. 09's
   successful capture's 1,001,228). Fixed by roughly doubling the total
   iteration budget (3000/condition instead of 2000) to give the
   GPU-metrics stream more margin to actually flush.
"""
import json
import torch

from _vendored_matmul_timing import w8a8_block_matmul
from vllm.utils.platform_utils import get_device_name_as_file_name

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
M = 2048
N, K = 17408, 5120  # gate_up_proj
NUM_WEIGHT_COPIES = 64  # matches the real model's actual layer count

ITERS_PER_BLOCK = 250
N_REPEATS = 16  # 16 * 250 = 4000 iterations/condition, 2x v1's total budget --
# extra margin in case the missing GPU_METRICS table in v1 was a
# duration/buffer-flush threshold issue, not just noise

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


def run_block(A, As, B_copies, config, n_iters, global_iter_offset):
    """Times n_iters calls, cycling B_copies by a GLOBAL iteration counter
    (not reset per block) so the same-B condition (len(B_copies)==1) and
    the cold condition (len(B_copies)==64) both index consistently across
    repeats -- avoids accidentally always starting the cold condition's
    each block back at B_copies[0]."""
    def run(i):
        B, Bs = B_copies[i % len(B_copies)]
        return w8a8_block_matmul(A, B, As, Bs, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    latencies = []
    for j in range(n_iters):
        i = global_iter_offset + j
        torch.accelerator.synchronize()
        start_event.record()
        run(i)
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event) * 1000)  # ms -> us
    return latencies


def main():
    device_name = get_device_name_as_file_name()
    print(f"device_name={device_name}", flush=True)

    json_path = (
        f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
        f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )
    with open(json_path) as f:
        tuned_configs = {int(k): v for k, v in json.load(f).items()}
    tuned_config = tuned_configs[M]
    print(f"tuned_config={tuned_config}", flush=True)

    print(f"Allocating A (fixed) + {NUM_WEIGHT_COPIES} distinct B copies "
          f"for gate_up_proj (N={N},K={K}), M={M}...", flush=True)
    A, As = make_A(M, K)
    B_copies = make_B_copies(NUM_WEIGHT_COPIES, N, K)
    total_mb = NUM_WEIGHT_COPIES * N * K / 1e6
    print(f"Total weight-copy footprint: {total_mb:.1f} MB "
          f"(single copy: {N * K / 1e6:.1f} MB, L40S L2 cache: 96 MB)", flush=True)

    conditions = {
        ("default", "same"): (DEFAULT_CONFIG, [B_copies[0]]),
        ("default", "cold"): (DEFAULT_CONFIG, B_copies),
        ("tuned", "same"): (tuned_config, [B_copies[0]]),
        ("tuned", "cold"): (tuned_config, B_copies),
    }
    # fixed round-robin order, repeated N_REPEATS times -- spreads each
    # condition's budget evenly across the whole run's timeline instead of
    # clustering it at one point, so a monotonic drift affects all 4
    # conditions roughly equally rather than favoring whichever ran later.
    order = list(conditions.keys())

    all_latencies = {c: [] for c in conditions}
    per_block_means = {c: [] for c in conditions}
    global_offset = {c: 0 for c in conditions}

    # warmup: touch every condition once before timing anything
    for c, (config, copies) in conditions.items():
        run_block(A, As, copies, config, 5, 0)
    torch.accelerator.synchronize()

    print(f"\n=== Running {N_REPEATS} repeats x {ITERS_PER_BLOCK} iters/block, "
          f"round-robin order {order} ===", flush=True)
    for r in range(N_REPEATS):
        for c in order:
            config, copies = conditions[c]
            lat = run_block(A, As, copies, config, ITERS_PER_BLOCK, global_offset[c])
            global_offset[c] += ITERS_PER_BLOCK
            all_latencies[c].extend(lat)
            block_mean = sum(lat) / len(lat)
            per_block_means[c].append(block_mean)
        print(f"  repeat {r+1}/{N_REPEATS}: " +
              "  ".join(f"{c[0]}-{c[1]}={per_block_means[c][-1]:.1f}us" for c in order),
              flush=True)

    print("\n=== Per-condition overall mean (all iterations) ===")
    means = {}
    for c in order:
        m = sum(all_latencies[c]) / len(all_latencies[c])
        means[c] = m
        print(f"  {c[0]:>8}-{c[1]:<4}: {m:.1f}us  (n={len(all_latencies[c])})")

    print("\n=== Trend check: first-half vs second-half block means, per condition ===")
    print("(if these differ substantially, a global drift is real and biases the comparison)")
    for c in order:
        blocks = per_block_means[c]
        half = len(blocks) // 2
        first_half = sum(blocks[:half]) / half
        second_half = sum(blocks[half:]) / (len(blocks) - half)
        drift_pct = (second_half - first_half) / first_half * 100
        print(f"  {c[0]:>8}-{c[1]:<4}: first-half={first_half:.1f}us  "
              f"second-half={second_half:.1f}us  drift={drift_pct:+.2f}%")

    print("\n=== Summary: same-B -> cold-cycling shift, per config ===")
    for role in ["default", "tuned"]:
        same_m = means[(role, "same")]
        cold_m = means[(role, "cold")]
        print(f"  {role}: {same_m:.1f}us -> {cold_m:.1f}us  "
              f"({(cold_m / same_m - 1) * 100:+.2f}%)")

    default_same, default_cold = means[("default", "same")], means[("default", "cold")]
    tuned_same, tuned_cold = means[("tuned", "same")], means[("tuned", "cold")]
    print(f"\n  isolated speedup, same-B:   {(default_same - tuned_same) / default_same * 100:+.1f}%")
    print(f"  isolated speedup, cold:     {(default_cold - tuned_cold) / default_cold * 100:+.1f}%")
    print("\nFor reference, real serving (from 05's profiling-results): "
          "default ~1181us (flat/faster than isolated), tuned ~2057us (+76%).")


if __name__ == "__main__":
    main()
