#!/usr/bin/env python3
"""GROUP_SIZE_M surgical reproduction (experiment 14).

`12`'s cold-weight-cycling test showed tuned's DRAM Read nearly doubles
(12%->22%) under cold-cycling while default stays flat (~9.2%) -- real,
correctly-directioned, but well short of real serving's 72.8%. Diffing the
two configs directly (rather than chasing another external-contention
hypothesis) turns up a candidate that's been sitting in the config file the
whole time: `GROUP_SIZE_M`, Triton's L2-cache-locality grid-swizzle knob
(see `vllm/model_executor/layers/quantization/utils/fp8_utils.py:772-780` --
the exact `num_pid_in_group`/`group_id`/`pid_m`/`pid_n` formula from
Triton's own matmul tutorial). default uses `GROUP_SIZE_M=32`, which at its
own `BLOCK_SIZE_M=64` and M=2048 gives `num_pid_m=32` -- GROUP_SIZE_M exactly
equals num_pid_m, meaning ONE group covers all of M: only a single ~0.64MB
slice of `B` needs to be resident at a time, reused by up to 32 concurrently-
scheduled blocks, before the grid moves to the next slice. tuned uses
`GROUP_SIZE_M=1` -- grouping disabled entirely, plain row-major order, which
spreads concurrently-active blocks across many different B-tiles at once.
Checking the full M-grid in the tuned JSON: every M-bucket at and above 512
for this exact shape (gate_up_proj, N=17408,K=5120) landed on
`GROUP_SIZE_M=1` -- consistent with an autotuner that always benchmarks with
the same warm, repeated B (so grouping has zero measurable effect in ITS
environment) picking the degenerate value by tie-breaking noise, blind to
the real cost that only appears once B is genuinely contended.

This experiment tests that directly and surgically: same cold-weight-cycling
design as `12` v2, but instead of only default vs. tuned, adds a THIRD
config -- tuned's exact M=2048 config with ONLY `GROUP_SIZE_M` changed from
1 to 16 (tuned's own num_pid_m at BLOCK_SIZE_M=128, M=2048 -- the analogous
"one group covers all of M" value, mirroring default's own strategy).
Everything else (BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, num_warps=8,
num_stages=3 -- all of tuned's other, genuinely-faster parameters) stays
identical. If GROUP_SIZE_M is really the mechanism, tuned-groupfix should
recover something close to default's flat DRAM-Read behavior under cold-
cycling, while keeping tuned's fast same-B duration.

3 configs x 2 weight-states (same/cold) = 6 conditions, round-robin
counterbalanced in 250-iteration blocks x 16 repeats, exactly matching `12`
v2's methodology (same clock-drift control).
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
N_REPEATS = 16  # 16 * 250 = 4000 iterations/condition, matching 12 v2

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}

# tuned's own num_pid_m at BLOCK_SIZE_M=128, M=2048: ceil(2048/128) = 16 --
# the value that makes GROUP_SIZE_M cover all of M in one group, mirroring
# default's own strategy (default's GROUP_SIZE_M=32 == its own num_pid_m=32).
TUNED_GROUPFIX_GROUP_SIZE_M = 16


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
    (not reset per block) -- see 12 v2 for why."""
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
    tuned_groupfix_config = dict(tuned_config)
    tuned_groupfix_config["GROUP_SIZE_M"] = TUNED_GROUPFIX_GROUP_SIZE_M
    print(f"tuned_config (original)  = {tuned_config}", flush=True)
    print(f"tuned_config (groupfix)  = {tuned_groupfix_config}", flush=True)
    print(f"default_config           = {DEFAULT_CONFIG}", flush=True)

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
        ("groupfix", "same"): (tuned_groupfix_config, [B_copies[0]]),
        ("groupfix", "cold"): (tuned_groupfix_config, B_copies),
    }
    order = list(conditions.keys())

    all_latencies = {c: [] for c in conditions}
    per_block_means = {c: [] for c in conditions}
    global_offset = {c: 0 for c in conditions}

    print("\n=== warmup ===", flush=True)
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
    for c in order:
        blocks = per_block_means[c]
        half = len(blocks) // 2
        first_half = sum(blocks[:half]) / half
        second_half = sum(blocks[half:]) / (len(blocks) - half)
        drift_pct = (second_half - first_half) / first_half * 100
        print(f"  {c[0]:>8}-{c[1]:<4}: first-half={first_half:.1f}us  "
              f"second-half={second_half:.1f}us  drift={drift_pct:+.2f}%")

    print("\n=== Summary: same-B -> cold-cycling shift, per config ===")
    for role in ["default", "tuned", "groupfix"]:
        same_m = means[(role, "same")]
        cold_m = means[(role, "cold")]
        print(f"  {role:>8}: {same_m:.1f}us -> {cold_m:.1f}us  "
              f"({(cold_m / same_m - 1) * 100:+.2f}%)")

    print("\nFor reference: 12's cold-weight-cycling result (duration): "
          "default 1370.0->1369.3us (-0.05%), tuned 1224.0->1288.4us (+5.26%).")
    print("12's matched DRAM Read: default 9.22%->9.19% (flat), "
          "tuned 11.96%->22.46% (nearly 2x).")
    print("If GROUP_SIZE_M is the mechanism: groupfix's cold-cycling shift "
          "should look much more like default's (small/flat) than tuned's "
          "(clearly rising), while groupfix's same-B duration should stay "
          "close to tuned's (keeping the other parameters' speed benefit).")


if __name__ == "__main__":
    main()
