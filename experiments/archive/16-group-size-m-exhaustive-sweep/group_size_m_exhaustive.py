#!/usr/bin/env python3
"""GROUP_SIZE_M exhaustive isolated-kernel comparison (experiment 16).

Directly settles the open question from `14`: is the autotuner's
GROUP_SIZE_M choice a real, reproducible signal, or is it noise-dominated
tie-breaking? For every M in gate_up_proj's tuned-config grid (N=17408,
K=5120), takes that M's ACTUAL winning config (BLOCK_SIZE_M, BLOCK_SIZE_N,
BLOCK_SIZE_K, num_warps, num_stages exactly as tuned) and varies ONLY
GROUP_SIZE_M across the tuner's own real candidate set {1, 16, 32, 64} --
same isolated, same-tensor-reused-across-conditions methodology the real
tuning script uses (see `benchmarks/kernels/benchmark_w8a8_block_fp8.py`).

Unlike the tuner's own 10-iteration, no-repeat, mean-only comparison, this
uses round-robin counterbalanced repeats (matching this project's own
established methodology from 12/14) specifically so we can compute a
per-condition standard error and actually tell whether the "winning" value
is statistically distinguishable from the alternatives, or within noise.
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

M_VALUES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
GROUP_CANDIDATES = [1, 16, 32, 64]  # the tuner's own actual search-space values

ITERS_PER_BLOCK = 50
N_REPEATS = 16  # 50*16 = 800 iterations/condition


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
    print(f"device_name={device_name}", flush=True)

    json_path = (
        f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
        f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )
    with open(json_path) as f:
        tuned_configs = {int(k): v for k, v in json.load(f).items()}

    conditions = []  # (M, group, config)
    for m in M_VALUES:
        base_config = dict(tuned_configs[m])
        actual_group = base_config["GROUP_SIZE_M"]
        for g in GROUP_CANDIDATES:
            cfg = dict(base_config)
            cfg["GROUP_SIZE_M"] = g
            conditions.append((m, g, cfg, actual_group))

    print(f"=== {len(M_VALUES)} M values x {len(GROUP_CANDIDATES)} GROUP_SIZE_M candidates "
          f"= {len(conditions)} conditions ===", flush=True)

    # allocate A/B once per M -- reused across all 4 GROUP_SIZE_M candidates for
    # that M, matching the real tuning script's own methodology exactly (same
    # tensor pair reused across the whole config sweep for a given M).
    tensors = {}
    for m in M_VALUES:
        tensors[m] = (*make_A(m, K), *make_B(N, K))
    torch.accelerator.synchronize()
    print("tensors allocated", flush=True)

    print("\n=== warmup ===", flush=True)
    for m, g, cfg, _ in conditions:
        A, As, B, Bs = tensors[m]
        run_block(A, As, B, Bs, cfg, 3)
    torch.accelerator.synchronize()

    per_block_means = {(m, g): [] for m, g, _, _ in conditions}
    actual_choice = {}

    print(f"\n=== {N_REPEATS} repeats x {ITERS_PER_BLOCK} iters/block, round-robin ===", flush=True)
    for r in range(N_REPEATS):
        for m, g, cfg, actual_group in conditions:
            actual_choice[m] = actual_group
            A, As, B, Bs = tensors[m]
            lat = run_block(A, As, B, Bs, cfg, ITERS_PER_BLOCK)
            per_block_means[(m, g)].append(sum(lat) / len(lat))
        print(f"  repeat {r + 1}/{N_REPEATS} done", flush=True)

    print("\n=== Results: mean +/- stderr (us), per (M, GROUP_SIZE_M) ===")
    header = f"{'M':>6} {'tuner picked':>13}"
    for g in GROUP_CANDIDATES:
        header += f"  GM={g:<3}mean+-SE      "
    header += "  test-winner  matches-tuner?"
    print(header)

    agree, disagree, within_noise = 0, 0, 0
    for m in M_VALUES:
        row = f"{m:>6} {actual_choice[m]:>13}"
        means, stderrs = {}, {}
        for g in GROUP_CANDIDATES:
            blocks = per_block_means[(m, g)]
            mean = statistics.mean(blocks)
            stderr = statistics.stdev(blocks) / (len(blocks) ** 0.5) if len(blocks) > 1 else 0.0
            means[g] = mean
            stderrs[g] = stderr
            row += f"  {mean:8.2f}+-{stderr:5.2f}us"
        winner = min(means, key=means.get)
        # is the winner's mean outside the tuner-picked value's mean +/- 2*SE (95%-ish)?
        tuner_g = actual_choice[m]
        tuner_mean, tuner_se = means[tuner_g], stderrs[tuner_g]
        winner_mean, winner_se = means[winner], stderrs[winner]
        gap = tuner_mean - winner_mean
        combined_se = (tuner_se ** 2 + winner_se ** 2) ** 0.5
        distinguishable = combined_se > 0 and abs(gap) > 2 * combined_se
        if winner == tuner_g:
            verdict = "MATCH"
            agree += 1
        elif not distinguishable:
            verdict = "tie (within noise)"
            within_noise += 1
        else:
            verdict = f"MISMATCH (real gap: {winner} beats tuner's {tuner_g} by {gap/tuner_mean*100:.1f}%)"
            disagree += 1
        row += f"  winner=GM={winner}  {verdict}"
        print(row)

    print(f"\n=== Summary across {len(M_VALUES)} M values ===")
    print(f"  Tuner's pick was the fastest in our test: {agree}/{len(M_VALUES)}")
    print(f"  Tuner's pick tied with the fastest (within 2*SE noise band): {within_noise}/{len(M_VALUES)}")
    print(f"  Tuner's pick was measurably beaten by another candidate: {disagree}/{len(M_VALUES)}")


if __name__ == "__main__":
    main()
