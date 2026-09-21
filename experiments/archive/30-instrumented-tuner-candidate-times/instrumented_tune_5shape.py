#!/usr/bin/env python3
"""EXP29C: same instrumentation as exp29, but exp29b's STRUCTURE.

exp29  = instrumented script, gate_up_proj only, M in {1024,2048}, 3 repeats
         each -> the same shape searched back-to-back. Result: GROUP_SIZE_M=1
         won 6/6.
exp29b = UNMODIFIED script, M=1024, all 5 shapes per run, 3 runs -> shapes
         interleaved between repeats. Result: GROUP_SIZE_M = 64, 1, 16
         (1/3), matching experiment 27's unmodified runs (0/6) rather than
         exp29.

Two things differ between exp29 and exp29b, so neither isolates a cause:
  (a) the instrumentation itself (a JSON record written + flushed inside the
      timing loop, between candidates), and
  (b) the shape sequence -- exp29 re-searches the SAME shape back-to-back, so
      B stays warm in L2 ACROSS searches, an extra warmth cushion exp29b
      never gets because four other shapes run in between.

(b) is the mechanistically interesting one: by this project's own theory,
more inter-launch warmth should differentially favour GROUP_SIZE_M=1 (it has
the worst intra-launch locality, so it has the most to gain from leftover
warmth). If that's the cause, exp29's 6/6 is an amplified demonstration of
the thesis rather than an artifact undermining it.

This script holds the instrumentation constant and adopts exp29b's structure
exactly (M=1024, all 5 shapes per run, 3 runs). So:
  exp29c vs exp29b  -> isolates the INSTRUMENTATION (same structure, differs
                       only by the in-loop logging)
  exp29c vs exp29   -> isolates the SHAPE SEQUENCE (same instrumentation,
                       differs only by interleaving)

RAW OUTPUT: every candidate of every search is written to
candidate_times.jsonl (full config, kernel_time_us, all 10 raw per-iteration
latencies, N, K, M, repeat, candidate_index). Nothing is aggregated away.
"""

import json
import os
import time
from datetime import datetime

import torch
from tqdm import tqdm

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

assert current_platform.is_cuda() or current_platform.is_rocm()

# Same 5 shapes, same order, as the real script's get_weight_shapes().
SHAPES = [(17408, 5120), (8192, 5120), (7168, 5120), (5120, 8704), (5120, 3072)]
BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
M_VALUES = [1024]
N_REPEATS = 3
OUT_DIR = "/repo/exp29c-results"
JSONL_PATH = os.path.join(OUT_DIR, "candidate_times.jsonl")


def w8a8_block_matmul(A, B, As, Bs, block_size, config, output_dtype=torch.float16):
    block_n, block_k = block_size[0], block_size[1]
    M = A.numel() // A.shape[-1]
    N_, K_ = B.shape
    C = A.new_empty(A.shape[:-1] + (N_,), dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_, META["BLOCK_SIZE_N"]),
        )

    kernel = _w8a8_triton_block_scaled_mm
    kernel[grid](
        A, B, C, As, Bs, M, N_, K_, block_n, block_k,
        A.stride(-2), A.stride(-1),
        B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1),
        As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        **config,
    )
    return C


def get_configs_compute_bound():
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


def benchmark_config(A, B, As, Bs, block_size, config, out_dtype=torch.float16, num_iters=10):
    """Verbatim from the real script, including the documented /10 bug, plus
    the raw per-iteration latencies the real script discards."""
    def run():
        w8a8_block_matmul(A, B, As, Bs, block_size, config, out_dtype)

    torch.accelerator.synchronize()
    for _ in range(5):
        run()
    torch.accelerator.synchronize()

    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)

    latencies: list[float] = []
    for _ in range(num_iters):
        torch.accelerator.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg, latencies


def make_tensors(M, N, K):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    factor_for_scale = 1e-2

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    n_tiles = (N + BLOCK_N - 1) // BLOCK_N
    k_tiles = (K + BLOCK_K - 1) // BLOCK_K
    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * factor_for_scale
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * factor_for_scale
    return A, B, As, Bs


def tune_instrumented(M, N, K, repeat, search_space, jsonl_fh):
    A, B, As, Bs = make_tensors(M, N, K)
    torch.accelerator.synchronize()

    best_config, best_time = None, float("inf")
    n_ok = n_skipped = 0

    for idx, config in enumerate(tqdm(search_space, desc=f"N={N} M={M} r{repeat}")):
        try:
            kernel_time, raw_latencies = benchmark_config(
                A, B, As, Bs, [BLOCK_N, BLOCK_K], config, OUT_DTYPE, num_iters=10
            )
        except triton.runtime.autotuner.OutOfResources:
            n_skipped += 1
            jsonl_fh.write(json.dumps({
                "M": M, "N": N, "K": K, "repeat": repeat, "candidate_index": idx,
                "config": config, "kernel_time_us": None,
                "raw_latencies_ms": None, "status": "OutOfResources",
            }) + "\n")
            continue

        n_ok += 1
        jsonl_fh.write(json.dumps({
            "M": M, "N": N, "K": K, "repeat": repeat, "candidate_index": idx,
            "config": config, "kernel_time_us": kernel_time,
            "raw_latencies_ms": raw_latencies, "status": "ok",
        }) + "\n")
        jsonl_fh.flush()

        if kernel_time < best_time:   # real script's strict `<`
            best_time = kernel_time
            best_config = config

    print(f"  N={N},K={K} M={M} r{repeat}: {n_ok} evaluated, {n_skipped} skipped", flush=True)
    print(f"  winner: {best_config}  ({best_time:.3f}us)", flush=True)
    return best_config, best_time


def main():
    torch.cuda.init()
    os.makedirs(OUT_DIR, exist_ok=True)
    search_space = get_configs_compute_bound()
    search_space = [c for c in search_space if BLOCK_K % c["BLOCK_SIZE_K"] == 0]
    print(f"search space size: {len(search_space)}", flush=True)
    print(f"writing raw per-candidate data to {JSONL_PATH}", flush=True)

    winners = []
    t0 = time.time()
    with open(JSONL_PATH, "a") as jsonl_fh:
        for repeat in range(1, N_REPEATS + 1):
            for (Nsh, Ksh) in SHAPES:          # all 5 shapes per run, like the real script
                for M in M_VALUES:
                    print(f"\n=== run {repeat}/{N_REPEATS}  N={Nsh},K={Ksh}  M={M} "
                          f"({datetime.now().ctime()}) ===", flush=True)
                    cfg, t = tune_instrumented(M, Nsh, Ksh, repeat, search_space, jsonl_fh)
                    winners.append({"M": M, "N": Nsh, "K": Ksh, "repeat": repeat,
                                    "winner": cfg, "winner_time_us": t})
                    with open(os.path.join(OUT_DIR, "winners.json"), "w") as f:
                        json.dump(winners, f, indent=2)

    print(f"\ntotal time: {time.time() - t0:.1f}s", flush=True)
    print("\n=== WINNERS (all shapes) ===")
    for w in winners:
        print(f"run{w['repeat']} N={w['N']},K={w['K']} M={w['M']}: "
              f"GROUP_SIZE_M={w['winner']['GROUP_SIZE_M']}  {w['winner']}")

    print("\n=== gate_up_proj GROUP_SIZE_M head-to-head, holding each run's "
          "winning other-params fixed (derived from candidate_times.jsonl) ===")
    rows = [json.loads(l) for l in open(JSONL_PATH) if l.strip()]
    for w in [x for x in winners if x["N"] == 17408]:
        base = {k: v for k, v in w["winner"].items() if k != "GROUP_SIZE_M"}
        matches = {}
        for r in rows:
            if (r["M"] != w["M"] or r["repeat"] != w["repeat"]
                    or r.get("N") != w["N"] or r["kernel_time_us"] is None):
                continue
            if all(r["config"].get(k) == v for k, v in base.items()):
                matches[r["config"]["GROUP_SIZE_M"]] = r["kernel_time_us"]
        if matches:
            fastest = min(matches, key=matches.get)
            print(f"run{w['repeat']}  base={base}")
            print("    times: " + "  ".join(f"GSM={g}:{matches[g]*10:8.1f}us"
                                            for g in sorted(matches)))
            print(f"    fastest={fastest}")


if __name__ == "__main__":
    main()
