#!/usr/bin/env python3
"""Does row-major's eviction knee track B's size? (+ reconcile exp32 vs exp16's ncu)

BACKGROUND
----------
Experiment 32 swept how many MiB of L2 are evicted between timed launches and
found a sharp asymmetry at gate_up_proj M=1024: GROUP_SIZE_M=1 degrades +52%
from warm to fully-evicted, while grouped orderings are flat (+1%). Its knee
sat at 2-8 MiB of eviction, matching the predicted slack of
96 - (B 85 + one A row-block 0.6 + one C row-block 4.25) = ~6 MiB.

That is only INDIRECT attribution: the knee position implies ~90MiB is being
retained, which can only be B, but nothing measured B's residency directly.

THIS EXPERIMENT -- two parts, both counter-free.

PART A: reconcile a contradiction now sitting in the record.
  At M=1024 fully cold, exp32 measured GROUP_SIZE_M=1 at 826us vs grouped
  570us (gap 256us). Experiment 16's ncu probe measured 581us vs 520us (gap
  61us). Grouped roughly agrees; row-major does not. The likeliest cause is
  trivial: ncu profiled the tuner's actual M=1024 winner, which has
  num_stages=4, whereas exp32 used num_stages=3. So this runs the sweep at
  BOTH num_stages values. If the gap closes at num_stages=4, the
  contradiction is explained and both datasets stand.

PART B: attribution by prediction.
  If B is the tensor whose retention matters, row-major's knee must move with
  B's size. Slack = L2 - (B + A_row_block + C_row_block), and both B and the
  C row-block shrink with N, so the predicted knee for K=5120, M=1024,
  BLOCK_SIZE_M/N=128:

    N=17408  B=85.0MiB  C_rowblk=4.25MiB  ->  predicted knee ~ 6.1MiB
    N=12288  B=60.0MiB  C_rowblk=3.00MiB  ->  predicted knee ~32.4MiB
    N= 8192  B=40.0MiB  C_rowblk=2.00MiB  ->  predicted knee ~53.4MiB
    N= 4096  B=20.0MiB  C_rowblk=1.00MiB  ->  predicted knee ~74.4MiB

  A knee that marches 6 -> 32 -> 53 -> 74 MiB confirms B is the retained
  tensor, without a profiler. A knee that stays put refutes the theory.

  N=17408 (gate_up_proj) and N=8192 (in_proj_qkvz) are real shapes from this
  model; N=12288 and N=4096 are synthetic controls chosen to spread B's size
  evenly. This also directly tests the shape-specificity claim -- the whole
  argument for why only gate_up_proj regresses is that only its B is large
  enough to leave no slack.

METHOD
------
Only GROUP_SIZE_M varies within each condition ({1,16,32,64}); BLOCK_SIZE_M/
N/K=128 and num_warps=8 fixed. One A/B/As/Bs set per shape, reused, as
tune() does. Eviction = writing X MiB to a scratch buffer, performed OUTSIDE
the timed region (write, synchronize, then start the timer). Conditions are
round-robined within each repeat so session drift cannot bias one relative
to another.

RAW OUTPUT: one JSONL record per (N, num_stages, GROUP_SIZE_M, evict_MiB,
repeat) block, containing every individual iteration latency.
"""

import json
import os
import statistics
import time

import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton

assert current_platform.is_cuda() or current_platform.is_rocm()

K = 5120
BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
M = 1024
GROUP_CANDIDATES = [1, 16, 32, 64]
BASE = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "num_warps": 8}

# (N, num_stages): first two are Part A (reconciliation), rest are Part B.
CONDITIONS = [
    (17408, 3),   # matches exp32
    (17408, 4),   # matches exp16's ncu config -> reconciliation
    (12288, 3),
    (8192, 3),
    (4096, 3),
]

EVICT_MIB = [0, 2, 4, 6, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 160]
N_REPEATS = 4
ITERS_PER_BLOCK = 30
OUT_DIR = "/repo/exp33-results"
JSONL_PATH = os.path.join(OUT_DIR, "knee_sweep_raw.jsonl")
MAX_EVICT_BYTES = max(EVICT_MIB) * 1024 * 1024
MIB = 1024 * 1024


def w8a8_block_matmul(A, B, As, Bs, config, out_dtype):
    m = A.numel() // A.shape[-1]
    n_, k_ = B.shape
    C = A.new_empty(A.shape[:-1] + (n_,), dtype=out_dtype)

    def grid(META):
        return (
            triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n_, META["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        A, B, C, As, Bs, m, n_, k_, BLOCK_N, BLOCK_K,
        A.stride(-2), A.stride(-1),
        B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1),
        As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        **config,
    )
    return C


def make_tensors(n):
    fi = torch.finfo(torch.float8_e4m3fn)
    A = ((torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fi.max
         ).clamp(min=fi.min, max=fi.max).to(torch.float8_e4m3fn)
    B = ((torch.rand(n, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fi.max
         ).clamp(min=fi.min, max=fi.max).to(torch.float8_e4m3fn)
    As = torch.rand(M, (K + BLOCK_K - 1) // BLOCK_K, dtype=torch.float32, device="cuda") * 1e-2
    Bs = torch.rand((n + BLOCK_N - 1) // BLOCK_N, (K + BLOCK_K - 1) // BLOCK_K,
                    dtype=torch.float32, device="cuda") * 1e-2
    return A, B, As, Bs


def time_block(A, B, As, Bs, config, evict_buf, evict_elems, n_iters):
    se, ee = torch.Event(enable_timing=True), torch.Event(enable_timing=True)
    lat = []
    for _ in range(n_iters):
        if evict_elems > 0:
            evict_buf[:evict_elems].zero_()
        torch.accelerator.synchronize()
        se.record()
        w8a8_block_matmul(A, B, As, Bs, config, OUT_DTYPE)
        ee.record()
        ee.synchronize()
        lat.append(se.elapsed_time(ee) * 1000.0)
    return lat


def predicted_knee(n):
    b = n * K / MIB
    a_rowblk = 128 * K / MIB
    c_rowblk = 128 * n * 2 / MIB
    return 96.0 - (b + a_rowblk + c_rowblk), b, c_rowblk


def main():
    torch.cuda.init()
    os.makedirs(OUT_DIR, exist_ok=True)
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    print(f"L2 = {l2/MIB:.1f} MiB   M={M}  K={K}  base={BASE}", flush=True)

    evict_buf = torch.empty(MAX_EVICT_BYTES // 4, dtype=torch.int32, device="cuda")
    fh = open(JSONL_PATH, "a")
    summary = {}
    t0 = time.time()

    for (n, stages) in CONDITIONS:
        key = f"N={n},num_stages={stages}"
        pk, b_mib, c_rb = predicted_knee(n)
        print(f"\n=== {key}   B={b_mib:.1f}MiB  C_rowblock={c_rb:.2f}MiB  "
              f"-> PREDICTED knee ~{pk:.1f}MiB ===", flush=True)

        A, B, As, Bs = make_tensors(n)
        torch.accelerator.synchronize()
        configs = {g: dict(BASE, GROUP_SIZE_M=g, num_stages=stages) for g in GROUP_CANDIDATES}
        for g in GROUP_CANDIDATES:
            time_block(A, B, As, Bs, configs[g], evict_buf, 0, 3)   # JIT warmup
        torch.accelerator.synchronize()

        means = {(g, x): [] for g in GROUP_CANDIDATES for x in EVICT_MIB}
        for rep in range(1, N_REPEATS + 1):
            for x in EVICT_MIB:
                elems = x * MIB // 4
                for g in GROUP_CANDIDATES:
                    lat = time_block(A, B, As, Bs, configs[g], evict_buf, elems, ITERS_PER_BLOCK)
                    means[(g, x)].append(statistics.mean(lat))
                    fh.write(json.dumps({
                        "N": n, "K": K, "M": M, "num_stages": stages,
                        "GROUP_SIZE_M": g, "evict_MiB": x, "repeat": rep,
                        "latencies_us": lat,
                    }) + "\n")
            fh.flush()
        print(f"  ({time.time()-t0:.0f}s elapsed)", flush=True)

        med = {(g, x): statistics.median(means[(g, x)]) for g in GROUP_CANDIDATES for x in EVICT_MIB}
        print(f"  {'evict':>6} | " + " | ".join(f"GSM={g:>2}" for g in GROUP_CANDIDATES), flush=True)
        for x in EVICT_MIB:
            print(f"  {x:>6} | " + " | ".join(f"{med[(g,x)]:7.1f}" for g in GROUP_CANDIDATES), flush=True)

        # knee = smallest evict where GSM=1 is >5% slower than its own evict=0
        base1 = med[(1, 0)]
        knee = next((x for x in EVICT_MIB if med[(1, x)] > base1 * 1.05), None)
        warm_cold = (med[(1, EVICT_MIB[-1])] / base1 - 1) * 100
        g_best0 = min(med[(g, 0)] for g in GROUP_CANDIDATES if g != 1)
        print(f"  --> GSM=1 warm {base1:.1f}us, fully-evicted {med[(1,EVICT_MIB[-1])]:.1f}us "
              f"({warm_cold:+.1f}%)", flush=True)
        print(f"  --> GSM=1 vs best grouped at evict=0: {(base1/g_best0-1)*100:+.2f}%", flush=True)
        print(f"  --> MEASURED knee = {knee} MiB   (PREDICTED {pk:.1f} MiB)", flush=True)

        summary[key] = {
            "B_MiB": b_mib, "C_rowblock_MiB": c_rb, "predicted_knee_MiB": pk,
            "measured_knee_MiB": knee, "gsm1_warm_us": base1,
            "gsm1_cold_us": med[(1, EVICT_MIB[-1])], "gsm1_warm_to_cold_pct": warm_cold,
            "gsm1_vs_best_grouped_at_evict0_pct": (base1 / g_best0 - 1) * 100,
            "median_us": {f"GSM={g},evict={x}": med[(g, x)]
                          for g in GROUP_CANDIDATES for x in EVICT_MIB},
        }
        with open(os.path.join(OUT_DIR, "summary.json"), "w") as sf:
            json.dump(summary, sf, indent=2)

        del A, B, As, Bs
        torch.cuda.empty_cache()

    fh.close()
    print(f"\n=== KNEE vs B SIZE ===")
    print(f"{'condition':<24} {'B MiB':>7} {'predicted':>10} {'measured':>9} {'warm->cold':>11}")
    for k, v in summary.items():
        print(f"{k:<24} {v['B_MiB']:>7.1f} {v['predicted_knee_MiB']:>10.1f} "
              f"{str(v['measured_knee_MiB']):>9} {v['gsm1_warm_to_cold_pct']:>10.1f}%")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
