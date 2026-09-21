#!/usr/bin/env python3
"""How much L2 residency does each GROUP_SIZE_M ordering actually depend on?

THE OPEN QUESTION this addresses
--------------------------------
It is established (experiments 30/30b, and the committed config) that the
unfixed tuner reproducibly selects GROUP_SIZE_M=1 for gate_up_proj at every
large-M anchor, and (experiment 28) that adding an L2 flush to the tuner
flips that to grouped values at all six anchors. The COLD half of the
mechanism is also measured: ncu (which flushes caches by default, so its
numbers are a cold measurement) showed GROUP_SIZE_M=1 re-fetching ~58MB
more DRAM than grouped at M=1024, which accounts for its 11.8% slowdown
almost exactly.

The WARM half is NOT explained. We know empirically that the tuner measures
GROUP_SIZE_M=1 as faster when the cache is warm, but not why -- and two
attempted explanations were wrong: (a) "A changes between launches" is false
for the tuner (tune() allocates ONE A and ONE B and reuses both), and (b)
"grouped sheds more of B" was asserted without justification -- total
eviction pressure is the same in both orderings (A+B+C = 124MiB touched
against a 96MiB L2), and capacity-wise both CAN end a launch with A+B (90MiB)
resident.

WHY THIS DESIGN
---------------
The two orderings depend on different amounts of resident data:
  GROUP_SIZE_M=1  (row-major)  reuses B: re-reads all 85MiB of B once per
                               M-tile (8x at M=1024). Needs ~85MiB resident.
  GROUP_SIZE_M>=8 (grouped)    reuses A: re-reads all of A once per N-tile
                               (136x). Needs only ~5MiB (M=1024) resident.

So instead of trying to attribute DRAM bytes per tensor (aggregate counters
cannot do that, and ncu here would need DCGM paused cluster-wide), measure
the dependency directly: evict X bytes of least-recently-used data between
timed launches and sweep X. Writing X bytes to a scratch buffer displaces
roughly X bytes of LRU data.

PREDICTION if the reuse-distance theory is right:
  - grouped should stay flat until X approaches ~90MiB (its 5MiB working set
    survives almost any partial eviction), then degrade.
  - GROUP_SIZE_M=1 should degrade MUCH earlier -- its ~90MiB working set has
    only ~6MiB of slack in a 96MiB L2, so even a small eviction starts
    costing it B re-reads.
  - X=0 reproduces the tuner's warm condition; large X reproduces the flush
    condition. If GROUP_SIZE_M=1 wins at X=0 and loses at large X, the
    crossover point is the quantitative answer to "how much residency does
    the tuner's regime hand it".

If instead both orderings degrade at the same X, the residency theory is
wrong and the warm-regime advantage is something else (L2 bandwidth,
latency-hiding, occupancy) -- which would be an equally useful result.

METHOD
------
gate_up_proj (N=17408, K=5120) at M in {1024, 2048}. Base config is the
winning family (BLOCK_SIZE_M/N/K=128, num_warps=8, num_stages=3); only
GROUP_SIZE_M varies across {1,16,32,64}. Eviction sizes swept from 0 to
160MiB. Conditions are round-robined within each repeat so monotonic drift
(experiment 29 measured up to +17% over a session) cannot bias one condition
relative to another. Eviction happens OUTSIDE the timed region (write, then
synchronize, then start the timer), same as experiment 24/28's flush design.

RAW OUTPUT: one JSONL record per (GROUP_SIZE_M, evict_MiB, repeat) block
containing every individual iteration latency -- nothing aggregated away.
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

N, K = 17408, 5120  # gate_up_proj
BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
M_VALUES = [1024, 2048]
GROUP_CANDIDATES = [1, 16, 32, 64]
BASE_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "num_warps": 8,
    "num_stages": 3,
}
EVICT_MIB = [0, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 160]
N_REPEATS = 5
ITERS_PER_BLOCK = 40
OUT_DIR = "/repo/exp32-results"
JSONL_PATH = os.path.join(OUT_DIR, "eviction_sweep_raw.jsonl")
MAX_EVICT_BYTES = max(EVICT_MIB) * 1024 * 1024


def w8a8_block_matmul(A, B, As, Bs, config, out_dtype):
    M = A.numel() // A.shape[-1]
    N_, K_ = B.shape
    C = A.new_empty(A.shape[:-1] + (N_,), dtype=out_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N_, META["BLOCK_SIZE_N"]),
        )

    _w8a8_triton_block_scaled_mm[grid](
        A, B, C, As, Bs, M, N_, K_, BLOCK_N, BLOCK_K,
        A.stride(-2), A.stride(-1),
        B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1),
        As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        **config,
    )
    return C


def make_tensors(M):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    A = ((torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
         ).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B = ((torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
         ).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    As = torch.rand(M, (K + BLOCK_K - 1) // BLOCK_K, dtype=torch.float32, device="cuda") * 1e-2
    Bs = torch.rand((N + BLOCK_N - 1) // BLOCK_N, (K + BLOCK_K - 1) // BLOCK_K,
                    dtype=torch.float32, device="cuda") * 1e-2
    return A, B, As, Bs


def time_block(A, B, As, Bs, config, evict_buf, evict_elems, n_iters):
    """Evict evict_elems*4 bytes of LRU data, then time one launch. Repeated."""
    start_event = torch.Event(enable_timing=True)
    end_event = torch.Event(enable_timing=True)
    lat = []
    for _ in range(n_iters):
        if evict_elems > 0:
            evict_buf[:evict_elems].zero_()   # displaces ~evict bytes of LRU data
        torch.accelerator.synchronize()       # eviction is OUTSIDE the timed region
        start_event.record()
        w8a8_block_matmul(A, B, As, Bs, config, OUT_DTYPE)
        end_event.record()
        end_event.synchronize()
        lat.append(start_event.elapsed_time(end_event) * 1000.0)  # ms -> us
    return lat


def main():
    torch.cuda.init()
    os.makedirs(OUT_DIR, exist_ok=True)
    dev = torch.cuda.current_device()
    try:
        l2 = torch.cuda.get_device_properties(dev).L2_cache_size
        print(f"device L2_cache_size = {l2} bytes = {l2/1024/1024:.1f} MiB", flush=True)
    except AttributeError:
        print("L2_cache_size not exposed", flush=True)

    evict_buf = torch.empty(MAX_EVICT_BYTES // 4, dtype=torch.int32, device="cuda")
    configs = {g: dict(BASE_CONFIG, GROUP_SIZE_M=g) for g in GROUP_CANDIDATES}
    print(f"base config (GROUP_SIZE_M varies): {BASE_CONFIG}", flush=True)

    fh = open(JSONL_PATH, "a")
    summary = {}
    t0 = time.time()

    for M in M_VALUES:
        A, B, As, Bs = make_tensors(M)
        torch.accelerator.synchronize()
        a_mib = A.numel() / 1024 / 1024
        b_mib = B.numel() / 1024 / 1024
        c_mib = M * N * 2 / 1024 / 1024
        print(f"\n=== M={M}  A={a_mib:.1f}MiB  B={b_mib:.1f}MiB  C={c_mib:.1f}MiB "
              f"(A+B={a_mib+b_mib:.1f}MiB) ===", flush=True)

        print("  JIT warmup for all 4 configs...", flush=True)
        for g in GROUP_CANDIDATES:
            time_block(A, B, As, Bs, configs[g], evict_buf, 0, 3)
        torch.accelerator.synchronize()

        means = {(g, x): [] for g in GROUP_CANDIDATES for x in EVICT_MIB}
        for rep in range(1, N_REPEATS + 1):
            for x in EVICT_MIB:                     # round-robin: drift can't bias
                elems = x * 1024 * 1024 // 4
                for g in GROUP_CANDIDATES:
                    lat = time_block(A, B, As, Bs, configs[g], evict_buf, elems,
                                     ITERS_PER_BLOCK)
                    means[(g, x)].append(statistics.mean(lat))
                    fh.write(json.dumps({
                        "M": M, "GROUP_SIZE_M": g, "evict_MiB": x, "repeat": rep,
                        "base_config": BASE_CONFIG,
                        "latencies_us": lat,
                    }) + "\n")
            fh.flush()
            print(f"  repeat {rep}/{N_REPEATS} done ({time.time()-t0:.0f}s)", flush=True)

        print(f"\n  --- M={M}: mean launch time (us) by eviction size ---", flush=True)
        hdr = "  evict_MiB | " + " | ".join(f"GSM={g:>2}" for g in GROUP_CANDIDATES) + " |  best | GSM1 vs best"
        print(hdr, flush=True)
        rows = {}
        for x in EVICT_MIB:
            vals = {g: statistics.median(means[(g, x)]) for g in GROUP_CANDIDATES}
            best = min(vals, key=vals.get)
            delta = (vals[1] / vals[best] - 1) * 100
            rows[x] = {"per_gsm_us": vals, "best": best, "gsm1_vs_best_pct": delta}
            print(f"  {x:>9} | " + " | ".join(f"{vals[g]:7.1f}" for g in GROUP_CANDIDATES)
                  + f" | GSM={best:<3} | {delta:+6.2f}%", flush=True)
        summary[M] = rows
        with open(os.path.join(OUT_DIR, "summary.json"), "w") as sf:
            json.dump(summary, sf, indent=2)

        del A, B, As, Bs
        torch.cuda.empty_cache()

    fh.close()
    print(f"\ntotal time {time.time()-t0:.0f}s", flush=True)
    print("\nInterpretation guide:")
    print("  evict=0 reproduces the tuner's warm condition; large evict reproduces the flush.")
    print("  If GSM=1 wins at evict=0 and loses as evict grows, the crossover is the")
    print("  amount of residency the tuner's regime was handing it. If all four degrade")
    print("  together, the residency theory is wrong and the warm advantage is elsewhere.")


if __name__ == "__main__":
    main()
