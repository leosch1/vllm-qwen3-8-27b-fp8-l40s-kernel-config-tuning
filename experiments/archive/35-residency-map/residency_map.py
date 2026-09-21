#!/usr/bin/env python3
"""What is actually still in L2 after a GEMM launch? A per-slice residency map.

WHY
---
Experiments 32/33 established, by eviction sweeps, that row-major
(GROUP_SIZE_M=1) depends on ~88MiB of residency at gate_up_proj's size while
grouped ordering depends on nothing. But that only identifies WHICH tensor
matters (B); it never measured what is actually resident at any moment.

Two competing pictures of the end-of-launch cache state were proposed and
neither has been tested:
  row-major : B was just swept end-to-end, so B should be roughly UNIFORMLY
              resident, and A largely evicted.
  grouped   : A is the hot tensor (touched once per N-tile), while each B
              column-block is touched exactly once -- so B's EARLY columns
              should be the oldest and most evicted, giving a residency
              GRADIENT across B, with A resident.

The gradient prediction is the mechanism originally hypothesised for why
grouped might suffer in a warm cache ("the necessary starting columns of B
were already evicted"). So far it rests only on a 2.2% timing difference.

METHOD -- residency probe by bandwidth
--------------------------------------
L2 and DRAM differ by ~an order of magnitude in bandwidth (L40S: L2 several
TB/s, DRAM 0.86TB/s), so timing a read of a known number of bytes tells us
where those bytes came from. Procedure, per (config, tensor, slice):

  1. run the GEMM once (establishes the end-of-launch cache state)
  2. immediately run a probe kernel that reads ONLY slice i of the tensor
     and accumulates it into a device scalar (so it cannot be optimised out)
  3. time the probe; bytes/time gives effective bandwidth
     -> high  => that slice was resident in L2
     -> ~DRAM => that slice had been evicted

Each probe is preceded by its own fresh GEMM launch, because the probe
itself pulls its slice into cache and would contaminate any later probe.
Slice order is shuffled deterministically per repeat so that thermal/clock
drift or any ordering effect cannot manufacture a gradient.

A calibration pass measures the same probe (a) right after flushing L2 and
(b) twice in a row, giving per-slice DRAM and L2 reference bandwidths to
compare against -- so "resident" vs "evicted" is judged against measured
endpoints on this exact hardware, not assumed constants.

RAW OUTPUT: one JSONL record per probe with every timing.
"""

import json
import os
import statistics
import time

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform

assert current_platform.is_cuda() or current_platform.is_rocm()

N, K = 17408, 5120          # gate_up_proj
M = 1024
BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
BASE = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "num_warps": 8, "num_stages": 3}
GROUP_CANDIDATES = [1, 16]   # row-major vs grouped; the contrast of interest
N_SLICES = 16                # B split into 16 equal row-slices (B rows == N dim)
N_REPEATS = 10
PROBE_ITERS = 1   # MUST be 1: a second read would hit L2 regardless
MIB = 1024 * 1024
OUT_DIR = "/repo/exp34b-results"
JSONL = os.path.join(OUT_DIR, "residency_raw.jsonl")


@triton.jit
def _probe_sum(PTR, OUT, n_elems, BLOCK: tl.constexpr):
    """Read n_elems int32 words; each block stores its own partial sum.

    NOTE: an earlier version used tl.atomic_add into a single scalar. That
    serialised every block on one address and made the probe measure atomic
    throughput (~250GB/s, cold and warm indistinguishable) instead of memory
    bandwidth. Per-block stores remove the contention entirely.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    v = tl.load(PTR + offs, mask=mask, other=0)
    tl.store(OUT + pid, tl.sum(v.to(tl.int32)))


PROBE_BLOCK = 2048


def probe(buf_i32, start_elem, n_elem, out_buf):
    grid = (triton.cdiv(n_elem, PROBE_BLOCK),)
    _probe_sum[grid](buf_i32[start_elem:], out_buf, n_elem, BLOCK=PROBE_BLOCK)


def w8a8(A, B, As, Bs, config):
    m = A.numel() // A.shape[-1]
    n_, k_ = B.shape
    C = A.new_empty(A.shape[:-1] + (n_,), dtype=OUT_DTYPE)

    def grid(META):
        return (triton.cdiv(m, META["BLOCK_SIZE_M"]) * triton.cdiv(n_, META["BLOCK_SIZE_N"]),)

    _w8a8_triton_block_scaled_mm[grid](
        A, B, C, As, Bs, m, n_, k_, BLOCK_N, BLOCK_K,
        A.stride(-2), A.stride(-1), B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1), As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0), **config,
    )
    return C


def main():
    torch.cuda.init()
    os.makedirs(OUT_DIR, exist_ok=True)
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    print(f"L2={l2/MIB:.1f}MiB  N={N} K={K} M={M}  base={BASE}", flush=True)

    fi = torch.finfo(torch.float8_e4m3fn)
    A = ((torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fi.max
         ).clamp(min=fi.min, max=fi.max).to(torch.float8_e4m3fn)
    B = ((torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fi.max
         ).clamp(min=fi.min, max=fi.max).to(torch.float8_e4m3fn)
    As = torch.rand(M, K // BLOCK_K, dtype=torch.float32, device="cuda") * 1e-2
    Bs = torch.rand(N // BLOCK_N, K // BLOCK_K, dtype=torch.float32, device="cuda") * 1e-2

    # int32 views for the probe kernel
    B_i32 = B.view(torch.int32).flatten()
    A_i32 = A.view(torch.int32).flatten()
    out_buf = torch.zeros(1 + B_i32.numel() // 256, dtype=torch.int32, device="cuda")
    flush_buf = torch.empty(int(l2 * 1.5) // 4, dtype=torch.int32, device="cuda")

    b_slice_elems = B_i32.numel() // N_SLICES
    a_slice_elems = A_i32.numel() // N_SLICES
    b_slice_mib = b_slice_elems * 4 / MIB
    a_slice_mib = a_slice_elems * 4 / MIB
    print(f"B total {B_i32.numel()*4/MIB:.1f}MiB -> {N_SLICES} slices of {b_slice_mib:.2f}MiB",
          flush=True)
    print(f"A total {A_i32.numel()*4/MIB:.1f}MiB -> {N_SLICES} slices of {a_slice_mib:.2f}MiB",
          flush=True)

    se, ee = torch.Event(enable_timing=True), torch.Event(enable_timing=True)

    def timed_probe(buf, start, n):
        torch.accelerator.synchronize()
        se.record()
        probe(buf, start, n, out_buf)
        ee.record()
        ee.synchronize()
        ms = se.elapsed_time(ee)
        gbps = (n * 4) / (ms / 1000.0) / 1e9
        return ms * 1000.0, gbps     # us, GB/s

    configs = {g: dict(BASE, GROUP_SIZE_M=g) for g in GROUP_CANDIDATES}
    for g in GROUP_CANDIDATES:      # JIT warmup
        w8a8(A, B, As, Bs, configs[g])
    timed_probe(B_i32, 0, b_slice_elems)
    torch.accelerator.synchronize()

    fh = open(JSONL, "a")

    # ---- calibration: measured DRAM and L2 endpoints for this exact probe ----
    print("\n=== calibration ===", flush=True)
    cal = {}
    for tag, buf, n in (("B", B_i32, b_slice_elems), ("A", A_i32, a_slice_elems)):
        cold, warm = [], []
        for _ in range(5):
            flush_buf.zero_(); torch.accelerator.synchronize()
            cold.append(timed_probe(buf, 0, n)[1])
            warm.append(timed_probe(buf, 0, n)[1])   # immediately again -> resident
        cal[tag] = {"dram_GBps": statistics.median(cold), "l2_GBps": statistics.median(warm)}
        print(f"  {tag}: after-flush {cal[tag]['dram_GBps']:7.1f} GB/s   "
              f"immediately-again {cal[tag]['l2_GBps']:7.1f} GB/s", flush=True)
        fh.write(json.dumps({"kind": "calibration", "tensor": tag,
                             "cold_GBps": cold, "warm_GBps": warm}) + "\n")

    # ---- residency map ----
    results = {}
    t0 = time.time()
    for g in GROUP_CANDIDATES:
        for tag, buf, n_el, sl_mib in (("B", B_i32, b_slice_elems, b_slice_mib),
                                       ("A", A_i32, a_slice_elems, a_slice_mib)):
            per_slice = {i: [] for i in range(N_SLICES)}
            for rep in range(N_REPEATS):
                order = [(i * 7 + rep * 3) % N_SLICES for i in range(N_SLICES)]  # shuffled
                for i in order:
                    flush_buf.zero_()                 # start from a known-cold cache
                    torch.accelerator.synchronize()
                    w8a8(A, B, As, Bs, configs[g])    # the launch under study
                    torch.accelerator.synchronize()
                    us, gbps = timed_probe(buf, i * n_el, n_el)
                    per_slice[i].append(gbps)
                    fh.write(json.dumps({
                        "kind": "probe", "GROUP_SIZE_M": g, "tensor": tag,
                        "slice": i, "repeat": rep, "slice_MiB": sl_mib,
                        "probe_us": us, "GBps": gbps,
                    }) + "\n")
                fh.flush()
            med = {i: statistics.median(per_slice[i]) for i in range(N_SLICES)}
            results[(g, tag)] = med
            dram, l2r = cal[tag]["dram_GBps"], cal[tag]["l2_GBps"]
            print(f"\n=== GROUP_SIZE_M={g}, tensor {tag}: residency map "
                  f"({sl_mib:.2f}MiB slices) ===", flush=True)
            print(f"  slice |   GB/s | resident fraction (0=DRAM {dram:.0f}, 1=L2 {l2r:.0f})", flush=True)
            for i in range(N_SLICES):
                frac = (med[i] - dram) / max(l2r - dram, 1e-9)
                bar = "#" * max(0, min(40, int(frac * 40)))
                print(f"  {i:>5} | {med[i]:6.1f} | {frac:5.2f} {bar}", flush=True)
            print(f"  ({time.time()-t0:.0f}s elapsed)", flush=True)

    fh.close()
    out = {f"GSM={g},{tag}": {str(i): v for i, v in results[(g, tag)].items()}
           for (g, tag) in results}
    out["calibration"] = cal
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== READING THE RESULT ===")
    print("  row-major (GSM=1): expect B roughly UNIFORMLY resident, A largely evicted.")
    print("  grouped  (GSM=16): expect a GRADIENT across B -- late slices resident,")
    print("                     early slices evicted -- and A resident.")
    print("  A flat, fully-evicted B under BOTH would mean neither picture is right.")
    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
