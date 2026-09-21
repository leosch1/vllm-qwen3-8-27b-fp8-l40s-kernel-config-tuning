#!/usr/bin/env python3
"""KV-cache-traffic reproduction (experiment 13).

`12-cold-weight-cycling-repro` tested whether forcing every gate_up_proj
call to read a *different weight tensor* (cycling through 64 copies)
reproduces the real-serving DRAM-bound regression. It partially worked --
tuned's DRAM Read rose from ~12% (same-B) to ~22% (cold-B) -- but real
serving shows ~72.8%. `11-interleaved-cache-locality-repro` separately
tested whether *other kernel types* (real conv/attention/recurrent kernels)
account for the rest -- that came back negative (both configs got *faster*,
not slower, when interleaved with diverse kernels).

Neither test modeled the one thing real serving does at volume that
neither design touched: **real per-request KV-cache read/write traffic**.
At concurrency=64, real serving continuously reads/writes PagedAttention
KV-cache blocks for up to 64 concurrent sequences, at up to 8192 tokens of
context each -- a large, continuous memory stream through L2 that has
nothing to do with which other *weight* tensor is being read. This
experiment tests that directly: same cold-weight-cycling mechanism as `12`,
but with a KV-cache-sized buffer (~20GB, matching the real deployment's
measured ~23-25GiB KV cache) genuinely read+written between every
gate_up_proj call, to see if *this* -- not other GEMM weights, not kernel
diversity -- is what pushes DRAM Read toward real serving's 72.8%.

Design: same 2x2 structure as `12` v2 (same-B/cold-B x default/tuned),
now crossed with a KV-traffic on/off toggle -- 8 conditions total, round-
robin counterbalanced exactly like `12` v2 (to control for the same
clock-drift confound already found and fixed there).

Simplifying assumption, deliberately documented (matching `11`'s own
practice of flagging scope reductions): each gate_up_proj call is treated
as "one layer," and the per-layer KV touch is sized as
total_kv_buffer / 64 (~320MB), touched via a real in-place read+write
op (not just allocation) at a sequentially-advancing offset, so after 64
calls (one full round) the entire buffer has been touched once. This is a
uniform-per-layer approximation -- the real model's hybrid architecture
(full_attention_interval=4) means only 16 of 64 layers do real
quadratic-attention KV reads at this scale, the other 48 are linear/mamba
attention with much smaller fixed-size state -- so this is deliberately a
maximally-aggressive, not a precisely-faithful, KV-traffic model. If this
*doesn't* move the needle, that's real evidence against sheer KV-cache
volume as the mechanism (not just against this specific approximation of
it).
"""
import json
import torch

from _vendored_matmul_timing import w8a8_block_matmul
from vllm.utils.platform_utils import get_device_name_as_file_name

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
M = 2048
N, K = 17408, 5120
NUM_WEIGHT_COPIES = 64  # matches the real model's actual layer count

KV_BUFFER_GB = 20
KV_BUFFER_BYTES = KV_BUFFER_GB * (1024**3)
KV_ELEMENTS = KV_BUFFER_BYTES // 2  # bf16, 2 bytes/element
KV_CHUNK_ELEMENTS = KV_ELEMENTS // NUM_WEIGHT_COPIES  # ~1 layer's worth

ITERS_PER_BLOCK = 250
N_REPEATS = 16  # matches 12's v2: 16*250 = 4000 iterations/condition

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 2,
}


def make_A(m, k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    A_fp32 = (torch.rand(m, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_info.max
    A = A_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    As = torch.rand(m, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
    return A, As


def make_B_copies(n_copies, n, k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    n_tiles = (n + BLOCK_N - 1) // BLOCK_N
    k_tiles = (k + BLOCK_K - 1) // BLOCK_K
    copies = []
    for _ in range(n_copies):
        B_fp32 = (torch.rand(n, k, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_info.max
        B = B_fp32.clamp(min=fp8_info.min, max=fp8_info.max).to(torch.float8_e4m3fn)
        Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * 1e-2
        copies.append((B, Bs))
    return copies


def run_gemm(A, As, B, Bs, config):
    return w8a8_block_matmul(A, B, As, Bs, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)


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

    print(f"Allocating A (fixed) + {NUM_WEIGHT_COPIES} distinct B copies...", flush=True)
    A, As = make_A(M, K)
    B_copies = make_B_copies(NUM_WEIGHT_COPIES, N, K)

    print(f"Allocating {KV_BUFFER_GB}GB KV-cache-sized buffer "
          f"({KV_CHUNK_ELEMENTS * 2 / 1e6:.1f}MB/touch, {NUM_WEIGHT_COPIES} touches/round)...",
          flush=True)
    kv_buffer = torch.zeros(KV_ELEMENTS, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    print("done allocating", flush=True)

    def touch_kv(global_iter):
        # sequentially-advancing chunk, so one full round (64 calls) touches
        # the entire buffer once -- real read + write, not just allocation
        chunk_idx = global_iter % NUM_WEIGHT_COPIES
        start = chunk_idx * KV_CHUNK_ELEMENTS
        end = start + KV_CHUNK_ELEMENTS
        kv_buffer[start:end].add_(1.0)

    def run_block(config, weight_copies, kv_on, n_iters, offset):
        start_event = torch.Event(enable_timing=True)
        end_event = torch.Event(enable_timing=True)
        latencies = []
        for j in range(n_iters):
            i = offset + j
            B, Bs = weight_copies[i % len(weight_copies)]
            torch.accelerator.synchronize()
            start_event.record()
            run_gemm(A, As, B, Bs, config)
            if kv_on:
                touch_kv(i)
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event) * 1000)  # us
        return latencies

    conditions = {
        ("default", "same", "noKV"): (DEFAULT_CONFIG, [B_copies[0]], False),
        ("default", "same", "KV"):   (DEFAULT_CONFIG, [B_copies[0]], True),
        ("default", "cold", "noKV"): (DEFAULT_CONFIG, B_copies, False),
        ("default", "cold", "KV"):   (DEFAULT_CONFIG, B_copies, True),
        ("tuned", "same", "noKV"):   (tuned_config, [B_copies[0]], False),
        ("tuned", "same", "KV"):     (tuned_config, [B_copies[0]], True),
        ("tuned", "cold", "noKV"):   (tuned_config, B_copies, False),
        ("tuned", "cold", "KV"):     (tuned_config, B_copies, True),
    }
    order = list(conditions.keys())

    all_latencies = {c: [] for c in conditions}
    per_block_means = {c: [] for c in conditions}
    offsets = {c: 0 for c in conditions}

    print("\n=== warmup ===", flush=True)
    for c, (config, copies, kv_on) in conditions.items():
        run_block(config, copies, kv_on, 5, 0)
    torch.accelerator.synchronize()

    print(f"\n=== {N_REPEATS} repeats x {ITERS_PER_BLOCK} iters/block, "
          f"round-robin order {order} ===", flush=True)
    for r in range(N_REPEATS):
        for c in order:
            config, copies, kv_on = conditions[c]
            lat = run_block(config, copies, kv_on, ITERS_PER_BLOCK, offsets[c])
            offsets[c] += ITERS_PER_BLOCK
            all_latencies[c].extend(lat)
            per_block_means[c].append(sum(lat) / len(lat))
        print(f"  repeat {r+1}/{N_REPEATS}: " +
              "  ".join(f"{'-'.join(c)}={per_block_means[c][-1]:.1f}us" for c in order),
              flush=True)

    print("\n=== Per-condition overall mean ===")
    means = {}
    for c in order:
        m = sum(all_latencies[c]) / len(all_latencies[c])
        means[c] = m
        print(f"  {'-'.join(c):>20}: {m:.1f}us  (n={len(all_latencies[c])})")

    print("\n=== Trend check: first-half vs second-half block means ===")
    for c in order:
        blocks = per_block_means[c]
        half = len(blocks) // 2
        fh = sum(blocks[:half]) / half
        sh = sum(blocks[half:]) / (len(blocks) - half)
        print(f"  {'-'.join(c):>20}: first-half={fh:.1f}us  second-half={sh:.1f}us  "
              f"drift={(sh - fh) / fh * 100:+.2f}%")

    print("\n=== Summary: effect of adding KV traffic, per config x weight-state ===")
    for role in ["default", "tuned"]:
        for weight_state in ["same", "cold"]:
            no_kv = means[(role, weight_state, "noKV")]
            kv = means[(role, weight_state, "KV")]
            print(f"  {role}-{weight_state}: no-KV={no_kv:.1f}us -> with-KV={kv:.1f}us  "
                  f"({(kv / no_kv - 1) * 100:+.2f}%)")

    print("\nFor reference: real serving DRAM Read -- default ~15.8%, tuned ~72.8%.")
    print("12's cold-weight-cycling-only result: default DRAM Read ~9.2% (flat), "
          "tuned ~12%(same)->22%(cold).")


if __name__ == "__main__":
    main()
