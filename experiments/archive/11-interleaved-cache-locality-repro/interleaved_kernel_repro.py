#!/usr/bin/env python3
"""Single-process, single-stream interleaved-kernel reproduction of the
gate_up_proj real-vs-isolated inflation, without vLLM or a second process.

Design (see experiments/11-interleaved-cache-locality-repro/README.md for
the full writeup):
  - No multiprocessing, no noisy-neighbor process -- unlike
    04-contention-simulation, everything here runs in ONE process on ONE
    default CUDA stream, matching the confirmed single-stream/single-context
    execution model of real vLLM serving.
  - Real per-decode-step call multiplicities across the 5 W8A8 shapes
    (64/48/16/64/64 -- gate_up_proj and down_proj/out_proj every layer,
    in_proj_qkvz on 48 linear-attention layers, qkv_proj on 16 full-attention
    layers, full_attention_interval=4) are replayed layer-by-layer, so
    gate_up_proj's kernel gets genuinely different compiled Triton kernels
    launched between its own repeats, instead of being called back-to-back
    with itself the way the isolated benchmark does.
  - Only gate_up_proj's own launches are timed (CUDA events recorded around
    just that call); the other 4 shapes' calls are real, executed, timed-out
    kernel launches too -- they're just not the thing being measured -- so
    the GPU pipeline is never artificially drained by a host-side sync until
    the very end, preserving the async pipelining real serving has.
"""

import json
import sys
import time

import torch

from _vendored_matmul_timing import w8a8_block_matmul
from vllm.utils.platform_utils import get_device_name_as_file_name

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
M = 2048  # the slow-cluster batch size this whole investigation converges on
NUM_STEPS = 200  # 200 * 64 = 12,800 timed gate_up_proj samples

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}

# (role, N, K) for all 5 real shapes
SHAPES = {
    "gate_up_proj": (17408, 5120),
    "in_proj_qkvz": (8192, 5120),
    "qkv_proj": (7168, 5120),
    "down_proj": (5120, 8704),
    "out_proj": (5120, 3072),
}


def make_tensors(M, N, K, block_n, block_k):
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min
    factor = 1e-2

    A_fp32 = (torch.rand(M, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    A = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B_fp32 = (torch.rand(N, K, dtype=torch.float32, device="cuda") - 0.5) * 2 * fp8_max
    B = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    As = torch.rand(M, k_tiles, dtype=torch.float32, device="cuda") * factor
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device="cuda") * factor
    return A, B, As, Bs


def load_tuned_config(device_name, N, K, M):
    path = (
        f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
        f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )
    with open(path) as f:
        configs = {int(k): v for k, v in json.load(f).items()}
    return configs[M]


def run_pass(config_mode, tensors, configs):
    """config_mode: 'default' or 'tuned'. Runs NUM_STEPS synthetic decode
    steps, each replaying all 64 layers' real call sequence, timing only
    gate_up_proj's own launches via CUDA events (no intermediate syncs)."""
    gu = tensors["gate_up_proj"]
    iq = tensors["in_proj_qkvz"]
    qk = tensors["qkv_proj"]
    dn = tensors["down_proj"]
    ot = tensors["out_proj"]

    def cfg(role):
        return DEFAULT_CONFIG if config_mode == "default" else configs[role]

    # warmup (JIT compile every kernel variant before timing starts)
    for role, t in tensors.items():
        w8a8_block_matmul(*t, [BLOCK_N, BLOCK_K], cfg(role), OUT_DTYPE)
    torch.cuda.synchronize()

    # Bug found the hard way: allocating a fresh torch.cuda.Event() pair
    # every iteration and never synchronizing until the very end lets tens
    # of thousands of live, un-freed events pile up -- CUDA event bookkeeping
    # overhead itself then scales with outstanding-event count and swamps
    # the actual kernel time (measured floor came out ~1150-1200us even at
    # the *minimum* sample, ~9x the known-correct isolated baseline). Fix:
    # flush (synchronize, read back elapsed_time, discard) every FLUSH_EVERY
    # iterations, bounding outstanding events while still preserving async,
    # unsynced pipelining *within* each flush window -- plenty of depth to
    # keep the GPU genuinely running ahead of the CPU, without the resource
    # buildup.
    FLUSH_EVERY = 256
    durations_us = []
    start_events, end_events = [], []

    def flush():
        if not start_events:
            return
        torch.cuda.synchronize()
        durations_us.extend(se.elapsed_time(ee) * 1000 for se, ee in zip(start_events, end_events))
        start_events.clear()
        end_events.clear()

    for step in range(NUM_STEPS):
        for layer_idx in range(64):
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            w8a8_block_matmul(*gu, [BLOCK_N, BLOCK_K], cfg("gate_up_proj"), OUT_DTYPE)
            ee.record()
            start_events.append(se)
            end_events.append(ee)
            if len(start_events) >= FLUSH_EVERY:
                flush()

            # full_attention_interval=4: every 4th layer is full-attention
            if (layer_idx + 1) % 4 == 0:
                w8a8_block_matmul(*qk, [BLOCK_N, BLOCK_K], cfg("qkv_proj"), OUT_DTYPE)
            else:
                w8a8_block_matmul(*iq, [BLOCK_N, BLOCK_K], cfg("in_proj_qkvz"), OUT_DTYPE)
            w8a8_block_matmul(*dn, [BLOCK_N, BLOCK_K], cfg("down_proj"), OUT_DTYPE)
            w8a8_block_matmul(*ot, [BLOCK_N, BLOCK_K], cfg("out_proj"), OUT_DTYPE)

    flush()
    return durations_us


def run_isolated_control(config_mode, tensors, configs):
    """Same timing method, but gate_up_proj called back-to-back with only
    itself -- no interleaving -- as an apples-to-apples control using the
    exact same event-based methodology as run_pass(), not benchmark_config()'s
    separately-implemented averaging."""
    gu = tensors["gate_up_proj"]
    config = DEFAULT_CONFIG if config_mode == "default" else configs["gate_up_proj"]

    for _ in range(5):
        w8a8_block_matmul(*gu, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)
    torch.cuda.synchronize()

    FLUSH_EVERY = 256
    durations_us = []
    start_events, end_events = [], []

    def flush():
        if not start_events:
            return
        torch.cuda.synchronize()
        durations_us.extend(se.elapsed_time(ee) * 1000 for se, ee in zip(start_events, end_events))
        start_events.clear()
        end_events.clear()

    for _ in range(NUM_STEPS * 64):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        w8a8_block_matmul(*gu, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)
        ee.record()
        start_events.append(se)
        end_events.append(ee)
        if len(start_events) >= FLUSH_EVERY:
            flush()
    flush()
    return durations_us


def summarize(label, durations_us):
    d = sorted(durations_us)
    n = len(d)
    mean = sum(d) / n
    median = d[n // 2]
    p10 = d[int(n * 0.10)]
    p90 = d[int(n * 0.90)]
    print(f"{label}: n={n} mean={mean:.1f}us median={median:.1f}us p10={p10:.1f}us p90={p90:.1f}us min={d[0]:.1f}us max={d[-1]:.1f}us")
    return mean


def main():
    device_name = get_device_name_as_file_name()
    print(f"device_name={device_name}, M={M}, NUM_STEPS={NUM_STEPS}")

    tensors = {role: make_tensors(M, N, K, BLOCK_N, BLOCK_K) for role, (N, K) in SHAPES.items()}
    configs = {role: load_tuned_config(device_name, N, K, M) for role, (N, K) in SHAPES.items()}
    print(f"tuned gate_up_proj config @ M={M}: {configs['gate_up_proj']}")

    # This node's GPU was previously idle (P8, ~210MHz vs 2520MHz max boost)
    # -- a brief warmup never gives NVIDIA's boost-clock ramp enough
    # sustained load to kick in, and every timing below would otherwise be
    # inflated ~10x by clock state, not by anything about interleaving.
    # Spin real load for a few seconds first and confirm the clock actually
    # came up before trusting any measurement.
    print("spinning up GPU clocks (sustained load, ~4s)...")
    spin_end = time.time() + 4.0
    gu = tensors["gate_up_proj"]
    gu_cfg = configs["gate_up_proj"]
    while time.time() < spin_end:
        for _ in range(20):
            w8a8_block_matmul(*gu, [BLOCK_N, BLOCK_K], gu_cfg, OUT_DTYPE)
        torch.cuda.synchronize()
    import subprocess
    clock = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,pstate", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"GPU clock after spin-up: {clock}")

    results = {}
    for config_mode in ["default", "tuned"]:
        print(f"\n=== {config_mode} ===")
        iso = run_isolated_control(config_mode, tensors, configs)
        results[f"{config_mode}_isolated"] = summarize(f"{config_mode} isolated (back-to-back, this script's own timing method)", iso)

        interleaved = run_pass(config_mode, tensors, configs)
        results[f"{config_mode}_interleaved"] = summarize(f"{config_mode} interleaved (real per-layer call sequence, single stream)", interleaved)

    print("\n=== summary ===")
    print(f"default: isolated={results['default_isolated']:.1f}us -> interleaved={results['default_interleaved']:.1f}us ({results['default_interleaved']/results['default_isolated']:.2f}x)")
    print(f"tuned:   isolated={results['tuned_isolated']:.1f}us -> interleaved={results['tuned_interleaved']:.1f}us ({results['tuned_interleaved']/results['tuned_isolated']:.2f}x)")
    print("\nreference (real production, from experiment 09, M=2048):")
    print("default: isolated=133.0us, real=1148.2us (8.63x)")
    print("tuned:   isolated=117.8us, real=2054.7us (17.44x)")


if __name__ == "__main__":
    main()
