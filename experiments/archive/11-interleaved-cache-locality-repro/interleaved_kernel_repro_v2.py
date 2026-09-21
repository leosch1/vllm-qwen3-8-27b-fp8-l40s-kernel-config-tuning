#!/usr/bin/env python3
"""Faithful version of experiment 11: same single-process/single-stream
design, but the "other kernels" interleaved between gate_up_proj repeats
are now genuinely different real kernels (real vLLM causal_conv1d, real
PyTorch SDPA attention, real silu activation) instead of just other W8A8
GEMM tile-config variants -- testing whether the negative result from the
"cheap version" was because kernel-family diversity (not just tile-config
diversity) is what actually matters for the cache-locality hypothesis.

Scoping decision, made without live cluster access to verify what's
installed (see README): the exact gated-delta-rule backend dispatches to
one of 3 optional-dependency implementations (flashinfer/fla/cutedsl)
depending on hardware/availability, each with a different call signature.
Rather than risk guessing wrong blind, this uses a real PyTorch reference
computation for that step (matching the actual math: exp-gated recurrent
state update, einsum, silu) instead of the exact backend kernel -- real GPU
work, genuinely different instruction profile from a GEMM, just not
byte-identical to whichever Triton/CUDA kernel vLLM would actually dispatch
to. causal_conv1d_fn IS vLLM's real, exact kernel (confirmed tractable,
single well-documented backend). Full attention uses PyTorch's native
scaled_dot_product_attention -- real, complex, fused, very different from
a GEMM -- rather than vLLM's own PagedAttention (which needs a KV-cache
block-table setup this harness doesn't have and doesn't need for the
question being asked: does genuinely different kernel-family diversity,
not just tile-config diversity, change the result).

Real model dims (Qwen/Qwen3.8-27B-FP8, TP=2), from the deployed model's own
config.json:
  full_attention_interval=4, num_hidden_layers=64
  head_dim=256, num_attention_heads=24 (12/rank), num_key_value_heads=4 (2/rank)
  linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_num_key_heads=16,
  linear_value_head_dim=128, linear_num_value_heads=48
"""

import json
import time

import torch
import torch.nn.functional as F

from _vendored_matmul_timing import w8a8_block_matmul
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.utils.platform_utils import get_device_name_as_file_name
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

BLOCK_N, BLOCK_K = 128, 128
OUT_DTYPE = torch.bfloat16
TUNED_DIR = "./tuned-configs"
M = 2048
NUM_STEPS = 200
FLUSH_EVERY = 256

# treat M=2048 as 8 sequences of 256 tokens each, matching this workload's
# real, established prompt length (see experiments/02, /03) -- rather than
# one implausible 2048-token sequence.
NUM_SEQS = 8
SEQ_LEN = M // NUM_SEQS

DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32, "num_warps": 4, "num_stages": 2,
}
SHAPES = {
    "gate_up_proj": (17408, 5120),
    "in_proj_qkvz": (8192, 5120),
    "qkv_proj": (7168, 5120),
    "down_proj": (5120, 8704),
    "out_proj": (5120, 3072),
}

# real model dims
CONV_DIM = 5120  # approx per-rank q+k+v combined width fed to the conv
CONV_WIDTH = 4
ATTN_HEADS, ATTN_KV_HEADS, ATTN_HEAD_DIM = 12, 2, 256  # per-rank (TP=2)
GDN_HEADS, GDN_HEAD_DIM = 24, 128  # per-rank (48/2, 16 or value heads/2 -- approx


def make_gemm_tensors(M, N, K, block_n, block_k):
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


def make_conv_inputs():
    """Real vLLM causal_conv1d_fn inputs, continuous-batching layout:
    NUM_SEQS sequences of SEQ_LEN tokens each, concatenated."""
    x = torch.randn(CONV_DIM, M, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(CONV_DIM, CONV_WIDTH, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(CONV_DIM, device="cuda", dtype=torch.bfloat16)
    conv_states = torch.zeros(NUM_SEQS + 1, CONV_DIM, CONV_WIDTH - 1, device="cuda", dtype=torch.bfloat16)
    query_start_loc = torch.arange(0, M + 1, SEQ_LEN, device="cuda", dtype=torch.int32)
    cache_indices = torch.arange(1, NUM_SEQS + 1, device="cuda", dtype=torch.int32)
    has_initial_state = torch.zeros(NUM_SEQS, device="cuda", dtype=torch.bool)
    return x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state


def run_conv(conv_inputs):
    x, weight, bias, conv_states, query_start_loc, cache_indices, has_initial_state = conv_inputs
    return causal_conv1d_fn(
        x, weight, bias=bias, conv_states=conv_states,
        query_start_loc=query_start_loc, cache_indices=cache_indices,
        has_initial_state=has_initial_state, activation="silu",
    )


# The recurrent update below runs a genuinely sequential Python loop (it's
# a per-token state update, matching the real math) -- at the true SEQ_LEN
# (256) x 9600 calls this is ~2.4M sequential einsum dispatches, wildly
# impractical. Decoupled from SEQ_LEN: still real, substantial, genuinely
# different GPU work each occurrence, just a smaller chunk than the real
# kernel would process in one call. Documented scope reduction, not a
# fidelity claim.
GDN_CHUNK_LEN = 8


def make_gdn_recurrent_inputs():
    """Real-scale tensors for a gated-delta-rule-style recurrent update --
    a real PyTorch reference computation (einsum/exp/silu), not the exact
    backend Triton/FlashInfer kernel (see module docstring)."""
    shape = (NUM_SEQS, GDN_CHUNK_LEN, GDN_HEADS, GDN_HEAD_DIM)
    q = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(NUM_SEQS, GDN_CHUNK_LEN, GDN_HEADS, device="cuda", dtype=torch.float32) * -0.1
    beta = torch.rand(NUM_SEQS, GDN_CHUNK_LEN, GDN_HEADS, device="cuda", dtype=torch.float32)
    state = torch.zeros(NUM_SEQS, GDN_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM, device="cuda", dtype=torch.float32)
    return q, k, v, g, beta, state


def run_gdn_recurrent(gdn_inputs):
    """One real chunk's worth of the actual gated-delta-rule math (exp-gated
    recurrent state update via einsum), run token-by-token across the chunk
    -- real GPU compute, genuinely different instruction profile from a
    GEMM, not the exact production kernel."""
    q, k, v, g, beta, state = gdn_inputs
    out = torch.empty_like(v)
    decay = torch.exp(g)  # (seqs, seq_len, heads)
    for t in range(GDN_CHUNK_LEN):
        qt = q[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()  # (seqs, heads, dim), fp32 to match state
        dt, bt = decay[:, t], beta[:, t]        # (seqs, heads)
        state.mul_(dt.unsqueeze(-1).unsqueeze(-1))
        delta = torch.einsum("shd,she->shde", kt, (vt - torch.einsum("shde,shd->she", state, kt)))
        state.add_(bt.unsqueeze(-1).unsqueeze(-1) * delta)
        out[:, t] = torch.einsum("shde,shd->she", state, qt).to(v.dtype)
    return F.silu(out)


def make_attn_inputs():
    shape = (NUM_SEQS, ATTN_HEADS, SEQ_LEN, ATTN_HEAD_DIM)
    kv_shape = (NUM_SEQS, ATTN_KV_HEADS, SEQ_LEN, ATTN_HEAD_DIM)
    q = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*kv_shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*kv_shape, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def run_attn(attn_inputs):
    q, k, v = attn_inputs
    # GQA: repeat kv heads to match query heads
    rep = ATTN_HEADS // ATTN_KV_HEADS
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def main():
    device_name = get_device_name_as_file_name()
    print(f"device_name={device_name}, M={M} ({NUM_SEQS} seqs x {SEQ_LEN} tok), NUM_STEPS={NUM_STEPS}")

    gemm_tensors = {role: make_gemm_tensors(M, N, K, BLOCK_N, BLOCK_K) for role, (N, K) in SHAPES.items()}
    configs = {role: load_tuned_config(device_name, N, K, M) for role, (N, K) in SHAPES.items()}
    print(f"tuned gate_up_proj config @ M={M}: {configs['gate_up_proj']}")

    conv_inputs = make_conv_inputs()
    gdn_inputs = make_gdn_recurrent_inputs()
    attn_inputs = make_attn_inputs()

    print("spinning up GPU clocks (sustained load, ~4s)...")
    spin_end = time.time() + 4.0
    gu = gemm_tensors["gate_up_proj"]
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

    def cfg(role, config_mode):
        return DEFAULT_CONFIG if config_mode == "default" else configs[role]

    def run_isolated_control(config_mode):
        gu_t = gemm_tensors["gate_up_proj"]
        config = DEFAULT_CONFIG if config_mode == "default" else configs["gate_up_proj"]
        for _ in range(5):
            w8a8_block_matmul(*gu_t, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)
        torch.cuda.synchronize()
        durations_us, start_events, end_events = [], [], []

        def flush():
            if not start_events:
                return
            torch.cuda.synchronize()
            durations_us.extend(se.elapsed_time(ee) * 1000 for se, ee in zip(start_events, end_events))
            start_events.clear(); end_events.clear()

        for _ in range(NUM_STEPS * 64):
            se = torch.cuda.Event(enable_timing=True)
            ee = torch.cuda.Event(enable_timing=True)
            se.record()
            w8a8_block_matmul(*gu_t, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)
            ee.record()
            start_events.append(se); end_events.append(ee)
            if len(start_events) >= FLUSH_EVERY:
                flush()
        flush()
        return durations_us

    def run_pass(config_mode):
        gu_t, iq, qk, dn, ot = (gemm_tensors[r] for r in ["gate_up_proj", "in_proj_qkvz", "qkv_proj", "down_proj", "out_proj"])
        # warmup every kernel path once
        for role, t in gemm_tensors.items():
            w8a8_block_matmul(*t, [BLOCK_N, BLOCK_K], cfg(role, config_mode), OUT_DTYPE)
        run_conv(conv_inputs)
        run_gdn_recurrent(gdn_inputs)
        run_attn(attn_inputs)
        torch.cuda.synchronize()

        durations_us, start_events, end_events = [], [], []

        def flush():
            if not start_events:
                return
            torch.cuda.synchronize()
            durations_us.extend(se.elapsed_time(ee) * 1000 for se, ee in zip(start_events, end_events))
            start_events.clear(); end_events.clear()

        for step in range(NUM_STEPS):
            for layer_idx in range(64):
                se = torch.cuda.Event(enable_timing=True)
                ee = torch.cuda.Event(enable_timing=True)
                se.record()
                w8a8_block_matmul(*gu_t, [BLOCK_N, BLOCK_K], cfg("gate_up_proj", config_mode), OUT_DTYPE)
                ee.record()
                start_events.append(se); end_events.append(ee)
                if len(start_events) >= FLUSH_EVERY:
                    flush()

                if (layer_idx + 1) % 4 == 0:
                    # full-attention layer: qkv_proj GEMM + real SDPA attention
                    w8a8_block_matmul(*qk, [BLOCK_N, BLOCK_K], cfg("qkv_proj", config_mode), OUT_DTYPE)
                    run_attn(attn_inputs)
                else:
                    # linear-attention layer: in_proj_qkvz GEMM + real conv + gdn recurrent
                    w8a8_block_matmul(*iq, [BLOCK_N, BLOCK_K], cfg("in_proj_qkvz", config_mode), OUT_DTYPE)
                    run_conv(conv_inputs)
                    run_gdn_recurrent(gdn_inputs)
                w8a8_block_matmul(*dn, [BLOCK_N, BLOCK_K], cfg("down_proj", config_mode), OUT_DTYPE)
                w8a8_block_matmul(*ot, [BLOCK_N, BLOCK_K], cfg("out_proj", config_mode), OUT_DTYPE)
        flush()
        return durations_us

    def summarize(label, durations_us):
        d = sorted(durations_us)
        n = len(d)
        mean = sum(d) / n
        print(f"{label}: n={n} mean={mean:.1f}us median={d[n//2]:.1f}us p10={d[int(n*0.1)]:.1f}us p90={d[int(n*0.9)]:.1f}us min={d[0]:.1f}us max={d[-1]:.1f}us")
        return mean

    results = {}
    for config_mode in ["default", "tuned"]:
        print(f"\n=== {config_mode} ===")
        results[f"{config_mode}_isolated"] = summarize(f"{config_mode} isolated (back-to-back)", run_isolated_control(config_mode))
        results[f"{config_mode}_interleaved"] = summarize(f"{config_mode} interleaved (real conv/gdn/attn per layer)", run_pass(config_mode))

    print("\n=== summary ===")
    print(f"default: isolated={results['default_isolated']:.1f}us -> interleaved={results['default_interleaved']:.1f}us ({results['default_interleaved']/results['default_isolated']:.2f}x)")
    print(f"tuned:   isolated={results['tuned_isolated']:.1f}us -> interleaved={results['tuned_interleaved']:.1f}us ({results['tuned_interleaved']/results['tuned_isolated']:.2f}x)")


if __name__ == "__main__":
    main()
