#!/usr/bin/env python3
"""Minimal script for ncu profiling -- NOT for timing (that's what
cold_weight_cycling_repro_v2.py is for). Deliberately tiny: ncu's
Application Replay mode reruns this WHOLE script once per hardware-counter
pass, so keeping total runtime small matters a lot here.

Layout, in execution order (each is one distinct kernel launch of
`_w8a8_triton_block_scaled_mm`, targeted individually via ncu's
--kernel-name + --launch-skip/--launch-count):

  0-4:   default warmup on B[0]                 (JIT/warmup, not profiled)
  5:     default, B[0] again (HOT -- just read)  <- "default-same" probe
  6-68:  default reads B[1..63] in turn           (evicts B[0] from L2)
  69:    default, B[0] again (COLD -- evicted)   <- "default-cold" probe
  70-74: tuned warmup on B[0]                     (JIT/warmup, not profiled)
  75:    tuned, B[0] again (HOT)                 <- "tuned-same" probe
  76-138:tuned reads B[1..63] in turn              (evicts B[0] from L2)
  139:   tuned, B[0] again (COLD)                 <- "tuned-cold" probe

So launch indices (0-based, among matching-kernel-name launches only) are:
5, 69, 75, 139 for same/cold x default/tuned respectively.
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
NUM_WEIGHT_COPIES = 64

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


def call(A, As, B, Bs, config):
    return w8a8_block_matmul(A, B, As, Bs, [BLOCK_N, BLOCK_K], config, OUT_DTYPE)


def main():
    device_name = get_device_name_as_file_name()
    json_path = (
        f"{TUNED_DIR}/N={N},K={K},device_name={device_name},"
        f"dtype=fp8_w8a8,block_shape=[{BLOCK_N},{BLOCK_K}].json"
    )
    with open(json_path) as f:
        tuned_configs = {int(k): v for k, v in json.load(f).items()}
    tuned_config = tuned_configs[M]

    A, As = make_A(M, K)
    B_copies = make_B_copies(NUM_WEIGHT_COPIES, N, K)
    torch.cuda.synchronize()

    for config, label in [(DEFAULT_CONFIG, "default"), (tuned_config, "tuned")]:
        for _ in range(5):
            call(A, As, *B_copies[0], config)          # warmup (HOT, indices 0-4 / 70-74)
        call(A, As, *B_copies[0], config)               # SAME probe (index 5 / 75)
        for i in range(1, NUM_WEIGHT_COPIES):
            call(A, As, *B_copies[i], config)           # evict B[0] (indices 6-68 / 76-138)
        call(A, As, *B_copies[0], config)               # COLD probe (index 69 / 139)
        torch.cuda.synchronize()

    print("done")


if __name__ == "__main__":
    main()
