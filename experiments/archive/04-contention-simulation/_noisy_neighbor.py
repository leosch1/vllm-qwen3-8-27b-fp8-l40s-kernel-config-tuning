#!/usr/bin/env python3
"""
Standalone background load for compare_under_contention.py. Runs as a
separate OS process (not a thread in the same process as the measurement
loop) so it isn't starved by the GIL while the main process is busy timing
kernels -- separate processes sharing one GPU is also a closer match to how
real contention happens: vLLM's TP workers are separate processes, not
threads.

Just repeats one large dense matmul on the GPU, forever, until killed.
"""

import sys

import torch

M = N = K = int(sys.argv[1]) if len(sys.argv) > 1 else 4096

a = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

while True:
    torch.matmul(a, b)
