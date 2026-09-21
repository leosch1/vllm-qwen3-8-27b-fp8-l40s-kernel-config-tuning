# 06 — Instance-variance tracer artifact

**Status:** confirmed, resolved. A genuine dead end that had to be ruled
out explicitly rather than assumed away — the kind of finding that's only
"boring" in hindsight.

## What

Investigate why the very first nsys capture of this whole investigation
(`pass1-default-concurrency64`, later dubbed "instance A") showed the tuned
config issuing **~11.85× more GEMM launches** than default for the identical
request workload — an effect large enough, if real, to fully explain the
c=64 regression on its own (net GEMM GPU-time 3.09× higher for tuned despite
each individual call being 3.84× faster). Resolved by redeploying and
capturing a second pod instance ("instance B") for all 4 config×concurrency
cells, then diffing real CUDA API call counts between the anomalous and
normal instances.

## Why

A launch-count multiplier this large needed a mechanism, and the leading
candidate (`SPLIT_K`-style config-driven kernel replication) was directly
checkable and directly false — the tuned JSON configs only set
`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, nothing that
would multiply launches. Before either building a "why tuned issues more
launches" narrative or dismissing the discrepancy as noise, it needed to be
pinned to a specific, checkable cause.

## How

1. **2-replicate matrix**: redeployed default and tuned, captured a second
   pod instance ("instance B") for each of the 4 cells (default/tuned ×
   c=1/c=64), on top of the original ("instance A") captures — 8 captures
   total, all on the same node (`gpu-node-a`), same nsys session.
2. **`cuda_api_sum`**: real CUDA API call counts (`cudaGraphLaunch`,
   `cudaLaunchKernel`, `cuLaunchKernel`, `cuLaunchKernelEx`) — independent of
   any GPU-side kernel-launch *decomposition* — compared between default
   instance A (anomalous) and tuned instance A (normal).
3. **Within-capture ramp-up check**: `cuda_gpu_trace` on instance A, GEMM
   event count bucketed across the capture's 10 equal time windows, to rule
   out a mid-capture warm-up effect.

## Results

Full launch-count matrix:

| | c=64 | c=1 |
|---|---:|---:|
| default, instance A | 45,568 | 10,240 |
| default, instance B | 536,576 (×2 identical repeats, same pod) | 1,310,720 |
| tuned, instance A | 540,160 | 1,310,720 |
| tuned, instance B | 537,088 | 1,310,720 |

Three of the four instances (default-B, tuned-A, tuned-B) converge tightly
regardless of config — within 0.7% of each other at c=64, and *exactly*
1,310,720 at c=1, all three. Only **default instance A** — the very first
pod deployed in this whole investigation — differs, by ~11.8× at c=64 and
~128× at c=1. Repeating the same pod twice (default instance B, c=64) gave
an *exact* match (536,576 = 536,576), confirming this isn't run-to-run
noise within a pod — whatever's responsible is fixed once per pod instance.
This reframed the question entirely: not "why does tuned differ from
default," but "what was different about that one specific first capture."

**Root cause: a tracer decomposition artifact, not a real execution
difference.** `cuda_api_sum` on default instance A vs. tuned instance A:

| CUDA API call | default instance A (anomalous) | tuned instance A (normal) |
|---|---:|---:|
| `cudaGraphLaunch` | 2,314 | 2,190 |
| `cudaLaunchKernel` | 174,756 | 173,076 |
| `cuLaunchKernel` | 84,118 | 83,648 |
| `cuLaunchKernelEx` | 154,246 | 151,754 |

Within 1–6% of each other — ordinary run-to-run noise, nothing like the
11.8× gap in reported GEMM launch count. **The real work done — including
the number of CUDA graph replays — was essentially identical.** The gap
exists entirely in how completely nsys's `--cuda-graph-trace=node` decomposed
each graph replay into visible per-kernel GPU events: ~2,300 graph launches
decomposed into ~20 recorded GEMM events per replay for instance A, vs. ~230
per replay for every other instance. Same graphs, same real content,
different tracer decomposition depth.

Ramp-up check: GEMM event count per time-bucket on instance A was roughly
uniform throughout (2,280–6,144 events/bucket, no ramp-up) — so it wasn't
CUPTI's node-tracing gradually engaging over the capture; whatever
determined the shallower decomposition was fixed for that entire session
from the start.

**Ruled out as causes**: CUDA graph capture-set variability (proven
deterministic via startup logs), a node-persistent JIT/driver cache (checked
directly — no such mount exists), a gradual within-capture warm-up (checked
directly — flat throughout). **Not resolved**: *why* that one specific
capture session got shallower decomposition — would need to diff actual
CUPTI/nsys internals or compiled kernel artifacts to go further, with no
guarantee of a clean answer; not pursued given the cost/benefit.

**What stands, unaffected by any of this**: every real `vllm bench serve`
result — client-side, not GPU-side kernel-launch counting — was consistent
and reproducible across every instance measured (default: 72.57 / 73.12 /
73.27ms at c=64 across 3 instances/runs, 31.74 / 31.90ms at c=1 across 2;
tuned: 73.93 / 73.82ms at c=64 across 2, 24.85 / 24.78ms at c=1 across 2).
Default instance A is excluded entirely from every later analysis in this
project (e.g. [`05-nsys-hardware-profiling-matrix`](../05-nsys-hardware-profiling-matrix/)'s
chart).

## Reproduce

Given a nsys capture, run `cuda_api_sum` and `cuda_gpu_kern_sum` (or the
equivalent `nsys export --type sqlite` tables `CUPTI_ACTIVITY_KIND_RUNTIME` /
`CUPTI_ACTIVITY_KIND_KERNEL`) side by side; a real execution difference
shows up in `cuda_api_sum` (actual API call counts), while a tracer artifact
shows up only in the kernel-level decomposition, not the API call counts
underneath it.
