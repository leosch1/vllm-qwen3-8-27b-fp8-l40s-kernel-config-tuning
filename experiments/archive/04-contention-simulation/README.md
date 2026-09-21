# 04 — Contention simulation

**Status:** actually run, with real, important results — **correcting an
earlier draft of this writeup**, which wrongly assumed the script was never
executed because no separate output log file existed for it (the run's
output was captured directly into the chart below instead). The specific
mechanism it exercises (a separate OS process competing for the same GPU)
was later shown to differ from the exact mechanism inside a single serving
process (see the reconciliation at the end) — but its central result,
tuned specifically regressing at large `M` for one shape, is real and lines
up with the eventual root cause found later in this project.

## What

Test, directly and causally, whether the tuned config's kernel-level
advantage shrinks or reverses when the GPU is busy with other concurrent
work, by rerunning the exact same default-vs-tuned kernel-level comparison
as [`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/) while
a separate background process continuously saturates the GPU with unrelated
matmuls.

## Why

By this point, [`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/)
had found a real, growing regression under load, and
[`03-gemm-traffic-vs-speedup`](../03-gemm-traffic-vs-speedup/) had just ruled
out overfitting-to-tuned-batch-sizes even after confirming real traffic does
reach the suspect `M` value. Contention was the next candidate: the tuned
configs' own launch parameters, compared against default right around where
the regression starts, use roughly double the concurrent warps and pipeline
depth versus default's fixed `num_warps=4, num_stages=2` — rational when you
have the whole GPU to yourself, but exactly the kind of choice that could
claim more of a shared, finite resource pool once something else is running.
This also fit a shape the simpler "GEMM is just a shrinking share of step
time" theory didn't: the regression kept worsening from c=48→64→96→128
rather than leveling off, which contention scaling *with* load fits better
than a fixed floor.

## How

`compare_under_contention.py` reruns `compare_default_vs_tuned.py`'s exact
comparison, at the same 18 batch sizes, 500 iterations/point, while a
separate OS process (`_noisy_neighbor.py`, launched via `subprocess.Popen`
— a real separate process, not a thread, so it isn't starved by the GIL and
more closely matches how real contention happens: vLLM's TP workers are
themselves separate processes) continuously runs an unrelated 4096×4096×4096
dense matmul on the same GPU, forever, until killed.

```bash
python3 compare_under_contention.py   # starts _noisy_neighbor.py itself, in the background
```

A second layer of analysis, built on top of this run plus the real per-call
`_M_COUNTS` histogram from
[`03-gemm-traffic-vs-speedup`](../03-gemm-traffic-vs-speedup/): does
batch-mix × per-kernel timing data actually *predict* the real end-to-end
TPOT gap seen in [`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/)?
`predicted tuned TPOT = actual default TPOT − (predicted default GEMM time −
predicted tuned GEMM time)`, with GEMM time weighted by the real per-decode-step
call multiplicities for the 5 shapes — **64/48/16/64/64**, not "1 call of
each shape" as a first, corrected-away pass assumed — which match the
model's real architecture (64 total layers, 48 linear-attention + 16
full-attention, confirmed against `/mnt/models/config.json`).

## Results

Full interactive detail (per-shape speedup, absolute "contention tax", the
TPOT-prediction panels, and raw data tables): [`chart.html`](./chart.html).

**The tuned config for `N=17408` (`gate_up_proj`) goes negative under
synthetic contention, specifically at large `M`:**

| M | speedup, isolated | speedup, under contention |
|---:|---:|---:|
| 256 | +12.7% | +10.6% |
| 512 | +14.7% | **−32.1%** |
| 1024 | +10.5% | **−18.8%** |
| 1536 | +10.6% | **−10.0%** |
| 2048 | +12.9% | **−4.6%** |
| 3072 | +12.2% | +1.3% |
| 4096 | +9.7% | −1.9% |

At `M=512` specifically, tuned's absolute time under contention (47.2µs)
is actually *slower* than default's (35.7µs) — a full reversal, not just a
shrinking win. This is the one shape and roughly the one `M` range
(`M≳500`) that the rest of this project eventually converges on as the real
root cause of the c=64 production regression — found here first, via a
synthetic reproduction, before the later in-session profiling work traced
it to real serving traffic directly.

The other 4 shapes stay positive at every point under contention (see the
per-shape panels in the chart) — the effect is concentrated on `N=17408`,
same as everywhere else in this project.

**Does this predict the real TPOT gap?** Once corrected to the real
64/48/16/64/64 per-step call multiplicities, GEMM goes from a negligible
~0.1% of TPOT (the naive "1 call per shape" assumption) to a real ~10-20%,
and the predicted-tuned-TPOT curve moves into a comparable range to the
actual measured gap (mean absolute error 2.8ms isolated-based, 2.6ms
contention-based, both vs. the real per-concurrency TPOT from
[`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/)). It still
undershoots specifically at `C=1` (predicts ~0.3ms of the actual ~7ms TPOT
drop) — direct per-call compute savings alone don't explain the very-low-concurrency
win either, so something beyond raw GEMM time is at play there too, not
just at the high-concurrency end.

## Reconciling with the later, in-session finding

Later profiling work in this project (real `nsys` GPU-metrics + kernel-trace
capture of the live server — see
[`05-nsys-hardware-profiling-matrix`](../05-nsys-hardware-profiling-matrix/))
established that the real CUDA execution model on this deployment is a
**single stream, single context per device** — kernels inside one vLLM
process serialize strictly, with zero true kernel-level *simultaneity*, ruling
out same-stream kernel overlap as the mechanism. That doesn't contradict this
experiment: `_noisy_neighbor.py` runs as a genuinely separate OS process with
its own CUDA context, competing for the same physical SMs, L2 cache, and
DRAM bandwidth via GPU time-slicing across contexts — a real hardware
contention mechanism distinct from "two kernels executing at the same
instant in one stream." The fact that this cruder, external form of
contention reproduces the same shape/`M` signature that real serving
(no synthetic noisy neighbor needed) was later found to independently
produce, is a striking piece of corroborating evidence that shared hardware
resource pressure (from *something* — other TP rank, attention kernels,
NCCL, cache/bandwidth pressure) is at least part of what's really going on
for `N=17408` specifically — even though the ultimate mechanism, established
in [`09-isolated-vs-real-flops-inflation`](../09-isolated-vs-real-flops-inflation/),
turned out not to require an external process at all.

## Reproduce

```bash
python3 compare_under_contention.py
```
Compare its output against `compare_default_vs_tuned.py`'s (isolated)
numbers for the same batch sizes and shapes.
