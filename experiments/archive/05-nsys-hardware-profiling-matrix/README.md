# 05 — nsys hardware-profiling matrix

**Status:** confirmed. This is the point where the investigation moved from
inference (isolated benchmarks + estimation) to direct measurement of the
real, deployed server.

## What

Capture real GPU hardware-counter kernel traces from the actual live
predictor pod under actual client load, via the NVIDIA Nsight Operator
(`components/nsight-operator`), across a 2×2×2 matrix: config
(default/tuned) × concurrency (1/64) × pod instance (2 replicates each, 8
captures total) — replacing estimation with direct measurement.

## Why

[`04-contention-simulation`](../04-contention-simulation/)'s "predicted vs.
actual TPOT" analysis was built from *isolated microbenchmark* GEMM times
weighted by real batch-mix counts — an estimate, not a direct measurement.
The next step was to capture actual hardware-counter kernel-time data from
the real predictor pod under real client load, so GEMM's real share of GPU
time (and the tuned-vs-default delta) could be cited directly instead of
inferred.

## How

`nsys profile` is injected into the predictor's `kserve-container` process
by the Nsight Operator's mutating webhook, result streamed to its own
MinIO-backed cloud storage. Two blocking prerequisites, both real and
non-obvious:

- **`dcgmi profile --pause`/`--resume` on the node's DCGM pod, held for the
  entire pod lifetime** (not just around the discrete capture window) — DCGM
  holds the GPU's exclusive hardware-counter resource otherwise and
  `nsys`/`ncu` fail outright. Discovered the hard way: resuming DCGM
  mid-pod-lifetime silently revokes the resource without crashing anything,
  breaking only the metrics stream for all subsequent captures on that pod.
- **`--cuda-graph-trace=node`** in `nsightToolArgs` (GitOps-managed via
  `ocp-config/argocd-applications/overlays/<cluster>/nsight-operator.yaml`'s
  `valuesObject`, never the vendored chart directly). Without it, nsys's
  default CUDA-graph handling collapses each vLLM decode step (CUDA-graph-captured
  by default) into one opaque placeholder-named launch, hiding every real
  GEMM/attention kernel underneath.

Per-run artifact: `capture.nsys-rep` (git-ignored, large — also durable in
Nsight Operator's own cloud storage, re-fetchable via `nsight_operator.py
download --session <id> --collection <id>`) plus a `kernel_summary.csv`
(the `cuda_gpu_kern_sum` per-kernel-name breakdown: `Count`, `mean_us`,
`sum_ms`, `pct_of_kernel_time`) and the matching `vllm bench serve`
`bench-result.txt`.

## Results

Full raw data and per-run breakdowns:
[`profiling-results/README.md`](./profiling-results/README.md) (this
project's primary running writeup) and its
`pass{1,2}-{default,tuned}-concurrency{1,64}[-v2]/` subdirectories. Interactive summary of the "reliable" instances (excludes
the anomalous instance — see
[`06-instance-variance-tracer-artifact`](../06-instance-variance-tracer-artifact/)):
[`chart.html`](./chart.html).

| Run | Mean TPOT | Total kernel time | Wall clock | GPU-busy ratio | GEMM share |
|---|---:|---:|---:|---:|---:|
| default, c=64 | 72.57ms | 70.74s | 83.64s | 42.3% | 35.4% (25.04s) |
| default, c=1 | 31.74ms | 13.41s | 84.60s | 7.9% | 10.0% (1.34s) |
| tuned, c=64 | 73.93ms | 168.36s | 87.42s | 96.3% | 45.9% (77.32s) |
| tuned, c=1 | 24.85ms | 127.62s | 67.39s | 94.7% | 75.3% (96.05s) |

(TP=2 — every "total kernel time"/"GEMM time" figure is summed across both
GPU devices, confirmed near-identical per-device via
`rank_stats_by_device.parquet`, so it can exceed wall clock by up to 2x;
GPU-busy ratio divides by `2 × wall clock` to normalize back to a genuine
per-GPU average.)

Real e2e TPOT at c=64 is essentially flat, slightly *worse* on tuned
(72.57ms → 73.93ms) — confirming
[`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/)'s finding directly
from hardware counters, not just client-side timing. Per GEMM call, tuned's
kernel genuinely is much faster (549.6µs → 143.1µs mean at c=64, in the
instance measured first) — the tuning itself works at the individual-kernel
level, exactly as [`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/)
found. What erases that win in aggregate turned out **not** to be a
launch-count multiplication (that read was a tracer artifact — see
[`06-instance-variance-tracer-artifact`](../06-instance-variance-tracer-artifact/))
but a duration-distribution tail specific to one shape at large `M` — see
[`09-isolated-vs-real-flops-inflation`](../09-isolated-vs-real-flops-inflation/).

**Kernel-time breakdown, GEMM excluded, confirms this is a single-kernel
effect, not general contention:** at c=64, GEMM's per-kernel delta
(tuned − default) is +2,187.6ms — over 11× larger than the next-biggest
kernel's delta (`silu_and_mul`, −189.5ms); at c=1, GEMM's delta is
−35,816.2ms, over 1,000× larger than anything else. NCCL AllReduce, the
attention/linear-attention kernels, quantization kernels — none of them move
meaningfully under tuned, at either concurrency. Not a general "everything
contends harder" effect; isolated almost entirely to the one kernel actually
being tuned.

**GPU-busy ratio** sits in the same ~95–98% band for both configs at *both*
concurrencies — c=1 is not meaningfully less saturated than c=64. Tuned's
busy ratio at c=1 (94.7–95.0%) is actually very slightly *lower* than
default's (96.1%) — the opposite of "tuned fills the GPU better." This
metric is normalized by wall-clock time: tuned finishes the identical real
work in *less* wall-clock time (each launch is genuinely faster), not
because it keeps the GPU busier over a similarly-long window. A single
decode stream (c=1) is a tight sequential dependency chain either way — very
little true idle time to fill in *either* config, so the efficiency gain
shows up as shorter total duration, not higher busy percentage.

**SM occupancy** (derived from real captured launch configs — grid/block
dims, registers/thread — against L40S/Ada Lovelace SM limits, not a live
occupancy counter): at c=64, default is flat at 16.7%; tuned's dominant case
(67% of launches) reaches 33.3% — roughly double, from more warps and a
modest register-count drop. At c=1 the gap is bigger: tuned's dominant case
(99% of launches) reaches 83.3% — 5× default, not 2×, which can't come from
doubled warps alone (caps at 2×) — needs the register count to also collapse
(224→48 registers/thread, 4.7×) so 5 whole 8-warp blocks fit per SM instead
of 2, traced to a genuinely smaller tile (`BLOCK_SIZE_M=16` vs. default's
fixed `64`) rather than just more warps.

## Reproduce

```bash
source /tmp/nsight-cli-venv/bin/activate   # python3.12 venv w/ nsight_operator.py deps
cd /private/tmp/nsight-operator-resources_v26.3.1
python3 nsight_operator.py download --session <session id> --collection <collection id> --output-dir <dir>
```
Then `nsys export --type sqlite --tables='.*KERNEL.*|StringIds'` on the
`.nsys-rep` for local SQLite/pandas analysis (faster and more memory-safe
than the Analysis service's server-side `nsys recipe` jobs, which have a
genuine, reproducible memory-scaling bug unrelated to input size).
