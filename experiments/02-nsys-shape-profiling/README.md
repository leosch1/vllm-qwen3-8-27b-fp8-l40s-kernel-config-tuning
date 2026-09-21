# 02 — nsys shape profiling

Supports [blog §3 "What is responsible for the performance degradation?"
and §4 "Isolated benchmark and real serving aren't measuring the same
thing"](../../blog/index.html).

## What

Capture real GPU hardware-counter kernel traces from the actual live
predictor pod under actual client load (not an isolated microbenchmark),
via NVIDIA Nsight Systems, at two concurrency levels (1 and 64) for both
the default and tuned configs. From those traces:

1. Map every `_w8a8_triton_block_scaled_mm` kernel launch to which of the
   5 real weight-matrix shapes it was, and a coarse batch-size (`M`) band.
2. Histogram launch durations, split by (shape, band), to see whether the
   tuned config's win holds up per-shape under real traffic, not just on
   average.
3. Average `GPU_METRICS` hardware counters (Tensor Active, DRAM Read) for
   the specific kernel/shape under investigation.

## Why

`01-first-measurement` found a real end-to-end regression at high
concurrency, but that measurement can't say *which* GEMM shape or batch
size is responsible — it's one aggregate number across the whole model.
Hardware-counter traces from the real, running server can.

## How

**Capture** — the serving process needs to run under `nsys profile`. On
this project's cluster that's handled by the NVIDIA Nsight Operator: a
mutating webhook rewrites the pod at admission time so it starts under
`nsys profile` automatically — nothing about the vLLM launch command
itself changes. Two things still need to be set up separately, neither of
them on that command line:

- **`dcgmi profile --pause`** on the node's DCGM pod, held for the entire
  capture pod's lifetime (not just around the capture window) — DCGM
  holds the GPU's exclusive hardware-counter resource otherwise, and
  `nsys` fails outright. Resume with `dcgmi profile --resume` once done.
- **`--cuda-graph-trace=node`** configured into the operator's own
  `nsightToolArgs` (a one-time operator setting, not a per-run flag).
  Without it, nsys's default CUDA-graph handling collapses each vLLM
  decode step (CUDA-graph-captured by default) into one opaque
  placeholder-named launch, hiding every real GEMM/attention kernel
  underneath.

```bash
dcgmi profile --pause
# deploy the serving pod with profiling enabled (operator-specific: an
# annotation or CR field the webhook watches for) and drive client load
# against it (vllm bench serve, same as 01) -- the capture itself needs
# no manual nsys invocation.
dcgmi profile --resume
```

Without an equivalent operator, the same capture is one manual command:
`nsys profile --cuda-graph-trace=node --gpu-metrics-device=all --output=capture.nsys-rep <the serving launch command>` —
DCGM still needs pausing first regardless of which route starts `nsys`.

**Analysis** — export the capture to SQLite, then extract one row per
`_w8a8_triton_block_scaled_mm` launch (duration, and the `N`/`K`/`M` its
grid dimensions imply):

```bash
nsys export --type sqlite --tables='.*KERNEL.*|StringIds' capture.nsys-rep
# query the resulting .sqlite for _w8a8_triton_block_scaled_mm launches,
# recover N/K (fixed per weight, listed in build_histograms.py's SHAPES)
# and M (from the launch's grid size) per row, write to launches.csv
python3 build_histograms.py launches.csv --label c64_tuned
```

`build_histograms.py` does the binning/categorization step this project's
analysis applied — same 69 log-spaced duration bins, same 5-shape and
6-band mapping — so its output lines up directly with
`results/duration-histograms.json`, which holds the actual 4 histograms
(`c1_default`, `c1_tuned`, `c64_default`, `c64_tuned`) the blog's §3/§5
charts render.

The Tensor Active / DRAM Read numbers (`results/hardware-counters.json`)
come from the same captures' `GPU_METRICS` stream, averaged over the
`gate_up_proj` launches specifically — no separate script, a direct mean
of that counter column filtered to the kernel/shape in question.

## Result

At **c=1**, the whole duration distribution shifts left under the tuned
config — mean drops 27% (100.6µs → 73.3µs).

At **c=64**, most of the distribution still shifts left, but one cluster
shifts sharply *right*: `gate_up_proj` launches around `M≈2000` nearly
double, ~1100µs → ~2000µs. That single cluster is enough to flip the
overall mean from an improvement to a regression (139.8µs → 143.5µs).

Averaged hardware counters for that specific kernel/shape:

| | Tensor Active | DRAM Read |
|---|---:|---:|
| isolated (`01`) | 58.2% | 14.9% |
| real serving | 28.3% | **72.8%** |

DRAM Read jumps 5x and Tensor Active roughly halves between the isolated
benchmark and real serving — the same kernel, computing the same matrix
multiply, is memory-bound in production in a way it never is in
isolation. That's the mechanism [`04-l2-flush-retune`](../04-l2-flush-retune/)
chases down.

## Files

- `build_histograms.py` — the launch-duration binning/categorization
  script (see "Analysis" above for its expected CSV input).
- `results/duration-histograms.json` — the 4 histograms behind blog §3's
  and §5's charts.
- `results/hardware-counters.json` — the Tensor Active/DRAM Read table
  behind blog §4.
