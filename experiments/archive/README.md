# Archive

The full research history behind [`../`](../) — every experiment run
during this investigation, not just the ones the blog cites. Flat, in
roughly chronological order. Raw `.nsys-rep`/`.sqlite` captures are
excluded (multi-GB each; the numbers they produced are in each folder's
own README/results); everything else is kept as originally written.

Four folders here overlap with the curated set and say so at the top:
`20`, `23`, `26`, `27` point back to `../0N-.../` for the parts already
documented there.

For a narrated, single-page walkthrough of all 35 with every diagram
inline, see [`overview.html`](./overview.html).

| # | Experiment |
|---:|---|
| 01 | [Isolated kernel benchmark](./01-isolated-kernel-benchmark/) |
| 02 | [End-to-end concurrency sweep](./02-e2e-concurrency-sweep/) |
| 03 | [GEMM traffic vs. speedup](./03-gemm-traffic-vs-speedup/) |
| 04 | [Contention simulation](./04-contention-simulation/) |
| 05 | [nsys hardware-profiling matrix](./05-nsys-hardware-profiling-matrix/) |
| 06 | [Instance-variance tracer artifact](./06-instance-variance-tracer-artifact/) |
| 07 | [GPU-metrics clock throttling](./07-gpu-metrics-clock-throttling/) |
| 08 | [Per-layer shape mapping](./08-per-layer-shape-mapping/) |
| 09 | [Isolated-vs-real FLOPs inflation](./09-isolated-vs-real-flops-inflation/) |
| 10 | [W8A8 call-plan recorder](./10-w8a8-call-plan-recorder/) |
| 11 | [Interleaved-kernel cache-locality reproduction](./11-interleaved-cache-locality-repro/) |
| 12 | [Cold-weight-cycling reproduction](./12-cold-weight-cycling-repro/) |
| 13 | [KV-cache-traffic reproduction](./13-kv-cache-traffic-repro/) |
| 14 | [GROUP_SIZE_M surgical reproduction](./14-group-size-m-repro/) |
| 15 | [GROUP_SIZE_M e2e validation](./15-groupfix-e2e-validation/) |
| 16 | [GROUP_SIZE_M exhaustive sweep](./16-group-size-m-exhaustive-sweep/) |
| 17 | [Cache-aware autotune rerun](./17-cache-aware-autotune-rerun/) |
| 18 | [out_proj / down_proj order-based disambiguation](./18-out-down-proj-order-disambiguation/) |
| 19 | [Cache-aware config e2e validation](./19-cache-aware-config-e2e-validation/) |
| 20 | [Cache-aware config isolated kernel benchmark](./20-cache-aware-isolated-kernel-benchmark/) — parts 1-10; part 11 in `../04/` |
| 21 | [groupfix real-serving nsys capture](./21-groupfix-real-serving-nsys-capture/) |
| 22 | [cache-aware config real-serving nsys capture](./22-cache-aware-real-serving-nsys-capture/) |
| 23 | [M=2048 single-anchor GROUP_SIZE_M fix](./23-m2048-singlefix/) — fully in `../03/` |
| 24 | [CUDA-graph dispatch histogram](./24-cudagraph-dispatch-histogram/) |
| 25 | [L2-flush autotune rerun](./25-l2-flush-autotune-rerun/) (generalizing the cache-aware fix; RHOAI-stack run, superseded for the blog by `29`) |
| 26 | [L2-flush isolated kernel benchmark](./26-l2-flush-isolated-kernel-benchmark/) — parts 1, 3, 4; part 2 in `../05/` |
| 27 | [L2-flush e2e validation](./27-l2-flush-e2e-validation/) — fully in `../06/` |
| 28 | [GROUP_SIZE_M order-bias probe](./28-group-size-m-order-bias-probe/) |
| 29 | [L2-flush autotune rerun, v0.27.1](./29-l2-flush-autotune-rerun-v0271/) — fully in `../04/` |
| 30 | [Instrumented tuner candidate times](./30-instrumented-tuner-candidate-times/) |
| 31 | [Original tuner full sweep, v0.27.1](./31-original-tuner-full-sweep-v0271/) |
| 32 | [Single-M=2048 isolated](./32-single-m-2048-isolated/) |
| 33 | [Eviction sweep](./33-eviction-sweep/) |
| 34 | [Knee vs. B size](./34-knee-vs-b-size/) |
| 35 | [Residency map](./35-residency-map/) |

## Known gaps

- One link in `11-interleaved-cache-locality-repro/README.md` points at
  `vllm/UPSTREAM_BUG_benchmark_config_avg.md` — a file in a separate
  `vllm` fork clone, outside this repo. Pre-existing, not something this
  archiving pass could resolve.
