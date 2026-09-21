# 07 — GPU-metrics clock throttling

**Status:** confirmed ruled out. No standalone chart exists for this one —
the analysis was done via direct SQLite queries against the raw
`GPU_METRICS` table rather than a visualization; it's folded into the
combined "ruled out" list in
[`../05-nsys-hardware-profiling-matrix/profiling-results/README.md`](../05-nsys-hardware-profiling-matrix/profiling-results/README.md).

## What

Capture live GPU hardware telemetry (clock frequency, SM/tensor
utilization, DRAM read/write bandwidth) alongside the kernel trace, and test
directly whether clock throttling explains why the tuned config's GEMM
kernel regresses at large `M` in real serving.

## Why

By the time [`05-nsys-hardware-profiling-matrix`](../05-nsys-hardware-profiling-matrix/)
and the per-layer reverse-mapping work
([`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/)) had
localized the regression to specific large-`M` launches of one shape
(`gate_up_proj`, N=17408), clock throttling was a natural mechanical
candidate: tuned's config at this shape is structurally heavier
(`BLOCK_SIZE_M=128, num_warps=8, num_stages=3`, 65536B dynamic shared
memory vs. default's `BLOCK_SIZE_M=64, num_warps=4, num_stages=2`, 25088B)
— plausibly enough to push power/thermal limits and trigger a clock dip
specifically during its own slow launches.

## How

Added `--gpu-metrics-devices=all --gpu-metrics-frequency=1000` (reduced from
the 10000Hz default to avoid an Analysis-service OOM) to `nsightToolArgs`
in `ocp-config/argocd-applications/overlays/<cluster>/nsight-operator.yaml`'s
`valuesObject` (GitOps-managed; the vendored chart itself was never
touched). This requires `dcgmi profile --pause` held for the pod's *entire*
lifetime, not just around discrete captures — a real, previously-hit bug:
resuming DCGM mid-pod-lifetime silently revokes the exclusive
hardware-counter resource without crashing anything, breaking only the
metrics stream for later captures on that same pod.

Extracted the raw `GPU_METRICS` and `TARGET_INFO_GPU_METRICS` tables via
`nsys export --type sqlite --tables='.*GPU_METRICS.*'` (POSIX BRE, not SQL
wildcards — `%pattern%` silently produces a 0-byte file). Key gotcha: clock
metrics (`metricId=0` "GPC Clock Frequency [MHz]", `metricId=1` "SYS Clock
Frequency") are stored as raw **Hz** despite the metric name, confirmed by
magnitude against L40S's real ~2520MHz boost clock (divide by 1e6). Other
metrics used: `metricId=10` GR Active, `14` SMs Active, `16` Tensor Active,
`29`/`30` DRAM Read/Write Bandwidth.

Four specific checks, all against the same fixed T0 (a bug from an earlier
pass — mixing two different "t0" definitions, first-kernel-ever vs.
first-GEMM-ever — produced a false "no real gap" claim that had to be
walked back once corrected):
1. A direct slow-cluster-vs-fast-baseline telemetry comparison (10,335
   slow-cluster launches ≥1000µs vs. 499,812 fast-baseline launches <200µs,
   same capture).
2. DRAM read/write bandwidth utilization for the same two populations, to
   check for HBM bandwidth contention (e.g. other concurrent requests'
   KV-cache reads competing with this kernel's weight loads) — something an
   isolated single-kernel benchmark structurally can't see.
3. Clock frequency immediately before/during/after tuned's specific slow,
   large-`M` launches, to distinguish a *pre-existing* throttled state
   (sustained surrounding load) from the kernel's *own* power draw
   triggering it.
4. Pearson correlation between individual launch duration and the clock
   frequency at that moment, within one *fixed* launch config (5,888
   launches, all identical gridX/blockX/registers) — so shape/M isn't a
   confound, and the check reduces to "does clock explain the variance,
   holding everything else fixed."

## Results

**Check 1 — direct telemetry comparison.** Real, measurable clock
throttling coincident with the slow cluster:

| Metric | Slow cluster (≥1000µs) | Fast baseline (<200µs) | Delta |
|---|---:|---:|---:|
| GPC clock | 2376.1 MHz | 2511.6 MHz | −135.5 MHz (−5.4%) |
| Tensor Active | 27.3% | 5.2% | +22.1pp |
| SMs Active | 95.9% | 62.9% | +33.0pp |
| GR Active | 100.0% | 99.9% | ~0 |

A genuine signal (slow cluster's clock distribution is wider and lower —
std 74 MHz, min 2147 MHz — vs. the fast baseline's tight, near-max
distribution — std 25 MHz, close to L40S's ~2520MHz rated boost). But a
5.4% clock deficit would only account for ~5.4% of a compute-bound kernel's
duration — nowhere near the ~2× real-vs-isolated gap being chased. Rules out
"no throttling is happening," but leaves most of the gap unexplained.

**Check 2 — DRAM bandwidth.** Slow cluster shows much higher DRAM *read*
utilization (79.2% vs. 51.99%, +27pp) but *lower* write utilization (5.1%
vs. 14.0%). Real, but mostly just a property of the shape itself (large-M
GEMM naturally reads more weight data per launch) — the isolated benchmark
of the same shape would very likely show similarly high read bandwidth on
its own, so this doesn't distinguish real-serving from isolated either.

**Check 3 — before/during/after clock, the one that actually undercuts the
throttling story.** Clock is *lowest before* the slow kernel even starts
(2295 MHz), *rises during* it (2376 MHz), and *keeps rising after* it ends
(2433 MHz) — a steady recovery trend, not a dip. The opposite of "this
kernel's own power draw throttles itself." These slow-cluster launches tend
to occur during/just-after periods when the GPU has already been under
sustained heavy load for a while (clock still recovering from that), rather
than the kernel itself causing the throttle.

**Check 4 — duration vs. clock, shape held fixed.** With shape/config held
completely fixed, duration is very tight (1930–2138µs, <2% CV) and clock
frequency barely correlates with it at all (Pearson **+0.18** — and what
little correlation exists runs *backwards*: higher clock → slightly
*longer* duration, not shorter).

**The clincher, from repeating checks 1 and 3 on default's own large-M
launches: default shows a much bigger, genuinely local clock dip (−19.8%)**
— yet default is *less* inflated overall (real-vs-isolated) than tuned at
the same shape/M. If clock throttling were driving the regression, the
config with the bigger dip should be the more inflated one. It's the
opposite. Clock throttling doesn't distinguish the two configs at all, let
alone explain why tuned specifically loses here.

**Net conclusion**: the clock-throttling signal is real but appears to be a
*correlate* of "the system was recently under heavy sustained load" rather
than a *cause* of these specific launches being slow — within a fixed
shape, clock explains essentially none of the duration variance. Ruling
this out (and DRAM bandwidth alongside it) was itself useful: it eliminated
the two most obvious mechanical explanations and left the
instruction/constant-cache locality hypothesis (see
[`09-isolated-vs-real-flops-inflation`](../09-isolated-vs-real-flops-inflation/))
as the most plausible remaining lead.

This also confirmed two other candidate mechanisms directly, using the same
telemetry: concurrent kernel contention (single-stream serialization
confirmed — zero true kernel-level concurrency on one device, both from the
trace data and a GUI screenshot) and NCCL overlap (`Overlap` metric = 0.0
throughout the capture).

## Reproduce

```bash
nsys export --type sqlite --tables='.*GPU_METRICS.*|.*KERNEL.*|StringIds' capture.nsys-rep
```
Then join `GPU_METRICS.rawTimestamp` against `CUPTI_ACTIVITY_KIND_KERNEL.start/end`
for the launches of interest, dividing clock `value` columns by 1e6 to get
real MHz.
