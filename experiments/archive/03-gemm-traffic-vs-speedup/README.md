# 03 — GEMM traffic vs. speedup

**Status:** confirmed and still stands as one piece of the picture — but it
was a test of a hypothesis (overfitting to tuned batch sizes) that turned out
**not** to be the main mechanism behind the c=64 regression. It correctly
ruled that hypothesis out even after directly confirming its premise, which
is why the investigation kept going into
[`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/).

## What

Instrument the real, deployed `fp8_utils.py` kernel dispatch code to count
every actual GEMM call by its exact `(N, K, M)`, live, during real serving —
then compare that real traffic distribution (which `M` values production
*actually* sends to the kernel, weighted by how often) against the
per-`M` speedup curve from
[`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/).

## Why

[`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/) found a
real but small (~3pp) generalization gap concentrated at specific held-out
`M` values (worst: `M=160`). That only matters for the c=64 regression
(discovered in
[`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/)) if real traffic
actually lands on those weak `M` values. That's directly testable — not
inferred — if we can see the real batch sizes hitting the kernel.

## How

Two counting lines patched directly into the live, deployed `fp8_utils.py`,
right where `M`/`N`/`K` are already computed before the kernel launch —
mounted over the installed file as a diagnostic-only change, never touching
upstream vLLM:

```python
with _M_COUNTS_LOCK:
    _M_COUNTS[(N, K, M)] += 1
```

Counts are dumped periodically to a JSON histogram file readable off the
pod. Captured once per concurrency level (1, 2, 4, 8, 16, 32, 48, 64, 96,
128) plus once for the one-time model-init sweep (warmup + cudagraph
capture — config-independent by construction, since it doesn't depend on
which kernel config is mounted), for both the default and tuned configs
separately.

One important scoping detail: this only counts real Python-level dispatch
calls. Once vLLM captures a CUDA graph, the kernel launch itself is
graph-replayed and never re-enters this Python code path — so this
instrumentation is blind to *that* kernel invocation. It still traces the
scheduler's real per-step batch sizes, since a given step's `M` is
determined before the graph-vs-eager choice.

## Results

Interactive traffic browser (concurrency-selectable, default vs. tuned side
by side): [`chart.html`](./chart.html).

**First reading — wrong in an interesting way.** At the original
`--max-num-seqs=64`, `M=160` (the worst offender from
[`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/)) **never
occurred** — a hard scheduler ceiling (`assert len(self.running) <=
max_num_seqs` in `scheduler.py`) kept decode batches at or below 64. Raising
`--max-num-seqs` to 128 let decode genuinely reach past that ceiling, and
`M=160` started showing up in real traffic, confirmed directly via this same
instrumentation.

**Yet the end-to-end delta barely moved even once the specific weak point
was demonstrably being hit.** The generalization gap this experiment set out
to test turned out not to be the explanation, even after directly confirming
its premise (real traffic does reach the weak `M`) — a genuinely useful
negative result, not a dead end.

### TTFT-specific traffic weighting

The same per-`M` real-traffic weighting, split by phase (decode calls,
`M≤192`, vs. prefill calls, `M>256`) explains part — but only part — of why
TTFT specifically degrades under tuning while TPOT stays comparable
(observed in [`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/)):

| traffic | traffic-weighted mean kernel speedup |
|---|---:|
| decode (`M≤192`) | +23.2% |
| prefill (`M>256`) | +9.1% |

Prefill genuinely benefits less from tuning — but that number is still
positive, so it's not the whole TTFT story either. The kernel itself has no
notion of prefill vs. decode (it just sees an `[M,K]` tensor); the leading
remaining hypothesis at the time was the same contention mechanism from
[`04-contention-simulation`](../04-contention-simulation/), plausibly
amplified for prefill specifically — one large prefill GEMM (M≈2000) holding
its larger, more resource-hungry tuned configuration for one long burst,
colliding with concurrent decode work needing the same SMs, and TTFT
(unlike the isolated kernel benchmark) includes that queueing delay, not
just raw compute time.

## Reproduce

Full diff (this mechanism plus the separate call-plan recorder from
[`10-w8a8-call-plan-recorder`](../10-w8a8-call-plan-recorder/), both merged
into the same file):
[`../10-w8a8-call-plan-recorder/fp8_utils_instrumentation.patch`](../10-w8a8-call-plan-recorder/fp8_utils_instrumentation.patch).
At minimum, patch `_M_COUNTS[(N, K, M)] += 1` into the deployed `fp8_utils.py` right
before the kernel's `grid(META)` closure, mount it over the installed file
(`subPath` ConfigMap mount, requires pod recreation to pick up), then drive
traffic at each concurrency level with `vllm bench serve` (same command as
[`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/)) while reading
back the periodically-dumped JSON histogram.
