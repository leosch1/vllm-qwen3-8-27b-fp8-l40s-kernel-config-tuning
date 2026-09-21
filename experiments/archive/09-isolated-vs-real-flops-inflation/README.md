# 09 — Isolated-vs-real FLOPs inflation

**Status:** confirmed. This is the project's core finding — the answer to
"why does c=64 regress" that everything from
[`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/) onward was
chasing, and the point where "concurrency" gets explicitly demoted from
cause to mediator.

**Caveat found later, now corrected via a direct hardware-counter
re-measurement.** The isolated-benchmark values below (`*_iso_us`) come from
vLLM's own `benchmark_config()`, which has a real upstream bug — it divides
its final average by `num_iters * 10` instead of `num_iters`, silently
under-reporting every isolated value by ~10x (see
`vllm/UPSTREAM_BUG_benchmark_config_avg.md` in the vllm fork repo, and
[`11-interleaved-cache-locality-repro`](../11-interleaved-cache-locality-repro/)
for how this was found). The **ratio** numbers originally on this page
(`*_ratio`, the "17.1x"/"8.85x" headline figures) combined that deflated
isolated value with an independent, correctly-measured real-serving value —
so they were overstated by roughly the same ~10x.

Corrected by rerunning `isolated_all_shapes.py` unmodified but with real
`nsys` hardware-counter capture (`--gpu-metrics-devices=all
--gpu-metrics-frequency=1000`, same method as
[`05`](../05-nsys-hardware-profiling-matrix/)/[`07`](../07-gpu-metrics-clock-throttling/))
instead of trusting `benchmark_config()`'s self-reported average — CUPTI's
kernel start/end timestamps don't go through the buggy division at all, so
this sidesteps the bug rather than patching it. Real hardware duration for
`gate_up_proj`, `M=2048`, both configs, 6,015 launches each:

| | isolated (nsys hardware-measured) | `benchmark_config()` self-reported | true/reported ratio |
|---|---:|---:|---:|
| default | 1302.3µs | 135.2µs | 9.63x |
| tuned | 1169.1µs | 122.1µs | 9.57x |

Both ~9.6x, confirming the documented bug's ~10x magnitude directly (residual
gap from CUPTI's measurement window vs. the internal CUDA-event timer, not a
second bug). Isolated speedup (tuned vs. default) barely moves either way —
10.2% here vs. the originally-reported +9.7% — exactly as expected, since a
same-function self-timing bug cancels in a same-function ratio.

**What changes: the real-vs-isolated inflation is not what the original
(buggy) numbers said.** Using these corrected isolated values against the
same real-serving durations already established
(default 1179.5–1183.2µs, tuned 2057.1–2057.3µs across the two capture
pairs):

| | real | isolated (corrected) | real/isolated |
|---|---:|---:|---:|
| default | ~1181µs | 1302.3µs | **0.91x — real is *faster* than isolated** |
| tuned | ~2057µs | 1169.1µs | **1.76x — real is 76% slower than isolated** |

Not "both configs inflate, tuned inflates more" (the original ~17x/~9x
framing) — **default shows no real-vs-isolated inflation at all** (if
anything, slightly faster in real serving); **tuned shows a genuine, large,
tuned-specific inflation.** This is a sharper result than the original,
not just a corrected magnitude: it rules out any explanation that would
predict *both* configs degrading from isolated to real (general system load,
generic contention) and narrows it specifically to whatever tuned's heavier
tile config does differently under real conditions. See the
`isolated-gpumetrics-capture/` subdirectory for the raw capture
(`full.sqlite`, an nsys-generated database with `CUPTI_ACTIVITY_KIND_KERNEL`
and `GPU_METRICS`) and
[`profiling-results/README.md`](../05-nsys-hardware-profiling-matrix/profiling-results/README.md)'s
Tensor/DRAM-utilization section for what closes this loop further — the
same isolated capture's Tensor Active/DRAM Read numbers.

## What

Directly compare, at the *identical* shape and *identical* batch size `M`,
the same GEMM's duration measured two ways: (a) the isolated microbenchmark
([`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/)'s
method, alone on an otherwise idle GPU) and (b) its real duration as
captured from actual live serving traffic
([`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/)'s
reverse-mapped per-launch durations). Then test whether GEMM "heaviness"
(FLOPs) predicts how large that real-vs-isolated gap gets, across all 5
real shapes at two `M` values (1 and 2048).

## Why

[`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/) had already
localized the regression to `gate_up_proj` (N=17408) at `M≳640`. But that
alone doesn't say *why* — the natural next question is whether it's simply
because this is the "biggest" GEMM in the model (most FLOPs), which would
predict any sufficiently large GEMM should show the same effect, or whether
it's specific to something about this shape's tuned config in particular.

## How

**FLOPs formula**: `2×M×K×N` for `C(M,N) = A(M,K) @ Bᵀ(K,N)`.

**Isolated measurement**: `isolated_all_shapes.py` (all 5 shapes at
M=2048) / `isolated_all_shapes_m1.py` (M=1) — both call the same vendored
`w8a8_block_matmul`/`benchmark_config` functions as
[`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/), run live
on the same pod for a fair hardware comparison (requires DCGM paused, same
as any hardware-counter work on this cluster).

**Real measurement**: durations reverse-mapped from the same captures
[`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/) used
(`pass1-default-concurrency64-v2` / `pass2-tuned-concurrency64-v2` and their
c=1 counterparts), for the exact same shape/M combinations measured
in isolation.

**Two normalizations**, since ratio alone isn't clean (tuned's own ratio for
`gate_up_proj` is actually *higher* at M=1, 25.6×, than at M=2048, 17.4×,
because the isolated baseline itself grows with M too):
- **Ratio**: real duration ÷ isolated duration.
- **Absolute difference**: real duration − isolated duration (µs).

## Results

Interactive detail across all 10 points (5 shapes × 2 M values, ~11,600×
span in FLOPs): [`chart.html`](./chart.html).

**The headline number, corrected** (see the caveat box at the top for the
original, bug-inflated version) — the specific launch signature this whole
GPU-metrics investigation converged on (`gate_up_proj`, `M∈(1920,2048]`):
corrected isolated duration 1169.1µs vs. this same launch's real-serving
duration of ~2057.3µs — a **1.76× inflation for tuned**, vs. default's own
equivalent large-M launch at the same shape showing **no inflation at all**
(~1181µs real vs. 1302.3µs isolated — real is if anything slightly faster).
**Not "both configs suffer, tuned suffers more" — only tuned inflates from
isolated to real at this shape/M. Default doesn't, at all.**

**FLOPs does not cleanly predict the effect.** Reading both the ratio and
absolute-difference panels together: the qualitative finding
(`gate_up_proj`'s tuned config is the outlier, not "big GEMMs in general")
holds under both normalizations — reassuring, since it isn't an artifact of
which normalization you pick. But with only 5 shapes — and `N` and `K`
varying together rather than independently across them — this doesn't
establish a precise functional law relating FLOPs to the effect. It shows
one large, repeatable anomaly at the one shape with by far the largest `N`,
and a roughly consistent, much smaller effect everywhere else. Honestly
inconclusive on the general question ("do heavier GEMMs inflate more,
categorically") even though the specific finding (this one shape, this one
`M` range) is solid.

**Mechanisms ruled out** for the inflation itself (see
[`07-gpu-metrics-clock-throttling`](../07-gpu-metrics-clock-throttling/)):
concurrent kernel contention (single-stream serialization confirmed — zero
true kernel-level concurrency on one device), NCCL overlap (`Overlap` metric
= 0.0 throughout), pre-launch scheduling/queueing gaps (negligible), clock
throttling as a *cause* (real but small for tuned, wrong-signed correlation,
and default shows a *bigger* local dip while being *less* inflated overall —
so clock doesn't distinguish the two configs either).

**Confirmed, refined form of the cache-locality hypothesis: tuned is
compute-bound in isolation and memory-bound in real serving; default is
compute-bound in both.** The isolated capture above has `--gpu-metrics`
enabled, so Tensor Active / DRAM Read can be measured for the *identical*
launches whose durations are quoted above — directly comparable to the real
capture's numbers already in
[`profiling-results/README.md`](../05-nsys-hardware-profiling-matrix/profiling-results/README.md):

| | Tensor Active (isolated) | Tensor Active (real) | DRAM Read (isolated) | DRAM Read (real) |
|---|---:|---:|---:|---:|
| default | 55.9% | 45.5% | 9.2% | 15.8% |
| tuned | 58.2% | 28.3% | 14.9% | **72.8%** |

In isolation, both configs look similar and clearly compute-bound (Tensor
Active ~56–58%, DRAM Read ~9–15%) — tuned isn't structurally memory-hungrier
than default here, contrary to what its heavier tile
(`BLOCK_SIZE_M=128, num_warps=8, num_stages=3`, 65536B dynamic shared memory
vs. default's `BLOCK_SIZE_M=64, num_warps=4, num_stages=2`, 25088B) might
suggest on its own. Only in real serving does tuned's DRAM Read jump ~5x to
72.8% (highest-utilized resource in that launch, by a wide margin) while its
Tensor Active roughly halves — a genuine compute-bound-to-memory-bound
transition that happens *only* for tuned and *only* in real serving. This is
consistent with (not yet a full mechanistic proof of) the original
cache-locality framing: default barely touches DRAM either way, so it's
largely insensitive to whatever DRAM/L2 state a *different* kernel type
running immediately before it (real serving) vs. an identical repeat of
itself (isolated) leaves behind. Tuned's heavier, deeper-pipelined design is
apparently fine when the memory subsystem answers instantly (isolated,
nothing else has ever touched it) but exposed exactly when it doesn't —
which is what a cold cache/row-buffer state from a preceding, structurally
different kernel would produce. `ncu` stall-reason data would still be the
direct proof (same two blockers as before — replay-mode
self-contamination and unconfirmed tool availability/permissions — remain
unaddressed), but this closes the isolated-vs-real half of the loop that
was previously untested.

## The conclusion this project converges on

**Concurrency has no direct causal role.** It's purely a mediator —
continuous batching under load pushes the scheduler's real batch size `M`
into the 640–2048 range, and it's specifically *real-serving* GEMM calls at
that `M` range, for this one weight shape, that degrade in a way the
identical isolated kernel benchmark at the identical `M` never shows. This
supersedes the contention-centric framing from
[`02-e2e-concurrency-sweep`](../02-e2e-concurrency-sweep/) and
[`04-contention-simulation`](../04-contention-simulation/) — real, but not
the root cause; a corroborating symptom at best.

## Reproduce

Both scripts below are included in this folder; each depends on the
repo-root `_vendored_matmul_timing.py` (same helper
[`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/) uses) and
that shape's tuned-config JSON from `./tuned-configs`:

```bash
python3 isolated_all_shapes.py      # M=2048, all 5 shapes
python3 isolated_all_shapes_m1.py   # M=1, all 5 shapes
```
Compare against the same shape/M's real duration, reverse-mapped per
[`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/)'s method from
a live capture.

### Reproducing the GPU-metrics-enabled isolated capture

`isolated-gpumetrics-capture/` (`capture.nsys-rep` + `full.sqlite`) was
produced by rerunning the exact same `isolated_all_shapes.py` above, but
inside an nsys-injected pod with `--gpu-metrics-devices=all
--gpu-metrics-frequency=1000`, instead of trusting `benchmark_config()`'s
self-reported average — same method
[`05`](../05-nsys-hardware-profiling-matrix/)/[`07`](../07-gpu-metrics-clock-throttling/)
use for real serving, just pointed at a one-off job instead of the live
predictor:

1. **A throwaway Job**, pinned to the cluster's idle secondary GPU node (not
   the live predictor's node), labeled `nvidia-nsight-profile: enabled`
   with its container named `kserve-container` (the injection webhook's
   `containerIncludePatterns` only matches that literal name), and running
   as the `qwen3-8-27b-fp8-pr-evidence-profiler` ServiceAccount (grants
   `SYS_ADMIN` + root via the `qwen3-8-27b-fp8-pr-evidence-profiling` SCC —
   without it, `nsys` fails with `ERR_NVGPUCTRPERM` even with DCGM paused;
   DCGM-pause and this SCC are two independent, both-required blockers, not
   redundant with each other).
2. `dcgmi profile --pause` on that node's DCGM pod first (same requirement
   and same gotcha as everywhere else in this project — must happen before
   the pod starts, and stay paused for its whole lifetime).
3. `nsight_operator.py profiler-start` / run the script via `oc exec` /
   `profiler-stop`, then `dcgmi profile --resume`. The CLI's own
   `download --session <id>` only fetches a whole session (all historical
   collections, large) with no per-collection filter in this version — much
   faster in practice to `oc cp` the `.nsys-rep` straight off the pod's own
   `/tmp/s3_cache/` (same trick already used once in
   [`profiling-results/README.md`](../05-nsys-hardware-profiling-matrix/profiling-results/README.md)),
   and to export locally using the same pod's own `nsys` binary (already
   present on any nsight-injected pod at
   `/mnt/nv/bin/nsight-systems/bin/nsys` — no local `nsys` install needed).
   `nsys export --type sqlite --tables=...` silently produced a 0-byte file
   for unclear reasons in this environment; `nsys stats` (which builds a
   complete, unfiltered SQLite as an internal step, left behind next to the
   `.nsys-rep`) worked fine and was used directly instead.
4. Local tooling (`ngc registry resource download-version`,
   `nsight_operator.py`, its `python3.12` venv) lives in `.nsight-cli/`
   at this repo's root — gitignored, not committed, machine-specific. The
   `ngc` CLI's own bundled certifi bundle failed TLS verification for
   this project's network path; worked around by fetching the resource
   files directly via `curl` against the NGC API's file-redirect endpoints
   instead of the `ngc` CLI.
