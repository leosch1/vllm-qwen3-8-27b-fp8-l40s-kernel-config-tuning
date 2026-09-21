# 12 — Cold-weight-cycling reproduction

**Status:** built and run; real, correctly-directioned, but partial
reproduction, confirmed directly via matched DRAM-Read/Tensor-Active
metrics (v2). Confirms the mechanism is real and contributes in the right
direction; rules out "this alone fully explains the effect." A follow-up
`ncu` stall-reason attempt narrows it further: the effect needs *sustained*
repeated streaming to appear -- a single isolated cache miss shows zero
measurable cost for either config, ruling out "simple per-access cache-hit-
vs-miss latency" as the direct mechanism.

## What

Test a sharper, more specific version of the cache-locality hypothesis than
`11-interleaved-cache-locality-repro` tested: does forcing every GEMM call
to read a genuinely *different* weight tensor (as real serving's 64 distinct
transformer layers do, each with their own `gate_up_proj` weight matrix)
reproduce the tuned-specific real-serving slowdown -- without any kernel-type
interleaving, multi-request scheduling, or CUDA graphs?

## Why

`09-isolated-vs-real-flops-inflation`'s corrected numbers (see its README)
and `05`'s Tensor-Active/DRAM-Read table established that tuned's
`gate_up_proj` launch is compute-bound in isolation and memory-bound only in
real serving. `11`'s two designs tested whether interleaving *other kernel
types* between repeats of `gate_up_proj` reproduces this -- both came back
negative. But those designs, as best can be determined, kept calling
`gate_up_proj` on the *same* weight tensor throughout, varying only which
other kernels ran in between. That's not actually the variable real serving
changes: real serving's 64 layers each have their *own* distinct ~89MB
weight matrix, so no single-shape-fixed-tensor test could have caught this.
This experiment isolates that one specific variable directly.

## How

`cold_weight_cycling_repro.py` -- single process, single CUDA stream, one
GEMM shape (`gate_up_proj`, N=17408, K=5120, M=2048), reimplementing
`_vendored_matmul_timing.benchmark_config()`'s CUDA-event-bracketing
methodology directly (not calling that function -- it has the documented
`/10` bug, and more fundamentally its `run()` closure can't vary the weight
tensor per iteration).

- **A** (activations) is created once and reused throughout -- real serving's
  activations do vary somewhat too, but they're much smaller (10.5MB vs
  `B`'s 89.1MB) and contribute far less L2 pressure, so holding it fixed
  isolates the variable that matters.
- **64 distinct copies of `B`** (matching the real model's actual layer
  count) are pre-allocated with independent random data -- 5.7GB total,
  trivial on a 48GB card. `B`'s single-copy size (89.1MB) is almost exactly
  the L40S's L2 cache size (96MB), so any copy is fully evicted by the other
  63 (~5.6GB) long before it's reused.
- Two conditions per config, same 2000-iteration/5-warmup methodology both
  times: **same-B** (every iteration hits `B_copies[0]`, matching
  `isolated_all_shapes.py`'s existing methodology) and **cold** (iteration
  `i` uses `B_copies[i % 64]`, guaranteeing a genuine cold fetch every time
  by construction).
- Run inside the same nsys-injected, DCGM-paused, SCC-elevated pod pipeline
  built for `09`'s isolated-GPU-metrics capture -- see that experiment's
  "Reproducing the GPU-metrics-enabled isolated capture" section for the
  full recipe (unchanged here).

## Results (v1 -- since superseded by v2 below, kept for the record)

| | same-B (baseline) | cold (64 distinct copies) | shift |
|---|---:|---:|---:|
| default | 1296.0µs | 1289.3µs | -0.5% |
| tuned | 1135.9µs | 1209.7µs | +6.5% |

v1 ran each condition as one long fixed-order block (default-same,
tuned-same, default-cold, tuned-cold) with no order control, and its
capture had no `GPU_METRICS` table at all (confirmed genuinely absent, not
an export-tool bug). Real question raised after this run: was default's
apparent -0.5% "improvement" a real effect, or just noise/order drift given
no counterbalancing? See v2.

## v2: counterbalanced ordering + longer capture + direct DRAM confirmation

`cold_weight_cycling_repro_v2.py` fixes both gaps: (1) splits each
condition's budget into 250-iteration blocks and round-robins through all 4
conditions 16 times (`default-same, default-cold, tuned-same, tuned-cold`,
repeated), so any monotonic drift over the run affects all 4 conditions
roughly equally instead of biasing whichever ran later; (2) roughly doubles
total iterations (4000/condition vs. 2000) after v1's capture -- which
processed only 229,231 events vs. `09`'s successful capture's 1,001,228 --
came back with no `GPU_METRICS` table at all, suggesting a
duration/buffer-flush threshold. That fix worked: v2's capture (58.9MB
sqlite vs. v1's 13.3MB) does contain `GPU_METRICS`.

**Trend check confirms v1's -0.5% was noise, not signal.** All four
conditions show a similar, small drift across the run (+1.2% to +2.25%,
first-half vs. second-half block means) -- a real but modest and *uniform*
effect (mild thermal drift, not boost-clock ramp, since it's a slowdown
not a speedup), not something that favors one condition over another. With
proper counterbalancing:

| | same-B | cold (64 copies) | shift |
|---|---:|---:|---:|
| default | 1370.0µs | 1369.3µs | **-0.05%** |
| tuned | 1224.0µs | 1288.4µs | **+5.26%** |

Isolated speedup narrows from +10.7% (same-B) to +5.9% (cold) -- consistent
with v1's +12.4%→+6.2%, same pattern, now on solid methodological footing
(n=4000/condition, order-balanced, drift-checked).

**Direct DRAM-metric confirmation, matched to the exact same launches these
durations come from:**

| | Duration (same→cold) | Tensor Active (same→cold) | DRAM Read (same→cold) |
|---|---|---|---|
| default | 1317.7→1316.0µs (flat) | 55.9%→55.8% (flat) | 9.22%→9.19% (**flat**) |
| tuned | 1171.3→1233.1µs (+5.3%) | 58.2%→54.5% (-3.6pp) | 11.96%→**22.46%** (**nearly 2x**) |

This directly answers the "are we sure the weights were actually being
fetched from DRAM?" question v1 couldn't: yes, for tuned -- DRAM Read
genuinely nearly doubles under cold-cycling, with a matching real drop in
Tensor Active and the same duration increase measured three independent
ways (timing, Tensor Active, DRAM Read) in the same direction. Default is a
clean null on *every* metric, not just duration -- no sensitivity to weight
cycling at all.

**Still confirmed as real and contributing, still confirmed as
insufficient alone.** Tuned's cold-cycling DRAM Read (22.46%) is well short
of real serving's 72.8% -- cold weight cycling explains part of the gap
(isolated same-B's 11.96% baseline is itself already nonzero, per `09`'s
L2-capacity math; cold-cycling roughly doubles it; real serving is over 3x
higher still). Something else in real serving -- genuine multi-request
scheduling/queueing dynamics, the combination of cold weights *and*
kernel-type diversity together rather than either alone, or something not
yet identified -- accounts for the remaining gap. This narrows `11`'s open
conclusion further rather than resolving it: three independent
single-process tests (11's two, this one) have each confirmed one candidate
mechanism contributes something real without fully reproducing the effect.

## `ncu` stall-reason attempt: a genuine surprise, narrows the mechanism further

`09`'s two blockers for `ncu` (replay-mode self-contamination against a
*live* process, and unconfirmed tool availability) turned out to both be
resolvable for a *synthetic, self-contained* repro like this one:
`nsight-compute` isn't bundled by the nsight-operator injection (it only
ships Nsight Systems), but it installs cleanly via `dpkg -i` (not
`apt-get install`, which fails under this SCC's dropped capabilities --
`apt`'s internal privilege-drop-to-`_apt` sandboxing needs capabilities the
profiling SCC doesn't grant) from the same NVIDIA devtools apt repo already
used for `nsys`. And since this is our own standalone script rather than
the live predictor, `--replay-mode application` (rerun the whole program
per hardware-counter pass, avoiding the same-kernel-back-to-back-replay
warming problem `09` flagged) is fully practical -- no need to attach to a
continuously-serving process at all. Deployed as a plain (non-nsys-injected
-- the nsys injection wraps every exec'd command, which would conflict with
`ncu` wanting the same CUPTI hardware-counter path) job under the same
profiling SCC + DCGM-pause prerequisites.

`ncu_probe.py`: a minimal script (not for timing -- `cold_weight_cycling_repro_v2.py`
already does that) with exactly 4 launches of interest, targeted individually
via `--kernel-name regex:_w8a8_triton_block_scaled_mm --launch-skip N
--launch-count 1`: read `B[0]` warm (right after touching it 5 times in a
warmup loop) vs. read `B[0]` cold (right after touching `B[1..63]` once each
-- 5.6GB streamed, more than enough to evict `B[0]`'s 89MB from a 96MB L2 by
pure capacity), for both configs.

**Result: no measurable difference at all, for either config.**

| | Duration | Memory Throughput | DRAM Throughput |
|---|---:|---:|---:|
| default-same | 1.13ms | 69.36% | 17.53% |
| default-cold | 1.12ms | 68.77% | 17.65% |
| tuned-same | 1.10ms | 46.85% | 39.79% |
| tuned-cold | 1.10ms | 46.91% | 39.77% |

Grid sizes confirmed correct (`(4352,1,1)` / `(2176,1,1)` matching default/
tuned exactly), so this isn't a targeting bug. This directly contradicts
what the large-scale (4000-iteration) test clearly showed for tuned
(DRAM Read 11.96%→22.46%, matched by a real duration increase) -- meaning
something about *how* cold was induced differs between the two tests in a
way that matters.

**What's different: a single isolated cache miss vs. sustained, rapid,
repeated streaming through many large tensors.** `ncu_probe.py`'s "cold"
condition is one genuine cache miss, surrounded by otherwise-normal
activity. The large-scale test's "cold" condition is thousands of
back-to-back reads continuously cycling through 64 different ~89MB tensors,
sustained across the entire measurement window. That a single isolated miss
costs nothing for either config, while the sustained version clearly costs
tuned something real, points the mechanism somewhere more specific than "is
this one read a cache hit or miss" -- more likely something at the memory
*controller* level that only manifests under sustained pressure (DRAM
row-buffer/bank-level contention, queue depth building up under continuous
rapid-fire access to many different addresses), not a per-access latency
difference that a single before/after snapshot could catch. Genuinely
useful to know, not a dead end: it rules out "simple cache-hit-vs-miss
latency" as the direct mechanism and redirects toward something that needs
sustained load to appear -- consistent with real serving being exactly that
(continuous, sustained request processing), unlike either of `09`'s single
before/after style tests.

Not yet tried: profiling a launch *embedded within* an ongoing sustained
cycle (e.g. launch index ~130 of a continuous run through the 64 copies,
rather than one clean isolated before/after) -- would test whether a
*short* sustained burst is enough to expose the effect at the single-launch
level, without needing the full 4000-iteration scale.

## Reproduce

```bash
# copy cold_weight_cycling_repro_v2.py, ../../../_vendored_matmul_timing.py,
# and this shape's tuned-config JSON onto an nsys-injected, SCC-elevated pod
# (see 09's "Reproducing the GPU-metrics-enabled isolated capture"), then:
python3 cold_weight_cycling_repro_v2.py

# for the ncu stall-reason attempt: same tuned-config JSON, plus
# ncu_probe.py and _vendored_matmul_timing.py, onto a PLAIN (non-injected)
# pod under the qwen3-8-27b-fp8-pr-evidence-profiler SCC with DCGM paused --
# install nsight-compute via `dpkg -i` (not apt-get, see above) from
# https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/,
# then:
ncu --replay-mode application \
    --kernel-name regex:_w8a8_triton_block_scaled_mm \
    --launch-skip <N> --launch-count 1 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section WarpStateStats \
    --export /tmp/ncu_report --force-overwrite \
    python3 ncu_probe.py
# launch indices (0-based, matching-kernel-name only): 5/69 = default same/cold,
# 75/139 = tuned same/cold -- see ncu_probe.py's docstring for the full layout.
```

(`cold_weight_cycling_repro.py`, the v1 script, is kept for the record but
superseded -- its fixed-order, no-drift-check design is why v2 exists.)
