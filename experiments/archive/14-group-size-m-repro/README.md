# 14 — GROUP_SIZE_M surgical reproduction

**Charts:** [`../15-groupfix-e2e-validation/report.html`](../15-groupfix-e2e-validation/report.html) --
visual summary of this experiment's mechanism finding alongside `15`'s e2e
validation attempt.

**Status:** built and run; **the mechanism.** Changing exactly one config
parameter (`GROUP_SIZE_M`, tuned's own value only) while holding every other
parameter fixed eliminates tuned's cold-weight-cycling DRAM-Read sensitivity
entirely -- confirmed directly via matched GPU_METRICS, cross-checked
against the exact duration-level prediction made in advance. This is the
most direct, surgical evidence in the whole investigation, and it doesn't
require any external contention, KV traffic, or exotic hardware mechanism:
it's a single, well-documented Triton grid-swizzle parameter that the
autotuner picked blind to its real-world cost.

## What

Diffing `default`'s and `tuned`'s full config dicts (rather than continuing
to chase external-environment hypotheses per `11`-`13`) surfaces one
parameter that stands out from the rest: `GROUP_SIZE_M`. Unlike
`BLOCK_SIZE_M`/`num_warps`/`num_stages` (genuine compute-shape/parallelism
choices), `GROUP_SIZE_M` is Triton's L2-cache-locality grid-swizzle knob --
straight out of Triton's own matmul tutorial, and confirmed directly in this
kernel's own source
(`vllm/model_executor/layers/quantization/utils/fp8_utils.py:772-780`):

```python
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

At M=2048 for this shape (`gate_up_proj`, N=17408, K=5120):

| | `BLOCK_SIZE_M` | `GROUP_SIZE_M` | `num_warps` | `num_stages` |
|---|---:|---:|---:|---:|
| default | 64 | **32** | 4 | 2 |
| tuned | 128 | **1** | 8 | 3 |

default's `BLOCK_SIZE_M=64` gives `num_pid_m = ceil(2048/64) = 32` -- exactly
equal to its own `GROUP_SIZE_M`. That means default groups its *entire* M
range under one fixed weight-tile: only a single ~0.64MB slice of `B` needs
to be resident at a time, reused by up to 32 concurrently-scheduled blocks,
before the grid moves to the next slice. tuned's `GROUP_SIZE_M=1` disables
grouping entirely -- plain row-major order, spreading concurrently-active
blocks across many different `B`-tiles at once, needing far more of the
89MB matrix simultaneously resident to get any reuse at all.

Checking the full tuned-config M-grid for this exact shape: every M-bucket
at and above 512 landed on `GROUP_SIZE_M=1` (256→32, 512→**1**, 1024→**1**,
1536→**1**, 2048→**1**, 3072→**1**, 4096→**1**). (Checked the other three
GEMM shapes in this deployment too -- noisier, no such clean collapse, so
this may be specific to this shape's dimensions rather than a universal law
across every shape this autotuner has ever tuned.)

**Confirmed directly by reading the actual tuning script** (not just
inferred from the config grid): `benchmarks/kernels/benchmark_w8a8_block_fp8.py`
is the source of these JSON files (its own docstring: *"Tune triton w8a8
block fp8... Then copy to model_executor/layers/quantization/utils/configs"*,
and it builds the exact `N=...,K=...,device_name=...,dtype=fp8_w8a8,block_shape=[...].json`
filename). `GROUP_SIZE_M` genuinely *is* swept -- candidates
`[1, 16, 32, 64]`, crossed with `BLOCK_SIZE_M∈{16,32,64,128,256}`,
`BLOCK_SIZE_N∈{32,64,128,256}`, `BLOCK_SIZE_K∈{64,128}`, `num_warps∈{4,8}`,
`num_stages∈{2,3,4,5}` -- 1280 configs, exhaustively timed per `(M,N,K)`
(plain Python loop, no Triton `@autotune`, no pruning heuristic). The blind
spot is in `tune()` itself (`benchmark_w8a8_block_fp8.py:202-230`):

```python
A_fp32 = ((torch.rand(M, K, ...) - 0.5) * 2 * fp8_max)  # ONE A, generated once
B_fp32 = ((torch.rand(N, K, ...) - 0.5) * 2 * fp8_max)  # ONE B, generated once
...
for config in tqdm(search_space):        # all 1280 configs
    kernel_time = benchmark_config(A, B, As, Bs, ...)   # same A, B every time
```

`A`/`B` are generated **once per `(M,N,K)`** and reused across **all 1280
candidate configs** in that sweep -- never re-randomized per config, never
cleared or flushed. By the time any given config's 5 warmup launches run,
`B` has already been touched by potentially hundreds of *other* candidates'
launches too -- there is no way for this harness to ever observe a cold-cache
condition. `GROUP_SIZE_M`'s entire benefit (reusing an L2-resident
weight-tile across concurrently-scheduled blocks) is structurally invisible
to it; only its small, real, fixed overhead (extra div/mod per thread block
for the swizzle remap) is ever measured. Neither benchmark script mentions
L2, cache, or locality anywhere (`grep`'d both, zero matches) -- `GROUP_SIZE_M`
is brute-forced as an opaque integer knob, with no special-casing based on
tensor footprint vs. L2 size. So `GROUP_SIZE_M=1` isn't a coverage gap or
tie-breaking noise -- it's the **correct answer to the wrong question**: a
genuine train/serve mismatch between the tuning objective (one tensor,
timed 1280 different ways, always warm) and real serving's actual access
pattern (many distinct tensors, contended L2).

One precision this surfaces: the real search space only offers
`{1, 16, 32, 64}` for `GROUP_SIZE_M`, not the `num_pid_m`-matched values
used below and in `15` for every M bucket (4, 8, 12, 16, 24, 32 for
M=512..4096). At M=2048 specifically, `16` was already a real candidate the
tuner tried and rejected -- directly comparable to what a corrected tuning
run would have chosen. The other buckets' patched values are a reasoned
extrapolation of the same principle, not literally values the tuner ever
evaluated.

**Does `tuned` actually beat `groupfix` under the tuner's own (idealized,
always-warm) measurement conditions -- i.e. is the autotuner's choice
self-consistent with what it actually measured?** Directly measured
below (`same-B` condition, the closest match to the tuner's own
fixed-tensor methodology): `tuned` (1219.5µs) beats `groupfix`
(1245.5µs) by **2.1%**. **Yes, as expected** -- this is a mechanism
sanity-check, not a claim about true isolated/steady-state kernel
performance: it confirms the autotuner's `GROUP_SIZE_M=1` pick is
reproducible and internally consistent with the same kind of
single-fixed-tensor, always-warm environment the tuning script itself
uses, which is exactly why the tuner picked it.

> **Clarification (from `16`): this does not mean `GROUP_SIZE_M=1` is
> actually faster in a real, well-isolated, steady-state sense --
> quite the opposite.** `16` measured a genuinely different question
> (true isolated performance under a much more rigorously isolated,
> no-condition-switching methodology, via `ncu`) and found `GROUP_SIZE_M=1`
> clearly *losing* to grouped values at M=1024, with every hardware
> counter agreeing. The two results are complementary, not contradictory:
> `tuned` wins here because this experiment deliberately reproduces the
> tuner's own idealized/always-warm testing conditions, and `tuned`
> loses in `16` because that experiment removes those idealized
> conditions and measures real cache-locality cost instead. That *is*
> the train/serve mismatch this experiment set out to demonstrate. One
> honest caveat on the precise magnitude: `16` also found that merely
> round-robinning between differently-*compiled* kernel variants (as this
> experiment's `same-B` comparison does, switching between
> `default`/`tuned`/`groupfix` every 250-iteration block) measurably
> dilutes a true effect's size relative to a fully isolated, no-switching
> measurement -- so the exact "2.1%" figure may run a bit low or high of
> what a `16`-style single-condition-only re-test at M=2048 would show,
> even though the qualitative direction (tuned wins under tuner-like
> conditions) is expected to hold. See
> `../16-group-size-m-exhaustive-sweep/README.md` ("Implication for
> experiment 14") for the full reasoning.

## Why

`12`'s cold-weight-cycling test already showed tuned's DRAM Read nearly
doubling under cold-cycling (12%→22%) while default stays flat -- real,
correctly-directioned, but well short of real serving's 72.8%. `13` ruled
out KV-cache traffic as the missing ingredient. Rather than continuing to
add more external-environment complexity, this experiment asks the more
basic question: does the config difference *by itself*, isolated from any
environmental factor, already explain the asymmetry `12` found? If so, no
further external mechanism is needed to explain *why* cold-cycling affects
tuned and not default -- only, perhaps, to explain the remaining gap to real
serving's full magnitude.

## How

`group_size_m_repro.py` -- same design as `12` v2 (single process, single
CUDA stream, `gate_up_proj` M=2048, 64 distinct `B` copies, round-robin
counterbalanced 250-iteration blocks × 16 repeats), extended with a THIRD
config: **`groupfix`** -- tuned's exact M=2048 config with *only*
`GROUP_SIZE_M` changed from 1 to 16 (tuned's own `num_pid_m` at
`BLOCK_SIZE_M=128`, M=2048 -- the value that makes one group cover all of M,
mirroring default's own strategy of `GROUP_SIZE_M == num_pid_m`).
`BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `num_warps=8`, `num_stages=3` --
every other tuned parameter -- stay untouched. 3 configs × 2 weight-states
(same/cold) = 6 conditions, same nsys-injected/DCGM-paused/SCC-elevated pod
pipeline as `09`/`12`/`13`.

Because `GROUP_SIZE_M` doesn't affect grid/block dimensions, registers, or
shared memory (only the pid→tile *mapping*), `tuned` and `groupfix` launch
with an *identical* kernel-launch signature (`gridX=2176`, 250 regs,
`blockX=256`, 64KB dynamic shared memory) -- unlike `13`'s KV-touch marker,
there's no distinguishing shape to classify samples by. Conditions were
instead recovered by **position**: kernels were tagged by grid-family
(`gridX=4352`=default vs. `gridX=2176`=tuned-or-groupfix), repeat-cycle
boundaries detected via the family transition back to default (start of a
new round-robin cycle), and each cycle's 1500 kernels split into six
consecutive 250-kernel sub-blocks in the script's fixed round-robin order
(`default-same, default-cold, tuned-same, tuned-cold, groupfix-same,
groupfix-cold`) -- verified directly against the data (250-kernel blocks
recovered cleanly for every complete repeat; only the first/last partial
repeats, from profiling starting/stopping mid-cycle, were shorter and
excluded from that check, not from the metric aggregation).

## Results

Duration (n=4000/condition):

| | same-B | cold (64 copies) | shift |
|---|---:|---:|---:|
| default | 1356.9µs | 1381.2µs | +1.80% |
| tuned | 1219.5µs | 1281.6µs | +5.10% |
| **groupfix** | 1245.5µs | 1231.3µs | **-1.14%** |

Trend check (first-half vs. second-half block means): all six conditions
drift +0.19% to +1.73% (one outlier at -1.19%), similar order of magnitude
throughout -- no condition-specific confound.

`groupfix`'s cold-cycling shift (-1.14%, i.e. flat/noise) is *smaller* than
default's own (+1.80%) and far below tuned's (+5.10%) -- while `groupfix`'s
same-B duration (1245.5µs) stays close to tuned's (1219.5µs, +2.1%), keeping
almost all of tuned's speed advantage over default (1356.9µs, -8.2%).

**Direct GPU_METRICS confirmation** (matched to the exact same launches,
via the positional classification described above):

| | Tensor Active (same→cold) | DRAM Read (same→cold) | DRAM Write (same→cold) |
|---|---|---|---|
| default | 56.6%→56.7% (flat) | 10.0%→9.3% (**flat**) | 7.3%→7.3% (flat) |
| tuned | 58.5%→55.3% (-3.2pp) | 11.9%→**22.4%** (**nearly 2x**) | 8.0%→7.8% (flat) |
| **groupfix** | 58.8%→58.8% (**flat**) | 10.1%→**10.2%** (**flat**) | 8.1%→8.2% (flat) |

This is as clean as this investigation has produced anywhere. `groupfix`'s
DRAM Read under cold-cycling (10.19%) is not just lower than tuned's
(22.38%) -- it's flatter than *default's own* same→cold shift (10.03%→9.29%),
and its same-B Tensor Active (58.79%) is essentially identical to tuned's
own same-B value (58.52%), confirming `groupfix` keeps tuned's other
speed-relevant properties untouched. Changing exactly one parameter, holding
everything else fixed, took tuned's cold-cycling-sensitive signature and
made it disappear.

## What this does and doesn't establish

**Does establish:** `GROUP_SIZE_M` is a real, sufficient, single-parameter
cause of the specific asymmetry `12` found (tuned sensitive to cold-weight
cycling, default not) -- not an artifact of `12`'s methodology, not a
side-effect of `BLOCK_SIZE_M`/`num_warps`/`num_stages` differing, and not
requiring any external contention mechanism to explain *that* asymmetry.
Given the autotuning JSON shows `GROUP_SIZE_M=1` consistently for every
large-M bucket of this exact shape, and the autotuning benchmark
methodology structurally cannot distinguish grouping choices under an
always-warm, single-tensor benchmark -- **confirmed directly by reading the
actual tuning script** (see above: one fixed `A`/`B` pair reused across all
1280 candidate configs, never re-randomized, never cleared), not just
inferred -- this is a fully coherent, mechanistic explanation requiring no
new hardware phenomenon: **the autotuner picked a value for a cache-locality
parameter that its own benchmark harness was blind to, and that value
turned out to be the worst possible one for real serving's actual access
pattern.**

**Doesn't (yet) establish (at the time this section was written):** that
`GROUP_SIZE_M` alone explains the *full* real-serving magnitude (72.8%
DRAM Read). This experiment reproduces `12`'s cold-cycling scale (~22%),
and shows that scale is fully attributable to `GROUP_SIZE_M` rather than
the other config differences -- but `12`'s own result was already known to
be well short of real serving's magnitude. Whether real serving's larger
effect is the *same* mechanism intensified by real contention or involves
something additional was still open at this point.

**Update from `15`:** the e2e question above (does the fix survive contact
with real serving traffic, at real serving's scale) is now answered,
independently of the DRAM-Read-magnitude question -- `groupfix` never
regresses relative to `default` anywhere across a full concurrency sweep
(1-128) on the real predictor, and closes an *increasing* gap against plain
`tuned` as concurrency rises (+7.4% at c=128, right where the original
regression concentrated). Whatever the remaining gap to 72.8% DRAM Read
represents mechanistically, it doesn't stop the config-level fix from
working end-to-end.

**Update from `16`:** the follow-up empirical test this section originally
called for -- an exhaustive isolated-kernel sweep across all M values and
`GROUP_SIZE_M` candidates -- was built and run, and went through its own
correction cycle before landing on a trustworthy answer. Its first pass
appeared to confirm `GROUP_SIZE_M=1` as a real, non-noise isolated-kernel
winner at M=512-3072; a follow-up `ncu` hardware-counter probe flatly
contradicted that at M=1024 (`GROUP_SIZE_M=1` clearly *losing*, with
consistent DRAM-bytes/L2-hit-rate/throughput evidence), and a decisive
single-M re-test traced the discrepancy to a methodological artifact in
the sweep's own round-robin design (cross-M cold-start contamination),
not a real effect.

Net result, and how it fits with this section's own `same-B` result
above: `16` answers a different question than the `same-B` comparison
does. `same-B` asks "does the tuner's choice hold up under the tuner's
*own* idealized, always-warm testing conditions" -- yes, as expected,
confirming the mechanism is self-consistent. `16` asks "does
`GROUP_SIZE_M=1` actually perform better in a real, rigorously isolated,
no-condition-switching sense" -- no: grouped values genuinely outperform
`GROUP_SIZE_M=1` once cross-condition measurement contamination is
removed, matching textbook cache-locality theory. Both are true at once,
and together they sharpen rather than undercut this experiment's core
claim: the autotuner didn't make an unreasonable choice given what it
measured (`same-B` confirms that), but what it measured was itself an
idealized, always-warm regime that both differs from real serving's
contended-cache access pattern *and*, per `16`, doesn't even hold up as
"faster in isolation" once measured with a methodology that isn't
vulnerable to the same kind of sequential-single-pass bias the tuning
script itself uses. See `16`'s README for the full argument.

## Reproduce

```bash
# copy group_size_m_repro.py, ../../../_vendored_matmul_timing.py, and this
# shape's tuned-config JSON onto an nsys-injected, SCC-elevated pod (see 09's
# "Reproducing the GPU-metrics-enabled isolated capture"). Note: the
# placeholder's nsys wrapper holds the exclusive GPU-metrics device lock as
# soon as it's launched (even under --start-later=true, before profiler-start
# is ever called) -- profiler-start then profiler-stop it once to release
# that lock before launching the real script, exactly as in 13.
cd /tmp && python3 group_size_m_repro.py
```
