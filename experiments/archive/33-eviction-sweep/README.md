# 33 — Eviction sweep

## What

`eviction_sweep.py` measures how much L2 residency `GROUP_SIZE_M=1`
(row-major) actually depends on, directly, without a profiler: write `X`
MiB to a scratch buffer between the GEMM and the timed launch (displacing
roughly `X` MiB of least-recently-used cache content), and sweep `X` from
0 to 160 MiB. `gate_up_proj` at M∈{1024,2048}, `GROUP_SIZE_M`∈{1,16,32,64},
base config otherwise fixed at the winning family
(`BLOCK_SIZE_M/N/K=128, num_warps=8, num_stages=3`). Conditions
round-robined within each repeat to cancel session-level thermal/clock
drift.

Prediction being tested: row-major needs ~90MiB resident (all of `B`,
~85MiB, plus slack), so should degrade almost immediately as eviction
grows; grouped needs only ~5MiB (one `A` row-block), so should stay flat
until eviction approaches ~90MiB. If instead both degrade together at the
same `X`, the residency theory is wrong and the warm-cache advantage comes
from something else (bandwidth, latency-hiding, occupancy).

## Result

At M=1024, `GROUP_SIZE_M=1` wins at `evict=0` (+0.00% vs. best) and
`evict=4` (+0.00%), but is already behind by `evict=8` (+2.1%) and keeps
degrading monotonically to +46.6% at `evict=80-160`. The grouped values
(16/32/64) stay essentially flat (within a few percent of each other)
across the entire 0–160MiB sweep. Same pattern at M=2048: `GROUP_SIZE_M=1`
wins only at `evict∈{0,4}`, then loses increasingly from `evict=8` onward
(+26.6% at `evict=128`). Full per-`(GROUP_SIZE_M, evict_MiB)` means:
[`exp32-results/summary.json`](./exp32-results/summary.json). Raw
per-iteration data: `exp32-results/eviction_sweep_raw.jsonl`.

The crossover happens far earlier (single-digit MiB) than the naive ~90MiB
prediction, but the qualitative asymmetry is exactly as predicted:
row-major loses its advantage almost immediately under any eviction
pressure, grouped orderings don't. [`34-knee-vs-b-size`](../34-knee-vs-b-size/)
follows up on the exact crossover point.

## Files

- `eviction_sweep.py` / `eviction-sweep-job.yaml` — the sweep script and
  its Job.
- `exp32-results/eviction_sweep_raw.jsonl` — raw per-iteration data.
- `exp32-results/summary.json` — per-`(M, evict_MiB)` means and win margins.
- `exp32-run.log` — full run transcript, including the formatted table.
