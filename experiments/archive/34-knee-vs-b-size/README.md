# 34 — Knee vs. B size

## What

Two-part follow-up to [`33-eviction-sweep`](../33-eviction-sweep/), which
found a sharp eviction "knee" for `GROUP_SIZE_M=1` at `gate_up_proj`
M=1024, but only indirectly implies which tensor (`B`) is responsible.

- **Part A**: reruns the eviction sweep at `gate_up_proj`
  (N=17408) with `num_stages` set to both 3 and 4, to check whether the
  discrepancy between `33`'s cold-vs-warm gap (826 vs 570µs, a 256µs gap)
  and experiment 16's `ncu` measurement (581 vs 520µs, a 61µs gap) is
  explained by `num_stages` differing between the two (the real tuner's
  actual M=1024 winner uses `num_stages=4`; `33` used 3).
- **Part B**: reruns the same eviction sweep at 4 different `N` values —
  N=17408 (`gate_up_proj`, real shape), N=8192 (`in_proj_qkvz`, real
  shape), and two synthetic controls (N=12288, N=4096) chosen to spread
  `B`'s size evenly. If `B`'s retention is what matters, the measured
  eviction "knee" should track `B`'s predicted size
  (`L2 − (B + one A row-block + one C row-block)`) — moving from ~6MiB
  predicted at N=17408 (`B`=85MiB) up to ~74MiB predicted at N=4096
  (`B`=20MiB).

## Result

**Part A**: `num_stages=3` and `num_stages=4` give effectively the same
measured knee (2 MiB) and a similar warm→cold degradation (50.4% vs.
51.7%) — `num_stages` does not explain the gap to experiment 16's `ncu`
numbers; that discrepancy remains unresolved.

**Part B**: the measured knee does **not** cleanly track `B`'s predicted
size:

| condition | B (MiB) | predicted knee (MiB) | measured knee (MiB) | warm→cold |
|---|---:|---:|---:|---:|
| N=17408 | 85.0 | 6.1 | 2 | +50.4% |
| N=12288 | 60.0 | 32.4 | 6 | +33.7% |
| N=8192 | 40.0 | 53.4 | 2 | +21.8% |
| N=4096 | 20.0 | 74.4 | 48 | +25.2% |

The predicted knee decreases monotonically as `B` shrinks, but the
measured knee doesn't follow it monotonically (2 → 6 → 2 → 48) — N=8192
in particular measures a knee as small as N=17408's despite predicting
the second-largest knee of the four. The warm→cold degradation is
substantial at every `N` tested (21.8%–51.7%), so the qualitative
row-major-is-cache-sensitive finding holds across shapes, but the specific
quantitative prediction (knee position tracking `B`'s size) is not
confirmed by this data. Full per-point data:
[`exp33-results/summary.json`](./exp33-results/summary.json), raw:
`exp33-results/knee_sweep_raw.jsonl`, full transcript: `exp33-run.log`.

## Files

- `knee_vs_b_size.py` / `knee-vs-bsize-job.yaml` — the sweep script and
  its Job.
- `exp33-results/knee_sweep_raw.jsonl` — raw per-iteration data.
- `exp33-results/summary.json` — per-condition predicted/measured knee and
  degradation.
- `exp33-run.log` — full run transcript, including the formatted tables.
