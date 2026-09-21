# 35 — Residency map

## What

`residency_map.py` measures what is actually still resident in L2 after a
single GEMM launch, per-slice, without a profiler. Method: run the GEMM
once, then immediately time a probe kernel that reads only one slice of
either operand (`A` or `B`) — bytes/time gives effective bandwidth, which
sits near L2 bandwidth if that slice was resident, or near DRAM bandwidth
if it had been evicted. A calibration pass measures the same probe
right-after-flush and twice-in-a-row, giving per-slice DRAM/L2 reference
bandwidths on this exact hardware. `gate_up_proj`, `GROUP_SIZE_M`∈{1,16},
16 slices each of `A` and `B`, slice order shuffled per repeat.

Two competing predictions:
- row-major (`GROUP_SIZE_M=1`): `B` was just swept end-to-end, so should
  be roughly *uniformly* resident, `A` largely evicted.
- grouped (`GROUP_SIZE_M=16`): each `B` column-block is touched once, so
  the *earliest*-touched (oldest) slices should be most evicted, giving a
  residency *gradient* across `B`, with `A` resident throughout.

## Result

Calibration: `B` after-flush 164.5 GB/s, immediately-reread 298.6 GB/s.
`A` after-flush 15.0 GB/s, immediately-reread 21.5 GB/s.

**`B`, `GROUP_SIZE_M=1`**: resident fraction bounces between 0.20 and 0.50
across all 16 slices with no clear trend — consistent with the "roughly
uniform" prediction, though clearly partial (not fully resident)
residency throughout, not the "swept end-to-end, fully warm" picture.

**`B`, `GROUP_SIZE_M=16`**: a real gradient — 0.17 at slice 0, climbing to
0.50–0.74 by slices 10–13 — confirming the predicted gradient directly.

**`A`, both `GROUP_SIZE_M=1` and `GROUP_SIZE_M=16`**: resident fractions
are near zero or negative (as low as −0.36) in both configs — i.e. `A`
reads no faster than the post-flush DRAM calibration in either case. This
does **not** confirm the "grouped keeps `A` resident" half of the
prediction; `A` looks essentially evicted regardless of ordering in this
data.

Full per-slice data: [`exp34b-results/summary.json`](./exp34b-results/summary.json).
Raw per-probe timings: `exp34b-results/residency_raw.jsonl`. Full
transcript with the formatted bar-chart tables: `exp34b-run.log`.

## Files

- `residency_map.py` / `residency-map-job.yaml` — the probe script and its
  Job.
- `exp34b-results/residency_raw.jsonl` — raw per-probe timing data.
- `exp34b-results/summary.json` — per-slice resident fractions + calibration.
- `exp34b-run.log` — full run transcript.
