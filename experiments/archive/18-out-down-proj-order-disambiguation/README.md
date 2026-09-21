# 18 — out_proj / down_proj order-based disambiguation

**Status:** confirmed. Resolves >99% of the one ambiguity
[`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/) left open, via
a second, independent, purely structural method — no new capture, no
cluster/Nsight Operator access needed.

## What

`out_proj` (`N=5120, K=3072`, the mixer/attention output projection) and
`down_proj` (`N=5120, K=8704`, the dense-FFN down-projection) share `N`, and
under both default's fixed tile formula and tuned's per-anchor lookup they
frequently produce the **byte-identical launch grid/block signature** — grid
dims alone can never tell them apart for the affected launches. This isn't
rare: across all 4 reliable-instance captures, **47.8–50.0% of every real
captured GEMM launch** falls into this exact collision (confirmed directly
by enumerating both shapes' real tuned-config JSONs against every observed
`(gridX, blockX)`).

`08` already resolved most of this via duration-histogram bimodality (the
two shapes' real launch durations cluster separately even when their grid
ties, since duration still depends on `K`), but explicitly left a residual
**~3–3.5%** genuinely ambiguous even after that.

This experiment tests a second, independent disambiguator: **real vLLM
program order**. Within one decoder layer, the forward pass issues kernels
in a fixed sequence:

```
{qkv_proj | in_proj_qkvz} → attention → out_proj → gate_up_proj → down_proj
```

`gate_up_proj` is never ambiguous (its grid divisor is unique among all 5
shapes), so it's a reliable anchor: the ambiguous 5120-shape launch
immediately **before** a `gate_up_proj` launch must be that layer's
`out_proj`; the one immediately **after** must be that layer's `down_proj`.

## Why

The GEMM-shape decomposition built for
[`chart-durations-by-shape*.html`](../05-nsys-hardware-profiling-matrix/)
had to lump `out_proj`/`down_proj` into one honest "either/or" bucket rather
than resolve them, since implementing `08`'s duration-clustering was out of
scope for those charts. Before deciding whether it's worth doing properly,
this experiment checks whether a *much* simpler structural trick — just
looking at launch order — is robust enough to use instead (or as a
cross-check on duration-clustering).

## How

No new data was captured and **no cluster/Nsight Operator access was used**
— everything needed already exists locally as `kernel_trace.sqlite` under
[`05-nsys-hardware-profiling-matrix/profiling-results/*-v2/`](../05-nsys-hardware-profiling-matrix/profiling-results/),
extracted during that experiment. This is pure offline reanalysis:

1. Read `start`, `end`, `gridX`, `blockX`, `deviceId`, `globalPid` for every
   real captured `_w8a8_triton_block_scaled_mm` launch, from the same 4
   reliable-instance captures `05` and `08` use (default/tuned × c=1/c=64).
2. **Confirmed launch order == program order first, before trusting it**:
   every capture has exactly 2 distinct `(deviceId, globalPid, streamId,
   contextId)` combinations — one CUDA stream per device, matching TP=2. No
   concurrent-stream interleaving to worry about, so sorting by `start`
   within one device reproduces the real kernel issue order.
3. Classify already-unambiguous shapes via `08`'s exact formula (default:
   fixed `BLOCK_SIZE_M=64, BLOCK_SIZE_N=128`; tuned: per-anchor lookup from
   each shape's own tuned JSON).
4. For every ambiguous (`out_proj_or_down_proj`) launch, look at its
   immediate neighbors in the per-device, time-ordered GEMM sequence:
   `down_proj` if the previous launch is `gate_up_proj`, `out_proj` if the
   next launch is `gate_up_proj`.

Script: [`resolve_order.py`](./resolve_order.py). Raw output:
[`results.json`](./results.json).

## Results

| capture | total launches | ambiguous (grid alone) | resolved by order | still unresolved | out_proj mean | down_proj mean | ratio (K ratio 2.83) |
|---|---:|---:|---:|---:|---:|---:|---:|
| default c=64 | 536,576 | 256,512 (47.8%) | 256,256 | 256 (0.10%) | 50.9µs | 130.9µs | 2.57 |
| tuned c=64 | 537,088 | 258,048 (48.0%) | 258,048 | 0 (0.00%) | 39.4µs | 105.8µs | 2.69 |
| default c=1 | 1,310,720 | 655,360 (50.0%) | 650,240 | 5,120 (0.78%) | 43.9µs | 101.5µs | 2.31 |
| tuned c=1 | 1,310,720 | 652,800 (49.8%) | 652,800 | 0 (0.00%) | 24.9µs | 68.9µs | 2.76 |

**Resolution rate: 99.2–100% of every ambiguous launch**, using nothing but
adjacency to an already-unambiguous `gate_up_proj` anchor — better than `08`'s
own duration-clustering result (~96.5–97% before its extra
`signature_maps.pkl` lookup step). The residual unresolved launches (0–0.78%)
are exactly the edge cases already known from other analyses (`unmapped`
launches sitting at a capture-window boundary breaking the immediate-neighbor
check), not a flaw in the method.

**Two independent sanity checks both hold:**

- **Count ratio**: the real architecture fires `out_proj` and `down_proj`
  exactly once per layer, same as `gate_up_proj` — so `total_out_proj /
  gate_up_proj` and `total_down_proj / gate_up_proj` should both land at
  ~1.0. They do: exactly `1.0000` for both c=1 captures, `0.963–0.968` for
  c=64 (the small shortfall is fully accounted for by the same handful of
  `unmapped`/boundary launches noted above).
- **Duration direction and magnitude**: `down_proj` (`K=8704`) should take
  measurably longer than `out_proj` (`K=3072`) at the same tile count, in
  roughly the ratio of their `K`s (`8704/3072 ≈ 2.83`). Every capture's
  order-resolved groups land at **2.31–2.76×** — the right direction, the
  right ballpark, entirely independent of the ordering logic itself (nothing
  about the resolution method references duration at all). If the ordering
  rule were resolving launches incorrectly, there's no reason this ratio
  would consistently land near 2.83.

**Conclusion: the ordering trick is robust enough to use as the primary
disambiguator** — simpler, deterministic (not a statistical cutoff), and
resolves strictly more launches than duration-clustering alone.

## Cluster access

**Not needed for this experiment.** Every input already exists on disk from
`05-nsys-hardware-profiling-matrix`'s captures
(`profiling-results/*-v2/kernel_trace.sqlite`) — this is entirely offline
reanalysis of existing data, no live pod, `nsys`, or Nsight Operator access
required.

## Reproduce

From the repo root:

```bash
python3 experiments/18-out-down-proj-order-disambiguation/resolve_order.py
```

## Next step (not done here)

The visualization charts in `05-nsys-hardware-profiling-matrix/` (
`chart-durations-by-shape.html`, `-and-mband.html`,
`-and-mband-visible-only.html`, `-full-resolution.html`) still show
`out_proj`/`down_proj` as one combined "ambiguous pair" color/legend entry,
by design choice at the time. Applying this experiment's per-launch
resolution to those charts (so `out_proj` and `down_proj` get their own
real colors/legend rows instead of the honest-but-coarser combined bucket)
is a follow-up, not yet implemented.
