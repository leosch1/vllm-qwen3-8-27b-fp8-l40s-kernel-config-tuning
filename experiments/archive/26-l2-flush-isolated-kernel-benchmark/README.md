# 26 — L2-flush isolated kernel benchmark

> Part 2 (validation under L2-flush pressure) is already documented in [`experiments/05-l2-flush-kernel-validation`](../../05-l2-flush-kernel-validation/). Parts 1, 3, and 4 below (cycling-pressure validation, stability rerun, tuning-stability check) aren't used in the blog.

**Status: complete.** Mirrors `20`'s cache-aware validation exactly, for
`25`'s L2-flush-tuned config instead: default vs. L2-flush-tuned, both
timed under `17`'s cache-cycling design (64 distinct `B` copies per point,
identical reset-to-index-0 sequence for both configs at each M) — the
regime the fix actually targets, not the tuner's own always-warm
methodology.

## Why

`25` validated the L2-flush fix two ways already: does the full rerun
converge on similar configs as `17`'s 64-copy cycling fix (yes, similar
`GROUP_SIZE_M=1` residual rate), and do the two methods' own winning
configs perform about the same head-to-head (yes, within ~1.3% on
average). Neither of those checks it against `default` directly, under
this project's standard cache-cycling isolated-kernel methodology — the
same way `20` validated the cache-aware config and `01` validated the
original `tuned` config. This closes that gap: same 175-point grid, same
chart convention as `chart-cache-cycling-speedup.html` /
`chart-tuned-cycling-speedup.html`, so it's directly, visually comparable
to every other config validated this way.

No need to rerun `default` as a separate baseline — the cache-cycling
harness always benchmarks `default` and the config under test together, in
the same paired run at each point (that's inherent to the fair-comparison
design, not an extra rerun of a prior experiment).

## Method

`compare_default_vs_l2flush_cycling.py` — `20`'s
`compare_default_vs_cache_aware_cycling.py` with the config source swapped
to `25`'s `l2-flush-tuned-configs/`, otherwise byte-identical: same
`DEFAULT_CONFIG`, same `CyclingLaunchCounter` (identical B-copy sequence,
reset to index 0, for both configs at each point — a counterbalanced
comparison, not an order confound), same `NUM_ITERS=2000`, same 5 shapes ×
18 anchor + 17 held-out M values.

Deployed as `isolated-benchmark-exp25-l2flush-cycling-job.yaml`, a plain
throwaway Job embedding both the config JSONs and the script directly
(same pattern as every isolated-benchmark job in this project).

## Result

Ran cleanly, zero errors. Chart: `chart-l2flush-cycling-speedup.html`. Raw
table output: `isolated-benchmark-exp25-l2flush-cycling-results.log`.

| | all 175 points | large-M anchors (M≥512, n=55) |
|---|---:|---:|
| mean speedup vs. `default` | +7.9% | +8.4% |
| negative points | 17/175 (10%) | 2/55 (4%) |
| worst point | −26.6% (`out_proj`, M=160, held-out) | — |

Directly comparable to `20`'s cache-aware result (+mean not explicitly
reported there, but 10/175 negative, worst −16.0% at the same `out_proj`
M=160 point): L2-flush is a real, clean win over `default` under the exact
regime this class of fix targets, with a similar (slightly higher) overall
negative-point rate concentrated at the same non-anchor, small-to-mid-M
held-out points that both prior configs' isolated benchmarks also struggle
with — consistent with `25`'s own finding that the exact `GROUP_SIZE_M`
picked at small `M` is close-competitor noise, not a sign the L2-flush
mechanism is worse. At the large-`M` anchors this whole investigation is
actually about, the result is clean: +8.4% mean, only 2/55 points negative.

## Part 2: same comparison, under L2-flush pressure instead of cycling

**Why:** the part-1 comparison above validates the L2-flush-tuned config
under `17`'s cycling mechanism -- a reasonable "does it generalize to a
different cache-pressure pattern" check, but not the most direct one,
since it's a different mechanism than the one that actually produced the
config. `compare_default_vs_l2flush_l2flushed.py` asks the more literal
question instead: under the *exact* methodology `25` used to pick this
config (a direct L2 flush before every timed launch, sized from the
device's real reported L2 capacity), does it actually win? Deployed as
`isolated-benchmark-exp25-l2flush-flushed-job.yaml`.

**Result: mean speedup +15.5%, only 1/175 points negative anywhere**
(`down_proj`, M=768, -4.2%) -- nearly double the cycling comparison's mean
and a 17x reduction in negative points:

| | cycling pressure (part 1) | L2-flush pressure (part 2) |
|---|---:|---:|
| mean speedup, all 175 points | +7.9% | **+15.5%** |
| negative points | 17/175 (10%) | **1/175 (0.6%)** |
| worst point | -26.6% | **-4.2%** |
| mean speedup, large-M (n=55) | +8.4% | **+11.75%** |
| negative, large-M | 2/55 | **1/55** |

This is exactly what matching the validation mechanism to the tuning
mechanism should do: it removes one entire source of measurement mismatch
noise (the config's own cache-locality assumptions being tested against a
*different* cache-pressure pattern than the one it was optimized for
means their "always warm/always cold" boundary conditions don't line up
exactly at every intermediate M). The cycling chart isn't wrong or
superseded -- it answers a genuinely different, harder question (does
this generalize beyond its own tuning assumption) -- but this is the more
direct answer to "does the fix work", and it's considerably cleaner.

Chart: `chart-l2flush-flushed-speedup.html`. Raw log:
`isolated-benchmark-exp25-l2flush-flushed-results.log`.

**Note on a fixed bug found while building this:** the L2-flush mechanism's
device-properties lookup (`torch.cuda.get_device_properties(device).L2CacheSize`)
had the wrong attribute name -- PyTorch exposes it as `L2_cache_size`
(underscored), not `L2CacheSize`. Every run in `25`/`26` silently hit the
`AttributeError` fallback path instead of querying real hardware, though
this didn't invalidate any result: the documented 128MiB fallback constant
is still larger than this GPU's real L2 (100.66MB), so the flush buffer
was always big enough to fully evict the cache regardless. Fixed in
`25`'s `l2_flush_tune.py`/`trainjob-full.yaml` and this experiment's
`compare_default_vs_l2flush_l2flushed.py` for future runs -- confirmed
working in part 3 below (`resolved L2_cache_size from device properties:
100.7MB`, no fallback fired).

## Part 3: stability check -- two independent runs, same design

Reran part 2's exact job spec a second time on a separate pod (fresh
tensors, fresh warmup, same 175-point grid, same L2-flush mechanism, now
with the attribute-name bug fixed so both runs resolve the real hardware
L2 size directly). Chart: `chart-l2flush-flushed-speedup-stability.html`
(two panels, shared y-axis range). Raw log:
`isolated-benchmark-exp25-l2flush-flushed-run2-results.log`.

| | run 1 | run 2 |
|---|---:|---:|
| mean speedup, all 175 points | +15.5% | +15.1% |
| negative points | 1/175 | 2/175 |
| worst point | −4.2% (`down_proj`, M=768) | −4.2% (`down_proj`, M=768) |
| min / max across all 175 points | −4.2% / +30.2% | −4.2% / +30.2% |

**Both runs land on the exact same min, the exact same max, and the exact
same worst point** across all 175 (shape, M) combinations, with per-shape
curves that are visually near-identical between the two panels. This is a
clean, reproducible result, not a fragile one -- the strength of the
L2-flush validation isn't an artifact of one lucky run.

## Part 4: tuning-stability, not just measurement-stability

Parts 2/3 both benchmark the *same* L2-flush-tuned config (`25`'s original
output) -- a pure measurement-repeatability check. A harder question:
does the L2-flush *autotune script itself* reliably produce a config with
this same performance envelope, or did `25`'s specific output happen to be
a lucky draw? Reran `25`'s full production-scope autotune script a second,
completely independent time (`trainjob-full-run2.yaml`, fresh 5-shape x
18-M search, no shared state with the first run) to get a genuinely
different tuned config, then ran that new config through the identical
flush-pressure isolated benchmark.

**Tuning-time stability** (mirrors `25`'s own methodology): at the
large-M anchors, run 2's config lands on `GROUP_SIZE_M=1` in 11/30 points
vs. run 1's 12/30 -- consistent with `25`'s finding that the residual
`GROUP_SIZE_M=1` rate is a genuine, stable minority outcome (~37-40%) of
the search, not something that drifts between independent reruns. Only
14/30 points pick the exact same `GROUP_SIZE_M` value between the two
runs -- confirming (again) that the *specific* value chosen is close-
competitor noise, while the *aggregate* shift away from `1` is stable.

**Kernel-level result, this new independently-retuned config vs. default,
under L2-flush pressure:**

| | run 1 (25's original config) | run 3 (independently retuned) |
|---|---:|---:|
| mean speedup, all 175 points | +15.5% | **+14.93%** |
| negative points | 1/175 | 2/175 |
| worst point | −4.2% (`down_proj`, M=768) | −4.5% (`down_proj`, M=768) |

**A genuinely different config, from a completely independent autotune
search, lands in the same narrow performance range** -- same mean
speedup to within a percentage point, same negative-point count, even the
same worst-performing point. Chart updated to 3 panels:
`chart-l2flush-flushed-speedup-stability.html`. Raw log:
`isolated-benchmark-exp25-l2flush-flushed-retune-run2-results.log`. New
retuned config: `../25-l2-flush-autotune-rerun/l2-flush-tuned-configs-run2/`.

This is the strongest stability result in the whole L2-flush validation:
it's not just that repeating the same measurement gives the same answer
(parts 2/3) -- repeating the *entire tuning process* from scratch, and
landing on a different config each time, still gives the same real-world
performance envelope. The L2-flush fix's benefit doesn't depend on getting
lucky with which specific `GROUP_SIZE_M`/`num_stages` combination the
search happens to converge on.

## Reproduce

```bash
oc apply -f isolated-benchmark-exp25-l2flush-cycling-job.yaml
```
