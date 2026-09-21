# 17 — Cache-aware autotune rerun

**Status:** built and run across four successive stages — a 2-value pilot,
a full 5-shape/18-M rerun, a 2-M noise-floor check, and a dedicated
10-repeat stability check at the one point that stayed ambiguous — because
each stage surfaced a question the previous one couldn't answer on its own.
Net result, once all four are read together: **real, and now precisely
quantified.** Every prior experiment that touched "what would a cache-aware
tuner pick" did it by hand-patching the tuner's *output* JSON (`groupfix`,
`15`) or by measuring the cache-eviction effect with a standalone script
(`12`) — never by actually fixing `tune()` itself and rerunning the real
search. This experiment does that: patch the one confirmed blind spot in
vLLM's real `benchmarks/kernels/benchmark_w8a8_block_fp8.py`, rerun its real
1280-config exhaustive search, unmodified otherwise, and see what it picks.

**Headline result:**

| M | original tuner's `GROUP_SIZE_M` | cache-aware tuner's `GROUP_SIZE_M` |
|---|---|---|
| 1024 | 1, 1, 1, 1 (4/4 runs) — **perfectly stable** | 16, 16, 16, 32 (4 runs) — grouped every time, usually 16 |
| 2048 | 1, 1, 1, 1 (4/4 runs) — **perfectly stable** | **16 in 11/14, 32 in 1/14, 1 in 2/14** — properly quantified across all stages (stage 4's dedicated 10-repeat check contributed 8/1/1) |

Two things stand out, together: **the original tuner's `GROUP_SIZE_M=1`
pick is not noise — it's a rock-solid, reproducible preference every single
time**, consistent with `14`'s mechanistic explanation (it's a real, if
small and locality-blind, win under the tuner's own always-warm
methodology, not a coin flip). **The cache-aware fix reliably moves away
from `GROUP_SIZE_M=1` toward a grouped value — 80% of the time landing on
`16` specifically, `15`'s hand-patched `groupfix` value — with the
remaining ~20% split between a different grouped value (`32`) and a rare
reversion to `1`.** That reversion is now known to be a genuine, if
infrequent (~10%), minority outcome of the tuner's single-pass, no-repeat
benchmarking methodology — not an artifact of running M=2048 deep inside a
long multi-shape job, since stage 4 reproduced it in a clean,
single-purpose process too.

## What

Take the real tuning script's `tune()` function and make exactly one
change: instead of one fixed, never-cleared `A`/`B` pair reused across all
1280 candidate configs (the confirmed blind spot from `14`/`16`), cycle `B`
through `N_WEIGHT_COPIES=64` independently-random copies (matching the real
model's 64 transformer layers — the same count `12` used), selected by a
counter that increments on *every individual kernel launch* — warmup and
timed alike — across the *entire* search for a given `M`, never reset
per-candidate-config.

That last detail matters and is deliberate, not incidental: `12`'s own
`ncu` follow-up found a single isolated cache miss costs nothing
measurable for either config — only continuous, rapid, sustained cycling
through many distinct tensors produces the effect. A counter reset per
candidate (touch a few different copies, then go back to copy 0 for the
next config) would just reproduce that "single isolated miss" pattern that's
already been shown to not matter. A global, never-reset counter keeps the
cache under continuous pressure for the whole run, matching `12`'s design
rather than accidentally recreating the thing `12`'s `ncu` follow-up ruled
out.

`A` is created once and reused throughout, unmodified from the real script
— `12` already established activations are much smaller (10.5MB vs `B`'s
89.1MB) and contribute far less L2 pressure, so holding it fixed isolates
the one variable that matters.

Everything else — search space (`get_configs_compute_bound()`, verbatim,
all 1280 configs), per-candidate benchmarking methodology (5 warmup + 10
timed CUDA-event-bracketed iterations, including the real script's own
documented `/10` timing bug, kept for exact methodology parity since it's a
constant factor that cancels in ranking), `OutOfResources` handling — is
copied unchanged from the real script.

## Why

`14` and `16` together established *that* the real tuning script is
structurally blind to cache-locality cost and *that* `GROUP_SIZE_M=1` is
worse than grouped values under a properly isolated measurement. But
nothing in this project had actually verified that fixing the tuner's own
search loop and rerunning it produces the same answer a human reasoning
about the mechanism came up with by hand (`groupfix`). That's a real gap:
it's possible a corrected tuner would pick some *other* `GROUP_SIZE_M`
value, or — more concerning for the "the tuner otherwise works fine"
narrative — that fixing this one blind spot would also shift other
parameters (`BLOCK_SIZE_M`, `num_warps`, `num_stages`) in ways nobody
predicted, since all 1280 configs are re-ranked, not just re-scored on one
axis.

## How — four stages

### Stage 1: 2-value pilot

Scope: `gate_up_proj` only, at **M=1024 and M=2048** — the two values with
the richest existing comparison data in this project (`14`'s `groupfix`
swap used M=2048; `16`'s `ncu`/single-M work used M=1024). Deployed as a
`TrainJob` (`trainjob.yaml`) on the exact same `runtimeRef`/image/Kueue
queue (`gpu-autotune`) as the real production `qwen3-8-27b-fp8-dense-autotune`
job, single GPU, results printed to stdout rather than written to the
production tuned-configs PVC (a one-off comparison run, not meant to
overwrite anything live).

**Result:** both M values, `GROUP_SIZE_M` was the *only* parameter that
changed (1→16), everything else identical to the original. Looked like an
unusually clean confirmation — clean enough to be worth double-checking at
scale before trusting it.

### Stage 2: full 5-shape × 18-M rerun

Same patch, but run exactly as the real production job does: all 5 GEMM
shapes, the full 18-value `M` grid, using the actual `tune_on_gpu()`/`main()`
driver and `save_configs()` (so real, standard-format tuned-config JSON
files come out, not just a comparison printout) — `trainjob-full.yaml`.
Completed in 72 minutes (faster than production's ~133-minute baseline for
the same scope, likely because JIT-compiled kernels get reused across
shapes/`M` values within one long-lived process).

**Result: much messier than the pilot.** Diffing each shape's output
against the original tuned-config JSON (`cache-aware-tuned-configs/` in
this folder holds all 5 real, standard-format files):

- At small `M` (1–256), *many* parameters differ from the original, not
  just `GROUP_SIZE_M` — `BLOCK_SIZE_M/N`, `num_warps`, `num_stages` all
  shift around too, across most shapes.
- At large `M` (≥512, the range that actually matters — `08` localized the
  real regression to `M≳640`), differences are mostly still `GROUP_SIZE_M`-
  dominated, but not cleanly: `gate_up_proj` at **M=2048 — the single
  most-studied point in this whole project — came back with `GROUP_SIZE_M`
  unchanged at 1** this time, contradicting the pilot's clean 1→16 result at
  the exact same shape and `M`.

That contradiction (stage 1 vs. stage 2, same shape, same `M`, same patch)
is what stage 3 exists to resolve.

### Stage 3: noise-floor check

The concern: `benchmark_config()`'s methodology (one 10-iteration timed
measurement per candidate, no repeats) is inherently noisy, and there was
no baseline for how much the *original, unmodified* tuner's own output
varies run-to-run with no fix applied at all. Without that, stage 2's messy
result couldn't be distinguished from "the fix has a real but noisy effect"
vs. "any two independent runs of this search disagree this much, fix or
not." `noise_floor_trainjob.yaml` runs, per `M∈{1024,2048}`,
`gate_up_proj` only, four `tune()` calls in alternating (not blocked) order
— `original`, `cache-aware`, `original`, `cache-aware` — so a monotonic
drift across the run can't bias one condition more than the other.

### Stage 4: dedicated M=2048 stability check

Stage 3 left `M=2048` at only 4 total trials (pilot + stage-2 + 2 noise-floor
repeats), one of which was the reversion to `1` — far too few to tell "a
genuine, if infrequent, minority outcome" from "the specific run that
produced it had something unusual going on" (stage 2's `1` result ran deep
inside a 90-`tune()`-call process; stage 3's didn't). `m2048_stability_trainjob.yaml`
runs **10 independent cache-aware-only `tune()` calls, back to back, in one
clean, single-purpose process** — same shape, same `M`, nothing else
sharing the run.

**Result:** `GROUP_SIZE_M` distribution over 10 runs — **`16`: 8, `32`: 1,
`1`: 1.** This settles the question stage 2 raised: the reversion to `1` is
a real, reproducible minority outcome (roughly 1 run in 10), not an artifact
specific to stage 2's particular execution context. `16` is the clear
majority (80%), matching `groupfix`'s value; `32` and `1` are both
infrequent alternates, consistent with `16`/`32`/`64` being close
competitors and `1` occasionally still winning a close single-pass,
no-repeat measurement by chance.

## Results — the aggregate picture

Combining all independent trials from all four stages (pilot + stage-2's
`M=1024`/`M=2048` entries + stage 3's two repeats each + stage 4's 10
repeats at `M=2048`):

| M | original tuner `GROUP_SIZE_M` | cache-aware tuner `GROUP_SIZE_M` |
|---|---|---|
| 1024 | 1, 1, 1, 1 (n=4) | 16, 16, 16, 32 (n=4) |
| 2048 | 1, 1, 1, 1 (n=4) | 16×11, 32×1, 1×2 (n=14 — stage 1+2+3's 4, plus stage 4's 10) — **`16` in 79%, some grouped value in 86%, `1` in 14%** |

**The original tuner is perfectly stable, every run, both `M` values.**
This is itself an important, clarifying result: `GROUP_SIZE_M=1` isn't a
noisy tie-break for the original tuner — it's a strong, reproducible
preference under its own always-warm methodology, exactly consistent with
`14`'s mechanistic story (a real, if locality-blind, win every single time
under that specific measurement regime).

**The cache-aware tuner reliably moves away from `GROUP_SIZE_M=1` — properly
quantified at M=2048 (n=14, thanks to stage 4): a grouped value 86% of the
time, `16` specifically 79% of the time — but which grouped value carries
real run-to-run noise** (`16` dominant, `32` a rare alternate at both `M`
values; `1` itself recurs about 1 run in 7, confirmed as a genuine minority
outcome rather than an artifact of any one run's context). `16` — `15`'s
hand-patched `groupfix` value — is the clear majority answer, not a
one-off; the minority outcomes (`32`, and the occasional reversion to `1`)
are consistent with genuine measurement noise in the specific choice among
several grouped values that likely perform close to each other, not
evidence against the direction of the effect. M=1024 only has n=4 so far
(no stage-4-equivalent dedicated check run there yet) — flagged as
follow-up if a precise rate is wanted at that `M` too.

`num_stages` shows some noise too, even in the *original* tuner (M=2048:
4 in one run, 3 in another) — a real, if smaller and less consequential,
baseline noise floor that isn't specific to the cache-aware patch.

At small `M` (1–256, from stage 2's full run), the much wider cross-parameter
noise is consistent with this project's own established L2-capacity
reasoning (`16`): the cache-locality effect shouldn't matter much until the
working set approaches L2 capacity, which only happens around `M≳512` for
this shape — below that, near-ties across *many* configs are expected to be
noise-dominated regardless of whether the cache fix is applied, not a sign
that the fix broke something. This wasn't directly re-tested with a
small-`M` noise floor (only `M=1024`/`2048` got the dedicated stage-3
check) — flagged as follow-up, not assumed to hold.

Caveat on absolute timing values reported anywhere in this experiment's raw
logs: they come from `benchmark_config()`'s own self-reported average,
which carries the same documented `/10` bug covered elsewhere in this
project (`experiments/README.md`'s "Known caveat" section) — a constant
factor that cancels in *ranking* (which config wins) but makes the absolute
microsecond values ~10x too small. Not corrected here since only the
ranking/winner matters for this experiment's question.

## What this does and doesn't establish

**Does establish:** the "fix the autotune script" recommendation is real,
not just plausible — a genuinely patched `tune()`, rerun end to end,
reliably shifts the tuner away from `GROUP_SIZE_M=1` toward a grouped
value, and the original tuner's `GROUP_SIZE_M=1` pick is confirmed to be a
strong, reproducible preference rather than noise, strengthening rather
than weakening the "train/serve mismatch" story from `14`. `15`'s
hand-picked `groupfix` value (`16`) is empirically the most common answer a
corrected search converges on.

**Doesn't establish (revised down from the pilot's overly clean framing):**
that `GROUP_SIZE_M` is the *only* parameter the fix ever touches (stage 2
shows real movement in other parameters, concentrated at small `M`); that
the fix deterministically picks one exact `GROUP_SIZE_M` value (it doesn't
— it reliably picks *a* grouped value, not reliably the *same* grouped
value); or whether `N_WEIGHT_COPIES=64` specifically (vs. some other count)
is what matters versus just "enough copies to exceed L2." A full-scale,
multi-repeat noise floor across all 5 shapes and all 18 `M` values (this
experiment only got that treatment at `M∈{1024,2048}`, one shape) would be
needed to fully generalize — flagged as follow-up, not assumed.

## Reproduce

```bash
# stage 1 -- 2-value pilot
oc apply -f trainjob.yaml
# stage 2 -- full 5-shape x 18-M rerun, real save_configs() output
oc apply -f trainjob-full.yaml
# stage 3 -- noise-floor check (2 M values x 4 alternating runs)
oc apply -f noise_floor_trainjob.yaml
# stage 4 -- dedicated M=2048 stability check (10 independent cache-aware-only runs)
oc apply -f m2048_stability_trainjob.yaml

# watch: oc get pods -n enterprise-ai | grep cache-aware
# logs:  oc logs <pod> -n enterprise-ai | grep -v 'it/s\]'   # strips tqdm noise
```

`cache_aware_tune.py` is the standalone, locally-runnable version of stage
1's logic (depends on this repo's `_vendored_matmul_timing.py` and
`tuned-configs/`, same convention as every other experiment script here).
Each `TrainJob` yaml embeds a fully self-contained copy of the relevant
script directly (no external file dependencies, since the pod only has
`/tmp` to work with, matching the real production autotune `TrainJob`'s own
pattern).

Raw outputs:
- `cache_aware_tune_run.log` / `cache_aware_tune_results.json` — stage 1
- `full_run.log` — stage 2's complete job log; `cache-aware-tuned-configs/`
  — all 5 shapes' real, standard-format tuned-config JSON files, directly
  diffable against `../../../tuned-configs/`
- `noise_floor_run.log` / `noise_floor_results.json` — stage 3
- `m2048_stability_run.log` / `m2048_stability_results.json` — stage 4
