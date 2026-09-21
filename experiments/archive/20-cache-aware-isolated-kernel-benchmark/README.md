# 20 — Cache-aware config isolated kernel benchmark

> Part 11 (the original tuned config re-measured under L2-flush pressure) is already documented in [`experiments/04-l2-flush-retune`](../../04-l2-flush-retune/). Parts 1-10 below (the cache-aware/64-copy-cycling fix) aren't used in the blog.

**Status:** built and run; **a real, previously-unknown caveat surfaces.**
`01` established the original tuned config never loses to `default` in
isolation, anywhere (0/90 anchors negative). This experiment asks the same
question of the *cache-aware* config (`17`'s round 2, already validated end
to end in `19`) — and the answer is different: **it does lose to `default`
at some points, concentrated in one mid-range `M` band, on two of the five
shapes, sometimes substantially (worst: -59.6%).**

This doesn't contradict `19`'s e2e result — real serving's aggregate
throughput across the whole traffic mix still came out ahead everywhere —
but it's a genuine, newly-discovered gap between "wins in aggregate
end-to-end" and "never loses at the kernel level," which `groupfix` (by
construction, a single-parameter patch) doesn't have.

## What

Same exact methodology as `01`: `benchmark_config()`/`w8a8_block_matmul`
vendored verbatim from vLLM's own tuner, same `DEFAULT_CONFIG`, same 18
tuner-anchor batch sizes plus the same 17 held-out (generalization) batch
sizes in between, same all 5 shapes, same `NUM_ITERS=2000`, same
max-output-diff numerical sanity check. The only substitution: the "other"
config comes from `17`'s cache-aware autotuning-script output (round 2)
instead of this repo's original always-warm `tuned-configs/`.

## Why

`17`/`19` validated the cache-aware config's *aggregate* production
behavior (never regresses across a full concurrency sweep) and its
*search-stability* profile (reliably picks a grouped `GROUP_SIZE_M`). Never
directly asked whether the config that comes out of the full 18-M search —
which, per `17`'s stage 2, moves *more than just* `GROUP_SIZE_M` at small
`M`, with real cross-parameter noise concentrated exactly there — is ever
isolated-kernel-worse than `default` on its own. That's a different,
narrower question than either prior experiment answered, and the natural
next check before treating this config as a strict, no-downside
replacement for `default`.

## Results

175 points (5 shapes × 35 M values each). **24/175 (13.7%) come back
negative** — 5 at the tuner's own 18 anchor points, 19 at held-out
(off-anchor) M values. All output-diff checks passed (`0.0000` everywhere)
— the two configs still agree numerically; this is purely a speed
difference, not a correctness issue.

Worst case per shape:

| shape | worst speedup (cache-aware vs. default) |
|---|---:|
| `gate_up_proj` (N=17408,K=5120) | **-59.6%** (M=56, held-out) |
| `in_proj_qkvz` (N=8192,K=5120) | -38.0% (M=80, held-out) |
| `qkv_proj` (N=7168,K=5120) | -7.0% (M=160, held-out) |
| `out_proj` (N=5120,K=3072) | -7.8% (M=200, held-out) |
| `down_proj` (N=5120,K=8704) | -5.4% (M=768, held-out) |

The negative anchor points (the tuner's own actual search points, not
interpolation artifacts) are all on `gate_up_proj`, all in one contiguous
mid-range band, plus one negligible one on `down_proj`:

| M | default (µs) | cache-aware (µs) | speedup |
|---:|---:|---:|---:|
| 48 | 7.17 | 9.48 | -32.2% |
| 64 | 7.57 | 11.75 | -55.2% |
| 96 | 9.77 | 12.40 | -26.9% |
| 128 | 9.37 | 11.82 | -26.2% |
| 512 (`down_proj`) | 22.59 | 22.72 | -0.6% (noise-floor) |

**Root cause, checked directly against the actual config values**: at
`gate_up_proj` M=48/64/96/128, the cache-aware search picked much smaller
tiles than `default`'s own `BLOCK_SIZE_M=64, BLOCK_SIZE_N=128` — e.g. M=64:
`BLOCK_SIZE_M=16, BLOCK_SIZE_N=64` (vs. default's 64×128) — which multiplies
the launch grid and, at exactly this mid-`M` range, costs more in overhead
than `GROUP_SIZE_M`'s (still-correct, still-grouped: `64` at this point)
benefit recovers. This is precisely the "much messier than the pilot" small-M
noise `17`'s stage 2 already flagged (cross-parameter movement, not just
`GROUP_SIZE_M`, concentrated below `M≈256`) — this experiment shows that
noise isn't just cosmetic: at a couple of specific mid-range anchors on the
model's biggest shape, it's a real, substantial isolated-kernel loss, not
merely "a different but similarly-fast" pick.

**Elsewhere, the pattern from `01` holds up well**: everywhere above
`M≈200` (all five shapes), cache-aware beats `default` by a similar
double-digit margin to the original tuned config (+9-14% at large `M`,
tapering the same way `01` documented) — and at very small `M` (≤~28-32) it
beats `default` by even more than the original tuned config did in several
shapes (e.g. `gate_up_proj` M=1: +43.1% vs. `01`'s original ~+50-70% range
for comparable M — same ballpark, not a regression there either).

## Part 2: the same 175 points, under cache-cycling pressure

The always-warm result above deliberately asks a narrow question (does the
new config ever lose even in the tuner's own idealized regime). The
question this project actually cares about is the mirror image: **does it
reliably win in the regime the fix targets** — cache pressure, not
always-warm? `compare_default_vs_cache_aware_cycling.py` reruns the
identical 175-point grid, but times both configs under `17`'s
cache-cycling design (64 distinct `B` copies, both configs seeing the
identical reset-to-index-0 cycling sequence at each point, so it's a fair,
counterbalanced comparison, not just "both configs are equally cold").

**Result: the severe mid-`M` dip almost entirely disappears.**

| shape | worst speedup (always-warm) | worst speedup (cache-cycling) |
|---|---:|---:|
| `gate_up_proj` | -59.6% | **-3.7%** |
| `in_proj_qkvz` | -38.0% | -16.0% |
| `qkv_proj` | -7.0% | -1.8% |
| `down_proj` | -5.4% | -4.7% |
| `out_proj` | -7.8% | -8.2% |

Negative points overall drop from 24/175 (13.7%) to **10/175 (5.7%)**, and
the ones that remain are far smaller and mostly at held-out (not anchor)
`M` values — `gate_up_proj`'s only negative anchor point under cycling is
M=96 at a negligible -0.6%, down from four substantial negative anchors
(-26% to -55%) in the always-warm pass. Best-case speedups stay strong
throughout (+18% to +26% across all 5 shapes). Full data:
`isolated-benchmark-cycling-results.log`.

**This directly confirms the mechanistic read from part 1**: the severe
always-warm dip was real, but specific to a regime (no cache pressure,
smaller/higher-overhead tiles penalized only by pure launch overhead) that
isn't the one real serving or the fix itself targets. Under the regime
that actually matters, the cache-aware config is a clean, close-to-uniform
win — not perfectly spotless (10 small residual negatives remain, mostly
off-anchor), but a qualitatively different picture from part 1's substantial,
anchor-level regressions.

## Part 3: groupfix, always-warm

Same always-warm methodology as part 1, run against `groupfix` (`15`'s
hand patch — only `gate_up_proj`'s large-M `GROUP_SIZE_M` values changed;
the other 4 shapes are the original tuned config, completely untouched).
`compare_default_vs_groupfix_warm.py` / `isolated-benchmark-groupfix-warm-job.yaml`.

| | cache-aware (part 1) | groupfix (part 3) |
|---|---:|---:|
| negative points | 24/175 (13.7%) | **8/175 (4.6%)** |
| negative anchor points | 5/90 | **1/90** (-0.8%, negligible) |
| worst case | -59.6% | **-11.6%** |
| best case | up to +43.1% | up to +75.3% |

**Dramatically cleaner than the cache-aware config's always-warm result,
and this makes complete mechanistic sense.** `groupfix` changes exactly
one parameter (`GROUP_SIZE_M`) — which has little to no effect when the
cache is always warm, since its entire benefit is cache-locality — at a
handful of large-M anchors on one shape, leaving every `BLOCK_SIZE_M/N/K`/
`num_warps`/`num_stages` value, and 4 of 5 shapes entirely, byte-identical
to the original tuned config `01` already showed never loses to `default`
when warm (0/90 anchors negative). The cache-aware config's much messier
always-warm result (part 1) isn't about `GROUP_SIZE_M` at all — it's the
side effect of a full independent retune moving *other* parameters too,
which `groupfix`'s surgical, single-parameter patch never does. Full data:
`isolated-benchmark-groupfix-warm-results.log`.

## Part 4: groupfix under cache-cycling pressure

Same cache-cycling methodology as part 2, same `groupfix` config.
`compare_default_vs_groupfix_cycling.py` / `isolated-benchmark-groupfix-cycling-job.yaml`.

| | cache-aware (part 2) | groupfix (part 4) |
|---|---:|---:|
| negative points | 10/175 (5.7%) | 13/175 (7.4%) |
| worst case | -16.0% (`in_proj_qkvz`, M=160) | -26.1% (`qkv_proj`, M=160) |
| best case | up to +26.4% | up to +31.6% |

**Comparable overall, not a clean win for either — a contrast worth
noting against part 3's clean always-warm result.** Under cache pressure,
`groupfix` is no cleaner than the full cache-aware retune, and its worst
case is actually a bit larger. Notably, the negative points on
`qkv_proj`/`down_proj`/`out_proj` — shapes `groupfix` never touches at
all, still running the original tuned config unmodified — show some of the
same kind of scattered, small-to-moderate cache-pressure-sensitive dips
(e.g. `qkv_proj` M=160: -26.1%, `down_proj` M=768: -7.5%). That's a
genuinely new data point this project hadn't measured before: **the
original tuned config itself has real, cache-pressure-sensitive weak spots
at scattered `M` values**, on shapes nobody has patched — meaning at least
part of this residual isolated-kernel noise isn't introduced by either
fix, it's a pre-existing property of the tuning process being read out for
the first time here. Put together with part 3, the picture is: `groupfix`
is nearly spotless when warm (matching the original tuned config almost
exactly, as expected from touching so little) but inherits the *original*
tuned config's own scattered cache-pressure sensitivity once cache
pressure is present — it was never designed to fix that, only `GROUP_SIZE_M`
at the one shape it targets.

## Part 5: the original, unpatched tuned config under cache-cycling pressure

One comparison this project had never actually run directly, despite all
the surrounding evidence: `default` vs. the plain original tuned config
(this repo's `tuned-configs/`, `GROUP_SIZE_M=1` at every large-M anchor on
`gate_up_proj`) under cache-cycling pressure, using `20`'s clean
2000-iteration, numerically-verified methodology — the most direct test
yet of `14`'s original mechanism finding. `compare_default_vs_tuned_cycling.py` /
`isolated-benchmark-tuned-cycling-job.yaml`.

**Result, at first glance surprising: tuned still beats default at most
`gate_up_proj` large-M anchors, even under cache pressure** — M=1024:
+6.0%, M=1536: +9.2%, M=2048: +5.8%, M=3072: +9.6%, M=4096: +8.5%. Only
M=512 is negative (-4.0%). Overall: 14/175 points negative, worst -20.5%
(`qkv_proj`, M=160) — comparable in scale to both `groupfix`'s (part 4)
and cache-aware's (part 2) own cycling results.

**This does not contradict `14`/`16` — it's answering a different
question.** `14`'s `same-B` comparison and `16`'s `ncu`/single-M work held
every parameter *except* `GROUP_SIZE_M` fixed and varied only that one
parameter, cleanly isolating its effect (confirmed there: grouped values
really do beat `GROUP_SIZE_M=1` under cache pressure/proper isolation).
This experiment compares the *full bundled* `tuned` config against
`default` — and `tuned` differs from `default` in more than
`GROUP_SIZE_M`: `BLOCK_SIZE_M=128` (2× default's 64) and `num_warps=8`
(2× default's 4) are also genuinely faster, `GROUP_SIZE_M`-independent
choices. In a single-shape, no-other-confounds isolated benchmark like
this one, those advantages are large enough to outweigh `GROUP_SIZE_M=1`'s
cache-locality cost at most points, even under cycling pressure. It's only
in *real serving* — where `GROUP_SIZE_M`'s cost compounds with genuinely
concurrent traffic across many shapes, KV-cache reads/writes, and other
kernel types simultaneously (the still-not-fully-explained gap between
`12`'s ~22% DRAM-Read reproduction and real serving's 72.8%) — that this
cost becomes large enough to flip the *aggregate* e2e result negative
(`02`'s original -4.1% at c=128). This single-shape cycling benchmark,
however clean, still isn't that regime — it reproduces part of the real
mechanism (confirmed: `GROUP_SIZE_M=1` really is worse in a properly
isolated, single-parameter sense) without reproducing the full real-serving
magnitude, the same open gap flagged back in `12`/`13`.

## Part 6: cache-aware and groupfix, directly against tuned (not default)

Parts 2/4/5 each compare one candidate config against `default` under
cache-cycling pressure. A natural follow-up: how do the two *fixes*
(cache-aware-tuned, groupfix) perform directly against the config they're
replacing — `tuned` — rather than against `default`? Computed with no new
benchmark run: all three cycling runs share the identical 64-B-copy
design, M grid, and hardware, so their absolute-microsecond durations are
directly comparable to each other. `parse_cross_compare_vs_tuned.py`
extracts the raw per-M durations already logged in Parts 2/4/5's result
logs and divides them directly (verified first that all three runs' own
`default` numbers track each other within 5% at every M, confirming this
cross-run comparison is sound).

**`chart-cache-aware-vs-tuned-cycling-speedup.html`** — cache-aware vs.
tuned. Mostly positive, especially at large M on `gate_up_proj` (M=512:
+13.6%, M=1024: +4.9%, M=2048: +5.2%) — the same anchors where `tuned`'s
`GROUP_SIZE_M=1` is worst. 28/175 points negative (worst -10.7%,
`out_proj` M=80), consistent with this being a full independent retune
that trades off differently at some M values even as it wins where it
matters most.

**`chart-groupfix-vs-tuned-cycling-speedup.html`** — groupfix vs. tuned.
A clean built-in sanity check: `groupfix` only changes `GROUP_SIZE_M`, and
only at 6 of `gate_up_proj`'s 18 M anchors (512, 1024, 1536, 2048, 3072,
4096) — everywhere else it's byte-identical to `tuned`. At every M where
the two configs are literally the same, the measured "speedup" is just
noise scattered around 0%, exactly as it should be. The real signal
concentrates precisely where the parameter changed: +12.7% (M=512) down to
+2.3% (M=2048) up to +5.3% (M=3072/4096) — confirming `GROUP_SIZE_M` alone
is what's carrying groupfix's advantage over `tuned` at large M, with
nothing else contaminating the comparison.

**⚠️ Caveat added after Part 8:** both charts in this part chain through
two separately-run `default` baselines (the same cross-run method Part
7/8 later re-examined). This particular pair of runs happened to agree
closely (the "identical-point" noise above tops out around ±5%), so these
two charts' numbers likely hold up — but that agreement was not
guaranteed, and Part 8 found a *different* run pair (tuned vs. singlefix)
where the same method swung as much as −18% at points with zero real
config difference. Treat this part's magnitudes as good-but-unverified
rather than re-deriving them with a same-run paired script, the way Part 8
did for singlefix.

## Part 7: singlefix under cache-cycling pressure — is the effect really that localized?

`23` built `singlefix`: byte-identical to the original `tuned` config
everywhere except `gate_up_proj`'s single M=2048 entry (`GROUP_SIZE_M`
1→16), and confirmed via real nsys capture that it recovers ~74% of
`groupfix`'s real-serving improvement. This part asks the isolated-kernel
version of the same question with the fine-grained M-sweep this regime
allows: does `singlefix`'s cycling speedup curve vs. `default` actually
track `tuned`'s curve almost everywhere and diverge only right at the one
anchor it changes — or is that expectation too clean?
`compare_default_vs_singlefix_cycling.py` /
`isolated-benchmark-singlefix-cycling-job.yaml`, same 175-point grid,
NUM_ITERS=2000, numerical sanity check as every other pass here.

**Confirmed at the pattern level. The exact magnitude below was corrected
in Part 8 — see there before citing a number.** Comparing `singlefix`'s
per-anchor speedup vs. `default` against `tuned`'s own (Part 5), on
`gate_up_proj`'s large-M anchors:

| M | tuned | singlefix | delta |
|---:|---:|---:|---:|
| 512 | −4.0% | −4.2% | −0.2pp |
| 768 | +1.8% | +0.6% | −1.2pp |
| 1024 | +6.0% | +5.4% | −0.6pp |
| 1280 | +5.4% | +5.2% | −0.2pp |
| 1536 | +9.2% | +7.7% | −1.5pp |
| 1792 | +6.7% | +6.7% | +0.0pp |
| **2048** | **+5.8%** | **+8.9%** | **+3.1pp** |
| 2560 (held-out, nearest anchor 2048) | +8.1% | +9.1% | +1.0pp |
| 3072 | +9.6% | +9.3% | −0.3pp |
| 3584 | +9.0% | +8.9% | −0.1pp |
| 4096 | +8.5% | +9.1% | +0.6pp |

Every point except M=2048 (and its held-out neighbor M=2560, which maps to
the same anchor) sits within ±1.5pp of `tuned` — indistinguishable from
run-to-run noise for a byte-identical config. M=2048 itself shows the
single largest positive delta of any point in the entire table: +3.1
percentage points, exactly where the one parameter change lives, and
nowhere else. `chart-singlefix-cycling-speedup.html` (the standard
5-shape/175-point view) and `chart-gate-up-proj-fix-comparison.html` (a
focused overlay of `tuned`/`singlefix`/`groupfix`/cache-aware-tuned on just
`gate_up_proj`, with a marker at M=2048) both show this directly —
`singlefix` hugs `tuned`'s curve everywhere except one visible bump at the
marked anchor, while `groupfix` and cache-aware-tuned (which both change
more than this one value) separate from `tuned` well before M=2048 and
stay separated after it.

**⚠️ Caveat added after Part 8:** this table's "delta" column is *chained*
through two separately-run `default` baselines (this run's own `default`,
and Part 5's own separate `default` run) — the same method Part 6 used for
cache-aware/groupfix vs. `tuned`. Part 8 found that method carries more
noise than its own internal sanity check could catch: at points where two
configs are provably byte-identical, chained deltas swung as much as
**−18%**, an order of magnitude bigger than the noise this table's own
numbers suggested. The *pattern* here (localized to M=2048) still holds —
Part 8's clean, same-run paired measurement confirms it independently —
but this table's own **+3.1pp** figure at M=2048 should not be quoted as
the effect size; Part 8's **+2.0%**, from a same-run paired comparison, is
the reliable number.

## Part 8: singlefix vs. tuned, paired in the same run — correcting Part 7's magnitude

Part 7 (and Part 6, for cache-aware/groupfix) measured "config A vs. tuned"
by chaining two *separate* Job runs, each already compared against
`default` on its own. That method's only cross-check was whether each
run's own `default` numbers looked internally consistent — it never
checked whether the *two runs' own `default` baselines agreed with each
other*, which is exactly what chaining requires. They don't, closely
enough: singlefix's run and tuned's run each report a different absolute
`default` duration at the same M (e.g. 138.89µs vs. 134.94µs at
`gate_up_proj` M=2048, a 2.9% gap on its own) — small on its own, but
compounding through two chained ratios, and at other points the gap runs
larger. Directly dividing the two runs' logged absolute durations (no
`default` in the loop at all) shows why this matters: at every point where
`singlefix` and `tuned` are byte-identical (every shape except
`gate_up_proj`, and every `gate_up_proj` anchor except M=2048), the
"speedup" should be exactly 0% — instead it swings as low as **−18.2%**
(`in_proj_qkvz`, M=256). That's pure cross-run noise with zero real config
difference behind it, an order of magnitude past what Part 6's own
same-run sanity check (a <5% spread in each run's own `default` numbers)
would have led anyone to expect.

**Fix: benchmark the two configs directly, in the same run.**
`compare_tuned_vs_singlefix_cycling.py` /
`isolated-benchmark-tuned-vs-singlefix-cycling-job.yaml` loads both
`tuned`'s and `singlefix`'s per-M configs and pits them head-to-head with
the same `CyclingLaunchCounter` counterbalanced design used everywhere
else in this experiment (64 B-copies, counter reset between the two
configs at each M point) — the same noise-canceling guarantee the
default-relative charts have always had, just with `tuned` as the baseline
instead of `default`.

**Result: dramatically cleaner, and the real effect size is smaller than
Part 7's chained estimate.** 90/175 points negative overall (a roughly
even scatter around zero — exactly what "no real difference" should look
like), worst −8.1% (`qkv_proj`, M=256, a shape the config change never
touches). On `gate_up_proj`'s large-M anchors, every point sits within
±1.6pp of zero except M=2048 itself:

| M | speedup (paired) |
|---:|---:|
| 512 | −0.2% |
| 1024 | −0.5% |
| 1536 | −1.6% |
| **2048** | **+2.0%** |
| 2560 (held-out) | +0.8% |
| 3072 | −1.0% |
| 4096 | −1.0% |

`chart-tuned-vs-singlefix-paired-cycling-speedup.html` (the standard
5-shape view) and `chart-singlefix-vs-tuned-methods-comparison.html` (both
methods plotted directly on top of each other, for `gate_up_proj`) show
the contrast: the chained/cross-run line drifts as low as −7.6% across a
curve that should be flat at zero; the paired line sits much closer to
zero everywhere, with +2.0% at M=2048.

**⚠️ Caveat added after Part 9 — this section's own headline claim was
also overclaimed.** Calling +2.0% "a clean, isolated effect" doesn't
survive checking it against this table's own null distribution: across
the 174 *other* points here (every one of which has zero real config
difference), stdev is 2.43pp and the range is [−8.1%, +12.3%] — 24% of
those known-zero points are at least as extreme as +2.0%. A single trial
at M=2048 cannot be distinguished from that noise. **Part 9 resolves this
properly** with repeated trials at the one point that matters, rather than
reading significance off a single-pass sweep — read that before citing a
number from this table.

**Standing lesson for this repo, beyond this one comparison:** a
same-run/own-`default` "sanity check" (Part 6's approach — do these
numbers look internally consistent?) is not the same claim as "these two
separately-run baselines agree with each other," and only the second
claim is what a cross-run chained comparison actually needs. Any future
"config A vs. config B" question in this project should default to a
same-run paired design (like this part, and like every default-relative
chart already does) rather than chaining through two separate
already-published `default` comparisons, even when each one's own
internal noise check passes.

## Part 9: is +2.0% at M=2048 real, or just noise? — a single-point stability check

Part 8's table showed a single reading of +2.0% at `gate_up_proj` M=2048
and called it "clean" and "isolated." It isn't, on its own: this
experiment's own 175-point sweep contains 174 *other* points with
provably zero real config difference (every point except `gate_up_proj`
M=2048), and that null population has stdev 2.43pp and ranges from −8.1%
to +12.3%. **24% of those known-zero points are at least as extreme as
+2.0%.** A single trial at M=2048 cannot be told apart from that noise —
2000-iteration averaging cancels random per-iteration jitter, but a
single measurement block still carries whatever run-to-run/point-to-point
drift (GPU clock/power state, thermal history) happened to be present at
the moment that one block ran, and 2000 repeats of the *same* block don't
touch that.

**Fix: stop trying to read significance off one pass over 175 different
points, and instead repeat the one point that matters.** Same design as
`17`'s dedicated M=2048 stability check for the autotune script itself:
`compare_tuned_vs_singlefix_m2048_repeated.py` /
`isolated-benchmark-m2048-stability-repeat-job.yaml` reruns the identical
paired, counterbalanced `tuned`-vs-`singlefix` comparison at exactly
`gate_up_proj` M=2048, 10 independent times (fresh `A`/`As`/64 B-copies
each repeat).

**Result: real, repeatable, and tightly distributed.**

| repeat | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| speedup | +2.27% | +2.43% | +2.17% | +2.66% | +1.94% | +1.43% | +1.33% | +0.83% | +2.02% | +0.33% |

**10/10 repeats positive.** Mean +1.74%, stdev 0.74pp — a third of the
175-point sweep's cross-point noise floor — giving a 95% CI of
**[+1.21%, +2.27%]**, entirely above zero. Part 8's original single
reading (+2.0%) sits comfortably inside this interval; it wasn't wrong,
it just wasn't verifiable on its own.

**What this resolves:** the 175-point sweep's 2.4pp noise floor is mostly
*cross-point* drift — different M values and shapes measured minutes
apart over the ~5-minute job, likely GPU clock/thermal state moving
between very differently-sized kernel launches — not noise inherent to
comparing `tuned` vs. `singlefix` at one fixed point (which the JIT
compiles as genuinely different kernel binaries, since `GROUP_SIZE_M` is
a `tl.constexpr`, confirmed by reading `_w8a8_triton_block_scaled_mm` in
vLLM's own source). Isolated back-to-back at a single point, that
switching cost is small (~0.74pp), and the real effect is easily visible
above it. The lesson isn't "isolated-kernel benchmarking can't see this
effect" — it's "a single pass over many different points, however many
iterations each point averages, is the wrong instrument for judging
whether one specific point's reading is signal or noise; repeat the point,
not the iteration count."

**This still doesn't explain the magnitude gap to real serving.** +1.74%
in properly-verified isolation vs. −46.1% (`tuned`→`singlefix`) in real
serving (`23`) is still a >25x difference — that gap is real and is the
same concurrency-compounding story documented in `09`/`12`/`13` (isolated
cache-cycling reproduces the *direction* of a cache-locality effect but
not remotely its *magnitude* under genuine concurrent multi-request
contention). Part 9 only closes the "is the small isolated number even
real" question, not the "why is real serving's effect so much bigger"
question — that one was already answered, separately, well before this
correction was needed.

## Part 10: within-block vs. between-block CIs, at full sweep scale

Part 9 resolved whether `gate_up_proj` M=2048's effect is real by
*repeating* that one point 10 times. A natural follow-up: what if we
instead compute a CI directly from the 2000 raw per-iteration latencies
inside a single measurement block (via the delta method on
`speedup% = 100*(1 - singlefix_us/tuned_us)`, using each block's own
SEM = stdev/√2000)? Does that within-block CI capture the real noise, or
does Part 9's between-repeat variability (mean +1.74%, stdev 0.74pp)
represent something the within-block statistics can't see?

**A 5-repeat check at M=2048 first (`compare_tuned_vs_singlefix_m2048_within_block.py`):**
every one of the 5 within-block CIs was tight (half-widths ~0.1–0.2pp) —
and two of them didn't even overlap each other (`[+1.29%,+1.66%]` vs.
`[+2.33%,+2.71%]`), direct proof the within-block CI is far too narrow to
describe run-to-run reproducibility. Quantified: between-repeat stdev
0.54pp vs. average within-block SE 0.10pp — a 5.6x gap in stdev, meaning
**~97% of the total variance is between-block drift the within-block
statistic never sees**, and only ~3% is the random per-iteration jitter
2000-iteration averaging actually cancels.

**Then the full 175-point sweep, with a within-block CI computed at every
point (`compare_tuned_vs_singlefix_cycling_with_ci.py` /
`isolated-benchmark-tuned-vs-singlefix-with-ci-job.yaml`).** This is the
decisive, full-scale version of the same check: **139 of 174 known-null
points (79.9%) have a within-block 95% CI that excludes zero** —
i.e. the within-block CI would call four out of every five points
"statistically significant," when the true rate should be ~5% since only
one point out of 175 has any real config difference at all. A few
examples, all provably null (`singlefix`≡`tuned` at every one of these):

| point | reading | within-block 95% CI |
|---|---:|---:|
| `out_proj` M=128 | +12.7% | [+12.17%, +13.25%] |
| `in_proj_qkvz` M=56 | +11.7% | [+11.31%, +12.18%] |
| `out_proj` M=6 | −8.9% | [−9.53%, −8.29%] |
| `qkv_proj` M=64 | −6.1% | [−6.76%, −5.48%] |

`gate_up_proj` M=2048 itself reads +1.4% with within-block CI
`[+1.25%, +1.62%]` in this particular run — which *also* excludes zero,
but that carries no diagnostic value on its own, since 80% of provably-null
points do exactly the same thing. **The within-block CI cannot
distinguish the one real point from noise; only Part 9's between-repeat CI
can.**

**Conclusion, stated precisely:** 2000-iteration averaging within one
measurement block does exactly what it claims — it estimates that one
block's own mean with high precision (SEM ~0.05–0.1µs on kernel durations
of tens to hundreds of µs). It says nothing about whether re-running the
same measurement would give the same answer, and for this benchmark
methodology on this cluster, it emphatically would not: ~97% of the
total variance lives between blocks, not within them. **Any CI reported
for a single-trial reading in this repo's isolated-kernel benchmarks
should be treated as meaningless unless it comes from repeating the whole
measurement independently — a within-block CI, however precise-looking,
is not a substitute.**

## Part 11: the original tuned config, under L2-flush pressure — closing part 5's open gap

Part 5 found the original `tuned` config's `BLOCK_SIZE_M`/`num_warps`
advantages mask `GROUP_SIZE_M=1`'s cache-locality cost under cache-cycling
pressure, and flagged an open question: this single-shape isolated
benchmark, however clean, doesn't reproduce the full magnitude of real
serving's regression, and it wasn't clear why. `25`'s general L2-flush
mechanism (a full flush of the device's real, reported L2 capacity before
every single timed launch — the most extreme cold-cache condition
possible, more severe than cycling through 64 weight copies) gives a way
to test whether that gap is a cache-pressure-*intensity* question.
`compare_default_vs_tuned_l2flushed.py` / `isolated-benchmark-tuned-l2flushed-job.yaml`.

**Result: this is the clearest, most dramatic demonstration of the
`GROUP_SIZE_M=1` regression this project has produced.** Under a full L2
flush, `gate_up_proj`'s large-M points — previously mildly positive or
near-zero under cache-cycling — swing sharply and monotonically negative
exactly where the original tuner picked `GROUP_SIZE_M=1`:

| M | speedup vs. cache-cycling (part 5) | speedup under L2-flush (this part) |
|---:|---:|---:|
| 512 | −4.0% | **−47.5%** |
| 768 | +1.8% | **−34.4%** |
| 1024 | +6.0% | **−23.7%** |
| 1280 | +5.4% | **−19.1%** |
| 1536 | +9.2% | **−11.4%** |
| 1792 | +6.7% | **−9.1%** |
| 2048 | +5.8% | **−6.0%** |
| 2560 | +8.1% | +0.2% |
| 3072 | +9.6% | +5.7% |
| 4096 | +8.5% | +5.4% |

No other shape shows anything close to this pattern — `in_proj_qkvz`,
`qkv_proj`, `down_proj`, and `out_proj` all stay solidly positive at every
large-M anchor, same as under cycling pressure. 13/175 points negative
overall.

### The M=40 outlier — explained (added later)

This part originally flagged `gate_up_proj` M=40 held-out at −64.6% as "a
single pathological tile-size interaction", didn't fit into the large-M
story, and set it aside. **That was wrong: it is the same mechanism, and it
should be folded in.**

M=40 is a held-out point. It snaps to anchor 32 (40 is equidistant from 32
and 48; `min()` breaks the tie by list order), whose config has
`BLOCK_SIZE_M=32` — so `ceil(40/32) = 2` M-tiles, where `default`'s
`BLOCK_SIZE_M=64` gives just 1. Combined with that config's
`GROUP_SIZE_M=1`, the two CTAs sharing a given `B` panel land **136 apart**
in dispatch order, so the 85MiB weight matrix is swept twice instead of
once. At M=40 the kernel is almost nothing *but* reading `B` (`A` is
0.2MiB, `C` is 1.3MiB), so a second sweep roughly doubles it, with no
compute to hide behind. Set `GROUP_SIZE_M >= 2` and the two CTAs become
adjacent, the matrix is swept once, and the penalty disappears.

So it is exactly the large-M mechanism — row-major re-reading the big
operand with the re-reads too far apart to hit cache — triggered by a
`BLOCK_SIZE_M` that creates a second M-tile out of 8 leftover rows.

**But two M-tiles alone isn't sufficient**, which is what makes this worth
recording. M=20 (anchor 16, `BLOCK_SIZE_M=16`) also has 2 M-tiles and also
`GROUP_SIZE_M=1`, also 272 CTAs — and measures **+1.2%**, not negative. The
difference is occupancy:

| | M-tiles | CTAs | smem/CTA | CTAs/SM | waves | measured |
|---|---:|---:|---:|---:|---:|---:|
| M=20 (anchor 16, `num_stages=3`) | 2 | 272 | 36KiB | 3 | **1** | +1.2% |
| M=40 (anchor 32, `num_stages=5`) | 2 | 272 | 80KiB | **1** | **2** | −64.6% |

M=20's config is light enough on shared memory that all 272 CTAs are
resident at once on 142 SMs, so both M-tiles run *concurrently* and each
`B` panel is fetched once. M=40's `num_stages=5` costs 80KiB/CTA, only one
fits per SM, so the tiles run as two sequential waves and `B` is genuinely
re-fetched. (Computed both ways for Triton allocating `num_stages` vs
`num_stages-1` buffers — same conclusion either way.)

**The corrected rule has three conjuncts:** `GROUP_SIZE_M=1`, *and* more
than one M-tile, *and* too little occupancy for those tiles to overlap.
`num_stages` therefore acts as a cache parameter here, not just a
latency-hiding one: it sets shared-memory footprint, which sets occupancy,
which decides whether row-major's re-reads are free or catastrophic.

Checked against every point in this part's chart: **every negative
`gate_up_proj` point satisfies the rule**, and the only negative point that
doesn't is M=112 at −1.6% (single sweep, noise-level). Past M≈512 the CTA
count exceeds residency regardless of `num_stages`, so waves are
unavoidable and each extra M-tile is a real extra sweep — with the penalty
shrinking as M grows (−47.5% → −6.0% → positive past ~2560) simply because
the fixed 85MiB of extra traffic becomes a smaller share of a kernel doing
proportionally more work.

**This closes part 5's open gap directly**: the reason the cache-cycling
benchmark didn't reproduce real serving's full regression magnitude isn't
that isolated single-shape benchmarking is fundamentally the wrong tool —
it's that 64-copy cycling wasn't cold enough to fully expose the effect
`BLOCK_SIZE_M`/`num_warps` were masking. Real serving interleaves five
GEMM shapes, attention, and KV-cache traffic between successive
`gate_up_proj` launches at the same `M`, plausibly evicting far more of L2
than cycling through 64 same-shape weight copies alone does. A full flush
is a closer proxy for that real condition than cycling is, and under it,
`gate_up_proj`'s large-M regression reappears at a magnitude much closer
to what real serving showed (`05`: `tuned` +76.5% *worse* than `default`
in that exact regime) than cycling's masked, near-neutral result did.

Chart: `chart-tuned-l2flushed-speedup.html`. Raw log:
`isolated-benchmark-tuned-l2flushed-results.log`.

## What this does and doesn't establish

**Does establish:** the cache-aware config is not a strict, no-downside
replacement for `default` under an always-warm measurement the way
`groupfix` — a single hand-verified parameter change — was shown to be in
`14`'s `same-B` comparison. A real autotuning rerun, even a "good draw" one
already validated end to end in `19`, can and does pick some
always-warm-isolated-kernel losers, concentrated in a specific mid-`M` band
on the model's two largest/most-`N`-heavy shapes. **But under cache-cycling
pressure — the regime that actually matters — that same config is a
near-uniform win**, confirming the always-warm losses are a measurement-
regime artifact, not evidence the fix is unreliable where it counts.

**Doesn't establish:** that either result contradicts `19`'s e2e finding.
`19` measured aggregate throughput across whatever `M` distribution real
continuous batching actually produces at each concurrency level — consistent
with part 2's cache-cycling result showing the fix winning almost
everywhere once cache pressure is present, matching real serving's own
access pattern more closely than the always-warm pass did. The always-warm
finding (part 1) remains a real, narrower caveat: a workload that somehow
ran this kernel with a genuinely warm, uncontended cache at exactly
M=48-160 on `gate_up_proj` could still see a real, substantial regression —
flagged as a residual, low-probability risk, not a live contradiction of
`19`'s validated result.

## Reproduce

```bash
oc apply -f isolated-benchmark-job.yaml
# oc logs <pod> -n enterprise-ai  once Succeeded
```

`compare_default_vs_cache_aware.py` is the standalone version of the logic
above (depends on this repo's `_vendored_matmul_timing.py`, `tuned-configs/`
style convention, same as `compare_default_vs_tuned.py` at the repo root) —
`isolated-benchmark-job.yaml` embeds a fully self-contained copy directly
(plus `17`'s round-2 JSON config files), matching `01`'s own
`tune-and-compare-job.yaml` pattern, since the pod only has `/tmp` to work
with. Plain `Job` (not `TrainJob`, no Kueue queue needed structurally,
though this cluster's Kueue webhook auto-suspended it anyway on the first
run — label `kueue.x-k8s.io/queue-name=gpu-autotune` after creation if it
comes up `Suspended`, or include the label in the manifest from the start
as the cycling variant below does).

A second pass, `compare_default_vs_cache_aware_cycling.py` /
`isolated-benchmark-cycling-job.yaml`, re-runs the identical 175-point grid
under cache-cycling pressure (17's `CyclingLaunchCounter` design, 64 `B`
copies, reset to the same sequence for both configs at each point) instead
of the always-warm methodology above — the regime the fix actually
targets, where a clean win is expected rather than merely hoped for. See
below once run.

Raw output: `isolated-benchmark-results.log` (all 175 points, all 5 shapes).
