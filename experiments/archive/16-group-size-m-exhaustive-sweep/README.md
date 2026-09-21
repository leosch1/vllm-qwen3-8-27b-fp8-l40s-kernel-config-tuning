# 16 — GROUP_SIZE_M exhaustive sweep (and its own methodology correcting itself)

**Status:** built and run in three successive stages, the last of which
**reverses** the first's headline finding. Kept as a single experiment
rather than split up because the correction *is* the finding: an
apparently clean, statistically-"real" result turned out to be a
measurement artifact of this project's own round-robin methodology, caught
only by pushing the isolation further. Read top-to-bottom — the middle
sections are wrong, on purpose, left in for the record.

**Bottom line:** at M=1024 (`gate_up_proj`, N=17408, K=5120), the most
isolated measurement available (`ncu` hardware counters, deeply pre-warmed,
no condition-switching) shows `GROUP_SIZE_M=1` **losing** to grouped values
by ~11.8%, with every metric (DRAM bytes moved, L2 hit rate, achieved DRAM
throughput%, SM throughput) pointing the same direction — plain textbook
cache-locality behavior. A from-scratch timing rerun that eliminates
cross-`M` tensor switching (but still round-robins across `GROUP_SIZE_M`
values) reproduces that ordering, `GROUP_SIZE_M=1` losing by 1.87%. The
original 72-condition exhaustive sweep, which round-robins across 18
different `M` values' tensors, found the **opposite** — `GROUP_SIZE_M=1`
winning by 5.7% at the same `M`. That sweep's design is now understood to
be the source of the discrepancy, not a real effect: cycling through 18
different `M`s' tensors means every block starts cold, and that cold-start
transient biased the result. See "Correction" below.

This does *not* undermine experiment 14's own "`tuned` beats `groupfix` by
2.1%" result — that comparison answers a different question (is the
tuner's choice self-consistent with the tuner's *own* idealized testing
conditions, which it is, as expected) than this experiment does (is
`GROUP_SIZE_M=1` actually faster under a rigorously isolated, no-
switching methodology, which it isn't). See "Implication for experiment
14" below for why these are complementary, not conflicting — and one
caveat on the precise 2.1% magnitude that does still apply.

## Stage 1 — does GROUP_SIZE_M follow a real trend across shapes/M?

`14` found tuned-config JSON collapsing to `GROUP_SIZE_M=1` for every
`gate_up_proj` M-bucket ≥512, and speculated this might be a general
pattern. Pulling the full `GROUP_SIZE_M` column for all four GEMM shapes
in this deployment across their entire tuned M-grid did **not** support a
clean general trend — a pooled grid-size-bucket table initially looked
like one, but broke down under a per-shape check: the "high grid"
bucket (grid size 1000–10000) was dominated by `gate_up_proj`'s own 5
data points, with only 7 independent points from the other three shapes,
2 of which disagreed with the supposed pattern. Even the `gate_up_proj`-only
pattern (a real, visible run of `GROUP_SIZE_M=1` at M≥512) could plausibly
be correlated noise from one tuning session rather than a real per-shape
law — nothing in the tuned-config JSON alone can distinguish "real
trend" from "noisy tie-break that happened to autocorrelate," since each
M is tuned once, no repeats. This is what motivated stage 2: settle it
empirically, not by pattern-matching the tuner's own (single-pass,
no-repeat) output.

## Stage 2 — the exhaustive sweep (`group_size_m_exhaustive.py`)

For every M in `gate_up_proj`'s tuned-config grid (18 values, 1–4096),
takes that M's actual winning `BLOCK_SIZE_M/N/K`/`num_warps`/`num_stages`
and varies only `GROUP_SIZE_M` across the tuner's own real candidate set
`{1, 16, 32, 64}` — same isolated, tensor-reused-across-conditions
methodology the real tuning script uses, but with round-robin
counterbalanced repeats (16 repeats × 50 iters/block = 800
iterations/condition) so a per-condition standard error is actually
computable, unlike the tuner's own single-pass 10-iteration mean.

**Result (18 M × 4 GROUP_SIZE_M = 72 conditions):**

- Tuner's pick was the fastest in our test: **9/18**
- Tuner's pick tied with fastest (within a rough 2×SE band): **7/18**
- Tuner's pick measurably beaten by another candidate: **2/18**
- A real, non-noise "`GROUP_SIZE_M=1` wins" signal across M=512–3072
  (matching the tuner's own choice there), a tie at M=2048, and a
  reversal at M=4096 (`GROUP_SIZE_M=64` beat `GROUP_SIZE_M=1` by ~11.2%).
- Specifically at **M=1024**: `GROUP_SIZE_M=1` = 563.05µs vs.
  `GROUP_SIZE_M=64` = 595.38µs — `GROUP_SIZE_M=1` **wins by 5.7%**, a gap
  well outside the block-to-block noise band.

Full per-condition log: `exhaustive_sweep_summary.log`.

At this point the working theory was pure L2-cache-capacity reasoning: at
low M, `A` (M×K, fp8) and `B` (N×K, fp8, fixed at ~89MB for this shape)
both fit inside the L40S's 96MB L2 together, so which `GROUP_SIZE_M` is
picked shouldn't matter — there's nothing to evict either way, and choice
of grouping is noise. Once the working set exceeds L2, grouping should
start to matter, and (per the standard tutorial argument) *grouped*
values should start winning as M grows, not `GROUP_SIZE_M=1`.

**A first pass at "where does A+B stop fitting in L2" only counted `A`
and `B`, omitting the output tensor `C`.** Recomputed including `C`
(M×N×2 bytes, bf16) — `A+B+C` exceeds L2 by M=128, which does *not*
line up with the onset of a real, measurable `GROUP_SIZE_M` effect
(~M=512 in this sweep). Conclusion at the time: `C` is write-only and
never read back mid-kernel, so it likely doesn't compete for L2
read-cache residency the way `A`/`B` do — `A+B` alone (not `A+B+C`)
remains the better, if still imperfect, predictor of where grouping
should start to matter.

**This left an unresolved, theoretically backwards result: `GROUP_SIZE_M=1`
(no grouping) was the one *winning* at the M values where the working set
was large enough for cache locality to matter (512–3072), while grouped
values only won back at the very largest M (4096).** Standard cache-locality
theory predicts the opposite — more grouping should help more as the
working set grows, not less. That contradiction, not just idle curiosity,
is what justified spending cluster time on `ncu`.

## Stage 3 — ncu ground truth (`ncu_probe.py`)

Nsight Compute, Application Replay mode, single-launch probes
(`--launch-skip`/`--launch-count 1`) after 10 dedicated warmup launches
per `GROUP_SIZE_M` value, at the one M (1024) where stage 2's signal was
cleanest. Metrics: `gpu__time_duration.sum`,
`sm__throughput.avg.pct_of_peak_sustained_elapsed`,
`dram__throughput.avg.pct_of_peak_sustained_elapsed`,
`lts__t_sector_hit_rate.pct`, `dram__bytes_read.sum`,
`dram__bytes_write.sum`.

| `GROUP_SIZE_M` | duration | DRAM read | DRAM write | DRAM throughput% | L2 hit rate | SM throughput |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 581.3µs | 170.4MB | 32.7MB | 42.4% | 90.4% | 50.6% |
| 16 | 520.1µs | 112.8MB | 13.2MB | 29.4% | 93.6% | 60.5% |
| 32 | 520.9µs | 112.8MB | 13.3MB | 29.4% | 93.6% | 60.6% |
| 64 | 518.6µs | 112.8MB | 12.9MB | 29.4% | 93.6% | 60.5% |

**Flatly contradicts stage 2: `GROUP_SIZE_M=1` is 11.8% *slower*, not
5.7% faster.** Every metric agrees and rules out the one alternative
theory that could have rescued `GROUP_SIZE_M=1`: greater memory-level
parallelism from spreading blocks across more distinct tiles at once
(which would predict *higher* achieved DRAM throughput% for
`GROUP_SIZE_M=1`). Instead `GROUP_SIZE_M=1` shows lower DRAM
throughput%, more total bytes moved, a lower L2 hit rate, and lower SM
throughput — plain, uncomplicated cache-locality cost, textbook-shaped in
every column. A back-of-envelope check also ruled out one other candidate
explanation (per-block `div`/`mod` overhead from the grid-swizzle
remapping, which `GROUP_SIZE_M=1` skips): the hypothesized savings
(~150ns) are roughly 1000× too small to explain the ~32,000ns/iteration
gap stage 2 reported at M=1024.

Full data: `ncu_probe_results.log`.

## Stage 4 — the decisive test (`group_size_m_single_m.py`)

Two designs now disagreed and only one causal difference separated them:
stage 2 round-robins across 18 *different M values'* tensors (a
completely different, much larger A/B pair every block), while the `ncu`
probe uses one fixed A/B pair with no switching at all. This script
isolates that variable directly: **M=1024 only, one A/B pair allocated
once for the entire script and never touched by any other M**,
round-robinning *only* across the 4 `GROUP_SIZE_M` candidates (32
repeats × 50 iters/block = 1600 iterations/condition — double stage 2's
per-condition sample).

```
=== Results: mean +/- stderr (us) ===
  GROUP_SIZE_M=  1:   597.62 +/-  3.35 us  (n=32 blocks x 50 iters)
  GROUP_SIZE_M= 16:   586.68 +/-  2.50 us  (n=32 blocks x 50 iters)
  GROUP_SIZE_M= 32:   590.63 +/-  2.54 us  (n=32 blocks x 50 iters)
  GROUP_SIZE_M= 64:   594.85 +/-  2.65 us  (n=32 blocks x 50 iters)

Winner: GROUP_SIZE_M=16
GROUP_SIZE_M=1 vs winner: +1.87%
```

Combined SE(`GM=1`, `GM=16`) ≈ 4.18µs; the 10.9µs gap is ~2.6× that,
outside this project's usual rough 2×SE noise band — a real, if modest,
effect, and critically **in the same direction as `ncu`**, not stage 2.

| measurement | design | GROUP_SIZE_M=1 at M=1024 |
|---|---|---|
| Stage 2 (`group_size_m_exhaustive.py`) | round-robin across 18 different Ms | **wins** by 5.7% |
| Stage 4 (`group_size_m_single_m.py`) | round-robin across only 4 GROUP_SIZE_M values, one M, one tensor pair | **loses** by 1.87% |
| Stage 3 (`ncu_probe.py`) | one fixed config at a time, 10 dedicated warmup launches, no switching | **loses** by ~11.8% |

Full data: `single_m_results.log`.

## Correction

**Stage 2's finding — that `GROUP_SIZE_M=1` measurably wins at
M=512–3072 — does not hold up and should not be cited as a real effect.**
The exhaustive sweep's round-robin design cycles through 72 conditions
spanning 18 different `M` values, each with its own, much larger A/B
tensor pair; every 50-iteration block for one `(M, GROUP_SIZE_M)`
condition is therefore immediately preceded by a block touching a
*completely different* M's tensors, meaning that condition starts cold
every single repeat and only partially re-warms within its own 50
iterations before being interrupted again. That is a real, systematic
bias, not a hypothetical one — removing exactly that one variable (stage
4) reverses the M=1024 result, and an independently-instrumented,
zero-switching measurement (`ncu`, stage 3) shows the same reversed
ordering with a much larger, hardware-counter-backed margin.

The size gradient across the three measurements (wins by 5.7% → loses by
1.87% → loses by 11.8%, as switching-between-conditions is progressively
removed) is itself informative: it suggests that *even* switching
between differently-compiled `GROUP_SIZE_M` kernels on the *same* tensor
pair (stage 4's remaining round-robin) still dilutes the true steady-state
gap somewhat, relative to a fully isolated, no-switching measurement
(`ncu`). Any timing methodology in this investigation that round-robins
between different compiled kernel variants — not just different tensors —
should be read with that residual bias in mind.

**This reopens, rather than answers, the stage-2-motivated puzzle "why is
`GROUP_SIZE_M=1` preferred at high M, when cache-locality theory predicts
the opposite?"** The honest answer is: it probably isn't, in reality —
the tuned-config JSON's own preference for `GROUP_SIZE_M=1` at M≥512 may
itself be a downstream symptom of the same class of artifact. The real
tuning script (`benchmarks/kernels/benchmark_w8a8_block_fp8.py`, see
`14`) evaluates all 1280 candidate configs for a given M *sequentially, in
a single pass, with only 5 warmup + 10 timed iterations per candidate and
no repeats* — structurally the same kind of "switch to a differently-compiled
kernel, measure immediately, move on" pattern that stage 4 showed still
carries a measurable, direction-relevant bias even under otherwise ideal
conditions (same M, same tensors, no repeats needed for it to matter,
just fewer than the 32 blocks used here). This is a plausible mechanism,
not something separately tested here — it hasn't been directly verified
that the tuner's own within-M search is biased toward `GROUP_SIZE_M=1` for
this reason (as opposed to `GROUP_SIZE_M` genuinely mattering less at
some M values than others, which the L2-capacity theory still broadly
supports for *low* M). It's flagged as the current best explanation for
stage 2's M≥512 pattern, not a settled conclusion.

## Implication for experiment 14

`14`'s `same-B` comparison (`tuned` 1219.5µs vs. `groupfix` 1245.5µs,
`tuned` winning by 2.1%, at M=2048) is **not** undermined by this
experiment's correction — it answers a different question. That
comparison deliberately reproduces the tuning script's *own* idealized
measurement regime (one fixed, always-warm tensor pair, no cross-tensor
eviction) to check whether the tuner's `GROUP_SIZE_M=1` choice is
self-consistent with what the tuner itself measured. It is, and that's
expected: it's a mechanism sanity-check, not a claim about true
isolated/steady-state performance. This experiment (`16`) measures the
latter — real cache-locality cost under a rigorously isolated, no-
condition-switching methodology — and correctly finds the opposite
ordering. Both are true simultaneously and together make the train/serve-
mismatch story sharper, not weaker: the autotuner's choice is locally
consistent with its own (idealized) objective, and that objective itself
doesn't reflect real steady-state performance even before considering
real serving's contended cache.

The one thing worth flagging precisely, not retracting: `14`'s `same-B`
design *does* round-robin between three differently-compiled kernel
variants (`default`/`tuned`/`groupfix`) every 250-iteration block, at a
fixed M=2048 (no cross-M switching — the specific vulnerability stage 2
had). Stage 4 above shows that kind of between-kernel-variant switching,
even with the tensor pair held fixed, is enough on its own to measurably
dilute a true effect's magnitude (though not necessarily its direction —
stage 4's `GM=1`-loses-by-1.87% is smaller than `ncu`'s `GM=1`-loses-by-
11.8%, but still the same sign). So the qualitative conclusion ("tuned
wins under tuner-like conditions") is expected to hold, but the precise
"2.1%" figure may not be exact — a `16`-style two-condition-only
round-robin (or an `ncu` probe) at M=2048 would give a more trustworthy
number. Flagged as follow-up work, not yet done.

**What this does *not* touch:** experiment 15's e2e validation (`groupfix`
never regressing vs. `default`, closing an increasing gap against `tuned`
across a real concurrency sweep) is a completely different kind of
measurement — real serving traffic, real CUDA graphs, no isolated-kernel
round-robin methodology at all — and is unaffected by any of this. If
anything, this correction makes the overall story a little sharper: the
tuner's own choice of `GROUP_SIZE_M=1` may not even be "correct-for-a-narrow-
objective-that-doesn't-match-production" so much as an artifact of the
tuner's own under-powered, no-repeat, sequential search methodology —
doubly unreliable, not just narrowly optimized.

## Reproduce

```bash
# All three scripts share the same pod/environment setup as 14/15 — copy
# alongside ../../../_vendored_matmul_timing.py and this shape's tuned-config
# JSON onto a GPU pod with vllm's site-packages on PYTHONPATH.
cd /tmp
python3 group_size_m_exhaustive.py     # stage 2 (produces the now-corrected result)
python3 group_size_m_single_m.py       # stage 4 (the decisive re-test)

# stage 3 needs ncu installed and DCGM paused for exclusive counter access
# (dcgmi profile --pause on the node, --resume when done) — see 09's nsys
# pod setup for the equivalent pattern.
ncu --replay-mode application --launch-skip <N> --launch-count 1 \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,\
dram__bytes_read.sum,dram__bytes_write.sum \
    python3 ncu_probe.py
```

## Open follow-up (not yet done)

- Repeat stage 3/4 at one or two more M values (e.g. 512, 3072) to check
  whether the reversal is general across the M=512–3072 range stage 2
  flagged, or specific to M=1024.
- Re-derive `14`'s `tuned`-vs-`groupfix` isolated comparison with a
  stage-4-style, two-condition-only round-robin (or an `ncu` probe) at
  M=2048 to get a trustworthy number for that specific claim.
- Cluster cleanup: delete the `ncu-groupsize-probe` job in the
  `enterprise-ai` namespace; resume DCGM on `gpu-node-b`
  (`dcgmi profile --resume`, pod `nvidia-dcgm-kv2cg`).
