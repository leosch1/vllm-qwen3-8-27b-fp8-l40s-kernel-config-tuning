# 25 — L2-flush autotune rerun (generalizing the cache-aware fix)

**Status: complete.**

**Headline result:** the full production-scope rerun (5 shapes × 18 `M`
values, real `save_configs()`/`main()` driver, same deployment pattern as
`17` stage 2) completed cleanly in 76 minutes with zero errors. The
`L2CacheSize` device-properties attribute isn't exposed by this cluster's
PyTorch build, so the documented 128MiB fallback fired as designed (logged,
not silently swallowed) — the mechanism itself is unaffected, since the
fallback is deliberately sized well above any current data-center GPU's L2.

At the large-`M` anchors that actually matter (`08`'s established `M≳512`
regression zone, `n=30`: 5 shapes × 6 `M` values `{512,1024,1536,2048,3072,4096}`):

| tuner | `GROUP_SIZE_M=1` rate |
|---|---:|
| original (always-warm) | 16/30 (53%) |
| `17`'s 64-copy cycling fix | 13/30 (43%) |
| this experiment's L2-flush fix | 12/30 (40%) |

Both fixes move the tuner away from `GROUP_SIZE_M=1` by a similar amount,
landing within 3 percentage points of each other — the general,
model-independent flush mechanism reproduces the essential character of
`17`'s model-specific 64-copy fix at full production scope, not just at
the one point (`gate_up_proj` M=2048) prototyped first. Neither fix
eliminates `GROUP_SIZE_M=1` entirely: both still land on it in roughly
2 out of every 5 large-`M` points, consistent with `17`'s own finding that
this is a genuine minority outcome of the search's single-pass, no-repeat
benchmarking methodology, not something either cache-locality fix is
expected to fully suppress.

**But exact per-point agreement is much weaker than that headline number
suggests**, and this is the important nuance: across all 90 (shape, `M`)
points, only 13 are byte-identical between the two methods, 17 more agree
on direction (both grouped, or both `1`) but disagree on the *specific*
`GROUP_SIZE_M` value, and 60 differ in some other parameter too
(`BLOCK_SIZE_M/N`, `num_warps`, `num_stages`) — especially at small `M`,
where this project's own L2-capacity reasoning (`16`) already predicts
near-ties across many configs regardless of which cache methodology is
used. Whether these disagreements reflect a real performance difference or
just noise between close competitors (`17`'s own finding: `16`/`32`/`64`
often perform within noise of each other) is exactly what the kernel-level
cross-check below is for.

Raw config diff: ad hoc, output captured above — reproducible directly
against `l2-flush-tuned-configs/`, `../17-cache-aware-autotune-rerun/cache-aware-tuned-configs-v2/`,
and `../../../tuned-configs/`.

## Kernel-level cross-check: does the disagreement matter?

Ran `compare_cyc_vs_flush_cycling_fully_randomized.py` (deployed as
`isolated-benchmark-exp24-cyc-vs-flush-job.yaml`) — all 5 shapes × 18 real
anchor `M` values × 2 config-sources (`cyc`, `flush`) × 2000 iterations =
360,000 launches, generated as one flat list and executed in a single
shuffled random order with a uniformly-random one-of-64 weight-copy chosen
per launch, this project's established cache-cycling fully-randomized
methodology (`01`/`20`) — an independent, third measurement that doesn't
favor either tuning method's own internal ranking. Full per-shape tables:
`cyc-vs-flush-cycling-fully-randomized-results.log`.

**Result: the two methods' winning configs perform almost identically,
across the board.**

| | n | mean \|Δ\| | max \|Δ\| |
|---|---:|---:|---:|
| all 90 (shape, M) points | 90 | 1.30% | 6.50% |
| points where `GROUP_SIZE_M` **matches** | 28 | 1.25% | 6.50% |
| points where `GROUP_SIZE_M` **differs** | 62 | 1.32% | 6.40% |

The mean absolute deviation is **the same** (within noise) whether the two
methods agree on the exact `GROUP_SIZE_M` value or not — direct
confirmation that the widespread per-point disagreement found above is
exactly what `17` already characterized: several `GROUP_SIZE_M` values
(`16`/`32`/`64`) are close competitors that perform within a percent or two
of each other, not a sign that either fix is picking meaningfully worse
configs than the other. Only 4/90 points exceed a 5% gap, and none exceed
6.5%, dwarfed by the actual effect size this whole investigation is about
(the original tuner's `GROUP_SIZE_M=1` pick costing ~8-20% at the shapes/M
values `14`/`16` measured).

**Does reverting to `GROUP_SIZE_M=1` at large `M` actually cost
performance, in this direct head-to-head?** At the 9 large-`M` (`≥512`)
points where exactly one method reverted to `1` while the other picked a
grouped value:

| shape | M | cyc | flush | Δ (flush vs cyc) | who reverted to 1 | did reverting lose? |
|---|---:|---:|---:|---:|---|---|
| gate_up_proj | 3072 | 1 | 16 | −4.5% | cyc | yes |
| out_proj | 1024 | 1 | 16 | −6.4% | cyc | yes |
| qkv_proj | 4096 | 32 | 1 | +0.7% | flush | yes |
| out_proj | 3072 | 1 | 32 | −0.7% | cyc | yes |
| qkv_proj | 1024 | 1 | 32 | −0.3% | cyc | yes (barely) |
| in_proj_qkvz | 2048 | 32 | 1 | +0.0% | flush | no (tie) |
| down_proj | 1536 | 64 | 1 | −1.0% | flush | no |
| down_proj | 2048 | 32 | 1 | −1.6% | flush | no |
| out_proj | 2048 | 1 | 16 | +0.7% | cyc | no |

5/9 (a rough coin flip) — consistent with the reversion to `1` being a
genuine but *usually small-magnitude* minority outcome of single-pass
noise, not a systematic loss. The two clearest real wins for staying
grouped (gate_up_proj M=3072, out_proj M=1024, both −4.5% to −6.4%) show
the effect is real when it does show up; the rest are within a percent or
two either way.

## Conclusion

The general, model-independent L2-flush mechanism is a valid replacement
for `17`'s model-specific 64-copy-cycling fix: run at full production
scope, it reproduces the same overall shift away from `GROUP_SIZE_M=1`
(40% vs. 43% residual-`1` rate, within the noise floor both methods
already have), and where the two methods pick different exact configs,
this project's own independent isolated-kernel measurement shows those
picks perform within ~1.3% of each other on average — i.e. the
disagreement is real but inconsequential, exactly the "close competitors"
pattern `17` already established for the grouped values themselves. This
is a solid basis for the upstream PR discussed earlier: flush L2 directly
(sized from the device's own reported capacity, with a documented
generous fallback), drop the model-specific copy count entirely, and treat
this as validated at full scope rather than a single-point prototype.

## Reproduce

## Why this experiment

`17` fixed the real tuning script's blind spot (`14`/`16`: `tune()` reuses
one fixed `B` across all 1280 candidate configs, so it never sees a
config's true cache-locality cost) by cycling `B` through 64
independently-random copies, one per real transformer layer count. That
patch worked and was validated end-to-end (`19`, `21`/`22`) — but the
`N_WEIGHT_COPIES=64` constant is arbitrary from a general standpoint: it's
sized to *this* model's layer count, not to the actual physical constraint
that makes the effect appear (the working set needs to exceed the GPU's L2
capacity). A different model with larger per-layer weights, or a different
GPU with a larger L2, could silently fall back into the always-warm
regime with that same hardcoded `64` — not something an upstream vLLM PR
should ship.

The general fix: don't proxy L2 eviction through copy count at all — flush
L2 directly, once, right before every timed kernel launch, using a buffer
sized from the actual device's reported L2 capacity
(`torch.cuda.get_device_properties(device).L2CacheSize`). This needs no
per-model tuning, needs only one `A`/`B` pair (not 64), and targets the
physical constraint directly instead of approximating it.

This experiment checks two things before treating that as a real
candidate for an upstream PR:

1. **Does a full, real rerun of the entire tuning script (5 shapes × 18 `M`
   values, not just the one point `17` prototyped first) converge on
   approximately the same configs as `17`'s 64-copy cycling version** —
   confirming the two methodologies aren't just theoretically equivalent
   but actually produce the same practical recommendation end to end?
2. **Do the two methodologies' *own* kernel-level timing measurements roughly
   agree** — i.e. if you take the flush-winner and the cycling-winner
   configs (at the points where they differ) and benchmark them against
   each other, are they within noise of each other, consistent with `17`'s
   own finding that `16`/`32`/`64` are close competitors under its
   methodology?

## What (planned)

1. **`l2_flush_tune.py`** — standalone reference copy, same shape as `17`'s
   `cache_aware_tune.py`, with `CyclingLaunchCounter`/64-copy allocation
   replaced by: one `A`/`B` pair (real script's original allocation
   pattern, unchanged), one `flush_buf` sized to `1.5×` the device's
   reported `L2CacheSize` (falling back to a documented, generous 128MiB
   constant with a printed warning if the attribute isn't present on this
   PyTorch build), and a `flush_l2()` call (`flush_buf.zero_()`) inserted
   immediately before every *timed* iteration only — not before the 5
   warmup iterations, which exist purely to trigger Triton JIT compilation
   and don't need to reflect a cold cache. Everything else (search space,
   5-warmup/10-timed CUDA-event bracketing, the documented `/10` timing
   bug, `OutOfResources` handling) copied unchanged, same as `17`.
2. **Full 5-shape × 18-`M` rerun**, using the real `save_configs()`/`main()`
   driver exactly like `17` stage 2's `trainjob-full.yaml` (same real
   production script, same one function patched, same `TrainJob`
   deployment pattern) — not just the 2-value pilot scope, since the whole
   point here is checking whether the general method reproduces `17`'s
   result *at production scope*, not just at the one point already
   prototyped.
3. **Diff the resulting configs** against both the original (always-warm)
   tuned-configs and `17`'s `cache-aware-tuned-configs-v2/` (the version
   validated end-to-end in `19`/`22`), per shape per `M`.
4. **Kernel-level cross-check**: for every `(shape, M)` point where the
   flush-based winner and the cycling-based winner disagree on
   `GROUP_SIZE_M` (or any other parameter), directly benchmark both
   configs against each other using this project's already-established
   cache-cycling isolated-kernel methodology (`01`/`20`'s
   `compare_default_vs_tuned_cycling_fully_randomized.py`-style harness),
   to see whether the disagreement reflects a real performance difference
   or just noise between close competitors — consistent with `17`'s own
   finding that `16`/`32`/`64` often perform within noise of each other.

## How

Reusing the exact deployment pattern already proven in `17`: a `TrainJob`
on the same `runtimeRef`/image/Kueue queue (`gpu-autotune`), single GPU,
embedding a fully self-contained script (no external file dependencies,
matching production's own `TrainJob` pattern). Results captured both to
stdout (`=== CONFIG_JSON_BEGIN ===` / `=== CONFIG_JSON_END ===` markers,
`17`'s own pattern for surviving pod cleanup) and to the real
`save_configs()` JSON files, copied out via `oc cp` before teardown.

## Reproduce

```bash
# full 5-shape x 18-M rerun (real production script, one function patched)
oc apply -f trainjob-full.yaml
# watch: oc get pods -n enterprise-ai | grep l2-flush
# logs:  oc logs <pod> -n enterprise-ai | grep -v 'it/s\]'
# (extract each shape's CONFIG_JSON_BEGIN/END block from the log into
#  l2-flush-tuned-configs/ -- see this README's history for the exact
#  extraction one-liner, or just re-derive it: same pattern 17 uses.)

# kernel-level cross-check: cyc-tuned vs flush-tuned configs, head to head
oc apply -f isolated-benchmark-exp24-cyc-vs-flush-job.yaml
```

## Files

- `l2_flush_tune.py` — standalone reference version (2-value pilot scope,
  `gate_up_proj` only, M∈{1024,2048}), same structure as `17`'s
  `cache_aware_tune.py`.
- `trainjob-full.yaml` — the actual deployed job: real production
  `benchmark_w8a8_block_fp8.py`, `tune()`'s weight-tensor handling replaced
  with the L2-flush mechanism (marked `EXP24 CHANGE`), full 5-shape × 18-`M`
  rerun via the real `save_configs()`/`main()` driver.
  `l2-flush-tuned-configs/` — the 5 resulting standard-format JSON files.
- `compare_cyc_vs_flush_cycling_fully_randomized.py` /
  `isolated-benchmark-exp24-cyc-vs-flush-job.yaml` — the kernel-level
  cross-check, generated by embedding both tuning methods' real config
  JSONs and this project's established cache-cycling fully-randomized
  harness into a throwaway Job.
  `cyc-vs-flush-cycling-fully-randomized-results.log` — full per-shape
  results tables.
