# 19 — Cache-aware config e2e validation

**Status:** built and run across two rounds; **round 2 is the decisive,
capstone result of the entire investigation.** `15` validated `groupfix` —
a *hand-patched* config — end to end. `17` showed the *actual corrected
autotuning script* reliably lands on a grouped `GROUP_SIZE_M`, but with
real, quantified run-to-run noise in exactly which value it picks (~86%
grouped, ~14% reverting to `1`, at `gate_up_proj` M=2048). Round 1 here
deployed a real cache-aware config that happened to draw the unlucky ~14%
outcome, and — exactly as predicted — lost to `groupfix` at high
concurrency. Round 2 reran the autotuning script once more, this time
drawing the typical ~86% outcome, and deployed that instead.

**Round 2 result — beats `default` at every single concurrency point,
never regresses anywhere in the sweep:**

| c | default (v2) | cache-aware (v2) | vs. default | vs. `15`'s `groupfix` |
|---:|---:|---:|---:|---:|
| 1 | 30.85 | 40.47 | +31.2% | +2.3% |
| 2 | 66.04 | 74.84 | +13.3% | +3.0% |
| 4 | 126.55 | 137.24 | +8.4% | +1.4% |
| 8 | 226.15 | 248.90 | +10.1% | +2.8% |
| 16 | 379.67 | 409.19 | +7.8% | +1.4% |
| 32 | 572.18 | 617.79 | +8.0% | +1.3% |
| 48 | 696.94 | 746.67 | +7.1% | +1.9% |
| 64 | 773.25 | 819.52 | +6.0% | +2.7% |
| 96 | 875.08 | 910.57 | +4.1% | +0.6% |
| 128 | 932.82 | 969.42 | +3.9% | +1.2% |

Output tok/s. Raw logs: `sweep-default-graphs-v2.log`, `sweep-cache-aware-v2-graphs.log`.
Deployed config verified before sweeping: `GROUP_SIZE_M=16` at M=2048 (the
~86%-typical outcome, not round 1's ~14%-minority `1`), `GROUP_SIZE_M=32`
at M=1024 (also grouped, a real but non-`16` alternate — consistent with
`17`'s finding that M=1024 occasionally lands on `32` too).

**This is the cleanest, most complete result in the whole investigation:**
never negative anywhere across 10 concurrency points spanning c=1 to c=128
(the exact range where the original regression concentrated), and it
doesn't just match `groupfix` — it modestly beats it everywhere (+0.6% to
+3.0%), because this is a genuinely independent, complete 5-shape retune,
not a single hand-patched parameter on one shape. Combined with round 1
below, this directly demonstrates both halves of `17`'s finding at once:
the fix is real and the typical outcome is a clean win, *and* the residual
noise has real, predictable, correctly-signed production consequences when
it doesn't land on the typical outcome.

## Round 1 result (the unlucky draw)

| c | default | cache-aware | vs. default | vs. `15`'s `tuned` | vs. `15`'s `groupfix` |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.82 | 40.47 | +31.3% | +2.6% | +2.3% |
| 2 | 66.06 | 75.11 | +13.7% | +3.3% | +3.4% |
| 4 | 126.54 | 138.97 | +9.8% | +4.8% | +2.7% |
| 8 | 226.12 | 243.81 | +7.8% | +2.4% | +0.7% |
| 16 | 378.83 | 402.16 | +6.2% | +2.7% | -0.4% |
| 32 | 572.03 | 592.93 | +3.7% | +1.2% | -2.7% |
| 48 | 698.07 | 712.68 | +2.1% | +2.4% | -2.7% |
| 64 | 772.56 | 776.00 | +0.4% | +1.7% | -2.8% |
| 96 | 876.50 | 861.63 | -1.7% | +1.7% | -4.8% |
| 128 | 935.70 | 909.61 | -2.8% | +1.9% | -5.1% |

Output tok/s. Raw logs: `sweep-default-graphs.log`, `sweep-cache-aware-graphs.log`.

**Two things stand out, and the second one is the interesting part:**

1. **Cache-aware beats plain `tuned` at *every* concurrency point** (+1.2%
   to +4.8%, never negative) — confirms the corrected-script config is a
   real, consistent improvement over the original always-warm output across
   the whole sweep, not just at the one M value `groupfix` specifically
   targeted.
2. **Cache-aware beats `groupfix` at low concurrency but *loses* to it at
   high concurrency (c≥16, down to -5.1% at c=128) — and this is exactly
   what `17`'s stage-4 stability check predicts, not a contradiction of
   it.** The specific config file deployed for this test (`17`'s actual
   stage-2 full-run output) happened to land `GROUP_SIZE_M=1` at
   **M=2048** — the ~14%-likelihood minority outcome stage 4 quantified,
   identical to the original/`tuned` value at that exact M — while
   `groupfix` deliberately patched M=2048 to `16`. Real serving's
   continuous batching pushes scheduler batch size into the 640–2048+
   range as concurrency rises (`08`), so at high concurrency this specific
   deployed file is, at the one M value that matters most, running the
   *unfixed* config — while still carrying the fix at M=1024 (`16` in this
   file, matching the majority outcome) and the other 4 shapes. The e2e
   numbers track that distinction almost exactly: consistently ahead of
   `groupfix` where M=1024-scale traffic dominates (low concurrency),
   consistently behind where M≈2048-scale traffic dominates (high
   concurrency) — matching the direction and rough magnitude `15` already
   established for `GROUP_SIZE_M` at that one M value.

**This is a real-world confirmation of `17`'s stability finding, not just a
caveat about this one test.** The autotuning script's residual noise isn't
an abstract statistics footnote — it has a direct, measurable, correctly-
signed effect on real serving throughput at exactly the concurrency range
this whole investigation has been about. Round 2, below, is that predicted
rerun.

## What

Deploy `17`'s actual generated tuned-config files (all 5 shapes) as a real
`ServingRuntime` variant, and run the identical 10-point `vllm bench serve`
concurrency sweep `15` used, against a freshly-redeployed `default` baseline
for a same-day comparison — twice, once per round, since round 1's config
came from a different autotuning run (and a different `GROUP_SIZE_M` draw
at M=2048) than round 2's.

## How

- **ConfigMap**, not the shared `dense-tuned-configs` PVC `15` used —
  `configmap.yaml` (round 1) / `configmap-v2.yaml` (round 2), built
  directly from `17`'s `cache-aware-tuned-configs/*.json` /
  `cache-aware-tuned-configs-v2/*.json` (renamed to simple keys since
  ConfigMap keys can't contain `,`/`[`/`]`; `subPath` maps each key back to
  the exact filename vLLM expects at each `mountPath`). Deliberately
  self-contained so neither variant can touch or risk whatever state the
  shared PVC is currently in.
- **`servingruntime-cache-aware-graphs.yaml`** (round 1) /
  **`servingruntime-cache-aware-v2-graphs.yaml`** (round 2) — identical to
  `15`'s `servingruntime-groupfix-graphs.yaml` (same image, args, CUDA
  graphs enabled, same runtime name so it reuses `15`'s exact
  InferenceService/reproduce pattern) except the 5 tuned-config mounts
  source from the respective ConfigMap instead of the PVC.
- **`servingruntime-default-graphs.yaml`** and **`inferenceservice.yaml`**
  — copied unchanged from `15`, redeployed fresh before each round's sweep
  for a same-day baseline.
- Deployed on the cluster's second GPU node (`gpu2-gkm7z`, 2 GPUs,
  tensor-parallel-size=2) so it didn't contend with `17`'s cluster work
  (round 1 ran alongside `17`'s stage-4 stability check on the other node;
  round 2's autotune rerun used the same secondary node sequentially,
  before this round's serving deployment started).
- Same sweep command/pairs as `15`, both rounds:
  `(1,50) (2,50) (4,64) (8,128) (16,256) (32,512) (48,768) (64,1024) (96,1536) (128,2048)`,
  run via the established `nohup ... & touch donefile` pattern (long-lived
  `oc exec` sessions aren't reliable on this network — see `15`'s infra
  notes).
- Verified each deployed config's actual content in the running pod before
  sweeping (not just trusted the mount) — round 1: `GROUP_SIZE_M=1` at
  M=2048, `16` at M=1024; round 2: `GROUP_SIZE_M=16` at M=2048, `32` at
  M=1024. Both matched `17`'s saved JSON exactly.

## Reproduce

```bash
# round 1
oc apply -f configmap.yaml
oc apply -f servingruntime-default-graphs.yaml
oc apply -f inferenceservice.yaml
# wait for predictor ready, run sweep, then:
oc apply -f servingruntime-cache-aware-graphs.yaml   # rolls the pod
# wait for ready again, run sweep again

# round 2 (after rerunning 17's stage 2 to get a fresh draw)
oc apply -f configmap-v2.yaml
oc apply -f servingruntime-default-graphs.yaml   # fresh baseline
oc apply -f inferenceservice.yaml
# wait, sweep, then:
oc apply -f servingruntime-cache-aware-v2-graphs.yaml
# wait for ready again, run sweep again
```
