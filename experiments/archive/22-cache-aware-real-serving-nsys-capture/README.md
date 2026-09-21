# 22 — cache-aware config real-serving nsys capture

**The same real nsys hardware-counter capture as `21`, this time against
`19` round 2's actual cache-aware-tuned config** (`17`'s patched autotune
script's real, independently-generated output — not a hand patch like
`groupfix`). Answers the natural follow-up to `21`: does the *other*
validated fix hold up in the same direct, source-level re-measurement, or
was `21`'s clean result specific to `groupfix`'s narrow single-parameter
change?

## Why this experiment

`21` re-measured `groupfix` at the exact shape/M-band that originally
showed the regression and found it recovered cleanly. But `groupfix` is a
minimal, single-parameter hand patch — its launches are classifiable by
reusing `tuned`'s own grid/block signatures, since `GROUP_SIZE_M` doesn't
affect them. The cache-aware config is a different kind of fix: a real,
independently-retuned 5-shape config where `BLOCK_SIZE_M`, `BLOCK_SIZE_N`,
`num_warps`, and `num_stages` all differ from `tuned`'s at many M anchors,
not just `GROUP_SIZE_M`. It was already validated end-to-end (`19`) and in
isolation (`20`), but never against this specific real-serving,
source-level measurement — and its classification can't reuse `21`'s
shortcut, so it needed its own reverse-mapping logic built from its own
config JSONs.

## Result

Weighted-mean duration, `gate_up_proj`, M 1024–2048 band, c=64:

| config | duration | vs. `default` | vs. `tuned` |
|---|---:|---:|---:|
| `default` (`05`'s original capture) | 1111.0µs | — | −43.4% |
| `tuned` (`05`'s original capture) | 1960.95µs | +76.5% | — |
| `groupfix` (`21`, new capture) | 1038.3µs | −6.5% | −47.1% |
| cache-aware-tuned (this experiment, new capture) | 1035.0µs | **−6.8%** | **−47.2%** |

The two independently-derived fixes land within 0.3% of each other — both
comfortably below `default`, both nowhere near `tuned`'s blowout. This is a
genuinely independent confirmation: the cache-aware config wasn't
hand-patched to fix this specific number the way `groupfix` was, it fell
out of rerunning the real autotune search with its cache-locality blind
spot patched (`17`). Two structurally different fixes, arrived at two
different ways, converge on the same real-serving outcome at the exact
kernel population that started this whole investigation.

`chart-cache-aware-vs-default-tuned-groupfix.html` extends `21`'s chart
with a sixth panel (`c64_cache_aware`) — same layout, directly comparable
to all five existing panels.

## Method

Same pipeline as `21`, with one real difference in the classification step:

1. **Deploy the cache-aware config with profiling enabled.** `configmap.yaml`
   (`19` round 2's actual 5 tuned-config JSONs, from
   `17-cache-aware-autotune-rerun/cache-aware-tuned-configs-v2/`),
   `servingruntime-cache-aware-profiling.yaml` (same base as `21`'s —
   CUDA graphs on, `runAsUser:0` + `SYS_ADMIN`, dedicated ConfigMap),
   `inferenceservice.yaml` (`nvidia-nsight-profile: enabled` restored, same
   as `21`).
2. **Drive real traffic, capture, extract.** Identical to `21`:
   `nsight_operator.py profiler-start`/`profiler-stop` against the
   already-registered session, `vllm bench serve` at c=64 (500 prompts,
   256/128 random input/output len) from a separate untraced `bench-client-throwaway`
   Job hitting the predictor's raw pod IP:8080, `nsys stats --report gputrace`
   to materialize the intermediate sqlite on-pod, `extract_kernels.py` to
   pull the `_w8a8_triton_block_scaled_mm` kernel's launch rows into a CSV.
3. **Classify from the cache-aware config's own JSONs, not `tuned`'s.**
   `21`'s shortcut (reuse `tuned`'s grid/block-signature classification,
   valid only because `groupfix` doesn't touch `BLOCK_SIZE_M`/`BLOCK_SIZE_N`/
   `num_warps`) does not apply here — the cache-aware config changes all of
   those at several M anchors. `build_cache_aware_mband.py` rebuilds the
   same classification logic (`18-out-down-proj-order-disambiguation/resolve_order.py`'s
   real-program-order disambiguation for the `out_proj`/`down_proj`
   collision — a property of vLLM's execution order, independent of which
   config is deployed) from the cache-aware config's own JSONs instead.
4. **Merge into the chart's series format.** `build_series_from_durs.py`
   bins the reclassified per-launch durations into the same log-spaced bin
   edges the original chart (`05`, carried through `21`) already uses, so
   the new `c64_cache_aware` series stacks directly alongside the existing
   five.

## New gotcha found in this experiment

- **DCGM profiling and `nsys --gpu-metrics-devices` hold the same hardware
  counter lock — resuming DCGM after `21`'s cleanup blocked this
  experiment's capture.** `21`'s cleanup correctly resumed DCGM profiling
  on the node once its own capture was done. Deploying `22`'s predictor
  onto that same node then immediately crash-looped with nsys's own
  "Illegal `--gpu-metrics-devices` usage... Already under profiling" error
  — not a leftover pod or stale nsight-operator session (both checked and
  clean), but DCGM itself holding the GPU-metrics counter lock. **Fix:**
  `dcgmi profile --pause` again before deploying a new profiling-enabled
  pod on a node where DCGM was previously resumed, then delete the
  crash-looping pod to force an immediate retry (rather than waiting out
  `CrashLoopBackOff`'s growing delay). Every experiment in this project
  that both profiles with nsys *and* leaves DCGM resumed afterward needs to
  remember to re-pause it before the next nsys-profiled deployment on the
  same node.
- All three of `21`'s gotchas (nsight-injector fork-wrapping a separate
  benchmark client, Kueue gating bare Pods forever, `oc cp` silently
  truncating large files) applied identically here and were handled the
  same way — no need to relitigate them, see `21`'s README.

## Files

- `configmap.yaml`, `servingruntime-cache-aware-profiling.yaml`,
  `inferenceservice.yaml` — the profiling-enabled cache-aware-config
  deployment.
- `bench-client-job.yaml` — the separate untraced benchmark-client Job (not
  saved for `21` at the time; saved here as the reusable reference for both).
- `extract_kernels.py` — SQL extraction script, run on-pod.
- `build_cache_aware_mband.py` — classifies launches from the cache-aware
  config's own JSONs (not `tuned`'s) and recovers M-bands.
- `build_series_from_durs.py` — bins a flat per-launch duration list into
  the chart's existing log-spaced bin edges.
- `cache_aware_series.json` — the resulting `c64_cache_aware` series.
- `chart-cache-aware-vs-default-tuned-groupfix.html` — `21`'s chart,
  extended with the sixth panel.

## Cleanup

All cluster resources from this experiment (the `qwen-27b-pr-evidence`
InferenceService/ServingRuntime, `cache-aware-tuned-configs-v2-profiling`
ConfigMap, `bench-client-throwaway` Job) were deleted after this capture,
and DCGM profiling was re-paused-then-resumed correctly on the node's
GPU-engine pod (`nvidia-dcgm-xt7j6`, node `gpu-node-a`).
