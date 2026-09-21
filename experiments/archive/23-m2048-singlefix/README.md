# 23 — M=2048 single-anchor GROUP_SIZE_M fix

> Fully covered in [`experiments/03-group-size-m-patch`](../../03-group-size-m-patch/).

**The narrowest possible intervention: identical to the original `tuned`
config everywhere, for all 5 shapes and every M anchor, except one single
value** — `gate_up_proj`'s M=2048 entry, where `GROUP_SIZE_M` changes from
`1` to `16` (the value both `groupfix` and cache-aware-tuned independently
land on there). Every other large-M anchor on `gate_up_proj` (512, 1024,
1536, 3072, 4096) stays at `GROUP_SIZE_M=1`, exactly as in plain `tuned`.
Same real nsys capture methodology as `21`/`22`.

## Why this experiment

`groupfix` turned out to *not* be the single-M patch it's often described
as — it changes `GROUP_SIZE_M` at 6 different `gate_up_proj` M anchors (512
through 4096), not just the M=2048 one that originally showed the
regression. That leaves an open question: does fixing *just* the specific
M-band that showed the blowout (`08`/`09`'s "M 1024–2048" bucket) recover
it on its own, or do the other large-M anchors — also `GROUP_SIZE_M=1`,
never individually tested — also matter in real serving? `singlefix`
isolates the answer: a config that changes exactly one value out of the
entire 5-shape config.

## Result

Weighted-mean duration, `gate_up_proj`, M 1024–2048 band, c=64:

| config | duration | vs. `default` | vs. `tuned` |
|---|---:|---:|---:|
| `default` (`05`'s original capture) | 1111.0µs | — | −43.4% |
| `tuned` (`05`'s original capture) | 1960.95µs | +76.5% | — |
| **singlefix** (this experiment, new capture) | **1057.25µs** | **−4.8%** | **−46.1%** |
| `groupfix` (`21`) | 1038.3µs | −6.5% | −47.1% |
| cache-aware-tuned (`22`) | 1035.0µs | −6.8% | −47.2% |

Fixing **one value** — `GROUP_SIZE_M` at exactly the M anchor that showed
the blowout, nothing else — recovers **~74% of `groupfix`'s full
improvement** over `default` (53.8µs of `groupfix`'s 72.7µs total gain).
The remaining gap between `singlefix` and `groupfix`/cache-aware-tuned is
real but small (~1.7–2.0 percentage points), consistent with the other
large-M anchors (512, 1024, 1536, 3072, 4096) contributing a genuine but
secondary effect. The specific M-band that started this whole
investigation is overwhelmingly explained, and overwhelmingly fixed, by
that one M-band's own `GROUP_SIZE_M` value.

`chart-singlefix-vs-all.html` extends `22`'s chart with a seventh panel
(`c64_singlefix`) — same layout, directly comparable to all six existing
panels, now showing default → tuned → singlefix → groupfix →
cache-aware-tuned in one place.

## Method

Identical pipeline to `21`/`22`:

1. **Build the config.** All 5 shapes' JSONs copied byte-for-byte from
   `tuned-configs/`, except `gate_up_proj`'s `"2048"` entry, where
   `GROUP_SIZE_M` is changed from `1` to `16`. Verified programmatically:
   diffing every other entry across all 5 files against the original
   confirms zero other differences.
2. **Deploy with profiling enabled.** `configmap.yaml`,
   `servingruntime-singlefix-profiling.yaml` (same base as `21`/`22`),
   `inferenceservice.yaml`.
3. **Drive real traffic, capture, extract.** Identical to `21`/`22`:
   `nsight_operator.py profiler-start`/`profiler-stop` against the
   already-registered session, `vllm bench serve` at c=64 (500 prompts,
   256/128 random input/output len) from a separate untraced
   `bench-client-throwaway` Job, `nsys stats --report gputrace` to
   materialize the intermediate sqlite on-pod, `extract_kernels.py` to pull
   the GEMM kernel's launch rows into a CSV (536,064 launches — matches
   `21`/`22`'s magnitude).
4. **Classify by reusing `tuned`'s signatures (like `groupfix`, unlike
   cache-aware-tuned).** `singlefix` only changes `GROUP_SIZE_M`, which
   doesn't appear in the grid/block-dimension formula — so its launches are
   byte-identical in signature to the original `tuned` capture's, and
   `21`'s classification logic (`build_singlefix_mband.py`, a direct copy
   of `build_groupfix_mband.py`) applies unmodified.
5. **Merge into the chart's series format.** `build_series_from_durs.py`
   (from `22`) bins the reclassified per-launch durations into the same
   log-spaced bin edges the original chart already uses.

## New gotcha found in this experiment

- **A freshly-stopped `.nsys-rep` capture isn't necessarily done being
  written when `profiler-stop` returns.** The first `nsys stats --report
  gputrace` attempt against this capture failed with `Exportation error:
  Section Table Reference magic number mismatch` — not the usual "report
  not found" error `21`/`22` saw. Checking the file's size twice a few
  seconds apart showed it was still growing (153MB → 173MB). Waiting for
  the size to stabilize and retrying (after removing the corrupt partial
  `.sqlite`) resolved it cleanly. Worth a short size-stability check before
  running `nsys stats` on a capture that was just stopped, rather than
  extracting immediately.
- Also hit (again) DCGM/nsys's shared GPU-metrics counter lock — `22`'s
  cleanup resumed DCGM, so this deployment needed `dcgmi profile --pause`
  again before it would start, same as `22`'s own gotcha with `21`'s
  cleanup. This is now a standing pattern for every experiment in this
  chain: always re-pause DCGM before deploying a new nsys-profiled pod on a
  node where a previous experiment's cleanup resumed it.

## Files

- `N=*.json` (5 files) — the singlefix config itself: byte-identical to
  `tuned-configs/` except `gate_up_proj`'s M=2048 `GROUP_SIZE_M`.
- `configmap.yaml`, `servingruntime-singlefix-profiling.yaml`,
  `inferenceservice.yaml` — the profiling-enabled deployment.
- `bench-client-job.yaml` — the separate untraced benchmark-client Job.
- `extract_kernels.py` — SQL extraction script, run on-pod.
- `build_singlefix_mband.py` — classifies launches by reusing `tuned`'s
  grid/block signatures (same approach as `21`'s groupfix script).
- `build_series_from_durs.py` — bins per-launch durations into the chart's
  existing log-spaced bin edges.
- `singlefix_series.json` — the resulting `c64_singlefix` series.
- `chart-singlefix-vs-all.html` — `22`'s chart, extended with the seventh
  panel.

## Cleanup

All cluster resources from this experiment (the `qwen-27b-pr-evidence`
InferenceService/ServingRuntime, `m2048-singlefix-configs-profiling`
ConfigMap, `bench-client-throwaway` Job) were deleted after this capture,
and DCGM profiling was resumed on the node's GPU-engine pod
(`nvidia-dcgm-xt7j6`, node `gpu-node-a`).
