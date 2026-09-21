# 24 — CUDA-graph dispatch histogram (replayed vs. eager, by M)

**A direct, source-level instrumentation of vLLM's `CudagraphDispatcher.dispatch()`
— the exact per-forward-step decision point — recording every real
dispatch call's `num_tokens` and resulting mode (eager vs. graph-replayed),
both at startup and under real serving traffic.** Answers the natural
follow-up to the blog's "Real GEMM traffic, by batch size" section: those
per-concurrency GEMM-call counts came from a Python-level wrapper counter
that graph replay bypasses entirely, so they understate real GEMM traffic
by roughly an order of magnitude and can't distinguish eager from replayed
launches at all. This experiment measures both, exactly.

## Why this experiment

Earlier in this investigation (`05`'s existing nsys captures), a direct
SQL query against `CUPTI_ACTIVITY_KIND_KERNEL.graphId` showed 92.9% of
real GEMM kernel launches during serving are graph-replayed, only 7.1%
eager — but that number is aggregated across all M, with no per-M
breakdown of which sizes are replayed and which fall back to eager
execution. Getting that breakdown from nsys archaeology alone would mean
decoding `graphId`→shape/M mappings after the fact from grid dimensions —
possible (as done for the dominant-`graphId` analysis in `05`), but
indirect and limited to whatever graphs happened to get captured in that
one recording. Instrumenting the actual dispatch call is exact, cheap (no
nsys profiling needed at all), and free of any inference step: `dispatch()`
fires once per real forward step, for every step, and already returns the
`CUDAGraphMode` that determines eager vs. replay before the kernel ever
launches.

## Result

Two very different regimes, both exactly as vLLM's source code predicts:

**Startup** (one-time model-init sweep) reproduces the full 35-point
mixed-prefill-decode capture-size sweep (`1, 2, 4, ..., 256`, mostly
replayed once each size is captured, with early eager passes before each
size's graph exists) plus one isolated eager-only call at **M=2048** — the
`profile_run()` memory-sizing pass confirmed earlier via source read to run
strictly before `capture_model()`, in eager mode.

**Real traffic** (c=64, 500 requests) looks nothing like startup:

| | eager launches | replayed launches |
|---|---:|---:|
| total | 21,504 | 245,504 |
| dominant M | 2048 (8,448) | **64 (209,920, 84.5% of all replayed)** |

- Replayed launches are overwhelmingly **M=64 — the target concurrency
  itself** — swamping every other replayed size combined. This is decode
  traffic hitting its steady-state batch size, exactly as the earlier
  `graphId`-decoding analysis in `05` predicted from indirect evidence
  (one dominant `graphId` at 64% of replays, inferred to be M≈64).
- Eager launches are almost entirely large-M prefill chunks (M=312 up to
  M=2048), never small decode-shaped M — confirming eager execution during
  real serving is prefill overflow past the piecewise-graph ceiling, not
  decode.
- **M=2048 recurs 33 times during real traffic**, not just once at
  startup — the scheduler regularly re-hits its `max_num_batched_tokens=2048`
  ceiling under sustained prefill load, each time forced eager since no
  graph is captured at that size for prefill.

`chart-dispatch-histogram.html` plots both regimes as two log-scaled
bar-chart panels, color-coded eager vs. replayed.

## Method

1. **Instrument the exact decision point.** Copied vLLM's real
   `vllm/v1/cudagraph_dispatcher.py` to `cudagraph_dispatcher_instrumented.py`,
   diff-verified as pure additions only: the original `dispatch()` body was
   renamed unchanged to `_dispatch_uninstrumented`; the new public
   `dispatch()` is a thin wrapper that calls it, then records
   `(current_label, num_tokens, mode.name)` into a module-level `Counter`
   before returning the same `(mode, desc)` unchanged. A background thread
   dumps the counter to `/tmp/dispatch_histogram.json` every second and
   refreshes a `current_label` from `/tmp/current_label.txt`, so a driver
   script can tag which phase (`startup`/`unknown`, `c64`, ...) each
   dispatch call belongs to without touching the instrumented file again.
2. **Deploy graph-enabled, config-independent.** `servingruntime.yaml` —
   no `--enforce-eager` (graphs must stay on; this is not `02`'s
   diagnostic eager-only pass), plain `default` kernel config (replay-vs-eager
   dispatch is config-independent, so no tuned-configs mount needed), one
   volume mount overlaying the instrumented file at
   `vllm/v1/cudagraph_dispatcher.py`. No nsys profiling, no SCC/service-account
   needs — `configmap.yaml` + `inferenceservice.yaml` are otherwise identical
   to `21`'s minus the profiling label.
3. **Capture startup, then drive real traffic.** Read `/tmp/dispatch_histogram.json`
   right after the pod became ready (captures the full startup sweep under
   the `unknown` label untouched), then wrote `c64` to `/tmp/current_label.txt`
   on the pod and ran `bench-client-job.yaml` (copied from `22`'s, IP updated
   to this pod) at concurrency 64, 500 prompts — same benchmark parameters
   used throughout this project. Read the histogram file again after the
   run completed.
4. **Convert dispatch-step counts to launch counts.** Each dispatch call
   is one forward step, not one kernel launch — every real step launches
   the GEMM kernel once per layer per shape (`gate_up_proj`=64, `down_proj`=64,
   `out_proj`=64, `in_proj_qkvz`=48, `qkv_proj`=16 → 256 launches/step,
   the same constant that explains the "multiples of 256" pattern seen
   throughout `03`/`10`'s data). Multiplied every per-M dispatch-step count
   by 256 to get the exact per-M launch counts plotted in the chart.
5. **Tear down.** Deleted the InferenceService, ServingRuntime, ConfigMap,
   and throwaway bench-client Job immediately after retrieving the final
   histogram — no capture artifacts needed to be preserved beyond the JSON
   already pulled locally.

## Files

- `cudagraph_dispatcher_instrumented.py` — instrumented copy of vLLM's
  `vllm/v1/cudagraph_dispatcher.py`, pure additions only (see Method §1).
- `configmap.yaml` — embeds the instrumented file as the ConfigMap's
  `cudagraph_dispatcher.py` key.
- `servingruntime.yaml` — graph-enabled, default config, single overlay
  mount for the instrumented file.
- `inferenceservice.yaml` — same shape as `21`'s, profiling label/service
  account removed (not needed for this experiment).
- `bench-client-job.yaml` — `22`'s bench-client Job, IP updated to this
  experiment's predictor pod.
- `chart-dispatch-histogram.html` — two-panel log-scaled bar chart
  (startup / real c=64 traffic), eager vs. replayed launch counts by M.
