# 10 — W8A8 call-plan recorder

**Status:** built and merged into the deployed instrumentation, but never
actually exercised — no tuned-side (or default-side) call-plan JSONL was
ever captured with it. Documented as an available-but-unused tool, not a
completed experiment with a result.

## What

An opt-in, ordered, per-call recorder for every real `w8a8_triton_block_scaled_mm`
dispatch — logging `(sequence, run_id, host_time_ns, weight_id, M, N, K,
grid_x, config)` as JSONL — plus a comparison tool
(`compare_w8a8_call_plans.py`) that diffs two such recordings by their
logical `(M, N, K)` shape sequence: total calls per side, shapes unique to
one side, and how many calls occupy the *same sequence position* in both
plans.

## Why

The existing `_M_COUNTS` histogram instrumentation (used in
[`03-gemm-traffic-vs-speedup`](../03-gemm-traffic-vs-speedup/)) only records
aggregate counts per `(N, K, M)` — it can't answer "does the tuned config
issue GEMM calls in the same *order*, interleaved with the same other work,
as default?" A different agent's session began this work independently,
building the ordered recorder and the comparison tool without initially
being aware of the existing, already-deployed `_M_COUNTS` instrumentation in
the same file. Reconciling that handoff meant merging the new call-plan
recorder into the existing `fp8_utils.py` ConfigMap correctly, preserving
both mechanisms rather than one clobbering the other.

## How

Env-gated (`VLLM_W8A8_CALL_PLAN_PATH`, `VLLM_W8A8_CALL_PLAN_RUN_ID`, both
read once at import time) — a no-op with zero overhead when unset, so it's
safe to leave merged into the standing instrumentation permanently:

```python
_W8A8_CALL_PLAN_PATH = os.getenv("VLLM_W8A8_CALL_PLAN_PATH")
_W8A8_CALL_PLAN_RUN_ID = os.getenv("VLLM_W8A8_CALL_PLAN_RUN_ID", "")
...
def _record_w8a8_call(B, M, N, K, config):
    if _W8A8_CALL_PLAN_PATH is None:
        return
    with _W8A8_CALL_PLAN_LOCK:
        _W8A8_CALL_PLAN.append({...})   # sequence, run_id, host_time_ns, weight_id, M, N, K, grid_x, config
```
Inserted at the real call site right before `def grid(META):` in
`w8a8_triton_block_scaled_mm` — the same function
[`03-gemm-traffic-vs-speedup`](../03-gemm-traffic-vs-speedup/)'s `_M_COUNTS`
line lives in, confirmed compatible and merged rather than one replacing the
other. Flushed periodically and at process exit
(`atexit.register(_flush_w8a8_call_plan)`).

Full diff, both mechanisms combined, as actually applied on top of
`vllm/model_executor/layers/quantization/utils/fp8_utils.py`:
[`fp8_utils_instrumentation.patch`](./fp8_utils_instrumentation.patch). This
is diagnostic-only and was never intended to reach the real upstream PR
(which adds only JSON config files, no code changes) — kept here as
reference/reproduction material, not applied to the tracked `vllm/` clone
used for the actual PR branches.

To exercise it: set `VLLM_W8A8_CALL_PLAN_PATH`/`VLLM_W8A8_CALL_PLAN_RUN_ID`
as env vars on the serving container (added to `servingruntime.yaml` for the
eager-mode experiment below, alongside `--enforce-eager`), run traffic, pull
the resulting JSONL off the pod, repeat for the other config, then:

```bash
python3 compare_w8a8_call_plans.py default_plan.jsonl tuned_plan.jsonl --output diff.json
```

One real constraint discovered while wiring this up: **CUDA graph replay
never re-enters this Python function**, so this recorder (like `_M_COUNTS`)
is blind to graph-replayed launches — it only sees calls made in eager mode,
or the first, graph-*capturing* occurrence of each shape. `--enforce-eager`
was added to `servingruntime.yaml` specifically so every call would actually
reach this instrumentation, with an explicit comment that eager mode's own
timing numbers are not representative of normal (graph-enabled) serving
performance — this run would only ever be used for call *ordering*, never
for latency claims.

## What actually happened

The instrumentation was merged, deployed, and confirmed syntactically
correct (`fp8-utils-instrumented` ConfigMap updated via `oc create
configmap ... --dry-run=client -o yaml | oc apply -f -`, and kept in sync
with the local `vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py`
copy). `compare_w8a8_call_plans.py` was read in full and confirmed
compatible with the merged instrumentation's JSONL schema. **Neither side
was ever actually run with the env vars set** — no default-side or
tuned-side call-plan JSONL exists anywhere in this project. By the time the
instrumentation was ready, the investigation's center of gravity had moved
to [`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/) and
[`09-isolated-vs-real-flops-inflation`](../09-isolated-vs-real-flops-inflation/),
which answered the shape/`M`-localization question a different way (via
nsys hardware traces, not Python-level call ordering) and didn't end up
needing the call-order comparison this tool was built for.

This tool remains available, merged into the standing instrumentation, for
a follow-up question this project didn't end up asking: whether tuned and
default dispatch GEMM calls in a meaningfully different *order or
interleaving* relative to other kernel types — which the "cold cache"
hypothesis in
[`09-isolated-vs-real-flops-inflation`](../09-isolated-vs-real-flops-inflation/)
would actually predict something about, if pursued.

## Reproduce

```bash
# on the serving container:
export VLLM_W8A8_CALL_PLAN_PATH=/tmp/call_plan.jsonl
export VLLM_W8A8_CALL_PLAN_RUN_ID=default-run-1
# run traffic (ideally --enforce-eager, or accept graph-capture-only coverage), then:
python3 compare_w8a8_call_plans.py default_plan.jsonl tuned_plan.jsonl
```
