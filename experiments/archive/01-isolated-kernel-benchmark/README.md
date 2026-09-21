# 01 — Isolated kernel benchmark

**Status:** confirmed, still stands. This is the foundation everything else in the
project reacts to or qualifies — not superseded by later findings.

**Note on which tuned configs are "canonical":** this repo's committed
[`tuned-configs/`](../../../tuned-configs/) (used throughout every experiment in
this project) is tuned specifically against `vllm/vllm-openai:v0.27.1`, the
latest vLLM version, since that's what the upstream PR must target. It is
**not** the same as the config set actually deployed to the production
OpenShift AI serving stack, which runs an older vLLM version and was
autotuned separately against it — the two are non-identical (different
`BLOCK_SIZE_N`/`GROUP_SIZE_M`/`num_warps`/`num_stages` at some batch-size
anchors) by design, not a stale duplicate or tuning-run variance. If you're
looking at `enterprise-ai/performance-tests/` and find two differing config
directories, that's why.

## What

Measure the tuned W8A8 block-FP8 Triton kernel config against vLLM's hardcoded
default config, at the kernel level only: one GEMM call, alone on an otherwise
idle GPU, timed with `torch.accelerator.synchronize()` bracketing each call.
Done for all 5 real weight shapes in this model
(`gate_up_proj` N=17408, `in_proj_qkvz` N=8192, `qkv_proj` N=7168, `down_proj`
N=5120/K=8704, `out_proj` N=5120/K=3072), first at the 18 batch sizes the
tuner itself searched, then at 17 more batch sizes deliberately chosen in the
gaps between those, to check whether the tuned config generalizes to
untested `M` or is narrowly overfit to its own anchors.

## Why

vLLM's own tuner (`benchmarks/kernels/benchmark_w8a8_block_fp8.py`) picks a
winning config per shape/batch-size internally and discards the losing
timings — it never reports a before/after number. Before trusting the tuned
configs enough to commit them (and later, before trusting them in a live
serving comparison), we needed our own direct measurement of how much faster
the tuned config actually is, and whether that speedup is real at batch
sizes the tuner didn't explicitly search.

## How

`compare_default_vs_tuned.py` reuses vLLM's own `w8a8_block_matmul` /
`benchmark_config` functions verbatim (not a reimplementation), constructing
tensors the same way the tuner's own `tune()` does — random FP8-clamped
inputs, per-block scale factors — then timing `NUM_ITERS` repetitions of each
config back to back, taking the mean. Each result comes with a `max output
diff` sanity check between the two configs' actual output tensors, confirming
that "faster" isn't cheating on numerics.

Applied at the 18 batch sizes vLLM's tuner searches by default:
`1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072,
4096`. Iteration count was pushed up over successive runs (500 → 1000 → 2000)
to check that results were stable and not noise — they were, so 2000 became
the standard.

To test generalization (not just interpolation, since production snaps any
real `M` to its *nearest* tuned anchor — see the lookup diagram in the repo's
`docs/index.html`), 17 more batch sizes were measured in between the anchors
(e.g. `M=3` between anchors 2 and 4, `M=160` between anchors 128 and 256),
each timed at its own *true*, off-anchor `M` but compared against whichever
config production would actually select for it (its nearest anchor's
config) — i.e. these numbers show what a real request at that exact size
would get, not what the tuner would get if it had searched that size too.

## Results

Kernel-level speedup (tuned vs. default), all 5 shapes, all 18 anchors —
consistent pattern: **+50-70% at small M, tapering to single digits by the
low thousands**. Full table in the top-level `README.md`'s "Kernel-level"
section; interactive version in [`chart.html`](./chart.html).

Held-out (generalization) results, 2000-iteration run, full detail in
[`held-out-results-2000iter-isolated.log`](./held-out-results-2000iter-isolated.log)
(also [`-500iter-`](./held-out-results-500iter-isolated.log) and
[`-1000iter-`](./held-out-results-1000iter-isolated.log) variants confirming
stability):

| | |
|---|---|
| tuned anchors ever negative | 0/90 |
| held-out points negative (stable across iteration counts) | 6/85 |
| held-out speedup vs. simple neighbor-interpolation | ~3 percentage points lower, on average |
| repeat offender | `M=160` (snaps to anchor 128) — negative in 3/5 shapes |

Real, but small: a genuine generalization gap exists, but it's easily lost
in the ~9-point run-to-run noise of any single measurement — which is why it
never showed up as an obvious outlier until specifically tested for.

**This generalization gap does not explain the later-discovered c=64
production regression.** It's ~3pp and confined to a handful of `M` values;
the production regression is 10s of percentage points and specific to one
shape (`gate_up_proj`, N=17408) across a broad, contiguous `M` range
(`M≳640`) that isn't concentrated on any interpolation gap — see
[`03-gemm-traffic-vs-speedup`](../03-gemm-traffic-vs-speedup/) and the later
per-layer work in [`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/)
for why. This experiment's real, useful role was ruling that hypothesis
*out*, not confirming it.

## Reproduce

```bash
pip install vllm==0.27.1
python3 compare_default_vs_tuned.py          # the 18 anchors
# held-out points: same script pattern, additional M values inserted,
# each looked up against its nearest anchor's tuned config
```

Kubernetes Job version: `k8s/compare-job.yaml` in the repo root.

Raw combined output of both the tuner itself
(`benchmark_w8a8_block_fp8.py`, all 5 shapes) and the default-vs-tuned
comparison run against `vllm/vllm-openai:v0.27.1`:
[`tuning-and-comparison-v0.27.1.log`](./tuning-and-comparison-v0.27.1.log).
Produced by the self-contained Kubernetes Job
[`tune-and-compare-job.yaml`](./tune-and-compare-job.yaml) — patches a
throwaway copy of the tuner's `get_weight_shapes()`, runs the tuner, then
runs the same default-vs-tuned comparison inline in one pod (no git clone
needed; distinct from `k8s/compare-job.yaml` in the repo root, which expects
the fork branch already cloned).
