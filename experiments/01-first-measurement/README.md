# 01 — First measurement

Supports [blog §2 "Yes, but..."](../../blog/index.html).

## What

Two independent measurements of the same question — does vLLM's
autotuned W8A8 block-FP8 kernel config actually help? — at two different
levels:

1. **Kernel-level**: one isolated GEMM call, alone on an otherwise idle
   GPU, timed directly. Default config vs. tuned config, for all 5 real
   weight-matrix shapes in the model, at the 18 batch sizes vLLM's own
   tuner searches plus 17 more chosen in the gaps between them (to check
   the tuned config generalizes, not just interpolates).
2. **End-to-end**: the same two configs, now serving real traffic through
   the full vLLM stack (`vllm bench serve`), swept across
   `--max-concurrency` 1 through 128.

## Why

vLLM's own tuner picks a winning config internally and discards the
losing timings — it never reports a before/after number. Before trusting
a tuned config in production, you need your own direct measurement of how
much faster it actually is, at both levels: kernel time alone, and real
serving traffic (which the kernel number alone can't guarantee, since
serving involves batching, scheduling, and everything else around the
GEMM call).

## How

**Producing the tuned config** — `benchmark_w8a8_block_fp8_original.py`
is vLLM's real, stock tuner (`benchmarks/kernels/benchmark_w8a8_block_fp8.py`),
with exactly one change: `get_weight_shapes()` returns this model's 5 real
weight-matrix shapes. Run at full scope (5 shapes × 18 batch sizes), it
produces [`original-tuned-config/`](./original-tuned-config/) — the input
both measurements below use.

**Kernel-level** — `compare_default_vs_tuned.py` reuses vLLM's own
`w8a8_block_matmul`/`benchmark_config` functions verbatim (vendored into
`_vendored_matmul_timing.py`, since `benchmarks/kernels/` isn't part of
the installed `vllm` package and can't be imported directly), constructing
tensors the same way the tuner's own `tune()` does. Every result comes
with a max-output-diff sanity check between the two configs' actual
output tensors, confirming "faster" isn't cheating on numerics. Reads
tuned configs from [`original-tuned-config/`](./original-tuned-config/).

**End-to-end** — `vllm bench serve` against a real running server, once
per concurrency level, once for each config (default vs. the same
original tuned config, mounted over vLLM's installed config path — see
`e2e-server-pod.yaml` for exactly where). Same client traffic profile at
every point (random dataset, 256 input / 128 output tokens).

## Result

**Kernel-level**: a clean, monotonic win everywhere — **+50-70% at small
`M`, tapering to single digits by the low thousands**, never negative at
any of the 18 tuned anchors. Full curve (mean across all 5 shapes):
[`results/kernel-level-mean-speedup.json`](./results/kernel-level-mean-speedup.json).

**End-to-end**: a smooth, monotonic *decay* through zero as concurrency
rises — not a step change, not noise:

| concurrency | tok/s delta |
|---:|---:|
| 1 | +28.0% |
| 2 | +9.9% |
| 4 | +4.8% |
| 8 | +5.6% |
| 16 | +3.5% |
| 32 | +2.5% |
| 48 | −0.0% |
| 64 | −0.8% |
| 96 | −2.9% |
| 128 | −4.1% |

Full data: [`results/e2e-concurrency-decay.json`](./results/e2e-concurrency-decay.json).

**The puzzle this sets up**: a shrinking advantage as concurrency grows is
exactly what the kernel-level curve predicts (large-`M` speedups taper to
single digits, and the scheduler batches more tokens together as load
increases). A *negative* one isn't — nothing in the kernel-level
measurement ever showed the tuned config losing outright. That
contradiction is what the rest of this project's experiments (`02`
onward) chase down.

## Reproduce

**Producing the tuned config** (expect ~2 hours for the full 5-shape sweep):

```bash
kubectl apply -f original-autotune-job.yaml
kubectl logs -f job/qwen3-8-27b-fp8-original-autotune
kubectl cp <pod>:/repo/original-tuned-config ./original-tuned-config
kubectl delete -f original-autotune-job.yaml
```

**Kernel-level**, against a GPU node directly:

```bash
pip install vllm==0.27.1
python3 compare_default_vs_tuned.py
```

or as a self-contained Kubernetes Job (clones this repo, runs the script,
sleeps so you can read the logs):

```bash
kubectl apply -f kernel-level-compare-job.yaml
kubectl logs -f job/qwen3-8-27b-fp8-kernel-level-compare
kubectl delete -f kernel-level-compare-job.yaml
```

**End-to-end**, against a Kubernetes cluster with GPU nodes:

```bash
kubectl create configmap qwen3-8-27b-fp8-tuned-configs --from-file=./original-tuned-config/
kubectl apply -f e2e-server-pod.yaml
kubectl wait --for=condition=Ready pod/qwen3-8-27b-fp8-e2e-server --timeout=10m
kubectl cp sweep.sh qwen3-8-27b-fp8-e2e-server:/tmp/sweep.sh
kubectl exec -it qwen3-8-27b-fp8-e2e-server -- bash /tmp/sweep.sh   # default run
# uncomment the tuned-configs volumeMounts/volumes block in e2e-server-pod.yaml,
# kubectl apply -f e2e-server-pod.yaml again, then:
kubectl exec -it qwen3-8-27b-fp8-e2e-server -- bash /tmp/sweep.sh   # tuned run
kubectl delete -f e2e-server-pod.yaml
```

## Files

- `benchmark_w8a8_block_fp8_original.py` / `original-autotune-job.yaml` —
  vLLM's stock tuner (shape list patched) and the Kubernetes Job to run
  it, producing `original-tuned-config/`.
- `compare_default_vs_tuned.py` / `_vendored_matmul_timing.py` — the
  kernel-level comparison script and its vendored vLLM helper functions.
- `original-tuned-config/` — the tuned config this experiment measures.
- `kernel-level-compare-job.yaml` — self-contained Kubernetes Job for the
  kernel-level comparison.
- `e2e-server-pod.yaml` / `sweep.sh` — standalone vLLM server + the
  `vllm bench serve` sweep script for the end-to-end comparison.
- `results/` — the exact numbers behind both charts in blog §2.
