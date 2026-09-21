# 06 — L2-flush end-to-end validation

Supports [blog §6's final chart and metrics table](../../blog/index.html)
— the "does it hold up in real, end-to-end serving?" question.

## What

The same end-to-end concurrency sweep as
[`01-first-measurement`](../01-first-measurement/), now for the retuned
config from [`04-l2-flush-retune`](../04-l2-flush-retune/) (committed at
[`tuned-configs/`](../../tuned-configs/)) instead of the original tuned
config. Real serving traffic, `vllm bench serve`, swept across
`--max-concurrency` 1 through 128.

## Why

`05` confirmed the fix at the kernel level. Kernel-level speed isn't
guaranteed to survive real serving — that's exactly the gap
`01-first-measurement` originally found for the *unpatched* config
(faster in isolation, worse at high concurrency in production). This is
the same check, for the fix.

## How

Identical methodology and identical scripts to `01-first-measurement`'s
end-to-end half — `e2e-server-pod.yaml` deploys a standalone vLLM server,
toggled between vLLM's default fallback config and this repo's
`tuned-configs/` (the l2-flush-retuned one) via the same
per-shape-JSON-file volume mount; `sweep.sh` runs the same 10-point
`vllm bench serve` sweep against it.

## Result

**Never worse than default, at any concurrency** — the previous tuned
config's e2e win decayed through zero and went negative above c=48; this
one stays positive at every single point, 1 through 128:

| concurrency | previous tuned config | l2-flush config |
|---:|---:|---:|
| 1 | +28.0% | **+30.5%** |
| 4 | +4.8% | **+7.0%** |
| 16 | +3.8% | **+7.5%** |
| 48 | +0.1% | **+6.7%** |
| 64 | −0.9% | **+5.9%** |
| 96 | −3.1% | **+4.1%** |
| 128 | −4.2% | **+3.8%** |

Full data: [`results/e2e-comparison.json`](./results/e2e-comparison.json).
Full tok/s + TTFT + TPOT table at every concurrency:
[`results/full-metrics-table.json`](./results/full-metrics-table.json) —
higher tok/s and lower TTFT/TPOT are both better; the l2-flush config
wins on tok/s and TPOT at every single point. TTFT is slightly worse at
low concurrency (c≤8, largest gap at c=4) but clearly better from c=16
on — see the full table for exact numbers.

This closes the loop the whole project started from: a config that's
faster in isolation and faster at every real concurrency level, with no
hidden regression at scale.

## Reproduce

```bash
kubectl create configmap qwen3-8-27b-fp8-l2flush-tuned-configs --from-file=../../tuned-configs/
kubectl apply -f e2e-server-pod.yaml
kubectl wait --for=condition=Ready pod/qwen3-8-27b-fp8-l2flush-e2e-server --timeout=10m
kubectl cp sweep.sh qwen3-8-27b-fp8-l2flush-e2e-server:/tmp/sweep.sh
kubectl exec -it qwen3-8-27b-fp8-l2flush-e2e-server -- bash /tmp/sweep.sh   # default run
# uncomment the tuned-configs volumeMounts/volumes block in e2e-server-pod.yaml,
# kubectl apply -f e2e-server-pod.yaml again, then:
kubectl exec -it qwen3-8-27b-fp8-l2flush-e2e-server -- bash /tmp/sweep.sh   # l2-flush run
kubectl delete -f e2e-server-pod.yaml
```

## Files

- `e2e-server-pod.yaml` / `sweep.sh` — standalone vLLM server + the
  `vllm bench serve` sweep script.
- `results/e2e-comparison.json` — tok/s delta vs. default, both the
  previous tuned config (reference, from an independent sweep — see
  `01-first-measurement` for how to reproduce that half) and the
  l2-flush config.
- `results/full-metrics-table.json` — tok/s, TTFT, TPOT at every
  concurrency point, default vs. l2-flush.
