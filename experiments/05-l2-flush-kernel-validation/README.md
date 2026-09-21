# 05 — L2-flush kernel-level validation

Supports [blog §6, "kernel-level speedup, new tuned config compared to
default" chart](../../blog/index.html).

## What

Benchmark the retuned config from [`04-l2-flush-retune`](../04-l2-flush-retune/)
(now committed at [`tuned-configs/`](../../tuned-configs/)) against the
default config, at the kernel level, across all 5 shapes and 35 batch
sizes — the same grid [`01-first-measurement`](../01-first-measurement/)
used for the original config. Both configs measured under a direct L2
flush before every launch: the same mechanism that actually produced the
winning config, so this asks the most literal possible question — under
the exact condition the tuner optimized for, does the config it picked
actually win?

## Why

`04` showed the *old* config regresses once measurement stops keeping the
cache warm. This experiment checks the *new* config doesn't have the same
problem — does fixing the tuning script's assumption actually produce a
config that's reliably better, not just differently biased?

## How

`compare_default_vs_l2flush_l2flushed.py` is
`01-first-measurement/compare_default_vs_tuned.py` with one change: every
timed launch is preceded by zeroing a buffer sized to the GPU's full L2
capacity (same flush mechanism as `04`), so neither config gets to
benefit from a warm cache the way `01`'s original always-warm benchmark
implicitly did.

## Result

**Mean speedup +15.5% across all 175 points. Only 1 point negative**
(`down_proj`, M=768, −4.2%) — and critically, `gate_up_proj`'s regression
at large `M` is gone; its curve is solidly positive at every anchor.
Full data: [`results/kernel-level-speedup.json`](./results/kernel-level-speedup.json).

## Reproduce

```bash
kubectl apply -f compare-job.yaml
kubectl logs -f job/qwen3-8-27b-fp8-l2flush-kernel-validate
kubectl delete -f compare-job.yaml
```

## Files

- `compare_default_vs_l2flush_l2flushed.py` — the comparison script.
- `compare-job.yaml` — Kubernetes Job to run it.
- `results/kernel-level-speedup.json` — the 175-point result behind blog
  §6's stability chart.
