# 04 — L2-flush retune

Supports [blog §6 "Patch the assumption, not just the config"](../../blog/index.html)
(the fix mechanism, and its first chart).

## What

Two things:

1. **Re-measure the original, unpatched tuned config's kernel-level
   speedup under a direct L2 flush** before every timed launch, instead
   of [`01-first-measurement`](../01-first-measurement/)'s always-warm
   regime — does the regression [`02`](../02-nsys-shape-profiling/) and
   [`03`](../03-group-size-m-patch/) found in real serving show up in an
   isolated benchmark too, once that benchmark stops keeping the cache
   artificially warm?
2. **Patch the autotuning script itself** with the same flush mechanism,
   and rerun it at full production scope (5 shapes × 18 batch sizes) to
   produce a new tuned config that doesn't carry the warm-cache
   assumption in the first place. This is what's committed at
   [`tuned-configs/`](../../tuned-configs/) today.

## Why

`03` fixed one value by hand and confirmed `GROUP_SIZE_M` was the
culprit. But the *reason* the tuner picked a bad value there is that its
own benchmarking loop reuses the same `A`/`B` tensors across all 1,280
candidate configs — by the time it's timing the 50th candidate, the
first 49 have already pulled the weight matrix into L2 and it never
leaves. The tuner is benchmarking a cache state real serving never has.
The fix isn't another hand patch — it's making the tuning script itself
measure under the condition it's actually optimizing for.

## How

The fix is the smallest change that makes the tuning script's own
benchmarking loop cache-realistic: flush the GPU's entire L2 cache right
before every timed launch.

```python
l2 = torch.cuda.get_device_properties(device).L2_cache_size
flush_buf = torch.empty(int(l2 * 1.5) // 4, dtype=torch.int32, device="cuda")

def benchmark_config(...):           # unchanged, apart from one line
    ...
    for _ in range(num_iters):
        flush_buf.zero_()            # ← the whole fix: evict L2, so this launch starts cold
        start_event.record()
        run()
        end_event.record()
```

`benchmark_w8a8_block_fp8_l2flush.py` applies exactly this patch to
vLLM's real, unmodified tuner (`benchmarks/kernels/benchmark_w8a8_block_fp8.py`
from `vllm/vllm-openai:v0.27.1`) — everything outside the
`L2-FLUSH PATCH`/`END L2-FLUSH PATCH` markers is the stock script, untouched.
`resolve_l2_bytes()` reads the GPU's actual reported L2 capacity, with a
documented, generous 128MiB fallback if that isn't exposed by the current
PyTorch build.

`compare_default_vs_tuned_l2flushed.py` applies the *same* flush
mechanism to `01-first-measurement`'s kernel-level comparison script, but
points it at `./original-tuned-config/` — the config the unpatched tuner
produced.

## Result

**Re-measured under flush, the original tuned config's `gate_up_proj`
regression is unambiguous** — previously mildly positive or near-zero
under a warm cache, it swings sharply negative exactly where the tuner
picked `GROUP_SIZE_M=1`:

| M | speedup, always-warm (`01`) | speedup, under L2-flush (this experiment) |
|---:|---:|---:|
| 512 | positive | **−47.5%** |
| 1024 | positive | **−23.7%** |
| 1536 | positive | **−11.4%** |
| 2048 | positive | **−6.0%** |
| 3072 | positive | +5.7% |
| 4096 | positive | +5.4% |

No other shape shows this pattern — every other shape stays solidly
positive at every large-`M` anchor, flushed or not. Full 175-point data:
[`results/original-tuned-under-flush.json`](./results/original-tuned-under-flush.json).

**The retuned config** (produced by `retune-job.yaml`, now committed at
[`tuned-configs/`](../../tuned-configs/)) no longer carries this
assumption — its kernel-level and end-to-end validation are in
[`05-l2-flush-kernel-validation`](../05-l2-flush-kernel-validation/) and
[`06-l2-flush-e2e-validation`](../06-l2-flush-e2e-validation/).

## Reproduce

**Re-measure the original config under flush:**

```bash
kubectl apply -f tuned-flushed-compare-job.yaml
kubectl logs -f job/qwen3-8-27b-fp8-tuned-flushed-compare
kubectl delete -f tuned-flushed-compare-job.yaml
```

**Rerun the fixed tuner at full scope** (produces a fresh
`tuned-configs/`-equivalent; expect ~2+ hours):

```bash
kubectl apply -f retune-job.yaml
kubectl logs -f job/qwen3-8-27b-fp8-l2-flush-retune
kubectl cp <pod>:/repo/l2-flush-tuned-configs ./l2-flush-tuned-configs
kubectl delete -f retune-job.yaml
```

## Files

- `benchmark_w8a8_block_fp8_l2flush.py` — vLLM's real tuner, patched with
  the L2-flush fix (patch clearly marked `L2-FLUSH PATCH`).
- `compare_default_vs_tuned_l2flushed.py` — `01`'s kernel-level
  comparison script, adapted to flush L2 before every launch.
- `original-tuned-config/` — the unpatched tuner's config output.
- `retune-job.yaml` / `tuned-flushed-compare-job.yaml` — Kubernetes Jobs
  for each script above.
- `results/original-tuned-under-flush.json` — the 175-point result behind
  blog §6's first chart.
