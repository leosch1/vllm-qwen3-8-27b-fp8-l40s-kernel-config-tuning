# Qwen3.8-27B-FP8 dense W8A8 block-FP8 kernel config tuning (NVIDIA L40S)

> Please check out https://leosch1.github.io/vllm-qwen3-8-27b-fp8-l40s-kernel-config-tuning/blog for the full writeup.

Tuning vLLM's dense W8A8 block-FP8 Triton GEMM kernel for [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (`tensor_parallel_size=2`) on NVIDIA L40S, and measuring whether it actually helps — including in real, concurrent serving, not just in isolation.

Backs [vllm-project/vllm#58005](https://github.com/vllm-project/vllm/pull/58005), contributing the resulting configs upstream.

Environment: `vllm/vllm-openai:v0.27.1`, 2x NVIDIA L40S.

## Background

### Why vLLM uses a Triton kernel for this model + GPU

`Qwen/Qwen3.8-27B-FP8` ships FP8-quantized weights with block-wise scaling.
To run this model on NVIDIA L40S GPUs, vLLM chooses to use a Triton kernel for GEMM.
See `w8a8_triton_block_scaled_mm` in `vllm/model_executor/layers/quantization/utils/fp8_utils.py`.

### The default config, and the warning it causes

Triton kernels take launch parameters. They have no single value for the best performance. The optimum depends on the matrix shape, batch size, and GPU.
vLLM ships one hardcoded fallback, used whenever nothing better is known:

```python
{
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}
```

Every time that fallback is used, vLLM logs a warning at startup:

```
WARNING [fp8_utils.py:851] Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at .../vllm/model_executor/layers/quantization/utils/configs/N=17408,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json
```

### From warning to tuned config

If a JSON file matching that exact path exists, vLLM uses it instead, picking whichever batch-size entry is closest to the actual request size.
vLLM already ships many such files for popular GPU/shape combinations, but has currently none for `Qwen/Qwen3.8-27B-FP8` on L40S. This project fills that gap.

Using vLLM's own tuning script, a grid search over the Triton launch parameters (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`) was run per GEMM shape and batch size, keeping the fastest combination for each. A first pass at this looked like a clean win in isolation but turned out to regress real, concurrent serving at higher load — traced to the tuner's own benchmarking loop implicitly assuming a warm L2 cache that real serving never has. **The methodology below is the corrected version**, which fixes that assumption in the tuner itself rather than patching around it. The regression, its root cause, and the fix are the subject of [the blog post](https://leosch1.github.io/vllm-qwen3-8-27b-fp8-l40s-kernel-config-tuning/blog/); this README just documents the end state.

## Config parameters

**Inside each JSON file** -- Triton kernel tile/scheduling parameters, one
set per batch size:
- `BLOCK_SIZE_M` / `BLOCK_SIZE_N` / `BLOCK_SIZE_K`: the GEMM's three dimensions (`[M,K] x [K,N] -> [M,N]`), tiled into chunks of this size for GPU execution.
- `GROUP_SIZE_M`: how many M-tiles get scheduled together — the parameter at the center of the regression/fix described in the blog.
- `num_warps`: concurrent warps (32-thread groups) per tile, controlling parallelism/occupancy.
- `num_stages`: software-pipelining depth - how many loop iterations ahead memory loads are prefetched.

**In each filename**, e.g.
`N=17408,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json`:
- `N`, `K`: the weight matrix's dimensions (`[N,K]`) this config applies to.
- `device_name`: which GPU model this was tuned for.
- `dtype`: the quantization format.
- `block_shape`: not the same as `BLOCK_SIZE_M/N/K` but the quantization block size (how many weight elements share one FP8 scale factor)

## Methodology

### 1. Tuning

Uses vLLM's own [benchmarks/kernels/benchmark_w8a8_block_fp8.py](https://github.com/vllm-project/vllm/blob/v0.27.1/benchmarks/kernels/benchmark_w8a8_block_fp8.py) with two patches, committed as two real commits on [leosch1/vllm@qwen3-8-27b-fp8-dense-tuning](https://github.com/leosch1/vllm/tree/qwen3-8-27b-fp8-dense-tuning) (branched from `v0.27.1`) so they're reviewable as plain diffs against the stock script:

**Patch 1 — [the model's real shapes](https://github.com/leosch1/vllm/commit/e0713dc7078efb7d67d9aa224ab3ccdd961c577d).** `get_weight_shapes()` only returns DeepSeek-V3's hardcoded shapes:

```diff
 def get_weight_shapes(tp_size):
+    return [
+        (17408, 5120),
+        (8192, 5120),
+        (7168, 5120),
+        (5120, 8704),
+        (5120, 3072),
+    ]
+
     # NOTE(HandH1998): The weight shapes only works for DeepSeek-V3.
     # Modify them, if you tune for another different model.
```

The needed shapes were captured from the live predictor logs' "Using default W8A8 Block FP8 kernel config" startup warnings, cross-checked against the model's `config.json`.

**Patch 2 — [flush L2 before every timed launch](https://github.com/leosch1/vllm/commit/53b0156cbce6d2d537457fd98b408cc22ebc6808).** The stock tuner reuses the same input tensors across all 1,280 candidate configs per shape, so by the time it's timing the 50th candidate the weight matrix has been sitting in L2 for a while and never leaves — a cache state real serving never has. The fix makes the tuner's own benchmarking loop measure under the condition it's actually optimizing for, threaded explicitly through `tune_on_gpu() → tune() → benchmark_config()` and toggleable via `--l2-flush`/`--no-l2-flush` (default: on):

```python
def benchmark_config(..., flush_l2=True):
    ...
    for i in range(num_iters):
        if flush_l2:
            get_flush_buf().zero_()  # evict A/B/C from L2, so this launch starts cold
        torch.accelerator.synchronize()
        start_event.record()
        run()
```

```bash
git clone --branch=qwen3-8-27b-fp8-dense-tuning https://github.com/leosch1/vllm.git
cd vllm/benchmarks/kernels
python3 benchmark_w8a8_block_fp8.py \
    --tp-size 2 --input-type fp8 --out-dtype bfloat16 \
    --block-n 128 --block-k 128 --save-path ./tuned-configs
```

The resulting 5 kernel config files are committed in [`tuned-configs/`](./tuned-configs). Kubernetes Job version: [`k8s/tune-job.yaml`](./k8s/tune-job.yaml) — self-contained, clones `leosch1/vllm@qwen3-8-27b-fp8-dense-tuning` and runs the tuner directly.

### 2. Kernel-level before/after comparison

vLLM has no tool that reports default-vs-tuned latency — the tuner picks a winner internally and discards the timing.

[`compare_default_vs_l2flush_l2flushed.py`](./compare_default_vs_l2flush_l2flushed.py) fills that gap, and — using the same reasoning as the tuning fix above — flushes L2 before every timed launch of *both* configs, so the comparison is measured under the same condition the tuner actually optimized for:

```bash
pip install vllm==0.27.1
python3 compare_default_vs_l2flush_l2flushed.py
```

Kubernetes Job version: [`k8s/compare-job.yaml`](./k8s/compare-job.yaml).

### 3. End-to-end serving before/after comparison

`vllm bench serve` runs against the model, both with and without the tuned configs, swept across `--max-concurrency` 1 through 128 (see [`experiments/06-l2-flush-e2e-validation/`](./experiments/06-l2-flush-e2e-validation/) for the exact server manifest and sweep script):

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --port=8080 --model=/mnt/models --served-model-name=qwen-27b \
    --tensor-parallel-size=2 --max-model-len=8192 --reasoning-parser=qwen3 \
    --enable-auto-tool-choice --tool-call-parser=qwen3_coder \
    --max-num-seqs=128
# env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm bench serve \
    --backend openai-chat --endpoint /chat/completions \
    --model qwen-27b --dataset-name random \
    --num-prompts <N> --random-input-len 256 --random-output-len 128 \
    --request-rate inf --max-concurrency <C> --temperature 0 \
    --save-result --save-detailed
```

## Results

### Kernel-level

Both configs measured under a direct L2 flush before every launch — the same condition the tuner optimized for. Full 175-point data (5 shapes × 35 batch sizes, including 17 held-out sizes the tuner never explicitly searched): [`experiments/05-l2-flush-kernel-validation/results/kernel-level-speedup.json`](./experiments/05-l2-flush-kernel-validation/results/kernel-level-speedup.json).

| shape (N, K) | mean speedup | worst point | negative points |
|---|---:|---:|---:|
| gate_up_proj (17408, 5120) | +8.8% | +3.8% | 0/35 |
| in_proj_qkvz (8192, 5120) | +13.0% | +2.7% | 0/35 |
| qkv_proj (7168, 5120) | +15.9% | +7.1% | 0/35 |
| down_proj (5120, 8704) | +19.0% | −4.2% | 1/35 (M=768) |
| out_proj (5120, 3072) | +20.8% | +8.5% | 0/35 |

175 points total, 1 negative — `gate_up_proj` at large `M` (the shape that produced the original regression under a warm-cache tuner) is now solidly positive at every anchor.

### End-to-end

Output tok/s delta vs. default, real serving, `vllm bench serve` swept across `--max-concurrency`. `tuned_original` is the first, warm-cache-tuned config (the one that motivated this fix); `l2_flush` is the current, committed one:

| concurrency | tuned_original | l2_flush |
|---:|---:|---:|
| 1 | +28.0% | **+30.5%** |
| 2 | +10.9% | **+13.7%** |
| 4 | +4.8% | **+7.0%** |
| 8 | +5.6% | **+10.2%** |
| 16 | +3.8% | **+7.5%** |
| 32 | +2.7% | **+8.1%** |
| 48 | +0.1% | **+6.7%** |
| 64 | −0.9% | **+5.9%** |
| 96 | −3.1% | **+4.1%** |
| 128 | −4.2% | **+3.8%** |

The first tuning pass decayed through zero and went negative above `c=48` — the regression this whole investigation is about. The current config never regresses, at any concurrency tested. Full metrics (tok/s, TTFT, TPOT at every point): [`experiments/06-l2-flush-e2e-validation/results/full-metrics-table.json`](./experiments/06-l2-flush-e2e-validation/results/full-metrics-table.json).

## More detail

- **[The blog post](https://leosch1.github.io/vllm-qwen3-8-27b-fp8-l40s-kernel-config-tuning/blog/)** — the full investigation: how the regression was found, ruled-out hypotheses, the root cause, and the fix. Written for a reader who wants the reasoning, not just the result.
- **[`experiments/`](./experiments/)** — a curated, numbered walkthrough (01–06) matching the blog's structure, each step self-contained and reproducible.
- **[`experiments/archive/`](./experiments/archive/)** — every experiment actually run during this project, including dead ends and corrected claims, with an [overview page](https://leosch1.github.io/vllm-qwen3-8-27b-fp8-l40s-kernel-config-tuning/experiments/archive/overview.html) narrating all of it.
