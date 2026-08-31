# Qwen3.8-27B-FP8 dense W8A8 block-FP8 kernel config tuning (NVIDIA L40S)

Tuning vLLM's dense W8A8 block-FP8 Triton GEMM kernel for [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (`tensor_parallel_size=2`) on NVIDIA L40S, and measuring whether it actually helps.

Also backs a PR to [vllm-project/vllm](https://github.com/vllm-project/vllm) contributing the resulting configs upstream: _TODO: link once opened_.

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
Using vLLM's own tuning script I did a grid-search on the different parameters (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages) per GEMM shape and batch size and kept the fastest combination for each.
Then I measured whether the custom config actually helps performance, both at the kernel level and end-to-end.

## Config parameters

**Inside each JSON file** -- Triton kernel tile/scheduling parameters, one
set per batch size:
- `BLOCK_SIZE_M` / `BLOCK_SIZE_N` / `BLOCK_SIZE_K`: the GEMM's three dimensions (`[M,K] x [K,N] -> [M,N]`), tiled into chunks of this size for GPU execution.
- `GROUP_SIZE_M`: how many M-tiles get scheduled together.
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

Used vLLM's own [benchmarks/kernels/benchmark_w8a8_block_fp8.py](https://github.com/vllm-project/vllm/blob/v0.27.1/benchmarks/kernels/benchmark_w8a8_block_fp8.py) with one small patch to work around the fact that `get_weight_shapes()` only returns DeepSeek-V3's hardcoded shapes:

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

The needed shapes were captured from the live predictor logs "Using default W8A8 Block FP8 kernel config" startup warnings, cross-checked against the model's `config.json`.

```bash
python3 benchmark_w8a8_block_fp8.py \
    --tp-size 2 --input-type fp8 --out-dtype bfloat16 \
    --block-n 128 --block-k 128 --save-path ./tuned-configs
```

The resulting 5 kernel config files are committed in [`tuned-configs/`](./tuned-configs).

The patch itself lives as a real commit on [leosch1/vllm@qwen3-8-27b-fp8-dense-tuning](https://github.com/leosch1/vllm/tree/qwen3-8-27b-fp8-dense-tuning) (branched from `v0.27.1`). Kubernetes Job version of the same steps: [`k8s/tune-job.yaml`](./k8s/tune-job.yaml), which clones that branch and runs the tuner.

### 2. Kernel-level before/after comparison

vLLM has no tool that reports default-vs-tuned latency, the tuner picks a winner internally and discards the timing.

The script in this repository `compare_default_vs_tuned.py` is filling that gap.

```bash
pip install vllm==0.27.1
python3 compare_default_vs_tuned.py
```

Kubernetes Job version: [`k8s/compare-job.yaml`](./k8s/compare-job.yaml).

### 3. End-to-end serving before/after comparison

`vllm bench serve` runs against the model, both with and without the tuned configs, at two concurrency profiles:
- low (`--request-rate 1 --max-concurrency 1`)
- high (`--request-rate inf --max-concurrency 64 --num-prompts 500`)

## Results

### Kernel-level

### N=17408, K=5120, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup | max output diff |
|---:|---:|---:|---:|---:|
| 1 | 19.49 | 5.55 | +71.5% | 0.0000 |
| 2 | 12.73 | 6.24 | +51.0% | 0.0000 |
| 4 | 10.30 | 5.36 | +48.0% | 0.0000 |
| 8 | 8.48 | 5.20 | +38.8% | 0.0000 |
| 16 | 7.98 | 5.47 | +31.4% | 0.0000 |
| 24 | 7.94 | 6.89 | +13.3% | 0.0000 |
| 32 | 8.67 | 6.33 | +27.0% | 0.0000 |
| 48 | 8.64 | 6.97 | +19.3% | 0.0000 |
| 64 | 7.64 | 6.49 | +15.1% | 0.0000 |
| 96 | 10.28 | 9.11 | +11.3% | 0.0000 |
| 128 | 10.78 | 9.21 | +14.5% | 0.0000 |
| 256 | 17.67 | 15.81 | +10.5% | 0.0000 |
| 512 | 34.71 | 31.96 | +7.9% | 0.0000 |
| 1024 | 66.65 | 60.37 | +9.4% | 0.0000 |
| 1536 | 93.44 | 81.80 | +12.5% | 0.0000 |
| 2048 | 125.55 | 115.59 | +7.9% | 0.0000 |
| 3072 | 189.90 | 176.64 | +7.0% | 0.0000 |
| 4096 | 264.12 | 252.02 | +4.6% | 0.0000 |

### N=8192, K=5120, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup | max output diff |
|---:|---:|---:|---:|---:|
| 1 | 13.47 | 4.45 | +66.9% | 0.0000 |
| 2 | 8.82 | 4.29 | +51.3% | 0.0000 |
| 4 | 7.95 | 4.84 | +39.1% | 0.0000 |
| 8 | 7.60 | 4.67 | +38.6% | 0.0000 |
| 16 | 7.47 | 4.57 | +38.8% | 0.0000 |
| 24 | 7.40 | 5.14 | +30.4% | 0.0000 |
| 32 | 7.86 | 4.98 | +36.7% | 0.0000 |
| 48 | 8.19 | 6.01 | +26.6% | 0.0000 |
| 64 | 7.53 | 5.70 | +24.3% | 0.0000 |
| 96 | 7.97 | 6.92 | +13.2% | 0.0000 |
| 128 | 7.73 | 6.97 | +9.8% | 0.0000 |
| 256 | 10.01 | 9.27 | +7.4% | 0.0000 |
| 512 | 17.39 | 15.24 | +12.3% | 0.0000 |
| 1024 | 29.58 | 27.59 | +6.7% | 0.0000 |
| 1536 | 42.33 | 39.27 | +7.2% | 0.0000 |
| 2048 | 55.57 | 50.94 | +8.3% | 0.0000 |
| 3072 | 83.97 | 78.24 | +6.8% | 0.0000 |
| 4096 | 113.82 | 103.65 | +8.9% | 0.0000 |

### N=7168, K=5120, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup | max output diff |
|---:|---:|---:|---:|---:|
| 1 | 11.79 | 5.93 | +49.7% | 0.0000 |
| 2 | 9.98 | 5.43 | +45.6% | 0.0000 |
| 4 | 8.11 | 4.31 | +46.9% | 0.0000 |
| 8 | 8.06 | 4.49 | +44.3% | 0.0000 |
| 16 | 7.43 | 4.37 | +41.2% | 0.0000 |
| 24 | 7.36 | 5.18 | +29.7% | 0.0000 |
| 32 | 7.34 | 5.11 | +30.3% | 0.0000 |
| 48 | 7.35 | 6.67 | +9.3% | 0.0000 |
| 64 | 7.90 | 6.24 | +21.1% | 0.0000 |
| 96 | 8.46 | 6.48 | +23.4% | 0.0000 |
| 128 | 8.37 | 6.64 | +20.6% | 0.0000 |
| 256 | 9.56 | 9.09 | +4.9% | 0.0000 |
| 512 | 16.11 | 15.32 | +4.9% | 0.0000 |
| 1024 | 29.37 | 27.04 | +7.9% | 0.0000 |
| 1536 | 37.22 | 34.45 | +7.4% | 0.0000 |
| 2048 | 49.27 | 45.05 | +8.6% | 0.0000 |
| 3072 | 71.82 | 66.88 | +6.9% | 0.0000 |
| 4096 | 98.15 | 89.84 | +8.5% | 0.0000 |

### N=5120, K=8704, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup | max output diff |
|---:|---:|---:|---:|---:|
| 1 | 14.44 | 5.28 | +63.5% | 0.0000 |
| 2 | 11.67 | 4.85 | +58.4% | 0.0000 |
| 4 | 10.42 | 4.93 | +52.7% | 0.0000 |
| 8 | 10.35 | 4.88 | +52.9% | 0.0000 |
| 16 | 10.47 | 5.13 | +51.0% | 0.0000 |
| 24 | 10.13 | 5.55 | +45.2% | 0.0000 |
| 32 | 11.87 | 5.42 | +54.3% | 0.0000 |
| 48 | 10.15 | 6.33 | +37.6% | 0.0000 |
| 64 | 10.08 | 6.15 | +39.0% | 0.0000 |
| 96 | 10.27 | 6.91 | +32.6% | 0.0000 |
| 128 | 10.33 | 8.27 | +19.9% | 0.0000 |
| 256 | 12.48 | 12.48 | +0.0% | 0.0000 |
| 512 | 23.28 | 21.02 | +9.7% | 0.0000 |
| 1024 | 34.66 | 32.89 | +5.1% | 0.0000 |
| 1536 | 46.80 | 42.94 | +8.3% | 0.0000 |
| 2048 | 59.51 | 53.75 | +9.7% | 0.0000 |
| 3072 | 87.40 | 80.46 | +7.9% | 0.0000 |
| 4096 | 121.28 | 111.65 | +7.9% | 0.0000 |

### N=5120, K=3072, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup | max output diff |
|---:|---:|---:|---:|---:|
| 1 | 7.89 | 3.95 | +50.0% | 0.0000 |
| 2 | 6.44 | 3.77 | +41.4% | 0.0000 |
| 4 | 5.86 | 3.90 | +33.5% | 0.0000 |
| 8 | 6.07 | 4.58 | +24.5% | 0.0000 |
| 16 | 5.67 | 3.88 | +31.6% | 0.0000 |
| 24 | 5.63 | 4.14 | +26.4% | 0.0000 |
| 32 | 5.87 | 4.13 | +29.6% | 0.0000 |
| 48 | 5.58 | 4.63 | +17.2% | 0.0000 |
| 64 | 5.70 | 4.70 | +17.5% | 0.0000 |
| 96 | 5.82 | 4.85 | +16.6% | 0.0000 |
| 128 | 5.86 | 5.55 | +5.3% | 0.0000 |
| 256 | 6.70 | 6.48 | +3.2% | 0.0000 |
| 512 | 10.32 | 9.75 | +5.5% | 0.0000 |
| 1024 | 15.37 | 13.70 | +10.9% | 0.0000 |
| 1536 | 18.67 | 18.47 | +1.1% | 0.0000 |
| 2048 | 23.93 | 21.50 | +10.1% | 0.0000 |
| 3072 | 31.97 | 29.49 | +7.7% | 0.0000 |
| 4096 | 43.10 | 41.28 | +4.2% | 0.0000 |

Consistent pattern across all 5 shapes: large speedup (+20-70%) at small batch sizes, tapering toward low single digits at large batch sizes.
The kernel goes from memory-bandwidth-bound to compute-bound as M grows, and tile/warp/pipeline-depth tuning matters far more in the memory-bound regime.

### End-to-end

Served under `vllm/vllm-openai:v0.27.1`:

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --port=8080 --model=/mnt/models --served-model-name=qwen-27b \
    --tensor-parallel-size=2 --max-model-len=8192 --reasoning-parser=qwen3 \
    --enable-auto-tool-choice --tool-call-parser=qwen3_coder \
    --max-num-seqs=64
# env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Benchmarked using `vllm bench serve`.

#### Low-concurrency (`--request-rate 1 --max-concurrency 1`)

```bash
vllm bench serve \
    --backend openai-chat --endpoint /chat/completions \
    --model qwen-27b --dataset-name random \
    --num-prompts 20 --random-input-len 256 --random-output-len 128 \
    --request-rate 1 --max-concurrency 1 --temperature 0 \
    --save-result --save-detailed
```

| Config | Successful | Failed | Duration (s) | Output tok/s | Peak output tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 20 | 0 | 88.96 | 28.78 | 93.00 | 98.12 | 153.07 | 126.12 | 568.88 | 33.53 | 31.73 | 60.01 | 33.35 | 31.68 | 69.93 |
| tuned | 20 | 0 | 68.99 | 37.10 | 41.00 | 126.52 | 284.22 | 147.49 | 2290.16 | 24.64 | 24.59 | 26.06 | 24.48 | 24.52 | 40.48 |

Output tok/s +28.9%, mean TPOT -26.5% -- a real, substantial win.

#### High-concurrency / throughput (`--request-rate inf --max-concurrency 64 --num-prompts 500`)

```bash
vllm bench serve \
    --backend openai-chat --endpoint /chat/completions \
    --model qwen-27b --dataset-name random \
    --num-prompts 500 --random-input-len 256 --random-output-len 128 \
    --request-rate inf --max-concurrency 64 --temperature 0 \
    --save-result --save-detailed
```

| Config | Successful | Failed | Duration (s) | Output tok/s | Peak output tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 500 | 0 | 84.44 | 757.96 | 1216.00 | 2584.25 | 1484.97 | 1380.37 | 4663.98 | 71.87 | 73.04 | 80.96 | 71.46 | 53.74 | 429.14 |
| tuned | 500 | 0 | 86.07 | 743.59 | 1648.00 | 2535.23 | 1999.32 | 1833.65 | 5457.73 | 69.54 | 69.79 | 83.13 | 69.17 | 50.57 | 528.96 |

Output tok/s -1.9%, mean TPOT -3.2% -- small and mixed (a small throughput
regression alongside a small TPOT improvement), not literally zero change,
but well within run-to-run noise -- not a meaningful win either way. This
isn't a contradiction of the low-concurrency result -- it's consistent with
the kernel-level tables above: at `--max-concurrency 64` the server peaks at
~127 concurrent requests, pushing the actual GEMM batch sizes into the
large-M / compute-bound regime, where the kernel-level tables already show
the tuned config's advantage shrinking to single digits. This config
primarily helps low-latency / low-batch serving, with negligible effect at
saturation.
