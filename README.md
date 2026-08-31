# Qwen3.8-27B-FP8 dense W8A8 block-FP8 kernel config tuning (NVIDIA L40S)

Tuning vLLM's dense W8A8 block-FP8 Triton GEMM kernel for [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (`tensor_parallel_size=2`) on NVIDIA L40S, and measuring whether it actually helps.

Also backs a PR to [vllm-project/vllm](https://github.com/vllm-project/vllm) contributing the resulting configs upstream: _TODO: link once opened_.

Environment: `vllm/vllm-openai:v0.27.1`, 2x NVIDIA L40S.

## Background

### Why vLLM uses a Triton kernel here

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
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 21.17 | 5.98 | +71.7% |
| 2 | 12.68 | 5.17 | +59.2% |
| 4 | 11.70 | 5.29 | +54.8% |
| 8 | 8.23 | 5.17 | +37.2% |
| 16 | 8.24 | 5.64 | +31.6% |
| 24 | 7.99 | 5.91 | +26.0% |
| 32 | 7.79 | 5.88 | +24.5% |
| 48 | 7.72 | 7.04 | +8.8% |
| 64 | 7.78 | 6.64 | +14.6% |
| 96 | 9.95 | 8.66 | +13.0% |
| 128 | 9.71 | 8.83 | +9.0% |
| 256 | 18.38 | 16.47 | +10.4% |
| 512 | 34.21 | 29.02 | +15.2% |
| 1024 | 62.90 | 61.32 | +2.5% |
| 1536 | 94.09 | 93.27 | +0.9% |
| 2048 | 124.72 | 125.64 | -0.7% |
| 3072 | 197.24 | 179.42 | +9.0% |
| 4096 | 263.28 | 256.54 | +2.6% |

### N=8192, K=5120, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 13.86 | 5.09 | +63.3% |
| 2 | 10.03 | 4.51 | +55.1% |
| 4 | 9.23 | 4.61 | +50.1% |
| 8 | 8.80 | 4.53 | +48.6% |
| 16 | 8.75 | 4.95 | +43.4% |
| 24 | 8.70 | 5.34 | +38.6% |
| 32 | 7.64 | 4.84 | +36.6% |
| 48 | 7.64 | 9.95 | -30.4% |
| 64 | 8.41 | 5.51 | +34.5% |
| 96 | 7.87 | 6.58 | +16.4% |
| 128 | 7.74 | 6.52 | +15.7% |
| 256 | 9.54 | 9.29 | +2.6% |
| 512 | 16.36 | 14.55 | +11.1% |
| 1024 | 29.90 | 26.30 | +12.0% |
| 1536 | 41.97 | 38.48 | +8.3% |
| 2048 | 54.17 | 50.82 | +6.2% |
| 3072 | 85.17 | 81.73 | +4.0% |
| 4096 | 116.38 | 111.00 | +4.6% |

### N=7168, K=5120, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 11.97 | 4.59 | +61.7% |
| 2 | 9.55 | 4.48 | +53.1% |
| 4 | 8.62 | 4.37 | +49.3% |
| 8 | 8.01 | 4.52 | +43.6% |
| 16 | 7.69 | 4.43 | +42.4% |
| 24 | 7.59 | 5.14 | +32.3% |
| 32 | 7.59 | 5.41 | +28.7% |
| 48 | 7.67 | 5.39 | +29.8% |
| 64 | 8.16 | 5.23 | +35.9% |
| 96 | 7.80 | 6.50 | +16.7% |
| 128 | 7.52 | 7.66 | -1.8% |
| 256 | 9.25 | 8.45 | +8.7% |
| 512 | 15.57 | 14.27 | +8.4% |
| 1024 | 27.88 | 26.13 | +6.3% |
| 1536 | 36.25 | 32.76 | +9.6% |
| 2048 | 48.65 | 44.32 | +8.9% |
| 3072 | 72.25 | 70.84 | +1.9% |
| 4096 | 101.00 | 94.90 | +6.0% |

### N=5120, K=8704, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 14.90 | 5.76 | +61.3% |
| 2 | 13.52 | 5.22 | +61.4% |
| 4 | 12.09 | 5.03 | +58.4% |
| 8 | 11.36 | 5.01 | +55.9% |
| 16 | 11.08 | 5.25 | +52.6% |
| 24 | 11.10 | 5.98 | +46.1% |
| 32 | 11.04 | 5.78 | +47.7% |
| 48 | 10.40 | 6.41 | +38.4% |
| 64 | 10.35 | 6.17 | +40.4% |
| 96 | 10.54 | 6.90 | +34.5% |
| 128 | 10.51 | 8.38 | +20.2% |
| 256 | 12.43 | 11.46 | +7.8% |
| 512 | 22.93 | 20.63 | +10.0% |
| 1024 | 33.95 | 31.82 | +6.3% |
| 1536 | 45.83 | 42.06 | +8.2% |
| 2048 | 58.78 | 53.56 | +8.9% |
| 3072 | 86.41 | 78.28 | +9.4% |
| 4096 | 123.35 | 118.56 | +3.9% |

### N=5120, K=3072, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 8.13 | 4.06 | +50.0% |
| 2 | 7.01 | 3.85 | +45.1% |
| 4 | 6.11 | 3.82 | +37.6% |
| 8 | 5.94 | 3.88 | +34.6% |
| 16 | 5.91 | 3.84 | +35.0% |
| 24 | 5.82 | 4.09 | +29.7% |
| 32 | 5.81 | 3.97 | +31.6% |
| 48 | 5.64 | 4.34 | +23.0% |
| 64 | 5.71 | 4.24 | +25.8% |
| 96 | 5.86 | 4.58 | +21.8% |
| 128 | 5.76 | 5.12 | +11.2% |
| 256 | 6.60 | 6.40 | +3.0% |
| 512 | 9.87 | 8.58 | +13.1% |
| 1024 | 14.45 | 13.26 | +8.2% |
| 1536 | 18.39 | 17.36 | +5.6% |
| 2048 | 24.59 | 21.57 | +12.3% |
| 3072 | 31.64 | 28.93 | +8.6% |
| 4096 | 43.60 | 39.61 | +9.2% |

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
# env: HF_HOME=/tmp/hf_home PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HOME=/tmp
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
