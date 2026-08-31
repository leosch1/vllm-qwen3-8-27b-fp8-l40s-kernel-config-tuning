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
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 19.14 | 5.32 | +72.2% |
| 2 | 15.32 | 7.43 | +51.5% |
| 4 | 13.34 | 5.35 | +59.9% |
| 8 | 8.28 | 5.14 | +37.9% |
| 16 | 9.07 | 5.81 | +36.0% |
| 24 | 7.98 | 6.12 | +23.3% |
| 32 | 7.77 | 5.86 | +24.5% |
| 48 | 7.71 | 7.15 | +7.2% |
| 64 | 7.68 | 6.58 | +14.3% |
| 96 | 10.78 | 8.93 | +17.1% |
| 128 | 9.97 | 8.96 | +10.2% |
| 256 | 18.65 | 16.31 | +12.5% |
| 512 | 33.36 | 29.03 | +13.0% |
| 1024 | 61.47 | 54.59 | +11.2% |
| 1536 | 91.93 | 79.86 | +13.1% |
| 2048 | 122.22 | 119.33 | +2.4% |
| 3072 | 188.36 | 175.49 | +6.8% |
| 4096 | 256.79 | 235.39 | +8.3% |

### N=8192, K=5120, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 13.47 | 4.52 | +66.4% |
| 2 | 8.89 | 5.06 | +43.1% |
| 4 | 8.12 | 4.46 | +45.2% |
| 8 | 7.62 | 5.03 | +34.0% |
| 16 | 7.68 | 4.60 | +40.0% |
| 24 | 7.70 | 5.31 | +31.1% |
| 32 | 7.77 | 4.85 | +37.6% |
| 48 | 7.54 | 5.76 | +23.5% |
| 64 | 7.34 | 5.49 | +25.2% |
| 96 | 7.78 | 6.42 | +17.4% |
| 128 | 7.99 | 7.88 | +1.4% |
| 256 | 10.72 | 8.97 | +16.3% |
| 512 | 17.62 | 14.96 | +15.1% |
| 1024 | 29.15 | 26.54 | +8.9% |
| 1536 | 41.81 | 39.99 | +4.3% |
| 2048 | 55.31 | 51.61 | +6.7% |
| 3072 | 82.86 | 73.00 | +11.9% |
| 4096 | 112.64 | 104.37 | +7.3% |

### N=7168, K=5120, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 12.43 | 4.61 | +62.9% |
| 2 | 9.20 | 4.74 | +48.5% |
| 4 | 8.60 | 5.63 | +34.6% |
| 8 | 7.97 | 5.91 | +25.8% |
| 16 | 9.05 | 4.65 | +48.6% |
| 24 | 8.41 | 4.86 | +42.2% |
| 32 | 7.65 | 4.94 | +35.5% |
| 48 | 7.87 | 5.33 | +32.3% |
| 64 | 7.28 | 5.60 | +23.2% |
| 96 | 8.10 | 6.58 | +18.8% |
| 128 | 8.34 | 6.95 | +16.7% |
| 256 | 9.36 | 9.39 | -0.3% |
| 512 | 16.50 | 15.35 | +7.0% |
| 1024 | 28.36 | 26.58 | +6.3% |
| 1536 | 37.18 | 33.02 | +11.2% |
| 2048 | 49.23 | 45.53 | +7.5% |
| 3072 | 71.90 | 66.62 | +7.3% |
| 4096 | 96.99 | 89.06 | +8.2% |

### N=5120, K=8704, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 14.95 | 6.36 | +57.5% |
| 2 | 13.51 | 5.24 | +61.2% |
| 4 | 10.86 | 6.16 | +43.3% |
| 8 | 11.42 | 5.41 | +52.6% |
| 16 | 10.37 | 4.97 | +52.0% |
| 24 | 10.21 | 5.59 | +45.3% |
| 32 | 10.27 | 5.71 | +44.4% |
| 48 | 10.09 | 6.44 | +36.2% |
| 64 | 10.18 | 6.45 | +36.7% |
| 96 | 10.89 | 7.88 | +27.6% |
| 128 | 11.03 | 9.23 | +16.4% |
| 256 | 13.23 | 12.04 | +9.0% |
| 512 | 22.77 | 21.22 | +6.8% |
| 1024 | 34.52 | 31.84 | +7.8% |
| 1536 | 46.74 | 41.77 | +10.6% |
| 2048 | 59.25 | 53.03 | +10.5% |
| 3072 | 87.52 | 77.21 | +11.8% |
| 4096 | 119.58 | 111.28 | +6.9% |

### N=5120, K=3072, device=NVIDIA_L40S
| batch size (M) | default (us) | tuned (us) | speedup |
|---:|---:|---:|---:|
| 1 | 9.33 | 4.22 | +54.7% |
| 2 | 7.26 | 4.41 | +39.2% |
| 4 | 6.20 | 5.34 | +13.9% |
| 8 | 6.13 | 4.37 | +28.8% |
| 16 | 5.80 | 4.06 | +30.0% |
| 24 | 5.66 | 4.90 | +13.4% |
| 32 | 5.66 | 3.98 | +29.8% |
| 48 | 5.98 | 4.26 | +28.8% |
| 64 | 5.65 | 4.45 | +21.2% |
| 96 | 6.24 | 4.66 | +25.3% |
| 128 | 5.79 | 5.07 | +12.5% |
| 256 | 6.64 | 6.80 | -2.5% |
| 512 | 10.08 | 8.81 | +12.7% |
| 1024 | 14.51 | 13.21 | +9.0% |
| 1536 | 18.25 | 17.17 | +5.9% |
| 2048 | 22.40 | 20.89 | +6.8% |
| 3072 | 31.37 | 28.80 | +8.2% |
| 4096 | 42.78 | 39.79 | +7.0% |

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
