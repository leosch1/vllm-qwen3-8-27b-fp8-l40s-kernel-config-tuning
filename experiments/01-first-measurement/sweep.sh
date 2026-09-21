#!/usr/bin/env bash
# Runs against the server started by e2e-server-pod.yaml (localhost:8080
# inside that pod). Prints one `vllm bench serve` summary per concurrency
# level; results/e2e-concurrency-decay.json is the output tok/s delta
# between a default-config run and a tuned-config run of this same sweep.
set -euo pipefail

CONCURRENCIES=(1 2 4 8 16 32 48 64 96 128)

for c in "${CONCURRENCIES[@]}"; do
  # 16x sample count relative to a single low-concurrency run, for stability.
  num_prompts=$((c * 16))
  echo "=== concurrency=$c num_prompts=$num_prompts ==="
  vllm bench serve \
    --backend openai-chat --base-url http://localhost:8080/v1 --endpoint /chat/completions \
    --model qwen-27b --dataset-name random \
    --num-prompts "$num_prompts" --random-input-len 256 --random-output-len 128 \
    --request-rate inf --max-concurrency "$c" --temperature 0
done
