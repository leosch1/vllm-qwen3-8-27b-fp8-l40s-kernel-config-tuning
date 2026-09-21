# 29 — L2-flush autotune rerun, v0.27.1

> Fully covered in [`experiments/04-l2-flush-retune`](../../04-l2-flush-retune/).

## What

Reran the L2-flush-patched autotuning script (`benchmark_w8a8_block_fp8_l2flush.py`)
at full production scope (5 shapes × 18 `M` values) on `vllm/vllm-openai:v0.27.1`
— the same stack that produced this repo's original `tuned-configs/` — instead
of the RHOAI production image the earlier L2-flush rerun (`25-l2-flush-autotune-rerun`)
used. This is what actually produced the config committed at
[`tuned-configs/`](../../../tuned-configs/) today.

## Files

- `benchmark_w8a8_block_fp8_l2flush.py` / `l2-flush-autotune-rerun-v0271-job.yaml`
  — the patched tuner and its Kubernetes Job.
- `l2-flush-tuned-configs-v0271/` — the resulting 5 config files.
- `exp28-run.log` — raw run output.
