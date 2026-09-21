# 28 — GROUP_SIZE_M order-bias probe

## What

Three separate scripts probing why the real tuning script reliably picks
`GROUP_SIZE_M=1` for `gate_up_proj` at large `M`, given other experiments
in this project measured grouped values as genuinely faster under careful
measurement:

- **`group_size_m_order_bias.py`** — tests whether the real tuner's own
  selection logic (`GROUP_SIZE_M` tested innermost, in fixed order
  `[1, 16, 32, 64]`, `best_time` updated only on strict `<`, so a tie keeps
  the earlier-tested candidate) could make `1` win more often than its
  true speed would predict, just because it's tested first and only a
  10-iteration sample separates it from the next candidate. Compares
  `GROUP_SIZE_M=1`'s win rate in forward order `[1,16,32,64]` against
  reversed order `[64,32,16,1]`, at `gate_up_proj` M=1024's actual winning
  config.
- **`group_size_m_order_bias_all_m.py`** — the same win-rate probe repeated
  at all six large-`M` anchors (512, 1024, 1536, 2048, 3072, 4096), each
  with its own actual winning config.
- **`version-diag-job.yaml`** — a quick diagnostic comparing vLLM/Triton/
  PyTorch versions and the actual `_w8a8_triton_block_scaled_mm` kernel
  source between `vllm/vllm-openai:v0.27.1` and the RHOAI production
  image, to check whether a stack difference (not just measurement noise)
  could explain different `GROUP_SIZE_M` outcomes between the two.
- **`real-tuner-repeat-m1024-job.yaml`** / **`rhoai-image-repeat-m1024-trainjob.yaml`**
  — reran the real, unmodified tuning script at `gate_up_proj` M=1024
  multiple independent times (v0.27.1 stack and RHOAI stack respectively),
  to see how often a fresh run actually lands on `GROUP_SIZE_M=1`.

## Result

**`version-diag-results.log`**: the Triton kernel source is byte-identical
between the two images. What differs is the compiler/runtime stack —
Triton 3.6.0 vs 3.7.1, PyTorch 2.11.0a0 vs 2.13.0, vLLM 0.24.0+rhaiv.9 vs
0.27.1 (CUDA version and GPU model are identical). The log names the
Triton JIT version as the most plausible candidate for different
`GROUP_SIZE_M` outcomes between stacks, but says explicitly this isn't
verified at the PTX/SASS level — flagged as the leading candidate, not a
proven root cause.

**`real-tuner-repeat-m1024-results.log`**: across 6 independent fresh
full-search reruns of the real tuner (v0.27.1 stack) at `gate_up_proj`
M=1024, `GROUP_SIZE_M` values seen were 16, 32, 32, 16, 64, 64 —
**`GROUP_SIZE_M=1` (the committed/historical value): 0/6.**

**`rhoai-image-repeat-m1024-results.log`**: 3 runs on the RHOAI stack
instead landed `GROUP_SIZE_M=1` at least twice (see the raw JSON in that
log for exact values).

No saved output exists in this folder for `group_size_m_order_bias.py` /
`group_size_m_order_bias_all_m.py`'s own win-rate numbers — only the
scripts and their methodology are preserved here, not a captured run.

## Files

- `group_size_m_order_bias.py` / `group-size-m-order-bias-job.yaml` and
  `group_size_m_order_bias_all_m.py` / `group-size-m-order-bias-all-m-job.yaml`
  — the order-bias win-rate probes (see "What" above) and their deployment
  manifests.
- `version-diag-job.yaml` / `version-diag-results.log` — the stack
  comparison.
- `real-tuner-repeat-m1024-job.yaml` / `real-tuner-repeat-m1024-results.log`
  — repeated real-tuner reruns, v0.27.1 stack.
- `rhoai-image-repeat-m1024-trainjob.yaml` / `rhoai-image-repeat-m1024-results.log`
  — repeated real-tuner reruns, RHOAI stack.
