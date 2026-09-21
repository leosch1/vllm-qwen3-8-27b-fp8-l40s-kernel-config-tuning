# PR evidence: e2e serving benchmark results

`qwen-27b-pr-evidence` (`Qwen/Qwen3.8-27B-FP8`, tensor-parallel-size=2), served
under `vllm/vllm-openai:v0.27.1` (see `README.md` in this directory). Same
`vllm bench serve` client/tokenizer setup as `performance-tests/benchmark-results.md`.

Note: these numbers are **not** directly comparable to the production
`qwen-27b` rows in `performance-tests/benchmark-results.md` -- this is a
different vLLM version end-to-end (not just a different kernel config), so
differences could come from many things besides the tuned config. The only
valid comparison here is default-vs-tuned *within this table*, both on the
same `v0.27.1` image.

## Low-concurrency (`--request-rate 1 --max-concurrency 1`, matches `performance-tests/benchmark-results.md`)

| Config | Successful | Failed | Duration (s) | Output tok/s | Peak output tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 20 | 0 | 84.44 | 30.32 | 32.00 | 103.37 | 150.26 | 123.14 | 555.81 | 31.78 | 31.68 | 33.27 | 31.56 | 31.75 | 34.27 |
| tuned (rerun pending) | 20 | 0 | 68.99 | 37.10 | 41.00 | 126.52 | 284.22 | 147.49 | 2290.16 | 24.64 | 24.59 | 26.06 | 24.48 | 24.52 | 40.48 |

## High-concurrency / throughput (`--request-rate inf --max-concurrency 64 --num-prompts 500`, matches vllm-project/vllm#23504's convention)

Server run with `--max-num-seqs=64` instead of production's `--max-num-seqs=1`
(see `servingruntime.yaml`) -- that setting exists only to avoid CUDA graph
capture OOM on our older RHOAI image and serializes the engine to one
sequence at a time, making a high-concurrency benchmark meaningless (and
triggering Route timeouts) if left in place. `v0.27.1` handled `--max-num-seqs=64`
without OOM.

| Config | Successful | Failed | Duration (s) | Output tok/s | Peak output tok/s | Total tok/s | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | P99 ITL (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | 500 | 0 | 83.54 | 766.06 | 1216.00 | 2611.85 | 1259.91 | 1165.28 | 3886.03 | 72.84 | 73.64 | 81.03 | 72.41 | 53.86 | 420.87 |
| tuned (rerun pending) | 500 | 0 | 86.07 | 743.59 | 1648.00 | 2535.23 | 1999.32 | 1833.65 | 5457.73 | 69.54 | 69.79 | 83.13 | 69.17 | 50.57 | 528.96 |

## Summary

- **Both profiles**: default-config numbers above are from a rerun (PASS 1);
  the tuned rows are still from the original run, pending a PASS 2 rerun for
  a like-for-like comparison -- don't compute deltas off either table until
  both rows in it are from the same rerun.
- Prior run's story (kept for reference, to be reconfirmed after PASS 2):
  low-concurrency showed a real win, high-concurrency showed essentially no
  difference, consistent with the kernel-level tables
  (`performance-tests/qwen3-8-27b-fp8-dense-pr-evidence-v0.27.1.log`) showing
  the tuned config's speedup shrinking to single digits at the large batch
  sizes a saturated `--max-concurrency 64` server actually runs at.
- Net: this dense config primarily helps low-latency / low-batch serving,
  with negligible effect at saturation -- worth stating plainly in the PR
  rather than only reporting the flattering low-concurrency number.
