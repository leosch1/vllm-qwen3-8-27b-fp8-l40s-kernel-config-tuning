# 32 — Isolated single-M run at M=2048 (unfixed tuner, v0.27.1)

Minimal-diff counterpart to experiment 31: identical job except
`batch_size=None` → `batch_size=2048`. Unmodified tuner
(leosch1/vllm@qwen3-8-27b-fp8-dense-tuning), gate_up_proj only, one cold run
per pod, one pod per node.

## Result — the two runs DISAGREE

| run | node | `GROUP_SIZE_M` at M=2048 |
|---|---|---|
| exp31 | `gpu2-gkm7z` | **16** |
| exp31b | `gpu-mh67p` | **1** |

Both selected `BLOCK_SIZE_M/N/K=128`, `num_warps=8`, `num_stages=3`;
`GROUP_SIZE_M` is the only parameter that differs.

## Consolidated picture

| context | M | `GROUP_SIZE_M=1` rate |
|---|---|---|
| isolated single-M | 1024 | 1/9 (experiment 28 ×6, experiment 30's exp29b run ×3) |
| full 18-M sweep | 1024 | 3/3 (committed, exp30, exp30b) |
| isolated single-M | 2048 | **1/2** (exp31, exp31b) |
| full 18-M sweep | 2048 | 3/3 (committed, exp30, exp30b) |

## Interpretation

The full-sweep context reliably produces `GROUP_SIZE_M=1` at both M values
(6/6 across the two anchors). Isolated cold runs do not, but the rate rises
with M: ~11% at M=1024, ~50% at M=2048.

That directional trend matches the margin data from experiment 30's raw
per-candidate timings. Within the large-tile family the winner comes from
(`BLOCK=128/128/128`, `num_warps=8`), `GROUP_SIZE_M=1`'s median advantage
over the best grouped value is **−1.29% at M=1024** and **−2.40% at
M=2048** — roughly double. A larger margin is harder for execution-context
effects (clock state, compile cadence) to overturn, so `GROUP_SIZE_M=1`
survives isolated runs more often at M=2048.

**But M=2048 is NOT context-robust.** An earlier claim in this
investigation — made off exp31b alone, before exp31 finished — that M=2048
"lands on `GROUP_SIZE_M=1` regardless of context" is **falsified** by
exp31's `16`. With n=2 the true isolated-run rate at M=2048 could plausibly
be anywhere from ~10% to ~90%; the only sound statements are (a) full sweeps
give `1` consistently, and (b) isolated runs give `1` more often at M=2048
than at M=1024.

## Caveat on sample size

n=2 here, n=9 at M=1024. The M=1024 side is reasonably solid; the M=2048
side is not. Anyone extending this should add repeats before quoting a rate.
