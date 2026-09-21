# 31 — Original (unfixed) tuner, FULL 18-M sweep, gate_up_proj, v0.27.1

**The missing control.** This repo had exactly ONE full-sweep run of the
unfixed tuner — the one that produced the committed `tuned-configs/`
(`experiments/01/tuning-and-comparison-v0.27.1.log`, Aug 2026). It had never
been repeated. Every unfixed rerun since (experiments 28, 30) used
`--batch-size 1024`, which starts a cold process and jumps straight to
M=1024 — NOT the real execution context.

Two fresh full sweeps were run today, concurrently, on two physically
separate nodes: exp30 on `gpu-node-a`, exp30b on
`gpu-node-b`. Unmodified script
(leosch1/vllm@qwen3-8-27b-fp8-dense-tuning, commit
2d8edac76fc6adcfe7fa46eae8c5d3fdf18bde30), gate_up_proj only,
`batch_size=None` (all 18 M), image `vllm/vllm-openai:v0.27.1`.

## Result: GROUP_SIZE_M, gate_up_proj

| M | committed | exp30 (gpu2) | exp30b (gpu1) |
|---:|---:|---:|---:|
| 1 | 32 | 32 | 16 |
| 2 | 64 | 1 | 32 |
| 4 | 32 | 64 | 16 |
| 8 | 64 | 16 | 64 |
| 16 | 1 | 32 | 32 |
| 24 | 16 | 16 | 64 |
| 32 | 1 | 16 | 1 |
| 48 | 64 | 16 | 32 |
| 64 | 64 | 1 | 16 |
| 96 | 64 | 64 | 16 |
| 128 | 16 | 64 | 32 |
| 256 | 32 | 32 | 32 |
| **512** | **1** | **1** | **1** |
| **1024** | **1** | **1** | **1** |
| **1536** | **1** | **1** | **1** |
| **2048** | **1** | **1** | **1** |
| **3072** | **1** | **1** | **1** |
| **4096** | **1** | **1** | **1** |

**Large M (512–4096): 6/6 on `GROUP_SIZE_M=1` in all three sweeps — 18/18
across three independent runs on two different GPUs.**

Small M (1–256), where `GROUP_SIZE_M` is a *provable no-op*
(`num_pid_m = ceil(M / BLOCK_SIZE_M) = 1`, so every candidate value compiles
to the same schedule): agreement is at chance level — 4/12, 3/12, 2/12
between pairs (chance = 3/12).

Full-config identity vs. committed at large M: exp30 5/6, exp30b 4/6;
exp30 vs exp30b 3/6. So the *other* parameters (`num_stages` especially)
wobble run to run — `GROUP_SIZE_M=1` at large M is **more** reproducible
than anything else in the config.

## What this establishes

1. **The committed config is not a fluke.** The unfixed tuner reproducibly
   selects `GROUP_SIZE_M=1` at every anchor where the parameter has any
   causal effect, on independent hardware.
2. **The determinism is specific to where the parameter matters.** Random
   in the no-op region, perfectly deterministic where it bites. That
   contrast is itself the evidence that this is a systematic preference,
   not measurement noise.
3. **It corrects experiments 28 and 30.** Their "fresh runs rarely
   reproduce `GROUP_SIZE_M=1`" finding (1 of 9 at M=1024) was an artifact
   of `--batch-size 1024`. Those results should NOT be cited as evidence
   that the tuner is unstable in this parameter.

## What it does NOT explain

The mechanism. The obvious candidate — GPU thermal drift over a long sweep
(experiment 30 measured +17% across a session) — is contradicted by our own
data: experiment 28's runs 2 and 3 executed on an already-hot GPU with a warm
Triton compile cache and still produced grouped values. So "hot by the time
it reaches large M" is not sufficient. Something else about the sequential
full-sweep execution context drives it, and it is not isolated.

## Files

- `original-tuner-full-sweep-job.yaml` / `original-tuner-full-sweep-gpu1-job.yaml`
- `exp30-results-gpu2/`, `exp30b-results-gpu1/` — the produced config JSONs
- `exp30-gpu2-run.log`, `exp30b-gpu1-run.log` — full run logs
