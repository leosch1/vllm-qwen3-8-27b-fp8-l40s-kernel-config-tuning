# 30 — Instrumented tuner candidate times

## What

`instrumented_tune.py` patches the real, unmodified v0.27.1 tuning script
with one change: `tune()` records every candidate's `(config,
measured_time)` instead of discarding losers. Where every other experiment
in this project can only see the tuner's final winner, this logs the full
per-candidate search — every `GROUP_SIZE_M` value's actual measured time,
in the real search's own execution order, for `gate_up_proj` at M=1024 and
M=2048, 3 repeats each.

`unmodified-tuner-gpu2-job.yaml` (exp29b) is a discriminating follow-up:
the same question, but running the real *unmodified* tuner (no
instrumentation) across all 5 shapes per run (shapes interleaved between
repeats), instead of `gate_up_proj` alone searched back-to-back.

`instrumented_tune_5shape.py` (exp29c, referenced in `exp29b`'s own
comments) was written to combine the instrumentation with the 5-shape
interleaved structure, to separate "does instrumentation itself change the
outcome" from "does searching the same shape back-to-back (vs. interleaved
with other shapes) change the outcome" — **this script was never run; no
results exist for it.**

## Result

**`exp29-results/` (instrumented, `gate_up_proj` only, same-shape
back-to-back): `GROUP_SIZE_M=1` won the real search's own head-to-head
6/6** (both M=1024 and M=2048, all 3 repeats each) — holding each repeat's
own winning `BLOCK_SIZE_M/N/K`/`num_warps`/`num_stages` fixed and
comparing all 4 `GROUP_SIZE_M` candidates' actual measured times,
`GROUP_SIZE_M=1` was the fastest of the 4 every single time. Full
per-candidate data (all evaluated configs, all repeats, raw per-iteration
latencies): `exp29-results/candidate_times.jsonl` (7,680 records).
Per-repeat winners: `exp29-results/winners.json`. Full run transcript
including the derived head-to-head comparison: `exp29-run.log`.

**`exp29b-run.log` (unmodified tuner, all 5 shapes per run, shapes
interleaved between repeats): `GROUP_SIZE_M` = 64, 1, 16 across 3 runs —
only 1/3 landing on `GROUP_SIZE_M=1`**, matching `28-group-size-m-order-bias-probe`'s
unmodified-tuner reruns (0/6) rather than this experiment's own
same-shape-repeated result (6/6).

The two results disagree, and since instrumentation and shape-sequence
both differ between them at once, this experiment doesn't establish which
one explains the difference — that's exactly what the never-run
`instrumented_tune_5shape.py` was meant to isolate.

## Files

- `instrumented_tune.py` / `instrumented-tuner-job.yaml` — the
  instrumented tuner and its Job.
- `instrumented_tune_5shape.py` — written, never run (see above).
- `unmodified-tuner-gpu2-job.yaml` — the discriminating unmodified-tuner,
  5-shape-interleaved run (exp29b).
- `exp29-results/candidate_times.jsonl`, `exp29-results/winners.json` —
  raw per-candidate and per-repeat data from the instrumented run.
- `exp29-run.log`, `exp29b-run.log` — full run transcripts.
