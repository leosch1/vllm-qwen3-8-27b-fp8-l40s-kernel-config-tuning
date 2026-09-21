# 03 — GROUP_SIZE_M single-anchor patch

Supports [blog §5 "Which parameter is most affected?"](../../blog/index.html).

## What

The narrowest possible intervention: identical to the original tuned
config everywhere, for all 5 shapes and every batch-size anchor, except
**one single value** — `gate_up_proj`'s `M=2048` entry, where
`GROUP_SIZE_M` changes from `1` to `16`. Every other anchor, including
`gate_up_proj`'s other large-`M` ones (512, 1024, 1536, 3072, 4096),
stays exactly as the original tuning run produced it. Deploy that config,
capture real serving traffic at c=64 the same way as
[`02-nsys-shape-profiling`](../02-nsys-shape-profiling/), and see whether
fixing just this one value recovers the regression that shape/batch-size
combination showed.

## Why

`02` identified `gate_up_proj` at `M≈2000` as the one cluster responsible
for erasing the tuned config's win at high concurrency, and named
`GROUP_SIZE_M` — the parameter controlling GEMM tile scheduling order —
as the likely culprit (per Triton's own docs, it "affects the L2 cache
hit rate"). This experiment tests that hypothesis with the smallest
change that could possibly confirm or refute it: touch nothing else,
change this one value, measure.

## How

1. **Build the config.** All 5 shapes' JSONs copied byte-for-byte from
   the original tuned config, except `gate_up_proj`'s `"2048"` entry's
   `GROUP_SIZE_M`. `singlefix-config/` holds the complete, ready-to-deploy
   5-file set — diff it against your own tuned config to confirm the
   single-value change.
2. **Deploy and capture**, identical methodology to `02`: DCGM paused,
   `nsys profile --cuda-graph-trace=node`, `vllm bench serve` at c=64
   (256/128 random input/output length) against a live server with this
   config mounted.
3. **Extract and classify.** `extract_kernels.py` pulls every
   `_w8a8_triton_block_scaled_mm` launch's `(start, end, gridX, blockX)`
   from the capture's exported sqlite. Since `GROUP_SIZE_M` doesn't affect
   grid or block dimensions, a launch's `(gridX, blockX)` pair is enough
   to match it against the original tuned config's already-known
   launch-shape signatures — reuse `02`'s shape/`M` mapping directly, no
   new classification logic needed.
4. **Bin and compare.** Feed the classified per-launch durations through
   `../02-nsys-shape-profiling/build_histograms.py` — same bin edges, same
   category scheme — so the result lines up directly against `02`'s
   `c64_default` and `c64_tuned` histograms.

## Result

The `gate_up_proj`/`M≈2000` cluster that dominated the c=64 histogram in
`02` moves back left, landing close to where the default config's
histogram already had it:

| config | that cluster's mean duration |
|---|---:|
| default | ~1111µs |
| tuned (unpatched) | ~1961µs |
| **singlefix** | **~1063µs** |

Changing **one value** out of the entire 5-shape config recovers the
regression. Full histogram: [`results/singlefix-histogram.json`](./results/singlefix-histogram.json)
— compare its `c64_singlefix` series directly against
[`02`'s `c64_default`/`c64_tuned`](../02-nsys-shape-profiling/results/duration-histograms.json).

This confirms `GROUP_SIZE_M` — and specifically its value at the exact
batch size that regressed — as the parameter responsible, not some
broader property of the tuned config. [`04-l2-flush-retune`](../04-l2-flush-retune/)
picks up from here: not by hand-patching individual anchors, but by fixing
the assumption in the tuning script that picks a bad `GROUP_SIZE_M` in
the first place.

## Files

- `singlefix-config/` — the complete, deployable 5-shape config (one
  value different from the original tuned config).
- `extract_kernels.py` — pulls raw per-launch rows from a captured
  nsys export's sqlite.
- `results/singlefix-histogram.json` — the resulting c=64 histogram.
