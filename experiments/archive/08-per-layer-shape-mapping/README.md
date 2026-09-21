# 08 — Per-layer shape mapping (+ GUI walkthrough)

**Status:** confirmed. This is the experiment that turned "the c=64
regression" from a single aggregate number into a shape-and-`M`-specific
finding — the foundation for
[`09-isolated-vs-real-flops-inflation`](../09-isolated-vs-real-flops-inflation/).
Merges what were originally two separate planned write-ups (per-layer
grid/block reverse-mapping, and a GUI walkthrough of one forward pass in
the Nsight Systems timeline) since they're really one investigation: the
GUI walkthrough is how the reverse-mapping logic was first validated by
eye before being trusted for the bulk numeric analysis.

## What

Reverse-engineer every real captured `_w8a8_triton_block_scaled_mm`
launch's logical GEMM shape `(N, K, M)` from its raw CUDA launch grid/block
dimensions (the only thing the trace directly records), map each of the 5
resulting shapes onto the model's actual named weight matrices
(`gate_up_proj`, `in_proj_qkvz`, `qkv_proj`, `out_proj`/`o_proj`,
`down_proj`), and validate the whole mapping by eye against one real forward
pass in the Nsight Systems GUI timeline before trusting it for aggregate
numbers.

## Why

The nsys trace never records the logical `N`/`K`/`M` a GEMM call was made
with — only the physical launch grid/block it produced. To answer "which
weight matrix is regressing, and at what batch size," that grid/block data
has to be inverted back to a shape, which requires knowing both configs'
exact tile-size formulas (not just guessing).

## How

**The formula, confirmed from real vLLM source** (`w8a8_triton_block_scaled_mm`'s
`grid(META)` function): `gridX = ceil(M/BLOCK_SIZE_M) × ceil(N/BLOCK_SIZE_N)`,
`blockX = num_warps × 32`. Default's config is a fixed formula always
(`BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, num_warps=4`) — unambiguous, no lookup
table. Tuned's config is a per-`M`-anchor lookup from that shape's own tuned
JSON — and critically, **both** `BLOCK_SIZE_M` *and* `BLOCK_SIZE_N` are
looked up (not just `M`, which was an early mistake — `BLOCK_SIZE_N` drops
to 64 at the smallest M-anchors for several shapes and was initially
hardcoded to 128, silently mislabeling some buckets until caught).

**The 5 real shapes**, all confirmed by reading actual vLLM source
(`qwen3_5.py`, `qwen3_next.py`, `qwen_gdn_linear_attn.py`), each traced to a
specific named projection:

| shape (N, K) | projection | layer scope |
|---|---|---|
| 17408, 5120 | `gate_up_proj` (dense FFN, fused gate+up) | all 64 layers |
| 8192, 5120 | `in_proj_qkvz` (linear-attn Q/K/V/Z) | 48 linear-attention layers |
| 7168, 5120 | `qkv_proj` (full-attn QKV+gate) | 16 full-attention layers |
| 5120, 3072 | `out_proj`/`o_proj` (mixer output) | all 64 layers |
| 5120, 8704 | `down_proj` (dense FFN down-proj) | all 64 layers |

(This model — `Qwen3_5ForConditionalGeneration` — is genuinely dense, not
MoE, with a hybrid linear-attention/full-attention structure:
`full_attention_interval=4` giving 48 linear-attention + 16 full-attention
layers out of 64 total, confirmed against the real deployed
`/mnt/models/config.json`.)

`N=5120` is genuinely ambiguous from grid/block alone — both `out_proj`
(K=3072) and `down_proj` (K=8704) produce identical launch dimensions,
coincidentally. Resolved instead via **duration-histogram bimodality**: each
sub-shape forms a tight, well-separated cluster in real launch duration, so
they're separable even though the launch signature itself can't tell them
apart. A `signature_maps.pkl`, built by enumerating `M=1..4096` against each
shape's real tuned-config JSON (correctly reading per-anchor
`BLOCK_SIZE_N`), resolves the remaining ~3–3.5% of launches population-wide
that are genuinely ambiguous even after that.

**GUI validation**: identified the timestamp range of one single real
forward pass in the Nsight Systems GUI timeline, then manually matched its
sequence of kernel launches (GEMM calls in their real firing order,
interleaved with attention/linear-attention/NCCL kernels) against the
model's known layer structure — confirming the reverse-mapping logic
produces a sequence that actually matches the real architecture (e.g. the
expected count and ordering of `gate_up_proj` vs. `qkv_proj`/`in_proj_qkvz`
calls per layer) before trusting it for the bulk per-layer duration
analysis below.

## Results

Interactive per-layer duration breakdowns:
[`chart-per-layer.html`](./chart-per-layer.html) (150-prompt capture) and
[`chart-per-layer-500prompts.html`](./chart-per-layer-500prompts.html) (500
prompts, more statistical power). Duration-by-launch-grid distributions
from the GPU-metrics-enabled captures:
[`chart-durations-by-grid.html`](./chart-durations-by-grid.html).

**The regression is one specific weight shape, not the whole model.** Of
the 5 W8A8 weight matrices, only `N=17408` (`gate_up_proj`) regresses under
tuning (−21.6% at 150 prompts, −27.4% at 500 prompts); the other 4 all get
faster (+3.6% to +23.4%). Per-layer proportions are nearly identical between
configs (e.g. N=5120: 47.46% of default's launches vs. 47.52% of tuned's) —
direct confirmation both captures are doing comparable logical work.

**Within `N=17408`, the regression is M-dependent, not one narrow
cluster** (500-prompt capture, rebinned onto a common M-axis):

| M range | default n | default mean | tuned n | tuned mean | delta |
|---|---:|---:|---:|---:|---:|
| 0–127 | 122,240 | 172.5µs | 121,600 | 170.7µs | +1.1% |
| 256–383 | 1,152 | 267.3µs | 1,280 | 231.1µs | +13.5% |
| 640–767 | 128 | 448.2µs | 896 | 780.1µs | −74.1% |
| 1280–1407 | 384 | 780.7µs | 128 | 1413.4µs | −81.0% |
| 1792–1919 | 4,224 | 1091.7µs | 4,992 | 1926.5µs | −76.5% |
| 1920–2047 | 4,864 | 1148.2µs | 4,352 | 2054.7µs | −78.9% |

Tuned wins or ties below `M~256`, then loses — consistently, at roughly the
same ~70–81% penalty — from `M~640` all the way up to `M~2048` (the largest
`M` this workload reaches at c=64).

**At c=1, the same shape shows zero regression anywhere in its M range** —
because c=1's workload never produces the `M` values where the problem
lives (decode is `M=1` for both configs, confirmed by matching launch
counts pairwise across every `gridX`; prefill tops out around the 256-token
prompt length):

| regime | default n | default mean | tuned n | tuned mean | delta |
|---|---:|---:|---:|---:|---:|
| decode (M~1) | 325,120 | 168.9µs | 325,120 | 135.5µs | +19.8% |
| prefill (M~256–320) | 2,560 | 236.7µs | 2,560 | 222.4µs | +6.0% |

## Reproduce

Given a kernel-trace SQLite export, group launches by `(gridX, blockX)`,
invert against each config's real formula (default: closed-form; tuned:
per-anchor lookup from the shape's tuned JSON, both `BLOCK_SIZE_M` and
`BLOCK_SIZE_N`) to recover `(N, K, M)`, then disambiguate `N=5120`'s two
sub-shapes by real launch-duration clustering rather than grid alone.
