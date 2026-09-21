# Real GPU hardware-counter profiling (Nsight Operator)

Captured via the NVIDIA Nsight Operator (`components/nsight-operator` in
ocp-config), which injects an `nsys profile` wrapper into the predictor's
`kserve-container` process and streams the result to its own MinIO-backed
cloud storage. `dcgmi profile --pause`/`--resume` on the node's DCGM pod is
required around every capture -- DCGM holds the GPU's exclusive
hardware-counter resource otherwise and `nsys`/`ncu` fail outright (see git
history for the full investigation of why real hardware-counter profiling was
blocked, and profiling-scc.yaml for the container-level requirements).

## Why this exists

The earlier `contention-results.html` artifact's "predicted vs actual" TPOT
comparison was built from *isolated microbenchmark* GEMM times weighted by
real batch-mix -- an estimate, not a direct measurement. This directory holds
actual hardware-counter kernel-time data captured from the real predictor
pod under real client load, so the GEMM kernel's real share of GPU time (and,
once pass 2 is captured, the tuned-vs-default delta) can be cited directly
instead of inferred.

## Raw data location

Each run's raw `.nsys-rep` file sits next to its `kernel_summary.csv` as
`capture.nsys-rep` -- present on disk for local inspection (e.g. opening in
the Nsight Systems GUI) but **git-ignored** (`*.nsys-rep` in the repo root's
`.gitignore`), not committed. They also already live durably in the Nsight
Operator's own cloud storage in-cluster regardless, keyed by session/
collection ID below, so a missing local copy is never fatal -- re-fetch any
of them with:

`pass2-tuned-concurrency1`'s raw file (272MB, larger than the others --
longer capture window at low request rate) failed to download via
`nsight_operator.py download` repeatedly (`ContentLengthError`, a
proxy/timeout issue on the larger HTTPS transfer -- not a corrupt capture,
`analysis` extraction from the same collection succeeded fine throughout).
Worked around by pulling it directly off the predictor pod's local
filesystem instead (nsys writes it to `/tmp/s3_cache/` on the container
before the coordinator uploads it, and it's still there after upload):
`oc cp -n enterprise-ai -c kserve-container <pod>:/tmp/s3_cache/<file>.nsys-rep capture.nsys-rep`.
Byte-for-byte match to the expected size (272,912,304 bytes).

```bash
source /tmp/nsight-cli-venv/bin/activate   # python3.12 venv w/ nsight_operator.py deps
cd /private/tmp/nsight-operator-resources_v26.3.1
python3 nsight_operator.py download --session <session id> --collection <collection id> --output-dir <dir>
```

(CLI resources: `ngc registry resource download-version
"nvidia/devtools/nsight-operator-resources:26.3.1"`, works fully
unauthenticated. Requires Python 3.12+.)

What's kept here instead: the session/collection IDs, the matching
`vllm bench serve` result, and a `kernel_summary.csv` per run -- the
per-kernel-name breakdown (`nsys` recipe `cuda_gpu_kern_sum`), sorted by
total GPU time, with `Count`, `mean_us` (mean kernel duration),
`sum_ms` (total time across all launches), and `pct_of_kernel_time`
(share of all GPU kernel time captured in that window).

## Runs

Session ID for all runs below: `d3123b6e-b352-4b14-9353-f19b27a027ee`
(node `gpu-node-a`). TP=2 -- every "total kernel time" /
"GEMM time" figure below is **summed across both GPU devices** (confirmed
via `rank_stats_by_device.parquet`: both devices contribute near-identical
per-kernel sums, as expected for symmetric TP=2 work), so it can exceed wall
clock by up to 2x. "GPU-busy ratio" divides by `2 * wall clock` to normalize
back to a genuine average per-GPU utilization.

| Run | Collection ID | Config | `vllm bench serve` args | Mean TPOT | Total kernel time | Wall clock | GPU-busy ratio (per-GPU avg) | `_w8a8_triton_block_scaled_mm` share |
|---|---|---|---|---|---|---|---|---|
| `pass1-default-concurrency64` | `26ff64c4-66fe-4eb4-8d21-d766ade1e27e` | default | `--max-concurrency 64 --num-prompts 500 --request-rate inf` | 72.57ms | 70.74s | 83.64s | 42.3% | 35.4% (25.04s) |
| `pass1-default-concurrency1` | `e14d7991-0adc-4081-8156-c1978de2ce0b` | default | `--max-concurrency 1 --num-prompts 20 --request-rate 1` | 31.74ms | 13.41s | 84.60s | 7.9% | 10.0% (1.34s) |
| `pass2-tuned-concurrency64` | `0f669f1f-9fb1-461c-a2aa-954fa3e9f79f` | tuned | `--max-concurrency 64 --num-prompts 500 --request-rate inf` | 73.93ms | 168.36s | 87.42s | 96.3% | 45.9% (77.32s) |
| `pass2-tuned-concurrency1` | `655dd8fc-7bcd-4873-a360-551c87b262ed` | tuned | `--max-concurrency 1 --num-prompts 20 --request-rate 1` | 24.85ms | 127.62s | 67.39s | 94.7% | 75.3% (96.05s) |

### pass2-tuned-concurrency64 vs pass1-default-concurrency64 -- unexpected result

Real e2e TPOT is essentially flat, slightly *worse* on the tuned config
(72.57ms -> 73.93ms). The kernel-level data explains why, and it's not what
the tuning was supposed to produce:

| `_w8a8_triton_block_scaled_mm` | default | tuned | ratio |
|---|---|---|---|
| Call count | 45,568 | 540,160 | **11.85x more calls** |
| Mean duration/call | 549.6us | 143.1us | **3.84x faster per call** |
| Total GEMM GPU-time | 25.04s | 77.32s | **3.09x more total time** |

Per-call, the tuned config's kernel genuinely is much faster (549.6us ->
143.1us) -- the tuning itself worked as intended at the individual-kernel
level. But something about the tuned config causes ~11.85x more GEMM
launches for the same request workload (same `--num-prompts 500
--max-concurrency 64`), which more than cancels out the per-call speedup:
net GEMM GPU-time is **3.09x higher**, not lower. That tracks with per-GPU
utilization jumping from 42.3% to 96.3% for essentially the same real-world
output throughput.

**Update after collecting a 2-replicate matrix (2 pod instances x 2 configs x
2 concurrency levels) -- see "Instance-to-instance variance" section below.**
The launch-count multiplication turned out to be a tracer artifact specific
to the very first capture of this whole investigation, not a real
tuned-vs-default or config-driven effect. Read that section first; the
paragraphs below are preserved as-written for the historical record but their
"32x tuned-specific" framing is superseded.

**Investigated, not yet resolved [superseded, see above].** Checked the actual tuned JSON configs
(`performance-tests/qwen3-8-27b-fp8-dense-v0.27.1-configs/*.json`) for a
`SPLIT_K` field that would directly explain a launch-count multiplier --
**not present**. The configs only set `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`,
`num_warps`, `num_stages`.

Cross-checked against the real Python-level dispatch count from the
`fp8-utils-instrumented` overlay's `/tmp/m_histogram.json` (counts real
`w8a8_triton_block_scaled_mm(N,K,M)` calls at the Python layer, cumulative
since the pod last restarted) -- saved here as
[`pass2-tuned-concurrency1/m_histogram.json`](./pass2-tuned-concurrency1/m_histogram.json)
(pulled off the tuned pod right after the `pass2-tuned-concurrency1`
capture, before that pod was replaced -- it's cumulative across *both*
`pass2-tuned-concurrency64` and `pass2-tuned-concurrency1` on that same
instance-A pod, not exclusive to the one capture it's filed next to; same
caveat applies to the equivalent `pass1-default-concurrency1/m_histogram.json`
and `pass1-default-concurrency1-v2/m_histogram.json`, each cumulative across
its own instance's two runs): **57,856** total calls across both
tuned runs combined (pod wasn't restarted between `pass2-tuned-concurrency64`
and `pass2-tuned-concurrency1`), versus **1,850,880** combined nsys kernel
launches (540,160 + 1,310,720) -- a clean **32x** ratio. That confirms each
logical Python-level GEMM call maps to ~32 separate GPU kernel launches
under the tuned config, on average across both runs -- but **not** why, and
not whether it's specific to the tuned config at all:

CUDA graphs are a real, live confound. vLLM CUDA-graph-captures repeated
decode steps (this is *why* `--cuda-graph-trace=node` was needed at all --
see above), and a graph *replay* executes pre-recorded GPU commands without
re-entering the Python function the histogram instruments. So the 32x gap
could equally be "CUDA-graph replay generally undercounts Python-side
instrumentation, for any config" rather than "the tuned config's kernel
genuinely issues more GPU launches per logical call." Distinguishing these
needs the same python-count-vs-nsys-count comparison run on the **default**
config for an apples-to-apples check -- not currently possible without
redeploying default again, since that pod (and its histogram, which isn't
persisted anywhere else) is gone, replaced by the tuned deployment.

**Bottom line so far:** the real, solidly-measured e2e numbers are the
`vllm bench serve` results themselves (concurrency=64: 72.57ms -> 73.93ms,
flat-to-worse; concurrency=1: 31.74ms -> 24.85ms, a real 21.7% improvement)
-- those don't depend on any of the above and can be cited directly. The
kernel-level GEMM-launch-count story is real and reproducible but its root
mechanism is not yet nailed down; don't cite "split-K" or any other specific
mechanism as the cause without redoing the python-vs-nsys check on a
default-config pod to rule out the CUDA-graph-replay explanation.

## Instance-to-instance variance (2-replicate matrix)

Redeployed and re-measured to get a second pod instance ("instance B") for
each of the 4 cells (default/tuned x c=64/c=1), on top of the original
("instance A") measurements above. Session `d3123b6e-b352-4b14-9353-f19b27a027ee`
throughout; all instances on node `gpu-node-a`.

`_w8a8_triton_block_scaled_mm` nsys launch counts, full matrix:

| | c=64 | c=1 |
|---|---|---|
| default, instance A | 45,568 | 10,240 |
| default, instance B | 536,576 (x2 identical repeats on the same pod) | 1,310,720 |
| tuned, instance A | 540,160 | 1,310,720 |
| tuned, instance B | 537,088 | 1,310,720 |

Directories: `pass1-default-concurrency64` / `pass1-default-concurrency1` =
instance A; `pass1-default-concurrency64-v2` / `pass1-default-concurrency1-v2`
/ `pass2-tuned-concurrency64-v2` / `pass2-tuned-concurrency1-v2` = instance B.
`pass2-tuned-concurrency64` / `pass2-tuned-concurrency1` = tuned instance A.
Each directory has `bench-result.txt` (the real `vllm bench serve` output for
that specific run) alongside `capture.nsys-rep` and `kernel_summary.csv`.

**Pattern:** three of the four instances (default-B, tuned-A, tuned-B)
converge tightly regardless of config -- within 0.7% of each other at c=64,
and *exactly* 1,310,720 at c=1, all three. Only **default instance A** -- the
very first pod deployed in this whole nsys investigation -- differs, by
~11.8x at c=64 and ~128x at c=1. Repeating the same pod twice (default
instance B, c=64) gave an *exact* match (536,576 = 536,576), confirming this
isn't run-to-run noise within a pod -- whatever's responsible is fixed once
per pod instance.

This reframed the whole question: not "why does tuned vs. default differ,"
and not "how variable are pod instances in general" (they're not, mostly) --
but "what was different about that one specific first capture."

### Root-caused: a tracer artifact, not a real execution difference

Ran `cuda_api_sum` (real CUDA API call counts, independent of any GPU-side
kernel-launch decomposition) on default instance A vs. tuned instance A:

| CUDA API call | default instance A (anomalous) | tuned instance A (normal) |
|---|---|---|
| `cudaGraphLaunch` | 2,314 | 2,190 |
| `cudaLaunchKernel` | 174,756 | 173,076 |
| `cuLaunchKernel` | 84,118 | 83,648 |
| `cuLaunchKernelEx` | 154,246 | 151,754 |

These are within 1-6% of each other -- ordinary run-to-run noise, nothing
like the 11.8x gap in `_w8a8_triton_block_scaled_mm`'s reported launch count.
**The real work done -- including the number of CUDA graph replays -- was
essentially identical.** The 11.8x/128x gap exists entirely in how
completely nsys's `--cuda-graph-trace=node` decomposed each graph replay
into visible per-kernel GPU events: ~2,300 graph launches decomposed into
~20 recorded GEMM events per replay for instance A, vs. ~230 per replay for
every other instance. Same graphs, same real content, different tracer
decomposition depth.

Checked whether this was a mid-capture warm-up effect (`cuda_gpu_trace` on
instance A, GEMM event count bucketed across the capture's 10 equal time
windows): roughly uniform throughout (2,280-6,144 events/bucket, no
ramp-up). So it wasn't CUPTI's node-tracing gradually engaging over the
course of the capture -- whatever determined the shallower decomposition was
fixed for that entire session from the start.

**Conclusion:** the GEMM-launch-count/contention-mechanism narrative from
earlier in this document doesn't hold -- it was chasing a tracer artifact
specific to the first capture, not a tuned-vs-default effect. Ruled out as
causes: CUDA graph capture-set variability (proven deterministic via startup
logs), a node-persistent JIT/driver cache (checked directly -- no such mount
exists), and a gradual within-capture warm-up (checked directly -- flat
throughout). What's not resolved: *why* that one specific capture session
got shallower decomposition. Would need to diff actual CUPTI/nsys internals
or compiled kernel artifacts to go further, with no guarantee of a clean
answer -- not pursued further given the cost/benefit.

**What stands, unaffected by any of this:** every real `vllm bench serve`
result, cited above and in each run's own `bench-result.txt` -- those come
from the client, not from GPU-side kernel-launch counting, and were
consistent and reproducible across every instance measured (default: 72.57 /
73.12 / 73.27ms at c=64 across 3 instances/runs, 31.74 / 31.90ms at c=1
across 2; tuned: 73.93 / 73.82ms at c=64 across 2, 24.85 / 24.78ms at c=1
across 2).

Both captures use `nsightToolArgs: --python-sampling=true
--trace-fork-before-exec=true --cuda-graph-trace=node` (set via
`ocp-config/argocd-applications/overlays/<cluster>/nsight-operator.yaml`'s
`valuesObject`) -- without `--cuda-graph-trace=node`, nsys's default
CUDA-graph handling collapses each vLLM decode step (CUDA-graph-captured by
default) into one opaque placeholder-named launch, hiding every real
GEMM/attention kernel underneath. `node` mode decomposes it; a residual
placeholder-named bucket remains in both runs regardless (see per-run notes),
but does not affect the `_w8a8_triton_block_scaled_mm` figures, which are
clearly named and resolved in both.

### pass1-default-concurrency64

Second-largest single GPU-time consumer behind NCCL's `AllReduce` (expected
-- TP=2 forces a cross-GPU reduce every layer). Residual unresolved-name
bucket (`Kernel2`, likely still-uncollapsed CUDA graph nodes): 6.3% (4.48s).
Full breakdown: `kernel_summary.csv`.

### pass1-default-concurrency1

GEMM's share of kernel time (10.0%) is roughly a third of what it is at
concurrency=64 (35.4%) -- batched matmuls amortize launch/kernel overhead far
better at scale, which is exactly the regime where the autotuned config
should matter most. Residual unresolved-name bucket at this concurrency is
larger and differently-shaped: a generically-named `kernel` bucket at 67.3%
(9.02s), likely prefill-path kernels launched via the CUDA driver API whose
names didn't resolve -- distinct from the `Kernel2` CUDA-graph issue seen at
concurrency=64. Doesn't affect the `_w8a8_triton_block_scaled_mm` figure.
Full breakdown: `kernel_summary.csv`.

## Why c=64 regresses: it's M, not concurrency

GPU-metrics-enabled captures (`*-gpumetrics*` directories; `--gpu-metrics-devices=all
--gpu-metrics-frequency=1000`, added to the same `nsightToolArgs`) plus a
reverse-mapping of each real launch's (N,K) shape from its grid/block
dimensions (`gemm-per-layer-durations.html` /
`gemm-per-layer-durations-500prompts.html`) let this be nailed down precisely.

**The regression is one specific weight shape, not the whole model.** Of the
5 W8A8 weight matrices, only N=17408 (K=5120) regresses under tuning
(-21.6% at 150 prompts, -27.4% at 500 prompts); the other 4 all get faster
(+3.6% to +23.4%).

**Within N=17408, the regression is M-dependent, not one narrow cluster.**
Rebinning every real launch onto a common M-axis (500-prompt captures,
`pass1-default-concurrency64-v2` / `pass2-tuned-concurrency64-v2`):

| M range | default n | default mean | tuned n | tuned mean | delta |
|---|---|---|---|---|---|
| 0-127 | 122,240 | 172.5us | 121,600 | 170.7us | +1.1% |
| 256-383 | 1,152 | 267.3us | 1,280 | 231.1us | +13.5% |
| 640-767 | 128 | 448.2us | 896 | 780.1us | -74.1% |
| 1280-1407 | 384 | 780.7us | 128 | 1413.4us | -81.0% |
| 1792-1919 | 4,224 | 1091.7us | 4,992 | 1926.5us | -76.5% |
| 1920-2047 | 4,864 | 1148.2us | 4,352 | 2054.7us | -78.9% |

Tuned wins or ties below M~256, then loses -- consistently, at roughly the
same ~70-81% penalty -- from M~640 all the way up to M~2048 (the largest M
this workload reaches at c=64). The specific launch signature characterized
in detail below (gridX=2176, M in (1920,2048]) is just the largest,
most-populated bin of this broader regression, not an isolated anomaly.

**At c=1, the same shape shows zero regression anywhere in its M range** --
because c=1's workload never produces the M values where the problem lives.
Decode is M=1 for both configs, confirmed by matching default's and tuned's
launch counts pairwise across every gridX pair, not just the one used here
(650,240 = 650,240; 325,120 = 325,120; 243,840 = 243,840; 81,280 = 81,280).
Default's fixed, table-free formula (`BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`
always) makes its side unambiguous: gridX=136 for this shape only arises from
`ceil(M/64)=1`, i.e. real M in (0,64] -- for pure single-request decode,
essentially all M=1. Tuned's matching gridX=272 looks superficially
different, but resolves to the same M=1: unlike default, tuned's
`BLOCK_SIZE_N` is *also* a per-M-anchor lookup value (not a fixed 128), and
its own M=1 anchor uses `BLOCK_SIZE_N=64`, giving `n_blocks=ceil(17408/64)=272`
and `ceil(1/16)x272=272` -- the same gridX, via a different but equally
real derivation. (An earlier version of this table mislabeled tuned's bucket
as "M=17-256", from assuming `BLOCK_SIZE_N=128` unconditionally; the
duration/count figures below were always correct, only that label was wrong.)
Prefill tops out around the 256-token prompt length (2,560 = 2,560 launches
both sides):

| regime | default n | default mean | tuned n | tuned mean | delta |
|---|---|---|---|---|---|
| decode (M~1) | 325,120 | 168.9us | 325,120 | 135.5us | +19.8% |
| prefill (M~256-320) | 2,560 | 236.7us | 2,560 | 222.4us | +6.0% |

**Conclusion: concurrency=64 has no direct causal role.** It's purely a
mediator -- continuous batching under load pushes M into the 640-2048 range,
and it's specifically *real-serving* GEMM calls at that M range, for this
one weight shape, that degrade. This is not a "large M is slow" story:
re-measured live on the same hardware, the *identical* isolated kernel
benchmark at the *identical* M=2048/shape/config still shows tuned winning
(+9.4% to +9.9% vs default, matching the original tuning claim) -- isolated
duration ~120.4us vs. this same launch's real-serving duration of ~2057.3us,
a ~17.1x inflation for tuned vs. only ~8.85x for default's own equivalent
large-M launch at the same shape (~1179.5us real vs. ~133.3us isolated).
Both configs suffer a large real-vs-isolated inflation at this shape/M range
-- it isn't unique to tuned in kind, only in degree (roughly 2x worse).

Ruled out as the mechanism (real trace/telemetry evidence, not inference):
concurrent kernel contention (confirmed single-stream serialization, no true
concurrent execution on one device), NCCL overlap (`Overlap` metric = 0.0
throughout), pre-launch scheduling/queueing gaps (negligible), and clock
throttling as a *cause* (real but small for tuned, -5.4% vs. a same-trace
fast-launch baseline; the before/during/after check shows clock *rising*
through tuned's slow launches, not dipping, i.e. these launches ride out an
already-elevated-load period rather than causing their own throttle; and
duration doesn't correlate with clock at all within one fixed launch config,
Pearson +0.18 in the wrong direction). Default's own large-M launches show a
much bigger, genuinely local clock dip (-19.8%) yet default is *less*
inflated overall -- so clock throttling doesn't distinguish the two configs
either.

Leading untested hypothesis: instruction/constant-cache locality. The
isolated benchmark repeats the identical kernel back-to-back (warm caches
every iteration); real serving sandwiches it between many different kernel
types every single occurrence (cold every time). Tuned's config at this
shape is structurally heavier (`BLOCK_SIZE_M=128, num_warps=8,
num_stages=3`, 65536B dynamic shared memory) than default's
(`BLOCK_SIZE_M=64, num_warps=4, num_stages=2`, 25088B) -- plausibly more
sensitive to a cold start each occurrence, consistent with winning big warm
and losing big cold. Not yet directly tested: would need `ncu` stall-reason
data (blocked by both replay-mode self-contamination for this specific
question, and unconfirmed tool availability/permissions in this
environment), or an external interleaved-kernel reproduction harness
(planned, not yet built) that recreates the "cold, context-switched"
condition without needing vLLM at all.

## Independent confirmation: `-gpumetrics-1khz-short` matched pair

A second, previously-unanalyzed pair --
`pass1-default-concurrency64-gpumetrics-1khz-short` /
`pass2-tuned-concurrency64-gpumetrics-1khz-short` (150 prompts each,
`--max-concurrency 64`, 1kHz GPU metrics, queried directly from the
already-exported `kernel_trace.sqlite`/`gpu_metrics.sqlite` since `nsys` CLI
isn't installed in this environment) -- reproduces the whole finding above
cleanly, with two things this pair specifically rules out: it has **matched
GEMM launch counts** (201,216 = 201,216, no repeat of the
tracer-artifact launch-count mismatch from instance A above), and it lets
the regression be tied directly to the client-visible latency percentiles
rather than only to aggregate kernel time.

**Total GEMM time is a wash; the distribution is what differs:**

| | default | tuned | delta |
|---|---:|---:|---:|
| `_w8a8_triton_block_scaled_mm` calls | 201,216 | 201,216 | -- |
| Total GEMM GPU-time (both GPUs) | 26,231.7ms | 26,262.6ms | +0.12% |
| Median call | 85.95us | 77.38us | -10.0% |
| p99 call | 1,168.4us | 2,011.6us | +72.2% |
| p99.9 / max call | 1,189.7 / 1,201.7us | 2,093.3 / 2,131.9us | +76% / +77% |

Tuned is genuinely faster at the median, as intended -- but has a much
fatter tail, and the two nearly cancel in the aggregate sum. That's why real
`vllm bench serve` TPOT doesn't improve here either (68.32ms -> 69.61ms).

**It's the same shape, same M-range, and it's config-independent which
steps hit it.** `gate_up_proj` (N=17408) fires once per layer per GPU per
step (64 layers x 2 GPUs = 128 combined launches/step, which cleanly
back-computes real step counts from launch counts -- almost every bucket
below is an exact multiple of 128):

| M-range (per step) | default: steps / mean | tuned: steps / mean |
|---|---:|---:|
| <=64/128 (decode-dominated) | ~365 steps / 170us | ~365 steps / 171us |
| mid (256-1408) | ~7 steps / 280-810us | ~5 steps / 238-1415us |
| **~1920-2048 (large batch)** | **21 steps / 1122-1183us** | **20 steps / 1926-2057us** |
| Total steps (this shape) | ~393 | ~393 |

Both configs hit the large-batch regime for almost exactly the same ~20 of
~393 steps (~5%) -- same real workload, same scheduler, config-independent.
What differs is only how expensive those ~20 steps are: default ~1.15ms/call
there, tuned ~2.0ms/call -- the same ~74-79% penalty already established
above, now shown to be concentrated in a specific, small, config-independent
subset of engine steps rather than spread across every decode step.

**This is why the client-side mean/p99 gap has the shape it does.** Mean
TPOT barely moves (+1.9%, 68.32ms -> 69.61ms) because ~95% of output tokens
are generated during ordinary small-M steps where the two configs are tied
or tuned is slightly ahead. P99 TPOT (+6.9%, 80.19ms -> 85.74ms) and P99 ITL
(+11.0%, 418.62ms -> 464.48ms) move much more, because P99 is defined by the
tail, and the tail is exactly where the ~20 slow steps (and everything
generated during them) land.

**Clock check, repeated on this pair, same result as above (backwards
correlation, still not causal):**

| | fast-baseline mean clock | slow-bucket mean clock | dip |
|---|---:|---:|---:|
| tuned (large-batch bucket) | 2515.5 MHz | 2498.8 MHz | -0.66% |
| default (large-batch bucket) | 2515.9 MHz | 2204.0 MHz | -12.4% |

Default's large-M launches take the much bigger clock hit here too, yet
default is still the faster config at this shape -- consistent with
`07-gpu-metrics-clock-throttling`'s conclusion that clock throttling is a
correlate of recent load, not a cause of the regression, and does not
distinguish the two configs.

### CUDA graph capture: confirmed irrelevant to this shape, not just unlikely

Checked directly via `graphId`/`graphNodeId` on `CUPTI_ACTIVITY_KIND_KERNEL`
rather than inferred from `max_num_seqs`: **92.9% of all GEMM launches are
graph-replayed in both configs, an identical count (186,880/201,216 both
sides)** -- vLLM only captures graphs for a fixed, small set of decode batch
sizes, and this workload's `max_num_seqs=128` never reaches the M~2000
large-batch regime. `graphId IS NULL` for **100%** of launches in the slow
bucket, both configs (`gridX=4352` default, `gridX=2176` tuned). Neither
config's regressed launch is ever graph-captured -- CUDA-graph replay is
ruled out as a distinguishing mechanism here directly, not just
argued unlikely.

### Register spilling: checked, ruled out

`localMemoryPerThread = 0` for every GEMM launch in both captures, every
bucket. No spilling anywhere; `registersPerThread` staying under the
allocation limit doesn't need qualification here.

### SM occupancy at the slow bucket specifically: a tie, not a tuned advantage

The project's established occupancy story (tuned reaching 2-5x default's
occupancy) is about the *dominant, small-M* decode bucket, where tuned's
tiny `BLOCK_SIZE_M=16` tile packs many blocks per SM. Recomputed for the
*slow*, large-M bucket specifically (48 max warps/SM on Ada/L40S):

| | regs/thread | threads/block | blocks/SM (reg-limited) | warps/SM | occupancy |
|---|---:|---:|---:|---:|---:|
| default (gridX=4352) | 243 | 128 | 2 | 8 | 16.7% |
| tuned (gridX=2176) | 250 | 256 | 1 | 8 | 16.7% |

Identical. At the large-M anchor, tuned's config (`BLOCK_SIZE_M=128,
num_warps=8`) needs 64,000 of the SM's 65,536 registers for just one
resident block -- leaving no room for a second, landing at the same total
warp count as default's two smaller blocks. Occupancy does not distinguish
the two configs at this specific bucket.

### What does distinguish them: tuned is memory-bound in real serving, default is compute-bound in both

Averaging `GPU_METRICS` (Tensor Active, DRAM Read/Write bandwidth
throughput) over each launch's own execution window, same two buckets:

| | Tensor Active | DRAM Read | DRAM Write |
|---|---:|---:|---:|
| default (gridX=4352) | **45.5%** | 15.8% | 10.1% |
| tuned (gridX=2176) | **28.3%** | **72.8%** | 5.5% |

At identical shape, identical M, identical occupancy: default's launch
spends most of its time doing tensor-core math (45.5% active, DRAM barely
touched). Tuned's launch sits at ~73% of peak DRAM read bandwidth -- the
highest-utilized resource by far in that launch -- while tensor cores sit
comparatively idle (28.3%). Default is compute-bound at this shape; tuned
is memory-bandwidth-bound.

This doesn't reinstate "concurrent contention" as a mechanism -- single-stream
serialization on one device is already confirmed above, so no other kernel
is literally executing at the same instant, stealing bandwidth mid-launch.
What it supports is a sharper, directly-measured version of the leading
instruction/cache-locality hypothesis: a compute-bound kernel barely touches
DRAM, so it's largely insensitive to whatever DRAM/L2 row/line state the
*previous*, different kernel left behind. A memory-bandwidth-bound kernel is
maximally exposed to exactly that.

**Closed the loop with a matching isolated capture** (same shape/M,
`--gpu-metrics` enabled on an idle-GPU one-off job -- see
[`09-isolated-vs-real-flops-inflation`](../../09-isolated-vs-real-flops-inflation/)'s
"Reproducing the GPU-metrics-enabled isolated capture" for how). The
DRAM-bound signature is **absent** in isolation, for both configs:

| | Tensor Active (isolated) | Tensor Active (real) | DRAM Read (isolated) | DRAM Read (real) |
|---|---:|---:|---:|---:|
| default | 55.9% | 45.5% | 9.2% | 15.8% |
| tuned | 58.2% | 28.3% | 14.9% | **72.8%** |

In isolation both configs are clearly compute-bound and look similar to each
other (Tensor Active ~56-58%, DRAM Read ~9-15%) -- tuned's heavier tile
isn't structurally more memory-hungry than default's on its own. Only in
real serving does tuned's DRAM Read jump ~5x while its Tensor Active
roughly halves; default barely moves either metric between isolated and
real. This is a genuine compute-bound-to-memory-bound transition specific
to tuned and specific to real serving, not a property of the tuned config
in isolation -- direct evidence for (not yet full proof of) the
cache/row-buffer-locality mechanism: tuned's config is fine when the memory
subsystem answers instantly (isolated, nothing else has ever touched it)
and exposed when it doesn't (real serving, sandwiched between structurally
different kernels every occurrence). `ncu` stall-reason data would still be
the direct mechanistic proof; not obtained here.
