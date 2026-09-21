# 13 — KV-cache-traffic reproduction

**Status:** built and run; clean negative result, confirmed directly via
matched DRAM-Read/Tensor-Active metrics (after fixing a config-label swap in
the classification query -- see "Correction found while writing this up"
below). Realistic-volume KV-cache traffic adds a **large, mostly
config-independent memory-bandwidth tax on top of whichever baseline was
already there** -- correctly directioned (tuned ends up a few points higher
than default in every with-KV row) but the gap it adds (+3-5pp) is nowhere
near real serving's ~57pp asymmetry (72.8% vs. 15.8%). Rules out "sheer
KV-cache traffic volume" as sufficient, on its own, to explain the remaining
gap between `12`'s partial reproduction and real serving's full effect.

## What

Test whether adding realistic-volume, continuously-read/written PagedAttention
KV-cache traffic -- something neither `12` (cold weight cycling only) nor
`11` (kernel-type diversity only) modeled -- is the missing ingredient that
pushes tuned's DRAM Read the rest of the way from `12`'s ~22% toward real
serving's 72.8%.

## Why

`12`'s cold-weight-cycling result confirmed the mechanism direction (tuned's
DRAM Read roughly doubles under cold-weight cycling, default stays flat) but
left a large gap unexplained: cold-cycling's 22% is still well short of real
serving's 72.8%. `11` already ruled out kernel-type diversity as the missing
piece (interleaving real conv/attention/recurrent kernels made both configs
*faster*, not slower). The one thing about real serving neither prior test
touched: at concurrency=64, vLLM continuously reads and writes PagedAttention
KV-cache blocks for up to 64 concurrent sequences -- a large, sustained
memory stream through L2/DRAM that has nothing to do with which *weight*
tensor is being read. This experiment adds that specific traffic directly to
see whether it -- not other GEMM weights, not kernel diversity -- accounts
for the rest of the gap.

## How

`kv_cache_traffic_repro.py` -- same single-process, single-CUDA-stream,
one-GEMM-shape (`gate_up_proj`, N=17408, K=5120, M=2048) design as `12` v2,
extended with a KV-traffic on/off toggle crossed with `12`'s existing
same-B/cold-B x default/tuned axes -- 8 conditions total (2×2×2), round-robin
counterbalanced in 250-iteration blocks × 16 repeats exactly like `12` v2, to
control for the same clock-drift confound found and fixed there.

- A 20GB `bfloat16` buffer is allocated up front (matching the real
  deployment's measured ~23-25GiB KV cache).
- When KV traffic is "on," every GEMM call is immediately followed by a real
  in-place read+write op (`.add_(1.0)`) on a ~320MB slice of that buffer
  (`20GB / 64`, i.e. one layer's worth), at a sequentially-advancing offset
  so one full 64-call round touches the entire buffer once.
- **Deliberately documented simplifying assumption** (matching `11`'s own
  practice of flagging scope reductions): each `gate_up_proj` call is treated
  as "one layer," and the per-layer touch is sized uniformly. The real
  model's hybrid architecture (`full_attention_interval=4`) means only 16 of
  64 layers do real quadratic-attention KV reads at this scale; the other 48
  are linear/mamba attention with much smaller fixed-size state. This is
  deliberately a maximally-aggressive, not a precisely-faithful, KV-traffic
  model -- if it doesn't move the needle, that's evidence against sheer
  KV-cache volume as the mechanism, not just against this specific
  approximation of it.
- Run inside the same nsys-injected, DCGM-paused, SCC-elevated pod pipeline
  built for `09`'s isolated-GPU-metrics capture (see that experiment's
  "Reproducing the GPU-metrics-enabled isolated capture" section).

## Results

Latency (n=4000/condition, all 8 conditions):

| | no-KV | with-KV | shift |
|---|---:|---:|---:|
| default-same | 1276.8µs | 2151.8µs | +68.5% |
| default-cold | 1280.1µs | 2155.4µs | +68.4% |
| tuned-same | 1145.5µs | 2241.6µs | +95.7% |
| tuned-cold | 1204.2µs | 2246.9µs | +86.6% |

Trend check (first-half vs. second-half block means, all 8 conditions):
drift ranges +1.8% to +5.4%, similar order of magnitude across all
conditions -- consistent with `12`'s modest uniform thermal drift, not a
condition-specific confound.

**Direct DRAM-metric confirmation** (matched to the exact same launches,
via kernel-shape classification: `gridX=4352`→default, `gridX=2176`→tuned --
derived arithmetically from each config's own `BLOCK_SIZE_M` [default=64,
tuned=128 at M=2048: `ceil(2048/64)×ceil(17408/128)=4352`,
`ceil(2048/128)×ceil(17408/128)=2176`] and cross-checked against `05`'s
independently-established register-count/gridX mapping [default=243regs@4352,
tuned=250regs@2176] -- see the correction note below on how this was caught;
plus detecting each GEMM's immediately-following `vectorized_elementwise_kernel`
launch to classify KV-on/off; block boundaries recovered directly from the
data and cross-checked against the expected same→cold alternation, not
assumed from iteration counts):

| | Tensor Active (noKV→KV) | DRAM Read (noKV→KV) | DRAM Write (noKV→KV) |
|---|---|---|---|
| default-same | 56.2%→31.1% | 9.4%→26.5% | 7.4%→23.4% |
| default-cold | 55.7%→31.1% | 9.3%→26.6% | 7.4%→23.4% |
| tuned-same | 58.5%→33.4% | 12.2%→29.2% | 8.2%→21.2% |
| tuned-cold | 55.2%→33.4% | 22.9%→31.4% | 7.9%→21.2% |

For reference: real serving DRAM Read is default ~15.8%, tuned ~72.8%
(a 4.6x ratio, tuned far higher). `12`'s cold-weight-cycling-only result:
default DRAM Read ~9.2% flat, tuned 12%→22% (same→cold) -- this experiment's
noKV column reproduces that almost exactly (default 9.4%→9.3% flat,
tuned 12.2%→22.9%), an independent replication of `12`, not a new result on
its own.

**Adding KV traffic raises DRAM Read for both configs by a similar absolute
amount (+17-22pp), and does not reproduce real serving's asymmetry.** Tuned
does end up somewhat higher than default in every with-KV row (26.5-26.6%
default vs. 29.2-31.4% tuned) -- correctly directioned, unlike what an
earlier, mislabeled pass of this analysis showed -- but the gap (roughly
+3-5pp) is nowhere near real serving's ~57pp gap (72.8% vs. 15.8%). Tensor
Active converges to almost exactly the same ~31-33% for all four with-KV
rows regardless of config or weight-state, and DRAM Read/Write both converge
to a similar range across all four. This is a clean negative: realistic-volume
KV-cache traffic, modeled this way, adds a large, mostly config-independent
memory-bandwidth tax on top of whichever baseline (default's flat ~9.4% or
tuned's cold-sensitive 12-23%) was already there -- it is not sufficient,
alone, to explain tuned's disproportionate real-serving regression.

**Correction found while writing this up:** an earlier pass of this analysis
had `gridX=2176`→`'default'` and `gridX=4352`→`'tuned'` in the classification
SQL -- backwards, a plain copy/paste error, not a new physical effect. It
produced a table that looked like a genuine puzzle (default appearing *more*
sensitive to cold-weight cycling than tuned, the opposite of `12`), which is
exactly what a config-label swap looks like once cold-cycling's real,
established asymmetry (tuned sensitive, default flat) gets attached to the
wrong name. Caught by rederiving each config's grid size arithmetically from
its own `BLOCK_SIZE_M` (shown above) and cross-checking against `05`'s
independently-established register-count/gridX table -- both agreed and
both contradicted the original labels. Worth recording plainly: not every
anomaly in this investigation has been a new hardware mechanism: this one
was self-inflicted, and the fix immediately made this experiment's
noKV numbers reproduce `12` almost exactly rather than contradict it.

## Infrastructure snag: the reused nsys session's exclusive GPU-metrics lock

The job template reused from `09`/`12` launches a `sleep infinity`
placeholder (wrapped in `nsys profile --gpu-metrics-devices=all
--start-later=true`) that waits for `/tmp/ready`, expecting the actual script
to run afterward via a separate `oc exec`. That separate `oc exec` is *also*
intercepted and wrapped in its own `nsys profile --gpu-metrics-devices=all`
call by the nsight-operator's injection mechanism -- and `--gpu-metrics-devices=all`
claims the GPU's metrics-sampling hardware exclusively, so the second wrap
failed outright (`Already under profiling`) while the placeholder's
collection was still open, even though the placeholder itself never touched
the GPU. Fixed by calling `profiler-stop` on the placeholder's collection
first (releasing the exclusive lock), then launching the real script via
`oc exec` in the background, then calling `profiler-start` again once its
*own* new collection ID appeared in its stderr -- which is why this capture's
window starts partway through the run (repeat ~11/16) rather than from the
first iteration.

## Reproduce

```bash
# copy kv_cache_traffic_repro.py, ../../../_vendored_matmul_timing.py, and this
# shape's tuned-config JSON onto an nsys-injected, SCC-elevated pod (see 09's
# "Reproducing the GPU-metrics-enabled isolated capture"), then:
#
# 1. touch /tmp/ready to unblock the placeholder
# 2. profiler-stop the placeholder's collection (frees the exclusive
#    gpu-metrics-devices=all lock -- see snag above)
# 3. launch the script via `oc exec ... -- sh -c 'cd /tmp && python3
#    kv_cache_traffic_repro.py'` in the background
# 4. profiler-start again once the new collection ID appears in its output
cd /tmp && python3 kv_cache_traffic_repro.py
```
