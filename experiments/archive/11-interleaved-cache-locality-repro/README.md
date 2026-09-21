# 11 — Interleaved-kernel cache-locality reproduction

**Status:** built and run; clean negative result for the design actually
tested (round-robin interleaving of the other 4 W8A8 shapes). Also
surfaced an unrelated, significant discovery along the way — a real
upstream vLLM bug that under-reports every isolated-benchmark absolute
microsecond value in this whole project by ~10x (see
[`../../vllm/UPSTREAM_BUG_benchmark_config_avg.md`](../../../vllm/UPSTREAM_BUG_benchmark_config_avg.md)
in the vllm fork repo) — deliberately not yet corrected here, since it
doesn't affect this experiment's own internal comparison (see below).

## What

Test the "instruction/constant-cache locality" hypothesis from
[`09-isolated-vs-real-flops-inflation`](../09-isolated-vs-real-flops-inflation/)
directly, without vLLM, without a second process — fixing the specific flaw
identified in [`04-contention-simulation`](../04-contention-simulation/)
(which used a genuinely different, cross-process contention mechanism that
doesn't occur in real single-tenant-per-GPU serving).

## Why

The isolated kernel benchmark repeats the *identical* compiled kernel
back-to-back — warm instruction/constant cache every time. Real serving
sandwiches the same call between many *different* kernel types every
occurrence — cold every time. If that's really what's driving the real-vs-
isolated gap, a single-process, single-stream harness that interleaves
genuinely different kernels between repeats of `gate_up_proj` — with no
second process, no cross-process GPU sharing at all — should reproduce at
least part of the effect. If it doesn't, that's real evidence against the
hypothesis in this form, not an inconclusive result.

## How

`interleaved_kernel_repro.py`, run as a real-vs-synthetic-load-free
Kubernetes Job with **no `nvidia-nsight-profile` injection label**
(`interleaved-repro-job.yaml`) — deliberately not the live serving pod,
to avoid two confounds discovered the hard way while building this:

1. **nsys injector overhead.** Running on the actual serving pod (which
   does carry the injection label, for the GPU-metrics work in
   [`07`](../07-gpu-metrics-clock-throttling/)) with DCGM paused (required
   for the wrapped process to launch at all) made every kernel launch ~10x
   slower — continuous 1000Hz GPU-metrics telemetry collection dominating
   over actual kernel time at this scale. Solved by running on a separate,
   uninjected pod instead — the namespace itself carries no injection
   label, only specific pods opted in, so a plain Job in the same namespace
   runs completely natively.
2. **GPU clock state.** The second GPU node this Job landed on
   (`gpu-node-b`, otherwise idle — the model-serving node's 2
   GPUs are both already in use) was sitting in `P8` (idle power state,
   ~210MHz vs. its 2520MHz max boost) — a brief warmup never gives NVIDIA's
   boost-clock ramp enough sustained load to kick in on a previously-idle
   GPU, and every measurement would otherwise be dominated by clock state,
   not kernel content. Fixed with an explicit ~4s sustained-load spin-up
   before any timed measurement, confirmed via `nvidia-smi` that the clock
   actually reached `P0` first.

A third, unrelated bug was also found and fixed here: allocating a fresh
`torch.cuda.Event()` pair every iteration (12,800+ of them) without ever
synchronizing until the very end lets outstanding, un-freed events pile up
— CUDA event bookkeeping overhead itself then scales with outstanding-event
count and swamps the real kernel time. Fixed by flushing (synchronize, read
back, discard) every 256 iterations — bounds outstanding events while still
preserving real async pipelining *within* each flush window.

**The actual design**, once those three were fixed: replay the model's real
per-decode-step call sequence layer-by-layer (64 layers,
`full_attention_interval=4`, matching the real per-step call multiplicities
established in [`04`](../04-contention-simulation/) — 64/48/16/64/64 across
the 5 shapes) at `M=2048`, all on one process, one default CUDA stream, no
multiprocessing anywhere. Only `gate_up_proj`'s own launches are timed
(CUDA events bracketing just that call); the other 4 shapes' calls are real,
executed kernel launches too — just not the thing being measured — so nothing
artificially drains the pipeline between them. Compared against a control
using the exact same timing methodology but calling only `gate_up_proj`
back-to-back with itself (no interleaving).

## Results

| | isolated (back-to-back) | interleaved (real per-layer sequence) | ratio |
|---|---:|---:|---:|
| default | 1348.4µs | 1318.2µs | 0.98x |
| tuned | 1258.5µs | 1248.7µs | 0.99x |

**No effect.** Neither config shows a meaningful penalty from interleaving —
both are within ~1-2% of their own isolated baseline, indistinguishable from
noise. Critically, there's no *tuned-specific* degradation either, which is
what would be needed to call this a reproduction of the real regression
(which is unique to tuned at high M, not something default also shows).

This comparison is unaffected by the `/10` upstream bug noted above, since
both arms here use the same from-scratch CUDA-event timing, not
`benchmark_config()` — the internal isolated-vs-interleaved ratio is exactly
as trustworthy as the absolute numbers are proportionally wrong, i.e. fully
trustworthy for this specific question.

**What this narrowed down:** interleaving *other W8A8 GEMM variants*
(different tile configs, same underlying kernel family) isn't enough
instruction/cache diversity to reproduce whatever real serving does to
`gate_up_proj`. That's what motivated the faithful version below.

## Faithful version: real attention/mamba kernels, not just other W8A8 shapes

`interleaved_kernel_repro_v2.py`. Same single-process/single-stream design
and timing methodology, but the interleaved kernels between `gate_up_proj`
repeats are now genuinely different real kernels, not other tile configs of
the same GEMM family:

- **Real vLLM `causal_conv1d_fn`** (`vllm/model_executor/layers/mamba/ops/causal_conv1d.py`)
  for the linear-attention layers' conv step — vLLM's actual, exact kernel,
  confirmed tractable to call directly with synthetic continuous-batching
  inputs (8 sequences × 256 tokens, matching this workload's real established
  prompt length rather than one implausible 2048-token sequence).
- **A real PyTorch reference computation** (exp-gated recurrent state
  update via `einsum`, matching the actual gated-delta-rule math) standing
  in for the recurrent step — a deliberate, documented scope reduction: the
  real backend dispatches to one of 3 optional-dependency implementations
  (flashinfer/fla/cutedsl) depending on hardware/availability, each with a
  different call signature, and guessing which one blind (this design
  decision was made while cluster connectivity was briefly down) risked
  wasted effort for uncertain payoff. Also decoupled from the real per-call
  token count (256) down to an 8-token chunk — the true per-token
  sequential Python loop at 256 tokens × 9,600 calls was ~2.4M sequential
  einsum dispatches, wildly impractical; still real, substantial, genuinely
  different GPU work each occurrence, just smaller in volume than the real
  kernel would process in one call.
- **PyTorch's native `scaled_dot_product_attention`** for full-attention
  layers — real, complex, fused, structurally very different from a GEMM —
  rather than vLLM's own PagedAttention, which needs a KV-cache block-table
  setup this question doesn't need (the question is whether kernel-*family*
  diversity matters, not exact fidelity to vLLM's specific attention
  kernel).

Real model dims used throughout (`Qwen/Qwen3.8-27B-FP8`, TP=2, from the
deployed model's own `config.json`): `full_attention_interval=4`,
`head_dim=256`, `num_attention_heads=24` (12/rank), `num_key_value_heads=4`
(2/rank), `linear_conv_kernel_dim=4`, `linear_key_head_dim=128`,
`linear_num_key_heads=16`, `linear_value_head_dim=128`,
`linear_num_value_heads=48`.

### A real methodological detour: node-dependent measurement instability

The first run (on the same pod experiment 11's first design had been
running on for ~9 hours) came back badly bimodal — every one of the four
passes (both configs × isolated/interleaved) split into a fast cluster
(~1000-1150µs) and a slow cluster (~3500-5900µs), inconsistently. Critically,
**the isolated control showed the exact same bimodality**, with zero other
kernel types involved at all — which immediately ruled out "correlates with
which kernel ran before it" as the explanation (there's no neighbor kernel
in the isolated control). A fresh pod, rescheduled on the same node, showed
the identical instability from a cold start too — ruling out "9 hours of
prior load" as the cause. `nvidia-smi -lgc` (explicit clock locking) isn't
permitted in this container (no elevated privileges), so the instability
couldn't be fixed directly. It resolved itself on a second fresh pod +
rerun with no code changes — most likely NVIDIA boost-clock ramp behavior
on this specific, otherwise-idle secondary node (`gpu-node-b`)
being less consistent than the primary serving node, which stays under
continuous real load. Re-ran a quick sanity check with the *original*
(cheap-version) script first on the fresh pod to confirm clean behavior
(got 0.99x/0.98x, matching the original clean run almost exactly) before
trusting the faithful-version run below.

### Results

| | isolated | interleaved (real conv+gdn+attn) | ratio |
|---|---:|---:|---:|
| default | 1419.9µs | 1134.3µs | 0.80x |
| tuned | 1288.8µs | 1136.7µs | 0.88x |

Clean run (tight distributions, no bimodality; GPU confirmed at `P0`,
1635MHz before timing started). **Not the pattern needed.** Both configs get
*faster* when interleaved with real, diverse kernels — not slower — and by
similar (not tuned-specific) magnitudes. Both interleaved values converge to
nearly the same absolute number (~1135µs) despite meaningfully different
isolated baselines, suggesting a general "diverse workload sustains
clock/power state more consistently than repeating one kernel" effect, not
a cache-locality-driven slowdown specific to `gate_up_proj` or to tuned.

**Conclusion: two independent designs — round-robin other W8A8 shapes, and
real conv/attention/recurrent kernels — both fail to reproduce the
tuned-specific real-serving regression in a single-process, single-stream
harness.** This meaningfully narrows the remaining hypothesis space: the
effect most likely needs something a single-process synthetic harness
structurally cannot produce — genuine multi-request scheduling/queueing
dynamics (many independent sequences dynamically batched by vLLM's actual
scheduler, not a fixed deterministic replay loop), or something specific to
**CUDA graph capture/replay** (real vLLM captures decode steps as CUDA
graphs; both harnesses here call every kernel in plain eager mode, a
genuinely untested variable neither design controlled for).

## Reproduce

```bash
oc apply -f interleaved-repro-job.yaml   # namespace: enterprise-ai, no injection label
# copy interleaved_kernel_repro.py (or _v2.py), _vendored_matmul_timing.py
# (repo root), and tuned-configs/*.json onto the pod at /tmp, then:
oc exec -n enterprise-ai <pod> -- sh -c "cd /tmp && python3 interleaved_kernel_repro.py"
oc exec -n enterprise-ai <pod> -- sh -c "cd /tmp && python3 interleaved_kernel_repro_v2.py"
```
If results come back bimodal/noisy, delete and recreate the Job to get a
fresh pod before trusting the numbers — this was a real, reproducible issue
on the specific secondary node used here, not a one-off fluke.
