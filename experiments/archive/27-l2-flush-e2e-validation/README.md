# 27 — L2-flush e2e validation

> Fully covered in [`experiments/06-l2-flush-e2e-validation`](../../06-l2-flush-e2e-validation/).

**Status: complete.** Mirrors `19`'s cache-aware e2e validation exactly,
for `25`'s L2-flush-tuned config instead: same 10-point concurrency sweep
(`vllm bench serve`, output tok/s, c ∈ {1,2,4,8,16,32,48,64,96,128}), same
CUDA-graphs-enabled real serving deployment, same shared PR-evidence
InferenceService/ServingRuntime pattern used throughout `15`/`19`/`21`/`22`.

## Why

`25`/`26` validate the L2-flush fix at the kernel level (isolated
benchmark, and head-to-head against `17`'s cycling fix). Neither is a real
end-to-end serving measurement — the same gap `19` closed for the
cache-aware config after `17`/`20`. This closes it for L2-flush: does the
general, model-independent fix actually deliver real throughput gains in
production-shaped serving traffic, not just in synthetic isolated
benchmarks.

**No need to rerun `default`** — reusing `19` round 2's `default` sweep
directly (same hardware, same image, same serving args, run same day as
that round's own comparison, so it's a valid baseline): `30.85, 66.04,
126.55, 226.15, 379.67, 572.18, 696.94, 773.25, 875.08, 932.82` tok/s at
c=1..128 (`../19-cache-aware-config-e2e-validation/sweep-default-graphs-v2.log`).
Only the L2-flush config itself needs a fresh deployment and sweep.

## Method

- **`configmap.yaml`** — `25`'s `l2-flush-tuned-configs/` (the real,
  standard-format output of the full production-scope L2-flush rerun),
  packaged the same way as `19`'s `configmap-v2.yaml`.
- **`servingruntime-l2-flush-graphs.yaml`** — `19`'s
  `servingruntime-cache-aware-v2-graphs.yaml` with only the config mount
  swapped — same image, same args, CUDA graphs enabled (no
  `--enforce-eager`), same `fp8-utils-instrumented` diagnostic mount kept
  for parity with the reused default baseline's own deployment.
- **`inferenceservice.yaml`** — identical to `19`'s (same name
  `qwen-27b-pr-evidence`, same profiler service account, same resources).
- Sweep: from inside the predictor pod, `vllm bench serve` at each
  (concurrency, num-prompts) pair `(1,50) (2,50) (4,64) (8,128) (16,256)
  (32,512) (48,768) (64,1024) (96,1536) (128,2048)`, `--dataset-name random
  --random-input-len 256 --random-output-len 128`, same as `15`/`19`.

## Result

Deployed config verified in the running pod before sweeping: `GROUP_SIZE_M=16`
at M=1024, `GROUP_SIZE_M=32` at M=2048 — matches `25`'s
`l2-flush-tuned-configs/` exactly. Full 10-point sweep completed cleanly,
0 failed requests at any concurrency. Raw log: `sweep-l2flush-graphs.log`.

**Never regresses vs. `default` anywhere in the sweep, and tracks `19`'s
cache-aware result almost exactly:**

| c | default | l2-flush | vs. default | vs. `15`'s groupfix | vs. `19`'s cache-aware v2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.85 | 40.25 | +30.5% | +1.7% | −0.5% |
| 2 | 66.04 | 75.07 | +13.7% | +3.3% | +0.3% |
| 4 | 126.55 | 135.42 | +7.0% | +0.1% | −1.3% |
| 8 | 226.15 | 249.27 | +10.2% | +3.0% | +0.1% |
| 16 | 379.67 | 408.28 | +7.5% | +1.1% | −0.2% |
| 32 | 572.18 | 618.54 | +8.1% | +1.5% | +0.1% |
| 48 | 696.94 | 743.64 | +6.7% | +1.5% | −0.4% |
| 64 | 773.25 | 819.25 | +5.9% | +2.6% | −0.0% |
| 96 | 875.08 | 911.39 | +4.1% | +0.7% | +0.1% |
| 128 | 932.82 | 968.23 | +3.8% | +1.1% | −0.1% |

Output tok/s. Chart (all three fixes together, same default baseline):
`chart-e2e-vs-default.html`. Also `chart-e2e-tuned-vs-l2flush.html` — a
cleaner two-line version against just the original, unfixed `tuned` config
(`15`'s sweep): `tuned` reproduces the founding regression (+28.0% at c=1
decaying to -4.2% at c=128), while `l2-flush` never crosses zero anywhere
in the sweep (+30.5% down to +3.8%) — the real-serving counterpart to
`20` part 11's kernel-level finding that the original `tuned` config's
`GROUP_SIZE_M=1` cost becomes clearly visible once measured under a
realistically cold cache.

**This is the cleanest possible confirmation available**: `l2-flush` and
`19`'s `cache-aware` config track each other within ±1.3% at every single
concurrency point — well inside normal sweep-to-sweep noise — despite
being produced by two structurally different tuning mechanisms (a
model-independent direct L2 flush vs. a model-specific 64-copy cycling
pool). At the real end-to-end serving level, not just in isolation, the
general fix delivers the same production benefit as the fix it's meant to
replace, and both comfortably beat `groupfix`'s single hand-patched
parameter at every concurrency above c=4.

Combined with `25` (tuning-rerun equivalence) and `26` (isolated-kernel
equivalence, both cache-pressure mechanisms), this closes the validation
loop for the L2-flush mechanism at all three levels this project uses:
tuning-time, kernel-level, and real end-to-end serving.

## Reproduce

```bash
oc apply -f configmap.yaml
oc apply -f servingruntime-l2-flush-graphs.yaml
oc apply -f inferenceservice.yaml
# wait for predictor ready, then from inside the pod:
vllm bench serve \
    --backend openai-chat --base-url http://localhost:8080/v1 --endpoint /chat/completions \
    --model qwen-27b-pr-evidence --tokenizer /mnt/models \
    --dataset-name random --num-prompts <N> --random-input-len 256 --random-output-len 128 \
    --request-rate inf --max-concurrency <C> --temperature 0
```
