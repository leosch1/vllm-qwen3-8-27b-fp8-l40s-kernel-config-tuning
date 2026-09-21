# 02 — End-to-end concurrency sweep

**Status:** the anomaly this experiment discovered is real and confirmed. The
*explanation* proposed for it at the time (resource contention, see
[`04-contention-simulation`](../04-contention-simulation/)) was later
superseded — see [`08-per-layer-shape-mapping`](../08-per-layer-shape-mapping/)
and [`09-isolated-vs-real-flops-inflation`](../09-isolated-vs-real-flops-inflation/)
for the actual root cause established later in this project.

## What

Serve the real model under `vllm/vllm-openai:v0.27.1`, both with the tuned
kernel configs and with vLLM's untuned default, and benchmark with `vllm
bench serve` at a sweep of concurrency levels: 1, 2, 4, 8, 16, 32, 48, 64,
96, 128 (`--max-num-seqs=128`). Same client traffic profile at every point
(random dataset, 256 input / 128 output tokens).

## Why

[`01-isolated-kernel-benchmark`](../01-isolated-kernel-benchmark/) established
a real, large kernel-level speedup — but a kernel-level number isn't an
end-to-end number. The natural next question: does the tuned config's win
survive real serving, at realistic traffic, at every load level a production
deployment would actually see — not just one point.

## How

```bash
vllm bench serve \
    --backend openai-chat --endpoint /chat/completions \
    --model qwen-27b --dataset-name random \
    --num-prompts <N> --random-input-len 256 --random-output-len 128 \
    --request-rate inf --max-concurrency <C> --temperature 0 \
    --save-result --save-detailed
```
run once per concurrency level `<C>` against each of the two live server
configs (default kernel config mounted / tuned kernel config mounted),
sample counts scaled up (16x) relative to a single low-concurrency run for
stability. Full raw logs, in this folder:
[`sweep-default-labeled.log`](./sweep-default-labeled.log) /
[`sweep-tuned-labeled.log`](./sweep-tuned-labeled.log) (also `-128.log`
variants from an earlier pass).

## Results

At `--max-concurrency 1` (closest match to the isolated kernel benchmark's
own regime — one request at a time):

| | |
|---|---|
| output tok/s | +21 – 29% across repeated runs |
| mean TPOT | −19 – 27% |
| kernel-level anchors ever negative (reference) | 0/90 |

A clean win, matching the kernel-level story almost exactly. Then the full
sweep:

**Output tok/s delta (tuned vs. default), by concurrency** — see
[`chart-decay.html`](./chart-decay.html):

| concurrency | delta |
|---:|---:|
| 1 | +28.0% |
| 2 | +9.9% |
| 4 | +4.8% |
| 8 | +5.6% |
| 16 | +3.5% |
| 32 | +2.5% |
| 48 | −0.0% |
| 64 | −0.8% |
| 96 | −2.9% |
| 128 | −4.1% |

A smooth, monotonic decay through zero — not a step change, not noise. **A
shrinking advantage as batch size grows is exactly what the kernel-level
table predicts** (large-M kernel speedups taper to single digits). **A
negative one isn't** — nothing in the isolated kernel measurements ever
showed the tuned config losing outright. This is the finding that opened the
whole rest of the investigation.

Absolute metrics behind the same sweep (output tok/s, mean TPOT, mean TTFT) —
see [`chart-absolute.html`](./chart-absolute.html), and
[`chart-absolute-zoomed.html`](./chart-absolute-zoomed.html) for the same
data with each panel's y-axis cropped to its own range so the gap is legible
— confirm the two configs are close enough in absolute terms that a
zero-based axis would hide the separation; TTFT specifically gets *worse*
under tuning as concurrency rises, even though TPOT stays roughly
comparable. (The TTFT-specific reasoning —
why prefill is hit differently than decode — lives in
[`03-gemm-traffic-vs-speedup`](../03-gemm-traffic-vs-speedup/), since it
needed the real per-M traffic breakdown from that experiment to explain.)

At the time, one candidate mechanism was proposed for the decline not
flattening out (contention scaling *with* concurrency rather than toward a
fixed floor, since GEMM's own step-time share should otherwise shrink and
flatten the curve) — see [`04-contention-simulation`](../04-contention-simulation/)
for that hypothesis and how it was eventually resolved by later,
in-session profiling work rather than by the synthetic-load script this
experiment originally proposed.

## Later, more rigorous rerun (upstream-PR-evidence style, left incomplete)

A second attempt at this same default-vs-tuned e2e comparison, on a
separate, non-production `qwen-27b-pr-evidence` deployment (so it never
touches, stops, or conflicts with the production `qwen-27b` InferenceService)
— matching the client/benchmark conventions of two prior merged vLLM PRs
(`vllm-project/vllm#23504`, `#52752`) for closer apples-to-apples framing if
this ever gets cited upstream. Deployment recipe:
[`pr-evidence-deployment-README.md`](./pr-evidence-deployment-README.md),
manifests [`pvc.yaml`](./pvc.yaml),
[`servingruntime.yaml`](./servingruntime.yaml) (default/tuned toggled by
commenting/uncommenting its tuned-config `volumeMounts`/`volumes` block),
[`inferenceservice.yaml`](./inferenceservice.yaml).

Results so far: [`pr-evidence-rerun-results.md`](./pr-evidence-rerun-results.md)
— **incomplete**. The default-config rows are from a fresh rerun; the
tuned-config rows are still carried over from the original run further above
(same numbers as this README's own table), pending a matching tuned rerun on
this exact deployment. Don't compute deltas from that file's tables until
both rows come from the same rerun — it's included here for completeness and
as a ready-to-resume starting point, not as a second confirmed result.

## Reproduce

Serve command (original sweep, both configs):
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --port=8080 --model=/mnt/models --served-model-name=qwen-27b \
    --tensor-parallel-size=2 --max-model-len=8192 --reasoning-parser=qwen3 \
    --enable-auto-tool-choice --tool-call-parser=qwen3_coder \
    --max-num-seqs=128
# env: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Then run the `vllm bench serve` command above once per concurrency level,
per config. For the separate, non-production PR-evidence deployment instead,
use the manifests and sequence in
[`pr-evidence-deployment-README.md`](./pr-evidence-deployment-README.md).
