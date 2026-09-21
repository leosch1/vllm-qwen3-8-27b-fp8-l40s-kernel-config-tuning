# 15 — GROUP_SIZE_M e2e validation

**Charts:** [`report.html`](./report.html) -- visual summary of both this
experiment and `14` (DRAM-Read mechanism, the corrected 10-point sweep, and
the eager-vs-graphs debugging story).

**Status:** built and run; **validated at the e2e level, with real margins.**
The first pass of this experiment produced a false lead (see "A false lead"
below) that made the whole comparison look noise-sized. Once corrected,
`groupfix` **never regresses relative to `default` anywhere in the 10-point
sweep** (+28.5% at c=1 down to +2.9% at c=128, always positive) and closes an
*increasing* gap against plain `tuned` as concurrency rises, reaching +7.4%
at c=128 -- exactly the point where the original regression was worst. This
closes the loop from `14`'s isolated kernel finding through to real serving
throughput.

## What

`14-group-size-m-repro` showed, at the isolated-kernel level, that a single
config parameter (`GROUP_SIZE_M`) is a real, sufficient cause of tuned's
cold-weight-cycling DRAM-Read sensitivity, and that fixing it recovers a
flat, default-like signature while keeping tuned's other speed advantages.
The natural next question: does this translate to real end-to-end serving
performance? Specifically, does a `tuned + groupfix` deployment stop showing
the negative e2e delta `02-e2e-concurrency-sweep` found at concurrency 48-128
(-0.0% to -4.1%)?

## How

Patched `gate_up_proj`'s tuned config JSON (`N=17408,K=5120,...json`): every
large-M bucket (>=512) that had `GROUP_SIZE_M=1` was changed to that
bucket's own `num_pid_m` (mirroring default's "one group covers all of M"
strategy, same logic validated in `14`):

| M | 512 | 1024 | 1536 | 2048 | 3072 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| `GROUP_SIZE_M` (original) | 1 | 1 | 1 | 1 | 1 | 1 |
| `GROUP_SIZE_M` (patched) | 4 | 8 | 12 | 16 | 24 | 32 |

The other 4 GEMM shapes' tuned configs are untouched. Deployed via the same
`qwen-27b-pr-evidence` InferenceService/ServingRuntime as `02` (recreated for
this test -- had been deleted at the end of the previous session), same
image/resources/traffic profile, benchmarked with the identical `vllm bench
serve` command and `--num-prompts` scaling `02` used at each concurrency
level (1→50, 2→50, 4→64, 8→128, 16→256, 32→512, 48→768, 64→1024, 96→1536,
128→2048 -- matching `02`'s exact sweep).

## A false lead: `--enforce-eager` was silently disabling CUDA graphs

The first full pass of this experiment (`default`/`tuned`/`groupfix`, all
three, full 10-point sweep) came back showing every config running at
roughly 30-80% of a week-old baseline's throughput, worse at low
concurrency, recovering toward the baseline at high concurrency -- and,
worryingly, `tuned` trailing `default` by a roughly *constant* 3-6%
basically everywhere, not just at high concurrency like the original
regression. GPU-level causes were checked directly and all ruled out:
clocks pinned at 2520MHz max boost throughout, power draw well under cap,
PCIe at full Gen4x16 on both GPUs, no NVLink involved (pure PCIe topology,
never present on this node), no other pods contending.

**The actual cause was in the deployment manifest, not the hardware.** The
shared `servingruntime.yaml` this whole experiment (and `02` before it) was
copied from carries `--enforce-eager`. Per `10-w8a8-call-plan-recorder`'s
own README: *"`--enforce-eager` was added to `servingruntime.yaml`
specifically so every call would actually reach this instrumentation, with
an explicit comment that eager mode's own timing numbers are not
representative of normal (graph-enabled) serving performance... Neither
side was ever actually run with the env vars set."* The call-plan recorder
that flag existed for was never exercised, but the flag itself was never
removed -- it silently disabled CUDA graph capture/replay for every e2e
benchmark run since, including this entire experiment's first pass, without
anyone re-checking whether it was still wanted.

Direct confirmation, `default` config, three spot-check points, eager vs.
graphs restored:

| c | eager (broken) | graphs (fixed) | original week-old baseline |
|---:|---:|---:|---:|
| 1 | 10.08 tok/s | 30.80 tok/s | 30.85 tok/s |
| 64 | 479.90 tok/s | 769.69 tok/s | 772.27 tok/s |
| 128 | 750.12 tok/s | 931.25 tok/s | 934.00 tok/s |

With graphs restored, `default` matches the original baseline within
99.3-100.0% at every point in the full sweep (below) -- there was no
environment drift, no hypervisor contention, nothing hardware-related at
all. A fixed ~70ms/step overhead at low concurrency, shrinking to ~32ms at
c=128 (the shape that looked like "host-side scheduling delay" in the first
pass) is exactly the signature of losing CUDA graph replay: eager mode
re-pays Python dispatch + kernel-launch overhead every step, a cost graphs
skip almost entirely -- large relative to a cheap low-concurrency step,
diluted once each step is doing 100+ms of real work. The
`servingruntime-default-graphs.yaml` / `servingruntime-groupfix-graphs.yaml`
variants in this folder are the corrected manifests (`--enforce-eager`
removed, everything else identical) used for the results below. The
`sweep-*-EAGER-INVALID-*.log` files are kept for the record, not used in any
conclusion.

## Results (corrected: CUDA graphs enabled, matching production's actual behavior)

Full 10-point sweep, all 3 configs, same day, same pod/node, one run per
condition:

| c | default | tuned | groupfix | tuned vs. default | **groupfix vs. default** | groupfix vs. tuned |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 30.80 | 39.43 | 39.57 | +28.0% | **+28.5%** | +0.4% |
| 2 | 65.58 | 72.72 | 72.65 | +10.9% | **+10.8%** | -0.1% |
| 4 | 126.49 | 132.60 | 135.30 | +4.8% | **+7.0%** | +2.0% |
| 8 | 225.52 | 238.05 | 242.02 | +5.6% | **+7.3%** | +1.7% |
| 16 | 377.23 | 391.46 | 403.64 | +3.8% | **+7.0%** | +3.1% |
| 32 | 570.96 | 586.16 | 609.67 | +2.7% | **+6.8%** | +4.0% |
| 48 | 695.45 | 696.17 | 732.48 | +0.1% | **+5.3%** | +5.2% |
| 64 | 769.69 | 762.97 | 798.35 | -0.9% | **+3.7%** | +4.6% |
| 96 | 873.95 | 847.13 | 905.07 | -3.1% | **+3.6%** | +6.8% |
| 128 | 931.25 | 892.30 | 958.12 | -4.2% | **+2.9%** | **+7.4%** |

Output tok/s, `vllm bench serve`, `--dataset-name random --random-input-len
256 --random-output-len 128`. Full raw logs: `sweep-default-graphs-*.log`,
`sweep-tuned-graphs-full.log`, `sweep-groupfix-graphs-*.log`.

**Three things stand out:**

1. **`tuned` vs. `default` reproduces `02`'s original regression almost
   exactly** (+28.0% at c=1 decaying through zero around c=48 to -4.2% at
   c=128, vs. the original +27.9%→-4.3%) -- direct, independent replication
   of the founding observation of this entire investigation, now on a
   corrected, valid measurement.
2. **`groupfix` never regresses relative to `default` anywhere in the
   sweep** -- every single point is positive, from +28.5% at c=1 down to
   +2.9% at c=128, right where the original regression was worst.
3. **`groupfix` vs. plain `tuned` grows *with* concurrency** (+0.4% at c=1,
   roughly tied at low-M where `GROUP_SIZE_M` was never touched, up to
   +7.4% at c=128) -- the fix's benefit is concentrated exactly where the
   mechanism identified in `14` predicts it should be: the large-M buckets
   where `GROUP_SIZE_M` was patched.

This is now a real, decisive, independent confirmation of `14`'s isolated
finding -- not just a non-contradictory data point sitting on top of it.

## Infra notes from running this

- **`Multi-Attach` on the PVC**: the `dense-tuned-configs` PVC is
  `ReadWriteOnce` and stays attached to whichever node the predictor pod is
  on, *even when no container's `volumeMounts` actually reference it*
  (declaring it under `.spec.volumes` is enough to trigger attachment).
  A throwaway job to edit the PVC's contents must be pinned
  (`nodeSelector: kubernetes.io/hostname`) to the same node as the running
  predictor, or it fails to schedule with `FailedAttachVolume`.
- **Long-lived `oc exec` sessions to run multi-stage benchmarks are not
  reliable** in this network -- hit repeated `read tcp ...:6443: read:
  operation timed out` disconnects partway through, at inconsistent points
  (as early as ~2 minutes in). Critically, **the remote process survives
  the local disconnect** (it isn't attached to a TTY) and keeps running
  server-side, invisibly consuming GPU/CPU until explicitly found and
  killed (`ps aux` in the container, then `kill -9`) -- checked for and
  killed stray processes before each retry to avoid contaminating results
  with concurrent unaccounted-for traffic. Fixed by running each stage as
  `nohup ... > logfile 2>&1 & ... touch donefile` *inside* the container via
  one short-lived `oc exec`, then polling for the done-marker file with
  separate, short `oc exec` calls (each individually resilient to a dropped
  connection, unlike one long attached stream).
- **A config-swap between two passes needs a real Deployment spec diff to
  trigger a pod restart**, even when only a PVC file's *content* changed
  and the mount path/spec is identical -- Kubernetes doesn't detect PVC
  content changes and won't roll the pod on its own. Toggling
  `VLLM_W8A8_CALL_PLAN_RUN_ID` (harmless, already present for a different
  diagnostic purpose) between passes was enough to force a real rollout
  each time; re-applying byte-identical YAML is a no-op.
- **Never trust an inherited flag in a shared manifest without checking
  what it does.** `--enforce-eager` was three lines away from the mounts
  being toggled in the exact file being edited for every pass of this
  experiment, and its own comment explained why it existed -- it still took
  a direct question from the user ("are you sure the pod is configured the
  same as production?") to prompt the diff against the real production
  manifest that surfaced it.

## Reproduce

```bash
# 1. patch (or restore) the gate_up_proj tuned-config JSON as above, upload
#    to the dense-tuned-configs PVC via a throwaway Job pinned to the
#    predictor's node (see infra notes)
# 2. oc apply -f servingruntime-default-graphs.yaml   (default, graphs)
#    oc apply -f servingruntime-groupfix-graphs.yaml  (tuned or groupfix,
#      depending on which config is currently on the PVC -- graphs)
# 3. from inside the predictor pod:
vllm bench serve \
    --backend openai-chat --base-url http://localhost:8080/v1 --endpoint /chat/completions \
    --model qwen-27b-pr-evidence --tokenizer /mnt/models \
    --dataset-name random --num-prompts <N> --random-input-len 256 --random-output-len 128 \
    --request-rate inf --max-concurrency <C> --temperature 0
# run once per (C, N) pair: (1,50) (2,50) (4,64) (8,128) (16,256) (32,512)
#                            (48,768) (64,1024) (96,1536) (128,2048)
```
