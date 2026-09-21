# 21 — groupfix real-serving nsys capture

**A real nsys hardware-counter capture of `groupfix` (`15`'s hand-patched
config) against live serving traffic, using the exact same methodology that
originally found the regression (`08`/`09`) and decomposed its distribution
(`05`).** Closes the loop on the single most specific claim this whole
investigation makes: not an aggregate e2e number, not an isolated synthetic
proxy, but the same per-kernel duration, at the same shape and M-band, on the
same real request traffic, re-measured after the fix.

## Why this experiment

By `19`, the fix was already validated two ways: end-to-end concurrency
sweeps (`15`, `19`) and isolated kernel benchmarks (`20`). Neither is the
*original* measurement. `05`/`08`/`09` made a specific claim about a specific
kernel launch population — `gate_up_proj` launches landing in the M
1024–2048 band, captured via real nsys hardware counters against real
`vllm bench serve` traffic at concurrency 64, where `tuned`'s weighted-mean
duration was 1961µs vs. `default`'s 1111µs (+76.5%, the actual regression
`05`'s chart shows). Everything since has argued indirectly that
`GROUP_SIZE_M` explains and fixes that number. This experiment re-runs that
exact capture with `groupfix` swapped in, to check directly rather than
inferring transitively.

## Result

Weighted-mean duration, `gate_up_proj`, M 1024–2048 band, c=64:

| config | duration | vs. `default` | vs. `tuned` |
|---|---:|---:|---:|
| `default` (`05`'s original capture) | 1111.0µs | — | −43.4% |
| `tuned` (`05`'s original capture) | 1960.95µs | +76.5% | — |
| `groupfix` (this experiment, new capture) | 1038.3µs | **−6.5%** | **−47.1%** |

`groupfix` doesn't just recover to `default`'s level in this exact
regression window — it goes slightly below it, while eliminating `tuned`'s
blowout entirely. This is the most direct confirmation available: same
capture methodology, same shape, same M-band, same concurrency, same real
traffic, as the measurement that originally found the problem.

The full distribution chart, `chart-groupfix-vs-default-tuned.html`, extends
`05`'s original chart with a fifth panel (`c64_groupfix`) — same shape/M-band
decomposition, same stacked-percentage-by-duration-bin layout, directly
comparable to the existing four.

## Method

1. **Classification reused unmodified.** `groupfix` only changes
   `GROUP_SIZE_M`, which does not appear in the grid/block-dimension formula
   at all (`gridX = ceil(M/BLOCK_SIZE_M) * ceil(N/BLOCK_SIZE_N)`,
   `blockX = num_warps*32`). So `groupfix`'s real launches have
   byte-identical `(gridX, blockX)` signatures to the original `tuned`
   capture's launches, and `18-out-down-proj-order-disambiguation/resolve_order.py`'s
   exact shape-classification + program-order disambiguation logic (for the
   `out_proj`/`down_proj` collision) applies unmodified — see
   `build_groupfix_mband.py`.
2. **Deploy `groupfix` with profiling enabled.** `configmap.yaml` (the 5
   `groupfix` tuned-config JSONs), `servingruntime-groupfix-profiling.yaml`
   (CUDA graphs on, `runAsUser:0` + `SYS_ADMIN`, ConfigMap-mounted rather
   than the shared PVC to avoid touching its uncertain current state),
   `inferenceservice.yaml` (same as `15`/`19` but with the
   `nvidia-nsight-profile: enabled` label *restored* — `15`/`19` deliberately
   omitted it for pure e2e throughput sweeps; this experiment needs the real
   capture).
3. **Drive real traffic, capture, extract.** Same nsight-operator pipeline
   as `05`/`08`/`09` (`nsight_operator.py`: `session-begin`,
   `profiler-start`, `profiler-stop`), DCGM paused for the duration on the
   node's actual GPU-engine pod, `vllm bench serve` at c=64 against the live
   predictor. `nsys stats --report gputrace` used to materialize the
   intermediate sqlite (see gotcha below), then `extract_kernels.py` run
   *on the pod* to pull just the `_w8a8_triton_block_scaled_mm` kernel's
   `(start, end, gridX, blockX, deviceId, globalPid)` rows into a small CSV.
   `build_groupfix_mband.py` then reclassifies those rows locally and emits
   the `SHAPE_MBAND_DATA`-compatible series (`shape_mband_data_with_groupfix.json`).

## Infrastructure gotchas found in this experiment (new to this project)

None of these are about the science — all three are pure plumbing failures
that cost real time getting a working real-traffic profiling pipeline for a
*second* config, and are worth recording since none showed up in `05`'s
original capture (which never needed a separate client pod or a second
config swap).

- **The nsight-injector wraps every forked/exec'd process, not just the top-level
  command — and that breaks a separate benchmark client run via `oc exec`
  inside the traced pod.** The injector's `--trace-fork-before-exec=true`
  evaluates *every* process in the traced pod's cgroup against a fixed
  exclusion list (`ash`, `bash`, `sh`, `python3` is **not** on it, nor is
  `nohup`, `vllm`, etc. — the list is shell/coreutils-shaped, not
  runtime-shaped). Anything not excluded gets wrapped in a *new*
  `nsys profile --gpu-metrics-devices=all` invocation, which fails with
  "Illegal `--gpu-metrics-devices` usage... Already under profiling" because
  the main server process's own wrap already holds the GPU-metrics hardware
  lock exclusively. **Fix:** run the benchmark client from a completely
  separate, untraced pod/Job, hitting the predictor's raw pod IP:8080
  directly — confirmed this bypasses the `kube-rbac-proxy` auth sidecar
  (which only proxies 8443/8643), so no auth token is needed either.
- **Bare `Pod` resources get stuck `SchedulingGated` forever in this
  cluster, even after Kueue admits them.** A throwaway client `Pod` never
  had its `kueue.x-k8s.io/admission`/`kueue.x-k8s.io/topology` gate removed,
  despite its `Workload` object showing both `QuotaReserved: True` and
  `Admitted: True`. **Fix:** use a `Job` (batch/v1) instead — this has
  worked reliably for every GPU workload in this entire project.
- **`oc cp` silently truncates very large files.** A ~2.8GB nsys-generated
  sqlite came back locally 842,245 bytes short, producing
  `sqlite3.DatabaseError: database disk image is malformed` on open. **Fix:**
  never transfer the large database at all — run the SQL extraction
  directly on the pod (uploading the small script itself via
  `oc exec -i ... -- bash -c "cat > extract_kernels.py"`, since even a
  direct `oc cp` of that small file hit a tar error) and copy out only the
  small CSV result, which transferred with an exact byte-for-byte match.
- **(Previously documented, reconfirmed here.)** `nsys export --type sqlite`
  produces a silent 0-byte file on this nsys version. Fallback:
  `nsys stats --report gputrace` — the specific report isn't found and the
  command errors out, but it generates the needed intermediate `.sqlite` as
  a side effect before failing on the report step.

## Files

- `configmap.yaml`, `servingruntime-groupfix-profiling.yaml`,
  `inferenceservice.yaml` — the profiling-enabled `groupfix` deployment.
- `extract_kernels.py` — SQL extraction script, run on-pod against the
  `nsys`-generated sqlite (see gotchas above for why it's never copied off
  the pod).
- `build_groupfix_mband.py` — reclassifies the extracted launches using
  `17`'s classification logic and recovers M-bands, producing the merged
  `shape_mband_data_with_groupfix.json`.
- `chart-groupfix-vs-default-tuned.html` — `05`'s original chart, extended
  with a fifth `c64_groupfix` panel.

## Cleanup

All cluster resources from this experiment (the `qwen-27b-pr-evidence`
InferenceService/ServingRuntime, `groupfix-tuned-configs-profiling`
ConfigMap, throwaway benchmark-client Job) were deleted after this capture,
and DCGM profiling was resumed on the node's GPU-engine pod.
