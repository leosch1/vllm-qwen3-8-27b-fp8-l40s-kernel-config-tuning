# PR evidence: e2e serving benchmarks

Temporary, NOT GitOps-managed resources for benchmarking `qwen-27b`
(`Qwen/Qwen3.8-27B-FP8`) end-to-end under `vllm/vllm-openai:v0.27.1` with the
freshly re-tuned dense W8A8 block-FP8 configs from
`performance-tests/qwen3-8-27b-fp8-dense-v0.27.1-configs/`, to get real
`vllm bench serve` numbers for the upstream PR (matching the precedent of
vllm-project/vllm#23504 and #52752, and this repo's own established
`performance-tests/benchmark-results.md` convention).

Separate name (`qwen-27b-pr-evidence`) from the production `qwen-27b`
`InferenceService`/`ServingRuntime` in `manifests/` -- doesn't touch, stop, or
conflict with production. Apply/delete manually with `oc apply -f` /
`oc delete -f`; never added to `manifests/kustomization.yaml`, so ArgoCD never
sees or manages these.

## Sequence

1. `oc apply -f pvc.yaml`
2. Upload the local tuned-config JSONs onto that PVC (see below).
3. `oc apply -f servingruntime.yaml inferenceservice.yaml` with
   `servingruntime.yaml`'s `volumeMounts`/`volumes` block commented out (as
   shipped) -- this is the **default-config** pass. Run both benchmark
   profiles.
4. Uncomment that block in `servingruntime.yaml`, `oc apply -f servingruntime.yaml`
   again, wait for the pod to roll -- this is the **tuned-config** pass. Rerun
   both benchmark profiles.
5. `oc delete -f .` when done.

## Uploading the tuned configs onto the PVC

No production pod has this PVC mounted, so use a throwaway busybox pod (same
trick used earlier this session, just in reverse -- uploading in instead of
pulling out):

```bash
oc apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: pr-evidence-config-upload
  namespace: enterprise-ai
spec:
  restartPolicy: Never
  containers:
    - name: upload
      image: docker.io/library/busybox:latest
      command: ["sleep", "300"]
      volumeMounts:
        - mountPath: /configs
          name: configs
  volumes:
    - name: configs
      persistentVolumeClaim:
        claimName: qwen3-8-27b-fp8-dense-v0271-configs
EOF

# wait for Running, then upload one file at a time via exec+cat -- `oc cp`'s
# tar stream was silently closing with no data in this environment (exit 0,
# but nothing landed on the PVC); this is more verbose but actually works:
cd performance-tests/qwen3-8-27b-fp8-dense-v0.27.1-configs
for f in *.json; do
  oc exec -i -n enterprise-ai pr-evidence-config-upload -- sh -c "cat > '/configs/$f'" < "$f"
done
oc exec -n enterprise-ai pr-evidence-config-upload -- ls -la /configs   # verify all 5 landed
oc delete pod -n enterprise-ai pr-evidence-config-upload
```

## Benchmark commands

Same `vllm bench serve` container/tokenizer setup already used throughout this
session, pinned to `v0.27.1` (not `:latest`) to match the exact version being
tested -- eliminates client/server version skew as a variable, same reasoning
as pinning the server image itself:

```bash
docker run --rm -it \
    -e BASE_URL_27B_PR_EVIDENCE="https://qwen-27b-pr-evidence-enterprise-ai.apps.<cluster-domain>/v1" \
    -e OCP_TOKEN="$(oc whoami -t)" \
    -v /tmp/qwen-27b-tokenizer:/vllm-workspace/qwen-27b-tokenizer:ro \
    --entrypoint /bin/bash \
    vllm/vllm-openai:v0.27.1
```

**Low-concurrency** (matches `performance-tests/benchmark-results.md`):
```bash
vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL_27B_PR_EVIDENCE" \
    --endpoint /chat/completions \
    --model qwen-27b-pr-evidence \
    --tokenizer /vllm-workspace/qwen-27b-tokenizer \
    --dataset-name random \
    --num-prompts 20 \
    --random-input-len 256 \
    --random-output-len 128 \
    --request-rate 1 \
    --max-concurrency 1 \
    --temperature 0 \
    --save-result --save-detailed --result-dir ./bench-results \
    --header "Authorization=Bearer $OCP_TOKEN"
```

**High-concurrency / throughput** (matches vllm-project/vllm#23504's convention):
```bash
vllm bench serve \
    --backend openai-chat \
    --base-url "$BASE_URL_27B_PR_EVIDENCE" \
    --endpoint /chat/completions \
    --model qwen-27b-pr-evidence \
    --tokenizer /vllm-workspace/qwen-27b-tokenizer \
    --dataset-name random \
    --num-prompts 500 \
    --random-input-len 256 \
    --random-output-len 128 \
    --request-rate inf \
    --max-concurrency 64 \
    --temperature 0 \
    --save-result --save-detailed --result-dir ./bench-results \
    --header "Authorization=Bearer $OCP_TOKEN"
```
(Scaled down from #23504's `--num-prompts 4096 --max-concurrency 2048`, which
targeted DeepSeek-V3 on a much larger multi-node cluster -- 500/64 is sized for
a single TP=2 L40S pair.)
