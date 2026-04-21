---
name: quickstart
description: Deploy a Dynamo recipe end-to-end -- recipe pick, namespace + HF secret, storage class, model cache, deploy, ready-wait, port-forward, smoke test
user-invocable: true
disable-model-invocation: true
---

# Dynamo Quickstart

Walk a user through deploying a Dynamo recipe end-to-end. Mirrors the canonical flow in [`recipes/README.md`](../../../recipes/README.md), but fills in the choices the README leaves to the user.

Optional argument: a recipe path (e.g. `llama-3-70b/vllm/agg`). If omitted, ask in Step 1.

## Step 1: Pick a recipe

Read [`recipes/README.md`](../../../recipes/README.md) and present the recipe table to the user filtered by what they have:

- **GPU type** (H100 / H200 / A100 / B200 / GB200) -- `kubectl describe node -l nvidia.com/gpu.product 2>/dev/null | grep -E 'gpu.product|Allocatable' | head -20`
- **Framework preference** (vLLM / SGLang / TensorRT-LLM)
- **Mode** (aggregated / disaggregated)

Resolve to a path of the form `<model>/<framework>/<mode>/`, e.g. `llama-3-70b/vllm/agg`. Confirm with the user before proceeding.

## Step 2: Set namespace and HuggingFace secret

```bash
export NAMESPACE=${NAMESPACE:-dynamo-demo}
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# HF token (required for gated models like Llama, Kimi, Qwen-VL)
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n "$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -
```

If `HF_TOKEN` is unset, prompt the user for it. If the model is not gated (e.g. `nemotron-3-super-fp8`), the secret is still created but unused -- harmless.

## Step 3: Verify the cluster is ready

Invoke the `verify-cluster` skill (or run its checks inline). The single most common failure here is `storageClassName: standard` not existing on the cluster. Auto-fix:

```bash
RECIPE=<recipe-path>  # e.g. llama-3-70b/vllm/agg
DEFAULT_SC=$(kubectl get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}')
echo "Default storage class: ${DEFAULT_SC:-<none -- abort and ask user>}"

# Patch model-cache.yaml in place if the default class differs from "standard"
yq -i "(.. | select(has(\"storageClassName\")).storageClassName) = \"$DEFAULT_SC\"" \
  recipes/$(echo "$RECIPE" | cut -d/ -f1)/model-cache/model-cache.yaml
```

## Step 4: Apply model-cache and wait for the download job

```bash
MODEL=$(echo "$RECIPE" | cut -d/ -f1)
kubectl apply -f recipes/$MODEL/model-cache/ -n "$NAMESPACE"
kubectl wait --for=condition=Complete job/model-download -n "$NAMESPACE" --timeout=6000s
```

The download can take 10-60 minutes depending on model size. Tail the logs to surface progress:

```bash
kubectl logs -f job/model-download -n "$NAMESPACE"
```

## Step 5: Patch the image tag and apply the deploy

The recipe `deploy.yaml` ships with a placeholder image. Pick the correct tag:

- **Released images** are at `nvcr.io/nvidia/ai-dynamo/{vllm,sglang,tensorrtllm,mocker}-runtime:<tag>`. Latest stable tag is in [`docs/reference/release-artifacts.md`](../../../docs/reference/release-artifacts.md). Match the tag to the cluster's GPU driver via the [Feature Support Matrix](../../../docs/backends/trtllm/README.md#feature-support-matrix).
- **Top-of-tree paths** (e.g. Kimi-k2.5, GLM-5-NVFP4) require a custom build. Run the `build-image` flow inline:
  ```bash
  ./container/build.sh --framework vllm --tag my-tag
  docker push <your-registry>/ai-dynamo/vllm-runtime:my-tag
  ```
  Then patch:
  ```bash
  yq -i "(.spec.services[].extraPodSpec.mainContainer.image) |= sub(\"nvcr\\.io/nvidia/ai-dynamo/(.+):.*\", \"<your-registry>/ai-dynamo/\\$1:my-tag\")" \
    recipes/$RECIPE/deploy.yaml
  ```

Apply:

```bash
kubectl apply -f recipes/$RECIPE/deploy.yaml -n "$NAMESPACE"
```

## Step 6: Wait for the deployment to be ready

```bash
DEPLOY_NAME=$(kubectl get dynamographdeployment -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
kubectl wait --for=condition=ready pod \
  -l nvidia.com/dynamo-graph-deployment-name="$DEPLOY_NAME" \
  -n "$NAMESPACE" --timeout=600s
```

If the wait times out, hand off to the `inspect-pods` skill to diagnose.

## Step 7: Port-forward and smoke test

```bash
kubectl port-forward svc/${DEPLOY_NAME}-frontend 8000:8000 -n "$NAMESPACE" &
PF_PID=$!
sleep 3

# Confirm the model is listed
curl -s http://localhost:8000/v1/models | jq '.data[].id'

# Send a known-good request
SERVED_MODEL=$(curl -s http://localhost:8000/v1/models | jq -r '.data[0].id')
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$SERVED_MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the single word: pong\"}],
    \"max_tokens\": 10
  }" | jq

kill $PF_PID 2>/dev/null
```

Success criterion: HTTP 200 with a non-empty completion. If the smoke test fails, hand off to `troubleshoot`.

## Optional: Enable observability

Pre-deploy, set the OTel env on the frontend service in `deploy.yaml`. Or post-deploy:

```bash
kubectl set env deployment/${DEPLOY_NAME}-frontend -n "$NAMESPACE" \
  OTEL_EXPORT_ENABLED=true \
  OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://tempo:4317 \
  OTEL_EXPORTER_OTLP_LOGS_ENDPOINT=http://loki-otlp:4317 \
  DYN_SYSTEM_PORT=8081
```

Adjust endpoints to match your cluster's Tempo/Loki install.

## Hand-offs

- Cluster preflight failed in Step 3 -> [`verify-cluster`](../verify-cluster/SKILL.md)
- Pods unhealthy in Step 6 -> [`inspect-pods`](../inspect-pods/SKILL.md)
- Smoke test failed in Step 7 with a known symptom -> [`troubleshoot`](../troubleshoot/SKILL.md)
