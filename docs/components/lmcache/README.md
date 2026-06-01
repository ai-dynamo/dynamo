# Dynamo + LMCache MP — Aggregated Serving on Kubernetes

Kubernetes manifests for running Dynamo aggregated vLLM serving with KV cache
offloaded to a per-node LMCache MP DaemonSet, sharing tensors with the worker
via cross-Pod CUDA IPC.

## Prerequisites

- K8s cluster with the NVIDIA GPU Operator.
- `kubectl` + `helm` with `KUBECONFIG` set.
- The Dynamo platform helm chart must include
  [PR #8414](https://github.com/ai-dynamo/dynamo/pull/8414) — it ships the
  relaxed `sharedMemory.disabled` CRD validation rule that our worker needs to
  opt out of the operator's auto-injected `/dev/shm` emptyDir. No NGC chart
  yet ships with this fix; clone `ai-dynamo/dynamo` and check out commit
  [`e7eb1c565f`](https://github.com/ai-dynamo/dynamo/commit/e7eb1c565f), then
  install from the local chart (see Step 1).

## Step 1 — Install the Dynamo platform

```bash
helm install dynamo-platform oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform \
  --version my-version \
  --namespace dynamo-system --create-namespace \
  --wait
```

> **How it works today** — NGC hasn't published a helm chart with the PR #8414
> CRD fix yet, so the clean `helm install` above doesn't work as-is. Two pieces
> are needed: the **chart** (carries the updated CRD) and the **operator image**
> (whose baked-in CRDs an init container re-applies on every restart). Until NGC
> ships both, the workaround is:
>
> 1. **Chart from local clone**: `git clone https://github.com/ai-dynamo/dynamo.git` and check out
>    commit [`e7eb1c565f`](https://github.com/ai-dynamo/dynamo/commit/e7eb1c565f)
>    (the version validated for this recipe), then install from the local path
>    with the NGC `1.1.1` operator image:
>    ```bash
>    # Required for `helm dependency build` — the platform chart pulls in
>    # NATS and Bitnami subcharts and helm needs to know their repos.
>    helm repo add nats https://nats-io.github.io/k8s/helm/charts/
>    helm repo add bitnami https://charts.bitnami.com/bitnami
>    helm repo update
>
>    cd dynamo/deploy/helm/charts
>    helm dependency build ./platform/
>    helm install dynamo-platform ./platform/ \
>      --namespace dynamo-system --create-namespace \
>      --set "dynamo-operator.controllerManager.manager.image.tag=1.1.1" \
>      --wait
>    ```
> 2. **Patch out the operator's CRD-applying init container** (it would otherwise
>    overwrite the chart's new CRDs on every restart):
>    ```bash
>    kubectl -n dynamo-system patch deployment dynamo-platform-dynamo-operator-controller-manager \
>      --type=json -p='[{"op":"replace","path":"/spec/template/spec/initContainers","value":[]}]'
>    ```
> 3. **Replace the CRDs by hand** (helm doesn't update CRDs on install/upgrade):
>    ```bash
>    kubectl replace -f platform/components/operator/crds/nvidia.com_dynamographdeployments.yaml
>    kubectl replace -f platform/components/operator/crds/nvidia.com_dynamocomponentdeployments.yaml
>    ```
>
> Alternative: build the operator image yourself from a commit that
> includes PR #8414 (so its init container re-applies the **new** CRDs) and
> point the chart at it via `--set ...image.tag=<your-tag>`. Avoids steps 2 and 3, but adds a registry of your own.

Verify the install:
```bash
kubectl get crd | grep nvidia.com
# expect:
#   dynamocomponentdeployments.nvidia.com
#   dynamographdeployments.nvidia.com
```

## Step 2 — Install the LMCache operator

```bash
kubectl apply -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml
```

> Tested with
> the LMCache operator image `lmcache/lmcache-operator:v0.1.1`.

Verify the install:
```bash
kubectl get crd | grep lmcache
# expect:
#   lmcacheengines.lmcache.lmcache.ai
```

## Step 3 — Create the namespace

```bash
kubectl create namespace dynamo-lmcache
```

## Step 4 — Create the HF token Secret

Both `Frontend` and `VllmDecodeWorker` reference `hf-token-secret` via `envFromSecret`.
The Secret must exist or the pods fail to start with `secret "hf-token-secret" not found`.

```bash
# Replace the dummy token below with a real HF token.
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
  namespace: dynamo-lmcache
type: Opaque
stringData:
  HF_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
EOF
```

## Step 5 — Deploy the LMCacheEngine

Replace `my-tag` below with the `lmcache/vllm-openai` image tag you want to run.

```bash
kubectl apply -f - <<'EOF'
apiVersion: lmcache.lmcache.ai/v1alpha1
kind: LMCacheEngine
metadata:
  name: lmcache-mp
  namespace: dynamo-lmcache
spec:
  image:
    repository: lmcache/vllm-openai
    tag: my-tag
    pullPolicy: IfNotPresent
  # L1 (CPU RAM) cache size — bump for production workloads.
  l1:
    sizeGB: 16
EOF
```

> Validated `my-tag` against `nightly-2026-04-25` (lmcache
> `0.4.5.dev31`). This pre-stable build is wire-compatible with the
> `vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3` worker image (which ships
> lmcache `0.4.4`).

Verify with:
```bash
kubectl -n dynamo-lmcache get lmcacheengine lmcache-mp
# expect: STATUS  Running
```

## Step 6 — Deploy the Dynamo worker

Edit `agg_lmcache.yaml`: replace `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag` (on
both `Frontend` and `VllmDecodeWorker`) with your Dynamo vllm-runtime image.

```bash
kubectl apply -n dynamo-lmcache -f examples/backends/vllm/deploy/agg_lmcache.yaml
```

> Validated `my-tag` against `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0-deepseek-v4-cuda13-dev.3`
> (vLLM `0.20.1`, lmcache `0.4.4`). Pair with the LMCacheEngine pin from
> Step 5.

Verify with:
```bash
kubectl -n dynamo-lmcache get pods -l nvidia.com/dynamo-component-type=worker
# expect: READY 1/1, STATUS Running
```

## Step 7 — Verify

Send the same long prompt twice:

```bash
kubectl -n dynamo-lmcache port-forward svc/vllm-agg-lmcache-frontend 8000:8000 >/dev/null &
trap 'kill %1' EXIT
sleep 4

PROMPT=$(python3 -c "print('the quick brown fox jumps over the lazy dog '*60)")
REQ="{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":5}"

for label in cold warm; do
  echo "--- $label ---"
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" -d "$REQ" \
    | python3 -m json.tool | grep -E "prompt_tokens|cached_tokens"
done
```

Then check LMCache server metrics:

```bash
LMC=$(kubectl -n dynamo-lmcache get pod -l app.kubernetes.io/instance=lmcache-mp -o name | head -1)
kubectl -n dynamo-lmcache exec "$LMC" -- curl -s localhost:9090/metrics | grep '^lmcache_mp_'
```

Expected: warm response shows `cached_tokens > 0`, and
`lmcache_mp_lookup_hit_tokens_total > 0`.

### Metrics endpoint by lmcache version

The LMCache operator declares three container ports on every DaemonSet pod:
`5555` (ZMQ), `8080` (control HTTP), `9090` (Prometheus). It also creates two
Services: `lmcache-mp` (5555 + 8080) and a headless `lmcache-mp-metrics` (9090).
Which port actually serves `/metrics` depends on the image:

| lmcache version | `/metrics` on `:9090` | `/metrics` on `:8080` | Metric labels |
|---|---|---|---|
| `nightly-2026-04-25` (lmcache `0.4.5.dev31`) | ✅ | 404 | bare counters, no labels |
| v0.4.5 stable (released 2026-05-15) and later | ✅ | ✅ | labeled with `{cache_salt, model_name}` |

## Cleanup

```bash
kubectl delete -n dynamo-lmcache -f examples/backends/vllm/deploy/agg_lmcache.yaml
kubectl -n dynamo-lmcache delete lmcacheengine lmcache-mp
kubectl delete namespace dynamo-lmcache
kubectl delete -f https://github.com/LMCache/LMCache/releases/download/operator-latest/install.yaml
helm uninstall dynamo-platform -n dynamo-system
kubectl delete namespace dynamo-system lmcache-operator-system
```
