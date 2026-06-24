---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Inference Gateway
---

## TL;DR

- **Gateway mode** routes inference requests through the [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) (GAIE) instead of Dynamo's built-in frontend router.
- **Dynamo EPP** (Endpoint Picker Plugin) runs at the gateway layer and performs token-aware, KV-cache-aware load balancing across your model workers.
- **Choose agentgateway** unless your cluster already standardizes on Istio as a Gateway API implementation. Both providers are fully supported.

## How It Works

```
HTTP client
 → Gateway (agentgateway or Istio — implements Gateway API)
 → EPP (Dynamo Endpoint Picker Plugin — KV-aware routing via ext_proc)
 → InferencePool → Workers (vLLM / SGLang / TRT-LLM)
```

GAIE extends the Kubernetes Gateway API with two new resource types — `InferencePool` and `InferenceModel` — and a standard plugin interface (ext_proc) that lets a custom EPP intercept requests at the gateway. Dynamo's EPP uses the same token-aware radix-tree KV router that the built-in frontend uses, so you get identical routing quality regardless of which gateway implementation you choose.

### Key Concepts

| Concept | What It Is |
|---------|------------|
| **Gateway API** | Kubernetes SIG-Network standard for expressing L4/L7 routing (`GatewayClass`, `Gateway`, `HTTPRoute`). |
| **GAIE** | Gateway API Inference Extension — adds `InferencePool`, `InferenceModel`, and the EPP plugin interface on top of Gateway API. |
| **EPP** | Endpoint Picker Plugin — the sidecar/deployment that intercepts requests via gRPC ext_proc and selects the best worker. Dynamo ships its own EPP with KV-aware routing. |
| **GatewayClass** | Cluster-scoped resource naming the gateway implementation (`agentgateway` or `istio`). |
| **Gateway** | Namespaced resource that instantiates a proxy bound to a `GatewayClass`. One `Gateway` per namespace is typical. |
| **HTTPRoute** | Routes HTTP traffic from a `Gateway` to an `InferencePool`. One per model deployment. |
| **InferencePool** | GAIE resource that groups model-server pods and wires them to the EPP. Created by the Dynamo operator from a `DynamoGraphDeployment`. |
| **DynamoGraphDeployment (DGD)** | Dynamo operator CRD that declares your model graph. The operator creates the `InferencePool`, EPP `Deployment`, and worker pods from it. |

### Agentgateway vs Istio

| | Agentgateway | Istio |
|-|---|---|
| **Good fit** | New clusters, no existing mesh requirement | Clusters already running Istio |
| **Install footprint** | Small: two Helm charts in `agentgateway-system` | Larger: full Istio control plane |
| **Istio sidecar conflict** | Needs `AgentgatewayParameters` annotation (handled by `install.sh`) | Native; no extra annotation needed |
| **Inference extension support** | Native; flag on by default | Requires `ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true` |
| **GatewayClass name** | `agentgateway` | `istio` |

> For detailed provider-specific installation, see
> [`deploy/inference-gateway/gateways/agentgateway/agentgateway.md`](../../deploy/inference-gateway/gateways/agentgateway/agentgateway.md)
> and
> [`deploy/inference-gateway/gateways/istio/istio.md`](../../deploy/inference-gateway/gateways/istio/istio.md).

## Prerequisites

- Kubernetes cluster with kubectl configured
- NVIDIA GPU drivers installed on worker nodes

## Installation Steps

### 1. Install Dynamo Platform

[See Quickstart Guide](./README.md) to install Dynamo Kubernetes Platform.
If you are installing from the source tree rather than a release chart, follow [Advanced: Build from Source](./installation-guide.md#advanced-build-from-source) and run `helm dep build ./platform/` before `helm install` so the vendored subcharts match the local chart contents.

### 2. Deploy Inference Gateway

Run the one-shot installer for your chosen provider. Both scripts install the Gateway API CRDs and GAIE CRDs, then install the gateway implementation and create an `inference-gateway` `Gateway` in `${NAMESPACE}`.

<Tabs>
<Tab title="Agentgateway">

```bash
cd deploy/inference-gateway
export NAMESPACE=my-model   # namespace for Gateway, HTTPRoute, and DynamoGraphDeployment
./gateways/agentgateway/install.sh
```

The script installs:
- Gateway API CRDs
- GAIE CRDs
- agentgateway control plane (CRDs + controller) into `agentgateway-system`
- `AgentgatewayParameters` resource that excludes Istio sidecar injection from `agentgateway-proxy` pods
- `Gateway` named `inference-gateway` in `${NAMESPACE}`

| Variable | Default | Description |
|----------|---------|-------------|
| `AGW_NAMESPACE` | `agentgateway-system` | Namespace for the agentgateway control plane. |
| `GATEWAY_API_VERSION` | `v1.5.1` | Gateway API release to install. |
| `IGW_LATEST_RELEASE` | `v1.2.1` | GAIE release to install. |
| `AGW_VERSION` | `v1.0.0` | agentgateway Helm chart version. |

For full manual steps see [agentgateway.md](../../deploy/inference-gateway/gateways/agentgateway/agentgateway.md).

</Tab>
<Tab title="Istio">

```bash
cd deploy/inference-gateway
export NAMESPACE=my-model   # namespace for Gateway, HTTPRoute, and DynamoGraphDeployment
./gateways/istio/install.sh
```

The script installs:
- Gateway API CRDs
- GAIE CRDs
- Istio (via `istioctl`) with `ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true`
- `Gateway` named `inference-gateway` in `${NAMESPACE}`

| Variable | Default | Description |
|----------|---------|-------------|
| `ISTIO_NAMESPACE` | `istio-system` | Namespace for the Istio control plane. |
| `GATEWAY_API_VERSION` | `v1.5.1` | Gateway API release to install. |
| `IGW_LATEST_RELEASE` | `v1.2.1` | GAIE release to install. |
| `ISTIO_VERSION` | `1.29.2` | Istio version (used when `istioctl` is not already on `PATH`). |

For full manual steps see [istio.md](../../deploy/inference-gateway/gateways/istio/istio.md).

</Tab>
</Tabs>

Verify the `Gateway` is programmed before proceeding:

```bash
kubectl get gateway inference-gateway -n ${NAMESPACE}

# Expected output
# NAME                CLASS          ADDRESS       PROGRAMMED   AGE
# inference-gateway   agentgateway   10.xx.xx.xx   True         30s
```

### 3. Setup Secrets

Do not forget docker registry secret if needed.

```bash
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=$DOCKER_SERVER \
  --docker-username=$DOCKER_USERNAME \
  --docker-password=$DOCKER_PASSWORD \
  --namespace=$NAMESPACE
```

Do not forget to include the HuggingFace token.

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

### 4. Build EPP Image (Optional)

You can either use the provided Dynamo FrontEnd image for the EPP image or build your own:

```bash
export DOCKER_SERVER=ghcr.io/nvidia/dynamo   # Container registry
export IMAGE_TAG=YOUR-TAG
cd deploy/inference-gateway/epp
make all        # Build Dynamo lib + Docker image + load locally
# make all-push to also push to registry
```

| Target | Description |
|--------|-------------|
| `make dynamo-lib` | Build Dynamo static library and copy to project |
| `make all` | Build Dynamo lib + Docker image + load locally |
| `make all-push` | Build Dynamo lib + Docker image + push to registry |

### 5. Deploy

Deploy the `DynamoGraphDeployment` and `HTTPRoute`. The `HTTPRoute` resolves the `Gateway` in the same namespace by default. If your `Gateway` lives in a different namespace, add `parentRefs[].namespace`:

```yaml
parentRefs:
  - group: gateway.networking.k8s.io
    kind: Gateway
    name: inference-gateway
    namespace: my-model   # only needed if the Gateway is in a different namespace
```

```bash
cd <dynamo-source-root>
# Choose disagg or agg example
kubectl apply -f examples/backends/vllm/deploy/gaie/disagg.yaml -n my-model
# or
kubectl apply -f examples/backends/vllm/deploy/gaie/agg.yaml -n my-model
# Apply the HTTPRoute
kubectl apply -f examples/backends/vllm/deploy/gaie/http-route.yaml -n my-model
```

Examples for other models are under `recipes/llama-3-70b/`:

```bash
# Update storageClassName in model-cache.yaml to match your cluster first
kubectl apply -f recipes/llama-3-70b/model-cache/model-cache.yaml   -n ${NAMESPACE}
kubectl apply -f recipes/llama-3-70b/model-cache/model-download.yaml -n ${NAMESPACE}

# Aggregated
kubectl apply -f recipes/llama-3-70b/vllm/agg/gaie/deploy.yaml    -n ${NAMESPACE}
kubectl apply -f recipes/llama-3-70b/vllm/agg/gaie/http-route.yaml -n ${NAMESPACE}

# Disaggregated
kubectl apply -f recipes/llama-3-70b/vllm/disagg-single-node/gaie/deploy.yaml    -n ${NAMESPACE}
kubectl apply -f recipes/llama-3-70b/vllm/disagg-single-node/gaie/http-route.yaml -n ${NAMESPACE}
```

Key deployment notes:

- When using GAIE the FrontEnd does not choose the workers. Routing is determined by the EPP.
- The FrontEnd must run with `--router-mode direct` so it respects the EPP's routing decisions passed via request headers.
- Use `frontendSidecar` on a worker service to have the operator inject a fully configured frontend sidecar:

```yaml
frontendSidecar:
  image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
  args:
    - --router-mode
    - direct
  envFromSecret: hf-token-secret
```

**Startup Probe Timeout:** The EPP has a default startup probe timeout of 30 minutes (10s × 180 failures). To allow 60 minutes:

```yaml
extraPodSpec:
  mainContainer:
    startupProbe:
      failureThreshold: 360   # 10s × 360 = 60 minutes
```

## Configuration

### Enabling KV-Aware Routing (default)

KV-aware routing uses live KV cache block events from workers so the EPP routes to the worker with the best prefix cache overlap.

1. **Workers — enable prefix caching and KV event publishing.**
   - **vLLM:** Pass `--enable-prefix-caching` and `--kv-events-config '{"enable_kv_cache_events":true}'`.
   - **SGLang:** Pass `--kv-events-config` with the appropriate endpoint.
   - **TRT-LLM:** Pass `--publish-events-and-metrics`.
2. **EPP — leave `DYN_USE_KV_EVENTS` at its default (`true`).** The EPP subscribes to worker KV events via the event plane (NATS/ZMQ).
3. **Block size — must be consistent.** `--block-size` on all workers must match `DYN_KV_CACHE_BLOCK_SIZE` on the EPP (default: 128).

### Disabling KV-Aware Routing

To fall back to approximate load-balanced routing:

1. **EPP:** Set `DYN_USE_KV_EVENTS=false`. The router tracks state locally with TTL decay instead of live KV events.
2. **Workers:** Pass `--no-enable-prefix-caching`.
3. **Optionally** set `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT=0` to skip prefix-overlap scoring and route by load only.

### EPP Routing Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_USE_KV_EVENTS` | `true` | Subscribe to live KV cache events from workers. |
| `DYN_KV_CACHE_BLOCK_SIZE` | `128` | Must match `--block-size` on all workers. |
| `DYN_BUSY_THRESHOLD` | (disabled) | Max allowed worker load before the router skips it. |
| `DYN_ENFORCE_DISAGG` | `false` | `true` = fail if no prefill workers; `false` = fall back to aggregated. |
| `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT` | `1.0` | Prefix-overlap credit multiplier (0–1). Lower = less bias toward prefix reuse. |
| `DYN_ROUTER_PREFILL_LOAD_SCALE` | `1.0` | Scale prompt-side prefill load before decode blocks are added. |
| `DYN_ROUTER_TEMPERATURE` | `0.0` | Sampling temperature for worker selection. `0` = deterministic top pick. |
| `DYN_ROUTER_REPLICA_SYNC` | `false` | Enable replica synchronization. |
| `DYN_ROUTER_TRACK_ACTIVE_BLOCKS` | `true` | Track active blocks. |
| `DYN_ROUTER_TRACK_OUTPUT_BLOCKS` | `false` | Track output blocks during generation. |

See the [KV cache routing design](../design-docs/router-design.md) for details.

### Service Mesh Integration (Istio)

When running under Istio as a **service mesh** (separate from using Istio as the gateway implementation), the mesh sidecar proxy may conflict with the EPP's TLS endpoint. Enable the automatic `DestinationRule` via the Dynamo platform Helm chart:

```bash
helm install dynamo deploy/helm/charts/platform \
  --set dynamo.serviceMesh.enabled=true
```

Or in a values file:

```yaml
dynamo:
  serviceMesh:
    enabled: true
    provider: "istio"
    istio:
      tlsMode: "SIMPLE"
      insecureSkipVerify: true
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dynamo.serviceMesh.enabled` | bool | `false` | Enable automatic `DestinationRule` generation for EPP services. |
| `dynamo.serviceMesh.provider` | string | `"istio"` | Service mesh provider. Only `"istio"` is supported. |
| `dynamo.serviceMesh.istio.tlsMode` | string | `"SIMPLE"` | TLS mode. Supported: `DISABLE`, `SIMPLE`, `MUTUAL`, `ISTIO_MUTUAL`. |
| `dynamo.serviceMesh.istio.insecureSkipVerify` | bool | `true` | Skip TLS cert verification (required when EPP uses self-signed certs). |

> [!NOTE]
> Istio CRDs (`networking.istio.io`) must be installed before enabling this feature. The operator detects Istio at startup and skips `DestinationRule` reconciliation if CRDs are absent.

When enabled, the operator produces a `DestinationRule` for each EPP service:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: <epp-service-name>
spec:
  host: <epp-service-name>.<namespace>.svc.cluster.local
  trafficPolicy:
    tls:
      mode: SIMPLE
      insecureSkipVerify: true
```

**Agentgateway + Istio sidecar conflict:** When `istio-injection=enabled` on the namespace, the Istio sidecar intercepts the `ext_proc` gRPC connection from `agentgateway-proxy` to EPP (port 9002), causing HTTP 500 responses. The agentgateway installer handles this automatically via `AgentgatewayParameters`. See [agentgateway.md § Step 3](../../deploy/inference-gateway/gateways/agentgateway/agentgateway.md#step-3-create-agentgatewayparameters-excludes-istio-sidecar-injection) for details and the manual Option B (cluster-wide patch).

## Verify Installation

```bash
kubectl get inferencepool -n ${NAMESPACE}
kubectl get httproute    -n ${NAMESPACE}
kubectl get service      -n ${NAMESPACE}
kubectl get gateway      -n ${NAMESPACE}
```

Sample output:

```
# kubectl get inferencepool
NAME        AGE
qwen-pool   33m

# kubectl get httproute
NAME        HOSTNAMES   AGE
qwen-route               33m
```

## Usage

### 1. Expose the Gateway

<Tabs>
<Tab title="Minikube">

```bash
# Terminal 1
minikube tunnel

# Terminal 2
GATEWAY_URL=$(kubectl get svc inference-gateway -n my-model -o jsonpath='{.spec.clusterIP}')
echo $GATEWAY_URL
```

</Tab>
<Tab title="Cluster (port-forward)">

```bash
# Terminal 1
kubectl port-forward svc/inference-gateway 8000:80 -n ${NAMESPACE}

# Terminal 2
GATEWAY_URL=http://localhost:8000
```

</Tab>
</Tabs>

### 2. Query Models

```bash
curl $GATEWAY_URL/v1/models | jq .
```

Sample output:

```json
{
  "data": [
    {
      "created": 1753768323,
      "id": "Qwen/Qwen3-0.6B",
      "object": "object",
      "owned_by": "nvidia"
    }
  ],
  "object": "list"
}
```

### 3. Send an Inference Request

```bash
MODEL_NAME="Qwen/Qwen3-0.6B"
curl $GATEWAY_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "'"${MODEL_NAME}"'",
      "messages": [{"role": "user", "content": "Hello, how are you?"}],
      "stream": false,
      "max_tokens": 30,
      "temperature": 0.0
    }'
```

If you have more than one `HTTPRoute` running, add a `Host` header and a `spec.hostnames` entry to your route:

```bash
curl -H "Host: llama3-70b-disagg.example.com" $GATEWAY_URL/v1/models | jq .
```

```yaml
spec:
  hostnames:
    - llama3-70b-disagg.example.com
```

## Cleanup

<Tabs>
<Tab title="Agentgateway">

```bash
kubectl delete dynamoGraphDeployment <your-dgd-name> -n ${NAMESPACE}
kubectl delete gateway inference-gateway                -n ${NAMESPACE}
kubectl delete agentgatewayparameters inference-gateway-params -n ${NAMESPACE}

AGW_NAMESPACE=${AGW_NAMESPACE:-agentgateway-system}
helm uninstall agentgateway      -n ${AGW_NAMESPACE}
helm uninstall agentgateway-crds -n ${AGW_NAMESPACE}
kubectl delete namespace ${AGW_NAMESPACE} --ignore-not-found

IGW_LATEST_RELEASE=${IGW_LATEST_RELEASE:-v1.2.1}
GATEWAY_API_VERSION=${GATEWAY_API_VERSION:-v1.5.1}
kubectl delete -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml" --ignore-not-found
kubectl delete -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml" --ignore-not-found
```

</Tab>
<Tab title="Istio">

```bash
kubectl delete dynamoGraphDeployment <your-dgd-name> -n ${NAMESPACE}
kubectl delete gateway inference-gateway               -n ${NAMESPACE}

ISTIO_NAMESPACE=${ISTIO_NAMESPACE:-istio-system}
IGW_LATEST_RELEASE=${IGW_LATEST_RELEASE:-v1.2.1}
GATEWAY_API_VERSION=${GATEWAY_API_VERSION:-v1.5.1}

istioctl uninstall -y \
  --set values.global.istioNamespace=${ISTIO_NAMESPACE} \
  --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true
kubectl delete namespace ${ISTIO_NAMESPACE} --ignore-not-found
kubectl delete gatewayclass istio istio-remote --ignore-not-found

kubectl delete -f "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml" --ignore-not-found
kubectl delete -f "https://github.com/kubernetes-sigs/gateway-api/releases/download/${GATEWAY_API_VERSION}/standard-install.yaml" --ignore-not-found
```

</Tab>
</Tabs>

## Troubleshooting

<Tabs>
<Tab title="Agentgateway">

**Gateway not showing `PROGRAMMED=True`**

```bash
kubectl describe gateway inference-gateway -n ${NAMESPACE}
kubectl get pods -n ${AGW_NAMESPACE:-agentgateway-system}
kubectl logs -n ${AGW_NAMESPACE:-agentgateway-system} deployment/agentgateway --tail=20
kubectl get gatewayclass agentgateway
```

**HTTPRoute not accepted**

```bash
kubectl describe httproute <route-name> -n ${NAMESPACE}
```

Verify `parentRefs` matches the `Gateway` name and `backendRefs` matches the `InferencePool` name.

**Inference requests return HTTP 500 with Istio installed**

Confirm the `AgentgatewayParameters` resource is present and `agentgateway-proxy` pods are running without an `istio-proxy` sidecar:

```bash
kubectl get agentgatewayparameters -n ${NAMESPACE}
kubectl get pods -n ${NAMESPACE} -l gateway.networking.k8s.io/gateway-name=inference-gateway \
  -o jsonpath='{.items[*].spec.containers[*].name}'
# Expected: agentgateway  (NOT "agentgateway istio-proxy")
```

**No address from LoadBalancer**

```bash
kubectl get gateway inference-gateway -n ${NAMESPACE} \
  -o jsonpath='{.status.addresses[0].value}'
```

If empty, confirm your cluster supports external load balancers or use port-forward to test locally.

</Tab>
<Tab title="Istio">

**Gateway not showing `PROGRAMMED=True`**

```bash
kubectl describe gateway inference-gateway -n ${NAMESPACE}
kubectl get pods -n ${ISTIO_NAMESPACE:-istio-system}
kubectl logs -n ${ISTIO_NAMESPACE:-istio-system} deployment/istiod --tail=20
```

Verify Istio was installed with `ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true`.

**HTTPRoute not accepted**

```bash
kubectl describe httproute <route-name> -n ${NAMESPACE}
```

Verify `parentRefs` matches the `Gateway` name and `backendRefs` matches the `InferencePool` name.

**No address from LoadBalancer**

```bash
kubectl get gateway inference-gateway -n ${NAMESPACE} \
  -o jsonpath='{.status.addresses[0].value}'
```

If empty, confirm your cluster supports external load balancers or use port-forward to test locally.

</Tab>
</Tabs>

## Gateway API Inference Extension Integration Details

This section documents the Dynamo EPP plugin implementation for GAIE.

### Router Bookkeeping

The EPP performs Dynamo router bookkeeping so the FrontEnd's router does not have to sync its state.

### Header Routing Hints

Since GAIE v1.5.0-rc.1, the EPP uses **headers and body mutations** to communicate routing decisions. The plugins set HTTP headers for worker targeting and inject pre-computed token IDs into the request body (`nvext.token_data`) so the frontend sidecar can skip redundant tokenization.

#### Headers Set by Dynamo Plugins

| Header | Description | Set By |
|--------|-------------|--------|
| `x-worker-instance-id` | Primary worker ID (decode worker in disagg mode) | kv-aware-scorer |
| `x-prefill-instance-id` | Prefill worker ID (disaggregated mode only) | kv-aware-scorer |
