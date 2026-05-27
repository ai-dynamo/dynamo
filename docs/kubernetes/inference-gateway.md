---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Inference Gateway (GAIE)
---

## Inference Gateway Setup with Dynamo

# Inference Gateway (GAIE)

Integrate Dynamo with the Gateway API Inference Extension for intelligent KV-aware request routing at the gateway layer.

## Features

- EPP's default kv-routing approach is not token-aware because the prompt is not tokenized. But the Dynamo plugin uses a token-aware KV algorithm. It employs the dynamo router which implements kv routing by running your model's tokenizer inline. The EPP plugin configuration lives in [`helm/dynamo-gaie/epp-config-dynamo.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/inference-gateway/standalone/helm/dynamo-gaie/epp-config-dynamo.yaml), following the checked-in GAIE/EPP configuration layout used by this repository.

- Dynamo Integration with the Inference Gateway supports Aggregated and Disaggregated Serving. A request only exercises disaggregated routing when the EPP config defines a `prefill` profile and prefill workers are available. The standalone [`epp-config-dynamo.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/inference-gateway/standalone/helm/dynamo-gaie/epp-config-dynamo.yaml) currently only defines a `decode` profile, while the recipe examples use separate aggregated and disaggregated configs under `recipes/llama-3-70b/vllm/agg/gaie/` and `recipes/llama-3-70b/vllm/disagg-single-node/gaie/`. Unless `DYN_ENFORCE_DISAGG=true`, deployments without a `prefill` profile or prefill workers fall back to aggregated serving.

- GAIE integration supports Data Parallelism.

- If you want to use LoRA deploy Dynamo without the Inference Gateway.

- These setups use [agentgateway](https://agentgateway.dev/) as the Inference Gateway implementation.

## Prerequisites

- Kubernetes cluster with kubectl configured
- NVIDIA GPU drivers installed on worker nodes

## Installation Steps

### 1. Install Dynamo Platform ###

[See Quickstart Guide](./README.md) to install Dynamo Kubernetes Platform.
If you are installing from the source tree rather than a release chart, follow [Path B: Custom Build from Source](./installation-guide.md#path-b-custom-build-from-source) and run `helm dep build ./platform/` before `helm install` so the vendored subcharts match the local chart contents.

### 2. Deploy Inference Gateway ###

First, deploy an inference gateway service. In this example, we'll install agentgateway with the inference extension enabled.

```bash
cd deploy/inference-gateway
export NAMESPACE=my-model # You can put the inference gateway into another namespace and then adjust your http-route.yaml
./scripts/install_gaie_crd_agentgateway.sh
```
This script installs the Gateway API CRDs, the GAIE CRDs, agentgateway into `agentgateway-system`, and a `Gateway` named `inference-gateway` into `${NAMESPACE}`.

#### f. Verify the Gateway is running

```bash
kubectl get gateway inference-gateway -n ${NAMESPACE}

# Sample output
# NAME                CLASS      ADDRESS   PROGRAMMED   AGE
# inference-gateway   agentgateway   <none>   True         1m
```


### 3. Setup secrets ###

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

### 4. Build EPP image (Optional)

You can either use the provided Dynamo FrontEnd image for the EPP image or you need to build your own Dynamo EPP custom image following the steps below.

```bash
# export env vars
export DOCKER_SERVER=ghcr.io/nvidia/dynamo	# Container registry
export IMAGE_TAG=YOUR-TAG # Or auto from git tag
cd deploy/inference-gateway/epp
make all # Do everything in one command
# or make all-push to also push


# Or step-by-step
make dynamo-lib # Build Dynamo library and copy to project
make image-load # Build Docker image and load locally
make image-push # Build and push to registry
make info # Check image tag
```

#### All-in-one Targets

| Target | Description |
|--------|-------------|
| `make dynamo-lib` | Build Dynamo static library and copy to project |
| `make all` | Build Dynamo lib + Docker image + load locally |
| `make all-push` | Build Dynamo lib + Docker image + push to registry |

### 4b. Build Rust EPP image (Optional — experimental)

A pure-Rust EPP implementation is available as an alternative to the Go-based EPP.
It replaces the Go EPP + CGO bridge with a single native Rust binary that implements
the Envoy ext_proc gRPC service and uses Dynamo's KV-aware router directly — no FFI
boundary, no Go runtime.

```bash
cd deploy/inference-gateway/ext-proc

# Build and load Docker image locally
make image-load
# Creates: dynamo/dynamo-rust-epp:<git-tag>

# Or build and push to a registry
export DOCKER_SERVER=ghcr.io/nvidia/dynamo
make image-push
```

To build the binary locally without Docker:

```bash
cd deploy/inference-gateway/ext-proc
make build
# Binary at: <repo-root>/target/release/dynamo-ext-proc
```

#### Rust EPP Makefile Targets

| Target | Description |
|--------|-------------|
| `make build` | Build the Rust EPP binary locally via cargo |
| `make image-load` | Build Docker image and load locally |
| `make image-push` | Build and push Docker image to registry |
| `make image-kind` | Build and load into a kind cluster |
| `make image-multiarch-push` | Build and push multi-arch image |
| `make fmt` / `make clippy` / `make test` | Development checks |
| `make info` | Show image tag and build configuration |

#### Rust EPP Configuration

The Rust EPP uses the same environment variables as the Go EPP for namespace
resolution and router configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `DYN_NAMESPACE_PREFIX` | *(unset)* | Dynamo discovery namespace (highest priority) |
| `DYN_NAMESPACE` | `vllm-agg` | Dynamo discovery namespace (fallback) |
| `DYN_COMPONENT_NAME` | `backend` | Dynamo component name |
| `DYN_ENFORCE_DISAGG` | `false` | Enforce disaggregated prefill/decode routing |
| `RUST_LOG` | `info` | Tracing log level filter |

The gRPC port is hardcoded to `9002` (matching the operator's `EPPGRPCPort` constant).

Namespace resolution follows the same logic as the Go EPP plugin:
`DYN_NAMESPACE_PREFIX` > `DYN_NAMESPACE` > `"vllm-agg"` (default).

The Rust EPP also respects the standard Dynamo router environment variables
(`DYN_OVERLAP_SCORE_WEIGHT`, `DYN_ROUTER_TEMPERATURE`, `DYN_USE_KV_EVENTS`, etc.)
documented in the Configuration section below.

> [!NOTE]
> The Rust EPP is experimental. It uses Dynamo's native discovery system
> (`DistributedRuntime`) instead of the GAIE Kubernetes controllers, so it
> does not require `InferencePool` or `InferenceModel` CRDs for endpoint
> discovery. It discovers workers through Dynamo's own registration mechanism.

#### `InferencePool` and the data plane (Istio, kGateway, Agentgateway)

Although the Rust EPP does not consult `InferencePool` for worker discovery,
the CRD is still required by the gateway **data plane**. Gateway
implementations (Istio, kGateway, Agentgateway) read `InferencePool` to:

1. Attach the `ext_proc` filter pointing at the EPP service.
2. Enable the `override_host` LB policy so the EPP's
   `x-gateway-destination-endpoint` header / dynamic-metadata is honored.
3. Scope which pods are eligible to receive traffic — the pool's selector
   becomes the `envoy.lb.subset_hint` metadata that the EPP intersects with
   its own discovered workers before picking one.

The Dynamo operator **auto-generates the `InferencePool`** for every
`DynamoGraphDeployment` ([`deploy/operator/internal/dynamo/epp/inference_pool.go`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/operator/internal/dynamo/epp/inference_pool.go)).
Its `Selector` matches the operator's worker-pod labels and its
`EndpointPickerRef` points at the EPP service on `9002`, so Dynamo's
discovery and the pool's pod set stay in sync automatically — users do not
hand-craft the pool.

**Using Istio instead of kGateway:**

- The only Istio-specific step is creating an Istio `Gateway` / `HTTPRoute`
  that references the operator-generated `InferencePool` as its `backendRef`.
  The DGD, the generated pool, and the Rust EPP image are all unchanged.
- The operator targets the stable `inference.networking.k8s.io/v1` API group,
  supported in Istio ≥ 1.27. Older Istio versions used the experimental
  `inference.networking.x-k8s.io` group and are not compatible.
- **mTLS to the EPP.** Istio expects mTLS between the gateway and the EPP
  service. The Rust EPP serves self-signed TLS on `9002` by default
  (`DYN_SECURE_SERVING=true`). See *Service Mesh Integration (Istio)* below
  for the `DestinationRule` the Dynamo Helm chart can generate so Istio
  terminates the EPP's TLS correctly.

> [!IMPORTANT]
> Model card discovery, worker liveness, KV-aware routing, and bookkeeping
> remain entirely in Dynamo's control. The `InferencePool` provides the
> data-plane envelope (which pods, which port, which EPP); Dynamo's
> discovery and the Rust EPP provide the routing intelligence inside that
> envelope. Customizing the pool selector by hand is supported but requires
> keeping it consistent with the operator's worker-pod labels — otherwise
> pods discovered by Dynamo will fail subset filtering and the EPP will
> return `RoutingFailed`.

### 5. Deploy

We recommend deploying Inference Gateway's Endpoint Picker as a Dynamo operator's managed component. Alternatively,
you could deploy it as a standalone pod.
Note that when deploying Dynamo with the Inference Gateway Extension each worker must have the FrontEnd as a sidecar.

#### 5.a. Deploy as a DGD component (recommended)

We provide an example for the Qwen vLLM below.
You have to deploy the Dynamo Graph and the `HTTPRoute`.
The example `http-route.yaml` resolves the `Gateway` in the same namespace as
the `HTTPRoute`, so the simplest path is to apply the route in the same
namespace where you installed the `Gateway` (i.e. `${NAMESPACE}`). If your
`Gateway` lives in a different namespace, add `parentRefs[].namespace` to point
at it explicitly:
```yaml
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: inference-gateway
      namespace: my-model # only needed if the Gateway is in a different namespace
```

```bash
cd <dynamo-source-root>
# kubectl get httproutes -n my-model # Make sure you do not have an incompatible HTTPRoute running, delete if so.
# Choose disagg or agg example
kubectl apply -f examples/backends/vllm/deploy/gaie/disagg.yaml -n my-model
# or
kubectl apply -f examples/backends/vllm/deploy/gaie/agg.yaml -n my-model
# make sure to apply the route
kubectl apply -f examples/backends/vllm/deploy/gaie/http-route.yaml -n my-model
```

Examples for other models can be found in the recipes folder.

```bash
# Deploy PVC, having first Update `storageClassName` in recipes/llama-3-70b/model-cache/model-cache.yaml to match your cluster before deploying
kubectl apply -f recipes/llama-3-70b/model-cache/model-cache.yaml  -n ${NAMESPACE}
kubectl apply -f recipes/llama-3-70b/model-cache/model-download.yaml  -n ${NAMESPACE}
```
We provide examples for llama-3-70b vLLM under the `recipes/llama-3-70b/vllm/agg/gaie/` for aggregated and `recipes/llama-3-70b/vllm/disagg-single-node/gaie/` for disaggregated serving.
Note for the aggregated serving you need to disable DYN_ENFORCE_DISAGG in epp config.
```bash
  - name: DYN_ENFORCE_DISAGG
    value: "false"
```
Use the proper folder in commands below.

```bash
# Deploy your Dynamo Graph.

# agg
kubectl apply -f recipes/llama-3-70b/vllm/agg/gaie/deploy.yaml -n ${NAMESPACE}
# Deploy the GAIE http-route CR. The route resolves the Gateway in the same namespace by default;
# if your Gateway is elsewhere, add parentRefs[].namespace before applying.
kubectl apply -f recipes/llama-3-70b/vllm/agg/gaie/http-route.yaml -n ${NAMESPACE}

# or disagg
kubectl apply -f recipes/llama-3-70b/vllm/disagg-single-node/gaie/deploy.yaml  -n ${NAMESPACE}
kubectl apply -f recipes/llama-3-70b/vllm/disagg-single-node/gaie/http-route.yaml -n ${NAMESPACE}
```

- When using GAIE the FrontEnd does not choose the workers. The routing is determined in the EPP.
- The FrontEnd must run with `--router-mode direct` so that it respects the EPP's routing decisions passed via request headers.
- Use the `frontendSidecar` field on a worker service to have the operator automatically inject a fully configured frontend sidecar container with all required Dynamo env vars, probes, and ports:

```yaml
frontendSidecar:
  image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1
  args:
    - --router-mode
    - direct
  envFromSecret: hf-token-secret
```

- The pre-selected worker (decode and prefill in case of the disaggregated serving) are passed in the request headers.
- The `--router-mode direct` flag ensures the routing respects this selection.

**Startup Probe Timeout:** The EPP has a default startup probe timeout of 30 minutes (10s × 180 failures).
If your model takes longer to load, increase the `failureThreshold` in the EPP's `startupProbe`. For example,
to allow 60 minutes for startup:

```yaml
extraPodSpec:
  mainContainer:
    startupProbe:
      failureThreshold: 360  # 10s × 360 = 60 minutes
```

**Gateway Namespace**
The example `http-route.yaml` resolves the `Gateway` in the same namespace as
the route. If you install the `Gateway` in one namespace and apply the route in
another, add `parentRefs[].namespace: <gateway-namespace>` to `http-route.yaml`.

#### 5.b. Deploy as a standalone pod

We do not recommend this method but there are hints on how to do this here.

##### 5.b.1 Deploy Your Model ###

##### 5.b.2 Install Dynamo GIE helm chart ###

```bash
cd deploy/inference-gateway/standalone

# Export the EPP image - use the Dynamo FrontEnd image or build your own EPP image (see section 4)
export EPP_IMAGE=<the-epp-image>
```
Create a model configuration file similar to the vllm_agg_qwen.yaml for your model.

```bash
helm upgrade --install dynamo-gaie ./helm/dynamo-gaie -n my-model -f ./vllm_agg_qwen.yaml --set-string extension.image=$EPP_IMAGE
```

By default, the Kubernetes discovery mechanism is used. If you prefer etcd, please use the `--set epp.dynamo.useEtcd=true` flag below.

```bash
helm upgrade --install dynamo-gaie ./helm/dynamo-gaie -n my-model -f ./vllm_agg_qwen.yaml --set-string extension.image=$EPP_IMAGE --set epp.dynamo.useEtcd=true
```

Key configurations include:

- An InferenceModel resource for the Qwen model
- A service for the inference gateway
- Required RBAC roles and bindings
- RBAC permissions
- dynamoGraphDeploymentName - the name of the Dynamo Graph where your model is deployed.


**Configuration**
You can configure the plugin by setting environment variables in the EPP component of your DGD in case of the operator-managed installation or in your [values.yaml](https://github.com/ai-dynamo/dynamo/blob/main/deploy/inference-gateway/standalone/helm/dynamo-gaie/values.yaml).

Common Vars for Routing Configuration:

**Enabling KV-Aware Routing (most precise)**

KV-aware routing uses live KV cache block events from workers so the EPP can route requests to the worker with the best prefix cache overlap. To enable it (default):

1. **Workers — enable prefix caching and KV event publishing.** Each worker must publish KV cache events to event plane (NATS/ZMQ) so the EPP's router can track per-worker cache state.
   - **vLLM:** Pass `--enable-prefix-caching` and `--kv-events-config '{"enable_kv_cache_events":true}'`.
   - **SGLang:** Pass `--kv-events-config` with the appropriate endpoint.
   - **TRT-LLM:** Pass `--publish-events-and-metrics`.
2. **EPP — leave `DYN_USE_KV_EVENTS` at its default (`true`).** The EPP subscribes to worker KV events via event plane (NATS/ZMQ) and uses them for prefix-overlap scoring.
3. **Block size — must be consistent.** The `--block-size` on all workers must match `DYN_KV_CACHE_BLOCK_SIZE` on the EPP (default: 128). Mismatched block sizes cause incorrect block hash computation.

**Disabling KV-Aware Routing**

To disable the EPP from listening for KV events (e.g., when prefix caching is off on workers, or for simpler load-balanced routing):

1. **EPP:** Set `DYN_USE_KV_EVENTS=false`. The router falls back to approximate mode (routing decisions are tracked locally with TTL decay instead of live KV events from workers).
2. **Workers:** Pass `--no-enable-prefix-caching` to disable prefix caching entirely. Without prefix caching, no KV events are generated regardless of other flags.
3. **Optionally** set `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT=0` on the EPP to skip prefix-overlap scoring altogether, making the router select workers based on load only.

- Set `DYN_BUSY_THRESHOLD` to configure the upper bound on how "full" a worker can be (often derived from kv_active_blocks or other load metrics) before the router skips it. If the selected worker exceeds this value, routing falls back to the next best candidate. By default the value is negative meaning this is not enabled.
- Set `DYN_ENFORCE_DISAGG=true` (default: `false`) to control per-request behavior when prefill workers are unavailable:
  - **`true` (recommended for disaggregated serving):** Requests fail with an error if prefill workers are not available. Use this when disaggregated serving is required and aggregated fallback is not acceptable.
  - **`false` (default):** Requests gracefully fall back to aggregated mode (skip prefill, route directly to decode) when prefill workers are not available. When prefill workers appear later, subsequent requests automatically use disaggregated routing.
- Set `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT` to control the device-local prefix-overlap credit multiplier, from 0.0 to 1.0. Higher values bias toward reusing workers with similar cached prefixes. (default: 1)
- Set `DYN_ROUTER_PREFILL_LOAD_SCALE` to scale adjusted prompt-side prefill load before decode blocks are added. (default: 1)
- Set `DYN_ROUTER_TEMPERATURE` (default: `0.0`) to soften or sharpen normalized worker sampling. Low temperature makes the router pick the top candidate deterministically; higher temperature lets lower-scoring workers through more often (exploration).
- `DYN_ROUTER_REPLICA_SYNC` — Enable replica synchronization (default: false)
- `DYN_ROUTER_TRACK_ACTIVE_BLOCKS` — Track active blocks (default: true)
- `DYN_ROUTER_TRACK_OUTPUT_BLOCKS` — Track output blocks during generation (default: false)
- See the [KV cache routing design](../design-docs/router-design.md) for details.

Stand-Alone installation only:
- Overwrite the `DYN_NAMESPACE` env var if needed to match your model's dynamo namespace.

**Service Mesh Integration (Istio)**

When running under a service mesh such as Istio, the mesh sidecar proxy may conflict with the EPP's own TLS serving, causing connection failures (double-TLS). To avoid this, the mesh must be told how to connect to the EPP service via an Istio `DestinationRule`.

The Dynamo operator can generate this DestinationRule automatically. Enable it by setting the `dynamo.serviceMesh` parameters when installing or upgrading the Dynamo platform Helm chart:

```bash
helm install dynamo deploy/helm/charts/platform \
  --set dynamo.serviceMesh.enabled=true
```

Or equivalently in a custom values file:

```yaml
dynamo:
  serviceMesh:
    enabled: true
    provider: "istio"
    istio:
      tlsMode: "SIMPLE"
      insecureSkipVerify: true
```

**Helm Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dynamo.serviceMesh.enabled` | bool | `false` | Enable automatic DestinationRule generation for EPP services. |
| `dynamo.serviceMesh.provider` | string | `"istio"` | Service mesh provider. Only `"istio"` is supported. |
| `dynamo.serviceMesh.istio.tlsMode` | string | `"SIMPLE"` | TLS mode for the DestinationRule. Supported values: `DISABLE`, `SIMPLE`, `MUTUAL`, `ISTIO_MUTUAL`. |
| `dynamo.serviceMesh.istio.insecureSkipVerify` | bool | `true` | Skip TLS certificate verification. Set to `true` when EPP uses self-signed certificates (the default). |

> [!NOTE]
> The Istio CRDs (`networking.istio.io`) must be installed on the cluster before enabling this feature. The operator detects Istio availability at startup — if the CRDs are not present, DestinationRule reconciliation is skipped even when `serviceMesh.enabled` is `true`.

When enabled, the operator produces a `DestinationRule` for each EPP service equivalent to:

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

If you are **not** using the Dynamo operator's Helm chart, you must create this `DestinationRule` manually for each EPP service. Without it, Istio's default mTLS policy will conflict with the EPP's gRPC TLS endpoint.

**Inference-gateway Istio sidecar exclusion**

When namespace-level Istio sidecar injection is enabled (`istio-injection=enabled`), the agentgateway-proxy pod also receives an Istio sidecar. This sidecar intercepts the ext_proc gRPC connection from agentgateway-proxy to EPP (port 9002) and routes it through `PassthroughCluster`, which breaks the connection and causes all inference requests to return HTTP 500 with an empty body.

The fix is to tell agentgateway to stamp `sidecar.istio.io/inject: "false"` on the proxy pod template so the Istio webhook skips that pod. EPP and worker pods still receive sidecars normally.

You have two options depending on how you set up the gateway:

***Option A: Per-gateway `AgentgatewayParameters` (recommended)***

This is what `install_gaie_crd_agentgateway.sh` does automatically. It only affects the `inference-gateway` proxy pods and leaves any other agentgateway-managed gateways untouched.

1. Create an `AgentgatewayParameters` resource in **the same namespace as the `inference-gateway` Gateway** (e.g. `dynamo-cloud`). It must be co-located with the `Gateway` because the Gateway API `spec.infrastructure.parametersRef` is a `LocalParametersReference` — it has no `namespace` field.

   ```yaml
   apiVersion: agentgateway.dev/v1alpha1
   kind: AgentgatewayParameters
   metadata:
     name: inference-gateway-params
     namespace: dynamo-cloud   # same as the Gateway
   spec:
     deployment:
       spec:
         template:
           metadata:
             annotations:
               sidecar.istio.io/inject: "false"
   ```

   Apply it with server-side apply (recommended by agentgateway):

   ```bash
   kubectl apply --server-side -n dynamo-cloud -f agentgateway-params.yaml
   ```

2. Wire the existing `Gateway` to use it. If the Gateway already exists, patch it in place:

   ```bash
   kubectl patch gateway inference-gateway -n dynamo-cloud --type='merge' -p '{
     "spec": {
       "infrastructure": {
         "parametersRef": {
           "group": "agentgateway.dev",
           "kind":  "AgentgatewayParameters",
           "name":  "inference-gateway-params"
         }
       }
     }
   }'
   ```

   Or include the `infrastructure` block directly in your `Gateway` manifest:

   ```yaml
   apiVersion: gateway.networking.k8s.io/v1
   kind: Gateway
   metadata:
     name: inference-gateway
     namespace: dynamo-cloud
   spec:
     gatewayClassName: agentgateway
     infrastructure:
       parametersRef:
         group: agentgateway.dev
         kind: AgentgatewayParameters
         name: inference-gateway-params
     listeners:
       - name: http
         port: 80
         protocol: HTTP
   ```

3. agentgateway will roll the proxy pod. Verify the new pod no longer has an `istio-proxy` container:

   ```bash
   kubectl get pod -l gateway.networking.k8s.io/gateway-name=inference-gateway \
     -n dynamo-cloud \
     -o jsonpath='{.items[0].spec.containers[*].name}{"\n"}'
   # Expect: agentgateway   (NOT "agentgateway istio-proxy")
   ```

***Option B: Patch the default `AgentgatewayParameters` CR (cluster-wide)***

The agentgateway controller creates a default `AgentgatewayParameters` resource named `agentgateway` in `agentgateway-system`. Any `Gateway` that does not set `spec.infrastructure.parametersRef` inherits this default. Patching it affects **all** agentgateway-managed proxies in the cluster.

```bash
kubectl patch agentgatewayparameters agentgateway -n agentgateway-system \
  --type='merge' -p '{
  "spec": {
    "deployment": {
      "spec": {
        "template": {
          "metadata": {
            "annotations": {
              "sidecar.istio.io/inject": "false"
            }
          }
        }
      }
    }
  }
}'
```

Use Option A instead if you have multiple agentgateway-managed gateways in the cluster and only want the `inference-gateway` proxy to skip injection.

The annotation is a no-op on clusters where Istio is not installed, so it is safe to set unconditionally.

> [!NOTE]
> With both the `DestinationRule` (for EPP) and the `AgentgatewayParameters` sidecar exclusion (for agentgateway-proxy) in place, end-to-end GAIE inference works correctly under Istio namespace-level injection.

### 6. Verify Installation ###

Check that all resources are properly deployed:

```bash
kubectl get inferencepool -n ${NAMESPACE}
kubectl get httproute -n ${NAMESPACE}
kubectl get service -n ${NAMESPACE}
kubectl get gateway -n ${NAMESPACE}
```

Sample output:

```bash
# kubectl get inferencepool
NAME        AGE
qwen-pool   33m

# kubectl get httproute
NAME        HOSTNAMES   AGE
qwen-route               33m
```

### 7. Usage ###

The Inference Gateway provides HTTP endpoints for model inference.

#### 1: Populate gateway URL for your k8s cluster ####

a. To test the integration in minikube, proceed as below:
Use minikube tunnel to expose the gateway to the host. This requires `sudo` access to the host machine. Alternatively, you can use port-forward to expose the gateway to the host as shown in alternative (b).

```bash
# in first terminal
ps aux | grep "minikube tunnel" | grep -v grep # make sure minikube tunnel is not already running.
minikube tunnel # start the tunnel

# in second terminal where you want to send inference requests
GATEWAY_URL=$(kubectl get svc inference-gateway -n my-model -o jsonpath='{.spec.clusterIP}') && echo $GATEWAY_URL
```

b. To test on a cluster use commands below:

use port-forward to expose the gateway to the host

```bash
# in first terminal
kubectl port-forward svc/inference-gateway 8000:80 -n ${NAMESPACE} # for NAMESPACE use the namespace where the Gateway service was created, for example my-model

# in second terminal where you want to send inference requests
GATEWAY_URL=http://localhost:8000
```

#### 2: Check models deployed to inference gateway ####

a. Query models:

```bash
# in the second terminal where you GATEWAY_URL is set
curl $GATEWAY_URL/v1/models | jq .
# or if you added the host name to http route:
curl -H "Host: llama3-70b-disagg.example.com" $GATEWAY_URL/v1/models | jq .
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

b. Send inference request to gateway:

```bash
MODEL_NAME="Qwen/Qwen3-0.6B"
curl $GATEWAY_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "'"${MODEL_NAME}"'",
      "messages": [
      {
          "role": "user",
          "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
      }
      ],
      "stream":false,
      "max_tokens": 30,
      "temperature": 0.0
    }'
```
or

```bash
MODEL_NAME="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
curl -H "Host: llama3-70b-disagg.example.com" http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
      "model": "'"${MODEL_NAME}"'",
      "messages": [
      {
          "role": "user",
          "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
      }
      ],
      "stream":false,
      "max_tokens": 30,
      "temperature": 0.0
    }'
```

Sample inference output:

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "audio": null,
        "content": "<think>\nOkay, I need to develop a character background for the user's query. Let me start by understanding the requirements. The character is an",
        "function_call": null,
        "refusal": null,
        "role": "assistant",
        "tool_calls": null
      }
    }
  ],
  "created": 1753768682,
  "id": "chatcmpl-772289b8-5998-4f6d-bd61-3659b684b347",
  "model": "Qwen/Qwen3-0.6B",
  "object": "chat.completion",
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 29,
    "completion_tokens_details": null,
    "prompt_tokens": 196,
    "prompt_tokens_details": null,
    "total_tokens": 225
  }
}
```

***If you have more than one HTTPRoute running on the cluster***
Add the host to your `http-route.yaml` and add the header
`curl -H "Host: llama3-70b-agg.example.com" ...` or `curl -H "Host: llama3-70b-disagg.example.com" http://localhost:8000/v1/models`

```bash
spec:
  hostnames:
    - llama3-70b-agg.example.com
```

### 8. Deleting the installation ###

If you need to uninstall run:

```bash
kubectl delete dynamoGraphDeployment vllm-agg
helm uninstall dynamo-gaie -n my-model

# To uninstall GAIE
# 1. Delete the inference-gateway
kubectl delete gateway inference-gateway --ignore-not-found

# 2. Uninstall agentgateway helm releases
helm uninstall agentgateway -n agentgateway-system
helm uninstall agentgateway-crds -n agentgateway-system

# 3. Delete the agentgateway-system namespace (optional, cleans up everything in it)
kubectl delete namespace agentgateway-system --ignore-not-found

# 4. Delete the Inference Extension CRDs
IGW_LATEST_RELEASE=v1.5.0-rc.2
kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml --ignore-not-found

# 5. Delete the Gateway API CRDs
GATEWAY_API_VERSION=v1.5.1
kubectl delete -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$GATEWAY_API_VERSION/standard-install.yaml --ignore-not-found
```

## Gateway API Inference Extension Integration

This section documents the updated plugin implementation for Gateway API Inference Extension **v1.5.0-rc.2**.

### Router bookkeeping operations

EPP performs Dynamo router book keeping operations so the FrontEnd's Router does not have to sync its state.


### Header Routing Hints

Since v1.5.0-rc.1, the EPP uses **headers and body mutations** for communicating routing decisions.
The plugins set HTTP headers for worker targeting and inject pre-computed token IDs
into the request body (`nvext.token_data`) so the frontend sidecar can skip redundant tokenization.

#### Headers Set by Dynamo Plugins

| Header | Description | Set By |
|--------|-------------|--------|
| `x-worker-instance-id` | Primary worker ID (decode worker in disagg mode) | kv-aware-scorer |
| `x-prefill-instance-id` | Prefill worker ID (disaggregated mode only) | kv-aware-scorer |
