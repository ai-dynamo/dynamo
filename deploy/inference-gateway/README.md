## Inference Gateway Setup with Dynamo

When integrating Dynamo with the Inference Gateway it is recommended to use the custom Dynamo EPP image.

1. **Dynamo EPP (Recommended):** The custom Dynamo EPP image integrates the Dynamo router directly into the gateway's endpoint picker. Using the `dyn-kv` plugin, it selects the optimal worker based on KV cache state and tokenized prompt before routing the request. The integration moves intelligent routing upstream to the gateway layer.

2. **Standard EPP (Fallback):** You can use the default GAIE EPP image, which treats the Dynamo deployment as a black box and routes requests round-robin. Routing intelligence remains within the Dynamo graph itself. Use this approach if you have a single Dynamo graph and don't need the custom EPP image.

EPP’s default kv-routing approach is not token-aware because the prompt is not tokenized. But the Dynamo plugin uses a token-aware KV algorithm. It employs the dynamo router which implements kv routing by running your model’s tokenizer inline. The EPP plugin configuration lives in [`helm/dynamo-gaie/epp-config-dynamo.yaml`](helm/dynamo-gaie/epp-config-dynamo.yaml) per EPP [convention](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/config-text/).

The setup provided here uses the Dynamo custom EPP by default. Set `epp.useDynamo=false` in your deployment to pick the approach 2.

Currently, these setups are only supported with the kGateway based Inference Gateway.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
  - [1. Install Dynamo Platform](#1-install-dynamo-platform)
  - [2. Deploy Inference Gateway](#2-deploy-inference-gateway)
  - [3. Deploy Your Model](#3-deploy-your-model)
  - [4. Install Dynamo GAIE Helm Chart](#4-install-dynamo-gaie-helm-chart)
  - [5. Verify Installation](#5-verify-installation)
  - [6. Usage](#6-usage)
  - [7. Deleting the Installation](#7-deleting-the-installation)
- [Gateway API Inference Extension v1.2.1 Integration](#gateway-api-inference-extension-v121-integration)
- [Body Injector](#body-injector)

## Prerequisites

- Kubernetes cluster with kubectl configured
- NVIDIA GPU drivers installed on worker nodes

## Installation Steps

### 1. Install Dynamo Platform ###

[See Quickstart Guide](../../docs/kubernetes/README.md) to install Dynamo Kubernetes Platform.

### 2. Deploy Inference Gateway ###

First, deploy an inference gateway service. In this example, we'll install `kgateway` based gateway implementation.
You can use the script below or follow the steps manually.

Script:

```bash
./scripts/install_gaie_crd_kgateway.sh
```

Manual steps:

#### a. Deploy the Gateway API CRDs

```bash
GATEWAY_API_VERSION=v1.4.1
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$GATEWAY_API_VERSION/standard-install.yaml
```

#### b. Install the Inference Extension CRDs

```bash
IGW_LATEST_RELEASE=v1.2.1
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml
```

#### c. Install kGateway CRDs and kGateway itself

kGateway needs the Agentgateway to support the Gateway Api  Inference Extension.

```bash
KGTW_VERSION=v2.1.1
helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION \
  kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds

helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway \
  oci://cr.kgateway.dev/kgateway-dev/charts/kgateway \
  --set inferenceExtension.enabled=true
```

#### d. Deploy the Inference Gateway

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/kgateway/gateway.yaml
```

#### e. Patch the Gateway to use kgateway GatewayClass

The manifest uses `gatewayClassName: agentgateway` but kGateway creates a GatewayClass named `kgateway`. Patch it:

```bash
kubectl patch gateway inference-gateway --type='json' \
  -p='[{"op": "replace", "path": "/spec/gatewayClassName", "value": "kgateway"}]'
```
**Note**: The manifest at `config/manifests/gateway/kgateway/gateway.yaml` uses `gatewayClassName: agentgateway`, but kGateway's helm chart creates a GatewayClass named `kgateway`. The patch command above fixes this mismatch.

#### f. Verify the Gateway is running

```bash
kubectl get gateway inference-gateway

# Sample output
# NAME                CLASS      ADDRESS   PROGRAMMED   AGE
# inference-gateway   kgateway             True         1m
```

#### g. Deploy the Body-Transformer service

**Why is this needed?**

Dynamo backend workers require routing information in the request body as an `nvext` field (not just headers). The GAIE EPP sets routing headers (`x-worker-instance-id`, etc.), but these headers need to be converted into body fields before reaching the backend.

The Body-Transformer is a lightweight ext_proc service that:
1. Reads the routing headers set by the GAIE EPP
2. Injects the `nvext` field into the JSON request body before forwarding to the Dynamo backend

```
Client: POST /v1/chat/completions {"model": "Qwen/Qwen3-0.6B", ...}
    ↓
kGateway
    ↓
EPP (Dynamo KV Scorer)
  → Selects worker: 1732649523291627853
  → Sets header: x-worker-instance-id=1732649523291627853
    ↓
Body Transformer
  → Reads header
  → Injects: {"nvext": {"backend_instance_id": 1732649523291627853}}
  → Body: 106 → 158 bytes
    ↓
Dynamo Backend (vllm-agg-frontend)
  → Receives request with nvext
  → Routes to correct worker
  → Returns response
    ↓
Client receives response
```

**Choosing an approach:**

| Approach | Works With | Pros | Cons |
|----------|------------|------|------|
| **Body-Transformer (ext_proc)** | All Envoy-based gateways (kGateway, Istio, Envoy Gateway, etc.) | Portable, works everywhere | Requires deploying an extra service |
| **Lua filters** | Istio, Envoy Gateway (NOT kGateway) | No extra service needed | Gateway-specific configuration |

- **For kGateway**: Use the Body-Transformer (required - kGateway doesn't support Lua filters)
- **For Istio/Envoy Gateway**: Choose either Body-Transformer OR Lua filters (see examples below)

Follow the [Body-Transformer README](body-transformer/README.md) to build and deploy this service.

### 3. Deploy Your Model ###

Follow the steps in [model deployment](../../examples/backends/vllm/deploy/README.md) to deploy `Qwen/Qwen3-0.6B` model in aggregate mode using [agg.yaml](../../examples/backends/vllm/deploy/agg.yaml) in `my-model` kubernetes namespace.

Sample commands to deploy model:

```bash
cd <dynamo-source-root>/examples/backends/vllm/deploy
kubectl apply -f agg.yaml -n my-model
```

Take a note of or change the DYNAMO_IMAGE in the model deployment file.

Do not forget docker registry secret if needed.

```bash
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=$DOCKER_SERVER \
  --docker-username=$DOCKER_USERNAME \
  --docker-password=$DOCKER_PASSWORD \
  --namespace=$NAMESPACE
```

Do not forget to include the HuggingFace token if required.

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Create a model configuration file similar to the vllm_agg_qwen.yaml for your model.
This file demonstrates the values needed for the Vllm Agg setup in [agg.yaml](../../examples/backends/vllm/deploy/agg.yaml)
Take a note of the model's block size provided in the model card.

### 4. Install Dynamo GAIE helm chart ###

The Inference Gateway is configured through the `inference-gateway-resources.yaml` file.

Deploy the Inference Gateway resources to your Kubernetes cluster by running the command below.

```bash
cd deploy/inference-gateway

# Export the Dynamo image you have used when deploying your model in Step 3.
export DYNAMO_IMAGE=<the-dynamo-image-you-have-used-when-deploying-the-model>
# Export the FrontEnd image tag provided by Dynamo (recommended) or build the Dynamo EPP image by following the commands later in this README.
export EPP_IMAGE=<the-epp-image-you-built>
```

```bash
helm upgrade --install dynamo-gaie ./helm/dynamo-gaie -n my-model -f ./vllm_agg_qwen.yaml --set-string extension.image=$EPP_IMAGE
```

Key configurations include:

- An InferenceModel resource for the Qwen model
- A service for the inference gateway
- Required RBAC roles and bindings
- RBAC permissions
- dynamoGraphDeploymentName - the name of the Dynamo Graph where your model is deployed.


**Configuration**
You can configure the plugin by setting environment vars in your [values-dynamo-epp.yaml].

- Overwrite the `DYN_NAMESPACE` env var if needed to match your model's dynamo namespace.
- Set `DYNAMO_BUSY_THRESHOLD` to configure the upper bound on how “full” a worker can be (often derived from kv_active_blocks or other load metrics) before the router skips it. If the selected worker exceeds this value, routing falls back to the next best candidate. By default the value is negative meaning this is not enabled.
- Set `DYNAMO_ROUTER_REPLICA_SYNC=true` to enable a background watcher to keep multiple router instances in sync (important if you run more than one KV router per component).
- By default the Dynamo plugin uses KV routing. You can expose `DYNAMO_USE_KV_ROUTING=false`  in your [values-dynamo-epp.yaml] if you prefer to route in the round-robin fashion.
- If using kv-routing:
  - Overwrite the `DYNAMO_KV_BLOCK_SIZE` in your [values-dynamo-epp.yaml](./values-dynamo-epp.yaml) to match your model's block size.The `DYNAMO_KV_BLOCK_SIZE` env var is ***MANDATORY*** to prevent silent KV routing failures.
  - Set `DYNAMO_OVERLAP_SCORE_WEIGHT` to weigh how heavily the score uses token overlap (predicted KV cache hits) versus other factors (load, historical hit rate). Higher weight biases toward reusing workers with similar cached prefixes.
  - Set `DYNAMO_ROUTER_TEMPERATURE` to soften or sharpen the selection curve when combining scores. Low temperature makes the router pick the top candidate deterministically; higher temperature lets lower-scoring workers through more often (exploration).
  - Set `DYNAMO_USE_KV_EVENTS=false` if you want to disable KV event tracking while using kv-routing
  - See the [KV cache routing design](../../docs/router/kv_cache_routing.md) for details.



Dynamo provides a custom routing plugin `pkg/epp/scheduling/plugins/dynamo_kv_scorer/plugin.go` to perform efficient kv routing.
The Dynamo router is built as a static library, the EPP router will call to provide fast inference.
You can either use the special FrontEnd image for the EPP_IMAGE in the Helm deployment command and proceed to the step 2 or you can [build the image yourself](#building-your-own-dynamo-epp-custom-image).


**Note**
You can also use the standard EPP image i.e. `us-central1-docker.pkg.dev/k8s-artifacts-prod/images/gateway-api-inference-extension/epp:v1.2.1` for the basic black box integration.

```bash
cd deploy/inference-gateway
helm upgrade --install dynamo-gaie ./helm/dynamo-gaie -n my-model -f ./vllm_agg_qwen.yaml

# Optionally export the standard EPP image if you do not want to use the default we suggest.
export EPP_IMAGE=us-central1-docker.pkg.dev/k8s-artifacts-prod/images/gateway-api-inference-extension/epp:v0.4.0
helm upgrade --install dynamo-gaie ./helm/dynamo-gaie -n my-model -f ./vllm_agg_qwen.yaml --set epp.useDynamo=false --set-string extension.image=$EPP_IMAGE
# Optionally overwrite the image --set-string extension.image=$EPP_IMAGE
```

### 5. Verify Installation ###

Check that all resources are properly deployed:

```bash
kubectl get inferencepool
kubectl get inferencemodel
kubectl get httproute
kubectl get service
kubectl get gateway
```

Sample output:

```bash
# kubectl get inferencepool
NAME        AGE
qwen-pool   33m

# kubectl get inferencemodel
NAME         MODEL NAME        INFERENCE POOL   CRITICALITY   AGE
qwen-model   Qwen/Qwen3-0.6B   qwen-pool        Critical      33m

# kubectl get httproute
NAME        HOSTNAMES   AGE
qwen-route               33m
```

### 6. Usage ###

The Inference Gateway provides HTTP endpoints for model inference.

#### 1: Populate gateway URL for your k8s cluster ####

```bash
export GATEWAY_URL=<Gateway-URL>
```

To test the gateway in minikube, use the following command:
a. User minikube tunnel to expose the gateway to the host
   This requires `sudo` access to the host machine. alternatively, you can use port-forward to expose the gateway to the host as shown in alternative (b).

```bash
# in first terminal
ps aux | grep "minikube tunnel" | grep -v grep # make sure minikube tunnel is not already running.
minikube tunnel & # start the tunnel

# in second terminal where you want to send inference requests
GATEWAY_URL=$(kubectl get svc inference-gateway -n my-model -o yaml -o jsonpath='{.spec.clusterIP}')
echo $GATEWAY_URL
```

b. use port-forward to expose the gateway to the host

```bash
# in first terminal
kubectl port-forward svc/inference-gateway 8000:80 -n default

# in second terminal where you want to send inference requests
GATEWAY_URL=http://localhost:8000
```

#### 2: Check models deployed to inference gateway ####

a. Query models:

```bash
# in the second terminal where you GATEWAY_URL is set

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

### 7. Deleting the installation ###

If you need to uninstall run:

```bash
kubectl delete dynamoGraphDeployment vllm-agg
helm uninstall dynamo-gaie -n my-model
```

---

## Gateway API Inference Extension v1.2.1 Integration

This section documents the updated plugin implementation for Gateway API Inference Extension **v1.2.1**.

### v1.2.1 API Changes

The v1.2.1 release introduces breaking changes to the plugin interfaces:

### Building for v1.2.1

The plugin code for v1.2.1 is in:
- `pkg/plugins/dynamo_kv_scorer/plugin.go`

#### Building your own Dynamo EPP custom image

```bash
# Build Dynamo library and copy to project
make dynamo-lib

# Build Docker image and load locally
make image-local-load

# Or do everything in one command
make all

# Check image tag
make info
```

#### All-in-one Targets

| Target | Description |
|--------|-------------|
| `make dynamo-lib` | Build Dynamo static library and copy to project |
| `make all` | Build Dynamo lib + Docker image + load locally |
| `make all-push` | Build Dynamo lib + Docker image + push to registry |
| `make all-kind` | Build Dynamo lib + Docker image + load to kind |

### Header-Only Routing for v1.2.1

In v1.2.1, the EPP uses a **header-only approach** for communicating routing decisions.
The plugins set HTTP headers that are forwarded to the backend workers.

#### Headers Set by Dynamo Plugins

| Header | Description | Set By |
|--------|-------------|--------|
| `x-worker-instance-id` | Primary worker ID (decode worker in disagg mode) | kv-aware-scorer |
| `x-prefiller-host-port` | Prefill worker ID (disaggregated mode only) | kv-aware-scorer |
| `x-dynamo-routing-mode` | `aggregated` or `disaggregated` | kv-aware-scorer |


## Body Injector

Dynamo backend workers require the `nvext` field in the JSON body (instead of reading headers).
You must deploy a **TrafficPolicy** that uses kGateway's transformation feature to read the headers set by the Dynamo plugins and inject the `nvext` field into the request body.

### How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Request Flow                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Client sends request:                                            │
│     POST /v1/chat/completions                                        │
│     Body: {"model": "llama", "messages": [...]}                      │
│                                                                      │
│  2. EPP (Dynamo KV Scorer) schedules request:                        │
│     → Sets headers:                                                  │
│       x-worker-instance-id: 42                                       │
│       x-dynamo-routing-mode: aggregated                              │
│                                                                      │
│  3. Body Transformer (ext_proc) reads headers and modifies body:     │
│     Body: {"model": "llama", "messages": [...],                      │
│            "nvext": {"backend_instance_id": 42}}                     │
│                                                                      │
│  4. Modified request forwarded to model server                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Body Modification

The transformation injects an `nvext` field into the JSON request body:

**Aggregated Mode (default):**
```json
{
  "model": "llama",
  "messages": [...],
  "nvext": {
    "backend_instance_id": 42
  }
}
```

**Disaggregated Mode:**
```json
{
  "model": "llama",
  "messages": [...],
  "nvext": {
    "prefill_worker_id": 10,
    "decode_worker_id": 42
  }
}
```

### Installation for kGateway

kGateway v2.1.x does NOT support Lua filters or advanced body transformation via Inja templating.
Instead, we use a separate **Body Transformer** ext_proc service.

1. **Build and deploy the Body Transformer:**

```bash
cd body-transformer

# Build the image
make image-build

# Load into minikube (if using minikube)
make image-load

# Deploy the service
kubectl apply -f deploy/deployment.yaml -n my-model
```

2. **Configure kGateway to use it:**

```bash
# Apply GatewayExtension, TrafficPolicy, and ReferenceGrant
kubectl apply -f deploy/kgateway-config.yaml
```

3. **Verify the configuration:**

```bash
# Check body-transformer is running
kubectl get pods -l app=body-transformer -n my-model

# Check TrafficPolicy status
kubectl get trafficpolicy nvext-body-injector -o yaml
```

The status should show `Accepted: True` and `Attached: True`.

See [body-transformer/README.md](body-transformer/README.md) for more details.

### Other Gateway Implementations

We provide Lua filter reference implementations for gateways that support them. These are provided as starting points and may require adjustments for your specific environment:

| Gateway | Configuration File | Resource Type |
|---------|-------------------|---------------|
| Istio | `config/lua-filter/istio-envoyfilter.yaml` | EnvoyFilter |
| Envoy Gateway | `config/lua-filter/envoygateway-patch.yaml` | EnvoyPatchPolicy |

**Note:** kGateway does not support Lua filters. Use the [Body-Transformer ext_proc service](body-transformer/README.md) instead.

For Istio:
```bash
kubectl apply -f config/lua-filter/istio-envoyfilter.yaml
```

For Envoy Gateway:
```bash
kubectl apply -f config/lua-filter/envoygateway-patch.yaml
```

