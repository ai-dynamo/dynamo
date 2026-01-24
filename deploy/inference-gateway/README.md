## Inference Gateway Setup with Dynamo

When integrating Dynamo with the Inference Gateway it is recommended to use the custom Dynamo EPP image.

1. **Dynamo EPP (Recommended):** The custom Dynamo EPP image integrates the Dynamo router directly into the gateway's endpoint picker. Using the `dyn-kv` plugin, it selects the optimal worker based on KV cache state and tokenized prompt before routing the request. The integration moves intelligent routing upstream to the gateway layer.

2. **Standard EPP (Fallback):** You can use the default GAIE EPP image, which treats the Dynamo deployment as a black box and routes requests round-robin. Routing intelligence remains within the Dynamo graph itself. Use this approach if you have a single Dynamo graph and don't need the custom EPP image.

EPP’s default kv-routing approach is not token-aware because the prompt is not tokenized. But the Dynamo plugin uses a token-aware KV algorithm. It employs the dynamo router which implements kv routing by running your model’s tokenizer inline. The EPP plugin configuration lives in [`helm/dynamo-gaie/epp-config-dynamo.yaml`](helm/dynamo-gaie/epp-config-dynamo.yaml) per EPP [convention](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/config-text/).

The setup provided here uses the Dynamo custom EPP by default. Set `epp.useDynamo=false` in your deployment to pick the approach 2.

Dynamo Integration with the Inference Gateway supports Aggregated and Disaggregated Serving.
If you want to use LoRA deploy Dynamo without the Inference Gateway or in the BlackBox approach with the Inference Gateway.

Currently, these setups are only supported with the kGateway based Inference Gateway.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
  - [1. Install Dynamo Platform](#1-install-dynamo-platform)
  - [2. Deploy Inference Gateway](#2-deploy-inference-gateway)
  - [3. Deploy Your Model](#3-deploy-your-model)
  - [4. Build EPP image](#4-build-epp-image)
  - [5. Install Dynamo GAIE helm chart](#5-install-dynamo-gaie-helm-chart)
  - [6. Verify Installation](#6-verify-installation)
  - [7. Usage](#7-usage)
  - [8. Deleting the installation](#8-deleting-the-installation)
- [Gateway API Inference Extension Details](#gateway-api-inference-extension-integration)
  - [Router bookkeeping operations](#router-bookkeeping-operations)
  - [Header Routing Hints](#header-routing-hints)


## Prerequisites

- Kubernetes cluster with kubectl configured
- NVIDIA GPU drivers installed on worker nodes

## Installation Steps

### 1. Install Dynamo Platform ###

[See Quickstart Guide](../../docs/kubernetes/README.md) to install Dynamo Kubernetes Platform.

### 2. Deploy Inference Gateway ###

First, deploy an inference gateway service. In this example, we'll install `kgateway` based gateway implementation.

```bash
cd deploy/inference-gateway
./scripts/install_gaie_crd_kgateway.sh
```
**Note**: The manifest at `config/manifests/gateway/kgateway/gateway.yaml` uses `gatewayClassName: agentgateway`, but kGateway's helm chart creates a GatewayClass named `kgateway`. The patch command in the script fixes this mismatch.

#### f. Verify the Gateway is running

```bash
kubectl get gateway inference-gateway

# Sample output
# NAME                CLASS      ADDRESS   PROGRAMMED   AGE
# inference-gateway   kgateway             True         1m
```


### 3. Deploy Your Model ###

Follow the steps in [model deployment](../../examples/backends/vllm/deploy/README.md) to deploy `Qwen/Qwen3-0.6B` model in aggregate mode using [agg.yaml](../../examples/backends/vllm/deploy/agg.yaml) in `my-model` kubernetes namespace.
Make sure to enable kv-routing by adding the env var in the FrontEnd.
```bash
    mainContainer:
      image: ...
      env:
        - name: DYN_ROUTER_MODE
          value: "kv"
```

Sample commands to deploy model:

```bash
cd <dynamo-source-root>
cd examples/backends/vllm/deploy
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

### 4. Build EPP image

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

### 5. Install Dynamo GAIE helm chart ###

The Inference Gateway is configured through the `inference-gateway-resources.yaml` file.

Deploy the Inference Gateway resources to your Kubernetes cluster by running the command below.

```bash
cd deploy/inference-gateway/

# Export the Dynamo image you have used when deploying your model in Step 3.
export DYNAMO_IMAGE=<the-dynamo-image-you-have-used-when-deploying-the-model>
# Export the FrontEnd image tag provided by Dynamo (recommended) or build the Dynamo EPP image by following the commands later in this README.
export EPP_IMAGE=<the-epp-image-you-built>
```

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
You can configure the plugin by setting environment vars in your [values-dynamo-epp.yaml].

- Overwrite the `DYN_NAMESPACE` env var if needed to match your model's dynamo namespace.
- Set `DYNAMO_BUSY_THRESHOLD` to configure the upper bound on how “full” a worker can be (often derived from kv_active_blocks or other load metrics) before the router skips it. If the selected worker exceeds this value, routing falls back to the next best candidate. By default the value is negative meaning this is not enabled.
- Set `DYNAMO_ENFORCE_DISAGG=true` if you want to enforce every request being served in the disaggregated manner. By default it is false meaning if the the prefill worker is not available the request will be served in the aggregated manner.
- By default the Dynamo plugin uses KV routing. You can expose `DYNAMO_USE_KV_ROUTING=false`  in your [values-dynamo-epp.yaml] if you prefer to route in the round-robin fashion.
- If using kv-routing:
  - Overwrite the `DYNAMO_KV_BLOCK_SIZE` in your [values-dynamo-epp.yaml](./values-dynamo-epp.yaml) to match your model's block size.The `DYNAMO_KV_BLOCK_SIZE` env var is ***MANDATORY*** to prevent silent KV routing failures.
  - Set `DYNAMO_OVERLAP_SCORE_WEIGHT` to weigh how heavily the score uses token overlap (predicted KV cache hits) versus other factors (load, historical hit rate). Higher weight biases toward reusing workers with similar cached prefixes.
  - Set `DYNAMO_ROUTER_TEMPERATURE` to soften or sharpen the selection curve when combining scores. Low temperature makes the router pick the top candidate deterministically; higher temperature lets lower-scoring workers through more often (exploration).
  - Set `DYNAMO_USE_KV_EVENTS=false` if you want to disable KV event tracking while using kv-routing
  - See the [KV cache routing design](../../docs/router/kv_cache_routing.md) for details.


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

### 6. Verify Installation ###

Check that all resources are properly deployed:

```bash
kubectl get inferencepool
kubectl get httproute
kubectl get service
kubectl get gateway
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

To test the gateway in minikube, use the following command:
a. User minikube tunnel to expose the gateway to the host
   This requires `sudo` access to the host machine. alternatively, you can use port-forward to expose the gateway to the host as shown in alternative (b).

```bash
# in first terminal
ps aux | grep "minikube tunnel" | grep -v grep # make sure minikube tunnel is not already running.
minikube tunnel # start the tunnel

# in second terminal where you want to send inference requests
GATEWAY_URL=$(kubectl get svc inference-gateway -n my-model -o jsonpath='{.spec.clusterIP}')
echo $GATEWAY_URL
```

b. use port-forward to expose the gateway to the host

```bash
# in first terminal
kubectl port-forward svc/inference-gateway 8000:80 -n my-model

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

### 8. Deleting the installation ###

If you need to uninstall run:

```bash
kubectl delete dynamoGraphDeployment vllm-agg
helm uninstall dynamo-gaie -n my-model

# To uninstall GAIE
# 1. Delete the inference-gateway
kubectl delete gateway inference-gateway --ignore-not-found

# 2. Uninstall kgateway helm releases
helm uninstall kgateway -n kgateway-system
helm uninstall kgateway-crds -n kgateway-system

# 3. Delete the kgateway-system namespace (optional, cleans up everything in it)
helm uninstall kgateway --namespace kgateway-system
kubectl delete namespace kgateway-system --ignore-not-found

# 4. Delete the Inference Extension CRDs
IGW_LATEST_RELEASE=v1.2.1
kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml --ignore-not-found

# 5. Delete the Gateway API CRDs
GATEWAY_API_VERSION=v1.4.1
kubectl delete -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$GATEWAY_API_VERSION/standard-install.yaml --ignore-not-found
```

## Gateway API Inference Extension Integration

This section documents the updated plugin implementation for Gateway API Inference Extension **v1.2.1**.

### Router bookkeeping operations

EPP performs Dynamo router book keeping operations so the FrontEnd's Router does not have to sync its state.


### Header Routing Hints

Since v1.2.1, the EPP uses a **header-only approach** for communicating routing decisions.
The plugins set HTTP headers that are forwarded to the backend workers.

#### Headers Set by Dynamo Plugins

| Header | Description | Set By |
|--------|-------------|--------|
| `x-worker-instance-id` | Primary worker ID (decode worker in disagg mode) | kv-aware-scorer |
| `x-prefill-instance-id` | Prefill worker ID (disaggregated mode only) | kv-aware-scorer |
