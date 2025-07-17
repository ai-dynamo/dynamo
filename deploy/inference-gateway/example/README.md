# Installing Inference Gateway with Dynamo (Experimental)

This is an experimental setup that treats each Dynamo deployment as a black box and routes traffic randomly among the deployments.

This guide provides instructions for setting up the Inference Gateway with Dynamo for managing and routing inference requests.

## Prerequisites

- Kubernetes cluster with kubectl configured
- NVIDIA GPU drivers installed on worker nodes

## Installation Steps

1. **Install Dynamo Cloud**

Follow the instructions in [deploy/cloud/README.md](../../deploy/cloud/README.md) to deploy Dynamo Cloud on your Kubernetes cluster. This will set up the necessary infrastructure components for managing Dynamo inference graphs.

2. **Launch 2 Dynamo Deployments**

Deploy 2 Dynamo aggregated graphs following the instructions in [examples/llm/README.md](../../examples/llm/README.md):

### Deploy Dynamo Graphs

Follow the commands to deploy 2 dynamo graphs -

```bash
# Set pre-built vLLM dynamo base container image
export VLLM_RUNTIME_IMAGE=<dynamo-vllm-base-image>
# for example:
# export VLLM_RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.3.1

# run the following commands from dynamo repo's root folder

# Deploy first graph
export DEPLOYMENT_NAME=llm-agg1
yq eval '
  .metadata.name = env(DEPLOYMENT_NAME) |
  .spec.services[].extraPodSpec.mainContainer.image = env(VLLM_RUNTIME_IMAGE)
' examples/vllm_v0/deploy/agg.yaml > examples/vllm_v0/deploy/agg1.yaml

kubectl apply -f examples/vllm_v0/deploy/agg1.yaml

# Deploy second graph
export DEPLOYMENT_NAME=llm-agg2
yq eval '
  .metadata.name = env(DEPLOYMENT_NAME) |
  .spec.services[].extraPodSpec.mainContainer.image = env(VLLM_RUNTIME_IMAGE)
' examples/vllm_v0/deploy/agg.yaml > examples/vllm_v0/deploy/agg2.yaml

kubectl apply -f examples/vllm_v0/deploy/agg2.yaml
```

3. **Deploy Inference Gateway**

First, deploy an inference gateway service. In this example, we'll install `kgateway` based gateway implementation.

Install the Inference Extension CRDs:
```bash
VERSION=v0.3.0
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$VERSION/manifests.yaml
```

Deploy an Inference Gateway. In this example, we'll install `Kgateway`:
```bash
KGTW_VERSION=v2.0.2

# Install the Kgateway CRDs
helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds

# Install Kgateway
helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --set inferenceExtension.enabled=true

# Deploy the Gateway
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.3.0/standard-install.yaml
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml
```

### Validate Resources
```bash
kubectl get gateway inference-gateway

# Sample output
# NAME                CLASS      ADDRESS   PROGRAMMED   AGE
# inference-gateway   kgateway             True         1m
```

4. **Apply Dynamo-specific manifests**

The Inference Gateway is configured through the `inference-gateway-resources.yaml` file.

Deploy the Inference Gateway resources to your Kubernetes cluster:

```bash
cd deploy/inference-gateway/example
kubectl apply -f resources
```

Key configurations include:
- An InferenceModel resource for the DeepSeek model
- A service for the inference gateway
- Required RBAC roles and bindings
- RBAC permissions

5. **Verify Installation**

Check that all resources are properly deployed:

```bash
kubectl get inferencepool
kubectl get inferencemodel
kubectl get httproute
```

Sample output:

```bash
# kubectl get inferencepool
NAME              AGE
dynamo-deepseek   6s

# kubectl get inferencemodel
NAME              MODEL NAME                                 INFERENCE POOL    CRITICALITY   AGE
deep-seek-model   deepseek-ai/DeepSeek-R1-Distill-Llama-8B   dynamo-deepseek   Critical      6s

# kubectl get httproute
NAME        HOSTNAMES   AGE
llm-route               6s
```

## Usage

The Inference Gateway provides HTTP/2 endpoints for model inference. The default service is exposed on port 9002.

### 1: Populate gateway URL for your k8s cluster
```bash
export GATEWAY_URL=<Gateway-URL>
```

To test the gateway in minikube, use the following command:
```bash
minikube tunnel &

GATEWAY_URL=$(kubectl get svc inference-gateway -o yaml -o jsonpath='{.spec.clusterIP}')
echo $GATEWAY_URL
```

### 2: Check models deployed to inference gateway

Query models:
```bash
curl $GATEWAY_URL/v1/models | jq .
```

Send inference request to gateway:

```bash
curl $GATEWAY_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```