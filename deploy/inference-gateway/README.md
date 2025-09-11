## Inference Gateway Setup with Dynamo

This guide demonstrates two setups.

- The basic setup treats each Dynamo deployment as a black box and routes traffic randomly among the deployments.
- The EPP-aware setup uses a custom Dynamo plugin `dyn-kv` to pick the best worker.

EPP’s default approach is token-aware only `by approximation` because it relies on the non-model tokenized text in the prompt. But the Dynamo plugin uses a token-aware KV algorithm. It employs the dynamo router which implements kv routing by running your model’s tokenizer inline. The EPP plugin configuration lives in [`helm/dynamo-gaie/epp-config-dynamo.yaml`](helm/dynamo-gaie/epp-config-dynamo.yaml) per EPP [convention](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/config-text/).

Currently, these setups are only supported with the kGateway based Inference Gateway.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Usage](#usage)

## Prerequisites

- Kubernetes cluster with kubectl configured
- NVIDIA GPU drivers installed on worker nodes

## Installation Steps

### 1. Install Dynamo Platform ###

[See Quickstart Guide](../../docs/guides/dynamo_deploy/README.md) to install Dynamo Cloud.


### 2. Deploy Inference Gateway ###

First, deploy an inference gateway service. In this example, we'll install `kgateway` based gateway implementation.
You can use the script below or follow the steps manually.

Script:
```bash
./install_gaie_crd_kgateway.sh
```

Manual steps:

a. Deploy the Gateway API CRDs:
```bash
GATEWAY_API_VERSION=v1.3.0
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$GATEWAY_API_VERSION/standard-install.yaml
```

b. Install the Inference Extension CRDs (Inference Model and Inference Pool CRDs)
```bash
INFERENCE_EXTENSION_VERSION=v0.5.1
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$INFERENCE_EXTENSION_VERSION/manifests.yaml -n  my-model
```

c. Install `kgateway` CRDs and kgateway.
```bash
KGATEWAY_VERSION=v2.0.3

# Install the Kgateway CRDs
helm upgrade -i --create-namespace --namespace kgateway-system --version $KGATEWAY_VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds

# Install Kgateway
helm upgrade -i --namespace kgateway-system --version $KGATEWAY_VERSION kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --set inferenceExtension.enabled=true
```

d. Deploy the Gateway Instance
```bash
kubectl create namespace my-model
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml -n  my-model
```

```bash
kubectl get gateway inference-gateway -n my-model

# Sample output
# NAME                CLASS      ADDRESS   PROGRAMMED   AGE
# inference-gateway   kgateway   x.x.x.x   True         1m
```

### 3. Deploy Your Model ###

Follow the steps in [model deployment](../../components/backends/vllm/deploy/README.md) to deploy `Qwen/Qwen3-0.6B` model in aggregate mode using [agg.yaml](../../components/backends/vllm/deploy/agg.yaml) in `my-model` kubernetes namespace.

Sample commands to deploy model:
```bash
cd <dynamo-source-root>/components/backends/vllm/deploy
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

Do not forget to include the the HuggingFace token if required.
```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Create a model configuraiont file similar to the vllm_agg_qwen.yaml for you model.
This file demonstrates the values needed for the Vllm Agg setup in [agg.yaml](../../components/backends/vllm/deploy/agg.yaml)


### 4. Install Dynamo GAIE helm chart ###

The Inference Gateway is configured through the `inference-gateway-resources.yaml` file.

Deploy the Inference Gateway resources to your Kubernetes cluster by running one of the commands below.

#### Basic Black Box Integration ####

For the basic black box integration run:

```bash
cd deploy/inference-gateway
helm install dynamo-gaie ./helm/dynamo-gaie -n my-model -f ./vllm_agg_qwen.yaml
```

#### EPP-aware Integration with the custom Dynamo Plugin ####

##### 1. Build the custom EPP image #####


##### 1.1 Clone the official GAIE repo in a separate folder #####

```bash
git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git
cd gateway-api-inference-extension
git checkout v0.5.1
```

##### 1.2 Run the script to build Dynamo Custom EPP #####

```bash
# Use your custom paths
cd deploy/inference-gateway
export DYNAMO_DIR=/path/to/dynamo
export EPP_DIR=/path/to/gateway-api-inference-extension-dynamo

# Run the script
./build-epp-dynamo.sh
```

Under the hood the script applies the Dynamo Patch to the EPP code base; creates a Dynamo Router static library and builds a custom EPP image with it.
Retag the freshly built image and push it to your registry.

##### 2. Deploy through helm #####

```bash
cd deploy/inference-gateway

# Export the Dynamo image you have used when deploying your model in Step 3.
export DYNAMO_IMAGE=<the-dynamo-image-you-have-used-when-deploying-the-model>
export EPP_IMAGE=<the-epp-image-you-built>  # i.e. docker.io/lambda108/epp-inference-extension-dynamo:v0.5.1-1

helm upgrade --install dynamo-gaie ./helm/dynamo-gaie \
  -n my-model \
  -f ./vllm_agg_qwen.yaml \
  -f ./values-epp-aware.yaml \
  --set eppAware.enabled=true \
  --set-string eppAware.eppImage=$EPP_IMAGE
```


Key configurations include:
- An InferenceModel resource for the Qwen model
- A service for the inference gateway
- Required RBAC roles and bindings
- RBAC permissions

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

### 7. Deleting the installation ###
If you need to uninstall run:

```bash
kubectl delete dynamoGraphDeployment vllm-agg
helm uninstall dynamo-gaie -n my-model
```
