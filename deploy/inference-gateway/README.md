## Inference Gateway Setup with Dynamo

When integrating Dynamo with the Inference Gateway you could either use the default EPP image provided by the extension or use the custom Dynamo image.

1. When using the Dynamo custom EPP image you will take advantage of the Dynamo router when EPP chooses the best worker to route the request to. This setup uses a custom Dynamo plugin `dyn-kv` to pick the best worker. In this case the Dynamo routing logic is moved upstream. We recommend this approach.

2. When using the GAIE-provided image for the EPP, the Dynamo deployment is treated as a black box and the EPP would route round-robin. In this case GAIE just fans out the traffic, and the smarts only remain within the Dynamo graph. Use this if you have one Dynamo graph and do not want to obtain the Dynamo EPP image. This is a "backup" approach.

The setup provided here uses the Dynamo custom EPP by default. Set `epp.useDynamo=false` in your deployment to pick the approach 2.

EPP’s default kv-routing approach is token-aware only `by approximation` because the prompt is tokenized with a generic tokenizer unaware of the model deployed. But the Dynamo plugin uses a token-aware KV algorithm. It employs the dynamo router which implements kv routing by running your model’s tokenizer inline. The EPP plugin configuration lives in [`helm/dynamo-gaie/epp-config-dynamo.yaml`](helm/dynamo-gaie/epp-config-dynamo.yaml) per EPP [convention](https://gateway-api-inference-extension.sigs.k8s.io/guides/epp-configuration/config-text/).

Currently, these setups are only supported with the kGateway based Inference Gateway.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Usage](#6-usage)

## Prerequisites

- Kubernetes cluster with kubectl configured
- NVIDIA GPU drivers installed on worker nodes

## Installation Steps

### 1. Install Dynamo Platform ###

[See Quickstart Guide](../../docs/kubernetes/README.md) to install Dynamo Cloud.

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
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$INFERENCE_EXTENSION_VERSION/manifests.yaml
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

Do not forget to include the HuggingFace token if required.

```bash
export HF_TOKEN=your_hf_token
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN} \
  -n ${NAMESPACE}
```

Create a model configuration file similar to the vllm_agg_qwen.yaml for your model.
This file demonstrates the values needed for the Vllm Agg setup in [agg.yaml](../../components/backends/vllm/deploy/agg.yaml)
Take a note of the model's block size provided in the model card.

### 4. Install Dynamo GAIE helm chart ###

The Inference Gateway is configured through the `inference-gateway-resources.yaml` file.

Deploy the Inference Gateway resources to your Kubernetes cluster by running the command below.

```bash
cd deploy/inference-gateway

# Export the Dynamo image you have used when deploying your model in Step 3.
export DYNAMO_IMAGE=<the-dynamo-image-you-have-used-when-deploying-the-model>
# Export the image tag provided by Dynamo (nvcr.io/nvstaging/ai-dynamo/epp-inference-extension-dynamo:v0.6.0-1) or you can build the Dynamo EPP image by following the commands later in this README.
export EPP_IMAGE=<the-epp-image-you-built>
```

```bash
helm upgrade --install dynamo-gaie ./helm/dynamo-gaie -n my-model -f ./vllm_agg_qwen.yaml --set-string extension.image=$EPP_IMAGE
# do not include --set-string extension.image=$EPP_IMAGE to use the default images
```

Key configurations include:

- An InferenceModel resource for the Qwen model
- A service for the inference gateway
- Required RBAC roles and bindings
- RBAC permissions
- values-dynamo-epp.yaml sets epp.dynamo.namespace=vllm-agg for the bundled example. Point it at your actual Dynamo namespace by editing that file or adding --set epp.dynamo.namespace=<namespace> (and likewise for epp.dynamo.component, epp.dynamo.kvBlockSize if they differ).


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
You can either use the image `nvcr.io/nvstaging/ai-dynamo/epp-inference-extension-dynamo:v0.6.0-1` for the EPP_IMAGE in the Helm deployment command and proceed to the step 2 or you can build the image yourself following the steps below.

##### 1. Build the custom EPP image #####

If you choose to build your own image use the steps below.

##### 1.1 Clone the official GAIE repo in a separate folder #####

```bash
git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git
cd gateway-api-inference-extension
git checkout v0.5.1
```

##### 1.2 Build the Dynamo Custom EPP #####

###### 1.2.1 Clone the official EPP repo ######

```bash
# Clone the official GAIE repo in a separate folder
cd path/to/gateway-api-inference-extension
git clone git@github.com:kubernetes-sigs/gateway-api-inference-extension.git
git checkout v0.5.1
```

###### 1.2.2 Run the script to build the EPP image ######

The script will apply a custom patch to the code with your GAIE repo and build the image for you to use.

```bash
# Use your custom paths
export DYNAMO_DIR=/path/to/dynamo
export GAIE_DIR=/path/to/gateway-api-inference-extension

# Run the script
cd deploy/inference-gateway
./build-epp-dynamo.sh
```

Under the hood the script applies the Dynamo Patch to the EPP code base; creates a Dynamo Router static library and builds a custom EPP image with it.
Re-tag the freshly built image and push it to your registry.

```bash
docker images
docker tag <your-new-id> <your-image-tag>
docker push  <your-image-tag>
```


**Note**
You can also use the standard EPP image`us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp:v0.4.0`. For the basic black box integration run:

```bash
cd deploy/inference-gateway
# Optionally export the standard EPP image if you do not want to use the default we suggest.
export EPP_IMAGE=us-central1-docker.pkg.dev/k8s-artifacts-prod/images/gateway-api-inference-extension/epp:v0.4.0
helm upgrade --install dynamo-gaie ./helm/dynamo-gaie -n my-model -f ./vllm_agg_qwen.yaml --set epp.useDynamo=false
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
