---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Google Kubernetes Engine (GKE)
---

## Pre-requisites

### Install gcloud CLI
https://cloud.google.com/sdk/docs/install

### Create a GKE cluster

```bash
export PROJECT_ID=<>
export REGION=<>
export ZONE=<>
export CLUSTER_NAME=<>
export GENERAL_PURPOSE_MACHINE_TYPE=n2-standard-4
export NUM_GENERAL_PURPOSE_NODES=1
export DISK_SIZE=200

gcloud container clusters create ${CLUSTER_NAME} \
 	--project=${PROJECT_ID} \
 	--location=${ZONE} \
	--subnetwork=default \
  --disk-size=${DISK_SIZE} \
  --machine-type=${GENERAL_PURPOSE_MACHINE_TYPE} \
  --num-nodes=${NUM_GENERAL_PURPOSE_NODES}
```

#### Create a GPU pool

```bash
export GPU_MACHINE_TYPE=g2-standard-4
export GPU_TYPE=nvidia-l4
export NUM_GPU_NODES=1
export GPU_COUNT=1

gcloud container node-pools create gpu-pool \
 	--accelerator type=${GPU_TYPE},count=${GPU_COUNT},gpu-driver-version=latest \
 	--project=${PROJECT_ID} \
 	--location=${ZONE} \
 	--cluster=${CLUSTER_NAME} \
  --machine-type=${GPU_MACHINE_TYPE} \
  --disk-size=${DISK_SIZE} \
  --num-nodes=${NUM_GPU_NODES}
```

## Install Dynamo Kubernetes Platform

Perform steps 2 and (optionally) 3 in [Dynamo Installation Guide](https://docs.nvidia.com/dynamo/kubernetes-deployment/start-here/installation-guide#overview). On GKE, [the recommended way](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/gpu-operator)
is not to install NVIDIA GPU operator. This is why you skip [step 1](../../installation-guide.md#step-1-install-the-gpu-operator).

## Deploy Inference Graph

In this section you deploy VLLM in the aggregated mode, serving `Qwen/Qwen3-0.6B`.

### Create a namespace for your inference graph deployments

```bash
export INFERENCE_NAMESPACE=my-inference-graphs
kubectl create namespace ${INFERENCE_NAMESPACE}
```

### Create a secret to store your Hugging Face token (required to download model weights)

```bash
export HF_TOKEN=<HF_TOKEN>
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=${HF_TOKEN} -n ${INFERENCE_NAMESPACE}
```

### Inspect the deployment gragh

Inspect https://github.com/ai-dynamo/dynamo/blob/main/examples/deployments/GKE/vllm/v1beta1/agg.yaml.
It contains DynamoGraphDeployment Custom Resource of the inference graph you will deploy.
You might want to download and edit it, or you can just apply the above manifest AS IS.

Note that the `args` field contains `LD_LIBRARY_PATH` and `PATH` to let GKE [find the correct GPU driver](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus).

For more manifests, check https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments/GKE/vllm.

### Perform the deployment

We will deploy a LLM model to the Dynamo platform. Here we use `Qwen/Qwen3-0.6B` model with VLLM and disaggregated deployment as an example.

In the deployment yaml file, some adjustments have to/ could be made:

1.  Wait for the graph deployment to become ready:

    ```bash
    kubectl wait --for=condition=Ready dgd/vllm-agg -n ${INFERENCE_NAMESPACE} --timeout=30m
    ```

    Once the inference graph is ready, you get the following output from the comamnd above:

    ```bash
    dynamographdeployment.nvidia.com/vllm-agg condition met
    ```

1.  Check the inference graph components:

    ```bash
    kubectl get pods -n ${INFERENCE_NAMESPACE}
    NAME                                                  READY   STATUS    RESTARTS   AGE
    vllm-agg-frontend-76898f5988-p5bw6                    1/1     Running   0          5m
    vllm-agg-vllmdecodeworker-2e88533b-6446cd9cdb-2n4sv   1/1     Running   0          5m
    ```

## Deploy the model

```bash
cd examples/deployments/GKE/vllm/v1beta1

kubectl apply -f disagg.yaml -n ${NAMESPACE}
```

**Expected output after successful deployment**

```bash
kubectl get pods
NAME                                                              READY   STATUS    RESTARTS   AGE
dynamo-platform-dynamo-operator-controller-manager-c665684ssqkx   2/2     Running   0          65m
dynamo-platform-etcd-0                                            1/1     Running   0          65m
dynamo-platform-nats-0                                            2/2     Running   0          65m
vllm-disagg-frontend-5954ddc4dd-4w2cb                             1/1     Running   0          11m
vllm-disagg-vllmdecodeworker-77844cfcff-ddn4v                     1/1     Running   0          11m
vllm-disagg-vllmprefillworker-55d5b74b4f-zrskh                    1/1     Running   0          11m
```

## Test the Deployment

```bash
export DEPLOYMENT_NAME=vllm-disagg

# Find the frontend pod
export FRONTEND_POD=$(kubectl get pods -n ${NAMESPACE} | grep "${DEPLOYMENT_NAME}-frontend" | sort -k1 | tail -n1 | awk '{print $1}')

# Forward the pod's port to localhost
kubectl port-forward deployment/vllm-disagg-frontend  8000:8000 -n ${NAMESPACE}

# disagg
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
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

### Response

```json
{"id":"chatcmpl-bd0670d9-0342-4eea-97c1-99b69f1f931f","choices":[{"index":0,"message":{"content":"Okay, here's a detailed character background for your intrepid explorer, tailored to fit the premise of Aeloria, with a focus on a","refusal":null,"tool_calls":null,"role":"assistant","function_call":null,"audio":null},"finish_reason":"stop","logprobs":null}],"created":1756336263,"model":"Qwen/Qwen3-0.6B","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":{"prompt_tokens":190,"completion_tokens":29,"total_tokens":219,"prompt_tokens_details":null,"completion_tokens_details":null}}
```
