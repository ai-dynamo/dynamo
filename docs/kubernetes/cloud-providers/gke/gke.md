---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Google Kubernetes Engine (GKE)
---

## Prerequisites

- [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) v1.24+
- [helm](https://helm.sh/docs/intro/install/) v3.0+
- [Hugging Face access token](https://huggingface.co/docs/hub/security-tokens), required to download model weights.
- A GKE cluster (v1.24+) with GPU nodes and GPU allocation plugins, for example
[NVIDIA Kubernetes Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) or
[DRA Driver for NVIDIA GPUs](https://dra-driver-nvidia-gpu.sigs.k8s.io/docs/).

Follow the instructions in the next section to create an example GKE cluster or proceed to
the [Install Dynamo Kubernetes Platform](#install-dynamo-kubernetes-platform) section.

## Create an example GKE cluster

If you already have a GKE cluster, proceed to
the [Install Dynamo Kubernetes Platform](#install-dynamo-kubernetes-platform) section.

Create a GKE cluster with one general purpose node and one GPU node.
The GPU node will have [NVIDIA Kubernetes Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) installed.

### Install gcloud CLI
https://cloud.google.com/sdk/docs/install

### Create a GKE cluster

1.  Set environment variables to store your cluster parameters:

    ```bash
    export PROJECT_ID=<>
    export REGION=<>
    export ZONE=<>
    export CLUSTER_NAME=<>
    export GENERAL_PURPOSE_MACHINE_TYPE=n2-standard-4
    export NUM_GENERAL_PURPOSE_NODES=1
    export DISK_SIZE=200
    ```

1.  Create a cluster:

    ```bash
    gcloud container clusters create ${CLUSTER_NAME} \
    --project=${PROJECT_ID} \
    --location=${ZONE} \
    --subnetwork=default \
    --disk-size=${DISK_SIZE} \
    --machine-type=${GENERAL_PURPOSE_MACHINE_TYPE} \
    --num-nodes=${NUM_GENERAL_PURPOSE_NODES}
    ```

#### Create a GPU node pool

1.  Set environment variables to store your node pool parameters:

    ```bash
    export GPU_MACHINE_TYPE=g2-standard-4
    export GPU_TYPE=nvidia-l4
    export NUM_GPU_NODES=1
    export GPU_COUNT=1
    ```

1.  Create a node pool:

    ```bash
    gcloud container node-pools create gpu-pool \
    --accelerator type=${GPU_TYPE},count=${GPU_COUNT},gpu-driver-version=default \
    --project=${PROJECT_ID} \
    --location=${ZONE} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=${GPU_MACHINE_TYPE} \
    --disk-size=${DISK_SIZE} \
    --num-nodes=${NUM_GPU_NODES}
    ```

## Install Dynamo Kubernetes Platform

Set `RELEASE_VERSION` to `1.2.1` in [step 2](../../installation-guide.md#step-2-install-the-dynamo-platform) of
[Dynamo Installation Guide](../../installation-guide.md#overview) and perform the step.
Optionally, perform
[step 3](../../installation-guide.md#step-3-install-optional-components).
You skip [step 1](../../installation-guide.md#step-1-install-the-gpu-operator) since on GKE
[the recommended way](https://cloud.google.com/kubernetes-engine/docs/how-to/gpu-operator#why)
is not to install NVIDIA GPU operator.

## Deploy an inference graph

Deploy vLLM in the aggregated mode, serving `Qwen/Qwen3-0.6B`.

### Create a namespace for your inference graph deployments

```bash
export INFERENCE_NAMESPACE=my-inference-graphs
kubectl create namespace ${INFERENCE_NAMESPACE}
```

### Create a secret to store your Hugging Face token

```bash
export HF_TOKEN=<your Hugging Face token>
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=${HF_TOKEN} -n ${INFERENCE_NAMESPACE}
```

### Inspect the deployment manifest

Clone the Dynamo GitHub repository:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
```

Inspect `examples/deployments/GKE/vllm/v1beta1/agg.yaml`.

It contains
[DynamoGraphDeployment](https://docs.nvidia.com/dynamo/latest/kubernetes-deployment/start-here/kubernetes-quickstart#understand-dynamo-deployment-resources)
[Custom Resource](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
of the inference graph you will deploy.
You might want to download and edit it, or you can just apply the above manifest as is.

Points to note in the container spec of `VllmDecodeWorker`:
- the `args` field contains `LD_LIBRARY_PATH` and `PATH` to let GKE [find the correct GPU driver](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus).
- the `resources` field uses the `nvidia.com/gpu` resource type to request one GPU.
Modify this field if you use other resource type to designate the GPUs in your cluster or if you use
[DRA](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/).

Verify that your cluster has nodes with GPUs and GPU allocation is configured:

```bash
kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.allocatable['nvidia\.com/gpu']"
```

```text
NAME                                        GPU
gke-my-cluster-default-pool-81c691d5-v7pn   <none>
gke-my-cluster-gpu-pool-6b65a2ea-7p04       1
```

For more manifests, check https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments/GKE/vllm.

### Deploy

1.  Deploy the DynamoGraphDeployment manifest:

    ```bash
    kubectl apply -n ${INFERENCE_NAMESPACE} -f examples/deployments/GKE/vllm/v1beta1/agg.yaml
    ```

1.  Wait for the graph deployment to become ready:

    ```bash
    kubectl wait --for=condition=Ready dgd/vllm-agg -n ${INFERENCE_NAMESPACE} --timeout=30m
    ```

    Once the inference graph is ready, you get the following output from the command above:

    ```bash
    dynamographdeployment.nvidia.com/vllm-agg condition met
    ```

1.  Check the inference graph components:

    ```bash
    kubectl get pods -n ${INFERENCE_NAMESPACE}
    ```

    ```text
    NAME                                                  READY   STATUS    RESTARTS   AGE
    vllm-agg-frontend-76898f5988-p5bw6                    1/1     Running   0          5m
    vllm-agg-vllmdecodeworker-2e88533b-6446cd9cdb-2n4sv   1/1     Running   0          5m
    ```

### Test

1.  In a separate terminal, forward the port of the frontend service of the inference graph to your local machine:

    ```bash
    kubectl port-forward service/vllm-agg-frontend 8000:8000 -n ${INFERENCE_NAMESPACE}
    ```

1.  Send a chat completion request:

    ```bash
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

    Expected response:
    ```json
    {"id":"chatcmpl-bd0670d9-0342-4eea-97c1-99b69f1f931f","choices":[{"index":0,"message":{"content":"Okay, here's a detailed character background for your intrepid explorer, tailored to fit the premise of Aeloria, with a focus on a","refusal":null,"tool_calls":null,"role":"assistant","function_call":null,"audio":null},"finish_reason":"stop","logprobs":null}],"created":1756336263,"model":"Qwen/Qwen3-0.6B","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":{"prompt_tokens":190,"completion_tokens":29,"total_tokens":219,"prompt_tokens_details":null,"completion_tokens_details":null}}
    ```

### Cleanup

Follow the steps in the subsections below to delete the components added in this guide.

#### Undeploy the inference graph

1.  Kill the port-forwarding command you ran in a separate terminal.

1.  Undeploy the inference graph.

    ```bash
    kubectl delete -n ${INFERENCE_NAMESPACE} -f examples/deployments/GKE/vllm/v1beta1/agg.yaml
    ```

1.  Delete the Hugging Face secret:

    ```bash
    kubectl delete secret hf-token-secret -n ${INFERENCE_NAMESPACE}
    ```

1.  Delete the namespace:

    ```bash
    kubectl delete namespace ${INFERENCE_NAMESPACE}
    ```

#### Undeploy Dynamo Kubernetes Platform

1.  Undeploy Dynamo Kubernetes Platform:

    ```bash
    export NAMESPACE="dynamo-system"
    helm delete dynamo-platform -n ${NAMESPACE}
    ```

1.  Delete the CRDs in the output of the following command:

    ```bash
    kubectl get crd -o name | grep 'dynamo.*\.nvidia\.com'
    ```

1.  Delete the Dynamo Kubernetes Platform namespace:

    ```bash
    export NAMESPACE="dynamo-system"
    kubectl delete namespace ${NAMESPACE}
    ```

#### Delete the GKE cluster

If you want to delete the GKE cluster you created in [Create a GKE Cluster](#create-a-gke-cluster), perform the following steps.

1.  Delete the node pool:

    ```bash
    gcloud container node-pools delete gpu-pool --project=${PROJECT_ID}  --location=${ZONE}  --cluster=${CLUSTER_NAME}
    ```

1.  Delete the cluster:

    ```bash
    gcloud container clusters delete ${CLUSTER_NAME} --project=${PROJECT_ID}  --location=${ZONE}
    ```
