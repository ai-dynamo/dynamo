---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Quickstart
---

Get a model running on Kubernetes in minutes.

## Prerequisites

- Kubernetes cluster (v1.24+) with GPU nodes
- [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) (v1.24+)
- [Helm](https://helm.sh/docs/intro/install/) (v3.0+) installed
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) installed on the cluster
- HuggingFace token secret on cluster

### HuggingFace token secret

Create a HuggingFace token secret for model downloads. If you don't have a token, see the HuggingFace [token guide](https://huggingface.co/docs/hub/en/security-tokens).

```bash
export HF_TOKEN=<your-hf-token>

kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN"
```

### GPU Operator quick install

If you don't have the GPU Operator yet:

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia --force-update
helm repo update nvidia
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace \
  --wait --timeout=600s
```

> [!TIP]
> If your cluster already provides GPU drivers (e.g., GKE with `gpu-driver-version=latest`, or AKS), add:
> ```bash
> --set driver.enabled=false --set toolkit.enabled=false
> ```

### Detailed installation

The GPU Operator is the only prerequisite for a basic deployment. For additional features like RDMA, Prometheus, or multinode scheduling with Grove/KAI Scheduler, see the [Installation Guide](installation-guide.md).

> [!TIP]
> If your GPU SKU and cloud provider are supported, you can use [AICR](https://github.com/NVIDIA/aicr) for rapid installation of prerequisites and the Dynamo Helm chart.

### Verify cluster is ready

Optionally, verify your cluster is ready:

```bash
./deploy/pre-deployment/pre-deployment-check.sh
```

## Install Dynamo

```bash
export NAMESPACE=dynamo-system
helm install dynamo-platform \
  oci://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform \
  --version "1.0.2" \
  --namespace "$NAMESPACE" \
  --create-namespace
```

Wait for the platform pods:

```bash
kubectl get pods -n $NAMESPACE
# Expected: dynamo-operator-*, etcd-*, nats-* pods all Running
```

## Deploy Your First Model

Deploy `Qwen/Qwen3-0.6B` using a DynamoGraphDeploymentRequest (DGDR).

The DGDR is the entrypoint for deploying models. It runs automatic profiling for your model/hardware and creates an auto-configured DynamoGraphDeployment (DGD). After that, the DGDR is completed and reaches a terminal state, similar to a K8s Job and can be cleaned up. The DGD is the resource that persists and serves your model.

```yaml
# qwen3-quickstart.yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-quickstart
spec:
  model: Qwen/Qwen3-0.6B
  backend: auto
  image: "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.0.2"
```

```bash
kubectl apply -f qwen3-quickstart.yaml -n $NAMESPACE
```

Watch the DGDR progress from `Pending` → `Profiling` → `Deploying` → `Deployed`:

```bash
kubectl get dgdr qwen3-quickstart -n $NAMESPACE -w
```

> [!NOTE]
> Dynamo supports vLLM, TensorRT-LLM, and SGLang backends. Setting `backend: auto` lets the profiler choose the best one for your model and hardware. See the [vLLM backend guide](../backends/vllm/README.md) for a backend guide example.


## Send a Request

Once the DGDR shows `Deployed`:

```bash
# Find and port-forward the frontend
FRONTEND_SVC=$(kubectl get svc -n $NAMESPACE -o name | grep frontend | head -1)
kubectl port-forward "$FRONTEND_SVC" 8000:8000 -n $NAMESPACE &

# Send a request
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is NVIDIA Dynamo?"}],
    "max_tokens": 200
  }' | python3 -m json.tool
```

## Cleanup

```bash
kubectl delete dgdr qwen3-quickstart -n $NAMESPACE
```

## Next Steps

- **[Installation Guide](installation-guide.md)** — Cloud provider setup, GPU Operator details, optional components (Grove, RDMA, model caching, Prometheus)
- **[Model Deployment Guide](model-deployment-guide.md)** — Strategy selection, model caching, planner, multinode, common pitfalls
- **[DGDR Reference](dgdr.md)** — Spec reference, lifecycle phases, monitoring commands, DGDR vs DGD
- **[Creating Deployments](deployment/create-deployment.md)** — Hand-craft a DGD spec for full control
