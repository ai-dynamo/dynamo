<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Deploying Inference Graphs to Kubernetes

High-level guide to Dynamo Kubernetes deployments. Start here, then dive into specific guides.

## 1. Install Platform First

```bash
# 1. Set environment
export NAMESPACE=dynamo-system
export RELEASE_VERSION=0.x.x # any version of Dynamo 0.3.2+ listed at https://github.com/ai-dynamo/dynamo/releases

# 2. Install CRDs
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# 3. Install Platform
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace
```

For more details or customization options (including multinode deployments), see **[Installation Guide for Dynamo Kubernetes Platform](/docs/kubernetes/installation_guide.md)**.

## 2. Choose Your Backend

Each backend has deployment examples and configuration options:

| Backend | Available Configurations |
|---------|--------------------------|
| **[vLLM](/components/backends/vllm/deploy/README.md)** | Aggregated, Aggregated + Router, Disaggregated, Disaggregated + Router, Disaggregated + Planner, Disaggregated Multi-node |
| **[SGLang](/components/backends/sglang/deploy/README.md)** | Aggregated, Aggregated + Router, Disaggregated, Disaggregated + Planner, Disaggregated Multi-node |
| **[TensorRT-LLM](/components/backends/trtllm/deploy/README.md)** | Aggregated, Aggregated + Router, Disaggregated, Disaggregated + Router, Disaggregated Multi-node |

## 3. Deploy Your First Model

```bash
export NAMESPACE=dynamo-cloud
kubectl create namespace ${NAMESPACE}

# to pull model from HF
export HF_TOKEN=<Token-Here>
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n ${NAMESPACE};

# Deploy any example (this uses vLLM with Qwen model using aggregated serving)
kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}

# Check status
kubectl get dynamoGraphDeployment -n ${NAMESPACE}

# Test it
kubectl port-forward svc/vllm-agg-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

## What's a DynamoGraphDeployment (DGD)?

It's a Kubernetes Custom Resource that defines your inference pipeline:
- Model configuration
- Resource allocation (GPUs, memory)
- Scaling policies
- Frontend/backend connections

Refer to the [API Reference and Documentation](/docs/kubernetes/api_reference.md) for more details.

## 📖 API Reference & Documentation

For detailed technical specifications of Dynamo's Kubernetes resources:

- **[API Reference](/docs/kubernetes/api_reference.md)** - Complete CRD field specifications for `DynamoGraphDeployment` and `DynamoComponentDeployment`
- **[Operator Guide](/docs/kubernetes/dynamo_operator.md)** - Dynamo operator configuration and management
- **[Create Deployment](/docs/kubernetes/create_deployment.md)** - Step-by-step deployment creation examples

### Choosing Your Architecture Pattern

When creating a deployment, select the architecture pattern that best fits your use case:

- **Development / Testing** - Use `agg.yaml` as the base configuration
- **Production with Load Balancing** - Use `agg_router.yaml` to enable scalable, load-balanced inference
- **High Performance / Disaggregated** - Use `disagg_router.yaml` for maximum throughput and modular scalability

### Frontend and Worker Components

You can run the Frontend on one machine (e.g., a CPU node) and workers on different machines (GPU nodes). The Frontend serves as a framework-agnostic HTTP entry point that:

- Provides OpenAI-compatible `/v1/chat/completions` endpoint
- Auto-discovers backend workers via etcd
- Routes requests and handles load balancing
- Validates and preprocesses requests

### Customizing Your Deployment

Example structure:
```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm
spec:
  services:
    Frontend:
      dynamoNamespace: my-llm
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: your-image
    VllmDecodeWorker:  # or SGLangDecodeWorker, TrtllmDecodeWorker
      dynamoNamespace: dynamo-dev
      componentType: worker
      replicas: 1
      envFromSecret: hf-token-secret  # for HuggingFace models
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: your-image
          command: ["/bin/sh", "-c"]
          args:
            - python3 -m dynamo.vllm --model YOUR_MODEL [--your-flags]
```

Worker command examples per backend:
```yaml
# vLLM worker
args:
  - python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B

# SGLang worker
args:
  - >-
    python3 -m dynamo.sglang
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    --tp 1
    --trust-remote-code

# TensorRT-LLM worker
args:
  - python3 -m dynamo.trtllm
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    --extra-engine-args engine_configs/agg.yaml
```

Key customization points include:
- **Model Configuration**: Specify model in the args command
- **Resource Allocation**: Configure GPU requirements under `resources.limits`
- **Scaling**: Set `replicas` for number of worker instances
- **Routing Mode**: Enable KV-cache routing by setting `DYN_ROUTER_MODE=kv` in Frontend envs
- **Worker Specialization**: Add `--is-prefill-worker` flag for disaggregated prefill workers

## Additional Resources

- **[Examples](/examples/README.md)** - Complete working examples
- **[Create Custom Deployments](/docs/kubernetes/create_deployment.md)** - Build your own CRDs
- **[Operator Documentation](/docs/kubernetes/dynamo_operator.md)** - How the platform works
- **[Helm Charts](/deploy/helm/README.md)** - For advanced users
- **[GitOps Deployment with FluxCD](/docs/kubernetes/fluxcd.md)** - For advanced users
- **[Logging](/docs/kubernetes/logging.md)** - For logging setup
- **[Multinode Deployment](/docs/kubernetes/multinode-deployment.md)** - For multinode deployment
- **[Grove](/docs/kubernetes/grove.md)** - For grove details and custom installation
- **[Monitoring](/docs/kubernetes/metrics.md)** - For monitoring setup
- **[Model Caching with Fluid](/docs/kubernetes/model_caching_with_fluid.md)** - For model caching with Fluid
