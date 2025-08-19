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
**[Dynamo Kubernetes Platform](dynamo_cloud.md)** - Main installation guide with 3 paths

## 2. Choose Your Backend

Each backend has deployment examples and configuration options:

| Backend | Available Configurations |
|---------|--------------------------|
| **[vLLM](../../components/backends/vllm/deploy/README.md)** | Aggregated, Aggregated + Router, Disaggregated, Disaggregated + Router, Disaggregated + Planner |
| **[SGLang](../../components/backends/sglang/deploy/README.md)** | Aggregated, Aggregated + Router, Disaggregated, Disaggregated + Planner, Disaggregated Multi-node |
| **[TensorRT-LLM](../../components/backends/trtllm/deploy/README.md)** | Aggregated, Aggregated + Router, Disaggregated, Disaggregated + Router | 

## 3. Deploy Your First Model

```bash
# Set same namespace from platform install
export NAMESPACE=dynamo-cloud

# Deploy any example (this uses vLLM with Qwen model using aggregated serving)
kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}

# Check status
kubectl get dynamoGraphDeployment -n ${NAMESPACE}

# Test it
kubectl port-forward svc/agg-vllm-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

## What's a DynamoGraphDeployment?

It's a Kubernetes Custom Resource that defines your inference pipeline:
- Model configuration
- Resource allocation (GPUs, memory)
- Scaling policies
- Frontend/backend connections

Example structure:
```yaml
apiVersion: dynamo.ai.nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-llm
spec:
  frontends:
    - type: http
      port: 8000
  backends:
    - type: vllm
      model: "Qwen/Qwen2-0.5B"
      gpus: 1
```

## Additional Resources

- **[Examples](../../examples/README.md)** - Complete working examples
- **[Create Custom Deployments](create_deployment.md)** - Build your own CRDs
- **[Operator Documentation](dynamo_operator.md)** - How the platform works
- **[Helm Charts](../../deploy/helm/README.md)** - For advanced users

