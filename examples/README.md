<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Dynamo Examples

This directory contains practical examples demonstrating how to deploy and use Dynamo for distributed LLM inference. Each example includes setup instructions, configuration files, and explanations to help you understand different deployment patterns and use cases.

> [!IMPORTANT]
> All DynamoGraphDeployment manifests use `nvidia.com/v1beta1`. Other custom resources that do not
> have a `v1beta1` API, such as `DynamoModel`, continue to use their supported version.
> To migrate an existing `v1alpha1` manifest, follow the
> [API version converter instructions](../deploy/utils/README.md).

**Want to see a specific example?**
Open a [GitHub issue](https://github.com/ai-dynamo/dynamo/issues) to request an example you'd like to see, or [open a pull request](https://github.com/ai-dynamo/dynamo/pulls) if you'd like to contribute your own!

## Basics & Tutorials

Learn fundamental Dynamo concepts through these introductory examples:

- **[Quickstart](https://docs.nvidia.com/dynamo/getting-started/quickstart)** - Simple local Dynamo setup across supported backends
- **[Disaggregated Serving](../docs/fern/pages/kubernetes/disaggregated-serving/overview.md)** - Prefill/decode separation for enhanced performance and scalability
- **[Multi-node TensorRT-LLM](../docs/fern/pages/developer-guide/additional-resources/tensorrt-llm-details/multinode-examples.md)** - Distributed inference across multiple nodes and GPUs

## Framework Support

These examples show how Dynamo broadly works using major inference engines.

If you want to see advanced, framework-specific deployment patterns and best practices, check out the [Examples Backends](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends) directory:
- **[vLLM](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm)** – vLLM-specific deployment and configuration
- **[SGLang](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang)** – SGLang integration examples and workflows
- **[TensorRT-LLM](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm)** – TensorRT-LLM workflows and optimizations

## Deployment Examples

Platform-specific manifests and templates for production environments. Deployment guides live under `docs/kubernetes/cloud-providers/`; each examples folder links to its guide.

- **[Amazon EKS](deployments/EKS/README.md)** - Manifests and templates ([deployment guide](../docs/fern/pages/kubernetes/installation/managed-kubernetes/eks/eks-setup.mdx))
- **[Azure AKS](deployments/AKS/README.md)** - Helm values ([deployment guide](../docs/fern/pages/kubernetes/installation/managed-kubernetes/azure/aks-setup.mdx))
- **[Amazon ECS](deployments/ECS/README.md)** - Task definitions ([deployment guide](../docs/fern/pages/kubernetes/installation/managed-kubernetes/eks/ecs.mdx))
- **[Google GKE](deployments/GKE/README.md)** - DGD manifests ([deployment guide](../docs/fern/pages/kubernetes/installation/managed-kubernetes/gcp/gke-setup.mdx))

## Runtime Examples

Low-level runtime examples for developers using Python<>Rust bindings:

- **[Hello World](custom_backend/hello_world/README.md)** - Minimal Dynamo runtime service demonstrating basic concepts

- **[Slime External Rollouts](rl/slime/README.md)** - Slime training with Dynamo SGLang routing, a shared `/generate` endpoint, and health-filtered worker controls

## Getting Started

1. **Choose your deployment pattern**: Start with the [Quickstart](https://docs.nvidia.com/dynamo/getting-started/quickstart) for a simple local deployment, or explore [Disaggregated Serving](../docs/fern/pages/kubernetes/disaggregated-serving/overview.md) for advanced architectures.

2. **Set up prerequisites**: Most examples require etcd and NATS services. You can start them using:
   ```bash
   docker compose -f dev/docker-compose.yml up -d
   ```

3. **Follow the example**: Each directory contains detailed setup instructions and configuration files specific to that deployment pattern.

## Prerequisites

Before running any examples, ensure you have:

- **Docker & Docker Compose** - For containerized services
- **CUDA-compatible GPU** - For LLM inference (except hello_world, which is non-GPU aware)
- **Python 3.9+** - For client scripts and utilities

### For Kubernetes Deployments

If you're running Kubernetes/cloud deployment examples (EKS, AKS, GKE), you'll also need:

| Tool | Minimum Version | Installation |
|------|-----------------|--------------|
| **kubectl** | v1.24+ | [Install kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) |
| **Helm** | v3.0+ | [Install Helm](https://helm.sh/docs/intro/install/) |

See the [Kubernetes Installation Guide](../docs/fern/pages/kubernetes/installation/install-dynamo.md#prerequisites) for detailed setup instructions and pre-deployment checks.
