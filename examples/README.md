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

> **Want to see a specific example?**
> Open a [GitHub issue](https://github.com/ai-dynamo/dynamo/issues) to request an example you'd like to see, or [open a pull request](https://github.com/ai-dynamo/dynamo/pulls) if you'd like to contribute your own!

## Basics & Tutorials

Learn fundamental Dynamo concepts through these introductory examples:

- **[Quickstart](basics/quickstart/README.md)** - Simple aggregated serving example with vLLM backend
- **[Disaggregated Serving](basics/disaggregated_serving/README.md)** - Prefill/decode separation for enhanced performance and scalability
- **[Multi-node](basics/multinode/README.md)** - Distributed inference across multiple nodes and GPUs

## Framework Support

These examples show how Dynamo broadly works using major inference engines.

If you want to see advanced, framework-specific deployment patterns and best practices, check out the [Examples Backends](../examples/backends/) directory:
- **[vLLM](backends/vllm/)** – vLLM-specific deployment and configuration
- **[SGLang](backends/sglang/)** – SGLang integration examples and workflows
- **[TensorRT-LLM](backends/trtllm/)** – TensorRT-LLM workflows and optimizations

## Deployment Examples

Platform-specific deployment guides for production environments:

- **[Amazon EKS](deployments/EKS/)** - Deploy Dynamo on Amazon Elastic Kubernetes Service
- **[Azure AKS](deployments/AKS/)** - Deploy Dynamo on Azure Kubernetes Service
- **[Amazon ECS](deployments/ECS/)** - Deploy Dynamo on Amazon Elastic Container Service
- **[Router Standalone](deployments/router_standalone/)** - Standalone router deployment patterns
- **Google GKE** - _Coming soon_

## Runtime Examples

Low-level runtime examples for developers using Python<>Rust bindings:

- **[Hello World](custom_backend/hello_world/README.md)** - Minimal Dynamo runtime service demonstrating basic concepts

## Getting Started

1. **Choose your deployment pattern**: Start with the [Quickstart](basics/quickstart/README.md) for a simple local deployment, or explore [Disaggregated Serving](basics/disaggregated_serving/README.md) for advanced architectures.

2. **Set up prerequisites**: Most examples require etcd and NATS services. You can start them using:
   ```bash
   docker compose -f deploy/docker-compose.yml up -d
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

See the [Kubernetes Installation Guide](../docs/kubernetes/installation_guide.md#prerequisites) for detailed setup instructions and pre-deployment checks.
