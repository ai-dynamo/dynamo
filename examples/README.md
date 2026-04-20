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

- **[Quickstart](/docs/getting-started/quickstart.md)** - Simple local Dynamo setup across supported backends
- **[Disaggregated Serving](/docs/features/disaggregated-serving/README.md)** - Prefill/decode separation for enhanced performance and scalability
- **[Multi-node TensorRT-LLM](/docs/backends/trtllm/multinode/trtllm-multinode-examples.md)** - Distributed inference across multiple nodes and GPUs

## Framework Support

These examples show how Dynamo broadly works using major inference engines.

If you want to see advanced, framework-specific deployment patterns and best practices, check out the [Examples Backends](/examples/backends/) directory:
- **[vLLM](/examples/backends/vllm/)** – vLLM-specific deployment and configuration
- **[SGLang](/examples/backends/sglang/)** – SGLang integration examples and workflows
- **[TensorRT-LLM](/examples/backends/trtllm/)** – TensorRT-LLM workflows and optimizations

## Deployment Examples

Platform-specific deployment guides for production environments:

- **[Amazon EKS](/examples/deployments/EKS/)** - Deploy Dynamo on Amazon Elastic Kubernetes Service
- **[Azure AKS](/examples/deployments/AKS/)** - Deploy Dynamo on Azure Kubernetes Service
- **[Amazon ECS](/examples/deployments/ECS/)** - Deploy Dynamo on Amazon Elastic Container Service
- **[Google GKE](/examples/deployments/GKE/)** - Deploy Dynamo on Google Kubernetes Engine

## Runtime Examples

Low-level runtime examples for developers using Python<>Rust bindings:

- **[Hello World](/examples/custom_backend/hello_world/README.md)** - Minimal Dynamo runtime service demonstrating basic concepts

## By Use Case

Indexed by what you want to *do*, not which backend you want to use. Each entry links to a runnable script or YAML and to the doc that explains it.

### Serving shapes

**Aggregated single-node**

- [vLLM `launch/agg.sh`](backends/vllm/launch/agg.sh) — one process, all GPUs. Smallest-footprint starting point. → [vLLM backend guide](../docs/backends/vllm/README.md)
- [TRT-LLM `launch/agg.sh`](backends/trtllm/launch/agg.sh) — aggregated TRT-LLM engine. → [TRT-LLM backend guide](../docs/backends/trtllm/README.md)
- [SGLang `launch/agg.sh`](backends/sglang/launch/agg.sh) → [SGLang backend guide](../docs/backends/sglang/README.md)

**Multi-node tensor parallelism**

- [vLLM `launch/multi_node_tp.sh`](backends/vllm/launch/multi_node_tp.sh) → [vLLM TP walkthrough](../docs/backends/vllm/README.md)
- SGLang multi-node → [SGLang examples](../docs/backends/sglang/sglang-examples.md)

**Disaggregated prefill / decode**

- [vLLM `launch/disagg.sh`](backends/vllm/launch/disagg.sh) → architecture: [Disaggregated Serving](../docs/features/disaggregated-serving/README.md)
- [SGLang `launch/disagg.sh`](backends/sglang/launch/disagg.sh) — note the known startup race, see [Troubleshooting](../docs/troubleshooting.md#disaggsh-503-on-first-requests).
- [TRT-LLM `launch/disagg.sh`](backends/trtllm/launch/disagg.sh)

### Advanced features

**GAIE (Gateway API Inference Extension)**

- [vLLM GAIE deploy YAMLs](backends/vllm/deploy/gaie/) → [`recipes/llama-3-70b/`](../recipes/llama-3-70b/) for production-ready equivalents.

**LoRA**

- [vLLM LoRA deploy YAMLs](backends/vllm/deploy/lora/) — multi-adapter hot-loading via the frontend.

**Multimodal**

- [`custom_backend/hello_world/`](custom_backend/hello_world/) — minimal custom-backend shape.
- Multimodal streaming on vLLM: see [Troubleshooting](../docs/troubleshooting.md#multimodal-streaming-fails-on-vllm) for current limitation.

**Tracing and telemetry**

- See [`docs/observability/README.md`](../docs/observability/README.md) for the required env-vars checklist. Example manifests under `backends/<backend>/deploy/` already include the right annotations.

**Fault tolerance**

- [`tests/fault_tolerance/`](../tests/fault_tolerance/) — templated vLLM deploys used for chaos testing.

### Cloud-specific deploys

- [GKE](deployments/GKE/)
- [ECS](deployments/ECS/)

### Diffusion (image/video)

- [diffusers/](diffusers/) — Dynamo frontend + diffusion backends.

## What lives where

- **`examples/`** — learning material, scripts, single-backend deploys. Usable as a starting point; not production-hardened.
- **`recipes/`** — production-ready K8s deploys per-model, per-backend, pinned to a release tag.

## Getting Started

1. **Choose your deployment pattern**: Start with the [Quickstart](/docs/getting-started/quickstart.md) for a simple local deployment, or explore [Disaggregated Serving](/docs/features/disaggregated-serving/README.md) for advanced architectures.

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

See the [Kubernetes Installation Guide](/docs/kubernetes/installation-guide.md#prerequisites) for detailed setup instructions and pre-deployment checks.
