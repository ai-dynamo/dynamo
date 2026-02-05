---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Welcome to NVIDIA Dynamo

The NVIDIA Dynamo Platform is a high-performance, low-latency inference framework designed to serve all AI models—across any framework, architecture, or deployment scale.

> [!TIP]
> **Discover the Latest Developments!**
>
> This guide is a snapshot of a specific point in time. For the latest information, examples, and Release Assets, see the [Dynamo GitHub repository](https://github.com/ai-dynamo/dynamo/releases/latest).

## Quickstart

Get started with Dynamo locally in just a few commands:

### 1. Install Dynamo

```bash
# Install uv (recommended Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install Dynamo
uv venv venv
source venv/bin/activate
# Use prerelease flag to install RC versions of flashinfer and/or other dependencies
uv pip install --prerelease=allow "ai-dynamo[sglang]"  # or [vllm], [trtllm]
```

### 2. Start etcd/NATS

```bash
# Fetch and start etcd and NATS using Docker Compose
VERSION=$(uv pip show ai-dynamo | grep Version | cut -d' ' -f2)
curl -fsSL -o docker-compose.yml https://raw.githubusercontent.com/ai-dynamo/dynamo/refs/tags/v${VERSION}/deploy/docker-compose.yml
docker compose -f docker-compose.yml up -d
```

### 3. Run Dynamo

```bash
# Start the OpenAI compatible frontend (default port is 8000)
python -m dynamo.frontend

# In another terminal, start an SGLang worker
python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B
```

### 4. Test Your Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50}'
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Backend Support** | vLLM, SGLang, and TensorRT-LLM backends |
| **Disaggregated Serving** | Separate prefill and decode for optimal performance |
| **KV Cache Routing** | Intelligent request routing based on KV cache state |
| **Kubernetes Native** | Full operator and Helm chart support |
| **Observability** | Prometheus metrics, Grafana dashboards, and tracing |

## Documentation Overview

### Backends
- [vLLM Backend](../backends/vllm/README.md) - High-throughput serving with vLLM
- [SGLang Backend](../backends/sglang/README.md) - Fast inference with SGLang
- [TensorRT-LLM Backend](../backends/trtllm/README.md) - Optimized inference with TensorRT-LLM

### Kubernetes Deployment
- [Installation Guide updated](../kubernetes/installation-guide.md) - Deploy Dynamo on Kubernetes
- [Operator Guide](../kubernetes/dynamo-operator.md) - Using the Dynamo Operator
- [Autoscaling](../kubernetes/autoscaling.md) - Automatic scaling configuration

### Architecture
- [System Architecture](../design-docs/architecture.md) - Overall system design
- [Disaggregated Serving](../design-docs/disagg-serving.md) - P/D separation architecture
- [Distributed Runtime](../design-docs/distributed-runtime.md) - Runtime internals

### Performance & Tuning
- [Performance Tuning](../performance/tuning.md) - Optimize your deployment
- [Benchmarking](../benchmarks/benchmarking.md) - Measure and compare performance
- [AI Configurator](../performance/aiconfigurator.md) - Automated configuration

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/ai-dynamo/dynamo/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/ai-dynamo/dynamo/discussions)
- **Reference**: [CLI Reference](../reference/cli.md) | [Glossary](../reference/glossary.md) | [Support Matrix](../reference/support-matrix.md)
