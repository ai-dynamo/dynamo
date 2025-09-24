<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
![Dynamo banner](./docs/images/frontpage-banner.png)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)
[![Discord](https://dcbadge.limes.pink/api/server/D92uqZRjCZ?style=flat)](https://discord.gg/D92uqZRjCZ)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ai-dynamo/dynamo)

| **[Roadmap](https://github.com/ai-dynamo/dynamo/issues/762)** | **[Support matrix](https://github.com/ai-dynamo/dynamo/blob/main/docs/support_matrix.md)** | **[Documentation](https://docs.nvidia.com/dynamo/latest/index.html)** | **[Examples](https://github.com/ai-dynamo/dynamo/tree/main/examples)** | **[Prebuilt containers](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)** | **[Design Proposals](https://github.com/ai-dynamo/enhancements)** | **[Blogs](https://developer.nvidia.com/blog/tag/nvidia-dynamo)**

# NVIDIA Dynamo

High-throughput, low-latency inference framework for serving generative AI models across multi-node distributed environments.

Large language models are quickly outgrowing the memory and compute budget of any single GPU. Tensor-parallelism solves the capacity problem by spreading each layer across many GPUs—and sometimes many servers—but it creates a new one: how do you coordinate those shards, route requests, and share KV cache fast enough to feel like one accelerator? This orchestration gap is exactly what NVIDIA Dynamo is built to close.

Dynamo is designed to be inference engine agnostic (supports TRT-LLM, vLLM, SGLang or others) and captures LLM-specific capabilities such as:

- **Disaggregated prefill & decode inference** – Maximizes GPU throughput and facilitates trade off between throughput and latency.
- **Dynamic GPU scheduling** – Optimizes performance based on fluctuating demand
- **LLM-aware request routing** – Eliminates unnecessary KV cache re-computation
- **Accelerated data transfer** – Reduces inference response time using NIXL.
- **KV cache offloading** – Leverages multiple memory hierarchies for higher system throughput

<p align="center">
  <img src="./docs/images/frontpage-architecture.png" alt="Dynamo architecture" width="600" />
</p>

Built in Rust for performance and Python for extensibility, Dynamo is fully open-source and driven by a transparent, OSS (Open Source Software) first development approach.

## Latest News

* [0.5.0] KVBM (KV Cache Block Manager) support now available in Dynamo for enhanced memory management and KV cache offloading from HBM to remote storage

## Framework Support Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|----------------------|----------------------------|----------------------------------------|
| [**Disaggregated Serving**](/docs/architecture/disagg_serving.md) | ✅ | ✅ | ✅ |
| [**Conditional Disaggregation**](/docs/architecture/disagg_serving.md#conditional-disaggregation) | 🚧 | 🚧 | 🚧 |
| [**KV-Aware Routing**](/docs/architecture/kv_cache_routing.md) | ✅ | ✅ | ✅ |
| [**SLA-Based Planner**](/docs/architecture/sla_planner.md) | ✅ | ✅ | ✅ |
| [**KVBM**](/docs/architecture/kvbm_architecture.md) | ✅ | 🚧 | ✅ |

To learn more about each framework and their capabilities, check out each framework's README!
- **[vLLM](components/backends/vllm/README.md)**
- **[SGLang](components/backends/sglang/README.md)**
- **[TensorRT-LLM](components/backends/trtllm/README.md)**

# Quick Start

**New to Dynamo?** **[Complete Quickstart Guide](docs/quickstart.md)** (Recommended)

## Local Development

### Prerequisites
- Ubuntu 24.04 (recommended) or compatible Linux
- NVIDIA GPU with CUDA support
- Docker & Docker Compose

### 1. Install Dynamo
```bash
# Install uv (recommended Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install Dynamo
uv venv venv
source venv/bin/activate
uv pip install "ai-dynamo[sglang]"  # or [vllm], [trtllm]
```

### 2. Start Infrastructure Services
```bash
# Start etcd and NATS (required for distributed communication)
docker compose -f deploy/docker-compose.yml up -d
```

### 3. Run Your First Model
```bash
# Terminal 1: Start frontend
python -m dynamo.frontend --http-port 8000

# Terminal 2: Start backend worker
python -m dynamo.sglang --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

### 4. Test It
```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

## Kubernetes Deployment

**Production deployments** **[Kubernetes Quickstart](docs/quickstart.md#kubernetes-quickstart)**

```bash
# Install platform
export NAMESPACE=dynamo-kubernetes
export RELEASE_VERSION=0.5.0

helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace

# Deploy model (example: vLLM aggregated)
kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}

# Test the deployment
kubectl port-forward svc/agg-vllm-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

**For detailed Kubernetes deployment guide**: [Kubernetes Documentation](docs/kubernetes/README.md)

## Next Steps

- Check out [Backends](components/backends) to deploy various workflow configurations (e.g. SGLang with router, vLLM with disaggregated serving, etc.)
- Run some [Examples](examples) to learn about building components in Dynamo and exploring various integrations.

### Benchmarking Dynamo

Dynamo provides comprehensive benchmarking tools to evaluate and optimize your deployments:

* **[Benchmarking Guide](docs/benchmarks/benchmarking.md)** – Compare deployment topologies (aggregated vs. disaggregated vs. vanilla vLLM) using GenAI-Perf
* **[Pre-Deployment Profiling](docs/benchmarks/pre_deployment_profiling.md)** – Optimize configurations before deployment to meet SLA requirements

# Supported Engines

Dynamo supports multiple inference engines. Choose your preferred backend:

| Engine | Install | Run Command | Notes |
|--------|---------|-------------|-------|
| **vLLM** | `uv pip install ai-dynamo[vllm]` | `python -m dynamo.vllm --model Qwen/Qwen3-0.6B` | Use `--context-length <value>` if KV cache doesn't fit in memory. Set `CUDA_VISIBLE_DEVICES` to specify GPUs. |
| **SGLang** | `uv pip install ai-dynamo[sglang]` | `python -m dynamo.sglang --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Requires `apt install -y libnuma-dev` dependency. |
| **TensorRT-LLM** | `uv pip install ai-dynamo[trtllm]` | `python -m dynamo.trtllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | Requires NVIDIA PyTorch container. See [TensorRT-LLM Quickstart](docs/quickstart.md#tensorrt-llm-backend) for setup. |

**Detailed engine guides**: [vLLM](components/backends/vllm/README.md) | [SGLang](components/backends/sglang/README.md) | [TensorRT-LLM](components/backends/trtllm/README.md)

# Development

<details>

<summary><strong>Building from Source</strong> (Click to expand)</summary>

**For contributors and advanced users**

### Prerequisites

**Ubuntu:**
```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

**macOS:**
- [Homebrew](https://brew.sh/)
```bash
# if brew is not installed on your system, install it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
- [Xcode](https://developer.apple.com/xcode/)

```bash
brew install cmake protobuf

## Check that Metal is accessible
xcrun -sdk macosx metal
```
If Metal is accessible, you should see an error like `metal: error: no input files`, which confirms it is installed correctly.

### Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Create a Python virtual env:

Follow the instructions in [uv installation](https://docs.astral.sh/uv/#installation) guide to install uv if you don't have `uv` installed. Once uv is installed, create a virtual environment and activate it.

- Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Create a virtual environment
```bash
uv venv dynamo
source dynamo/bin/activate
```

### Install build tools

```bash
uv pip install pip maturin
```

[Maturin](https://github.com/PyO3/maturin) is the Rust<->Python bindings build tool.

### Build the Rust bindings

```bash
cd lib/bindings/python
maturin develop --uv
```

### Install the wheel

```bash
cd $PROJECT_ROOT
uv pip install .
# For development, use
export PYTHONPATH="${PYTHONPATH}:$(pwd)/components/frontend/src:$(pwd)/components/planner/src:$(pwd)/components/backends/vllm/src:$(pwd)/components/backends/sglang/src:$(pwd)/components/backends/trtllm/src:$(pwd)/components/backends/llama_cpp/src:$(pwd)/components/backends/mocker/src"
```

> [!Note]
> Editable (`-e`) does not work because the `dynamo` package is split over multiple directories, one per backend.

You should now be able to run `python -m dynamo.frontend`.

Remember that nats and etcd must be running (see earlier).

Set the environment variable `DYN_LOG` to adjust the logging level; for example, `export DYN_LOG=debug`. It has the same syntax as `RUST_LOG`.

If you use vscode or cursor, we have a .devcontainer folder built on [Microsofts Extension](https://code.visualstudio.com/docs/devcontainers/containers). For instructions see the [ReadMe](.devcontainer/README.md) for more details.

</details>