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

| **[Roadmap](https://github.com/ai-dynamo/dynamo/issues/762)** | **[Documentation](https://docs.nvidia.com/dynamo/latest/index.html)** | **[Examples](https://github.com/ai-dynamo/dynamo/tree/main/examples)** | **[Design Proposals](https://github.com/ai-dynamo/enhancements)** |

# NVIDIA Dynamo

High-throughput, low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments.

## Latest News

* [08/05] Deploy `openai/gpt-oss-120b` with disaggregated serving on NVIDIA Blackwell GPUs using Dynamo [➡️ link](./components/backends/trtllm/gpt-oss.md)

## Why Dynamo?

LLMs are quickly outgrowing the memory and compute budget of any single GPU. Tensor-parallelism solves the capacity problem by spreading each layer across many GPUs—and sometimes many servers—but it creates a new one: how do you coordinate those shards, route requests, and share KV cache fast enough to feel like one accelerator? This orchestration gap is exactly what NVIDIA Dynamo is built to close.

<p align="center">
  <img src="./docs/images/frontpage-gpu-vertical.png" alt="Multi Node Multi-GPU topology" width="600" />
</p>

Dynamo is designed to be inference engine agnostic (supports TRT-LLM, vLLM, SGLang, and others) and captures LLM-specific optimizations such as:

- **Disaggregated prefill & decode inference** – Maximizes GPU throughput and facilitates trade off between throughput and latency.
- **Dynamic GPU scheduling** – Optimizes performance based on fluctuating demand
- **LLM-aware request routing** – Eliminates unnecessary KV cache re-computation
- **Accelerated data transfer** – Reduces inference response time using NIXL
- **KV cache offloading** – Leverages multiple memory hierarchies for higher system throughput

Built in Rust for performance and in Python for extensibility, Dynamo is fully open-source and driven by a transparent, OSS (Open Source Software) first development approach.

<p align="center">
  <img src="./docs/images/frontpage-architecture.png" alt="Dynamo architecture" width="600" />
</p>

## Framework Support Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|----------------------|----------------------------|----------------------------------------|
| [**Disaggregated Serving**](/docs/architecture/disagg_serving.md) | ✅ | ✅ | ✅ |
| [**Conditional Disaggregation**](/docs/architecture/disagg_serving.md#conditional-disaggregation) | 🚧 | 🚧 | 🚧 |
| [**KV-Aware Routing**](/docs/architecture/kv_cache_routing.md) | ✅ | ✅ | ✅ |
| [**Load Based Planner**](/docs/architecture/load_planner.md) | 🚧 | 🚧 | 🚧 |
| [**SLA-Based Planner**](/docs/architecture/sla_planner.md) | ✅ | ✅ | 🚧 |
| [**KVBM**](/docs/architecture/kvbm_architecture.md) | 🚧 | 🚧 | 🚧 |

## Local Deployment

> [!NOTE]
> **Requirements:** Ubuntu 24.04 with x86_64 CPU recommended. See [Support Matrix](docs/support_matrix.md) for details.

### 1. Start control-plane services (etcd + NATS)

Dynamo uses etcd and NATS to coordinate across your infrastructure. Start them with Docker:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### 2. Install Dynamo with your preferred engine. 

We publish Python wheels specialized for each of our supported engines: vLLM, SGLang, TRT-LLM, and llama.cpp. The example that follows uses SGLang; continue reading for other engines.
```bash
# Create Python environment
uv venv venv
source venv/bin/activate
uv pip install pip

# Choose your engine
uv pip install "ai-dynamo[sglang]" # or "ai-dynamo[vllm]", "ai-dynamo[trtllm]"  
 
```

### 3. Start Dynamo services

Open two terminals:

**Terminal 1 - Start the frontend:**
```bash
python -m dynamo.frontend --http-port 8080
```

**Terminal 2 - Start an engine worker:**
```bash
# For SGLang
python -m dynamo.sglang \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --skip-tokenizer-init
```

### 4. Send your first request!

```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [{"role": "user", "content": "Hello! What is Dynamo?"}],
    "stream": false,
    "max_tokens": 64
  }'
```

**Troubleshooting:** Run `python deploy/dynamo_check.py` to verify environment connectivity and configuration.

## Kubernetes Deployment

> [!NOTE]
> **Prerequisites:**
> - Kubernetes cluster (v1.24+)
> - Helm (v3.0+)
> - kubectl configured

### 1. Install Dynamo Kubernetes Platform

```bash
# Set configuration
export NAMESPACE=dynamo-kubernetes
export RELEASE_VERSION=0.4.1

# Install CRDs
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# Install Platform
kubectl create namespace ${NAMESPACE}
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE}
```

### 2. Deploy a model

```bash
# Deploy vLLM with Qwen model
kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}
```

### 3. Access your deployment

```bash
# Port forward to access locally
kubectl port-forward svc/agg-vllm-frontend 8000:8000 -n ${NAMESPACE}

# Test the deployment
curl http://localhost:8000/v1/models
```

**Next steps:**
- Explore [Kubernetes deployment guide](docs/guides/dynamo_deploy/README.md) for production configurations
- Check out engine-specific examples: [vLLM](components/backends/vllm/), [SGLang](components/backends/sglang/), [TRT-LLM](components/backends/trtllm/)

## Inference Engines

### vLLM

```
uv pip install ai-dynamo[vllm]
```

Run the backend/worker like this:
```
python -m dynamo.vllm --help
```

vLLM attempts to allocate enough KV cache for the full context length at startup. If that does not fit in your available memory pass `--context-length <value>`.

To specify which GPUs to use set environment variable `CUDA_VISIBLE_DEVICES`.

### SGLang

```
# Install libnuma
apt install -y libnuma-dev

uv pip install ai-dynamo[sglang]
```

Run the backend/worker like this:
```
python -m dynamo.sglang.worker --help
```

You can pass any sglang flags directly to this worker, see https://docs.sglang.ai/advanced_features/server_arguments.html . See there to use multiple GPUs.

### TensorRT-LLM

It is recommended to use [NGC PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for running the TensorRT-LLM engine.

> [!Note]
> Ensure that you select a PyTorch container image version that matches the version of TensorRT-LLM you are using.
> For example, if you are using `tensorrt-llm==1.0.0rc6`, use the PyTorch container image version `25.06`.
> To find the correct PyTorch container version for your desired `tensorrt-llm` release, visit the [TensorRT-LLM Dockerfile.multi](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docker/Dockerfile.multi) on GitHub. Switch to the branch that matches your `tensorrt-llm` version, and look for the `BASE_TAG` line to identify the recommended PyTorch container tag.

> [!Important]
> Launch container with the following additional settings `--shm-size=1g --ulimit memlock=-1`

### Install prerequisites
```
# Optional step: Only required for Blackwell and Grace Hopper
uv pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Required until the trtllm version is bumped to include this pinned dependency itself
uv pip install "cuda-python>=12,<13"

sudo apt-get -y install libopenmpi-dev
```

> [!Tip]
> You can learn more about these prequisites and known issues with TensorRT-LLM pip based installation [here](https://nvidia.github.io/TensorRT-LLM/installation/linux.html).

### After installing the pre-requisites above, install Dynamo
```
uv pip install ai-dynamo[trtllm]
```

Run the backend/worker like this:
```
python -m dynamo.trtllm --help
```

To specify which GPUs to use set environment variable `CUDA_VISIBLE_DEVICES`.

## Developing Locally

<details>
<summary><b>Expand local development setup instructions</b></summary>

### 1. Install libraries

**Ubuntu:**
```
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

**macOS:**
- [Homebrew](https://brew.sh/)
```
# if brew is not installed on your system, install it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
- [Xcode](https://developer.apple.com/xcode/)

```
brew install cmake protobuf

### Check that Metal is accessible
xcrun -sdk macosx metal
```
If Metal is accessible, you should see an error like `metal: error: no input files`, which confirms it is installed correctly.


### 2. Install Rust

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 3. Create a Python virtual env:

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

### 4. Install build tools

```
uv pip install pip maturin
```

[Maturin](https://github.com/PyO3/maturin) is the Rust<->Python bindings build tool.

## 5. Build the Rust bindings

```
cd lib/bindings/python
maturin develop --uv
```

### 6. Install the wheel

```
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

If you use vscode or cursor, we have a .devcontainer folder built on [Microsofts Extension](https://code.visualstudio.com/docs/devcontainers/containers).

</details>
