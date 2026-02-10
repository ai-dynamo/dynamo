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

![Dynamo banner](./docs/images/frontpage-banner.png)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ai-dynamo/dynamo)
[![Discord](https://dcbadge.limes.pink/api/server/D92uqZRjCZ?style=flat)](https://discord.gg/D92uqZRjCZ) ![Community Contributors](https://img.shields.io/badge/community_contributors-70%2B-brightgreen)

| **[Roadmap](https://github.com/ai-dynamo/dynamo/issues/5506)** | **[Support Matrix](https://github.com/ai-dynamo/dynamo/blob/main/docs/reference/support-matrix.md)** | **[Docs](https://docs.nvidia.com/dynamo/latest/index.html)** | **[Recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes)** | **[Examples](https://github.com/ai-dynamo/dynamo/tree/main/examples)** | **[Prebuilt Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)** | **[Design Proposals](https://github.com/ai-dynamo/enhancements)** | **[Blogs](https://developer.nvidia.com/blog/tag/nvidia-dynamo)**

# NVIDIA Dynamo

High-throughput, low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments.

## Why Dynamo

<p align="center">
  <img src="./docs/images/frontpage-gpu-vertical.png" alt="Multi Node Multi-GPU topology" width="600" />
</p>

Large language models exceed single-GPU capacity. Tensor parallelism spreads layers across GPUs but creates coordination challenges. Dynamo closes this orchestration gap.

Dynamo is inference engine agnostic (supports TRT-LLM, vLLM, SGLang) and provides:

- **Disaggregated Prefill & Decode** – Maximizes GPU throughput with latency/throughput trade-offs
- **Dynamic GPU Scheduling** – Optimizes performance based on fluctuating demand
- **LLM-Aware Request Routing** – Eliminates unnecessary KV cache re-computation
- **Accelerated Data Transfer** – Reduces inference response time using NIXL
- **KV Cache Offloading** – Leverages multiple memory hierarchies for higher throughput

Built in Rust for performance and Python for extensibility, Dynamo is fully open-source with an OSS-first development approach.

## Backend Feature Support

| | [SGLang](docs/backends/sglang/README.md) | [TensorRT-LLM](docs/backends/trtllm/README.md) | [vLLM](docs/backends/vllm/README.md) |
|---|:----:|:----------:|:--:|
| **Best For** | High-throughput serving | Maximum performance | Broadest feature coverage |
| [**Disaggregated Serving**](docs/design_docs/disagg_serving.md) | ✅ | ✅ | ✅ |
| [**KV-Aware Routing**](docs/components/router/README.md) | ✅ | ✅ | ✅ |
| [**SLA-Based Planner**](docs/components/planner/planner_guide.md) | ✅ | ✅ | ✅ |
| [**KVBM**](docs/components/kvbm/README.md) | 🚧 | ✅ | ✅ |
| [**Multimodal**](docs/features/multimodal/README.md) | ✅ | ✅ | ✅ |
| [**Tool Calling**](docs/agents/tool-calling.md) | ✅ | ✅ | ✅ |

> **[Full Feature Matrix →](docs/reference/feature-matrix.md)** — Detailed compatibility including LoRA, Request Migration, Speculative Decoding, and feature interactions.

## Dynamo Architecture

<p align="center">
  <img src="./docs/images/frontpage-architecture.png" alt="Dynamo architecture" width="600" />
</p>

> **[Architecture Deep Dive →](docs/design_docs/architecture.md)**

## Latest News

- [12/05] [Moonshot AI's Kimi K2 achieves 10x inference speedup with Dynamo on GB200](https://quantumzeitgeist.com/kimi-k2-nvidia-ai-ai-breakthrough/)
- [12/02] [Mistral AI runs Mistral Large 3 with 10x faster inference using Dynamo](https://www.marktechpost.com/2025/12/02/nvidia-and-mistral-ai-bring-10x-faster-inference-for-the-mistral-3-family-on-gb200-nvl72-gpu-systems/)
- [12/01] [InfoQ: NVIDIA Dynamo simplifies Kubernetes deployment for LLM inference](https://www.infoq.com/news/2025/12/nvidia-dynamo-kubernetes/)

## Get Started

| Path | Use Case | Time | Requirements |
|------|----------|------|--------------|
| [**Local Quick Start**](#local-quick-start) | Test on a single machine | ~5 min | 1 GPU, Ubuntu 24.04 |
| [**Kubernetes Deployment**](#kubernetes-deployment) | Production multi-node clusters | ~30 min | K8s cluster with GPUs |
| [**Building from Source**](#building-from-source) | Contributors and development | ~15 min | Ubuntu, Rust, Python |

Want to help shape the future of distributed LLM inference? See the **[Contributing Guide](CONTRIBUTING.md)**.

# Local Quick Start

The following examples require a few system level packages.
Recommended to use Ubuntu 24.04 with a x86_64 CPU. See [docs/reference/support-matrix.md](docs/reference/support-matrix.md)

## Install Dynamo

### Option A: Containers (Recommended)

Containers have all dependencies pre-installed. No setup required.

```bash
# SGLang
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.8.1

# TensorRT-LLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1

# vLLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.8.1
```

> **Tip:** To run frontend and worker in the same container, either run processes in background with `&` (see below), or open a second terminal and use `docker exec -it <container_id> bash`.

See [Release Artifacts](docs/reference/release-artifacts.md#container-images) for available versions.

### Option B: Install from PyPI

The Dynamo team recommends the `uv` Python package manager, although any way works.

```bash
# Install uv (recommended Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv venv
source venv/bin/activate
uv pip install pip
```

Install system dependencies and the Dynamo wheel for your chosen backend:

**SGLang**

```bash
sudo apt install python3-dev
uv pip install "ai-dynamo[sglang]"
```

> **Note:** For CUDA 13 (B300/GB300), the container is recommended. See [SGLang install docs](https://docs.sglang.io/get_started/install.html) for details.

**TensorRT-LLM**

```bash
sudo apt install python3-dev
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]"
```

> **Note:** TensorRT-LLM requires `pip` due to a transitive Git URL dependency that `uv` doesn't resolve. We recommend using the [TensorRT-LLM container](docs/reference/release-artifacts.md#container-images) for broader compatibility.

**vLLM**

```bash
sudo apt install python3-dev libxcb1
uv pip install "ai-dynamo[vllm]"
```

## Run Dynamo

> **Tip (Optional):** Before running Dynamo, verify your system configuration with `python3 deploy/sanity_check.py`

Dynamo provides a simple way to spin up a local set of inference components including:

- **OpenAI Compatible Frontend** – High performance OpenAI compatible http api server written in Rust.
- **Basic and Kv Aware Router** – Route and load balance traffic to a set of workers.
- **Workers** – Set of pre-configured LLM serving engines.

Start the frontend:

> **Tip:** To run in a single terminal (useful in containers), append `> logfile.log 2>&1 &` to run processes in background. Example: `python3 -m dynamo.frontend --store-kv file > dynamo.frontend.log 2>&1 &`

```bash
# Start an OpenAI compatible HTTP server with prompt templating, tokenization, and routing.
# For local dev: --store-kv file avoids etcd (workers and frontend must share a disk)
python3 -m dynamo.frontend --http-port 8000 --store-kv file
```

In another terminal (or same terminal if using background mode), start a worker for your chosen backend:

```bash
# SGLang
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --store-kv file

# TensorRT-LLM
python3 -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --store-kv file

# vLLM (note: uses --model, not --model-path)
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --store-kv file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

> **Note:** For dependency-free local development, disable KV event publishing (avoids NATS):
> - **vLLM:** Add `--kv-events-config '{"enable_kv_cache_events": false}'`
> - **SGLang:** No flag needed (KV events disabled by default)
> - **TensorRT-LLM:** No flag needed (KV events disabled by default)
>
> **TensorRT-LLM only:** The warning `Cannot connect to ModelExpress server/transport error. Using direct download.` is expected and can be safely ignored.
>
> See [Service Discovery and Messaging](#service-discovery-and-messaging) for details.

#### Send a Request

```bash
curl localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq
```

Rerun with `curl -N` and change `stream` in the request to `true` to get the responses as soon as the engine issues them.

# Kubernetes Deployment

For production deployments on Kubernetes clusters with multiple GPUs.

## Prerequisites

- Kubernetes cluster with GPU nodes
- [Dynamo Platform installed](docs/kubernetes/README.md)
- HuggingFace token for model downloads

## Production Recipes

Pre-built deployment configurations for common models and topologies:

| Model | Framework | Mode | GPUs | Recipe |
|-------|-----------|------|------|--------|
| Llama-3-70B | vLLM | Aggregated | 4x H100 | [View](recipes/llama-3-70b/vllm/) |
| DeepSeek-R1 | SGLang | Disaggregated | 8x H200 | [View](recipes/deepseek-r1/sglang/) |
| Qwen3-32B-FP8 | TensorRT-LLM | Aggregated | 8x GPU | [View](recipes/qwen3-32b-fp8/trtllm/) |

See [recipes/README.md](recipes/README.md) for the full list and deployment instructions.

## Cloud Deployment Guides

- [Amazon EKS](examples/deployments/EKS/)
- [Google GKE](examples/deployments/GKE/)

# Building from Source

For contributors who want to build Dynamo from source rather than installing from PyPI.

## 1. Install Libraries

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

## Check that Metal is accessible
xcrun -sdk macosx metal
```

If Metal is accessible, you should see an error like `metal: error: no input files`, which confirms it is installed correctly.

## 2. Install Rust

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## 3. Create a Python Virtual Environment

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

## 4. Install Build Tools

```
uv pip install pip maturin
```

[Maturin](https://github.com/PyO3/maturin) is the Rust<->Python bindings build tool.

## 5. Build the Rust Bindings

```
cd lib/bindings/python
maturin develop --uv
```

## 6. Install GPU Memory Service

The GPU Memory Service is a Python package with a C++ extension. It requires only Python development headers and a C++ compiler (g++).

```bash
cd $PROJECT_ROOT
uv pip install -e lib/gpu_memory_service
```

## 7. Install the Wheel

```
cd $PROJECT_ROOT
uv pip install -e .
```

## 8. Run the Frontend

```bash
python3 -m dynamo.frontend
```

## 9. Configure for Local Development

- Pass `--store-kv file` to avoid external dependencies (see [Service Discovery and Messaging](#service-discovery-and-messaging))
- Set `DYN_LOG` to adjust the logging level (e.g., `export DYN_LOG=debug`). Uses the same syntax as `RUST_LOG`

> **Note:** VSCode and Cursor users can use the `.devcontainer` folder for a pre-configured dev environment. See the [devcontainer README](.devcontainer/README.md) for details.

# Advanced Topics

## Benchmarking

Dynamo provides comprehensive benchmarking tools:

- **[Benchmarking Guide](docs/benchmarks/benchmarking.md)** – Compare deployment topologies using AIPerf
- **[SLA-Driven Deployments](docs/components/planner/planner_guide.md)** – Optimize deployments to meet SLA requirements

## Frontend OpenAPI Specification

The OpenAI-compatible frontend exposes an OpenAPI 3 spec at `/openapi.json`. To generate without running the server:

```bash
cargo run -p dynamo-llm --bin generate-frontend-openapi
```

This writes to `docs/reference/api/openapi.json`.

## Service Discovery and Messaging

Dynamo uses TCP for inter-component communication. On Kubernetes, native resources ([CRDs + EndpointSlices](docs/kubernetes/service_discovery.md)) handle service discovery. External services are optional for most deployments:

| Deployment | etcd | NATS | Notes |
|------------|------|------|-------|
| **Local Development** | ❌ Not required | ❌ Not required | Pass `--store-kv file`; vLLM also needs `--kv-events-config '{"enable_kv_cache_events": false}'` |
| **Kubernetes** | ❌ Not required | ❌ Not required | K8s-native discovery; TCP request plane |

> **Note:** KV-Aware Routing requires NATS for prefix caching coordination.

For Slurm or other distributed deployments (and KV-aware routing):

- [etcd](https://etcd.io/) can be run directly as `./etcd`.
- [nats](https://nats.io/) needs JetStream enabled: `nats-server -js`.

To quickly setup both: `docker compose -f deploy/docker-compose.yml up -d`

See [SGLang on Slurm](examples/backends/sglang/slurm_jobs/README.md) and [TRT-LLM on Slurm](examples/basics/multinode/trtllm/README.md) for deployment examples.

## More News

- [11/20] [Dell integrates PowerScale with Dynamo's NIXL for 19x faster TTFT](https://www.dell.com/en-us/dt/corporate/newsroom/announcements/detailpage.press-releases~usa~2025~11~dell-technologies-and-nvidia-advance-enterprise-ai-innovation.htm)
- [11/20] [WEKA partners with NVIDIA on KV cache storage for Dynamo](https://siliconangle.com/2025/11/20/nvidia-weka-kv-cache-solution-ai-inferencing-sc25/)
- [11/13] [Dynamo Office Hours Playlist](https://www.youtube.com/playlist?list=PL5B692fm6--tgryKu94h2Zb7jTFM3Go4X)
- [10/16] [How Baseten achieved 2x faster inference with NVIDIA Dynamo](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/)

<!-- Reference links for Feature Compatibility Matrix -->
[disagg]: docs/design_docs/disagg_serving.md
[kv-routing]: docs/components/router/README.md
[planner]: docs/components/planner/planner_guide.md
[kvbm]: docs/components/kvbm/README.md
[mm]: examples/multimodal/
[migration]: docs/fault_tolerance/request_migration.md
[lora]: examples/backends/vllm/deploy/lora/README.md
[tools]: docs/agents/tool-calling.md
