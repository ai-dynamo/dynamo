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

<p align="center">
  <a href="https://github.com/ai-dynamo/dynamo/issues/762"><b>Roadmap</b></a> &nbsp;|&nbsp;
  <a href="https://docs.nvidia.com/dynamo/latest/index.html"><b>Documentation</b></a> &nbsp;|&nbsp;
  <a href="https://github.com/ai-dynamo/examples"><b>Examples</b></a> &nbsp;|&nbsp;
  <a href="https://github.com/ai-dynamo/enhancements"><b>Design Proposals</b></a>
</p>

## NVIDIA Dynamo

**High-throughput, low-latency inference framework for serving generative AI and reasoning models in multi-node distributed environments.**

Large language models are quickly outgrowing the memory and compute budget of any single GPU. Tensor-parallelism solves the capacity problem by spreading each layer across many GPUs—and sometimes many servers—but it creates a new one: how do you coordinate those shards, route requests, and share KV cache fast enough to feel like one accelerator? This orchestration gap is exactly what NVIDIA Dynamo is built to close.

<p align="center">
  <img src="./docs/images/frontpage-architecture.png" alt="Dynamo architecture" width="600"/>
</p>

NVIDIA Dynamo is designed to be inference engine agnostic and captures LLM-specific capabilities such as:

- **Disaggregated prefill & decode inference** – Maximizes GPU throughput and facilitates trade-off between throughput and latency
- **Dynamic GPU scheduling** – Optimizes performance based on fluctuating demand  
- **LLM-aware request routing** – Eliminates unnecessary KV cache re-computation
- **Accelerated data transfer** – Reduces inference response time using NIXL
- **KV cache offloading** – Leverages multiple memory hierarchies for higher system throughput

Built in Rust for performance and in Python for extensibility, Dynamo is fully open-source and driven by a transparent, OSS (Open Source Software) first development approach.

## Framework Support Matrix

| Feature | [vLLM] | [SGLang] | [TensorRT-LLM] |
|---------|----------------------|----------------------------|----------------------------------------|
| **Disaggregated Serving** | ✅ | ✅ | ✅ |
| **Conditional Disaggregation** | ✅ | 🚧 | 🚧 |
| **KV-Aware Routing** | ✅ | ✅ | ✅ |
| **SLA-Based Planner** | ✅ | ❌ | ❌ |
| **Load Based Planner** | ✅ | ❌ | ❌ |
| **KVBM** | 🚧 | ❌ | ❌ |
| **Kubernetes Deployment** | ✅ | 🚧 | 🚧 |


To learn more about each framework and their capabilities, check out each framework's README!

- **[vLLM](examples/llm/README.md)** 
- **[SGLang](examples/sglang/README.md)** 
- **[TensorRT-LLM](examples/tensorrt_llm/README.md)** 

## Deployment Architectures

### Aggregated Serving
Single-instance deployment where both prefill and decode are handled by the same worker.

```
+------+      +-----------+      +------------------+
| HTTP |----->| processor |----->|      Worker      |
|      |<-----|           |<-----|   (Prefill +     |
+------+      +-----------+      |     Decode)      |
                                 +------------------+
```

**Best for:** Small to medium workloads, simple deployment

### Disaggregated Serving
Distributed deployment where prefill and decode are handled by separate, independently scalable workers.

```
+------+      +-----------+      +------------------+     +---------------+
| HTTP |----->| processor |----->| Decode Worker    |<--->| Prefill       |
|      |<-----|           |<-----|                  |     | Worker        |
+------+      +-----------+      +------------------+     +---------------+
                                          |
                                          v
                                 +------------------+
                                 |   Prefill Queue  |
                                 +------------------+
```

**Best for:** High throughput, independent scaling, optimized hardware utilization

### KV-Aware Routing
Intelligent request routing based on KV cache hit rates across workers.

```
+------+      +-----------+      +------------------+     +------------------+
| HTTP |----->| processor |----->|    KV Router     |---->|     Worker 1     |
|      |<-----|           |<-----|                  |---->|     Worker 2     |
+------+      +-----------+      +------------------+     |        ...       |
                                          |               +------------------+
                                          v
                                 +------------------+
                                 |   KV Indexer    |
                                 +------------------+
```

**Best for:** High cache hit rates, shared context workloads

## Installation

### Using pip
Using `pip` is our recommended way to install Dynamo.

```bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0
python3 -m venv venv
source venv/bin/activate

pip install "ai-dynamo[all]"
```

### Using conda
```bash
git clone https://github.com/ai-dynamo/dynamo.git
conda activate <ENV_NAME>
pip install nixl # Or install https://github.com/ai-dynamo/nixl from source

# To install ai-dynamo-runtime from source# To install ai-dynamo-runtime from source
cargo build --release
cd lib/bindings/python
pip install .
cd ../../../
pip install ".[all]"

# To test
docker compose -f deploy/metrics/docker-compose.yml up -d
cd examples/sglang
./launch/agg.sh
```

## Local Development

> [!NOTE]
> If you use vscode or cursor, check out our [.devcontainer setup](.devcontainer/README.md). Otherwise, to develop locally, we recommend working inside of the container.

```bash
# This builds the vllm container by default. You can change the framework by passing the --framework flag.
./container/build.sh
# This will mount your current working dynamo directory inside of the container
./container/run.sh -it --mount-workspace

# Setup dynamo
cargo build --release
mkdir -p /workspace/deploy/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/http /workspace/deploy/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/llmctl /workspace/deploy/sdk/src/dynamo/sdk/cli/bin
cp /workspace/target/release/dynamo-run /workspace/deploy/sdk/src/dynamo/sdk/cli/bin

uv pip install -e .
export PYTHONPATH=$PYTHONPATH:/workspace/deploy/sdk/src:/workspace/components/planner/src
```