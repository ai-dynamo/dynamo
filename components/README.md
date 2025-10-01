<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dynamo Components

This directory contains the core components that make up the Dynamo inference framework. Each component serves a specific role in the distributed LLM serving architecture, enabling high-throughput, low-latency inference across multiple nodes and GPUs.

## Supported Inference Engines

Dynamo supports multiple inference engines (with a focus on SGLang, vLLM, and TensorRT-LLM), each with their own deployment configurations and capabilities:

- **[vLLM](backends/vllm/README.md)** - High-performance LLM inference with native KV cache events and NIXL-based transfer mechanisms
- **[SGLang](backends/sglang/README.md)** - Structured generation language framework with ZMQ-based communication
- **[TensorRT-LLM](backends/trtllm/README.md)** - NVIDIA's optimized LLM inference engine with TensorRT acceleration

Each engine provides launch scripts for different deployment patterns in their respective `/launch` & `/deploy` directories.

## Core Components

### [Backends](src/dynamo/)

The backends directory contains inference engine integrations and implementations, with a key focus on:

- **vLLM** - Full-featured vLLM integration with disaggregated serving, KV-aware routing, and SLA-based planning
- **SGLang** - SGLang engine integration supporting disaggregated serving and KV-aware routing
- **TensorRT-LLM** - TensorRT-LLM integration with disaggregated serving capabilities


### [Frontend](src/dynamo/frontend/)

The frontend component provides the HTTP API layer and request processing:

- **OpenAI-compatible HTTP server** - RESTful API endpoint for LLM inference requests
- **Pre-processor** - Handles request preprocessing and validation
- **Router** - Routes requests to appropriate workers based on load and KV cache state
- **Auto-discovery** - Automatically discovers and registers available workers

### [Planner](src/dynamo/planner/)

The planner component monitors system state and dynamically adjusts worker allocation:

- **Dynamic scaling** - Scales prefill/decode workers up and down based on metrics
- **SLA-based planning** - Ensures inference performance targets are met
- **Load-based planning** - Optimizes resource utilization based on demand

## Getting Started

To get started with Dynamo components:

1. **Choose an inference engine** from the supported backends
2. **Set up required services** (etcd and NATS) using Docker Compose
3. **Configure** your chosen engine using Python wheels or building an image
4. **Run deployment scripts** from the engine's launch directory
5. **Monitor performance** using the metrics component

For detailed instructions, see the README files in each component directory and the main [Dynamo documentation](../docs/).
