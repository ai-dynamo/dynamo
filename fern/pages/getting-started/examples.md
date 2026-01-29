---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Examples"
---

Explore practical examples to get started with NVIDIA Dynamo.

## Quick Start Examples

The [examples directory](https://github.com/ai-dynamo/dynamo/tree/main/examples) in the Dynamo repository contains ready-to-run examples for various use cases.

### Backend Examples

| Backend | Description | Link |
|---------|-------------|------|
| **vLLM** | Run inference with vLLM backend | [examples/backends/vllm](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm) |
| **SGLang** | Run inference with SGLang backend | [examples/backends/sglang](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang) |
| **TensorRT-LLM** | Run inference with TensorRT-LLM backend | [examples/backends/trtllm](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm) |

### Deployment Examples

| Example | Description | Link |
|---------|-------------|------|
| **Basic Deployment** | Simple single-node deployment | [examples/basics](https://github.com/ai-dynamo/dynamo/tree/main/examples/basics) |
| **Kubernetes** | Deploy on Kubernetes | [examples/deployments](https://github.com/ai-dynamo/dynamo/tree/main/examples/deployments) |
| **Multimodal** | Vision and multimodal models | [examples/multimodal](https://github.com/ai-dynamo/dynamo/tree/main/examples/multimodal) |

### Custom Backend Examples

Learn how to create custom backends:

| Example | Description | Link |
|---------|-------------|------|
| **Custom Backend** | Build your own backend | [examples/custom_backend](https://github.com/ai-dynamo/dynamo/tree/main/examples/custom_backend) |

## Running Examples

Most examples can be run directly after installing Dynamo:

```bash
# Clone the repository
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

# Navigate to an example
cd examples/backends/sglang

# Follow the README in each example directory
```
