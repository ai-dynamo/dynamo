---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Quickstart
---

This guide covers running Dynamo **using the CLI on your local machine or VM**.

> [!IMPORTANT]
> **Looking to deploy on Kubernetes instead?**
> See the [Kubernetes Installation Guide](../kubernetes/installation-guide.md)
> and [Kubernetes Quickstart](../kubernetes/README.md) for cluster deployments.

## Choose Your Install Path

| Path | Best For | Guide |
|---|---|---|
| **Local Install** | Running Dynamo on a single machine or VM | [Local Installation](local-installation.md) |
| **Kubernetes** | Production multi-node cluster deployments | [Kubernetes Deployment Guide](../kubernetes/README.md) |
| **Building from Source** | Contributors and local development | [Building from Source](building-from-source.md) |

## Install Dynamo

**Option A: Containers (Recommended)**

Containers have all dependencies pre-installed. No setup required.

```bash
# SGLang
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1

# TensorRT-LLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.1

# vLLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1
```

See [Release Artifacts](../reference/release-artifacts.md#container-images) for available
versions and backend guides for run instructions: [SGLang](../backends/sglang/README.md) |
[TensorRT-LLM](../backends/trtllm/README.md) | [vLLM](../backends/vllm/README.md)

**Option B: Install from PyPI**

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
uv pip install --prerelease=allow "ai-dynamo[sglang]"
```

**TensorRT-LLM**

```bash
sudo apt install python3-dev
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]"
```

**vLLM**

```bash
sudo apt install python3-dev libxcb1
uv pip install --prerelease=allow "ai-dynamo[vllm]"
```

## Run Dynamo

Start the frontend, then start a worker for your chosen backend.

> [!TIP]
> To run in a single terminal (useful in containers), append `> logfile.log 2>&1 &`
> to run processes in background. Example: `python3 -m dynamo.frontend --discovery-backend file > dynamo.frontend.log 2>&1 &`

```bash
# Start the OpenAI compatible frontend (default port is 8000)
# --discovery-backend file avoids needing etcd. Frontend and workers must share a disk.
# The event plane automatically defaults to ZMQ (no NATS required) with this backend.
python3 -m dynamo.frontend --discovery-backend file
```

In another terminal (or same terminal if using background mode), start a worker:

**SGLang**

```bash
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file
```

**TensorRT-LLM**

```bash
python3 -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --discovery-backend file
```

**vLLM**

```bash
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

## Test Your Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50}'
```
