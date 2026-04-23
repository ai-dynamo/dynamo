---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
sidebar-title: Local Installation
description: Install and run Dynamo on a local machine or VM with containers or PyPI
---

# Local Installation

This guide walks through installing and running Dynamo on a local machine or VM with one or more GPUs. By the end, you'll have a working OpenAI-compatible endpoint serving a model.

For production multi-node clusters, see the [Kubernetes Deployment Guide](../kubernetes/README.md). To build from source for development, see [Building from Source](building-from-source.md).

## System Requirements

| Requirement | Supported |
|---|---|
| **GPU** | NVIDIA Ampere, Ada Lovelace, Hopper, Blackwell |
| **OS** | Ubuntu 22.04, Ubuntu 24.04 |
| **Architecture** | x86_64, ARM64 (ARM64 requires Ubuntu 24.04) |
| **CUDA** | 12.9+ or 13.0+ (B300/GB300 require CUDA 13) |
| **Python** | 3.10, 3.12 |
| **Driver** | 575.51.03+ (CUDA 12) or 580.00.03+ (CUDA 13) |

TensorRT-LLM does not support Python 3.11.

For the full compatibility matrix including backend framework versions, see the [Support Matrix](../reference/support-matrix.md).

## Install Dynamo

### Option A: Containers (Recommended)

Containers have all dependencies pre-installed. No setup required.

```bash
# SGLang
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1

# TensorRT-LLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.1

# vLLM
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1
```

To run frontend and worker in the same container, either:

- Run processes in background with `&` (see Run Dynamo section below), or
- Open a second terminal and use `docker exec -it <container_id> bash`

See [Release Artifacts](../reference/release-artifacts.md#container-images) for available
versions and backend guides for run instructions: [SGLang](../backends/sglang/README.md) |
[TensorRT-LLM](../backends/trtllm/README.md) | [vLLM](../backends/vllm/README.md)

### Option B: Install from PyPI

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

For CUDA 13 (B300/GB300), the container is recommended. See
[SGLang install docs](https://docs.sglang.io/get_started/install.html) for details.

**TensorRT-LLM**

```bash
sudo apt install python3-dev
pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu130
pip install --pre --extra-index-url https://pypi.nvidia.com "ai-dynamo[trtllm]"
```

TensorRT-LLM requires `pip` due to a transitive Git URL dependency that
`uv` doesn't resolve. We recommend using the TensorRT-LLM container for
broader compatibility. See the [TRT-LLM backend guide](../backends/trtllm/README.md)
for details.

**vLLM**

```bash
sudo apt install python3-dev libxcb1
uv pip install --prerelease=allow "ai-dynamo[vllm]"
```

## Run Dynamo

### Discovery Backend

Dynamo components discover each other through a shared backend. Two options are available:

| Backend | When to Use | Setup |
|---|---|---|
| **File** | Single machine, local development | No setup -- pass `--discovery-backend file` to all components. The event plane automatically defaults to ZMQ (no NATS required). |
| **etcd** | Multi-node, production | Requires a running etcd instance (default if no flag is specified). The event plane defaults to NATS. |

This guide uses `--discovery-backend file`. For etcd setup, see [Service Discovery](../kubernetes/service-discovery.md).

### Verify Installation (Optional)

Verify the CLI is installed and callable:

```bash
python3 -m dynamo.frontend --help
```

If you cloned the repository, you can run additional system checks:

```bash
python3 deploy/sanity_check.py
```

### Start the Frontend

```bash
# Start the OpenAI compatible frontend (default port is 8000)
python3 -m dynamo.frontend --discovery-backend file
```

To run in a single terminal (useful in containers), append `> logfile.log 2>&1 &`
to run processes in background:

```bash
python3 -m dynamo.frontend --discovery-backend file > dynamo.frontend.log 2>&1 &
```

### Start a Worker

In another terminal (or same terminal if using background mode), start a worker for your chosen backend:

**SGLang**

```bash
python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file
```

**TensorRT-LLM**

```bash
python3 -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --discovery-backend file
```

The warning `Cannot connect to ModelExpress server/transport error. Using direct download.`
is expected in local deployments and can be safely ignored.

**vLLM**

```bash
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file \
  --kv-events-config '{"enable_kv_cache_events": false}'
```

### KV Events Configuration

For dependency-free local development, disable KV event publishing (avoids NATS):

- **vLLM:** Add `--kv-events-config '{"enable_kv_cache_events": false}'`
- **SGLang:** No flag needed (KV events disabled by default)
- **TensorRT-LLM:** No flag needed (KV events disabled by default)

KV events are disabled by default for all backends. Add `--kv-events-config` explicitly only when you want KV event publishing enabled.

## Test Your Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50}'
```

## Troubleshooting

**CUDA/driver version mismatch**

Run `nvidia-smi` to check your driver version. Dynamo requires driver 575.51.03+ for CUDA 12 or 580.00.03+ for CUDA 13. B300/GB300 GPUs require CUDA 13. See the [Support Matrix](../reference/support-matrix.md) for full requirements.

**Model doesn't fit on GPU (OOM)**

The default model `Qwen/Qwen3-0.6B` requires ~2GB of GPU memory. Larger models need more VRAM:

| Model Size | Approximate VRAM |
|---|---|
| 7B | 14-16 GB |
| 13B | 26-28 GB |
| 70B | 140+ GB (multi-GPU) |

Start with a small model and scale up based on your hardware.

**Python 3.11 with TensorRT-LLM**

TensorRT-LLM does not support Python 3.11. If you see installation failures with TensorRT-LLM, check your Python version with `python3 --version`. Use Python 3.10 or 3.12 instead.

**Container runs but GPU not detected**

Ensure you passed `--gpus all` to `docker run`. Without this flag, the container won't have access to GPUs:

```bash
# Correct
docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1

# Wrong -- no GPU access
docker run --network host --rm -it nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.0.1
```

## Next Steps

- [Backend Guides](../backends/sglang/README.md) -- Backend-specific configuration and features
- [Disaggregated Serving](../features/disaggregated-serving/README.md) -- Scale prefill and decode independently
- [KV Cache Aware Routing](../components/router/router-guide.md) -- Smart request routing
- [Kubernetes Deployment](../kubernetes/README.md) -- Production multi-node deployments
