---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang
---

## Use the Latest Release

We recommend using the [latest stable release](https://github.com/ai-dynamo/dynamo/releases/latest) of Dynamo to avoid breaking changes.

---

Dynamo SGLang integrates [SGLang](https://github.com/sgl-project/sglang) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation while maintaining full compatibility with SGLang's native engine arguments. It supports LLM inference, embedding models, multimodal vision models, and diffusion-based generation (LLM, image, video).

## Prerequisites

- **CUDA toolkit headers** for bare-metal builds (e.g. `nvcc`, `cuda_runtime.h`). See [CUDA Requirements](../../getting-started/local-installation.md#system-requirements). Not required when running the pre-built `sglang-runtime` container.
- **`HF_TOKEN`** for gated models. Export it on every node that pulls the model weights, and accept the model license on the Hugging Face model page before launch:

  ```bash
  export HF_TOKEN=hf_...
  ```

## Installation

### Install Latest Release

We recommend using [uv](https://github.com/astral-sh/uv) to install:

```bash
uv venv --python 3.12 --seed
uv pip install --prerelease=allow "ai-dynamo[sglang]"
```

This installs the latest stable release of Dynamo with the compatible SGLang version.

### Install for Development

<Accordion title="Development installation in a virtual environment (recommended)">
Requires Rust and the CUDA toolkit (`nvcc`).

```bash
# install dynamo
uv venv --python 3.12 --seed
uv pip install maturin nixl
cd $DYNAMO_HOME/lib/bindings/python
maturin develop --uv
cd $DYNAMO_HOME
uv pip install -e .
# install sglang
git clone https://github.com/sgl-project/sglang.git
# you can optionally checkout any sglang branch
cd sglang && uv pip install -e "python"
```

This is the ideal way for agents to develop. You can provide the path to both repos and the virtual environment and have it rerun these commands as it makes changes
</Accordion>

### Docker

Two paths are supported. Pick the one that matches how you plan to develop.

#### Pre-built Dynamo SGLang container (recommended)

Pull and launch the published `sglang-runtime` image from NGC. See [release artifacts](../../reference/release-artifacts.md) for the current tag and CUDA variants.

```bash
docker run --gpus all -it --rm \
    --network host --shm-size=10G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --cap-add CAP_SYS_PTRACE --ipc host \
    -v $HOME/.cache/huggingface:/home/dynamo/.cache/huggingface \
    nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.1.1
```

Mount the host Hugging Face cache (`-v $HOME/.cache/huggingface:/home/dynamo/.cache/huggingface`) so each container restart doesn't re-download model weights. The container runs as user `dynamo` (UID 1000), which is why the in-container path is `/home/dynamo/.cache/huggingface`.

#### Build from source inside upstream SGLang container

Pull and launch the upstream SGLang image, then build Dynamo from source inside it:

```bash
docker run --gpus all -it --rm \
    --network host --shm-size=10G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --ipc host \
    lmsysorg/sglang:v{sglang_version}
```

Install build dependencies and Rust inside the container:

```bash
apt-get update -qq && apt-get install -y -qq \
    build-essential libclang-dev curl git > /dev/null 2>&1

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

pip install maturin[patchelf]
```

Clone and build Dynamo:

```bash
cd /sgl-workspace/
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo

cd lib/bindings/python/
maturin build -o /tmp
pip install /tmp/ai_dynamo_runtime*.whl

cd /sgl-workspace/dynamo/
pip install -e .
```

## Feature Support Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| [**Disaggregated Serving**](../../design-docs/disagg-serving.md) | ✅ | Prefill/decode separation with NIXL KV transfer |
| [**KV-Aware Routing**](../../components/router/README.md) | ✅ | |
| [**SLA-Based Planner**](../../components/planner/planner-guide.md) | ✅ | |
| [**Multimodal Support**](../../features/multimodal/multimodal-sglang.md) | ✅ | Image via EPD, E/PD, E/P/D patterns |
| [**Diffusion Models**](sglang-diffusion.md) | ✅ | LLM diffusion, image, and video generation |
| [**Request Cancellation**](../../fault-tolerance/request-cancellation.md) | ✅ | Aggregated full; disaggregated decode-only |
| [**Graceful Shutdown**](../../fault-tolerance/graceful-shutdown.md) | ✅ | Discovery unregister + grace period |
| [**Observability**](sglang-observability.md) | ✅ | Metrics, tracing, and Grafana dashboards |
| [**KVBM**](../../components/kvbm/README.md) | ❌ | Planned |

## Quick Start

### Python / CLI Deployment

Start infrastructure services for local development:

```bash
docker compose -f deploy/docker-compose.yml up -d
```


Launch an aggregated serving deployment:

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/agg.sh
```

Verify the deployment:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Explain why Roger Federer is considered one of the greatest tennis players of all time"}],
    "stream": true,
    "max_tokens": 30
  }'
```
### Disaggregated Serving

Launch a disaggregated Qwen3-0.6B deployment (smallest model, useful for plumbing validation):

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/disagg.sh
```

> **Performance caveat:** Qwen3-0.6B is small enough that the disaggregated pathway is dominated by transport overhead and will often look slower than aggregated. Use it for plumbing validation, not benchmarks. Switch to Qwen3-32B-FP8 or larger for realistic disagg numbers.

### Multi-Node TP

SGLang supports multi-node tensor parallelism via the native `--dist-init-addr`, `--nnodes`, and `--node-rank` flags. See [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html) for the canonical reference; the same flags work with `python -m dynamo.sglang`. For a Kubernetes deployment example, see [`disagg-multinode.yaml`](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy/disagg-multinode.yaml).

### Kubernetes Deployment

You can deploy SGLang with Dynamo on Kubernetes using a `DynamoGraphDeployment`. For more details, see the [SGLang Kubernetes Deployment Guide](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy).

## Next Steps

- **[Reference Guide](sglang-reference-guide.md)**: Worker types, architecture, and configuration
- **[Examples](sglang-examples.md)**: All deployment patterns with launch scripts
- **[Disaggregation](sglang-disaggregation.md)**: P/D architecture and KV transfer details
- **[Diffusion](sglang-diffusion.md)**: LLM, image, and video diffusion models
- **[Observability](sglang-observability.md)**: Metrics, tracing, and Grafana dashboards
- **[Deploying SGLang with Dynamo on Kubernetes](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy)**: Kubernetes deployment guide
