---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang
---

## Use the Latest Release

We recommend using the [latest stable release](https://github.com/ai-dynamo/dynamo/releases/latest) of Dynamo to avoid breaking changes.

---

Dynamo SGLang integrates [SGLang](https://github.com/sgl-project/sglang) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation while maintaining full compatibility with SGLang's native engine arguments. It supports LLM inference, embedding models, multimodal vision models, and diffusion-based generation (LLM, image, video).

## Installation

### Install Latest Release

We recommend using [uv](https://github.com/astral-sh/uv) to install:

```bash
uv venv --python 3.12 --seed
uv pip install "ai-dynamo[sglang]"
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

<Accordion title="Development installation inside SGLang container">

Pull and launch the SGLang runtime image:

```bash
docker run --gpus all -it --rm \
    --network host --shm-size=10G \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --ulimit nofile=65536:65536 \
    --ipc host \
    lmsysorg/sglang:v{sglang_version}
```

Inside the container, install build dependencies and Rust:

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

</Accordion>

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
### Kubernetes Deployment

You can deploy SGLang with Dynamo on Kubernetes using a `DynamoGraphDeployment`. For more details, see the [SGLang Kubernetes Deployment Guide](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy).

## Next Steps

- **[Reference Guide](sglang-reference-guide.md)**: Worker types, architecture, and configuration
- **[Examples](sglang-examples.md)**: All deployment patterns with launch scripts
- **[Disaggregation](sglang-disaggregation.md)**: P/D architecture and KV transfer details
- **[Diffusion](sglang-diffusion.md)**: LLM, image, and video diffusion models
- **[Observability](sglang-observability.md)**: Metrics, tracing, and Grafana dashboards
- **[Deploying SGLang with Dynamo on Kubernetes](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy)**: Kubernetes deployment guide
