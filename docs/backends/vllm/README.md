---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM
---

# LLM Deployment using vLLM

Dynamo vLLM integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation while maintaining full compatibility with vLLM's native engine arguments. Dynamo leverages vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting to enable KV-aware routing and P/D disaggregation.

## Installation

### Install Latest Release

We recommend using [uv](https://github.com/astral-sh/uv) to install:

```bash
uv venv --python 3.12 --seed
uv pip install "ai-dynamo[vllm]"
```

This installs Dynamo with the compatible vLLM version.

### Development Setup

For development, use the [devcontainer](https://github.com/ai-dynamo/dynamo/tree/main/.devcontainer) which has all dependencies pre-installed.

---

<Accordion title="Build and run container">

We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts). If you'd like to build your own container from source:

```bash
python container/render.py --framework vllm --output-short-filename
docker build -f container/rendered.Dockerfile -t dynamo:latest-vllm .
```

```bash
./container/run.sh -it --framework VLLM [--mount-workspace]
```

</Accordion>

## Feature Support Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| [**Disaggregated Serving**](../../design-docs/disagg-serving.md) | ✅ | Prefill/decode separation with NIXL KV transfer |
| [**KV-Aware Routing**](../../components/router/README.md) | ✅ | |
| [**SLA-Based Planner**](../../components/planner/planner-guide.md) | ✅ | |
| [**KVBM**](../../components/kvbm/README.md) | ✅ | |
| [**LMCache**](../../integrations/lmcache-integration.md) | ✅ | |
| [**FlexKV**](../../integrations/flexkv-integration.md) | ✅ | |
| [**Multimodal Support**](vllm-omni.md) | ✅ | Via vLLM-Omni integration |
| [**Observability**](vllm-observability.md) | ✅ | Metrics and monitoring |
| **WideEP** | ✅ | Support for DeepEP |
| **DP Rank Routing** | ✅ | [Hybrid load balancing](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/?h=external+dp#hybrid-load-balancing) via external DP rank control |
| [**LoRA**](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/lora/README.md) | ✅ | Dynamic loading/unloading from S3-compatible storage |
| **GB200 Support** | ✅ | Container functional on main |

## Quick Start

Start infrastructure services for local development:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Launch an aggregated serving deployment:

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg.sh
```

## Next Steps

- **[Reference Guide](vllm-reference-guide.md)**: Configuration, arguments, and operational details
- **[Examples](vllm-examples.md)**: All deployment patterns with launch scripts
- **[Observability](vllm-observability.md)**: Metrics and monitoring
- **[vLLM-Omni](vllm-omni.md)**: Multimodal model serving
- **[Kubernetes Deployment](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/README.md)**: Kubernetes deployment guide
- **[vLLM Documentation](https://docs.vllm.ai/en/stable/)**: Upstream vLLM serve arguments
