---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM
---

Dynamo vLLM integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation while maintaining full compatibility with vLLM's native engine arguments. Dynamo leverages vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting to enable KV-aware routing and P/D disaggregation.

## Experimental Sidecar

The experimental sidecar path runs the Dynamo worker outside vLLM and connects
through vLLM's native gRPC API. It keeps vLLM's native server and argument
surface while separating Dynamo and engine dependencies. See
[vLLM Sidecar](sidecar.md) for current readiness and launch examples.

## Installation

### Install Latest Release

We recommend using [uv](https://github.com/astral-sh/uv) to install:

```bash
uv venv --python 3.12 --seed
uv pip install "ai-dynamo[vllm]"
```

This installs Dynamo with the compatible vLLM version.

---

### Container

We have public images available on [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts):

```bash
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:<version>
./container/run.sh -it --framework VLLM --image nvcr.io/nvidia/ai-dynamo/vllm-runtime:<version>
```

<Accordion title="Build from source">

```bash
python container/render.py --framework vllm --output-short-filename
docker build -f container/rendered.Dockerfile -t dynamo:latest-vllm .
```

```bash
./container/run.sh -it --framework VLLM [--mount-workspace]
```

</Accordion>

### Development Setup

For development, use the [devcontainer](https://github.com/ai-dynamo/dynamo/tree/v1.4.0/.devcontainer) which has all dependencies pre-installed.

## Feature Support Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| [**Disaggregated Serving**](../../../concepts/system-architecture/disaggregated-serving.md) | ✅ | Prefill/decode separation with NIXL KV transfer |
| [**KV-Aware Routing**](../../router/overview.md) | ✅ | |
| [**SLA-Based Planner**](../../planner/planner-guide.md) | ✅ | |
| [**KVBM**](../../kvbm/overview.md) | ✅ | |
| [**LMCache**](../../../../../cli/kv-cache-offloading/overview.mdx) | ✅ | CUDA 12.9 and arm64/aarch64 containers may require building LMCache from source |
| [**FlexKV**](../../../../../cli/kv-cache-offloading/overview.mdx) | ✅ | Requires a separate FlexKV build |
| [**Multimodal Support**](../../../../../use-cases/diffusion/overview.md) | ✅ | Via vLLM-Omni integration |
| [**Observability**](observability.md) | ✅ | Metrics and monitoring |
| **WideEP** | ✅ | Support for DeepEP |
| **DP Rank Routing** | ✅ | [Hybrid load balancing](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/?h=external+dp#hybrid-load-balancing) via external DP rank control |
| [**LoRA**](https://github.com/ai-dynamo/dynamo/tree/v1.4.0/examples/backends/vllm/launch/lora/README.md) | ✅ | Dynamic loading/unloading from S3-compatible storage |
| **GB200 Support** | ✅ | Container functional on main |

## Feature Interactions

vLLM offers the broadest feature coverage in Dynamo, with full support for disaggregated serving, KV-aware routing, KV block management, LoRA adapters, and multimodal inference including video and audio. The matrix below shows which feature pairs are validated to work together.

**Legend:** ✅ Supported &nbsp;|&nbsp; 🚧 Work in Progress / Experimental / Limited

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | LoRA | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | — | | | | | | | | | |
| **KV-Aware Routing** | ✅ | — | | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | — | | | | | | | |
| **KV Block Manager** | ✅ | ✅ | ✅ | — | | | | | | |
| **Multimodal** | ✅ | ✅<sup>1</sup> | — | ✅ | — | | | | | |
| **Request Migration** | ✅ | ✅ | ✅ | ✅ | ✅ | — | | | | |
| **Request Cancellation** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | | | |
| **LoRA** | ✅ | ✅<sup>2</sup> | — | ✅ | — | ✅ | ✅ | — | | |
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | |
| **Speculative Decoding** | ✅ | ✅ | — | ✅ | — | ✅ | ✅ | — | ✅ | — |

> **Notes:**
> 1. **Multimodal + KV-Aware Routing**: Image-aware KV routing is supported in the documented vLLM paths. The default Rust frontend path supports model families handled by `llm-multimodal`; the Python chat-processor path delegates to vLLM's multimodal processor. ([Source](../../../../../use-cases/multimodal-serving/multimodal-kv-routing.md))
> 2. **KV-Aware LoRA Routing**: vLLM supports routing requests based on LoRA adapter affinity.
> 3. **Audio Support**: vLLM supports audio models like Qwen2-Audio (experimental). ([Source](multimodal.md))
> 4. **Video Support**: vLLM supports video input with frame sampling. ([Source](multimodal.md))
> 5. **Speculative Decoding**: Eagle3 support documented. ([Source](../../../../additional-resources/speculative-decoding/speculative-decoding-with-vllm.md))

## Quick Start

Start infrastructure services for local development:

```bash
docker compose -f dev/docker-compose.yml up -d
```

Launch an aggregated serving deployment:

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg.sh
```

> **Running launch scripts standalone.** The `launch/*.sh` scripts expect etcd and NATS to be reachable on localhost. Bring them up first (run from the repo root, or use the absolute path shown):
>
> ```bash
> docker compose -f "$DYNAMO_HOME/dev/docker-compose.yml" up -d
> ```
>
> Then run the launch script. Without these, workers register but the frontend cannot discover them and requests hang.

## Next Steps

- **[Reference Guide](reference-guide.md)**: Configuration, arguments, and operational details
- **[Examples](../../../../../recipes/cli-templates/vllm.mdx)**: Local deployment launch scripts
- **[KV Cache Offloading](kv-cache-offloading.md)**: KVBM, LMCache, and FlexKV integrations
- **[Observability](observability.md)**: Metrics and monitoring
- **[vLLM-Omni](../../../../../use-cases/diffusion/overview.md)**: Multimodal model serving
- **[Kubernetes Deployment](https://github.com/ai-dynamo/dynamo/tree/v1.4.0/examples/backends/vllm/deploy/README.md)**: Kubernetes deployment guide
- **[vLLM Documentation](https://docs.vllm.ai/en/stable/)**: Upstream vLLM serve arguments
