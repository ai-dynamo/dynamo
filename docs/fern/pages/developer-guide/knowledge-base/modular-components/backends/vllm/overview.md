---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM
subtitle: Run vLLM engines with Dynamo on NVIDIA or Intel GPUs
---

Dynamo vLLM integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime on NVIDIA and Intel GPUs. It enables disaggregated serving, KV-aware routing, and request cancellation while maintaining compatibility with vLLM's native engine arguments. Dynamo uses vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting to enable KV-aware routing and prefill/decode (P/D) disaggregation.

## Experimental Sidecar

The experimental sidecar path runs the Dynamo worker outside vLLM and connects
through vLLM's native gRPC API. It keeps vLLM's native server and argument
surface while separating Dynamo and engine dependencies. See
[vLLM Sidecar](sidecar.md) for current readiness and launch examples.

## Installation

<Tabs>
<Tab title="NVIDIA GPU">

We recommend using [uv](https://github.com/astral-sh/uv) to install:

```bash
uv venv --python 3.12 --seed
uv pip install "ai-dynamo[vllm]"
```

This installs Dynamo with the compatible vLLM version.

### Container Image

Use the published runtime image from the [NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts):

```bash
docker pull nvcr.io/nvidia/ai-dynamo/vllm-runtime:<version>
./container/run.sh -it --framework VLLM --image nvcr.io/nvidia/ai-dynamo/vllm-runtime:<version>
```

<Accordion title="Build from source">

```bash
python container/render.py --framework vllm --output-short-filename
docker build -f container/rendered.Dockerfile -t dynamo:latest-vllm .

./container/run.sh -it --framework VLLM [--mount-workspace]
```

</Accordion>

</Tab>
<Tab title="Intel GPU">

Build the vLLM XPU runtime image from the Dynamo source tree. XPU images support `amd64` hosts:

```bash
python3 container/render.py \
  --framework vllm \
  --device xpu \
  --target runtime \
  --output-short-filename

docker build \
  --file container/rendered.Dockerfile \
  --tag dynamo:latest-vllm-xpu \
  .
```

Run the image with access to the Intel GPU devices under `/dev/dri`:

```bash
./container/run.sh \
  -it \
  --framework VLLM \
  --device xpu \
  --image dynamo:latest-vllm-xpu \
  --mount-workspace
```

The `--device xpu` option mounts `/dev/dri`, adds the host render group when available, and disables
the NVIDIA container runtime. Inside the container, XPU launchers set `VLLM_TARGET_DEVICE=xpu` and
use `ZE_AFFINITY_MASK` to select devices.

</Tab>
</Tabs>

### Development Setup

For NVIDIA GPU development, use the
[devcontainer](https://github.com/ai-dynamo/dynamo/tree/main/.devcontainer), which has the CUDA
dependencies pre-installed. For Intel GPU development, use the source-built XPU runtime image.

## Feature Support Matrix

<Tabs>
<Tab title="NVIDIA GPU">

| Feature | Status | Notes |
|---------|--------|-------|
| [**Disaggregated Serving**](../../../concepts/system-architecture/disaggregated-serving.md) | ✅ | Prefill/decode separation with NIXL KV transfer |
| [**KV-Aware Routing**](../../router/overview.md) | ✅ | |
| [**SLA-Based Planner**](../../planner/planner-guide.md) | ✅ | |
| [**LMCache**](../../../../../cli/kv-cache-offloading/overview.mdx) | ✅ | CUDA 12.9 and arm64/aarch64 containers may require building LMCache from source |
| [**FlexKV**](../../../../../cli/kv-cache-offloading/overview.mdx) | ✅ | Requires a separate FlexKV build |
| [**Multimodal Support**](../../../../../use-cases/diffusion/overview.md) | ✅ | Via vLLM-Omni integration |
| [**Observability**](observability.md) | ✅ | Metrics and monitoring |
| **WideEP** | ✅ | Support for DeepEP |
| **DP Rank Routing** | ✅ | [Hybrid load balancing](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/?h=external+dp#hybrid-load-balancing) via external DP rank control |
| [**LoRA**](../../../../../cli/operations/lora-adapters.md) | ✅ | Dynamic loading/unloading from `file://`, `s3://`, and `hf://` sources; adapter-aware KV routing |
| [**Tool Calling**](../../../../../use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx) | ✅ | |
| **GB200 Support** | ✅ | Container functional on main |

</Tab>
<Tab title="Intel GPU">

| Feature | Status | Intel GPU Notes |
|---------|--------|-----------------|
| [**Disaggregated Serving**](../../../concepts/system-architecture/disaggregated-serving.md) | ✅ | P/D and multimodal E/P/D launchers use NIXL with XPU KV buffers |
| [**KV-Aware Routing**](../../router/overview.md) | ✅ | KV events and approximate cache tracking |
| [**SLA-Based Planner**](../../planner/planner-guide.md) | ✅ | XPU P/D deployment manifest includes the Planner |
| **KVBM** | — | |
| [**LMCache**](../../../../../cli/kv-cache-offloading/overview.mdx) | ✅ | Aggregated and multiprocess deployments |
| [**FlexKV**](../../../../../cli/kv-cache-offloading/overview.mdx) | — | |
| [**Multimodal Support**](multimodal.md) | ✅ | Image and video input, frontend decoding, and multimodal KV routing |
| [**Observability**](observability.md) | ✅ | Metrics and tracing deployment manifests |
| **WideEP** | — | |
| **DP Rank Routing** | — | |
| [**Tool Calling**](../../../../../use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx) | ✅ | Multimodal tool calling with the `hermes` parser |

> [!NOTE]
> An em dash means the feature is not supported on Intel GPU.

</Tab>
</Tabs>

> [!NOTE]
> vLLM LoRA support applies to generative workers. `--enable-lora` cannot be combined with `--realtime` or `--classify-worker`; embedding and other pooling workers do not expose the LoRA lifecycle endpoints.

## Feature Interactions

vLLM offers the broadest feature coverage in Dynamo, including disaggregated serving, KV-aware
routing, KV block management, LoRA adapters, and multimodal inference. Select an accelerator to see
the supported feature pairs.

<Tabs>
<Tab title="NVIDIA GPU">

**Legend:** ✅ Supported &nbsp;|&nbsp; 🚧 Work in Progress / Experimental / Limited
&nbsp;|&nbsp; — Not supported &nbsp;|&nbsp; N/A Same feature

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | LoRA | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | N/A | | | | | | | | | |
| **KV-Aware Routing** | ✅ | N/A | | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | N/A | | | | | | | |
| **KV Block Manager** | ✅ | ✅ | ✅ | N/A | | | | | | |
| **Multimodal** | ✅ | ✅<sup>1</sup> | ✅ | ✅ | N/A | | | | | |
| **Request Migration** | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | | | | |
| **Request Cancellation** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | | | |
| **LoRA** | ✅ | ✅<sup>2</sup> | — | ✅ | — | ✅ | ✅ | N/A | | |
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | N/A | |
| **Speculative Decoding** | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ | — | ✅ | N/A |

> **Notes:**
> 1. **Multimodal + KV-Aware Routing**: Image-aware KV routing is supported in the documented vLLM paths. The default Rust frontend path supports model families handled by `llm-multimodal`; the Python chat-processor path delegates to vLLM's multimodal processor. ([Source](../../../../../use-cases/multimodal-serving/multimodal-kv-routing.md))
> 2. **KV-Aware LoRA Routing**: vLLM supports routing requests based on LoRA adapter affinity.
> 3. **Audio Support**: vLLM supports audio models like Qwen2-Audio (experimental). ([Source](multimodal.md))
> 4. **Video Support**: vLLM supports video input with frame sampling. ([Source](multimodal.md))
> 5. **Speculative Decoding**: Eagle3 support documented. ([Source](../../../../additional-resources/speculative-decoding/speculative-decoding-with-vllm.md))

</Tab>
<Tab title="Intel GPU">

**Legend:** ✅ Supported &nbsp;|&nbsp; — Not supported &nbsp;|&nbsp; N/A Same feature

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | N/A | | | | | | | | |
| **KV-Aware Routing** | ✅ | N/A | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | N/A | | | | | | |
| **KV Block Manager** | — | — | — | N/A | | | | | |
| **Multimodal** | ✅ | ✅ | ✅ | — | N/A | | | | |
| **Request Migration** | ✅ | ✅ | ✅ | — | ✅ | N/A | | | |
| **Request Cancellation** | ✅ | ✅ | ✅ | — | ✅ | ✅ | N/A | | |
| **Tool Calling** | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | N/A | |
| **Speculative Decoding** | ✅ | ✅ | ✅ | — | — | ✅ | ✅ | ✅ | N/A |

> **Notes:**
> 1. **Disaggregated + KV-Aware Routing**: The XPU GDR launcher and Kubernetes manifest combine P/D
>    disaggregation with KV-aware routing.
> 2. **Disaggregated + Multimodal**: The XPU launcher provides encode/prefill/decode disaggregation.
> 3. **Multimodal + KV-Aware Routing**: XPU launchers cover the Rust multimodal router and the vLLM
>    chat-processor path.
> 4. **Request Lifecycle Features**: Request migration and cancellation run in Dynamo's request
>    plane and do not depend on accelerator-specific engine code.
> 5. **Speculative Decoding**: The vLLM XPU build supports N-gram speculative decoding.
> 6. **SLA-Based Planner**: The Planner operates on runtime metrics and replica state. It supports
>    multimodal deployments and consumes speculative decode metadata without accelerator-specific
>    code paths.

</Tab>
</Tabs>

## Quick Start

Start infrastructure services for local development:

```bash
docker compose -f dev/docker-compose.yml up -d
```

Launch an aggregated serving deployment:

<Tabs>
<Tab title="NVIDIA GPU">

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/agg.sh
```

</Tab>
<Tab title="Intel GPU">

Run these commands inside the XPU runtime container from the
[Installation](#installation) section:

```bash
cd $DYNAMO_HOME/examples/backends/vllm
bash launch/xpu/agg_xpu.sh
```

The launcher defaults to `ZE_AFFINITY_MASK=0`. Set it before launch to select another Intel GPU:

```bash
ZE_AFFINITY_MASK=1 bash launch/xpu/agg_xpu.sh
```

</Tab>
</Tabs>

> **Running launch scripts standalone.** The `launch/*.sh` scripts expect etcd and NATS to be reachable on localhost. Bring them up first (run from the repo root, or use the absolute path shown):
>
> ```bash
> docker compose -f "$DYNAMO_HOME/dev/docker-compose.yml" up -d
> ```
>
> Then run the launch script. Without these, workers register but the frontend cannot discover them and requests hang.

Verify the deployment:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'
```

## Next Steps

- **[Reference Guide](reference-guide.md)**: Configuration, arguments, and operational details
- **[Examples](../../../../../recipes/cli-templates/vllm.mdx)**: Local deployment launch scripts
- **[Intel GPU Examples](../../../../../recipes/cli-templates/vllm.mdx#xpu)**: XPU launchers and configuration
- **[KV Cache Offloading](kv-cache-offloading.md)**: LMCache and FlexKV integrations
- **[Observability](observability.md)**: Metrics and monitoring
- **[vLLM-Omni](../../../../../use-cases/diffusion/overview.md)**: Multimodal model serving
- **[Kubernetes Deployment](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/README.md)**: Kubernetes deployment guide
- **[vLLM Documentation](https://docs.vllm.ai/en/stable/)**: Upstream vLLM serve arguments
