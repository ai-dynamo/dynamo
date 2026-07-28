---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM
subtitle: Run vLLM engines with Dynamo on NVIDIA or Intel GPUs
---

Dynamo vLLM integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime on NVIDIA and Intel GPUs. It enables disaggregated serving, KV-aware routing, and request cancellation while maintaining compatibility with vLLM's native engine arguments. Dynamo uses vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting to enable KV-aware routing and prefill/decode (P/D) disaggregation.

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
| [**Disaggregated Serving**](../../design-docs/disagg-serving.md) | ✅ | Prefill/decode separation with NIXL KV transfer |
| [**KV-Aware Routing**](../../components/router/README.md) | ✅ | |
| [**SLA-Based Planner**](../../components/planner/planner-guide.md) | ✅ | |
| [**KVBM**](../../components/kvbm/README.md) | ✅ | |
| [**LMCache**](../../cli/kv-cache-offloading.mdx) | ✅ | CUDA 12.9 and arm64/aarch64 containers may require building LMCache from source |
| [**FlexKV**](../../cli/kv-cache-offloading.mdx) | ✅ | Requires a separate FlexKV build |
| [**Multimodal Support**](../../features/diffusion/README.md) | ✅ | Via vLLM-Omni integration |
| [**Observability**](vllm-observability.md) | ✅ | Metrics and monitoring |
| **WideEP** | ✅ | Support for DeepEP |
| **DP Rank Routing** | ✅ | [Hybrid load balancing](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/?h=external+dp#hybrid-load-balancing) via external DP rank control |
| [**LoRA**](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/lora/README.md) | ✅ | Dynamic loading/unloading from S3-compatible storage |
| [**Tool Calling**](../../tool-calling/README.mdx) | ✅ | |
| **GB200 Support** | ✅ | Container supported |

</Tab>
<Tab title="Intel GPU">

| Feature | Status | Intel GPU Notes |
|---------|--------|-----------------|
| [**Disaggregated Serving**](../../design-docs/disagg-serving.md) | ✅ | P/D and multimodal E/P/D launchers use NIXL with XPU KV buffers |
| [**KV-Aware Routing**](../../components/router/README.md) | ✅ | KV events and approximate cache tracking |
| [**SLA-Based Planner**](../../components/planner/planner-guide.md) | ✅ | XPU P/D deployment manifest includes the Planner |
| [**KVBM**](../../components/kvbm/README.md) | — | |
| [**LMCache**](../../cli/kv-cache-offloading.mdx) | ✅ | Aggregated and multiprocess deployments |
| [**FlexKV**](../../cli/kv-cache-offloading.mdx) | — | |
| [**Multimodal Support**](../../features/multimodal/multimodal-vllm.md) | ✅ | Image and video input, frontend decoding, and multimodal KV routing |
| [**Observability**](vllm-observability.md) | ✅ | Metrics and tracing deployment manifests |
| **WideEP** | — | |
| **DP Rank Routing** | — | |
| **LoRA** | — | |
| [**Tool Calling**](../../tool-calling/README.mdx) | ✅ | Multimodal tool calling with the `hermes` parser |

> [!NOTE]
> An em dash means the feature is not supported on Intel GPU.

</Tab>
</Tabs>

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
> 1. **Multimodal + KV-Aware Routing**: Image-aware KV routing is supported in the documented vLLM paths. The default Rust frontend path supports model families handled by `llm-multimodal`; the Python chat-processor path delegates to vLLM's multimodal processor. ([Source](../../features/multimodal/multimodal-kv-routing.md))
> 2. **KV-Aware LoRA Routing**: vLLM supports routing requests based on LoRA adapter affinity.
> 3. **Audio Support**: vLLM supports audio models like Qwen2-Audio (experimental). ([Source](../../features/multimodal/multimodal-vllm.md))
> 4. **Video Support**: vLLM supports video input with frame sampling. ([Source](../../features/multimodal/multimodal-vllm.md))
> 5. **Speculative Decoding**: Eagle3 support documented. ([Source](../../features/speculative-decoding/speculative-decoding-vllm.md))

</Tab>
<Tab title="Intel GPU">

**Legend:** ✅ Supported &nbsp;|&nbsp; — Not supported &nbsp;|&nbsp; N/A Same feature

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | LoRA | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | N/A | | | | | | | | | |
| **KV-Aware Routing** | ✅ | N/A | | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | N/A | | | | | | | |
| **KV Block Manager** | — | — | — | N/A | | | | | | |
| **Multimodal** | ✅ | ✅ | ✅ | — | N/A | | | | | |
| **Request Migration** | ✅ | ✅ | ✅ | — | ✅ | N/A | | | | |
| **Request Cancellation** | ✅ | ✅ | ✅ | — | ✅ | ✅ | N/A | | | |
| **LoRA** | — | — | — | — | — | — | — | N/A | | |
| **Tool Calling** | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | — | N/A | |
| **Speculative Decoding** | ✅ | ✅ | ✅ | — | — | ✅ | ✅ | — | ✅ | N/A |

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

### Rust Backend Preview

The Python vLLM backend remains the recommended entry point for production
deployments and examples. The Rust backend is a development preview for
validating the Rust `LLMEngine` integration with vLLM's engine-core client.
Use it when working on the Rust backend contract, cancellation, metrics,
or P/D wiring; use `python -m dynamo.vllm` or
`python -m dynamo.vllm.unified_main` for the most complete vLLM feature
coverage.

> [!NOTE]
> The Rust backend depends on vLLM's engine-core crates, which are not yet
> published to crates.io and are pulled as git dependencies. They are gated
> behind the off-by-default `vllm_rs` cargo feature, so the default workspace
> build does not require the git sources and the crate is excluded from the
> published Dynamo crates. You must pass `--features vllm_rs` to build or run it.

To run the Rust backend locally, start the same infrastructure services and
frontend, then launch the Rust worker in another terminal:

```bash
docker compose -f dev/docker-compose.yml up -d

python -m dynamo.frontend --http-port 8000
```

```bash
DYN_SYSTEM_PORT=8081 cargo run -p dynamo-vllm-rs-backend --features vllm_rs -- Qwen/Qwen3-0.6B -- \
  --enforce-eager \
  --max-model-len 4096
```

The Rust worker starts a managed vLLM engine-core process and registers with
the Dynamo frontend using the same discovery path as the Python unified
backend. The Rust backend is expected to become the default only after it
reaches feature and operational parity with the Python vLLM backend.

## Next Steps

- **[Reference Guide](vllm-reference-guide.md)**: Configuration, arguments, and operational details
- **[Examples](vllm-examples.mdx)**: Local deployment launch scripts
- **[Intel GPU Examples](vllm-examples.mdx#xpu)**: XPU launchers and configuration
- **[KV Cache Offloading](vllm-kv-offloading.md)**: KVBM, LMCache, and FlexKV integrations
- **[Observability](vllm-observability.md)**: Metrics and monitoring
- **[vLLM-Omni](../../features/diffusion/README.md)**: Multimodal model serving
- **[Kubernetes Deployment](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/README.md)**: Kubernetes deployment guide
- **[vLLM Documentation](https://docs.vllm.ai/en/stable/)**: Upstream vLLM serve arguments
