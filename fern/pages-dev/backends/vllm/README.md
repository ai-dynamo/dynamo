---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM
subtitle: vLLM engines run in Dynamo's distributed runtime with disaggregated serving, NIXL KV transfer, and KV-aware routing.
---

Dynamo vLLM integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation while maintaining full compatibility with vLLM's native engine arguments. Dynamo leverages vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting to enable KV-aware routing and P/D disaggregation.

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

For development, use the [devcontainer](https://github.com/ai-dynamo/dynamo/tree/main/.devcontainer) which has all dependencies pre-installed.

## Feature Support Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| [**Disaggregated Serving**](../../design-docs/disagg-serving.md) | ✅ | Prefill/decode separation with NIXL KV transfer |
| [**KV-Aware Routing**](../../components/router/README.md) | ✅ | |
| [**SLA-Based Planner**](../../components/planner/planner-guide.md) | ✅ | |
| [**KVBM**](../../components/kvbm/README.md) | ✅ | |
| [**LMCache**](../../integrations/lmcache-integration.md) | ✅ | CUDA 12.9 and arm64/aarch64 containers may require building LMCache from source |
| [**FlexKV**](../../integrations/flexkv-integration.md) | ✅ | |
| [**Multimodal Support**](../../features/multimodal/multimodal-vllm.md) | ✅ | Aggregated image/video serving on legacy and unified Python backends |
| [**Observability**](vllm-observability.md) | ✅ | Metrics and monitoring |
| **WideEP** | ✅ | Support for DeepEP |
| **DP Rank Routing** | ✅ | [Hybrid load balancing](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/?h=external+dp#hybrid-load-balancing) via external DP rank control |
| [**LoRA**](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch/lora/README.md) | ✅ | Dynamic loading/unloading from S3-compatible storage |
| **GB200 Support** | ✅ | Container functional on main |

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

### Rust Backend Preview

The Python vLLM backend remains the recommended entry point for production
deployments and examples. The Rust backend is a development preview for
validating the Rust `LLMEngine` integration with vLLM's engine-core client.
Use it when working on the Rust backend contract, cancellation, metrics,
or P/D wiring; use `python -m dynamo.vllm` or
`python -m dynamo.vllm.unified_main` for the most complete vLLM feature
coverage.

<Note>
The Rust backend depends on vLLM's engine-core crates, which are not yet
published to crates.io and are pulled as git dependencies. They are gated
behind the off-by-default `vllm_rs` cargo feature, so the default workspace
build does not require the git sources and the crate is excluded from the
published Dynamo crates. You must pass `--features vllm_rs` to build or run it.
</Note>

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
- **[Examples](vllm-examples.md)**: All deployment patterns with launch scripts
- **[KV Cache Offloading](vllm-kv-offloading.md)**: KVBM, LMCache, and FlexKV integrations
- **[Observability](vllm-observability.md)**: Metrics and monitoring
- **[vLLM-Omni](vllm-omni.md)**: Multimodal model serving
- **[Kubernetes Deployment](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/README.md)**: Kubernetes deployment guide
- **[vLLM Documentation](https://docs.vllm.ai/en/stable/)**: Upstream vLLM serve arguments
