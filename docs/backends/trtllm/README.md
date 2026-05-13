---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: TensorRT-LLM
---

## Use the Latest Release

We recommend using the [latest stable release](https://github.com/ai-dynamo/dynamo/releases/latest) of Dynamo to avoid breaking changes.

---

Dynamo TensorRT-LLM integrates [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, multi-node deployments, and request cancellation. It supports LLM inference, multimodal models, video diffusion, and advanced features like speculative decoding and attention data parallelism.

## Feature Support Matrix

### Core Dynamo Features

| Feature | TensorRT-LLM | Notes |
|---------|--------------|-------|
| [**Disaggregated Serving**](../../design-docs/disagg-serving.md) | ✅ |  |
| [**Conditional Disaggregation**](../../design-docs/disagg-serving.md) | 🚧 | Not supported yet |
| [**KV-Aware Routing**](../../components/router/README.md) | ✅ |  |
| [**SLA-Based Planner**](../../components/planner/planner-guide.md) | ✅ |  |
| [**Load Based Planner**](../../components/planner/README.md) | 🚧 | Planned |
| [**KVBM**](../../components/kvbm/README.md) | ✅ | |

### Large Scale P/D and WideEP Features

| Feature            | TensorRT-LLM | Notes                                                           |
|--------------------|--------------|-----------------------------------------------------------------|
| **WideEP**         | ✅           |                                                                 |
| **DP Rank Routing**| ✅           |                                                                 |
| **GB200 Support**  | ✅           |                                                                 |

## Prerequisites

- **`yq`** for in-place YAML edits. Install with `wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/local/bin/yq && chmod +x /usr/local/bin/yq` or `pip install yq` (the latter is a different tool with the same name but similar syntax). If neither is available, a `sed` fallback is shown inline where `yq` is used.

## Container / driver matrix

| Container tag | Backend version | CUDA | Min NVIDIA driver |
|---|---|---|---|
| `tensorrtllm-runtime:1.0.2` | TRT-LLM `v1.3.0rc5.post1` | `v13.1` | `580+` |
| `vllm-runtime:1.0.2` | vLLM `v0.16.0` | `v12.9` | `575+` |
| `vllm-runtime:1.0.2-cuda13` | vLLM `v0.16.0` | `v13.0` | `580+` |
| `sglang-runtime:1.0.2` | SGLang `v0.5.9` | `v12.9` | `575+` |
| `sglang-runtime:1.0.2-cuda13` | SGLang `v0.5.9` | `v13.0` | `580+` |

Source of truth: [`docs/reference/support-matrix.md`](../../reference/support-matrix.md#cuda-and-driver-requirements) and [`docs/reference/release-artifacts.md`](../../reference/release-artifacts.md). If those differ from the values above, the source-of-truth files win.

## Quick Start

**Step 1 (host terminal):** Start infrastructure services:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

**Step 2 (host terminal):** Pull and run the prebuilt container:

```bash
DYNAMO_VERSION=1.0.2
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:$DYNAMO_VERSION
docker run --gpus all -it --network host --ipc host \
  nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:$DYNAMO_VERSION
```

> [!NOTE]
> The `DYNAMO_VERSION` variable above can be set to any specific available version of the container.
> To find the available `tensorrtllm-runtime` versions for Dynamo, visit the [NVIDIA NGC Catalog for Dynamo TensorRT-LLM Runtime](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime).

**Step 3 (inside the container):** Launch an aggregated serving deployment (uses `Qwen/Qwen3-0.6B` by default):

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/agg.sh
```

The launch script will automatically download the model and start the TensorRT-LLM engine. You can override the model by setting `MODEL_PATH` and `SERVED_MODEL_NAME` environment variables before running the script.

**Step 4 (host terminal):** Verify the deployment:

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

## Deploy

Deploy TensorRT-LLM with Dynamo on Kubernetes using a `DynamoGraphDeployment`. Before `kubectl apply`, substitute the container image tag in the deployment YAML. The `sed` fallback is shown inline for environments without `yq`:

```bash
# yq
yq -i '(.spec.services[].extraPodSpec.mainContainer.image) |= sub(":1\.0\.2", ":<your-tag>")' deploy.yaml
# sed fallback
sed -i.bak 's|:1\.0\.2|:<your-tag>|g' deploy.yaml
```

For full Kubernetes deployment instructions, see the [TensorRT-LLM Kubernetes Deployment Guide](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy/README.md).

## Next Steps

- **[Reference Guide](trtllm-reference-guide.md)**: Features, configuration, and operational details
- **[Examples](trtllm-examples.md)**: All deployment patterns with launch scripts
- **[KV Cache Transfer](trtllm-kv-cache-transfer.md)**: KV cache transfer methods for disaggregated serving
- **[Observability](trtllm-observability.md)**: Metrics and monitoring
- **[Multinode Examples](multinode/trtllm-multinode-examples.md)**: Multi-node deployment with SLURM
- **[Deploying TensorRT-LLM with Dynamo on Kubernetes](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy/README.md)**: Kubernetes deployment guide
