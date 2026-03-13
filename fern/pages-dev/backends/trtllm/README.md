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

## Quick Start

**Step 1 (host terminal):** Start infrastructure services:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

**Step 2 (host terminal):** Pull and run the prebuilt container:

```bash
DYNAMO_VERSION=1.0.0
docker pull nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:$DYNAMO_VERSION
docker run --gpus all -it --network host --ipc host \
  nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:$DYNAMO_VERSION
```

<Note>
The `DYNAMO_VERSION` variable above can be set to any specific available version of the container.
To find the available `tensorrtllm-runtime` versions for Dynamo, visit the [NVIDIA NGC Catalog for Dynamo TensorRT-LLM Runtime](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/tensorrtllm-runtime).
</Note>

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

### Kubernetes Deployment

You can deploy TensorRT-LLM with Dynamo on Kubernetes using a `DynamoGraphDeployment`. For more details, see the [TensorRT-LLM Kubernetes Deployment Guide](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy/README.md).

## Next Steps

- **[Reference Guide](trtllm-reference-guide.md)**: Features, configuration, and operational details
- **[Examples](trtllm-examples.md)**: All deployment patterns with launch scripts
- **[KV Cache Transfer](trtllm-kv-cache-transfer.md)**: KV cache transfer methods for disaggregated serving
- **[Prometheus Metrics](trtllm-prometheus.md)**: Metrics and monitoring
- **[Multinode Examples](multinode/trtllm-multinode-examples.md)**: Multi-node deployment with SLURM
- **[Deploying TensorRT-LLM with Dynamo on Kubernetes](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy/README.md)**: Kubernetes deployment guide
