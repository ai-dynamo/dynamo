---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: TensorRT-LLM Sidecar
subtitle: Run Dynamo beside a stock TensorRT-LLM engine through native gRPC.
---

<Warning>
**Experimental.** The TensorRT-LLM sidecar, launcher, packaging, and feature
coverage can change without notice.
</Warning>

`dynamo-trtllm-sidecar` is a CPU-only Dynamo worker that connects to
TensorRT-LLM's native gRPC service. It preserves the upstream engine process
and argument surface while using Dynamo for request handling and distributed
serving. See the
[Sidecar Backends](../../../concepts/system-architecture/sidecar-backends.md) page for the common
architecture.

## Readiness

| Deployment path | Aggregated | Disaggregated |
|---|---|---|
| Local launcher | Validated on one GPU | Not supported |
| Kubernetes example | Validated | Not supported |

This table covers launch topology only. The
[TensorRT-LLM feature matrix](overview.md#feature-support-matrix) describes the
in-process backend; sidecar feature parity is still under evaluation. The
current native gRPC contract does not provide the prefill/decode handoff needed
for disaggregated serving. See the
[TensorRT-LLM sidecar README](https://github.com/ai-dynamo/dynamo/blob/v1.4.1/lib/sidecar/trtllm/README.md)
for other protocol limitations.

## Launch Locally

From a Dynamo source checkout, build or install Dynamo so
`dynamo-trtllm-sidecar` is on `PATH`. Install a TensorRT-LLM release that
provides `tensorrt_llm.commands.serve --grpc`.

Start Dynamo's local discovery services, then run the aggregated launcher:

```bash
docker compose -f dev/docker-compose.yml up -d
./lib/sidecar/trtllm/launch/agg.sh --model Qwen/Qwen3-0.6B
```

The launcher starts the Dynamo frontend, TensorRT-LLM engine, and sidecar. It
binds TensorRT-LLM's native gRPC endpoint to loopback.

Verify the frontend:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32
  }'
```

## Deploy on Kubernetes

No published TensorRT-LLM sidecar image is available yet. Follow the
[Kubernetes quick start](https://github.com/ai-dynamo/dynamo/blob/v1.4.1/lib/sidecar/trtllm/README.md#deploy-on-kubernetes-quick-start)
to build the CPU-only sidecar image and pair it with a stock upstream
TensorRT-LLM image. The source tree includes an
[aggregated deployment manifest](https://github.com/ai-dynamo/dynamo/blob/v1.4.1/lib/sidecar/trtllm/deploy/agg.yaml).
