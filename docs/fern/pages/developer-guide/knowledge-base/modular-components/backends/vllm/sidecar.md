---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: vLLM Sidecar
subtitle: Run Dynamo beside a stock vLLM engine through native gRPC.
---

> [!WARNING]
> **Experimental.** The vLLM sidecar, launchers, packaging, and feature coverage
> can change without notice.

`dynamo-vllm-sidecar` is a CPU-only Dynamo worker that connects to vLLM's native
gRPC service. It preserves the upstream engine process and argument surface
while using Dynamo for request handling and distributed serving. See the
[Sidecar Backends](../../../concepts/system-architecture/sidecar-backends.md) page for the common
architecture.

## Readiness

| Deployment path | Aggregated | Disaggregated |
|---|---|---|
| Local launcher | Validated on one GPU | Validated on two GPUs with NIXL |
| Kubernetes example | Validated | Validated with NIXL |

This table covers launch topology only. The
[vLLM feature matrix](overview.md#feature-support-matrix) describes the in-process
backend; sidecar feature parity is still under evaluation. See the
[vLLM sidecar README](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/vllm/README.md)
for current protocol limitations.

## Launch Locally

From a Dynamo source checkout, build or install Dynamo so
`dynamo-vllm-sidecar` is on `PATH`. Install a vLLM build that provides
`vllm-rs` and its native gRPC server.

Start Dynamo's local discovery services, then run the aggregated launcher:

```bash
docker compose -f dev/docker-compose.yml up -d
./lib/sidecar/vllm/launch/agg.sh --model Qwen/Qwen3-0.6B
```

To run separate prefill and decode engines on two GPUs:

```bash
./lib/sidecar/vllm/launch/disagg.sh --model Qwen/Qwen3-0.6B
```

Each launcher starts the Dynamo frontend, the vLLM engine process or processes,
and the matching sidecar workers. It binds the native gRPC endpoints to
loopback.

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

No published sidecar image is available yet. Follow the
[Kubernetes quick start](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/vllm/README.md#deploy-on-kubernetes-quick-start)
to build `dynamo-sidecar`, which contains all three engine-specific sidecar
executables. The vLLM manifests pass `vllm` to the image entrypoint and pair it
with a stock upstream vLLM image. The source tree includes
[aggregated](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/vllm/deploy/agg.yaml)
and
[disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/vllm/deploy/disagg.yaml)
manifests.
