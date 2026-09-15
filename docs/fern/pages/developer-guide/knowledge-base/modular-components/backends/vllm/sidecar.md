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

| Deployment path | Aggregated | P+D | E+PD | E+P+D |
|---|---|---|---|---|
| Local launcher | Validated on one GPU | Validated on two GPUs with NIXL | Validated on two GPUs with Embedding Cache transfer | Validated on three GPUs with Embedding Cache transfer and NIXL |
| Kubernetes example | Validated | Validated with NIXL | Not available | Not available |

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

For image requests, run a separate encoder with an aggregated prefill/decode engine on two GPUs:

```bash
./lib/sidecar/vllm/launch/disagg_multimodal_e_pd.sh
```

To separate encoder, prefill, and decode across three GPUs:

```bash
./lib/sidecar/vllm/launch/disagg_multimodal_epd.sh
```

The encoder-disaggregated launchers currently support images only. They use `Qwen/Qwen2.5-VL-3B-Instruct` and vLLM's `ECExampleConnector`, and require the producer and consumer to share the same EC storage path. E+P+D requires vLLM Rust frontend support for metadata-only remote-prefill decode from [vLLM #54814](https://github.com/vllm-project/vllm/pull/54814) or a later release containing it. Decode uses NIXL without an EC connector because the gRPC frontend removes EC parameters before submitting the request to EngineCore.

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
executables. The vLLM manifests run `dynamo-vllm-sidecar` as the container
command and pair it with a stock upstream vLLM image. The source tree includes
[aggregated](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/vllm/deploy/agg.yaml)
and
[disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/vllm/deploy/disagg.yaml)
manifests.
