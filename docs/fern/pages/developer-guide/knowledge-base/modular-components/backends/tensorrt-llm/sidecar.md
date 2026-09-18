---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: TensorRT-LLM Sidecar
subtitle: Run Dynamo beside a TensorRT-LLM engine through its OpenEngine gRPC API.
---

> [!WARNING]
> **Experimental.** The TensorRT-LLM sidecar, launcher, packaging, and feature
> coverage can change without notice.

`dynamo-trtllm-sidecar` is a CPU-only Dynamo worker that connects to
TensorRT-LLM's OpenEngine gRPC API (`openengine.v1`), served by
`trtllm-serve --grpc --grpc-protocol openengine`. It preserves the upstream
engine process and argument surface while using Dynamo for request handling and
distributed serving. See the
[Sidecar Backends](../../../concepts/system-architecture/sidecar-backends.md) page for the common
architecture.

## Readiness

| Deployment path | Aggregated | Disaggregated |
|---|---|---|
| Local launcher | Validated on one GPU | Validated on one GPU, both engines co-located; the launcher defaults to two |
| Kubernetes example | Validated | Validated, prefill and decode on separate pods |

This table covers launch topology only. The
[TensorRT-LLM feature matrix](overview.md#feature-support-matrix) describes the
in-process backend; sidecar feature parity is still under evaluation.
Disaggregated prefill/decode is supported over the OpenEngine contract: a
prefill worker marks its request `context_only` and returns the `PrefillReady`
KV handoff that a decode worker replays. Running it needs an engine with a KV
cache transceiver configured on both legs. See the
[TensorRT-LLM sidecar README](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/trtllm/README.md)
for other protocol limitations.

## Launch Locally

From a Dynamo source checkout, build or install Dynamo so
`dynamo-trtllm-sidecar` is on `PATH`. You need a TensorRT-LLM build that
serves `--grpc --grpc-protocol openengine` —
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc27.dev202609170000` or newer, the
first releases carrying the OpenEngine servicer. The launcher installs the
pinned OpenEngine Python bindings itself, which those releases do not ship.

Start Dynamo's local discovery services, then run the aggregated launcher:

```bash
docker compose -f dev/docker-compose.yml up -d
./lib/sidecar/trtllm/launch/agg.sh --model Qwen/Qwen3-0.6B
```

The launcher starts the Dynamo frontend, TensorRT-LLM engine, and sidecar. It
binds TensorRT-LLM's OpenEngine gRPC endpoint to loopback, which is
unauthenticated and plaintext.

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

For prefill and decode on two GPUs, run the disaggregated launcher instead. It
starts both engines with a NIXL cache transceiver, which the KV handoff needs on
both legs:

```bash
./lib/sidecar/trtllm/launch/disagg.sh --model Qwen/Qwen3-0.6B
```

The frontend serves the same endpoint either way.

## Deploy on Kubernetes

No published sidecar image is available yet. Follow the
[Kubernetes quick start](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/trtllm/README.md#deploy-on-kubernetes-quick-start)
to build `dynamo-sidecar`, which contains all three engine-specific sidecar
executables. The TensorRT-LLM manifest runs `dynamo-trtllm-sidecar` as the
container command and pairs it with a TensorRT-LLM image that serves
OpenEngine gRPC. Layer the pinned OpenEngine Python bindings onto
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc27.dev202609170000` or newer and
push the result: the release ships the servicer but not the bindings it and the
manifest's health probes import. The source tree includes an
[aggregated deployment manifest](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/trtllm/deploy/agg.yaml)
and a
[disaggregated one](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/trtllm/deploy/disagg.yaml)
that runs prefill and decode as separate worker pods. Read the disaggregated
manifest's header before applying it: it requests `rdma/ib` on both engines for
the KV transfer, which you drop if your fabric does not need it.
