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

Build or install Dynamo from a source checkout so `dynamo-trtllm-sidecar` is on
`PATH`. The engine must serve `--grpc --grpc-protocol openengine`, which means
`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc27.dev202609170000` or newer; the
launchers install the pinned OpenEngine Python bindings, which those releases do
not ship.

```bash
docker compose -f dev/docker-compose.yml up -d          # local discovery services
./lib/sidecar/trtllm/launch/agg.sh --model Qwen/Qwen3-0.6B
```

Each launcher starts the Dynamo frontend, the engine, and the sidecar, and binds
the engine's gRPC endpoint to loopback — it is unauthenticated and plaintext.
`launch/disagg.sh` is the same on two GPUs, with a NIXL cache transceiver on both
engines for the KV handoff. Either way the frontend serves the usual endpoint:

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

The source tree ships an
[aggregated manifest](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/trtllm/deploy/agg.yaml) and a
[disaggregated one](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/trtllm/deploy/disagg.yaml) that runs
prefill and decode as separate worker pods. Both need two images you build
yourself: `dynamo-sidecar`, which carries all three engine-specific executables,
and a TensorRT-LLM image with the pinned OpenEngine bindings layered on — the
release ships the servicer but not the bindings that it and the manifests'
health probes import.

Read the disaggregated manifest's header before applying it: it requests
`rdma/ib` on both engines, which you drop if your fabric does not expose it. The
[README](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/trtllm/README.md#deploy-on-kubernetes) has the full walkthrough.
