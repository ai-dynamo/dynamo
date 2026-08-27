---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: SGLang Sidecar
subtitle: Run Dynamo beside a stock SGLang engine through native gRPC.
---

<Warning>
**Experimental.** The SGLang sidecar, launchers, packaging, and feature
coverage can change without notice.
</Warning>

`dynamo-sglang-sidecar` is a CPU-only Dynamo worker that connects to SGLang's
native gRPC service. It preserves the upstream engine process and argument
surface while using Dynamo for request handling and distributed serving. See
the [Sidecar Backends](../../../concepts/system-architecture/sidecar-backends.md) page for the common
architecture.

## Readiness

| Deployment path | Aggregated | Disaggregated |
|---|---|---|
| Local launcher | Validated on one GPU | Validated on two GPUs with NIXL |
| Kubernetes example | Validated | Validated with NIXL |

This table covers launch topology only. The
[SGLang feature matrix](overview.md#feature-support-matrix) describes the
in-process backend; sidecar feature parity is still under evaluation. See the
[SGLang sidecar README](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/sglang/README.md)
for current protocol details.

## Launch Locally

From a Dynamo source checkout, build or install Dynamo so
`dynamo-sglang-sidecar` is on `PATH`. Install SGLang v0.5.16 or later, which
provides the native `--grpc-port` server option.

Start Dynamo's local discovery services, then run the aggregated launcher:

```bash
docker compose -f dev/docker-compose.yml up -d
./lib/sidecar/sglang/launch/agg.sh --model Qwen/Qwen3-0.6B
```

To run separate prefill and decode engines on two GPUs:

```bash
./lib/sidecar/sglang/launch/disagg.sh --model Qwen/Qwen3-0.6B
```

Each launcher starts the Dynamo frontend, the SGLang engine process or
processes, and the matching sidecar workers. It binds SGLang's HTTP and native
gRPC endpoints to loopback.

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
[Kubernetes quick start](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/sglang/README.md#deploy-on-kubernetes-quick-start)
to build `dynamo-sidecar`, which contains all three engine-specific sidecar
executables. The SGLang manifests run `dynamo-sglang-sidecar` as the container
command and pair it with a stock upstream SGLang image. The source tree includes
[aggregated](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/sglang/deploy/agg.yaml)
and
[disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/sglang/deploy/disagg.yaml)
manifests.
