---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sidecar Backends
subtitle: Run Dynamo beside a stock inference engine through its native gRPC API.
---

> [!WARNING]
> **Experimental.** Sidecar packaging, launchers, and API coverage are still
> evolving. The sidecar path does not yet match every feature of the in-process
> backends.

A Dynamo sidecar runs beside the inference engine process. It registers the
engine with Dynamo discovery and forwards engine events into the Dynamo event
plane. Today, requests also pass through the sidecar. The target design routes
requests directly to the engine's native gRPC service.

## Design Goals

- Keep the upstream engine's native serve path and argument surface.
- Move toward public, versioned gRPC contracts with explicit backward
  compatibility instead of importing private engine APIs.
- Isolate Dynamo and engine dependencies in separate processes.
- Attribute failures through engine-specific and Dynamo-specific logs and health
  checks.
- Reuse Dynamo's frontend, routing, planning, and disaggregated-serving
  orchestration.

## Architecture

```mermaid
flowchart LR
  F[Dynamo Frontend]
  subgraph W[Same host or Kubernetes pod]
    direction TB
    S[Dynamo Sidecar] <-->|Native gRPC| E[Inference Engine]
  end
  S -->|Discovery and Event planes| F
  F -->|Request plane*<br/>Native gRPC| E
```

<sup>*</sup> The direct request path is the target design. Today, requests pass
through the sidecar.

In the target design, the frontend and router resolve the engine endpoint
through discovery, then send requests directly to the engine. The sidecar stays
off the request path and uses the engine's native gRPC service for metadata and
event integration with Dynamo's discovery and event planes.

## Container Packaging

The unified CPU-only sidecar image contains three engine-specific executables:

| Engine | Container command |
|---|---|
| vLLM | `dynamo-vllm-sidecar` |
| SGLang | `dynamo-sglang-sidecar` |
| TensorRT-LLM | `dynamo-trtllm-sidecar` |

The image has no default entrypoint. Each Kubernetes deployment selects the
matching executable with `command`. The inference engine remains in a separate
GPU container, so the sidecar image does not include vLLM, SGLang,
TensorRT-LLM, CUDA, or engine-specific Python dependencies.

No published unified sidecar image is available yet. Build `dynamo-sidecar`
from the
[sidecar Dockerfile](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/Dockerfile).

## Target Responsibilities

| Layer | Responsibility |
|---|---|
| Dynamo frontend and router | OpenAI-compatible API, preprocessing, routing, and direct native gRPC requests to the engine |
| Dynamo sidecar | Engine registration and discovery, plus metadata and event forwarding |
| Inference engine | Native gRPC request serving, scheduling, sampling, token generation, KV cache, and GPU execution |

## Current Readiness

| Backend | Local launcher | Kubernetes example |
|---|---|---|
| [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/sidecar.md) | Aggregated and disaggregated | Aggregated and disaggregated |
| [SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/sidecar.md) | Aggregated and disaggregated | Aggregated and disaggregated |
| [TensorRT-LLM](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/sidecar.md) | Aggregated | Aggregated |

Disaggregated launch paths require multiple GPUs and use NIXL for KV transfer.
This table describes validated launch topologies, not feature parity with the
in-process backends.

See the
[sidecar source, unified Dockerfile, and engine-specific READMEs](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar)
for implementation details.
