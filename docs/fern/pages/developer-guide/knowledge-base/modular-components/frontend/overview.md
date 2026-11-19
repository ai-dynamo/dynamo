---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Frontend
subtitle: API gateway that exposes OpenAI HTTP and KServe gRPC endpoints and handles request preprocessing, routing, and response formatting.
---

The Dynamo Frontend is the API gateway for serving LLM inference requests. It provides OpenAI-compatible HTTP endpoints and KServe gRPC endpoints, handling request preprocessing, routing, and response formatting.

## Feature Matrix

| Feature | Status |
|---------|--------|
| OpenAI Chat Completions API (`/v1/chat/completions`) | ✅ Supported |
| OpenAI Completions API (`/v1/completions`) | ✅ Supported |
| OpenAI Embeddings API (`/v1/embeddings`) | ✅ Supported |
| OpenAI Responses API (`/v1/responses`) | ✅ Supported |
| OpenAI Models API (`/v1/models`) | ✅ Supported |
| Image Generation (`/v1/images/generations`) | ✅ Supported |
| Video Generation (`/v1/videos/generations`) | ✅ Supported |
| Anthropic Messages API (`/v1/messages`) | 🧪 Experimental |
| KServe gRPC v2 API | ✅ Supported |
| Streaming responses (SSE) | ✅ Supported |
| Multi-model serving | ✅ Supported |
| Integrated KV-aware routing | ✅ Supported |
| Tool calling | ✅ Supported |
| TLS (HTTPS) | ✅ Supported |
| Swagger UI (`/docs`) | ✅ Supported |
| NVIDIA request extensions (`nvext`) | ✅ Supported |

## Quick Start

### Prerequisites

- Dynamo platform installed
- `etcd` and `nats-server -js` running
- At least one backend worker registered

### HTTP Frontend

```bash
python -m dynamo.frontend --http-port 8000
```

This starts an OpenAI-compatible HTTP server with integrated pre/post processing and routing. Backends are auto-discovered when they call `register_model`.

The frontend does the pre and post processing. To do this it will need access to the model configuration files: `config.json`, `tokenizer.json`, `tokenizer_config.json`, etc. It does not need the weights.

Frontend will download the files it needs from Hugging Face, no setup is required. However we recommend setting up [modelexpress-server](https://github.com/ai-dynamo/modelexpress) and a shared folder such as a Kubernetes PVC. This ensures the model is only downloaded once across the whole cluster.

If the model is not available on Hugging Face, such as a private or customized model, you will need to make the model files available locally at the same file path as on the backend. The backend's `--model-path <here>` will need to exist on the frontend and contain at least the configuration (JSON) files.

### KServe gRPC Frontend

```bash
python -m dynamo.frontend --kserve-grpc-server
```

See the [KServe gRPC Frontend](frontend-guide.md) page for KServe-specific configuration and message formats.

### Kubernetes

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: frontend-example
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0
          command:
            - python3
            - -m
            - dynamo.frontend
            - --http-port
            - "8000"
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--http-port` | 8000 | HTTP server port |
| `--kserve-grpc-server` | false | Enable KServe gRPC server |
| `--router-mode` | `round-robin` | Routing strategy: `round-robin`, `random`, `kv`, `direct`, `least-loaded`, `device-aware-weighted` (`least-loaded` uses synchronous prefill fallback in disaggregated prefill mode) |

See the [KServe gRPC Frontend](frontend-guide.md) page for KServe configuration options.

## Responses API Output

Dynamo serves `/v1/responses` by converting each request to Chat Completions and converting the result back. The converted output differs from the OpenAI Responses API in these ways:

- The existing `reasoning.summary` request option enables raw reasoning output. Dynamo returns it in `content` as `reasoning_text` parts and streams it through `reasoning_text` events. The `summary` array stays empty.
- Assistant messages carry `phase`. When the model's output parser delimits assistant messages, as the Muse Spark parser does, each message keeps the phase the parser reports. Otherwise a function call closes preceding text as `commentary`. Resumed reasoning also closes preceding text as `commentary` when the request can call tools, even if the reasoning text is hidden. When no tool can be called, hidden reasoning keeps an undelimited message open. Text open at the end is `final_answer`.
- For messages the parser does not delimit, streaming sets `phase` on `response.output_item.done` and in `response.completed`, not on `response.output_item.added`.
- Non-streaming responses preserve recorded post-parse order when it matches the final aggregated content and calls. With ordered deltas, a text, call, text sequence produces a `commentary` message, the call, and a `final_answer` message in either mode.
- A legacy parser can combine visible text and reasoning into one delta without preserving their relative order. In that case streaming treats mixed text as a continuation of an open message when no reasoning span is active. Non-streaming falls back to the aggregate fields and places text before calls when their order is unknown.
- When a request replays earlier output, consecutive assistant messages merge into one assistant turn regardless of `phase`, unless the model's chat template renders phases, as Muse Spark's does.

## Next Steps

| Document | Description |
|----------|-------------|
| [Configuration Reference](../../../../reference/components/frontend-configuration.mdx) | All CLI arguments, env vars, and HTTP endpoints |
| [Protocol Field Handling](protocol-field-handling.md) | Ownership, transport, and end-to-end review contract for request and response fields |
| [KServe gRPC Frontend](frontend-guide.md) | KServe endpoints, backend registration, and flow-control tuning |
| [Python Route Extensions](python-route-extensions.md) | Add trusted HTTP routes from a Python package or importable module |
| [NVIDIA Request Extensions (nvext)](../../../additional-resources/nvidia-request-extensions-nvext.md) | Custom request fields for routing hints and cache control |
| [Router Documentation](../router/overview.md) | KV-aware routing configuration |
