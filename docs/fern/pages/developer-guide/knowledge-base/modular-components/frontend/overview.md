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
  graphs:
    - name: frontend
      replicas: 1
      services:
        - name: Frontend
          image: nvcr.io/nvidia/dynamo/dynamo-vllm:latest
          command:
            - python
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
| `--router-mode` | `round-robin` | Routing strategy: `round-robin`, `random`, `power-of-two`, `kv`, `direct`, `least-loaded`, `device-aware-weighted` (`power-of-two` and `least-loaded` use synchronous prefill fallback in disaggregated prefill mode) |

See the [KServe gRPC Frontend](frontend-guide.md) page for KServe configuration options.

## Responses API Output

Dynamo serves `/v1/responses` by converting each request to Chat Completions and converting the result back. The converted output differs from the OpenAI Responses API in these ways:

- Reasoning items are always returned when the model produces reasoning, with the raw trace in `content` as `reasoning_text` parts. When `reasoning.summary` is set, `summary` mirrors the raw trace and streaming emits `reasoning_summary_*` events instead of `reasoning_text` events.
- Assistant messages carry `phase`. Text followed by a function call is `commentary`, and text still open when the response ends is `final_answer`. Streamed text followed by more reasoning is `commentary` when the request can call tools (it defines tools and `tool_choice` is not `none`), and `final_answer` otherwise.
- Streaming sets `phase` on `response.output_item.done` and in `response.completed`, not on `response.output_item.added`.
- A non-streaming response has no ordering information for text and calls, so its text becomes one `commentary` message placed before the calls. A streamed text, call, text sequence returns a `commentary` message, the call, and a `final_answer` message.

## Next Steps

| Document | Description |
|----------|-------------|
| [Configuration Reference](../../../../reference/components/frontend-configuration.mdx) | All CLI arguments, env vars, and HTTP endpoints |
| [Protocol Field Handling](protocol-field-handling.md) | Ownership, transport, and end-to-end review contract for request and response fields |
| [KServe gRPC Frontend](frontend-guide.md) | KServe endpoints, backend registration, and flow-control tuning |
| [Python Route Extensions](python-route-extensions.md) | Add trusted HTTP routes from a Python package or importable module |
| [NVIDIA Request Extensions (nvext)](../../../additional-resources/nvidia-request-extensions-nvext.md) | Custom request fields for routing hints and cache control |
| [Router Documentation](../router/overview.md) | KV-aware routing configuration |
