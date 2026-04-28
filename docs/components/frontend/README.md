---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Frontend
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

**Recommended: worker-self-hosted metadata.** Each worker exposes its metadata files on its system status server (`DYN_SYSTEM_PORT`, automatically set to `9090` by the Kubernetes operator). At registration the frontend fetches each file over HTTP from `http://<worker>/v1/metadata/<slug>/<filename>` and verifies its blake3 checksum against the model's discovery card before caching it under `~/.cache/dynamo/mdc/`. This path requires no shared filesystem and no HuggingFace access on the frontend pod, works uniformly for public, private, and custom models, and is the default. Set `DYN_SELF_HOST_METADATA=false` on the worker (or pass `self_host_metadata=False` to `register_model(...)`) only if you have a specific reason to opt out. For non-Kubernetes setups, set `DYN_SYSTEM_PORT` explicitly on the worker — otherwise the system status server doesn't run and self-host silently no-ops, falling through to the legacy paths below.

**Legacy fallback paths** (used for opt-out workers, pre-PR workers, or environments where `DYN_SYSTEM_PORT` isn't set): the frontend can download files directly from Hugging Face for public models (optionally via [modelexpress-server](https://github.com/ai-dynamo/modelexpress) when configured), or read them from the same local path the worker used — the backend's `--model-path <here>` must exist on the frontend pod with at least the configuration JSON files. For private or customized models not on Hugging Face, the local-path approach is the only legacy option.

### KServe gRPC Frontend

```bash
python -m dynamo.frontend --kserve-grpc-server
```

See the [Frontend Guide](frontend-guide.md) for KServe-specific configuration and message formats.

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
| `--router-mode` | `round-robin` | Routing strategy: `round-robin`, `random`, `kv`, `direct`, `least-loaded`, `device-aware-weighted` (`power-of-two` and `least-loaded` use synchronous prefill fallback in disaggregated prefill mode) |

See the [Frontend Guide](frontend-guide.md) for full configuration options.

## Next Steps

| Document | Description |
|----------|-------------|
| [Configuration Reference](configuration.md) | All CLI arguments, env vars, and HTTP endpoints |
| [Frontend Guide](frontend-guide.md) | KServe gRPC configuration and integration |
| [NVIDIA Request Extensions (nvext)](nvext.md) | Custom request fields for routing hints and cache control |
| [Router Documentation](../router/README.md) | KV-aware routing configuration |
