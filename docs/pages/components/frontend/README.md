---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Frontend
---

# Frontend

The Dynamo Frontend is the API gateway for serving LLM inference requests. It provides OpenAI-compatible HTTP endpoints and KServe gRPC endpoints, handling request preprocessing, routing, and response formatting.

## Feature Matrix

| Feature | Status |
|---------|--------|
| OpenAI Chat Completions API | ✅ Supported |
| OpenAI Completions API | ✅ Supported |
| KServe gRPC v2 API | ✅ Supported |
| Streaming responses | ✅ Supported |
| Multi-model serving | ✅ Supported |
| Integrated routing | ✅ Supported |
| Tool calling | ✅ Supported |

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
| `--router-mode` | `round_robin` | Routing strategy: `round_robin`, `random`, `kv` |

See the [Frontend Guide](frontend-guide.md) for full configuration options.

## Next Steps

| Document | Description |
|----------|-------------|
| [Frontend Guide](frontend-guide.md) | KServe gRPC configuration and integration |
| [Router Documentation](../router/README.md) | KV-aware routing configuration |
