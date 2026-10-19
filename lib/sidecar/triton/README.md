<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Triton sidecar

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Flags and behavior may change without notice.

`dynamo-triton-sidecar` connects Dynamo to a separately managed NVIDIA Triton Inference Server over its KServe v2 gRPC API, its KServe v2 HTTP/REST API, or both.
It registers one Dynamo tensor endpoint for every ready Triton model and forwards unary model-infer requests without loading Triton or GPU libraries into the sidecar process.

Choose the protocols at runtime by passing their endpoints. At least one is required.

```bash
# gRPC only
dynamo-triton-sidecar --grpc-endpoint localhost:8001 --discovery-backend file

# HTTP only
dynamo-triton-sidecar --http-endpoint localhost:8000 --discovery-backend file

# Both
dynamo-triton-sidecar \
  --grpc-endpoint localhost:8001 \
  --http-endpoint localhost:8000 \
  --discovery-backend file
```

| Flag | Environment variable | Value |
|------|----------------------|-------|
| `--grpc-endpoint` | `DYN_SIDECAR_GRPC_ENDPOINT` | `host:port` or a plaintext `http://` / `grpc://` URL |
| `--http-endpoint` | `DYN_SIDECAR_HTTP_ENDPOINT` | `host:port` or an `http://` / `https://` URL, optionally with a path prefix |

When both protocols are enabled:

- Model discovery and model configuration use gRPC.
- Startup waits until both endpoints report ready, and a model is served only when it is ready on both.
- Inference requests alternate between gRPC and HTTP.

The HTTP transport uses Triton's binary tensor data extension for every input and output, so tensors travel as raw bytes as they do over gRPC.
Triton's JSON model configuration is converted to the same protobuf `ModelConfig` that gRPC returns, so Dynamo exposes identical model metadata on either protocol.
The `--grpc-connect-attempt-timeout-secs`, `--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs` flags also govern the HTTP transport; `--grpc-connections` applies only to gRPC.

Ready models are discovered through Triton's `RepositoryIndex` API. Use one
or more `--model-name <name>` options to skip repository auto-discovery.

## Support matrix

| Feature | Supported |
|---------|-----------|
| KServe v2 unary tensor inference | Yes |
| KServe v2 gRPC transport | Yes |
| KServe v2 HTTP/REST transport | Yes, with the binary tensor data extension |
| Multiple models per Triton server | Yes |
| Typed and raw tensor contents | Yes |
| Classification output parameters | Passed through |
| Bidirectional streaming inference | No |
| Shared-memory tensor I/O | No |
| Dynamic model repository updates | No; restart the sidecar |
