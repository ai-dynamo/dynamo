<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Triton sidecar

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Flags and behavior may change without notice.

`dynamo-triton-sidecar` connects Dynamo to a separately managed NVIDIA Triton Inference Server over its KServe v2 gRPC API.
It registers one Dynamo tensor endpoint for every ready Triton model and forwards unary model-infer requests without loading Triton or GPU libraries into the sidecar process.

Start Triton with its gRPC service enabled, then run:

```bash
dynamo-triton-sidecar \
  --grpc-endpoint localhost:8001 \
  --discovery-backend file
```

Ready models are discovered through Triton's gRPC `RepositoryIndex` API. Use one
or more `--model-name <name>` options to skip repository auto-discovery.

## Support matrix

| Feature | Supported |
|---------|-----------|
| KServe v2 unary tensor inference | Yes |
| Multiple models per Triton server | Yes |
| Typed and raw tensor contents | Yes |
| Classification output parameters | Passed through |
| Bidirectional streaming inference | No |
| Shared-memory tensor I/O | No |
| Dynamic model repository updates | No; restart the sidecar |
