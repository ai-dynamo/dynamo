<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Source: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/3d1f5cee1552b8208f3009c75f8bc856f27e0eff/rust/proto/inference.proto) and [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/3d1f5cee1552b8208f3009c75f8bc856f27e0eff/rust/proto/control.proto)
- Revision: `3d1f5cee1552b8208f3009c75f8bc856f27e0eff`
- `inference.proto` SHA-256: `6152c306583166ecd691c9c715cab950523e8d1ed2db3dc2bcb538f6ca90e56f`
- `control.proto` SHA-256: `390c88e94f1b68421c54c6d9440f2088d2709a432549c7a0fe94d35ce7b37476`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
