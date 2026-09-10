<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference source: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/3d1f5cee1552b8208f3009c75f8bc856f27e0eff/rust/proto/inference.proto) at `3d1f5cee1552b8208f3009c75f8bc856f27e0eff`
- Control source: [`rust/proto/control.proto`](https://github.com/JulienDarve/vllm/blob/20905fbeda0eb9760e1782d6ea5a0f96d4fe2457/rust/proto/control.proto) from [JulienDarve/vllm#3](https://github.com/JulienDarve/vllm/pull/3) at `20905fbeda0eb9760e1782d6ea5a0f96d4fe2457`
- `inference.proto` SHA-256: `6152c306583166ecd691c9c715cab950523e8d1ed2db3dc2bcb538f6ca90e56f`
- `control.proto` SHA-256: `5ea0f7b09652882e0692de334d27585965b625553801c6739e44de3e200eeaac`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
