<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference source: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/3d1f5cee1552b8208f3009c75f8bc856f27e0eff/rust/proto/inference.proto) at `3d1f5cee1552b8208f3009c75f8bc856f27e0eff`
- Control source: [`rust/proto/control.proto`](https://github.com/JulienDarve/vllm/blob/0a7bb87ee19b0cdcb8b12f1400521cd9598fcd04/rust/proto/control.proto) from [JulienDarve/vllm#3](https://github.com/JulienDarve/vllm/pull/3) at `0a7bb87ee19b0cdcb8b12f1400521cd9598fcd04`
- `inference.proto` SHA-256: `6152c306583166ecd691c9c715cab950523e8d1ed2db3dc2bcb538f6ca90e56f`
- `control.proto` SHA-256: `8dc94a80be71e1883c7345bb65053b1645982a4065f0fde67c6ddb22598064bd`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
