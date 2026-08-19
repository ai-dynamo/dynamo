<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Source: [`rust/proto/inference.proto`](https://github.com/connorcarpenter15/vllm/blob/2dd9ba09bb6e59d4f6b5794319c8cc5c9c59de2f/rust/proto/inference.proto) and [`rust/proto/control.proto`](https://github.com/connorcarpenter15/vllm/blob/2dd9ba09bb6e59d4f6b5794319c8cc5c9c59de2f/rust/proto/control.proto)
- Commit: `2dd9ba09bb6e59d4f6b5794319c8cc5c9c59de2f`
- Source `inference.proto` SHA-256: `078a3d2a94bd03a96fdfdfa31c13a805d00575b365dec5b3f8ed82d36f065e85`
- Source `control.proto` SHA-256: `cc966251a41541e83a1598f0b38d00bba05cdd11cbcf95ea84f32a27a4405611`
- Vendored `control.proto` SHA-256: `ddd085ffe6e242c2c7a7c05a9866ecbe11c5e84ecbe024fde4fe86942a62a73a`

`inference.proto` is copied without modification. `control.proto` retains only the RPCs and messages used by the sidecar while preserving their source field numbers. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
