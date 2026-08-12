<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Source: [`rust/proto/inference.proto`](https://github.com/connorcarpenter15/vllm/blob/c48311247e0d5dc31eb3b275e608ead3c837c6d4/rust/proto/inference.proto) and [`rust/proto/control.proto`](https://github.com/connorcarpenter15/vllm/blob/c48311247e0d5dc31eb3b275e608ead3c837c6d4/rust/proto/control.proto)
- Commit: `c48311247e0d5dc31eb3b275e608ead3c837c6d4`
- `inference.proto` SHA-256: `2a2efafa75f0b1c3cb1dfd12fc0dc71aeb599d5f21b5628ab45a6308906c4dc3`
- `control.proto` SHA-256: `60180d43bf6d57e41e8ba4cce9b9a9d93a2c8f0c998060153c6bf25cc728ee85`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
