<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Source: [`rust/proto/inference.proto`](https://github.com/connorcarpenter15/vllm/blob/2d2c3af18c52e8e4efa4b0b4903843b15c0dba0e/rust/proto/inference.proto) and [`rust/proto/control.proto`](https://github.com/connorcarpenter15/vllm/blob/2d2c3af18c52e8e4efa4b0b4903843b15c0dba0e/rust/proto/control.proto)
- Commit: `2d2c3af18c52e8e4efa4b0b4903843b15c0dba0e`
- `inference.proto` SHA-256: `a0d196dc240683e1c09abb54f324d4428d0c122a6802b44916ad2d96b491b06c`
- `control.proto` SHA-256: `cd4e7a8043f19d05929a2f59f5a5442894a037ef2d65832d3f7099992b1f1dbd`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
