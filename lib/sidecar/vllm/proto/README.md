<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Compatibility release: vLLM `0.28.0`
- Local `inference.proto` SHA-256: `6152c306583166ecd691c9c715cab950523e8d1ed2db3dc2bcb538f6ca90e56f`
- Local `control.proto` SHA-256: `c8363fd4397187a44e667d3d04ada30401e078ab6763ed5144f674184dd8d787`

The local schemas are wire-compatible with vLLM `0.28.0` and retain only the fields consumed by the sidecar plus additive compatibility fields. `ParallelismInfo.world_size` is additive and unavailable from a vLLM `0.28.0` server, so RL worker discovery remains disabled with that release. Update the compatibility release and checksums together. `dynamo-vllm-sidecar` generates and exports these types for `dynamo-vllm-mocker-server`.
