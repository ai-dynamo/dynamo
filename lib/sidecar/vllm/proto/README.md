<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Source: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/v0.27.1/rust/proto/inference.proto) and [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/v0.27.1/rust/proto/control.proto)
- Tag: `v0.27.1` (commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`)
- `inference.proto` SHA-256: `4e6c90467ea308bbed9c175c60943241064c7cf7f8007e837fd235d9d61e69e2`
- `control.proto` SHA-256: `390c88e94f1b68421c54c6d9440f2088d2709a432549c7a0fe94d35ce7b37476`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
