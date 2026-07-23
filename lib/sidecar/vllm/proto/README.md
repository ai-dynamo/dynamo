<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Source: `rust/proto/vllm_grpc.proto`
- Release: `v0.25.1`
- Commit: `752a3a504485790a2e8491cacbb35c137339ad34`
- SHA-256: `7cccd0e1b2e54f189550e1090cc80321fc2bbd188c98a4e22701b28fdeb177b6`

The file is copied without modification. Update the revision and checksum when
updating the protocol. `dynamo-vllm-sidecar` generates and temporarily exports
these types for `dynamo-vllm-mocker-server`; both consumers will move to the
upstream package once vLLM publishes it.
