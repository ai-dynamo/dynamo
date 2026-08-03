<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Sources: `rust/proto/inference.proto` and `rust/proto/control.proto`
- Commit: `a0d13bb5e70487ea5cb59ca43444ac14c3aaddef`
- `inference.proto` SHA-256: `09fca71821b9c8a4f1a7196960fc301fca7cd967847cb88c59b164f26167faca`
- `control.proto` SHA-256: `57917ab1ac0be8f5216b041167903d650be408bcb8b0b8cc4c8a78e591c1cc5c`

The files are copied without modification. Update the revision and checksums when updating the protocol. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`; both consumers will move to the upstream package once vLLM publishes it.
