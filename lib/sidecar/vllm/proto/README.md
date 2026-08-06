<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference source: [`rust/proto/inference.proto`](https://github.com/connorcarpenter15/vllm/blob/2d2c3af18c52e8e4efa4b0b4903843b15c0dba0e/rust/proto/inference.proto) at `2d2c3af18c52e8e4efa4b0b4903843b15c0dba0e`
- RL Control source: [`rust/proto/control.proto`](https://github.com/connorcarpenter15/vllm/blob/7f3ab290464ac319e867b0d011d11dd6b2ff37f4/rust/proto/control.proto) from [connorcarpenter15/vllm#22](https://github.com/connorcarpenter15/vllm/pull/22) at `7f3ab290464ac319e867b0d011d11dd6b2ff37f4`
- `inference.proto` SHA-256: `a0d196dc240683e1c09abb54f324d4428d0c122a6802b44916ad2d96b491b06c`
- `control.proto` SHA-256: `ec414b622e2412c59215aa0de4413625be5c0fdd32ccb4f6102dd4418127aa64`

The vendored Control schema composes the RL RPCs above with field 10 from [vllm-project/vllm#51178](https://github.com/vllm-project/vllm/pull/51178), which advertises explicit data-parallel rank routing. Update the source revisions and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
