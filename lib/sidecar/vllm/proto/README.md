<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference source: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/42156466db66f6d54cbea6075af82304cdfdaa6a/rust/proto/inference.proto) at `42156466db66f6d54cbea6075af82304cdfdaa6a`
- RL Control source: [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/2991f864083fdd5c60aa140d4fe1a561585a85dc/rust/proto/control.proto) from [vllm-project/vllm#51316](https://github.com/vllm-project/vllm/pull/51316) and [vllm-project/vllm#53204](https://github.com/vllm-project/vllm/pull/53204) at `2991f864083fdd5c60aa140d4fe1a561585a85dc`
- `inference.proto` SHA-256: `4c04f91d4967d1ba873fff6f546df138bc15cd29565c707c8554163392bb609a`
- `control.proto` SHA-256: `c8363fd4397187a44e667d3d04ada30401e078ab6763ed5144f674184dd8d787`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
