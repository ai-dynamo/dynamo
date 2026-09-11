<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference source: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/1f9444a34ff4ebfba4d65c68971bb5306a11aa92/rust/proto/inference.proto) at `1f9444a34ff4ebfba4d65c68971bb5306a11aa92`, extended by [vLLM #55047](https://github.com/vllm-project/vllm/pull/55047) at `090b63a7d7f4d9ebafcd8499dbaf7ebd7dce3275` and [vLLM #52723](https://github.com/vllm-project/vllm/pull/52723) at `9211ec20e69a71f1d7056ad2946331c6dc0eb873`
- RL Control source: [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/2991f864083fdd5c60aa140d4fe1a561585a85dc/rust/proto/control.proto) from [vllm-project/vllm#51316](https://github.com/vllm-project/vllm/pull/51316) and [vllm-project/vllm#53204](https://github.com/vllm-project/vllm/pull/53204) at `2991f864083fdd5c60aa140d4fe1a561585a85dc`
- `inference.proto` SHA-256: `522085198ae5d60256261542cd667285c426fa8de3c5eafc0beaea3fa79e2a2d`
- `control.proto` SHA-256: `c8363fd4397187a44e667d3d04ada30401e078ab6763ed5144f674184dd8d787`

The control file is copied without modification. The inference file matches the current-main vLLM base plus the documented pull-request extension above. Update the revisions and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.

The initial preprocessed multimodal transport accepts inline `kwargs_data` only. Native TITO cache-only references remain outside this PR because they require cache affinity across the Dynamo routing boundary.
