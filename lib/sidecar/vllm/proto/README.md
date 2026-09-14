<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference base source: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/42156466db66f6d54cbea6075af82304cdfdaa6a/rust/proto/inference.proto) at `42156466db66f6d54cbea6075af82304cdfdaa6a`
- RL Control base source: [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/2991f864083fdd5c60aa140d4fe1a561585a85dc/rust/proto/control.proto) from [vllm-project/vllm#51316](https://github.com/vllm-project/vllm/pull/51316) and [vllm-project/vllm#53204](https://github.com/vllm-project/vllm/pull/53204) at `2991f864083fdd5c60aa140d4fe1a561585a85dc`
- Dynamo adds `GenerateRequest.native_sampling_params_json`, `ServerInfo.supports_native_sampling_params_json`, and the preprocessed multimodal feature transport from [vLLM #55047](https://github.com/vllm-project/vllm/pull/55047); the sidecar advertises native Generate support only when the worker reports the corresponding extension.
- `inference.proto` SHA-256: `91f723d8f48755806ef05c9aa0cc0d43bc7fb23b0f94ddd7a7e6f0b4f2224b13`
- `control.proto` SHA-256: `abfb3829c8e142cadd649943d41ea67f8f75ff070686575153ff2b6f936312aa`

Update the base revisions, extensions, and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.

The initial preprocessed multimodal transport accepts inline `kwargs_data` only. Native TITO cache-only references remain outside this PR because they require cache affinity across the Dynamo routing boundary.
