<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference base source: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/42156466db66f6d54cbea6075af82304cdfdaa6a/rust/proto/inference.proto) at `42156466db66f6d54cbea6075af82304cdfdaa6a`
- RL Control base source: [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/2991f864083fdd5c60aa140d4fe1a561585a85dc/rust/proto/control.proto) from [vllm-project/vllm#51316](https://github.com/vllm-project/vllm/pull/51316) and [vllm-project/vllm#53204](https://github.com/vllm-project/vllm/pull/53204) at `2991f864083fdd5c60aa140d4fe1a561585a85dc`
- Dynamo adds `GenerateRequest.native_sampling_params_json` and `ServerInfo.supports_native_sampling_params_json`; the sidecar advertises native Generate support only when the worker reports this extension.
- `inference.proto` SHA-256: `134be6a7e3ee6318ae814f7a40d49dda7b139798a44222a495ce39074731789c`
- `control.proto` SHA-256: `abfb3829c8e142cadd649943d41ea67f8f75ff070686575153ff2b6f936312aa`

Update the base revisions, extensions, and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
