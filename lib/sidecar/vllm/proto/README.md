<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Upstream base commit: [`1f9444a34ff4ebfba4d65c68971bb5306a11aa92`](https://github.com/vllm-project/vllm/commit/1f9444a34ff4ebfba4d65c68971bb5306a11aa92)
  ([vllm-project/vllm#52840](https://github.com/vllm-project/vllm/pull/52840), "[Rust Frontend][gRPC] Add LoRA lifecycle control")
- Sources: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/1f9444a34ff4ebfba4d65c68971bb5306a11aa92/rust/proto/inference.proto)
  and [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/1f9444a34ff4ebfba4d65c68971bb5306a11aa92/rust/proto/control.proto)
- Dynamo adds `GenerateRequest.native_sampling_params_json` and `ServerInfo.supports_native_sampling_params_json`; the sidecar advertises native Generate support only when the worker reports this extension.
- `inference.proto` SHA-256: `00da71dba972ccde40e1c39ee37d9a18714f8fa6f46767f15d14f875a3358f80`
- `control.proto` SHA-256: `98e67fcc85429fe33e6ee0ea7c94af0950da5eb17939934eeff28c7f021eb17d`

Update the base revision, extensions, and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
