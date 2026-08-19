<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference source: [`rust/proto/inference.proto`](https://github.com/biswapanda/vllm/blob/100f970b4ee45036a6015f4fa852cf0610556633/rust/proto/inference.proto) at `100f970b4ee45036a6015f4fa852cf0610556633`
- RL Control source: [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/76ebe5a217d7536a5661272c680f0b1e3a62f5be/rust/proto/control.proto) from [vllm-project/vllm#51316](https://github.com/vllm-project/vllm/pull/51316) at `76ebe5a217d7536a5661272c680f0b1e3a62f5be`
- `inference.proto` SHA-256: `bad6c9101f81ab8b8b78bf3e7d70992796d859dbe81219fa6de119caf9a6d94a`
- `control.proto` SHA-256: `db72b0782142054293b07fd48247cc821c048213b9c95dbc37fb0d81dde8f46f`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
