<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

- Inference source: [`rust/proto/inference.proto`](https://github.com/biswapanda/vllm/blob/f0250a93161bac650fd7df7fb61017a25edd5888/rust/proto/inference.proto) at `f0250a93161bac650fd7df7fb61017a25edd5888`
- RL Control source: [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/76ebe5a217d7536a5661272c680f0b1e3a62f5be/rust/proto/control.proto) from [vllm-project/vllm#51316](https://github.com/vllm-project/vllm/pull/51316) at `76ebe5a217d7536a5661272c680f0b1e3a62f5be`
- `inference.proto` SHA-256: `55dcc2413caba3bd45f78dbe60c7b2578f3e0860083c26f2173f31e1d4820a93`
- `control.proto` SHA-256: `db72b0782142054293b07fd48247cc821c048213b9c95dbc37fb0d81dde8f46f`

The files are copied without modification. Update the revision and checksums together. `dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
