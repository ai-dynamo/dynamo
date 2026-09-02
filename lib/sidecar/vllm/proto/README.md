<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored vLLM protocol

Both files are copied from vLLM without modification.

- Upstream commit: [`1f9444a34ff4ebfba4d65c68971bb5306a11aa92`](https://github.com/vllm-project/vllm/commit/1f9444a34ff4ebfba4d65c68971bb5306a11aa92)
  ([vllm-project/vllm#52840](https://github.com/vllm-project/vllm/pull/52840), "[Rust Frontend][gRPC] Add LoRA lifecycle control")
- Sources: [`rust/proto/inference.proto`](https://github.com/vllm-project/vllm/blob/1f9444a34ff4ebfba4d65c68971bb5306a11aa92/rust/proto/inference.proto)
  and [`rust/proto/control.proto`](https://github.com/vllm-project/vllm/blob/1f9444a34ff4ebfba4d65c68971bb5306a11aa92/rust/proto/control.proto)
- `inference.proto` SHA-256: `078a3d2a94bd03a96fdfdfa31c13a805d00575b365dec5b3f8ed82d36f065e85`
- `control.proto` SHA-256: `1a050496e7d0f919f398d150d4bff1660d5a5eac57951137aeb0ca5970436696`

`vendored_protos_match_the_merged_vllm_release` and `lora_wire_fields_keep_their_upstream_numbers`
in `lib/sidecar/vllm/src/tests.rs` pin these checksums and the LoRA-relevant field numbers
respectively, so the vendored copies cannot silently drift from upstream.
When resyncing, copy both files verbatim and update the commit and both checksums together.

`dynamo-vllm-sidecar` generates and temporarily exports these types for `dynamo-vllm-mocker-server`.
