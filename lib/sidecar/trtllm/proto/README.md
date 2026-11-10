<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored OpenEngine protocol

These files are an exact snapshot of the canonical OpenEngine encoder-handoff
proposal below, including `Abort.encoder_handoff`. The proposal is still under
review and is not in a published BSR release. Merge the canonical protocol
before merging this consumer. Encoder handoffs require a matching engine
implementation.

- Source: [`proto/openengine/v1`](https://github.com/ai-dynamo/openengine/tree/9d18a3859ea6b5b08d98038d80f1203add036ea8/proto/openengine/v1)
- Commit: `9d18a3859ea6b5b08d98038d80f1203add036ea8`

TensorRT-LLM's matching transport branch vendors the same revision. The schema
adds encoder roles, handoff references, and typed cleanup/required-handoff capabilities.
Peer tensor transfer stays engine-owned. The existing shutdown RPC is kept in sync;
this sidecar does not implement shutdown.

SHA-256:

| File | SHA-256 |
|------|---------|
| `openengine/v1/error.proto` | `b1c559a3b7847c55a247145f1493805b56b58491a320fb26e6ebcd7072b62fb3` |
| `openengine/v1/generation.proto` | `0ca6bd8b14619b5b31284a702c1d43f20974436315c9f28027f4505117e1f22f` |
| `openengine/v1/kv.proto` | `12903e3ed9633b9ae181effa5a5383dc942ab975a0682e5b448e460fb02d72a7` |
| `openengine/v1/lifecycle.proto` | `1134a4854a43e7eba57d5a9c4a0da29bac727a63a97e3b8aca51c126ebd2405c` |
| `openengine/v1/lora.proto` | `c1aa58fca3ab1b40f7c75dbbd17280b2b52ee295256479b6b0fc74f924047ef9` |
| `openengine/v1/model.proto` | `07c261c084ee46377e046f1558a29e74f08dae326667ac42e43111b30141836a` |
| `openengine/v1/openengine.proto` | `69bec091250f4b1b21f1aab3285626f73c3709b95ee6f86b4ca64138bbbe316d` |
| `openengine/v1/server.proto` | `deb379508d5e93c719cc0c5ebe222e66af7d8784384d3cc270e95d0b48274d96` |

Update the revision and checksums together. Add or remove entries in `build.rs`'s
`PROTOS` list when the schema file set changes.
