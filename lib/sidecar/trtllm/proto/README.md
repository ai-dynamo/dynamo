<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored OpenEngine protocol

These files are an exact snapshot of the canonical OpenEngine encoder-handoff
proposal below, including the `Abort.encoder_cache` release target. The proposal
is still under review and is not in a published BSR release. Merge the canonical
protocol before merging this consumer. Encoder handoffs require a matching engine
implementation.

- Source: [`proto/openengine/v1`](https://github.com/ai-dynamo/openengine/tree/4f2d17edc16d8b430df34c049c13cb5aedf0ba11/proto/openengine/v1)
- Commit: `4f2d17edc16d8b430df34c049c13cb5aedf0ba11`

TensorRT-LLM's matching transport branch vendors the same revision. The schema
adds encoder roles, cache references, and typed cleanup/required-cache capabilities.
Peer tensor transfer stays engine-owned. The existing shutdown RPC is kept in sync;
this sidecar does not implement shutdown.

SHA-256:

| File | SHA-256 |
|------|---------|
| `openengine/v1/error.proto` | `b1c559a3b7847c55a247145f1493805b56b58491a320fb26e6ebcd7072b62fb3` |
| `openengine/v1/generation.proto` | `8244d740b46028740f1ec0ceded4192288b752281ebffce218affb371e432644` |
| `openengine/v1/kv.proto` | `4f5c46a65a7998ecf34ea37117b325b0f0a5a70149702d6bca919726b40526cd` |
| `openengine/v1/lifecycle.proto` | `9361c927d4a1746331e8d764c4537ea79861ade2a206ddbcca833b6d39959776` |
| `openengine/v1/lora.proto` | `c1aa58fca3ab1b40f7c75dbbd17280b2b52ee295256479b6b0fc74f924047ef9` |
| `openengine/v1/model.proto` | `f4d87663faa3e3208b8eee1110b99e42f0b2ffcef31194d435182ab46e0ae2a4` |
| `openengine/v1/openengine.proto` | `69bec091250f4b1b21f1aab3285626f73c3709b95ee6f86b4ca64138bbbe316d` |
| `openengine/v1/server.proto` | `deb379508d5e93c719cc0c5ebe222e66af7d8784384d3cc270e95d0b48274d96` |

Update the revision and checksums together. Add or remove entries in `build.rs`'s
`PROTOS` list when the schema file set changes.
