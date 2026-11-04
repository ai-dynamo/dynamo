<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored OpenEngine protocol

These files are exact copies of the canonical OpenEngine encoder-handoff
proposal at the immutable revision below. The proposal is available for review;
using its encoder fields requires a matching engine implementation.

- Source: [`proto/openengine/v1`](https://github.com/ai-dynamo/openengine/tree/9b6f34f9efb17aa2c6b7e0ad737b78106a8ecca3/proto/openengine/v1)
- Commit: `9b6f34f9efb17aa2c6b7e0ad737b78106a8ecca3`

TensorRT-LLM's matching transport branch vendors the same revision. The schema
adds encoder roles, cache references and an engine-to-engine transfer RPC. Its
existing shutdown RPC is kept in sync; this sidecar does not implement shutdown.

SHA-256:

| File | SHA-256 |
|------|---------|
| `openengine/v1/error.proto` | `b1c559a3b7847c55a247145f1493805b56b58491a320fb26e6ebcd7072b62fb3` |
| `openengine/v1/generation.proto` | `ffe30cacfb9e42e00c70ea9bc29a83b65865ee67a6ceb46ce3f5d3b9a8cb36cd` |
| `openengine/v1/kv.proto` | `4f5c46a65a7998ecf34ea37117b325b0f0a5a70149702d6bca919726b40526cd` |
| `openengine/v1/lifecycle.proto` | `a0ce097480a929c6acf5e0e145bfdb1bb8998e4c7ee95ec316ce481e2558a379` |
| `openengine/v1/lora.proto` | `c1aa58fca3ab1b40f7c75dbbd17280b2b52ee295256479b6b0fc74f924047ef9` |
| `openengine/v1/model.proto` | `ca767ecd917282d0a1e2e40716a2658c372c9abbfadd043ef8100db2ff39fee7` |
| `openengine/v1/openengine.proto` | `464962a220ff8f5ae64d7468c7e7943aae9f3e29b24d95a5e7dddee99b91aef6` |
| `openengine/v1/server.proto` | `deb379508d5e93c719cc0c5ebe222e66af7d8784384d3cc270e95d0b48274d96` |

Update the revision and checksums together. Add or remove entries in `build.rs`'s
`PROTOS` list when the schema file set changes.
