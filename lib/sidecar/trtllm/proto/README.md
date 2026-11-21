<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Vendored OpenEngine protocol

This is an exact snapshot of the native encoded-media contract at published
[OpenEngine source commit 2a442e0](https://github.com/ai-dynamo/openengine/tree/2a442e03912d6c98cdb81b73ab67e05b69d9ea75/proto/openengine/v1).
It is not yet a released protocol/runtime. Matching TRT-LLM bindings are required;
EPD startup checks the engine schema release against `REVISION`.

The optional Encoder service provides unary Encode and idempotent output release,
separately from streaming Inference.Generate and Control. Framework-native
references stay opaque to the sidecar. It adds no encoder-specific capability
discovery. Tensor transfer remains engine-owned.

SHA-256:

| File | SHA-256 |
| --- | --- |
| `openengine/v1/encoded_media.proto` | `1342b19cdab5b9dea54496ec6ef8e3ac9e9beb3aeb720c36cdc6c232478a991c` |
| `openengine/v1/encoding.proto` | `140a4407ca1172b99f8739ab474498d50e411f99595441e73bd533336dfa2de6` |
| `openengine/v1/error.proto` | `b1c559a3b7847c55a247145f1493805b56b58491a320fb26e6ebcd7072b62fb3` |
| `openengine/v1/generation.proto` | `9808bad8bb20839a1efa18402f5694f89be7c05220e0028aba39875b9950ad21` |
| `openengine/v1/kv.proto` | `4f5c46a65a7998ecf34ea37117b325b0f0a5a70149702d6bca919726b40526cd` |
| `openengine/v1/lifecycle.proto` | `a0ce097480a929c6acf5e0e145bfdb1bb8998e4c7ee95ec316ce481e2558a379` |
| `openengine/v1/lora.proto` | `c1aa58fca3ab1b40f7c75dbbd17280b2b52ee295256479b6b0fc74f924047ef9` |
| `openengine/v1/model.proto` | `5166cdcdaa260b1f5de2fa8d170dacbab1595af9413e73f09c6902b25f553284` |
| `openengine/v1/openengine.proto` | `6a9fe99dfde89ccc1d3dbf5558ca1c067e2805c3f230b1d034c8ef8fe636f659` |
| `openengine/v1/server.proto` | `a6ce9934a3f4ff508ec7c9b8fefdb69bd8025a63812c42b2e1ac0d8765ce0db8` |

Update REVISION, schema files and checksums together. Update build.rs's PROTOS
list when the file set changes.
