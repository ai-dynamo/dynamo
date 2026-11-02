<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# dynamo-multimodal

Multimodal identity shared by the frontend, engines, and the router: the per-block and per-request multimodal metadata (`BlockMmObjectInfo`, `BlockExtraInfo`, `RequestMmObjectInfo`, `RequestExtraInfo` and its `to_block_level` split), the pad and identifier rules (`hash_mm_identifier`, `pad_value_for_mm_hash`, `MM_PAD_SHIFT_VALUE`, `MM_PAD_HASH_MASK`, pinned to SGLang's `MultimodalItem._compute_pad_value`), and the vLLM `extra_keys` marker (`mark_mm_hash_for_extra_key`, `parse_mm_hash_from_extra_key`). Both ends of the wire compute a multimodal block's identity with these rules, so the block hash the router recomputes from a request matches the one the engine published for the blocks it stored.

A leaf crate: `[dependencies]` is exactly `serde` and `xxhash-rust`, no Dynamo crate. `dynamo-kv-hashing` consumes `BlockExtraInfo` when it folds `mm_hash` into a block hash; `dynamo-kv-events` parses the marker back out of vLLM extra keys and normalizes placeholder runs into `BlockExtraInfo`; `dynamo-scheduler`, `dynamo-kv-router`, `dynamo-llm`, and the Python bindings carry the metadata along the request path. The layering gate (`scripts/router_layering_check.sh`) checks the dependency set; `scripts/router_public_api_check.sh` keeps the public surface in `public-api.txt`.

Deliberately elsewhere: the vLLM extra-key codec (`DYNAMO_CACHE_SALT_PREFIX`, `extra_keys_to_cache_namespace`, `extra_keys_to_block_mm_infos`, which take the ZMQ wire item) and the placeholder-run normalization stay in `dynamo-kv-events`; `compute_block_hash*` stays in `dynamo-kv-hashing`; model-specific multimodal routing (`preprocessor/mm_routing/*`, the embedding caches, the encoder router) stays in `lib/llm`; `dynamo_kv_hashing::RequestMmObjectInfo` is a different shape with the same name for the PLH request and is not this type.

Part of the router split (see `lib/router/README.md`).
