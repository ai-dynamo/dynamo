// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(missing_docs)]

//! Multimodal identity shared by the frontend, engines, and the router.
//!
//! A multimodal block is a block whose tokens stand in for an image, a video, or another
//! object the model consumes. Its KV cache identity is the block's tokens plus the `mm_hash`
//! of every object it touches, and both ends of the wire must compute that identity the same
//! way: the frontend when it hashes a request before routing, the engine when it publishes
//! the blocks it stored. This crate holds the data types and the rules that make the two
//! agree, and nothing else: no token hashing (`dynamo-kv-hashing`), no event codec
//! (`dynamo-kv-events`), no model-specific placeholder handling (`dynamo-llm`).
//!
//! The data path:
//!
//! 1. The frontend maps each object's identifier to its routing hash with
//!    [`hash_mm_identifier`]. For SGLang it fills the object's placeholder run with the
//!    [`pad_value_for_mm_hash`] token, the same pad value the engine derives on its side; for
//!    vLLM it carries the hash through `extra_keys` as the marker string
//!    [`mark_mm_hash_for_extra_key`] produces.
//! 2. The request's objects and their token offsets are a [`RequestExtraInfo`];
//!    [`RequestExtraInfo::to_block_level`] splits it into one [`BlockExtraInfo`] per block, and
//!    `dynamo-kv-hashing` folds each block's `mm_hash`es into that block's hash.
//! 3. The engine's KV events come back carrying the marker or the pad tokens;
//!    `dynamo-kv-events` parses the marker with [`parse_mm_hash_from_extra_key`] and
//!    normalizes the pad runs into the same [`BlockExtraInfo`], so the indexer matches what
//!    the frontend hashed.
//!
//! A leaf crate: it depends on `serde` and `xxhash-rust` and on no Dynamo crate.

mod extra_key;
mod info;
mod pad;

pub use extra_key::{mark_mm_hash_for_extra_key, parse_mm_hash_from_extra_key};
pub use info::{BlockExtraInfo, BlockMmObjectInfo, RequestExtraInfo, RequestMmObjectInfo};
pub use pad::{MM_PAD_HASH_MASK, MM_PAD_SHIFT_VALUE, hash_mm_identifier, pad_value_for_mm_hash};
