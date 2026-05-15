// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(missing_docs)]

//! Universal `Request → Vec<PositionalLineageHash>` contract for KV cache identity.
//!
//! See `README.md` for the design rationale (the three-representation problem, why PLH
//! wins, the multimodal gap, and the extension-vs-PLH-alone caveat). This crate is the
//! single contract that the router, consolidator, kvbm, and framework workers should
//! converge on for block hashing.
//!
//! # Pure-computation contract
//!
//! This crate is intentionally a library, not a service. It contains no async, no
//! transports, no traits over runtime/scheduler config, and no event types. Those
//! belong in higher layers (kvbm-connector, consolidator, kv-router).

mod block;
mod compute;
mod error;
mod request;
mod salt;

pub use block::UniversalBlock;
pub use error::KvHashingError;
pub use request::{Request, RequestBuilder, RequestMmObjectInfo};

// Re-export the underlying primitives so consumers can depend solely on this crate.
pub use dynamo_tokens::{
    BlockHash, MM_SLOT_TAG_PLACEHOLDER, MM_SLOT_TAG_TOKEN, MmInfoError, PositionalLineageHash,
    SaltHash, SequenceHash, Token, TokenBlockMmInfo, compute_block_bytes_with_mm,
    compute_block_hash, compute_hash_v2, compute_salt_hash_from_bytes, validate_and_sort_mm_info,
};
