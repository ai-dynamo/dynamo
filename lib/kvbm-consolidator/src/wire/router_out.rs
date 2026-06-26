// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Egress wire format: kv-router-compatible event stream.
//!
//! Encoded as msgpack. [`EventBatch`] is a 3-tuple so it serializes as `[ts, events, rank]`
//! (matching vLLM's `msgspec(array_like=True)` envelope).

use serde::Serialize;

/// Batch envelope: `(timestamp, events, data_parallel_rank)`.
#[derive(Debug, Serialize)]
pub struct EventBatch(pub f64, pub Vec<Event>, pub Option<i32>);

/// kv-router-compatible event variant. Hash fields are u64 — obtained by projecting
/// a [`kvbm_logical::SequenceHash`] via [`crate::hash::router_block_hash`].
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    #[serde(rename = "BlockStored")]
    BlockStored {
        block_hashes: Vec<u64>,
        parent_block_hash: Option<u64>,
        token_ids: Vec<i32>,
        block_size: i32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        lora_name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    #[serde(rename = "BlockRemoved")]
    BlockRemoved {
        block_hashes: Vec<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        medium: Option<String>,
    },
    #[serde(rename = "AllBlocksCleared")]
    AllBlocksCleared {},
}
