// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire-format types for vLLM ZMQ KV event streams.
//!
//! These types mirror the Python `msgspec`-defined structures emitted by vLLM
//! engines over ZMQ PUB sockets. They are independent of the dynamo runtime
//! and can be used by any crate that needs to decode the raw ZMQ payloads.

use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use rmp_serde as rmps;

use crate::protocols::{PlacementEvent, WorkerWithDpRank};

mod convert;
mod deserialize;
mod extra_keys;
mod filter;
#[cfg(test)]
mod tests;
mod types;

pub use convert::{convert_event, create_stored_block_from_parts, create_stored_blocks};
pub use extra_keys::{extra_keys_to_block_mm_infos, parse_mm_hash_from_extra_key};
pub use types::{BlockHashValue, ExtraKeyItem, KvEventBatch, KvTokenIds, RawKvEvent};

pub fn decode_event_batch(payload: &[u8]) -> Result<KvEventBatch, rmps::decode::Error> {
    rmps::from_slice(payload)
}

#[derive(Debug, Clone)]
pub struct ZmqEventNormalizer {
    kv_block_size: u32,
    warning_count: Arc<AtomicU32>,
}

impl ZmqEventNormalizer {
    pub fn new(kv_block_size: u32) -> Self {
        Self {
            kv_block_size,
            warning_count: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn with_warning_count(kv_block_size: u32, warning_count: Arc<AtomicU32>) -> Self {
        Self {
            kv_block_size,
            warning_count,
        }
    }

    pub fn normalize(
        &self,
        raw: RawKvEvent,
        event_id: u64,
        worker: WorkerWithDpRank,
    ) -> Option<PlacementEvent> {
        convert_event(
            raw,
            event_id,
            self.kv_block_size,
            worker,
            &self.warning_count,
        )
    }
}
