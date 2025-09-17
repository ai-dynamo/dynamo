// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::tokens::{SequenceHash, Token};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum RouterRequest {
    // ini
    #[serde(rename = "new")]
    New {
        tokens: Vec<Token>,
    },
    MarkPrefill,
    MarkFree,
}

impl Default for RouterRequest {
    fn default() -> Self {
        RouterRequest::New { tokens: vec![] }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum RouterResponse {
    New { worker_id: i64, overlap_blocks: u32 },
    PrefillMarked { success: bool },
    FreeMarked { success: bool },
}

#[derive(Debug)]
pub struct WorkerSelectionResult {
    /// The worker id of the selected worker
    pub worker_id: i64,

    /// The total number of blocks required to prefill the request
    pub required_blocks: u64,

    /// The number of blocks that the selected worker may already have cached.
    /// This is not a guarantee, but an estimate.
    pub overlap_blocks: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ForwardPassMetrics {
    pub worker_stats: WorkerStats,
    pub kv_stats: KvStats,
    pub spec_decode_stats: Option<SpecDecodeStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WorkerStats {
    // https://lmsys.org/blog/2024-12-04-sglang-v0-4/#data-parallelism-attention-for-deepseek-models
    pub data_parallel_rank: Option<u32>,
    pub request_active_slots: u64,
    pub request_total_slots: u64,
    pub num_requests_waiting: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct KvStats {
    pub kv_active_blocks: u64,
    pub kv_total_blocks: u64,
    // percentage represented as a float from 0 to 1
    pub gpu_cache_usage_perc: f32,
    // percentage represented as a float from 0 to 1
    pub gpu_prefix_cache_hit_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct PredictiveLoadMetrics {
    pub kv_active_blocks: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LoadMetrics {
    EngineLoadMetrics(ForwardPassMetrics),
    PredictiveLoadMetrics(PredictiveLoadMetrics),
}

impl LoadMetrics {
    pub fn kv_active_blocks(&self) -> u64 {
        match self {
            LoadMetrics::EngineLoadMetrics(metrics) => metrics.kv_stats.kv_active_blocks,
            LoadMetrics::PredictiveLoadMetrics(metrics) => metrics.kv_active_blocks,
        }
    }
}

impl Default for LoadMetrics {
    fn default() -> Self {
        LoadMetrics::PredictiveLoadMetrics(PredictiveLoadMetrics::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct SpecDecodeStats {
    pub num_spec_tokens: Option<u32>,
    pub num_drafts: Option<u32>,
    pub num_draft_tokens: Option<u32>,
    pub num_accepted_tokens: Option<u32>,
    pub num_accepted_tokens_per_pos: Option<Vec<u32>>,
}

/// A [`LocalBlockHash`] is a hash computed from the tokens_ids, extra_token_ids and the optional
/// lora_id of a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct LocalBlockHash(pub u64);

/// A sequence aware hash of a block where the hash is computed from the tokens_ids, extra_token_ids
/// and the optional lora_id of a block, PLUS the hash of the parent block.
///
/// In this case, the hashing function is external and unknown.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExternalSequenceBlockHash(pub u64);

// Implement From trait for convenient conversion
impl From<u64> for ExternalSequenceBlockHash {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<i64> for ExternalSequenceBlockHash {
    /// Bitwise reinterpretation: preserves all bits, including negatives.
    /// This is lossless, but negative i64 values will appear as large u64 values.
    fn from(value: i64) -> Self {
        Self(value as u64)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrefillEvent {
    pub request_id: String,
    pub worker_id: i64,
    pub data: PrefillEventData,
    pub router_id: Uuid,
}

/// Represents the different stages of prefilling tokens for a request.
///
/// Each variant contains a `usize` representing the number of tokens
/// that are pending prefill in the request.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PrefillEventData {
    NewPrefill(usize),
    UpdatePrefill(usize),
    CompletePrefill,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActiveSequenceEvent {
    pub request_id: String,
    pub worker_id: i64,
    pub data: ActiveSequenceEventData,
    pub router_id: Uuid,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ActiveSequenceEventData {
    AddRequest {
        token_sequence: Vec<SequenceHash>,
        isl: usize,
        overlap: u32,
    },
    Free,
    MarkPrefillCompleted,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActiveBlockEvent {
    pub request_id: String,
    pub data: ActiveBlockEventData,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ActiveBlockEventData {
    NewBlock(Vec<SequenceHash>),
    FreeBlock,
}

/// Represents a collection of cache events and a shutdown flag.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheEvents {
    /// A list of cache events.
    pub events: Vec<KvCacheEvent>,
    /// A flag indicating whether the cache is shutting down.
    pub shutdown: bool,
}

/// Represents a single cache event with an ID and associated data.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheEvent {
    /// The unique identifier of the event.
    pub event_id: u64,
    /// The data associated with the event.
    pub data: KvCacheEventData,
}

/// Represents the data associated with a cache event.
///
/// Data is either stored or removed.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheEventData {
    Stored(KvCacheStoreData),
    Removed(KvCacheRemoveData),
    Cleared,
}

/// Represents the data associated with a stored cache event.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheStoreData {
    /// The optional hash of the parent block.
    pub parent_hash: Option<ExternalSequenceBlockHash>,
    /// A list of stored blocked data.
    pub blocks: Vec<KvCacheStoredBlockData>,
}

/// Represents data for a stored block.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheStoredBlockData {
    /// The hash of the block.
    pub block_hash: ExternalSequenceBlockHash,
    /// The hash of the tokens in the block.
    pub tokens_hash: LocalBlockHash,
}

/// Represents the data associated with a removed cache event.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KvCacheRemoveData {
    /// A list of block hashes to remove.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

impl Serialize for LocalBlockHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for LocalBlockHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Ok(LocalBlockHash(value))
    }
}

impl Serialize for ExternalSequenceBlockHash {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for ExternalSequenceBlockHash {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = u64::deserialize(deserializer)?;
        Ok(ExternalSequenceBlockHash(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn test_local_block_hash_serialization() {
        let hash = LocalBlockHash(12345);
        let serialized = serde_json::to_string(&hash).unwrap();
        assert_eq!(serialized, "12345");

        let deserialized: LocalBlockHash = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, hash);
    }

    #[test]
    fn test_external_sequence_block_hash_serialization() {
        let hash = ExternalSequenceBlockHash(67890);
        let serialized = serde_json::to_string(&hash).unwrap();
        assert_eq!(serialized, "67890");

        let deserialized: ExternalSequenceBlockHash = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, hash);
    }

    #[test]
    fn test_kv_cache_events_serialization() {
        let event_data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(1)),
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(2),
                tokens_hash: LocalBlockHash(3),
            }],
        });

        let event = KvCacheEvent {
            event_id: 1,
            data: event_data,
        };

        let events = KvCacheEvents {
            events: vec![event],
            shutdown: false,
        };

        let serialized = serde_json::to_string(&events).unwrap();
        let deserialized: KvCacheEvents = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.events.len(), 1);
        assert_eq!(deserialized.events[0].event_id, 1);
        if let KvCacheEventData::Stored(store_data) = &deserialized.events[0].data {
            assert_eq!(store_data.parent_hash.unwrap().0, 1);
            assert_eq!(store_data.blocks.len(), 1);
            assert_eq!(store_data.blocks[0].block_hash.0, 2);
            assert_eq!(store_data.blocks[0].tokens_hash.0, 3);
        } else {
            panic!("Expected KvCacheEventData::Stored variant");
        }
        assert!(!deserialized.shutdown);
    }

    #[test]
    fn test_kv_cache_remove_data_serialization() {
        let remove_data = KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(4), ExternalSequenceBlockHash(5)],
        };

        let serialized = serde_json::to_string(&remove_data).unwrap();
        let deserialized: KvCacheRemoveData = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.block_hashes.len(), 2);
        assert_eq!(deserialized.block_hashes[0].0, 4);
        assert_eq!(deserialized.block_hashes[1].0, 5);
    }
}
