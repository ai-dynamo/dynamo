// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::tokens::{SequenceHash, Token};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A worker identifier.
pub type WorkerId = u64;

/// A data parallel rank identifier.
pub type DpRank = u32;

/// A worker identifier combined with its data parallel rank.
/// Used for routing decisions in data parallel setups.
/// dp_rank = 0 indicates either DP not enabled or the first rank.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WorkerWithDpRank {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
}

impl WorkerWithDpRank {
    pub fn new(worker_id: WorkerId, dp_rank: DpRank) -> Self {
        Self { worker_id, dp_rank }
    }

    pub fn from_worker_id(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            dp_rank: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum RouterRequest {
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
    New {
        worker_id: WorkerId,
        #[serde(default)]
        dp_rank: DpRank,
        overlap_blocks: u32,
    },
    PrefillMarked {
        success: bool,
    },
    FreeMarked {
        success: bool,
    },
}

#[derive(Debug)]
pub struct WorkerSelectionResult {
    /// The full worker information including dp_rank
    pub worker: WorkerWithDpRank,

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
    pub data_parallel_rank: Option<DpRank>,
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

/// Active load metrics for a worker, used for busy detection.
///
/// Published by workers (with only `active_decode_blocks`) and by the scheduler
/// (with both `active_decode_blocks` and `active_prefill_tokens`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ActiveLoad {
    pub worker_id: WorkerId,
    #[serde(default)]
    pub dp_rank: DpRank,
    /// Number of active KV cache blocks on the worker (decode phase).
    pub active_decode_blocks: Option<u64>,
    /// Number of active prefill tokens (from scheduler's view).
    pub active_prefill_tokens: Option<u64>,
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
    pub worker_id: WorkerId,
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
    pub worker: WorkerWithDpRank,
    pub data: ActiveSequenceEventData,
    pub router_id: Uuid,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ActiveSequenceEventData {
    AddRequest {
        token_sequence: Option<Vec<SequenceHash>>,
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
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KvCacheEvent {
    /// The unique identifier of the event.
    pub event_id: u64,
    /// The data associated with the event.
    pub data: KvCacheEventData,
    /// The data parallel rank of the worker emitting this event (0 if DP not enabled).
    #[serde(default)]
    pub dp_rank: DpRank,
}

/// Represents the data associated with a cache event.
///
/// Data is either stored or removed.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheEventData {
    Stored(KvCacheStoreData),
    Removed(KvCacheRemoveData),
    Cleared,
}

/// Represents the data associated with a stored cache event.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KvCacheStoreData {
    /// The optional hash of the parent block.
    pub parent_hash: Option<ExternalSequenceBlockHash>,
    /// A list of stored blocked data.
    pub blocks: Vec<KvCacheStoredBlockData>,
}

/// Multimodal object information within a block.
/// Offsets are relative to the block (0 to block_size-1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockMmObjectInfo {
    /// Hash identifying this multimodal object
    pub mm_hash: u64,
    /// Token offset ranges where this MM object's placeholders appear within THIS block
    /// Each tuple is (start_offset, end_offset) relative to block start
    pub offsets: Vec<(usize, usize)>,
}

/// Extra metadata for a block containing multimodal objects
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlockExtraInfo {
    /// All multimodal objects referenced in this block
    pub mm_objects: Vec<BlockMmObjectInfo>,
}

/// Request-level multimodal object information.
/// Offsets are relative to the entire request token sequence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestMmObjectInfo {
    /// Hash identifying this multimodal object
    pub mm_hash: u64,
    /// Token offset ranges where this MM object's placeholders appear in the ENTIRE request
    /// Each tuple is (start_offset, end_offset) relative to request start
    pub offsets: Vec<(usize, usize)>,
}

/// Request-level multimodal metadata
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestExtraInfo {
    /// All multimodal objects in this request
    pub mm_objects: Vec<RequestMmObjectInfo>,
}

impl RequestExtraInfo {
    /// Convert request-level MM info to block-level MM info for a sequence of blocks.
    ///
    /// This function splits request-level offsets (relative to the entire request token sequence)
    /// into block-level offsets (relative to each block).
    ///
    /// # Arguments
    /// * `block_size` - The size of each block in tokens
    /// * `total_tokens` - Total number of tokens in the request
    ///
    /// # Returns
    /// A vector of `Option<BlockExtraInfo>` where each element corresponds to a block.
    /// `None` indicates a block with no multimodal objects.
    pub fn to_block_level(
        &self,
        block_size: usize,
        total_tokens: usize,
    ) -> Vec<Option<BlockExtraInfo>> {
        let num_blocks = total_tokens.div_ceil(block_size);
        let mut block_infos: Vec<Option<BlockExtraInfo>> = vec![None; num_blocks];

        for req_mm_obj in &self.mm_objects {
            for (req_start, req_end) in &req_mm_obj.offsets {
                // Find which blocks this offset range spans
                let start_block = req_start / block_size;
                let end_block = (req_end.saturating_sub(1)) / block_size;

                let upper_bound = end_block.min(num_blocks - 1) + 1;
                for (block_idx, block_info_opt) in block_infos
                    .iter_mut()
                    .enumerate()
                    .take(upper_bound)
                    .skip(start_block)
                {
                    let block_start_global = block_idx * block_size;
                    let block_end_global = ((block_idx + 1) * block_size).min(total_tokens);

                    // Calculate the intersection of this MM object's range with this block
                    let local_start = (*req_start).max(block_start_global) - block_start_global;
                    let local_end = (*req_end).min(block_end_global) - block_start_global;

                    if local_start < local_end {
                        let block_info = block_info_opt
                            .get_or_insert_with(|| BlockExtraInfo { mm_objects: vec![] });

                        // Check if we already have this mm_hash in this block
                        if let Some(existing) = block_info
                            .mm_objects
                            .iter_mut()
                            .find(|obj| obj.mm_hash == req_mm_obj.mm_hash)
                        {
                            // Add the offset range to existing object
                            existing.offsets.push((local_start, local_end));
                        } else {
                            // Create new MM object entry for this block
                            block_info.mm_objects.push(BlockMmObjectInfo {
                                mm_hash: req_mm_obj.mm_hash,
                                offsets: vec![(local_start, local_end)],
                            });
                        }
                    }
                }
            }
        }

        block_infos
    }
}

/// Represents data for a stored block.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KvCacheStoredBlockData {
    /// The hash of the block.
    pub block_hash: ExternalSequenceBlockHash,
    /// The hash of the tokens in the block.
    pub tokens_hash: LocalBlockHash,
    /// Extra multimodal metadata for this block
    /// Note: Do NOT use skip_serializing_if with bincode - it breaks deserialization
    /// because bincode is positional and expects all fields to be present.
    #[serde(default)]
    pub mm_extra_info: Option<BlockExtraInfo>,
}

/// Represents the data associated with a removed cache event.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
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

// ------
// Tests
// ------
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
                mm_extra_info: None,
            }],
        });

        let event = KvCacheEvent {
            event_id: 1,
            data: event_data,
            dp_rank: 0,
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
