// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::{SequenceHash, Token};
use serde::{Deserialize, Serialize};
use xxhash_rust::xxh3;

/// Seed for XXH3 hashing, consistent with indexer.rs
pub const XXH3_SEED: u64 = 1337;

/// Compute hash of data using XXH3 with the standard seed.
pub fn compute_hash(data: &[u8]) -> u64 {
    xxh3::xxh3_64_with_seed(data, XXH3_SEED)
}

/// Compute the hash of a local block.
pub fn compute_block_hash(data: &[u8]) -> LocalBlockHash {
    LocalBlockHash(compute_hash(data))
}

/// Compute the hash for a sequence of tokens, optionally including multimodal metadata.
///
/// When multimodal extra info is provided, the mm_hashes are included in the hash computation
/// to ensure that blocks with identical tokens but different multimodal objects produce
/// different hashes.
pub fn compute_block_hash_for_seq(
    tokens: &[u32],
    kv_block_size: u32,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
) -> Vec<LocalBlockHash> {
    tokens
        .chunks_exact(kv_block_size as usize)
        .enumerate()
        .map(|(block_idx, chunk)| {
            let mut bytes: Vec<u8> = chunk.iter().flat_map(|&num| num.to_le_bytes()).collect();

            // Include MM hashes in the block hash computation if present
            if let Some(mm_infos) = block_mm_infos
                && let Some(Some(block_mm_info)) = mm_infos.get(block_idx)
            {
                let mut mm_hashes: Vec<u64> = block_mm_info
                    .mm_objects
                    .iter()
                    .map(|obj| obj.mm_hash)
                    .collect();
                mm_hashes.sort_unstable();

                for mm_hash in mm_hashes {
                    bytes.extend_from_slice(&mm_hash.to_le_bytes());
                }
            }

            compute_block_hash(&bytes)
        })
        .collect()
}

/// Compute rolling sequence hashes for a vector of block hashes.
///
/// - The first block's sequence hash equals its block hash
/// - Subsequent blocks' sequence hash = hash([parent_sequence_hash, current_block_hash], seed)
pub fn compute_seq_hash_for_block(block_hashes: &[LocalBlockHash]) -> Vec<SequenceHash> {
    if block_hashes.is_empty() {
        return Vec::new();
    }

    let mut sequence_hashes = Vec::with_capacity(block_hashes.len());
    sequence_hashes.push(block_hashes[0].0);

    for i in 1..block_hashes.len() {
        let parent_seq_hash = sequence_hashes[i - 1];
        let current_block_hash = block_hashes[i].0;

        let combined = [parent_seq_hash, current_block_hash];
        let bytes: Vec<u8> = combined.iter().flat_map(|&num| num.to_le_bytes()).collect();
        let seq_hash = compute_hash(&bytes);
        sequence_hashes.push(seq_hash);
    }

    sequence_hashes
}

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
    pub router_id: u64,
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
    pub router_id: u64,
    #[serde(default)]
    pub lora_name: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ActiveSequenceEventData {
    AddRequest {
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlap: u32,
        expected_output_tokens: Option<u32>,
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
// Router Event Types
// ------

/// Errors that can occur during KV Cache Event processing.
#[derive(Debug, thiserror::Error)]
pub enum KvCacheEventError {
    #[error("Failed to find parent block")]
    ParentBlockNotFound,

    #[error("Failed to find block")]
    BlockNotFound,

    #[error("Invalid block sequence")]
    InvalidBlockSequence,
}

/// A [`KvCacheEvent`] on a specific LLM worker denoted by [`WorkerId`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterEvent {
    /// The ID of the worker emitting the event.
    pub worker_id: WorkerId,
    /// The cache event associated with the worker.
    pub event: KvCacheEvent,
}

impl RouterEvent {
    /// Create a new `RouterEvent`.
    ///
    /// ### Arguments
    ///
    /// * `worker_id` - The ID of the worker emitting the event.
    /// * `event` - The cache event.
    ///
    /// ### Returns
    ///
    /// A new `RouterEvent`.
    pub fn new(worker_id: WorkerId, event: KvCacheEvent) -> Self {
        Self { worker_id, event }
    }
}

/// Scores representing the overlap of workers (with their dp_rank).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapScores {
    /// Map of worker (with dp_rank) to score.
    pub scores: std::collections::HashMap<WorkerWithDpRank, u32>,
    /// List of frequencies that the blocks have been accessed. Entries with value 0 are omitted.
    pub frequencies: Vec<usize>,
    /// Map of worker to their tree size (number of blocks in the tree for that worker).
    pub tree_sizes: std::collections::HashMap<WorkerWithDpRank, usize>,
}

impl Default for OverlapScores {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlapScores {
    /// Create a new `OverlapScores`.
    ///
    /// ### Returns
    ///
    /// A new `OverlapScores`.
    pub fn new() -> Self {
        Self {
            scores: std::collections::HashMap::new(),
            frequencies: Vec::with_capacity(32),
            tree_sizes: std::collections::HashMap::new(),
        }
    }

    /// Update the scores with a set of workers.
    ///
    /// ### Arguments
    ///
    /// * `workers` - An iterator over `WorkerWithDpRank` references.
    pub fn update_scores<'a, I>(&mut self, workers: I)
    where
        I: IntoIterator<Item = &'a WorkerWithDpRank>,
    {
        for worker in workers {
            let score = self.scores.entry(*worker).or_insert(0);
            *score += 1;
        }
    }

    /// Add an entry in the frequency list.
    pub fn add_frequency(&mut self, frequency: usize) {
        if frequency != 0 {
            self.frequencies
                .last()
                .inspect(|elem| debug_assert!(**elem >= frequency));
            self.frequencies.push(frequency);
        }
    }
}

// ------
// TokensWithHashes
// ------

/// A container for tokens with lazily computed block and sequence hashes.
///
/// This struct avoids redundant hash computations by caching results:
/// - `get_or_compute_block_hashes()` computes block hashes if not cached
/// - `get_or_compute_seq_hashes()` computes seq hashes if not cached,
///   and will also compute block hashes first if needed (since seq hashes depend on them)
#[derive(Debug, Clone)]
pub struct TokensWithHashes {
    tokens: Vec<u32>,
    block_size: u32,
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    block_hashes: Option<Vec<LocalBlockHash>>,
    seq_hashes: Option<Vec<SequenceHash>>,
}

impl TokensWithHashes {
    /// Creates a new TokensWithHashes from tokens and block size.
    pub fn new(tokens: Vec<u32>, block_size: u32) -> Self {
        Self {
            tokens,
            block_size,
            block_mm_infos: None,
            block_hashes: None,
            seq_hashes: None,
        }
    }

    /// Adds multimodal extra info for blocks.
    pub fn with_mm_infos(mut self, infos: Vec<Option<BlockExtraInfo>>) -> Self {
        self.block_mm_infos = Some(infos);
        self
    }

    /// Returns a reference to the tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Returns the number of tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the block size.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Returns the multimodal extra info, if set.
    pub fn block_mm_infos(&self) -> Option<&[Option<BlockExtraInfo>]> {
        self.block_mm_infos.as_deref()
    }

    /// Returns block hashes, computing them if not already cached.
    pub fn get_or_compute_block_hashes(&mut self) -> &[LocalBlockHash] {
        if self.block_hashes.is_none() {
            self.block_hashes = Some(compute_block_hash_for_seq(
                &self.tokens,
                self.block_size,
                self.block_mm_infos.as_deref(),
            ));
        }
        self.block_hashes.as_ref().unwrap()
    }

    /// Returns sequence hashes, computing them if not already cached.
    /// This will also compute block hashes if they haven't been computed yet,
    /// since sequence hashes depend on block hashes.
    pub fn get_or_compute_seq_hashes(&mut self) -> &[SequenceHash] {
        if self.seq_hashes.is_none() {
            // Ensure block hashes are computed first
            let block_hashes = self.get_or_compute_block_hashes();
            self.seq_hashes = Some(compute_seq_hash_for_block(block_hashes));
        }
        self.seq_hashes.as_ref().unwrap()
    }

    /// Returns cached block hashes without computing. Returns None if not yet computed.
    pub fn block_hashes(&self) -> Option<&[LocalBlockHash]> {
        self.block_hashes.as_deref()
    }

    /// Returns cached seq hashes without computing. Returns None if not yet computed.
    pub fn seq_hashes(&self) -> Option<&[SequenceHash]> {
        self.seq_hashes.as_deref()
    }
}

// ------
// Tests
// ------
#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json;

    #[test]
    fn test_router_event_new() {
        let worker_id = 0;
        let kv_cache_event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(0),
                    mm_extra_info: None,
                    tokens_hash: LocalBlockHash(13226331709069118873),
                }],
            }),
            dp_rank: 0,
        };
        let router_event = RouterEvent::new(worker_id, kv_cache_event);

        assert_eq!(router_event.worker_id, worker_id);
        assert_eq!(router_event.event.event_id, 1);
        if let KvCacheEventData::Stored(store_op) = &router_event.event.data {
            assert_eq!(store_op.blocks.len(), 1);
            assert_eq!(
                store_op.blocks[0].tokens_hash,
                compute_block_hash(b"test data")
            );
            assert_eq!(store_op.blocks[0].block_hash, ExternalSequenceBlockHash(0));
        } else {
            panic!("Expected KvCacheEventData::Stored");
        }
    }

    #[test]
    fn test_overlap_scores_default() {
        let overlap_scores: OverlapScores = Default::default();
        assert!(overlap_scores.scores.is_empty());
    }

    #[rstest]
    #[case(11)]
    #[case(32)]
    #[case(64)]
    fn test_compute_block_hash_for_seq(#[case] kv_block_size: u32) {
        // create a sequence of kv_block_size elements
        let sequence = (0..kv_block_size).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size, None);
        assert_eq!(hashes.len(), 1);

        // create a sequence of kv_block_size + 1 elements
        let sequence = (0..(kv_block_size + 1)).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size, None);
        assert_eq!(hashes.len(), 1);

        // create a sequence of 2 * kv_block_size + 1 elements
        let sequence = (0..(2 * kv_block_size + 1)).collect::<Vec<u32>>();
        let hashes = compute_block_hash_for_seq(&sequence, kv_block_size, None);
        assert_eq!(hashes.len(), 2);
    }

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
