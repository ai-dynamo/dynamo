// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache Status Tracker
//!
//! Maintains the state of KV cache blocks across different event sources (vLLM and KVBM)
//! and determines when to emit STORE/REMOVE events.
//!
//! - Tracks by EVENT SOURCE (vLLM vs KVBM) instead of storage tier
//! - vLLM source: G1 (GPU) events from vLLM worker
//! - KVBM source: G2/G3 (host pinned/disk) events from KVBM
//! - Deduplication: Uses SequenceHash as the key
//!   - Always computes sequence hash using KVBM's xxHash3 method, regardless of source
//!   - SequenceHash = first block: Hash(tokens), subsequent: Hash([parent_seq_hash, block_hash])
//! - Emit Store: Only when a block is first stored from ANY source
//! - Emit Remove: Only when a block is removed from ALL sources

use std::collections::{HashMap, HashSet};

/// LocalBlockHash type (content hash from tokens only)
type LocalBlockHash = u64;

/// SequenceHash type (position-aware hash, includes parent context)
type SequenceHash = u64;

/// Seed for xxHash3 computation (must match the indexer's seed)
const XXH3_SEED: u64 = 1337;

/// Compute a LocalBlockHash from token IDs (content only)
fn compute_local_block_hash(token_ids: &[u32]) -> LocalBlockHash {
    let bytes: Vec<u8> = token_ids
        .iter()
        .flat_map(|&num| num.to_le_bytes())
        .collect();
    xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED)
}

/// Compute a SequenceHash from parent sequence hash and current block hash
/// This mirrors the indexer's sequence hash computation for consistent tracking
///
/// For the first block (no parent): sequence_hash = block_hash
/// For subsequent blocks: sequence_hash = hash([parent_sequence_hash, current_block_hash])
fn compute_sequence_hash(
    parent_sequence_hash: Option<SequenceHash>,
    current_block_hash: LocalBlockHash,
) -> SequenceHash {
    match parent_sequence_hash {
        None => {
            // First block: sequence hash equals block hash
            current_block_hash
        }
        Some(parent_hash) => {
            // Subsequent block: combine parent sequence hash with current block hash
            let combined = [parent_hash, current_block_hash];
            let bytes: Vec<u8> = combined.iter().flat_map(|&num| num.to_le_bytes()).collect();
            xxhash_rust::xxh3::xxh3_64_with_seed(&bytes, XXH3_SEED)
        }
    }
}

/// Event source for KV cache events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventSource {
    /// Events from vLLM worker (G1/GPU)
    Vllm,
    /// Events from KVBM
    Kvbm,
}

impl std::str::FromStr for EventSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vllm" | "VLLM" | "GPU" => Ok(EventSource::Vllm),
            "kvbm" | "KVBM" => Ok(EventSource::Kvbm),
            _ => Err(format!("Unknown event source: {}", s)),
        }
    }
}

impl EventSource {
    /// Convert to string representation
    pub fn to_str(&self) -> &'static str {
        match self {
            EventSource::Vllm => "vllm",
            EventSource::Kvbm => "kvbm",
        }
    }
}

/// Storage tier information (for metadata/debugging only)
///
/// Note: This does NOT determine the event source!
/// The event source (vLLM vs KVBM) is determined by WHERE the event originates:
/// - Events from vLLM's ZMQ subscriber → EventSource::Vllm
/// - Events from KVBM's DynamoEventManager → EventSource::Kvbm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    Device,     // GPU / G1
    HostPinned, // CPU / G2
    Disk,       // Disk / G3
}

impl StorageTier {
    /// Parse from vLLM's medium string (e.g., "GPU", "CPU_TIER1", "CPU_TIER2")
    pub fn from_vllm_medium(s: &str) -> Option<Self> {
        match s {
            "GPU" => Some(StorageTier::Device),
            "CPU_TIER1" => Some(StorageTier::HostPinned),
            "CPU_TIER2" => Some(StorageTier::Disk),
            _ => None,
        }
    }

    /// Convert to vLLM's medium string
    pub fn to_vllm_medium(&self) -> &'static str {
        match self {
            StorageTier::Device => "GPU",
            StorageTier::HostPinned => "CPU_TIER1",
            StorageTier::Disk => "CPU_TIER2",
        }
    }

    /// Convert to string representation
    pub fn to_str(&self) -> &'static str {
        match self {
            StorageTier::Device => "device",
            StorageTier::HostPinned => "host_pinned",
            StorageTier::Disk => "disk",
        }
    }
}

/// Legacy type alias for backward compatibility
#[deprecated(note = "Use StorageTier instead")]
pub type StorageMedium = StorageTier;

/// Minimal metadata for tracking which event sources have a block
/// All other metadata (tokens, parent, etc.) is stored in the ConsolidatedEvent when queued
#[derive(Debug, Clone)]
pub struct BlockMetadata {
    /// Event sources where this block exists (vLLM and/or KVBM)
    pub sources: HashSet<EventSource>,
    /// The first external block hash seen for this token sequence (for output events)
    /// Different sources may have different external hashes, but they all represent the same token content
    pub first_block_hash: String,
}

impl BlockMetadata {
    pub fn new(source: EventSource, block_hash: String) -> Self {
        let mut sources = HashSet::new();
        sources.insert(source);

        Self {
            sources,
            first_block_hash: block_hash,
        }
    }

    /// Check if this block exists in any source
    pub fn exists_in_any_source(&self) -> bool {
        !self.sources.is_empty()
    }

    /// Add a source to this block
    /// Returns true if this is a new source (wasn't already present)
    pub fn add_source(&mut self, source: EventSource) -> bool {
        self.sources.insert(source)
    }

    /// Remove a source from this block
    /// Returns true if the source was present and removed
    pub fn remove_source(&mut self, source: EventSource) -> bool {
        self.sources.remove(&source)
    }
}

/// Event to be published to the router
#[derive(Debug, Clone)]
pub enum ConsolidatedEvent {
    /// Block stored (first time across all sources)
    Store {
        block_hash: String,
        parent_hash: Option<String>,
        token_ids: Vec<u32>,
        block_size: usize,
        lora_id: Option<i32>,
        source: String, // The source where it was first stored (vllm or kvbm)
    },
    /// Block removed (removed from all sources)
    Remove {
        block_hash: String,
        source: String, // The source where it was last removed
    },
    /// All blocks cleared
    ClearAll,
}

/// Cache Status Tracker
///
/// Deduplication logic:
/// - Uses SequenceHash (computed from tokens + parent) as the key for deduplication
/// - SequenceHash is position-aware: same tokens at different positions = different keys
/// - Always uses KVBM's xxHash3 hashing function, regardless of source
/// - This allows vLLM and KVBM blocks at the same position to be deduplicated
/// - Emit Store: Only when a block is first stored from ANY source
/// - Emit Remove: Only when a block is removed from ALL sources
#[derive(Debug)]
pub struct CacheStatusTracker {
    /// Map of SequenceHash -> BlockMetadata (tracking which sources have this block)
    /// The key is position-aware: includes parent context
    blocks: HashMap<SequenceHash, BlockMetadata>,

    /// Reverse mapping: external_block_hash -> SequenceHash (that we computed)
    /// Needed because remove events only provide external hash, not token IDs
    /// Maps each source's external hash to our computed sequence hash
    hash_mapping: HashMap<String, SequenceHash>,

    /// Queue of events to be published
    event_queue: Vec<ConsolidatedEvent>,
}

impl Default for CacheStatusTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheStatusTracker {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            hash_mapping: HashMap::new(),
            event_queue: Vec::new(),
        }
    }

    /// Handle a STORE event
    ///
    /// Returns true if a consolidated STORE event should be published.
    /// Only publishes when a block is stored for the FIRST TIME from ANY source.
    ///
    /// # Arguments
    /// * `block_hash` - The external block hash (from vLLM or KVBM)
    /// * `source` - The event source (vLLM or KVBM) that stored this block
    /// * `token_ids` - The token IDs in this block (for content-based deduplication)
    /// * `tier` - Optional storage tier information (for metadata/debugging)
    #[allow(clippy::too_many_arguments)]
    pub fn handle_store(
        &mut self,
        block_hash: String,
        source: EventSource,
        token_ids: Vec<u32>,
        parent_hash: Option<String>,
        block_size: usize,
        lora_id: Option<i32>,
        tier: Option<StorageTier>,
        data_parallel_rank: Option<i32>,
    ) -> bool {
        // Compute LocalBlockHash from token IDs (content only)
        let local_block_hash = compute_local_block_hash(&token_ids);

        // Resolve parent sequence hash from parent's external hash (if provided)
        let parent_sequence_hash = parent_hash
            .as_ref()
            .and_then(|ph| self.hash_mapping.get(ph).copied());

        // Compute SequenceHash using KVBM's hashing method (position-aware)
        // This ensures consistent deduplication regardless of source
        let sequence_hash = compute_sequence_hash(parent_sequence_hash, local_block_hash);

        tracing::debug!(
            "Computing sequence_hash for block: local_block_hash={}, parent_seq_hash={:?}, sequence_hash={}",
            local_block_hash,
            parent_sequence_hash,
            sequence_hash
        );

        if let Some(metadata) = self.blocks.get_mut(&sequence_hash) {
            // Block already exists from another source (deduplication!), just add the new source
            let is_new_source = metadata.add_source(source);

            // Add this external hash to the mapping so remove events from this source can find the block
            // Multiple external hashes (from different sources) can map to the same SequenceHash
            self.hash_mapping.insert(block_hash.clone(), sequence_hash);

            if is_new_source {
                tracing::debug!(
                    "DEDUP: Block {} (seq_hash={}) added to source {:?} (already exists in {} source(s), {} tokens, external_hash={})\n  Token IDs: {:?}",
                    &metadata.first_block_hash[..16.min(metadata.first_block_hash.len())],
                    sequence_hash,
                    source,
                    metadata.sources.len(),
                    token_ids.len(),
                    &block_hash[..16.min(block_hash.len())],
                    &token_ids
                );
            } else {
                tracing::debug!(
                    "Block {} (seq_hash={}) already in source {:?}, external_hash={}\n  Token IDs: {:?}",
                    &metadata.first_block_hash[..16.min(metadata.first_block_hash.len())],
                    sequence_hash,
                    source,
                    &block_hash[..16.min(block_hash.len())],
                    &token_ids
                );
            }
            // Don't publish a new STORE event (block already exists)
            false
        } else {
            // First time seeing this block from any source - create metadata and queue STORE event
            let metadata = BlockMetadata::new(source, block_hash.clone());

            tracing::debug!(
                "New block {} (seq_hash={}) stored in source {:?} (tier={:?}): {} tokens, block_size={}, parent={}, lora={:?}, dp_rank={:?}\n  Token IDs: {:?}",
                &block_hash[..16.min(block_hash.len())],
                sequence_hash,
                source,
                tier,
                token_ids.len(),
                block_size,
                parent_hash
                    .as_ref()
                    .map(|p| &p[..16.min(p.len())])
                    .unwrap_or("none"),
                lora_id,
                data_parallel_rank,
                &token_ids
            );

            self.blocks.insert(sequence_hash, metadata);

            // Add to hash mapping so remove events can find the block by external hash
            self.hash_mapping.insert(block_hash.clone(), sequence_hash);

            // Queue a STORE event with full metadata
            self.event_queue.push(ConsolidatedEvent::Store {
                block_hash: block_hash.clone(),
                parent_hash,
                token_ids,
                block_size,
                lora_id,
                source: source.to_str().to_string(),
            });

            tracing::debug!(
                "Block {} (seq_hash={}) stored in first source {:?}, will publish STORE event (total tracked: {}, hash_mapping: {})",
                block_hash,
                sequence_hash,
                source,
                self.blocks.len(),
                self.hash_mapping.len()
            );
            true
        }
    }

    /// Handle a REMOVE event
    ///
    /// Returns true if a consolidated REMOVE event should be published.
    /// Only publishes when a block is removed from ALL sources.
    ///
    /// # Arguments
    /// * `block_hash` - The external block hash to remove
    /// * `source` - The event source (vLLM or KVBM) that removed this block
    pub fn handle_remove(&mut self, block_hash: &str, source: EventSource) -> bool {
        // Look up the SequenceHash from the external block hash
        let sequence_hash = match self.hash_mapping.get(block_hash) {
            Some(&hash) => hash,
            None => {
                tracing::warn!(
                    "Attempted to remove unknown block {} from source {:?} (not in hash_mapping)",
                    block_hash,
                    source
                );
                return false;
            }
        };

        if let Some(metadata) = self.blocks.get_mut(&sequence_hash) {
            // Remove the source
            let was_removed = metadata.remove_source(source);
            if !was_removed {
                tracing::warn!(
                    "Attempted to remove source {:?} from block {} but it wasn't present",
                    source,
                    block_hash
                );
                return false;
            }

            // Remove this external hash immediately when the source removes it
            // This keeps hash_mapping clean
            // Each external hash belongs to exactly one source, so when that source
            // removes the block, we can safely remove the hash_mapping entry
            self.hash_mapping.remove(block_hash);

            tracing::debug!(
                "Removed hash_mapping entry for {} (hash_mapping size: {})",
                block_hash,
                self.hash_mapping.len()
            );

            // Check if this was the last source
            if !metadata.exists_in_any_source() {
                // Block is gone from all sources - remove from tracker and publish REMOVE
                let first_block_hash = metadata.first_block_hash.clone();
                self.blocks.remove(&sequence_hash);

                // Double-check: clean up any stray hash mappings (should be empty by now)
                // This is a safety check
                let stray_count_before = self.hash_mapping.len();
                self.hash_mapping
                    .retain(|_ext_hash, &mut seq_hash| seq_hash != sequence_hash);
                let stray_count = stray_count_before - self.hash_mapping.len();

                if stray_count > 0 {
                    tracing::warn!(
                        "Found {} stray hash_mapping entries for seq_hash={} after all sources removed - cleaned up (hash_mapping size now: {})",
                        stray_count,
                        sequence_hash,
                        self.hash_mapping.len()
                    );
                }

                self.event_queue.push(ConsolidatedEvent::Remove {
                    block_hash: first_block_hash.clone(),
                    source: source.to_str().to_string(),
                });

                tracing::debug!(
                    "Block {} (seq_hash={}) removed from last source {:?}, will publish REMOVE event (total tracked: {}, hash_mapping: {})",
                    first_block_hash,
                    sequence_hash,
                    source,
                    self.blocks.len(),
                    self.hash_mapping.len()
                );
                true
            } else {
                // Block still exists in other sources
                tracing::debug!(
                    "Block {} (seq_hash={}) removed from source {:?}, still in {} source(s): {:?} (hash_mapping: {})",
                    &metadata.first_block_hash[..16.min(metadata.first_block_hash.len())],
                    sequence_hash,
                    source,
                    metadata.sources.len(),
                    metadata.sources,
                    self.hash_mapping.len()
                );
                false
            }
        } else {
            tracing::warn!(
                "Attempted to remove block {} from source {:?} but block not tracked",
                &block_hash[..16.min(block_hash.len())],
                source
            );
            false
        }
    }

    /// Handle a CLEAR_ALL event
    pub fn handle_clear_all(&mut self) {
        let num_blocks = self.blocks.len();
        tracing::debug!("Clearing all {} blocks from tracker", num_blocks);
        self.blocks.clear();
        self.hash_mapping.clear();
        self.event_queue.push(ConsolidatedEvent::ClearAll);
    }

    /// Drain all pending events to be published
    pub fn drain_events(&mut self) -> Vec<ConsolidatedEvent> {
        let events = std::mem::take(&mut self.event_queue);
        if !events.is_empty() {
            tracing::debug!(
                "Draining {} pending kv event(s) for publishing",
                events.len()
            );
        }
        events
    }

    /// Get the number of tracked blocks
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Get sources for a specific block by external block hash
    pub fn get_block_sources(&self, external_block_hash: &str) -> Option<&HashSet<EventSource>> {
        // Look up the local hash from external hash, then get sources
        let local_hash = self.hash_mapping.get(external_block_hash)?;
        self.blocks.get(local_hash).map(|m| &m.sources)
    }

    /// Legacy method for backwards compatibility
    #[deprecated(note = "Use get_block_sources instead")]
    pub fn get_block_tiers(&self, block_hash: &str) -> Option<&HashSet<EventSource>> {
        self.get_block_sources(block_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_first_store_publishes() {
        let mut tracker = CacheStatusTracker::new();

        let should_publish = tracker.handle_store(
            "hash1".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3],
            None,
            3,
            None,
            Some(StorageTier::Device),
            None, // data_parallel_rank
        );

        assert!(should_publish);
        assert_eq!(tracker.num_blocks(), 1);
        assert_eq!(tracker.drain_events().len(), 1);
    }

    #[test]
    fn test_duplicate_store_no_publish() {
        let mut tracker = CacheStatusTracker::new();

        tracker.handle_store(
            "hash1".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3],
            None,
            3,
            None,
            Some(StorageTier::Device),
            None,
        );
        tracker.drain_events(); // Clear first event

        let should_publish = tracker.handle_store(
            "hash1".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3],
            None,
            3,
            None,
            Some(StorageTier::Device),
            None,
        );

        assert!(!should_publish);
        assert_eq!(tracker.drain_events().len(), 0);
    }

    #[test]
    fn test_multi_source_store() {
        let mut tracker = CacheStatusTracker::new();

        // First store from vLLM
        tracker.handle_store(
            "vllm_hash1".to_string(), // vLLM's external hash
            EventSource::Vllm,
            vec![1, 2, 3],
            None,
            3,
            None,
            Some(StorageTier::Device),
            None,
        );
        tracker.drain_events();

        // Second store from KVBM - should not publish (same tokens, different external hash)
        let should_publish = tracker.handle_store(
            "kvbm_hash1".to_string(), // KVBM's external hash (different from vLLM)
            EventSource::Kvbm,
            vec![1, 2, 3], // Same tokens!
            None,
            3,
            None,
            Some(StorageTier::HostPinned),
            None,
        );

        assert!(!should_publish);
        #[allow(deprecated)]
        let sources = tracker.get_block_tiers("vllm_hash1").unwrap();
        assert_eq!(sources.len(), 2); // vllm and kvbm
    }

    #[test]
    fn test_remove_from_single_source_publishes() {
        let mut tracker = CacheStatusTracker::new();

        tracker.handle_store(
            "hash1".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3],
            None,
            3,
            None,
            Some(StorageTier::Device),
            None,
        );
        tracker.drain_events();

        let should_publish = tracker.handle_remove("hash1", EventSource::Vllm);

        assert!(should_publish);
        assert_eq!(tracker.num_blocks(), 0);
        let events = tracker.drain_events();
        assert_eq!(events.len(), 1);
        matches!(events[0], ConsolidatedEvent::Remove { .. });
    }

    #[test]
    fn test_remove_from_multi_source_no_publish() {
        let mut tracker = CacheStatusTracker::new();

        // Store from vLLM - first STORE event published
        tracker.handle_store(
            "vllm_hash1".to_string(), // vLLM's external hash
            EventSource::Vllm,
            vec![1, 2, 3],
            None,
            3,
            None,
            Some(StorageTier::Device),
            None,
        );
        // Store from KVBM - no event, block already exists (same tokens, different external hash)
        tracker.handle_store(
            "kvbm_hash1".to_string(), // KVBM's external hash (different from vLLM)
            EventSource::Kvbm,
            vec![1, 2, 3], // Same tokens!
            None,
            3,
            None,
            Some(StorageTier::HostPinned),
            None,
        );
        tracker.drain_events();

        // Remove from vLLM - should not publish (still in KVBM)
        let should_publish = tracker.handle_remove("vllm_hash1", EventSource::Vllm);

        assert!(!should_publish);
        assert_eq!(tracker.num_blocks(), 1);
        assert_eq!(tracker.drain_events().len(), 0);

        // Remove from KVBM (last source) - should publish REMOVE event
        let should_publish = tracker.handle_remove("kvbm_hash1", EventSource::Kvbm);

        assert!(should_publish);
        assert_eq!(tracker.num_blocks(), 0);
    }

    #[test]
    fn test_sequence_hash_first_block() {
        let mut tracker = CacheStatusTracker::new();

        // First block (no parent)
        let should_publish = tracker.handle_store(
            "block1".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3, 4],
            None, // No parent
            4,
            None,
            Some(StorageTier::Device),
            None,
        );

        assert!(should_publish);
        assert_eq!(tracker.num_blocks(), 1);

        let events = tracker.drain_events();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_sequence_hash_with_parent() {
        let mut tracker = CacheStatusTracker::new();

        // First block
        tracker.handle_store(
            "block1".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3, 4],
            None,
            4,
            None,
            Some(StorageTier::Device),
            None,
        );
        tracker.drain_events();

        // Second block with parent
        let should_publish = tracker.handle_store(
            "block2".to_string(),
            EventSource::Vllm,
            vec![5, 6, 7, 8],
            Some("block1".to_string()), // Has parent
            4,
            None,
            Some(StorageTier::Device),
            None,
        );

        assert!(should_publish);
        assert_eq!(tracker.num_blocks(), 2);
    }

    #[test]
    fn test_same_tokens_different_position_different_blocks() {
        let mut tracker = CacheStatusTracker::new();

        // First occurrence: tokens [1,2,3,4] at position 0 (no parent)
        tracker.handle_store(
            "block1".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3, 4],
            None,
            4,
            None,
            Some(StorageTier::Device),
            None,
        );
        tracker.drain_events();

        // Second occurrence: SAME tokens [1,2,3,4] but at position 1 (with parent)
        // This should be treated as a DIFFERENT block due to sequence hash
        let should_publish = tracker.handle_store(
            "block2".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3, 4],           // Same tokens!
            Some("block1".to_string()), // But different parent
            4,
            None,
            Some(StorageTier::Device),
            None,
        );

        // Should publish because sequence hash differs (different position)
        assert!(should_publish);
        assert_eq!(tracker.num_blocks(), 2);
    }

    #[test]
    fn test_clear_all() {
        let mut tracker = CacheStatusTracker::new();

        // Add multiple blocks
        tracker.handle_store(
            "block1".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3, 4],
            None,
            4,
            None,
            Some(StorageTier::Device),
            None,
        );

        tracker.handle_store(
            "block2".to_string(),
            EventSource::Kvbm,
            vec![5, 6, 7, 8],
            None,
            4,
            None,
            Some(StorageTier::HostPinned),
            None,
        );

        assert_eq!(tracker.num_blocks(), 2);

        // Clear all
        tracker.handle_clear_all();

        assert_eq!(tracker.num_blocks(), 0);

        // Verify hash_mapping is also cleared
        let should_publish = tracker.handle_remove("block1", EventSource::Vllm);
        assert!(!should_publish); // Should fail because block is gone
    }

    #[test]
    fn test_deduplication_across_sources_with_parent() {
        let mut tracker = CacheStatusTracker::new();

        // vLLM stores block 1 (parent)
        tracker.handle_store(
            "vllm_parent".to_string(),
            EventSource::Vllm,
            vec![1, 2, 3, 4],
            None,
            4,
            None,
            Some(StorageTier::Device),
            None,
        );
        tracker.drain_events();

        // vLLM stores block 2 (child of block 1)
        tracker.handle_store(
            "vllm_child".to_string(),
            EventSource::Vllm,
            vec![5, 6, 7, 8],
            Some("vllm_parent".to_string()),
            4,
            None,
            Some(StorageTier::Device),
            None,
        );
        tracker.drain_events();

        // KVBM stores the SAME child block (same tokens, same parent)
        // but with a different external hash
        let should_publish = tracker.handle_store(
            "kvbm_child".to_string(), // Different external hash
            EventSource::Kvbm,
            vec![5, 6, 7, 8],                // Same tokens
            Some("vllm_parent".to_string()), // Same parent
            4,
            None,
            Some(StorageTier::HostPinned),
            None,
        );

        // Should NOT publish (deduplication)
        assert!(!should_publish);
        // Should still have 2 blocks (parent + child)
        assert_eq!(tracker.num_blocks(), 2);
    }

    #[test]
    fn test_remove_non_existent_block() {
        let mut tracker = CacheStatusTracker::new();

        let should_publish = tracker.handle_remove("non_existent", EventSource::Vllm);

        assert!(!should_publish);
        assert_eq!(tracker.num_blocks(), 0);
    }

    #[test]
    fn test_compute_local_block_hash_deterministic() {
        let tokens1 = vec![1, 2, 3, 4];
        let tokens2 = vec![1, 2, 3, 4];
        let tokens3 = vec![1, 2, 3, 5]; // Different

        let hash1 = compute_local_block_hash(&tokens1);
        let hash2 = compute_local_block_hash(&tokens2);
        let hash3 = compute_local_block_hash(&tokens3);

        // Same tokens should produce same hash
        assert_eq!(hash1, hash2);
        // Different tokens should produce different hash
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_compute_sequence_hash_deterministic() {
        let block_hash1 = compute_local_block_hash(&[1, 2, 3, 4]);
        let block_hash2 = compute_local_block_hash(&[5, 6, 7, 8]);

        // First block: sequence hash = block hash
        let seq_hash1 = compute_sequence_hash(None, block_hash1);
        assert_eq!(seq_hash1, block_hash1);

        // Second block with parent
        let seq_hash2_v1 = compute_sequence_hash(Some(seq_hash1), block_hash2);
        let seq_hash2_v2 = compute_sequence_hash(Some(seq_hash1), block_hash2);

        // Same parent + block should produce same sequence hash
        assert_eq!(seq_hash2_v1, seq_hash2_v2);

        // Same block but different parent should produce different sequence hash
        let different_parent = compute_local_block_hash(&[9, 10, 11, 12]);
        let seq_hash2_different = compute_sequence_hash(Some(different_parent), block_hash2);
        assert_ne!(seq_hash2_v1, seq_hash2_different);
    }
}
