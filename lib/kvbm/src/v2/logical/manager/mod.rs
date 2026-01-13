// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block Manager v2

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::{BlockId, logical::pools::backends::LineageBackend, utils::tinylfu::TinyLFUTracker};

use super::{
    blocks::{
        Block, BlockMetadata, BlockRegistry, CompleteBlock, ImmutableBlock, MutableBlock,
        RegisteredBlock, state::Reset,
    },
    pools::{
        ActivePool, BlockDuplicationPolicy, InactivePool, InactivePoolBackend, ResetPool,
        ReusePolicy, SequenceHash,
        backends::{HashMapBackend, LruBackend, MultiLruBackend},
    },
};

/// Capacity settings for TinyLFU frequency tracker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyTrackingCapacity {
    /// Small capacity: 2^18 (262,144) entries
    Small,
    /// Medium capacity: 2^21 (2,097,152) entries - default
    Medium,
    /// Large capacity: 2^24 (16,777,216) entries
    Large,
}

impl FrequencyTrackingCapacity {
    /// Get the size in number of entries
    pub fn size(&self) -> usize {
        match self {
            Self::Small => 1 << 18,
            Self::Medium => 1 << 21,
            Self::Large => 1 << 24,
        }
    }

    /// Create a new TinyLFUTracker with this capacity
    pub fn create_tracker(&self) -> Arc<TinyLFUTracker<u128>> {
        Arc::new(TinyLFUTracker::new(self.size()))
    }
}

impl Default for FrequencyTrackingCapacity {
    fn default() -> Self {
        Self::Medium
    }
}

/// Type alias for upgrade function used in BlockManager
type UpgradeFn<T> = Arc<dyn Fn(SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> + Send + Sync>;

/// Configuration for different inactive pool backends
pub enum InactiveBackendConfig {
    /// HashMap with configurable reuse policy
    HashMap { reuse_policy: Box<dyn ReusePolicy> },
    /// Simple LRU - capacity automatically set to block_count
    Lru,
    /// Multi-level LRU with 4 fixed levels - capacity automatically set to block_count
    MultiLru {
        /// Frequency thresholds: [cold->warm, warm->hot, hot->very_hot]
        /// Default: [3, 8, 15]
        frequency_thresholds: [u8; 3],
    },
    /// Lineage backend
    Lineage,
}

/// Builder for BlockManager configuration
pub struct BlockManagerConfigBuilder<T: BlockMetadata> {
    /// Number of blocks in the pool
    block_count: Option<usize>,

    /// Size of each block in tokens (must be power of 2, 1-1024)
    /// Default: 16
    block_size: Option<usize>,

    /// Block registry for tracking blocks and frequency
    registry: Option<BlockRegistry>,

    /// Inactive pool backend configuration
    inactive_backend: Option<InactiveBackendConfig>,

    /// Policy for handling duplicate sequence hashes
    duplication_policy: Option<BlockDuplicationPolicy>,

    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<T>,
}

/// Error types for BlockManager builder
#[derive(Debug, thiserror::Error)]
pub enum BlockManagerBuilderError {
    #[error("Block count must be greater than 0")]
    InvalidBlockCount,
    #[error("Block size mismatch: expected {expected} tokens, got {actual}")]
    BlockSizeMismatch { expected: usize, actual: usize },
    #[error("Invalid backend configuration: {0}")]
    InvalidBackend(String),
    #[error("Builder validation failed: {0}")]
    ValidationError(String),
}

/// Error types for BlockManager reset operations
#[derive(Debug, thiserror::Error)]
pub enum BlockManagerResetError {
    #[error("Reset pool count mismatch: expected {expected}, got {actual}")]
    BlockCountMismatch { expected: usize, actual: usize },
}

/// BlockManager v2 with pluggable inactive pool backends
pub struct BlockManager<T: BlockMetadata> {
    reset_pool: ResetPool<T>,
    active_pool: ActivePool<T>,
    inactive_pool: InactivePool<T>,
    block_registry: BlockRegistry,
    duplication_policy: BlockDuplicationPolicy,
    upgrade_fn: UpgradeFn<T>,
    allocate_mutex: Mutex<()>,
    total_blocks: usize,
    block_size: usize,
}

impl<T: BlockMetadata> BlockManager<T> {
    /// Create a new builder for BlockManager
    ///
    /// # Example
    /// ```ignore
    /// let tracker = FrequencyTrackingCapacity::Medium.create_tracker();
    /// let registry = BlockRegistry::with_frequency_tracker(tracker);
    ///
    /// let manager = BlockManager::builder()
    ///     .block_count(1000)
    ///     .registry(registry)
    ///     .with_multi_lru_backend()
    ///     .build()?;
    /// ```
    pub fn builder() -> BlockManagerConfigBuilder<T> {
        BlockManagerConfigBuilder::default()
    }

    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        let _guard = self.allocate_mutex.lock();
        let mut blocks = self.reset_pool.allocate_blocks(count);
        match self.inactive_pool.allocate_blocks(count - blocks.len()) {
            Some(remaining) => {
                blocks.extend(remaining);
                Some(blocks)
            }
            None => None,
        }
    }

    /// Reset the inactive pool by draining all blocks and returning them to the reset pool.
    ///
    /// This method:
    /// 1. Acquires the inactive pool lock and allocates all blocks
    /// 2. Releases the inactive pool lock
    /// 3. Drops the allocated blocks, which returns them to the reset pool via RAII
    /// 4. Verifies the reset pool contains the expected number of blocks
    ///
    /// # Returns
    /// - `Ok(())` if all blocks were successfully returned to the reset pool
    /// - `Err(BlockManagerResetError::BlockCountMismatch)` if the final count doesn't match
    ///   (this can happen under contention when blocks are in active use)
    pub fn reset_inactive_pool(&self) -> Result<(), BlockManagerResetError> {
        // 1. Allocate all blocks from inactive pool (acquires lock internally)
        let blocks = self.inactive_pool.allocate_all_blocks();

        // 2. Drop blocks - RAII returns them to reset pool
        drop(blocks);

        // 3. Verify block count (may fail under contention - that's OK)
        let reset_count = self.reset_pool.len();
        if reset_count != self.total_blocks {
            return Err(BlockManagerResetError::BlockCountMismatch {
                expected: self.total_blocks,
                actual: reset_count,
            });
        }

        Ok(())
    }

    pub fn register_blocks(&self, blocks: Vec<CompleteBlock<T>>) -> Vec<ImmutableBlock<T>> {
        blocks
            .into_iter()
            .map(|block| {
                let handle = self
                    .block_registry
                    .register_sequence_hash(block.sequence_hash());
                let registered_block =
                    handle.register_block(block, self.duplication_policy, &self.inactive_pool);
                ImmutableBlock::new(registered_block, self.upgrade_fn.clone())
            })
            .collect()
    }

    pub(crate) fn register_mutable_block_from_existing<U: BlockMetadata>(
        &self,
        block: MutableBlock<T>,
        existing: &ImmutableBlock<U>,
    ) -> ImmutableBlock<T> {
        let handle = existing.registration_handle();

        assert!(
            handle.is_from_registry(&self.block_registry),
            "Attempted to register block with handle from different registry"
        );

        let registered_block =
            handle.register_mutable_block(block, self.duplication_policy, &self.inactive_pool);

        ImmutableBlock::new(registered_block, self.upgrade_fn.clone())
    }

    /// Register a mutable block with an explicit sequence hash.
    ///
    /// This is used when the block content comes from a remote source (e.g., RDMA pull)
    /// and we know the sequence hash but don't have an existing local block to copy from.
    ///
    /// # Arguments
    /// * `block` - The mutable block to register
    /// * `seq_hash` - The sequence hash for this block (from remote)
    pub(crate) fn register_mutable_block_with_hash(
        &self,
        block: MutableBlock<T>,
        seq_hash: SequenceHash,
    ) -> ImmutableBlock<T> {
        // Register the sequence hash to get a handle
        let handle = self.block_registry.register_sequence_hash(seq_hash);

        // Register the block using the handle
        let registered_block =
            handle.register_mutable_block(block, self.duplication_policy, &self.inactive_pool);

        ImmutableBlock::new(registered_block, self.upgrade_fn.clone())
    }

    /// Match blocks does a linear search through the [SequenceHash] array, stopping on the first miss.
    pub fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
        tracing::debug!(
            num_hashes = seq_hash.len(),
            inactive_pool_len = self.inactive_pool.len(),
            "match_blocks called"
        );

        // First try to match against active blocks
        let mut matched: Vec<ImmutableBlock<T>> = Vec::with_capacity(seq_hash.len());
        matched.extend(
            self.active_pool
                .find_matches(seq_hash, true)
                .into_iter()
                .map(|block| ImmutableBlock::new(block, self.upgrade_fn.clone())),
        );

        let active_matched = matched.len();
        tracing::debug!(active_matched, "Matched from active pool");

        // If we didn't match all hashes, try inactive blocks for the remaining ones
        let remaining_hashes = &seq_hash[matched.len()..];
        if !remaining_hashes.is_empty() {
            let inactive_found: Vec<_> = self.inactive_pool.find_blocks(remaining_hashes, true);
            let inactive_matched = inactive_found.len();
            tracing::debug!(
                remaining_to_check = remaining_hashes.len(),
                inactive_matched,
                "Matched from inactive pool"
            );
            matched.extend(
                inactive_found
                    .into_iter()
                    .map(|block| ImmutableBlock::new(block, self.upgrade_fn.clone())),
            );
        }

        tracing::debug!(total_matched = matched.len(), "match_blocks result");
        tracing::trace!(matched = ?matched, "matched blocks");
        matched
    }

    /// Scan for all blocks matching any of the given hashes.
    /// Unlike `match_blocks`, this does NOT stop on first miss.
    /// Returns a HashMap of found sequence hashes to immutable blocks.
    ///
    /// # Arguments
    /// * `seq_hashes` - Sequence hashes to scan for
    /// * `touch` - Whether to update frequency tracking (for MultiLRU)
    pub fn scan_matches(
        &self,
        seq_hashes: &[SequenceHash],
        touch: bool,
    ) -> HashMap<SequenceHash, ImmutableBlock<T>> {
        let mut result = HashMap::new();

        // 1. Check active pool for all hashes (read-only, no touch needed)
        let active_found = self.active_pool.scan_matches(seq_hashes);
        for (hash, block) in active_found {
            result.insert(hash, ImmutableBlock::new(block, self.upgrade_fn.clone()));
        }

        // 2. Build remaining hashes set
        let remaining: Vec<SequenceHash> = seq_hashes
            .iter()
            .filter(|h| !result.contains_key(h))
            .copied()
            .collect();

        // 3. Scan inactive pool for remaining (acquires blocks, may touch)
        if !remaining.is_empty() {
            let inactive_found = self.inactive_pool.scan_blocks(&remaining, touch);
            for (hash, block) in inactive_found {
                result.insert(hash, ImmutableBlock::new(block, self.upgrade_fn.clone()));
            }
        }

        result
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub fn available_blocks(&self) -> usize {
        self.reset_pool.len() + self.inactive_pool.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn duplication_policy(&self) -> &BlockDuplicationPolicy {
        &self.duplication_policy
    }

    /// Get a reference to the block registry
    #[allow(dead_code)]
    pub(crate) fn block_registry(&self) -> &BlockRegistry {
        &self.block_registry
    }
}

impl<T: BlockMetadata> Default for BlockManagerConfigBuilder<T> {
    fn default() -> Self {
        Self {
            block_count: None,
            block_size: Some(16), // Default to 16 tokens per block
            registry: None,
            inactive_backend: None,
            duplication_policy: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: BlockMetadata> BlockManagerConfigBuilder<T> {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of blocks in the pool
    pub fn block_count(mut self, count: usize) -> Self {
        self.block_count = Some(count);
        self
    }

    /// Set the block size (number of tokens per block)
    ///
    /// # Requirements
    /// - Must be >= 1 and <= 1024
    /// - Must be a power of 2
    ///
    /// # Panics
    /// Panics if the block size doesn't meet requirements
    pub fn block_size(mut self, size: usize) -> Self {
        assert!(
            (1..=1024).contains(&size),
            "block_size must be between 1 and 1024, got {}",
            size
        );
        assert!(
            size.is_power_of_two(),
            "block_size must be a power of 2, got {}",
            size
        );
        self.block_size = Some(size);
        self
    }

    /// Set the duplication policy
    pub fn duplication_policy(mut self, policy: BlockDuplicationPolicy) -> Self {
        self.duplication_policy = Some(policy);
        self
    }

    /// Set the block registry
    pub fn registry(mut self, registry: BlockRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Use simple LRU backend (capacity automatically set to block_count)
    pub fn with_lru_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::Lru);
        self
    }

    /// Use multi-level LRU backend with 4 fixed priority levels
    /// Default thresholds: [3, 8, 15] for transitions between:
    /// - Cold (0-2 hits) -> Warm (3-7 hits) -> Hot (8-14 hits) -> Very Hot (15 hits)
    pub fn with_multi_lru_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::MultiLru {
            frequency_thresholds: [3, 8, 15],
        });
        self
    }

    /// Use multi-level LRU with custom frequency thresholds
    ///
    /// # Requirements
    /// - Thresholds must be in ascending order: cold_to_warm < warm_to_hot < hot_to_very_hot
    /// - hot_to_very_hot must be <= 15 (4-bit counter maximum)
    /// - cold_to_warm must be >= 1 (to distinguish from never-accessed blocks)
    ///
    /// # Arguments
    /// * `cold_to_warm` - Minimum frequency to move from Cold to Warm level
    /// * `warm_to_hot` - Minimum frequency to move from Warm to Hot level
    /// * `hot_to_very_hot` - Minimum frequency to move from Hot to Very Hot level
    ///
    /// # Panics
    /// Panics if thresholds don't meet the requirements above
    pub fn with_multi_lru_backend_custom_thresholds(
        mut self,
        cold_to_warm: u8,
        warm_to_hot: u8,
        hot_to_very_hot: u8,
    ) -> Self {
        // Validate ascending order
        assert!(
            cold_to_warm < warm_to_hot && warm_to_hot < hot_to_very_hot,
            "Thresholds must be in ascending order: {} < {} < {} failed",
            cold_to_warm,
            warm_to_hot,
            hot_to_very_hot
        );

        // Validate maximum value (4-bit counter limit)
        assert!(
            hot_to_very_hot <= 15,
            "hot_to_very_hot threshold ({}) must be <= 15 (4-bit counter maximum)",
            hot_to_very_hot
        );

        // Additional validation: ensure reasonable gaps between levels
        assert!(
            cold_to_warm >= 1,
            "cold_to_warm threshold must be >= 1 to distinguish from zero-access blocks"
        );

        self.inactive_backend = Some(InactiveBackendConfig::MultiLru {
            frequency_thresholds: [cold_to_warm, warm_to_hot, hot_to_very_hot],
        });
        self
    }

    /// Use HashMap backend with custom reuse policy
    pub fn with_hashmap_backend(mut self, reuse_policy: Box<dyn ReusePolicy>) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::HashMap { reuse_policy });
        self
    }

    /// Use lineage backend
    pub fn with_lineage_backend(mut self) -> Self {
        self.inactive_backend = Some(InactiveBackendConfig::Lineage);
        self
    }

    /// Validate the configuration
    fn validate(&self) -> Result<(), String> {
        let registry = self.registry.as_ref().ok_or("registry is required")?;

        let block_count = self.block_count.ok_or("block_count is required")?;

        if block_count == 0 {
            return Err("block_count must be greater than 0".to_string());
        }

        // Validate block_size
        let block_size = self.block_size.unwrap_or(16);
        if !block_size.is_power_of_two() || !(1..=1024).contains(&block_size) {
            return Err(format!(
                "Invalid block_size {}: must be a power of 2 between 1 and 1024",
                block_size
            ));
        }

        // Additional validation for MultiLRU thresholds at build time
        if let Some(InactiveBackendConfig::MultiLru {
            frequency_thresholds,
        }) = &self.inactive_backend
        {
            let [t1, t2, t3] = frequency_thresholds;
            if !(*t1 < *t2 && *t2 < *t3) {
                return Err(format!(
                    "Invalid thresholds [{}, {}, {}]: must be in ascending order",
                    t1, t2, t3
                ));
            }
            if *t3 > 15 {
                return Err(format!(
                    "Invalid threshold {}: maximum frequency is 15 (4-bit counter)",
                    t3
                ));
            }

            // Validate MultiLRU requires frequency tracking
            if !registry.has_frequency_tracking() {
                return Err(
                    "MultiLRU backend requires a registry with frequency tracking".to_string(),
                );
            }
        }

        Ok(())
    }

    /// Build the BlockManager
    pub fn build(mut self) -> Result<BlockManager<T>, BlockManagerBuilderError> {
        // First validate the configuration
        self.validate()
            .map_err(BlockManagerBuilderError::ValidationError)?;

        let block_count = self.block_count.unwrap();
        let block_size = self.block_size.unwrap_or(16);

        // Use provided registry
        let registry = self.registry.unwrap();

        // Create reset pool
        let blocks: Vec<Block<T, Reset>> = (0..block_count as BlockId)
            .map(|id| Block::new(id, block_size))
            .collect();
        let reset_pool = ResetPool::new(blocks, block_size);

        // Create backend based on configuration
        let backend: Box<dyn InactivePoolBackend<T>> = match self.inactive_backend.take() {
            Some(InactiveBackendConfig::HashMap { reuse_policy }) => {
                tracing::info!("Using HashMap for inactive pool");
                Box::new(HashMapBackend::new(reuse_policy))
            }
            Some(InactiveBackendConfig::Lru) => {
                // Capacity automatically set to block_count
                let capacity = NonZeroUsize::new(block_count).expect("block_count must be > 0");
                tracing::info!("Using LRU for inactive pool");
                Box::new(LruBackend::new(capacity))
            }
            Some(InactiveBackendConfig::MultiLru {
                frequency_thresholds,
            }) => {
                // Require frequency tracker for MultiLRU
                let frequency_tracker = registry.frequency_tracker().ok_or_else(|| {
                    BlockManagerBuilderError::InvalidBackend(
                        "MultiLRU backend requires a registry with frequency tracking".to_string(),
                    )
                })?;

                // Total capacity = block_count, distributed across 4 levels
                let capacity_per_level = block_count.div_ceil(4); // Round up division
                let level_capacity =
                    NonZeroUsize::new(capacity_per_level).expect("capacity per level must be > 0");

                tracing::info!(
                    "Using MultiLRU inactive backend with thresholds: {:?}",
                    frequency_thresholds
                );
                Box::new(
                    MultiLruBackend::new_with_thresholds(
                        level_capacity,
                        &frequency_thresholds,
                        frequency_tracker,
                    )
                    .map_err(|e| BlockManagerBuilderError::InvalidBackend(e.to_string()))?,
                )
            }
            Some(InactiveBackendConfig::Lineage) => {
                tracing::info!("Using Lineage inactive backend");
                Box::new(LineageBackend::default())
            }
            None => {
                tracing::info!("Using default inactive backend: Lineage");
                Box::new(LineageBackend::default())
            }
        };

        // Create pools
        let inactive_pool = InactivePool::new(backend, &reset_pool);
        let active_pool = ActivePool::new(inactive_pool.clone());

        // Create upgrade function that captures InactivePool for unified lookup
        let inactive_pool_clone = inactive_pool.clone();
        let upgrade_fn = Arc::new(
            move |seq_hash: SequenceHash| -> Option<Arc<dyn RegisteredBlock<T>>> {
                // Use InactivePool's unified lookup (handles both active and inactive)
                inactive_pool_clone.find_or_promote_dyn(seq_hash)
            },
        );

        Ok(BlockManager {
            reset_pool,
            active_pool,
            inactive_pool,
            block_registry: registry,
            duplication_policy: self
                .duplication_policy
                .unwrap_or(BlockDuplicationPolicy::Allow),
            upgrade_fn,
            allocate_mutex: Mutex::new(()),
            total_blocks: block_count,
            block_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::blocks::BlockError;
    use super::*;
    use crate::KvbmSequenceHashProvider;
    use dynamo_tokens::TokenBlockSequence;
    use rstest::rstest;

    #[derive(Debug, Clone, PartialEq)]
    struct TestBlockData {
        value: u32,
    }

    /// Helper function to create a token block with specific data
    fn create_token_block(tokens: &[u32]) -> dynamo_tokens::TokenBlock {
        let token_sequence = TokenBlockSequence::from_slice(tokens, tokens.len() as u32, Some(42));
        if let Some(block) = token_sequence.blocks().first() {
            block.clone()
        } else {
            let mut partial = token_sequence.into_parts().1;
            partial.commit().expect("Should be able to commit")
        }
    }

    /// Helper function to create a token block using fill_iota pattern
    fn create_test_token_block_from_iota(start: u32) -> dynamo_tokens::TokenBlock {
        // Use fill_iota to generate [start, start+1, start+2, start+3]
        let tokens: Vec<u32> = (start..start + 4).collect();
        create_token_block(&tokens)
    }

    fn create_test_token_block_8_from_iota(start: u32) -> dynamo_tokens::TokenBlock {
        // Generate 8 sequential tokens starting from start
        let tokens: Vec<u32> = (start..start + 8).collect();
        create_token_block(&tokens)
    }

    /// Helper function to create a token block with exactly 16 tokens for testing
    #[expect(dead_code)]
    fn create_token_block_16() -> dynamo_tokens::TokenBlock {
        let tokens: Vec<u32> = (100..116).collect(); // 16 tokens: 100, 101, ..., 115
        create_token_block(&tokens)
    }

    /// Helper function to create a basic manager for testing
    fn create_test_manager(block_count: usize) -> BlockManager<TestBlockData> {
        let registry = BlockRegistry::with_frequency_tracker(
            FrequencyTrackingCapacity::default().create_tracker(),
        );

        BlockManager::<TestBlockData>::builder()
            .block_count(block_count)
            .block_size(4) // Most tests use 4-token blocks
            .registry(registry)
            .with_lru_backend()
            .build()
            .expect("Should build manager")
    }

    // ============================================================================
    // BUILDER PATTERN TESTS
    // ============================================================================

    mod builder_tests {
        use super::*;

        #[test]
        fn test_builder_default() {
            let registry = BlockRegistry::new();
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .registry(registry)
                .build()
                .expect("Should build with defaults");

            // Verify we can allocate blocks
            let blocks = manager.allocate_blocks(5);
            assert!(blocks.is_some());
            assert_eq!(blocks.unwrap().len(), 5);
        }

        #[test]
        fn test_builder_with_lru_backend() {
            let registry = BlockRegistry::new();
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .registry(registry)
                .with_lru_backend()
                .build()
                .expect("Should build with LRU backend");

            // Verify we can allocate blocks
            let blocks = manager.allocate_blocks(10);
            assert!(blocks.is_some());
            assert_eq!(blocks.unwrap().len(), 10);
        }

        #[test]
        fn test_builder_with_multi_lru_backend() {
            let registry = BlockRegistry::with_frequency_tracker(
                FrequencyTrackingCapacity::Small.create_tracker(),
            );
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .registry(registry)
                .with_multi_lru_backend()
                .build()
                .expect("Should build with MultiLRU backend");

            // Verify we can allocate blocks
            let blocks = manager.allocate_blocks(8);
            assert!(blocks.is_some());
            assert_eq!(blocks.unwrap().len(), 8);
        }

        #[test]
        fn test_builder_with_custom_multi_lru_thresholds() {
            let registry = BlockRegistry::with_frequency_tracker(
                FrequencyTrackingCapacity::Medium.create_tracker(),
            );
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .registry(registry)
                .with_multi_lru_backend_custom_thresholds(2, 6, 12)
                .build()
                .expect("Should build with custom thresholds");

            // Verify we can allocate blocks
            let blocks = manager.allocate_blocks(4);
            assert!(blocks.is_some());
            assert_eq!(blocks.unwrap().len(), 4);
        }

        #[test]
        fn test_builder_with_duplication_policy() {
            let registry = BlockRegistry::new();
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(50)
                .registry(registry)
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .with_lru_backend()
                .build()
                .expect("Should build with duplication policy");

            let blocks = manager.allocate_blocks(2);
            assert!(blocks.is_some());
            assert_eq!(blocks.unwrap().len(), 2);
        }

        #[test]
        fn test_builder_validation_zero_blocks() {
            let registry = BlockRegistry::new();
            let result = BlockManager::<TestBlockData>::builder()
                .block_count(0)
                .registry(registry)
                .build();

            assert!(result.is_err());
            if let Err(err) = result {
                assert!(
                    err.to_string()
                        .contains("block_count must be greater than 0")
                );
            }
        }

        #[test]
        fn test_builder_validation_missing_block_count() {
            let registry = BlockRegistry::new();
            let result = BlockManager::<TestBlockData>::builder()
                .registry(registry)
                .with_lru_backend()
                .build();

            assert!(result.is_err());
            if let Err(err) = result {
                assert!(err.to_string().contains("block_count is required"));
            }
        }

        #[test]
        fn test_builder_validation_missing_registry() {
            let result = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .with_lru_backend()
                .build();

            assert!(result.is_err());
            if let Err(err) = result {
                assert!(err.to_string().contains("registry is required"));
            }
        }

        #[test]
        #[should_panic(expected = "must be <= 15")]
        fn test_builder_invalid_threshold_too_high() {
            BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .with_multi_lru_backend_custom_thresholds(2, 6, 20); // 20 > 15, should panic
        }

        #[test]
        #[should_panic(expected = "must be in ascending order")]
        fn test_builder_invalid_threshold_order() {
            BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .with_multi_lru_backend_custom_thresholds(6, 2, 10); // Not ascending, should panic
        }

        #[test]
        fn test_builder_multi_lru_requires_frequency_tracking() {
            let registry = BlockRegistry::new(); // No frequency tracking
            let result = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .registry(registry)
                .with_multi_lru_backend()
                .build();

            assert!(result.is_err());
            if let Err(err) = result {
                assert!(err.to_string().contains("frequency tracking"));
            }
        }
    }

    // ============================================================================
    // BLOCK ALLOCATION TESTS
    // ============================================================================

    mod allocation_tests {
        use super::*;

        #[test]
        fn test_allocate_single_block() {
            let manager = create_test_manager(10);

            let initial_available = manager.available_blocks();
            let initial_total = manager.total_blocks();
            assert_eq!(initial_available, 10);

            let blocks = manager.allocate_blocks(1).expect("Should allocate 1 block");
            assert_eq!(blocks.len(), 1);

            // Verify available blocks decreased
            assert_eq!(manager.available_blocks(), initial_available - 1);
            assert_eq!(manager.total_blocks(), initial_total);

            let block = blocks.into_iter().next().unwrap();
            // Verify block has a valid ID
            let _block_id = block.block_id();

            // Drop the block and verify it returns to pool
            drop(block);
            assert_eq!(manager.available_blocks(), initial_available);
            assert_eq!(manager.total_blocks(), initial_total);
        }

        #[test]
        fn test_allocate_multiple_blocks() {
            let manager = create_test_manager(20);

            let initial_available = manager.available_blocks();
            let initial_total = manager.total_blocks();
            assert_eq!(initial_available, 20);

            let blocks = manager
                .allocate_blocks(5)
                .expect("Should allocate 5 blocks");
            assert_eq!(blocks.len(), 5);

            // Verify available blocks decreased correctly
            assert_eq!(manager.available_blocks(), initial_available - 5);
            assert_eq!(manager.total_blocks(), initial_total);

            // Verify all blocks have unique IDs
            let mut block_ids = Vec::new();
            for block in blocks {
                let id = block.block_id();
                assert!(!block_ids.contains(&id), "Block IDs should be unique");
                block_ids.push(id);
            }

            // All blocks should return to pool automatically on drop
            assert_eq!(manager.available_blocks(), initial_available);
            assert_eq!(manager.total_blocks(), initial_total);
        }

        #[test]
        fn test_allocate_all_blocks() {
            let manager = create_test_manager(10);

            let blocks = manager
                .allocate_blocks(10)
                .expect("Should allocate all blocks");
            assert_eq!(blocks.len(), 10);
        }

        #[test]
        fn test_allocate_more_than_available() {
            let manager = create_test_manager(5);

            let result = manager.allocate_blocks(10);
            assert!(
                result.is_none(),
                "Should not allocate more blocks than available"
            );
        }

        #[test]
        fn test_allocate_zero_blocks() {
            let manager = create_test_manager(10);

            let blocks = manager
                .allocate_blocks(0)
                .expect("Should allocate 0 blocks");
            assert_eq!(blocks.len(), 0);
        }

        #[test]
        fn test_sequential_allocations() {
            let manager = create_test_manager(10);

            let total_blocks = manager.total_blocks();
            assert_eq!(manager.available_blocks(), total_blocks);

            let blocks1 = manager.allocate_blocks(3).expect("First allocation");
            assert_eq!(blocks1.len(), 3);
            assert_eq!(manager.available_blocks(), total_blocks - 3);

            let blocks2 = manager.allocate_blocks(4).expect("Second allocation");
            assert_eq!(blocks2.len(), 4);
            assert_eq!(manager.available_blocks(), total_blocks - 7);

            let blocks3 = manager.allocate_blocks(3).expect("Third allocation");
            assert_eq!(blocks3.len(), 3);
            assert_eq!(manager.available_blocks(), 0);

            // Should have no blocks left
            let blocks4 = manager.allocate_blocks(1);
            assert!(blocks4.is_none(), "Should not have any blocks left");

            // Drop blocks in reverse order and verify counts
            drop(blocks3);
            assert_eq!(manager.available_blocks(), 3);

            drop(blocks2);
            assert_eq!(manager.available_blocks(), 7);

            drop(blocks1);
            assert_eq!(manager.available_blocks(), total_blocks);
            assert_eq!(manager.total_blocks(), total_blocks);
        }
    }

    // ============================================================================
    // BLOCK LIFECYCLE AND POOL RETURN TESTS
    // ============================================================================

    mod lifecycle_tests {
        use super::*;

        #[test]
        fn test_mutable_block_returns_to_reset_pool() {
            let manager = create_test_manager(10);

            let initial_available = manager.available_blocks();
            let initial_total = manager.total_blocks();
            assert_eq!(initial_available, 10);
            assert_eq!(initial_total, 10);

            {
                let blocks = manager
                    .allocate_blocks(3)
                    .expect("Should allocate 3 blocks");
                assert_eq!(blocks.len(), 3);

                // Available blocks should decrease
                assert_eq!(manager.available_blocks(), initial_available - 3);
                assert_eq!(manager.total_blocks(), initial_total); // Total never changes
            } // MutableBlocks dropped here - should return to reset pool

            // Available blocks should return to original count
            assert_eq!(manager.available_blocks(), initial_available);
            assert_eq!(manager.total_blocks(), initial_total);
        }

        #[test]
        fn test_complete_block_returns_to_reset_pool() {
            let manager = create_test_manager(10);

            let initial_available = manager.available_blocks();
            let initial_total = manager.total_blocks();

            {
                let mutable_blocks = manager.allocate_blocks(2).expect("Should allocate blocks");
                assert_eq!(manager.available_blocks(), initial_available - 2);

                let _complete_blocks: Vec<_> = mutable_blocks
                    .into_iter()
                    .enumerate()
                    .map(|(i, block)| {
                        let tokens = vec![400 + i as u32, 401 + i as u32, 402 + i as u32];
                        let token_block = create_token_block(&tokens);
                        block.complete(token_block)
                    })
                    .collect();

                // Blocks are still unavailable while in Complete state
                assert_eq!(manager.available_blocks(), initial_available - 2);
            } // CompleteBlocks dropped here - should return to reset pool

            // Available blocks should return to original count since blocks weren't registered
            assert_eq!(manager.available_blocks(), initial_available);
            assert_eq!(manager.total_blocks(), initial_total);
        }

        #[test]
        fn test_registered_block_lifecycle() {
            let manager = create_test_manager(10);

            let initial_available = manager.available_blocks();
            let initial_total = manager.total_blocks();

            // Step 1: Allocate and complete blocks
            let token_block = create_test_token_block_from_iota(500);
            let seq_hash = token_block.kvbm_sequence_hash();

            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            assert_eq!(manager.available_blocks(), initial_available - 1);

            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block)
                .expect("Should complete block");

            // Still unavailable while in Complete state
            assert_eq!(manager.available_blocks(), initial_available - 1);

            // Step 2: Register the block
            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            assert_eq!(immutable_blocks.len(), 1);
            let immutable_block = immutable_blocks.into_iter().next().unwrap();

            // Block is still not available (it's now in active/inactive pools, not reset)
            assert_eq!(manager.available_blocks(), initial_available - 1);

            {
                // Step 3: Use the block and verify it can be matched
                let matched_blocks = manager.match_blocks(&[seq_hash]);
                assert_eq!(matched_blocks.len(), 1);
                assert_eq!(matched_blocks[0].sequence_hash(), seq_hash);

                // Still not available while being used
                assert_eq!(manager.available_blocks(), initial_available - 1);
            } // matched blocks dropped here

            // Step 4: Drop the original registered block
            drop(immutable_block);

            // Block should now be available again (moved to inactive pool when ref count reached 0)
            assert_eq!(manager.available_blocks(), initial_available);
            assert_eq!(manager.total_blocks(), initial_total);
        }

        #[test]
        fn test_concurrent_allocation_and_return() {
            use std::sync::Arc;
            use std::thread;

            let manager = Arc::new(create_test_manager(20));
            let initial_total = manager.total_blocks();

            let handles: Vec<_> = (0..5)
                .map(|i| {
                    let manager_clone = Arc::clone(&manager);
                    thread::spawn(move || {
                        // Each thread allocates and drops some blocks
                        for j in 0..3 {
                            let blocks = manager_clone.allocate_blocks(2);
                            if let Some(blocks) = blocks {
                                // Complete one block
                                let token_block =
                                    create_test_token_block_from_iota((600 + i * 10 + j) as u32);
                                let complete_block = blocks
                                    .into_iter()
                                    .next()
                                    .unwrap()
                                    .complete(token_block)
                                    .expect("Should complete block");

                                // Register and drop
                                let _immutable_blocks =
                                    manager_clone.register_blocks(vec![complete_block]);
                                // blocks automatically dropped at end of scope
                            }
                        }
                    })
                })
                .collect();

            // Wait for all threads to complete
            for handle in handles {
                handle.join().unwrap();
            }

            // All blocks should eventually be available again
            assert_eq!(manager.total_blocks(), initial_total);
            // Available might be less than total if some blocks are in inactive pool,
            // but total should be preserved
        }

        #[test]
        fn test_full_block_lifecycle() {
            let manager = create_test_manager(10);
            let total_blocks = manager.total_blocks();
            assert_eq!(manager.available_blocks(), total_blocks);

            // Step 1: Allocate 5 blocks
            let mutable_blocks = manager
                .allocate_blocks(5)
                .expect("Should allocate 5 blocks");
            assert_eq!(manager.available_blocks(), total_blocks - 5);
            assert_eq!(manager.total_blocks(), total_blocks);

            // Step 2: Complete 3 blocks, drop 2 mutable blocks
            let mut mutable_blocks_iter = mutable_blocks.into_iter();
            let complete_blocks: Vec<_> = (0..3)
                .map(|i| {
                    let block = mutable_blocks_iter.next().unwrap();
                    let tokens = vec![
                        700 + i as u32,
                        701 + i as u32,
                        702 + i as u32,
                        703 + i as u32,
                    ];
                    let token_block = create_token_block(&tokens);
                    block.complete(token_block).expect("Should complete block")
                })
                .collect();
            let mutable_part: Vec<_> = mutable_blocks_iter.collect();

            drop(mutable_part); // Drop 2 mutable blocks

            // Should have 2 blocks returned to reset pool
            assert_eq!(manager.available_blocks(), total_blocks - 3);

            // Step 3: Register the 3 completed blocks
            let immutable_blocks = manager.register_blocks(complete_blocks);
            assert_eq!(immutable_blocks.len(), 3);

            // Still 3 blocks unavailable (now in active pool)
            assert_eq!(manager.available_blocks(), total_blocks - 3);

            // Step 4: Match and use one of the blocks
            let seq_hash = create_test_token_block_from_iota(700).kvbm_sequence_hash();
            let matched_blocks = manager.match_blocks(&[seq_hash]);
            assert_eq!(matched_blocks.len(), 1);

            // Step 5: Drop one registered block, keep others
            drop(immutable_blocks.into_iter().next());

            // Still have registered blocks in use, so available count depends on ref counting
            let available_after_drop = manager.available_blocks();
            assert!(available_after_drop >= total_blocks - 3);
            assert!(available_after_drop <= total_blocks);

            // Step 6: Drop everything
            drop(matched_blocks);

            // Eventually all blocks should be available again
            // (Some might be in inactive pool, but available_blocks counts both reset and inactive)
            assert_eq!(manager.total_blocks(), total_blocks);
            let final_available = manager.available_blocks();
            assert_eq!(final_available, total_blocks); // Allow for some blocks in inactive pool
        }
    }

    // ============================================================================
    // BLOCK SIZE VALIDATION TESTS
    // ============================================================================

    mod block_size_tests {

        use super::*;

        #[test]
        fn test_default_block_size() {
            let manager = create_test_manager(10);
            assert_eq!(manager.block_size(), 4); // create_test_manager uses block_size(4)
        }

        #[test]
        fn test_custom_block_size() {
            let registry = BlockRegistry::new();
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(32)
                .registry(registry)
                .build()
                .expect("Should build with custom block size");
            assert_eq!(manager.block_size(), 32);
        }

        #[test]
        fn test_block_size_validation_correct_size() {
            let manager = create_test_manager(10);
            let token_block = create_test_token_block_from_iota(100); // 4 tokens

            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let mutable_block = mutable_blocks.into_iter().next().unwrap();

            // Should succeed since token_block has exactly 4 tokens
            let result = mutable_block.complete(token_block);
            assert!(result.is_ok());
        }

        #[test]
        fn test_block_size_validation_wrong_size() {
            // Create a manager expecting 8-token blocks
            let registry = BlockRegistry::new();
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(8)
                .registry(registry)
                .with_lru_backend()
                .build()
                .expect("Should build manager");
            let token_block = create_test_token_block_from_iota(1); // 4 tokens, expected 8

            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let mutable_block = mutable_blocks.into_iter().next().unwrap();

            // Should fail since token_block has 4 tokens but manager expects 8
            let result = mutable_block.complete(token_block);
            assert!(result.is_err());

            if let Err(BlockError::BlockSizeMismatch {
                expected,
                actual,
                block: _,
            }) = result
            {
                assert_eq!(expected, 8);
                assert_eq!(actual, 4);
            } else {
                panic!("Expected BlockSizeMismatch error");
            }
        }

        #[rstest]
        #[case(1)]
        #[case(2)]
        #[case(4)]
        #[case(8)]
        #[case(16)]
        #[case(32)]
        #[case(64)]
        #[case(128)]
        #[case(256)]
        #[case(512)]
        #[case(1024)]
        fn test_builder_block_size_power_of_two(#[case] size: usize) {
            let registry = BlockRegistry::new();
            let result = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(size)
                .registry(registry)
                .build();
            assert!(result.is_ok(), "Block size {} should be valid", size);
        }

        #[test]
        #[should_panic(expected = "block_size must be a power of 2")]
        fn test_builder_block_size_not_power_of_two() {
            BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(15); // Not a power of 2
        }

        #[test]
        #[should_panic(expected = "block_size must be between 1 and 1024")]
        fn test_builder_block_size_too_large() {
            BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(2048); // Too large
        }

        #[test]
        #[should_panic(expected = "block_size must be between 1 and 1024")]
        fn test_builder_block_size_zero() {
            BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(0); // Zero is invalid
        }

        #[test]
        #[should_panic(expected = "block_size must be a power of 2")]
        fn test_builder_validation_invalid_block_size() {
            BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(7); // Not a power of 2, panics immediately
        }

        #[test]
        fn test_different_block_sizes() {
            // Test with block size 4
            let registry_4 = BlockRegistry::new();
            let manager_4 = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(4)
                .registry(registry_4)
                .build()
                .expect("Should build with block size 4");

            let token_block_4 = create_test_token_block_from_iota(10); // 4 tokens
            let mutable_blocks = manager_4
                .allocate_blocks(1)
                .expect("Should allocate blocks");
            let result = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block_4);
            assert!(result.is_ok());

            // Test with block size 8
            let registry_8 = BlockRegistry::new();
            let manager_8 = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(8)
                .registry(registry_8)
                .build()
                .expect("Should build with block size 8");

            let token_block_8 = create_test_token_block_8_from_iota(20); // 8 tokens
            let mutable_blocks = manager_8
                .allocate_blocks(1)
                .expect("Should allocate blocks");
            let result = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block_8);
            assert!(result.is_ok());
        }
    }

    // ============================================================================
    // BLOCK REGISTRATION AND DEDUPLICATION TESTS
    // ============================================================================

    mod registration_tests {
        use super::*;

        #[test]
        fn test_register_single_block() {
            let manager = create_test_manager(10);

            let token_block = create_test_token_block_from_iota(150);
            let expected_hash = token_block.kvbm_sequence_hash();
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block)
                .expect("Should complete block");

            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            assert_eq!(immutable_blocks.len(), 1);

            let immutable_block = immutable_blocks.into_iter().next().unwrap();
            assert_eq!(immutable_block.sequence_hash(), expected_hash);
        }

        #[test]
        fn test_register_multiple_blocks() {
            let manager = create_test_manager(10);

            let mut complete_blocks = Vec::new();
            let mut expected_hashes = Vec::new();

            for i in 0..3 {
                let tokens = vec![100 + i, 101 + i, 102 + i, 103 + i];
                let token_block = create_token_block(&tokens);
                expected_hashes.push(token_block.kvbm_sequence_hash());

                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                complete_blocks.push(complete_block);
            }

            let immutable_blocks = manager.register_blocks(complete_blocks);
            assert_eq!(immutable_blocks.len(), 3);

            for (i, immutable_block) in immutable_blocks.iter().enumerate() {
                assert_eq!(immutable_block.sequence_hash(), expected_hashes[i]);
            }
        }

        #[rstest]
        #[case(BlockDuplicationPolicy::Allow, 200, "allow", false)]
        #[case(BlockDuplicationPolicy::Reject, 300, "reject", true)]
        fn test_deduplication_policy(
            #[case] policy: BlockDuplicationPolicy,
            #[case] iota_base: u32,
            #[case] policy_name: &str,
            #[case] expect_same_block_id: bool,
        ) {
            let registry = BlockRegistry::new();
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(4)
                .registry(registry)
                .duplication_policy(policy)
                .with_lru_backend()
                .build()
                .expect("Should build manager");

            let token_block = create_test_token_block_from_iota(iota_base);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Register the same sequence hash twice
            let complete_block1 = {
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block.clone())
                    .expect("Should complete block")
            };

            let complete_block2 = {
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block")
            };

            let immutable_blocks1 = manager.register_blocks(vec![complete_block1]);
            let immutable_blocks2 = manager.register_blocks(vec![complete_block2]);

            assert_eq!(immutable_blocks1.len(), 1);
            assert_eq!(immutable_blocks2.len(), 1);

            // Both should have the same sequence hash
            assert_eq!(immutable_blocks1[0].sequence_hash(), seq_hash);
            assert_eq!(immutable_blocks2[0].sequence_hash(), seq_hash);

            // Check block IDs based on policy
            if expect_same_block_id {
                // Duplicates are rejected - same block ID
                assert_eq!(
                    immutable_blocks1[0].block_id(),
                    immutable_blocks2[0].block_id(),
                    "With {} policy, duplicates should reuse the same block ID",
                    policy_name
                );
            } else {
                // Duplicates are allowed - different block IDs
                assert_ne!(
                    immutable_blocks1[0].block_id(),
                    immutable_blocks2[0].block_id(),
                    "With {} policy, duplicates should have different block IDs",
                    policy_name
                );
            }
        }

        #[test]
        fn test_register_mutable_block_from_existing_reject_returns_block_to_reset_pool() {
            let registry = BlockRegistry::new();
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(2)
                .block_size(4)
                .registry(registry)
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .build()
                .expect("Should build manager");

            let blocks = manager
                .allocate_blocks(2)
                .expect("Should allocate two blocks");
            let mut iter = blocks.into_iter();
            let primary_mutable = iter.next().expect("Should have first block");
            let duplicate_mutable = iter.next().expect("Should have second block");

            let primary_id = primary_mutable.block_id();
            let duplicate_id = duplicate_mutable.block_id();

            let token_block = create_test_token_block_from_iota(42);
            let primary_complete = primary_mutable
                .complete(token_block)
                .expect("Should complete primary block");

            let mut registered = manager.register_blocks(vec![primary_complete]);
            let primary_immutable = registered.pop().expect("Should register primary block");

            let result =
                manager.register_mutable_block_from_existing(duplicate_mutable, &primary_immutable);

            assert_eq!(
                result.block_id(),
                primary_id,
                "Should reuse existing primary when duplicates are rejected"
            );

            assert_eq!(
                manager.available_blocks(),
                1,
                "Rejected duplicate should be returned to the reset pool"
            );

            let mut returned_blocks = manager
                .allocate_blocks(1)
                .expect("Should allocate returned reset block");
            let returned_block = returned_blocks
                .pop()
                .expect("Should contain one returned block");

            assert_eq!(
                returned_block.block_id(),
                duplicate_id,
                "Returned block should be the rejected duplicate"
            );
        }
    }

    // ============================================================================
    // BLOCK MATCHING TESTS
    // ============================================================================

    mod matching_tests {
        use super::*;

        #[test]
        fn test_match_no_blocks() {
            let manager = create_test_manager(10);

            let seq_hashes = vec![create_test_token_block_from_iota(400).kvbm_sequence_hash()];
            let matched_blocks = manager.match_blocks(&seq_hashes);
            assert_eq!(matched_blocks.len(), 0);
        }

        #[test]
        fn test_match_single_block() {
            let manager = create_test_manager(10);

            let token_block = create_test_token_block_from_iota(500);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Register a block
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block)
                .expect("Should complete block");
            let _immutable_blocks = manager.register_blocks(vec![complete_block]);

            // Try to match it
            let matched_blocks = manager.match_blocks(&[seq_hash]);
            assert_eq!(matched_blocks.len(), 1);
            assert_eq!(matched_blocks[0].sequence_hash(), seq_hash);
        }

        #[test]
        fn test_match_multiple_blocks() {
            let manager = create_test_manager(10);

            let mut seq_hashes = Vec::new();

            // Register multiple blocks
            for i in 0..4 {
                let tokens = vec![600 + i, 601 + i, 602 + i, 603 + i];
                let token_block = create_token_block(&tokens);
                seq_hashes.push(token_block.kvbm_sequence_hash());

                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                let _immutable_blocks = manager.register_blocks(vec![complete_block]);
            }

            // Match all blocks
            let matched_blocks = manager.match_blocks(&seq_hashes);
            assert_eq!(matched_blocks.len(), 4);

            for (i, matched_block) in matched_blocks.iter().enumerate() {
                assert_eq!(matched_block.sequence_hash(), seq_hashes[i]);
            }
        }

        #[test]
        fn test_match_partial_blocks() {
            let manager = create_test_manager(10);

            let mut seq_hashes = Vec::new();

            // Register only some blocks
            for i in 0..3 {
                let tokens = vec![700 + i, 701 + i, 702 + i, 703 + i];
                let token_block = create_token_block(&tokens);
                seq_hashes.push(token_block.kvbm_sequence_hash());

                if i < 2 {
                    // Only register first 2 blocks
                    let mutable_blocks =
                        manager.allocate_blocks(1).expect("Should allocate blocks");
                    let complete_block = mutable_blocks
                        .into_iter()
                        .next()
                        .unwrap()
                        .complete(token_block)
                        .expect("Should complete block");
                    let _immutable_blocks = manager.register_blocks(vec![complete_block]);
                }
            }

            // Try to match all 3 - should only get 2
            let matched_blocks = manager.match_blocks(&seq_hashes);
            assert_eq!(matched_blocks.len(), 2);

            for matched_block in matched_blocks {
                assert!(seq_hashes[0..2].contains(&matched_block.sequence_hash()));
            }
        }

        #[test]
        fn test_match_blocks_returns_immutable_blocks() {
            let manager = create_test_manager(10);

            let token_block = create_test_token_block_from_iota(800);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Register a block
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block)
                .expect("Should complete block");
            let _immutable_blocks = manager.register_blocks(vec![complete_block]);

            // Match and verify it's an ImmutableBlock
            let matched_blocks = manager.match_blocks(&[seq_hash]);
            assert_eq!(matched_blocks.len(), 1);

            let immutable_block = &matched_blocks[0];
            assert_eq!(immutable_block.sequence_hash(), seq_hash);

            // Test that we can downgrade it
            let weak_block = immutable_block.downgrade();
            assert_eq!(weak_block.sequence_hash(), seq_hash);
        }
    }

    // ============================================================================
    // IMMUTABLE BLOCK AND WEAK BLOCK TESTS
    // ============================================================================

    mod immutable_block_tests {
        use super::*;

        #[test]
        fn test_immutable_block_downgrade_upgrade() {
            let manager = create_test_manager(10);

            let token_block = create_test_token_block_from_iota(100);
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block)
                .expect("Should complete block");

            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            let immutable_block = immutable_blocks.into_iter().next().unwrap();

            // Test downgrade to WeakBlock
            let weak_block = immutable_block.downgrade();
            assert_eq!(weak_block.sequence_hash(), immutable_block.sequence_hash());

            // Test upgrade from WeakBlock
            let upgraded_block = weak_block.upgrade().expect("Should be able to upgrade");
            assert_eq!(
                upgraded_block.sequence_hash(),
                immutable_block.sequence_hash()
            );
            assert_eq!(upgraded_block.block_id(), immutable_block.block_id());
        }

        #[test]
        fn test_weak_block_upgrade_after_drop() {
            let manager = create_test_manager(10);

            let token_block = create_test_token_block_from_iota(200);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Create a weak block
            let weak_block = {
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                let immutable_blocks = manager.register_blocks(vec![complete_block]);
                let immutable_block = immutable_blocks.into_iter().next().unwrap();

                // Downgrade to weak
                immutable_block.downgrade()
            }; // immutable_block is dropped here

            // The upgrade function should still find the block through the pools
            let upgraded_block = weak_block.upgrade();

            // The result depends on whether the block is still in the pools
            if let Some(block) = upgraded_block {
                assert_eq!(block.sequence_hash(), seq_hash);
            }
        }

        #[test]
        fn test_weak_block_upgrade_nonexistent() {
            let manager = create_test_manager(10);

            let token_block = create_token_block(&[999, 998, 997, 996]); // Keep non-sequential for this test

            // Create an ImmutableBlock and immediately downgrade it
            let weak_block = {
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                let immutable_blocks = manager.register_blocks(vec![complete_block]);
                let immutable_block = immutable_blocks.into_iter().next().unwrap();
                immutable_block.downgrade()
            };

            // Force eviction by filling up the pool with other blocks
            for i in 0..10 {
                let tokens = vec![1000 + i, 1001 + i, 1002 + i, 1003 + i];
                let token_block = create_token_block(&tokens);
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                let _immutable_blocks = manager.register_blocks(vec![complete_block]);
            }

            // Try to upgrade - might fail if the original block was evicted
            let upgraded_block = weak_block.upgrade();
            assert!(upgraded_block.is_none());
            // // This test just verifies that upgrade doesn't panic, result can be None
            // if let Some(block) = upgraded_block {
            //     assert_eq!(
            //         block.sequence_hash(),
            //         create_token_block(&[999, 998, 997, 996]).sequence_hash()
            //     );
            // }
        }

        #[test]
        fn test_multiple_weak_blocks_same_sequence() {
            let manager = create_test_manager(10);

            let token_block = create_test_token_block_from_iota(150);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Create multiple weak blocks from the same immutable block
            let (weak1, weak2, weak3) = {
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                let immutable_blocks = manager.register_blocks(vec![complete_block]);
                let immutable_block = immutable_blocks.into_iter().next().unwrap();

                let w1 = immutable_block.downgrade();
                let w2 = immutable_block.downgrade();
                let w3 = immutable_block.downgrade();
                (w1, w2, w3)
            };

            // All weak blocks should have the same sequence hash
            assert_eq!(weak1.sequence_hash(), seq_hash);
            assert_eq!(weak2.sequence_hash(), seq_hash);
            assert_eq!(weak3.sequence_hash(), seq_hash);

            // All should be able to upgrade
            let upgraded1 = weak1.upgrade().expect("Should upgrade");
            let upgraded2 = weak2.upgrade().expect("Should upgrade");
            let upgraded3 = weak3.upgrade().expect("Should upgrade");

            assert_eq!(upgraded1.sequence_hash(), seq_hash);
            assert_eq!(upgraded2.sequence_hash(), seq_hash);
            assert_eq!(upgraded3.sequence_hash(), seq_hash);
        }
    }

    // ============================================================================
    // UPGRADE FUNCTION TESTS
    // ============================================================================

    mod upgrade_function_tests {
        use super::*;

        #[test]
        fn test_upgrade_function_finds_active_blocks() {
            let manager = create_test_manager(10);

            let token_block = create_test_token_block_from_iota(250);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Register a block (this puts it in active pool initially)
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block)
                .expect("Should complete block");
            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            let immutable_block = immutable_blocks.into_iter().next().unwrap();

            // Create a weak block and test upgrade
            let weak_block = immutable_block.downgrade();
            let upgraded = weak_block
                .upgrade()
                .expect("Should find block in active pool");
            assert_eq!(upgraded.sequence_hash(), seq_hash);
        }

        #[test]
        fn test_upgrade_function_finds_inactive_blocks() {
            let manager = create_test_manager(20);

            let token_block = create_test_token_block_from_iota(350);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Register a block
            let weak_block = {
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                let immutable_blocks = manager.register_blocks(vec![complete_block]);
                let immutable_block = immutable_blocks.into_iter().next().unwrap();
                immutable_block.downgrade()
            };

            // Force the block to potentially move to inactive pool by creating many other blocks
            for i in 0..10 {
                let tokens = vec![400 + i, 401 + i, 402 + i, 403 + i];
                let token_block = create_token_block(&tokens);
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                let _immutable_blocks = manager.register_blocks(vec![complete_block]);
            }

            // Try to upgrade - should still find the original block
            let upgraded = weak_block.upgrade();
            if let Some(block) = upgraded {
                assert_eq!(block.sequence_hash(), seq_hash);
            }
        }
    }

    // ============================================================================
    // ERROR HANDLING AND EDGE CASE TESTS
    // ============================================================================

    mod error_handling_tests {
        use super::*;

        #[test]
        fn test_allocation_exhaustion() {
            let manager = create_test_manager(3);

            // Allocate all blocks
            let blocks1 = manager
                .allocate_blocks(2)
                .expect("Should allocate 2 blocks");
            let blocks2 = manager.allocate_blocks(1).expect("Should allocate 1 block");

            // Try to allocate more - should fail
            let blocks3 = manager.allocate_blocks(1);
            assert!(
                blocks3.is_none(),
                "Should not be able to allocate when pool is empty"
            );

            // Drop some blocks and try again
            drop(blocks1);
            drop(blocks2);

            // Blocks should be returned to pool automatically
            let blocks4 = manager.allocate_blocks(1);
            assert!(
                blocks4.is_some(),
                "Should be able to allocate after blocks are returned"
            );
        }

        #[test]
        fn test_empty_sequence_matching() {
            let manager = create_test_manager(10);

            let matched_blocks = manager.match_blocks(&[]);
            assert_eq!(matched_blocks.len(), 0);
        }

        #[test]
        fn test_register_empty_block_list() {
            let manager = create_test_manager(10);

            let immutable_blocks = manager.register_blocks(vec![]);
            assert_eq!(immutable_blocks.len(), 0);
        }
    }

    // ============================================================================
    // INTEGRATION TESTS
    // ============================================================================

    mod integration_tests {
        use super::*;

        #[test]
        fn test_full_lifecycle_single_block() {
            let manager = create_test_manager(10);

            // 1. Allocate a mutable block
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
            let mutable_block = mutable_blocks.into_iter().next().unwrap();
            let block_id = mutable_block.block_id();

            // 2. Complete the block
            let token_block = create_test_token_block_from_iota(1);
            let seq_hash = token_block.kvbm_sequence_hash();
            let complete_block = mutable_block
                .complete(token_block)
                .expect("Should complete block");

            assert_eq!(complete_block.block_id(), block_id);
            assert_eq!(complete_block.sequence_hash(), seq_hash);

            // 3. Register the block
            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            let immutable_block = immutable_blocks.into_iter().next().unwrap();

            assert_eq!(immutable_block.block_id(), block_id);
            assert_eq!(immutable_block.sequence_hash(), seq_hash);

            // 4. Match the block
            let matched_blocks = manager.match_blocks(&[seq_hash]);
            assert_eq!(matched_blocks.len(), 1);
            assert_eq!(matched_blocks[0].sequence_hash(), seq_hash);

            // 5. Create weak reference and upgrade
            let weak_block = immutable_block.downgrade();
            let upgraded_block = weak_block.upgrade().expect("Should upgrade");
            assert_eq!(upgraded_block.sequence_hash(), seq_hash);
        }

        #[rstest]
        #[case("lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lru_backend())]
        #[case("multi_lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_multi_lru_backend())]
        fn test_multiple_blocks_different_backends(
            #[case] backend_name: &str,
            #[case] backend_builder: fn(
                BlockManagerConfigBuilder<TestBlockData>,
            ) -> BlockManagerConfigBuilder<TestBlockData>,
        ) {
            let registry = BlockRegistry::with_frequency_tracker(
                FrequencyTrackingCapacity::default().create_tracker(),
            );
            let manager = backend_builder(
                BlockManager::<TestBlockData>::builder()
                    .block_count(20)
                    .block_size(4)
                    .registry(registry),
            )
            .build()
            .expect("Should build");

            // Allocate, complete, and register blocks using BlockSequenceBuilder
            let base = 1000; // Use fixed base since we only test one backend per test now
            let tokens: Vec<u32> = (base as u32..base as u32 + 20).collect(); // 5 blocks * 4 tokens each = 20 tokens

            let mut seq_hashes = Vec::new();
            let mut complete_blocks = Vec::new();

            // Create token blocks from sequence
            let token_blocks = {
                let token_seq = dynamo_tokens::TokenBlockSequence::from_slice(&tokens, 4, Some(42));
                token_seq.blocks().to_vec()
            };

            for token_block in token_blocks.iter() {
                let seq_hash = token_block.kvbm_sequence_hash();
                seq_hashes.push(seq_hash);

                // Allocate mutable block and complete it
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block.clone())
                    .expect("Should complete block");
                complete_blocks.push(complete_block);
            }

            // Register all blocks
            let _immutable_blocks = manager.register_blocks(complete_blocks);

            // Verify all blocks can be matched
            let matched_blocks = manager.match_blocks(&seq_hashes);
            assert_eq!(
                matched_blocks.len(),
                5,
                "Manager with {} backend should match all blocks",
                backend_name
            );
        }

        #[test]
        fn test_concurrent_allocation_simulation() {
            let manager = create_test_manager(50);

            // Simulate concurrent allocations by interleaving operations
            let mut all_blocks = Vec::new();
            let mut all_hashes = Vec::new();

            // Phase 1: Allocate and complete some blocks
            for i in 0..10 {
                let tokens = vec![2000 + i, 2001 + i, 2002 + i, 2003 + i];
                let token_block = create_token_block(&tokens);
                all_hashes.push(token_block.kvbm_sequence_hash());

                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                all_blocks.push(complete_block);
            }

            // Phase 2: Register half the blocks
            let mut remaining_blocks = all_blocks.split_off(5);
            let _immutable_blocks1 = manager.register_blocks(all_blocks);

            // Phase 3: Allocate more blocks while some are registered
            for i in 10..15 {
                let tokens = vec![2000 + i, 2001 + i, 2002 + i, 2003 + i];
                let token_block = create_token_block(&tokens);
                all_hashes.push(token_block.kvbm_sequence_hash());

                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block");
                remaining_blocks.push(complete_block);
            }

            // Phase 4: Register remaining blocks
            let _immutable_blocks2 = manager.register_blocks(remaining_blocks);

            // Phase 5: Verify we can match all registered blocks
            let matched_blocks = manager.match_blocks(&all_hashes);
            assert_eq!(
                matched_blocks.len(),
                15,
                "Should match all registered blocks"
            );
        }

        #[test]
        fn test_shared_registry_across_managers() {
            // Create shared registry with frequency tracking
            let tracker = FrequencyTrackingCapacity::Medium.create_tracker();
            let registry = BlockRegistry::with_frequency_tracker(tracker);

            #[derive(Clone, Debug)]
            struct G1;

            #[derive(Clone, Debug)]
            struct G2;

            // Create two managers with different metadata types and policies
            let manager1 = BlockManager::<G1>::builder()
                .block_count(100)
                .block_size(4)
                .registry(registry.clone())
                .duplication_policy(BlockDuplicationPolicy::Allow)
                .with_multi_lru_backend()
                .build()
                .expect("Should build manager1");

            let manager2 = BlockManager::<G2>::builder()
                .block_count(100)
                .block_size(4)
                .registry(registry.clone())
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .with_multi_lru_backend()
                .build()
                .expect("Should build manager2");

            // Verify both managers work
            assert_eq!(manager1.total_blocks(), 100);
            assert_eq!(manager2.total_blocks(), 100);

            // Verify they share the same registry (frequency tracking works across both)
            let token_block = create_test_token_block_from_iota(3000);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Register in manager1
            let mutable_blocks1 = manager1.allocate_blocks(1).expect("Should allocate");
            let complete_block1 = mutable_blocks1
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block.clone())
                .expect("Should complete");
            let _immutable1 = manager1.register_blocks(vec![complete_block1]);

            // Both managers should see the registered block count in shared registry
            assert!(registry.is_registered(seq_hash));
        }
    }
}
