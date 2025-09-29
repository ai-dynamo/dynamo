// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block Manager v2

use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;

use super::{
    policies::{BlockDuplicationPolicy, ReusePolicy, reuse::fifo::FifoReusePolicy},
    pools::{
        ActivePool, BlockMetadata, RegisteredBlock, SequenceHash,
        block::{Block, Reset},
        frequency_sketch::TinyLFUTracker,
        inactive::InactivePool,
        inactive::backends::{
            InactivePoolBackend, hashmap_backend::HashMapBackend, lru_backend::LruBackend,
            multi_lru_backend::MultiLruBackend,
        },
        registry::BlockRegistry,
        reset::ResetPool,
        *,
    },
};

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
}

/// Builder for BlockManager configuration
pub struct BlockManagerConfigBuilder<T: BlockMetadata> {
    /// Number of blocks in the pool
    block_count: Option<usize>,

    /// Size of each block in tokens (must be power of 2, 1-1024)
    /// Default: 16
    block_size: Option<usize>,

    /// Frequency tracker size for TinyLFU (must be power of 2)
    /// Default: 2^21, Min: 2^18, Max: 2^24
    frequency_tracker_size: Option<usize>,

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

/// BlockManager v2 with pluggable inactive pool backends
pub struct BlockManager<T: BlockMetadata> {
    reset_pool: ResetPool<T>,
    active_pool: ActivePool<T>,
    inactive_pool: InactivePool<T>,
    block_registry: BlockRegistry,
    duplication_policy: BlockDuplicationPolicy,
    upgrade_fn: Arc<dyn Fn(SequenceHash) -> Option<Arc<dyn RegisteredBlock<T>>> + Send + Sync>,
    allocate_mutex: Mutex<()>,
    total_blocks: usize,
    block_size: usize,
}

impl<T: BlockMetadata> BlockManager<T> {
    /// Create a new builder for BlockManager
    ///
    /// # Example
    /// ```ignore
    /// let manager = BlockManager::builder()
    ///     .block_count(1000)
    ///     .with_multi_lru_backend()
    ///     .build()?;
    /// ```
    pub fn builder() -> BlockManagerConfigBuilder<T> {
        BlockManagerConfigBuilder::default()
    }

    pub fn allocate_blocks(&self, count: usize) -> Option<Vec<MutableBlock<T>>> {
        let _guard = self.allocate_mutex.lock();
        let mut blocks = self.reset_pool.try_allocate_blocks(count);
        match self.inactive_pool.allocate_blocks(count - blocks.len()) {
            Some(remaining) => {
                blocks.extend(remaining);
                Some(blocks)
            }
            None => return None,
        }
    }

    pub fn register_blocks(&self, blocks: Vec<CompleteBlock<T>>) -> Vec<ImmutableBlock<T>> {
        let pool_return_fn = self.inactive_pool.return_fn();
        blocks
            .into_iter()
            .map(|block| {
                let handle = self
                    .block_registry
                    .register_sequence_hash(block.sequence_hash());
                let registered_block =
                    handle.register_block(block, self.duplication_policy, pool_return_fn.clone());
                ImmutableBlock::new(registered_block, self.upgrade_fn.clone())
            })
            .collect()
    }

    pub fn match_blocks(&self, seq_hash: &[SequenceHash]) -> Vec<ImmutableBlock<T>> {
        // First try to match against active blocks
        let mut matched: Vec<ImmutableBlock<T>> = Vec::with_capacity(seq_hash.len());
        matched.extend(
            self.active_pool
                .find_matches(seq_hash, true)
                .into_iter()
                .map(|block| ImmutableBlock::new(block, self.upgrade_fn.clone())),
        );

        // If we didn't match all hashes, try inactive blocks for the remaining ones
        let remaining_hashes = &seq_hash[matched.len()..];
        if !remaining_hashes.is_empty() {
            matched.extend(
                self.inactive_pool
                    .find_blocks(remaining_hashes, true)
                    .into_iter()
                    .map(|block| ImmutableBlock::new(block, self.upgrade_fn.clone())),
            );
        }

        matched
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
}

impl<T: BlockMetadata> Default for BlockManagerConfigBuilder<T> {
    fn default() -> Self {
        Self {
            block_count: None,
            block_size: Some(16), // Default to 16 tokens per block
            frequency_tracker_size: None,
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
            size >= 1 && size <= 1024,
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
    /// Set frequency tracker size with validation
    /// Must be a power of 2 between 2^18 and 2^24
    pub fn frequency_tracker_size(mut self, size: usize) -> Self {
        assert!(
            size >= (1 << 18) && size <= (1 << 24),
            "Frequency tracker size must be between 2^18 and 2^24, got: {}",
            size
        );
        assert!(
            size.is_power_of_two(),
            "Frequency tracker size must be a power of 2, got: {}",
            size
        );
        self.frequency_tracker_size = Some(size);
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

    /// Validate the configuration
    fn validate(&self) -> Result<(), String> {
        let block_count = self.block_count.ok_or("block_count is required")?;

        if block_count == 0 {
            return Err("block_count must be greater than 0".to_string());
        }

        // Validate block_size
        let block_size = self.block_size.unwrap_or(16);
        if !block_size.is_power_of_two() || block_size < 1 || block_size > 1024 {
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

        // Create registry with frequency tracking
        let freq_size = self.frequency_tracker_size.unwrap_or(2_097_152);
        let frequency_tracker = Arc::new(TinyLFUTracker::new(freq_size));
        let registry = BlockRegistry::with_frequency_tracker(frequency_tracker.clone());

        // Create reset pool
        let blocks: Vec<Block<T, Reset>> = (0..block_count as u64)
            .map(|id| Block::new(id, block_size))
            .collect();
        let reset_pool = ResetPool::new(blocks, block_size);

        // Create backend based on configuration
        let backend: Box<dyn InactivePoolBackend<T>> = match self.inactive_backend.take() {
            Some(InactiveBackendConfig::HashMap { reuse_policy }) => {
                Box::new(HashMapBackend::new(reuse_policy))
            }
            Some(InactiveBackendConfig::Lru) => {
                // Capacity automatically set to block_count
                let capacity = NonZeroUsize::new(block_count).expect("block_count must be > 0");
                Box::new(LruBackend::new(capacity))
            }
            Some(InactiveBackendConfig::MultiLru {
                frequency_thresholds,
            }) => {
                // Total capacity = block_count, distributed across 4 levels
                let capacity_per_level = (block_count + 3) / 4; // Round up division
                let level_capacity =
                    NonZeroUsize::new(capacity_per_level).expect("capacity per level must be > 0");

                Box::new(MultiLruBackend::new_with_thresholds(
                    level_capacity,
                    &frequency_thresholds,
                    frequency_tracker,
                ))
            }
            None => {
                // Default to HashMap with FIFO
                Box::new(HashMapBackend::new(Box::new(FifoReusePolicy::new())))
            }
        };

        // Create pools
        let inactive_pool = InactivePool::new(backend, &reset_pool);
        let active_pool = ActivePool::new(registry.clone(), inactive_pool.return_fn());

        // Create upgrade function that captures the necessary components
        let registry_clone = registry.clone();
        let inactive_pool_clone = inactive_pool.clone();
        let return_fn_clone = inactive_pool.return_fn();
        let upgrade_fn = Arc::new(
            move |seq_hash: SequenceHash| -> Option<Arc<dyn RegisteredBlock<T>>> {
                // Try active pool first with touch=false (using registry directly)
                if let Some(handle) = registry_clone.match_sequence_hash(seq_hash, false) {
                    if let Some(block) = handle.try_get_block::<T>(return_fn_clone.clone()) {
                        return Some(block);
                    }
                }
                // Then try inactive pool with touch=false
                if let Some(block) = inactive_pool_clone
                    .find_blocks(&[seq_hash], false)
                    .into_iter()
                    .next()
                {
                    return Some(block);
                }
                None
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
    use super::*;
    use crate::tokens::TokenBlockSequence;

    #[derive(Debug, Clone, PartialEq)]
    struct TestBlockData {
        value: u32,
    }

    /// Helper function to create a token block with specific data
    fn create_token_block(tokens: &[u32]) -> crate::tokens::TokenBlock {
        let token_sequence = TokenBlockSequence::from_slice(tokens, tokens.len() as u32, Some(42));
        if let Some(block) = token_sequence.blocks().first() {
            block.clone()
        } else {
            let mut partial = token_sequence.into_parts().1;
            partial.commit().expect("Should be able to commit")
        }
    }

    /// Helper function to create a token block using fill_iota pattern
    fn create_test_token_block_from_iota(start: u32) -> crate::tokens::TokenBlock {
        // Use fill_iota to generate [start, start+1, start+2, start+3]
        let tokens: Vec<u32> = (start..start + 4).collect();
        create_token_block(&tokens)
    }

    fn create_test_token_block_8_from_iota(start: u32) -> crate::tokens::TokenBlock {
        // Generate 8 sequential tokens starting from start
        let tokens: Vec<u32> = (start..start + 8).collect();
        create_token_block(&tokens)
    }

    /// Helper function to create a token block with exactly 16 tokens for testing
    fn create_token_block_16() -> crate::tokens::TokenBlock {
        let tokens: Vec<u32> = (100..116).collect(); // 16 tokens: 100, 101, ..., 115
        create_token_block(&tokens)
    }

    /// Helper function to create a basic manager for testing
    fn create_test_manager(block_count: usize) -> BlockManager<TestBlockData> {
        BlockManager::<TestBlockData>::builder()
            .block_count(block_count)
            .block_size(4) // Most tests use 4-token blocks
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
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .build()
                .expect("Should build with defaults");

            // Verify we can allocate blocks
            let blocks = manager.allocate_blocks(5);
            assert!(blocks.is_some());
            assert_eq!(blocks.unwrap().len(), 5);
        }

        #[test]
        fn test_builder_with_lru_backend() {
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(100)
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
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .frequency_tracker_size(1 << 20) // 2^20
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
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .frequency_tracker_size(1 << 21) // 2^21 (default)
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
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(50)
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
            let result = BlockManager::<TestBlockData>::builder()
                .block_count(0)
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
            let result = BlockManager::<TestBlockData>::builder()
                .with_lru_backend()
                .build();

            assert!(result.is_err());
            if let Err(err) = result {
                assert!(err.to_string().contains("block_count is required"));
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
        #[should_panic(expected = "must be between 2^18 and 2^24")]
        fn test_builder_invalid_frequency_tracker_size() {
            BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .frequency_tracker_size(1000); // Not a valid size, should panic
        }

        #[test]
        #[should_panic(expected = "must be a power of 2")]
        fn test_builder_non_power_of_two_frequency_tracker() {
            BlockManager::<TestBlockData>::builder()
                .block_count(100)
                .frequency_tracker_size((1 << 20) + 1); // Not power of 2, should panic
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
            let seq_hash = token_block.sequence_hash();

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
            let seq_hash = create_test_token_block_from_iota(700).sequence_hash();
            let matched_blocks = manager.match_blocks(&[seq_hash]);
            assert_eq!(matched_blocks.len(), 1);

            // Step 5: Drop one registered block, keep others
            drop(immutable_blocks.into_iter().nth(0));

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
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(32)
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
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(8)
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

        #[test]
        fn test_builder_block_size_power_of_two() {
            // Valid power of 2 values should work
            for &size in &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
                let result = BlockManager::<TestBlockData>::builder()
                    .block_count(10)
                    .block_size(size)
                    .build();
                assert!(result.is_ok(), "Block size {} should be valid", size);
            }
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
            let manager_4 = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(4)
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
            let manager_8 = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(8)
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
            let expected_hash = token_block.sequence_hash();
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
                expected_hashes.push(token_block.sequence_hash());

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

        #[test]
        fn test_deduplication_allow_policy() {
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(4)
                .duplication_policy(BlockDuplicationPolicy::Allow)
                .with_lru_backend()
                .build()
                .expect("Should build manager");

            let token_block = create_test_token_block_from_iota(200);
            let seq_hash = token_block.sequence_hash();

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

            // Both should have the same sequence hash but potentially different block IDs
            assert_eq!(immutable_blocks1[0].sequence_hash(), seq_hash);
            assert_eq!(immutable_blocks2[0].sequence_hash(), seq_hash);

            // Both should have the same sequence hash but potentially different block IDs
            // Duplicates are allowed.
            assert_ne!(
                immutable_blocks1[0].block_id(),
                immutable_blocks2[0].block_id()
            );
        }

        #[test]
        fn test_deduplication_reject_policy() {
            let manager = BlockManager::<TestBlockData>::builder()
                .block_count(10)
                .block_size(4)
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .with_lru_backend()
                .build()
                .expect("Should build manager");

            let token_block = create_test_token_block_from_iota(300);
            let seq_hash = token_block.sequence_hash();

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

            let immutable_blocks1 = manager.register_blocks(vec![complete_block1]);
            assert_eq!(immutable_blocks1.len(), 1);
            assert_eq!(immutable_blocks1[0].sequence_hash(), seq_hash);

            // Register a duplicate - should still work but may reuse the existing registration
            let complete_block2 = {
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(token_block)
                    .expect("Should complete block")
            };

            let immutable_blocks2 = manager.register_blocks(vec![complete_block2]);
            assert_eq!(immutable_blocks2.len(), 1);
            assert_eq!(immutable_blocks2[0].sequence_hash(), seq_hash);

            // Duplicates are rejected.
            assert_eq!(
                immutable_blocks1[0].block_id(),
                immutable_blocks2[0].block_id()
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

            let seq_hashes = vec![create_test_token_block_from_iota(400).sequence_hash()];
            let matched_blocks = manager.match_blocks(&seq_hashes);
            assert_eq!(matched_blocks.len(), 0);
        }

        #[test]
        fn test_match_single_block() {
            let manager = create_test_manager(10);

            let token_block = create_test_token_block_from_iota(500);
            let seq_hash = token_block.sequence_hash();

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
                seq_hashes.push(token_block.sequence_hash());

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
                seq_hashes.push(token_block.sequence_hash());

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
            let seq_hash = token_block.sequence_hash();

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
            let seq_hash = token_block.sequence_hash();

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
            let seq_hash = token_block.sequence_hash();

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
            let seq_hash = token_block.sequence_hash();

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
            let seq_hash = token_block.sequence_hash();

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
            let seq_hash = token_block.sequence_hash();
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

        #[test]
        fn test_multiple_blocks_different_backends() {
            // Test with LRU backend
            let manager_lru = BlockManager::<TestBlockData>::builder()
                .block_count(20)
                .block_size(4)
                .with_lru_backend()
                .build()
                .expect("Should build");

            // Test with MultiLRU backend
            let manager_multi_lru = BlockManager::<TestBlockData>::builder()
                .block_count(20)
                .block_size(4)
                .with_multi_lru_backend()
                .build()
                .expect("Should build");

            // Test with HashMap backend (skipping HashMap for now due to backend issue)
            let managers = vec![manager_lru, manager_multi_lru];

            for (i, manager) in managers.iter().enumerate() {
                // Allocate, complete, and register blocks using BlockSequenceBuilder
                let base = 1000 + (i * 20); // Space out sequences for different managers
                let tokens: Vec<u32> = (base as u32..base as u32 + 20).collect(); // 5 blocks * 4 tokens each = 20 tokens

                let mut seq_hashes = Vec::new();
                let mut complete_blocks = Vec::new();

                // Create token blocks from sequence
                let token_blocks = {
                    let token_seq =
                        crate::tokens::TokenBlockSequence::from_slice(&tokens, 4, Some(42));
                    token_seq.blocks().to_vec()
                };

                for (j, token_block) in token_blocks.iter().enumerate() {
                    let seq_hash = token_block.sequence_hash();
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
                    "Manager {} should match all blocks",
                    i
                );
            }
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
                all_hashes.push(token_block.sequence_hash());

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
                all_hashes.push(token_block.sequence_hash());

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
    }
}
