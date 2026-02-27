// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test manager creation helpers.

use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::Result;

use crate::blocks::{BlockMetadata, BlockRegistry};
use crate::events::EventsManager;
use crate::manager::{BlockManager, FrequencyTrackingCapacity};
use crate::SequenceHash;

use super::token_blocks;

/// Create a basic test manager with LRU backend.
pub fn create_test_manager<T: BlockMetadata>(block_count: usize) -> BlockManager<T> {
    let registry = BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
        .build();

    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(4) // Most tests use 4-token blocks
        .registry(registry)
        .with_lru_backend()
        .build()
        .expect("Should build manager")
}

/// Create a test manager with custom block size.
pub fn create_test_manager_with_block_size<T: BlockMetadata>(
    block_count: usize,
    block_size: usize,
) -> BlockManager<T> {
    let registry = BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
        .build();

    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(block_size)
        .registry(registry)
        .with_lru_backend()
        .build()
        .expect("Should build manager")
}

/// Builder for creating test BlockRegistry with optional events integration.
///
/// # Example
///
/// ```ignore
/// // Simple registry
/// let registry = TestRegistryBuilder::new().build();
///
/// // With events manager
/// let events_manager = Arc::new(EventsManager::builder().build());
/// let registry = TestRegistryBuilder::new()
///     .events_manager(events_manager)
///     .build();
///
/// // With custom frequency tracking
/// let registry = TestRegistryBuilder::new()
///     .frequency_tracking(FrequencyTrackingCapacity::Large)
///     .build();
/// ```
#[derive(Default)]
pub struct TestRegistryBuilder {
    events_manager: Option<Arc<EventsManager>>,
    frequency_tracking: FrequencyTrackingCapacity,
}

impl TestRegistryBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self {
            events_manager: None,
            frequency_tracking: FrequencyTrackingCapacity::Medium,
        }
    }

    /// Sets the events manager for distributed event coordination.
    pub fn events_manager(mut self, manager: Arc<EventsManager>) -> Self {
        self.events_manager = Some(manager);
        self
    }

    /// Sets the frequency tracking capacity.
    ///
    /// Default: Medium
    pub fn frequency_tracking(mut self, capacity: FrequencyTrackingCapacity) -> Self {
        self.frequency_tracking = capacity;
        self
    }

    /// Builds the BlockRegistry.
    pub fn build(self) -> BlockRegistry {
        let mut builder =
            BlockRegistry::builder().frequency_tracker(self.frequency_tracking.create_tracker());

        if let Some(events_manager) = self.events_manager {
            builder = builder.event_manager(events_manager);
        }

        builder.build()
    }
}

/// Builder for creating test BlockManagers.
///
/// # Example
///
/// ```ignore
/// // Simple manager (creates its own registry)
/// let manager = TestManagerBuilder::<G1>::new()
///     .block_count(100)
///     .block_size(4)
///     .build();
///
/// // With explicit registry (for events integration)
/// let events_manager = Arc::new(EventsManager::builder().build());
/// let registry = TestRegistryBuilder::new()
///     .events_manager(events_manager.clone())
///     .build();
/// let manager = TestManagerBuilder::<G1>::new()
///     .block_count(100)
///     .block_size(4)
///     .registry(registry)
///     .build();
///
/// // Convenience: with events manager (creates registry internally)
/// let manager = TestManagerBuilder::<G1>::new()
///     .block_count(100)
///     .block_size(4)
///     .events_manager(events_manager)
///     .build();
/// ```
pub struct TestManagerBuilder<T: BlockMetadata> {
    block_count: Option<usize>,
    block_size: Option<usize>,
    registry: Option<BlockRegistry>,
    events_manager: Option<Arc<EventsManager>>,
    frequency_tracking: FrequencyTrackingCapacity,
    _phantom: PhantomData<T>,
}

impl<T: BlockMetadata> Default for TestManagerBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BlockMetadata> TestManagerBuilder<T> {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self {
            block_count: None,
            block_size: None,
            registry: None,
            events_manager: None,
            frequency_tracking: FrequencyTrackingCapacity::Medium,
            _phantom: PhantomData,
        }
    }

    /// Sets the number of blocks in the pool.
    pub fn block_count(mut self, count: usize) -> Self {
        self.block_count = Some(count);
        self
    }

    /// Sets the tokens per block (must be power of 2, 1-1024).
    pub fn block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size);
        self
    }

    /// Sets the registry to use.
    ///
    /// If not set, a registry will be created based on `frequency_tracking`
    /// and `events_manager` settings.
    pub fn registry(mut self, registry: BlockRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Sets the events manager for distributed event coordination.
    ///
    /// This is a convenience method that creates a registry with the events manager.
    /// If you also call `registry()`, this setting is ignored.
    pub fn events_manager(mut self, manager: Arc<EventsManager>) -> Self {
        self.events_manager = Some(manager);
        self
    }

    /// Sets the frequency tracking capacity for auto-created registry.
    ///
    /// Ignored if `registry()` is called.
    ///
    /// Default: Medium
    pub fn frequency_tracking(mut self, capacity: FrequencyTrackingCapacity) -> Self {
        self.frequency_tracking = capacity;
        self
    }

    /// Builds the BlockManager.
    ///
    /// # Panics
    ///
    /// Panics if `block_count` or `block_size` are not set.
    pub fn build(self) -> BlockManager<T> {
        let block_count = self.block_count.expect("block_count is required");
        let block_size = self.block_size.expect("block_size is required");

        let registry = self.registry.unwrap_or_else(|| {
            let mut builder =
                TestRegistryBuilder::new().frequency_tracking(self.frequency_tracking);
            if let Some(events_manager) = self.events_manager {
                builder = builder.events_manager(events_manager);
            }
            builder.build()
        });

        BlockManager::<T>::builder()
            .block_count(block_count)
            .block_size(block_size)
            .registry(registry)
            .with_lru_backend()
            .build()
            .expect("Should build test manager")
    }
}

/// Populate a BlockManager with token blocks and return their sequence hashes.
///
/// This function:
/// 1. Allocates blocks from the manager
/// 2. Completes them with provided token blocks
/// 3. Registers them
/// 4. Drops the immutable blocks (returns to inactive pool)
///
/// # Returns
/// Vec of sequence hashes for the registered blocks (in order)
pub fn populate_manager_with_blocks<T: BlockMetadata>(
    manager: &BlockManager<T>,
    token_blocks: &[dynamo_tokens::TokenBlock],
) -> Result<Vec<SequenceHash>> {
    let blocks = manager
        .allocate_blocks(token_blocks.len())
        .ok_or_else(|| anyhow::anyhow!("Failed to allocate {} blocks", token_blocks.len()))?;

    let complete_blocks: Vec<_> = blocks
        .into_iter()
        .zip(token_blocks.iter())
        .map(|(block, token_block)| {
            block
                .complete(&token_block)
                .map_err(|e| anyhow::anyhow!("Failed to complete block: {:?}", e))
        })
        .collect::<Result<Vec<_>>>()?;

    let seq_hashes: Vec<SequenceHash> = complete_blocks.iter().map(|b| b.sequence_hash()).collect();

    let immutable_blocks = manager.register_blocks(complete_blocks);

    // Drop immutable blocks - they return to inactive pool via RAII
    drop(immutable_blocks);

    Ok(seq_hashes)
}

/// Quick setup: create manager and populate with sequential token blocks.
///
/// # Arguments
/// * `block_count` - Number of blocks
/// * `block_size` - Tokens per block
/// * `start_token` - Starting token value for sequence
///
/// # Returns
/// (BlockManager, Vec<SequenceHash>)
pub fn create_and_populate_manager<T: BlockMetadata>(
    block_count: usize,
    block_size: usize,
    start_token: u32,
    registry: BlockRegistry,
) -> Result<(BlockManager<T>, Vec<SequenceHash>)> {
    let manager = TestManagerBuilder::<T>::new()
        .block_count(block_count)
        .block_size(block_size)
        .registry(registry)
        .build();

    let token_sequence = token_blocks::create_token_sequence(block_count, block_size, start_token);
    let seq_hashes = populate_manager_with_blocks(&manager, token_sequence.blocks())?;

    Ok((manager, seq_hashes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct TestMetadata;

    #[test]
    fn test_create_test_manager() {
        let manager = TestManagerBuilder::<TestMetadata>::new()
            .block_count(100)
            .block_size(16)
            .build();
        assert_eq!(manager.total_blocks(), 100);
        assert_eq!(manager.block_size(), 16);
        assert_eq!(manager.available_blocks(), 100);
    }

    #[test]
    fn test_populate_manager_with_blocks() {
        let manager = TestManagerBuilder::<TestMetadata>::new()
            .block_count(50)
            .block_size(4)
            .build();
        let token_seq = token_blocks::create_token_sequence(10, 4, 0);

        let seq_hashes =
            populate_manager_with_blocks(&manager, token_seq.blocks()).expect("Should populate");

        assert_eq!(seq_hashes.len(), 10);
        // Blocks should be in inactive pool after population
        assert_eq!(manager.available_blocks(), 50);
    }

    #[test]
    fn test_create_and_populate_manager() {
        let registry = TestRegistryBuilder::new().build();
        let (manager, hashes) = create_and_populate_manager::<TestMetadata>(32, 4, 100, registry)
            .expect("Should create");

        assert_eq!(hashes.len(), 32);
        assert_eq!(manager.total_blocks(), 32);
        assert_eq!(manager.available_blocks(), 32);

        // Verify blocks can be matched
        let matched = manager.match_blocks(&hashes);
        assert_eq!(matched.len(), 32);
    }
}
