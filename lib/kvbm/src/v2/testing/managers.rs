// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BlockManager testing utilities.

use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::Result;

use crate::{
    logical::{
        blocks::{BlockMetadata, BlockRegistry},
        events::EventsManager,
        manager::{BlockManager, FrequencyTrackingCapacity},
    },
    v2::logical::pools::SequenceHash,
};

use super::token_blocks;

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

/// Create a test registry with medium frequency tracking.
#[deprecated(note = "Use TestRegistryBuilder instead")]
pub fn create_test_registry() -> BlockRegistry {
    TestRegistryBuilder::new().build()
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
///
/// # Example
/// ```ignore
/// let manager = create_test_manager::<G2>(100, 4);
/// let token_seq = token_blocks::create_token_sequence(32, 4, 0);
/// let seq_hashes = populate_manager_with_blocks(&manager, token_seq.blocks())?;
/// assert_eq!(seq_hashes.len(), 32);
/// ```
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
                .complete(token_block.clone())
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
///
/// # Example
/// ```ignore
/// let (manager, hashes) = create_and_populate_manager::<G2>(32, 4, 0)?;
/// assert_eq!(hashes.len(), 32);
/// assert_eq!(manager.available_blocks(), 32);  // All in inactive pool
/// ```
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

// =============================================================================
// Multi-Instance Population Helper
// =============================================================================

use std::ops::Range;

use super::token_blocks::create_token_sequence;
use crate::BlockId;
use crate::v2::KvbmSequenceHashProvider;
use crate::v2::physical::transfer::FillPattern;
use dynamo_tokens::TokenBlockSequence;

/// Specification for a single instance's population.
pub struct InstancePopulationSpec<'a, M: BlockMetadata> {
    /// The block manager to populate.
    pub manager: &'a BlockManager<M>,
    /// Range of blocks from the full sequence to use (e.g., 0..16).
    pub block_range: Range<usize>,
    /// Optional fill pattern for physical blocks.
    /// Requires access to the physical layout via a fill function.
    pub fill_pattern: Option<FillPattern>,
}

/// Result of populating a single instance.
pub struct InstancePopulationResult {
    /// The instance index (0-based order of add_instance calls).
    pub instance_index: usize,
    /// Block IDs that were allocated for this instance.
    pub block_ids: Vec<BlockId>,
    /// Sequence hashes for the populated blocks.
    pub hashes: Vec<SequenceHash>,
}

/// Results from populating multiple instances.
pub struct PopulatedInstances {
    /// The full token sequence used for population.
    token_sequence: TokenBlockSequence,
    /// All sequence hashes from the full sequence.
    all_hashes: Vec<SequenceHash>,
    /// Per-instance results.
    instance_results: Vec<InstancePopulationResult>,
}

impl PopulatedInstances {
    /// Returns all sequence hashes from the full token sequence.
    pub fn all_hashes(&self) -> &[SequenceHash] {
        &self.all_hashes
    }

    /// Returns the token sequence.
    pub fn token_sequence(&self) -> &TokenBlockSequence {
        &self.token_sequence
    }

    /// Returns the block IDs for a specific instance.
    pub fn instance_block_ids(&self, instance_index: usize) -> Option<&[BlockId]> {
        self.instance_results
            .get(instance_index)
            .map(|r| r.block_ids.as_slice())
    }

    /// Returns the sequence hashes for a specific instance.
    pub fn instance_hashes(&self, instance_index: usize) -> Option<&[SequenceHash]> {
        self.instance_results
            .get(instance_index)
            .map(|r| r.hashes.as_slice())
    }

    /// Returns the number of instances that were populated.
    pub fn instance_count(&self) -> usize {
        self.instance_results.len()
    }

    /// Returns all instance results.
    pub fn instance_results(&self) -> &[InstancePopulationResult] {
        &self.instance_results
    }
}

/// Builder for populating multiple instances with blocks from a shared token sequence.
///
/// This reduces boilerplate when setting up multi-instance tests where each instance
/// gets a different slice of a continuous token sequence.
///
/// # Example
///
/// ```ignore
/// // BEFORE: 12+ lines per instance
/// let full_sequence = create_token_sequence(32, 16, 0);
/// let inst1_blocks: Vec<_> = full_sequence.blocks()[0..16].to_vec();
/// let inst1_hashes = populate_manager_with_blocks(&inst1.g2_manager(), &inst1_blocks)?;
/// let inst1_matched = inst1.g2_manager().match_blocks(&inst1_hashes);
/// let inst1_block_ids: Vec<_> = inst1_matched.into_iter().map(|b| b.block_id()).collect();
/// // ... repeat for inst2, inst3 ...
///
/// // AFTER: 5 lines total
/// let population = MultiInstancePopulator::builder()
///     .total_blocks(32)
///     .block_size(16)
///     .add_instance(&inst1.g2_manager(), 0..16)
///     .add_instance(&inst2.g2_manager(), 16..24)
///     .add_instance(&inst3.g2_manager(), 24..32)
///     .build()?;
///
/// // Access results
/// let all_hashes = population.all_hashes();
/// let inst1_ids = population.instance_block_ids(0).unwrap();
/// ```
pub struct MultiInstancePopulatorBuilder<'a, M: BlockMetadata> {
    total_blocks: Option<usize>,
    block_size: Option<usize>,
    start_token: u32,
    instances: Vec<InstancePopulationSpec<'a, M>>,
}

impl<'a, M: BlockMetadata> Default for MultiInstancePopulatorBuilder<'a, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, M: BlockMetadata> MultiInstancePopulatorBuilder<'a, M> {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            total_blocks: None,
            block_size: None,
            start_token: 0,
            instances: Vec::new(),
        }
    }

    /// Sets the total number of blocks in the token sequence.
    pub fn total_blocks(mut self, count: usize) -> Self {
        self.total_blocks = Some(count);
        self
    }

    /// Sets the number of tokens per block.
    pub fn block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size);
        self
    }

    /// Sets the starting token value (default: 0).
    pub fn start_token(mut self, token: u32) -> Self {
        self.start_token = token;
        self
    }

    /// Adds an instance to populate with blocks from the specified range.
    ///
    /// # Arguments
    /// * `manager` - The BlockManager to populate
    /// * `block_range` - Range of block indices from the full sequence (e.g., 0..16)
    pub fn add_instance(mut self, manager: &'a BlockManager<M>, block_range: Range<usize>) -> Self {
        self.instances.push(InstancePopulationSpec {
            manager,
            block_range,
            fill_pattern: None,
        });
        self
    }

    /// Adds an instance with a fill pattern specification.
    ///
    /// Note: The fill pattern is stored but not automatically applied.
    /// Use the returned PopulatedInstances with a physical layer to fill blocks.
    pub fn add_instance_with_pattern(
        mut self,
        manager: &'a BlockManager<M>,
        block_range: Range<usize>,
        fill_pattern: FillPattern,
    ) -> Self {
        self.instances.push(InstancePopulationSpec {
            manager,
            block_range,
            fill_pattern: Some(fill_pattern),
        });
        self
    }

    /// Builds the populated instances.
    ///
    /// # Panics
    ///
    /// Panics if `total_blocks` or `block_size` are not set, or if any
    /// block range is out of bounds.
    pub fn build(self) -> Result<PopulatedInstances> {
        let total_blocks = self.total_blocks.expect("total_blocks is required");
        let block_size = self.block_size.expect("block_size is required");

        // Create the full token sequence
        let token_sequence = create_token_sequence(total_blocks, block_size, self.start_token);
        let full_blocks = token_sequence.blocks();

        // Generate all hashes from the sequence
        let all_hashes: Vec<SequenceHash> =
            full_blocks.iter().map(|b| b.kvbm_sequence_hash()).collect();

        // Populate each instance
        let mut instance_results = Vec::with_capacity(self.instances.len());
        for (idx, spec) in self.instances.into_iter().enumerate() {
            // Validate range
            if spec.block_range.end > total_blocks {
                anyhow::bail!(
                    "Instance {} block_range {:?} exceeds total_blocks {}",
                    idx,
                    spec.block_range,
                    total_blocks
                );
            }

            // Extract blocks for this instance
            let instance_blocks: Vec<_> = full_blocks[spec.block_range.clone()].to_vec();

            // Populate the manager
            let hashes = populate_manager_with_blocks(spec.manager, &instance_blocks)?;

            // Match to get block IDs
            let matched = spec.manager.match_blocks(&hashes);
            let block_ids: Vec<BlockId> = matched.into_iter().map(|b| b.block_id()).collect();

            instance_results.push(InstancePopulationResult {
                instance_index: idx,
                block_ids,
                hashes,
            });
        }

        Ok(PopulatedInstances {
            token_sequence,
            all_hashes,
            instance_results,
        })
    }
}

/// Convenience type alias for the builder.
pub type MultiInstancePopulator<'a, M> = MultiInstancePopulatorBuilder<'a, M>;

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
