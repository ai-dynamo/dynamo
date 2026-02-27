// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BlockManager testing utilities.
//!
//! Contains core manager/registry builders, population helpers, and the
//! `MultiInstancePopulator` which bridges logical and physical layers.
//!
//! Note: Due to version boundaries between workspace crates and git-sourced
//! kvbm-logical/kvbm-physical, these utilities use workspace-local types directly
//! rather than re-exporting from kvbm-logical::testing.

use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;

use kvbm_logical::{
    blocks::{BlockMetadata, BlockRegistry},
    events::EventsManager,
    manager::{BlockManager, FrequencyTrackingCapacity},
};

use crate::{BlockId, SequenceHash};

use kvbm_logical::KvbmSequenceHashProvider;
use kvbm_common::tokens::TokenBlockSequence;
use kvbm_physical::transfer::FillPattern;

use super::token_blocks;

/// Builder for creating test BlockRegistry with optional events integration.
#[derive(Default)]
pub struct TestRegistryBuilder {
    events_manager: Option<Arc<EventsManager>>,
    frequency_tracking: FrequencyTrackingCapacity,
}

impl TestRegistryBuilder {
    pub fn new() -> Self {
        Self {
            events_manager: None,
            frequency_tracking: FrequencyTrackingCapacity::Medium,
        }
    }

    pub fn events_manager(mut self, manager: Arc<EventsManager>) -> Self {
        self.events_manager = Some(manager);
        self
    }

    pub fn frequency_tracking(mut self, capacity: FrequencyTrackingCapacity) -> Self {
        self.frequency_tracking = capacity;
        self
    }

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

    pub fn block_count(mut self, count: usize) -> Self {
        self.block_count = Some(count);
        self
    }

    pub fn block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size);
        self
    }

    pub fn registry(mut self, registry: BlockRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    pub fn events_manager(mut self, manager: Arc<EventsManager>) -> Self {
        self.events_manager = Some(manager);
        self
    }

    pub fn frequency_tracking(mut self, capacity: FrequencyTrackingCapacity) -> Self {
        self.frequency_tracking = capacity;
        self
    }

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
pub fn populate_manager_with_blocks<T: BlockMetadata>(
    manager: &BlockManager<T>,
    token_blocks: &[kvbm_common::tokens::TokenBlock],
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
    drop(immutable_blocks);

    Ok(seq_hashes)
}

/// Quick setup: create manager and populate with sequential token blocks.
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

/// Specification for a single instance's population.
pub struct InstancePopulationSpec<'a, M: BlockMetadata> {
    pub manager: &'a BlockManager<M>,
    pub block_range: Range<usize>,
    pub fill_pattern: Option<FillPattern>,
}

/// Result of populating a single instance.
pub struct InstancePopulationResult {
    pub instance_index: usize,
    pub block_ids: Vec<BlockId>,
    pub hashes: Vec<SequenceHash>,
}

/// Results from populating multiple instances.
pub struct PopulatedInstances {
    token_sequence: TokenBlockSequence,
    all_hashes: Vec<SequenceHash>,
    instance_results: Vec<InstancePopulationResult>,
}

impl PopulatedInstances {
    pub fn all_hashes(&self) -> &[SequenceHash] {
        &self.all_hashes
    }

    pub fn token_sequence(&self) -> &TokenBlockSequence {
        &self.token_sequence
    }

    pub fn instance_block_ids(&self, instance_index: usize) -> Option<&[BlockId]> {
        self.instance_results
            .get(instance_index)
            .map(|r| r.block_ids.as_slice())
    }

    pub fn instance_hashes(&self, instance_index: usize) -> Option<&[SequenceHash]> {
        self.instance_results
            .get(instance_index)
            .map(|r| r.hashes.as_slice())
    }

    pub fn instance_count(&self) -> usize {
        self.instance_results.len()
    }

    pub fn instance_results(&self) -> &[InstancePopulationResult] {
        &self.instance_results
    }
}

/// Builder for populating multiple instances with blocks from a shared token sequence.
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
    pub fn new() -> Self {
        Self {
            total_blocks: None,
            block_size: None,
            start_token: 0,
            instances: Vec::new(),
        }
    }

    pub fn total_blocks(mut self, count: usize) -> Self {
        self.total_blocks = Some(count);
        self
    }

    pub fn block_size(mut self, size: usize) -> Self {
        self.block_size = Some(size);
        self
    }

    pub fn start_token(mut self, token: u32) -> Self {
        self.start_token = token;
        self
    }

    pub fn add_instance(mut self, manager: &'a BlockManager<M>, block_range: Range<usize>) -> Self {
        self.instances.push(InstancePopulationSpec {
            manager,
            block_range,
            fill_pattern: None,
        });
        self
    }

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

    pub fn build(self) -> Result<PopulatedInstances> {
        let total_blocks = self.total_blocks.expect("total_blocks is required");
        let block_size = self.block_size.expect("block_size is required");

        let token_sequence =
            token_blocks::create_token_sequence(total_blocks, block_size, self.start_token);
        let full_blocks = token_sequence.blocks();

        let all_hashes: Vec<SequenceHash> =
            full_blocks.iter().map(|b| b.kvbm_sequence_hash()).collect();

        let mut instance_results = Vec::with_capacity(self.instances.len());
        for (idx, spec) in self.instances.into_iter().enumerate() {
            if spec.block_range.end > total_blocks {
                anyhow::bail!(
                    "Instance {} block_range {:?} exceeds total_blocks {}",
                    idx,
                    spec.block_range,
                    total_blocks
                );
            }

            let instance_blocks: Vec<_> = full_blocks[spec.block_range.clone()].to_vec();
            let hashes = populate_manager_with_blocks(spec.manager, &instance_blocks)?;

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

        let matched = manager.match_blocks(&hashes);
        assert_eq!(matched.len(), 32);
    }
}
