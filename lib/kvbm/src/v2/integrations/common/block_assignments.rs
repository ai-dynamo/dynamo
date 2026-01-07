// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block assignment traits and operations.
//!
//! This module provides a trait-based abstraction for managing the assignment of
//! physical blocks to logical token blocks (identified by sequence hashes).
//!
//! # Design
//!
//! The traits abstract over two different storage strategies:
//!
//! - **Connector** (ID-based): Stores `BlockId` and `(SequenceHash, BlockId)` tuples.
//!   The connector doesn't own blocks - vLLM owns them.
//!
//! - **Scheduler** (RAII blocks): Stores `MutableBlock<G1>` and `ImmutableBlock<G1>`.
//!   The scheduler owns blocks via RAII guards.
//!
//! Core algorithms like `apply_new_blocks`, `filter_block_ids`, and `transition_with`
//! are implemented once via blanket implementations on `BlockAssignmentOps`.

use std::ops::Range;

use crate::v2::{BlockId, SequenceHash};

// Re-export the existing trait for sequence hash access
pub use crate::v2::KvbmSequenceHashProvider;

// ============================================================================
// Core Traits
// ============================================================================

/// Trait representing an assigned block with its sequence hash.
///
/// This is implemented by types that represent blocks which have been
/// assigned to a specific position in the token sequence.
pub trait AssignedBlock {
    /// Get the physical block ID.
    fn block_id(&self) -> BlockId;

    /// Get the sequence hash identifying the logical token block.
    fn sequence_hash(&self) -> SequenceHash;
}

/// Trait representing an unassigned block (pending assignment to a sequence hash).
///
/// This is implemented by types that represent blocks which have been
/// allocated but not yet paired with a logical token block.
pub trait UnassignedBlock {
    /// Get the physical block ID.
    fn block_id(&self) -> BlockId;
}

// ============================================================================
// Storage Trait
// ============================================================================

/// Core trait for block assignment storage.
///
/// This trait abstracts over the different storage strategies used by
/// the connector (stores IDs) vs scheduler (stores RAII blocks).
///
/// Implementations provide the underlying storage containers and basic
/// operations. The `BlockAssignmentOps` trait then provides algorithms
/// via blanket implementation.
pub trait BlockAssignmentStorage {
    /// Type for blocks that haven't been assigned to a sequence hash yet.
    type Unassigned: UnassignedBlock;

    /// Type for blocks that have been assigned to a sequence hash.
    type Assigned: AssignedBlock;

    /// Get the assigned blocks.
    fn assigned(&self) -> &[Self::Assigned];

    /// Get the unassigned blocks.
    fn unassigned(&self) -> &[Self::Unassigned];

    /// Get mutable access to unassigned blocks.
    fn unassigned_mut(&mut self) -> &mut Vec<Self::Unassigned>;

    /// Extend the assigned blocks collection.
    fn extend_assigned(&mut self, blocks: impl IntoIterator<Item = Self::Assigned>);

    /// Take all unassigned blocks, leaving the collection empty.
    fn take_unassigned(&mut self) -> Vec<Self::Unassigned>;

    /// Take up to `count` unassigned blocks.
    fn take_unassigned_n(&mut self, count: usize) -> Vec<Self::Unassigned> {
        let unassigned = self.unassigned_mut();
        let take_count = count.min(unassigned.len());
        unassigned.drain(0..take_count).collect()
    }

    /// Extend the unassigned blocks collection.
    fn extend_unassigned(&mut self, blocks: impl IntoIterator<Item = Self::Unassigned>);

    /// Clear all blocks (both assigned and unassigned).
    fn clear(&mut self);

    // ========================================================================
    // Default implementations for common accessors
    // ========================================================================

    /// Number of assigned blocks.
    fn num_assigned(&self) -> usize {
        self.assigned().len()
    }

    /// Number of unassigned blocks.
    fn num_unassigned(&self) -> usize {
        self.unassigned().len()
    }

    /// Total number of blocks (assigned + unassigned).
    fn total_blocks(&self) -> usize {
        self.num_assigned() + self.num_unassigned()
    }

    /// Check if there are no blocks.
    fn is_empty(&self) -> bool {
        self.assigned().is_empty() && self.unassigned().is_empty()
    }

    /// Get all block IDs (assigned first, then unassigned).
    fn all_block_ids(&self) -> Vec<BlockId> {
        let mut ids: Vec<BlockId> = self.assigned().iter().map(|b| b.block_id()).collect();
        ids.extend(self.unassigned().iter().map(|b| b.block_id()));
        ids
    }
}

// ============================================================================
// Operations Trait (with blanket implementation)
// ============================================================================

/// Trait for block assignment operations.
///
/// This trait provides the core algorithms for managing block assignments.
/// It has a blanket implementation for any type implementing `BlockAssignmentStorage`.
pub trait BlockAssignmentOps: BlockAssignmentStorage {
    /// Assign new blocks to sequence positions by pairing with sequence hashes.
    ///
    /// Blocks are paired with sequence hashes in order, starting from where
    /// previous assignments ended. Any excess blocks (more blocks than available
    /// sequence hashes) are stored as unassigned.
    ///
    /// # Arguments
    ///
    /// * `new_blocks` - New unassigned blocks to add and potentially assign
    /// * `sequence_hashes` - All sequence hashes from the token sequence
    ///
    /// # Returns
    ///
    /// The range of indices into the assigned collection for newly assigned blocks.
    ///
    /// # Type Parameters
    ///
    /// * `H` - Type implementing `KvbmSequenceHashProvider` (e.g., `TokenBlock`)
    fn apply_new_blocks<H: KvbmSequenceHashProvider>(
        &mut self,
        new_blocks: Vec<Self::Unassigned>,
        sequence_hashes: &[H],
    ) -> Range<usize>
    where
        Self::Assigned: From<(Self::Unassigned, SequenceHash)>,
    {
        let start_idx = self.num_assigned();

        // Add new blocks to unassigned first
        self.extend_unassigned(new_blocks);

        // Take all unassigned blocks
        let all_unassigned = self.take_unassigned();

        // Pair blocks with sequence hashes starting from where we left off
        let mut iter = all_unassigned.into_iter();
        let newly_assigned: Vec<Self::Assigned> = sequence_hashes
            .iter()
            .skip(start_idx)
            .zip(&mut iter)
            .map(|(hash_source, block)| {
                Self::Assigned::from((block, hash_source.kvbm_sequence_hash()))
            })
            .collect();

        self.extend_assigned(newly_assigned);

        // Remaining blocks go back to unassigned
        self.extend_unassigned(iter);

        let end_idx = self.num_assigned();
        start_idx..end_idx
    }

    /// Transition unassigned blocks to assigned using a closure.
    ///
    /// The closure captures any external dependencies needed for the transition
    /// (e.g., `KVCacheManager` for the scheduler). On success, blocks are moved
    /// to assigned. On failure, blocks are returned to unassigned.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of unassigned blocks to transition
    /// * `transition` - Closure that performs the transition
    ///
    /// # Returns
    ///
    /// * `Ok(n)` - Number of blocks successfully transitioned
    /// * `Err(e)` - Error from the transition closure; blocks returned to unassigned
    ///
    /// # Example (Scheduler)
    ///
    /// ```ignore
    /// let token_blocks = request.get_token_blocks_range(start..end);
    /// request.block_state.transition_with(count, |mutable_blocks| {
    ///     kv_cache.complete_and_register(mutable_blocks, token_blocks)
    ///         .map_err(|returned| (returned, anyhow::anyhow!("Registration failed")))
    /// })?;
    /// ```
    fn transition_with<F, E>(&mut self, count: usize, transition: F) -> Result<usize, E>
    where
        F: FnOnce(Vec<Self::Unassigned>) -> Result<Vec<Self::Assigned>, (Vec<Self::Unassigned>, E)>,
    {
        let pending = self.take_unassigned_n(count);
        if pending.is_empty() {
            return Ok(0);
        }

        match transition(pending) {
            Ok(assigned) => {
                let n = assigned.len();
                self.extend_assigned(assigned);
                Ok(n)
            }
            Err((returned, error)) => {
                self.extend_unassigned(returned);
                Err(error)
            }
        }
    }

    /// Filter block IDs to find only those not yet known (assigned or unassigned).
    ///
    /// Validates that the prefix of `all_block_ids` matches the known blocks
    /// (assigned first, then unassigned). Returns the suffix containing unknown blocks.
    ///
    /// # Arguments
    ///
    /// * `all_block_ids` - Complete list of block IDs from the scheduler, in order
    ///
    /// # Returns
    ///
    /// Block IDs that are not yet known (the suffix after assigned + unassigned).
    ///
    /// # Panics
    ///
    /// Panics if the prefix doesn't match known blocks (indicates a bug).
    fn filter_block_ids(&self, all_block_ids: Vec<BlockId>) -> Vec<BlockId> {
        let num_assigned = self.num_assigned();
        let num_unassigned = self.num_unassigned();
        let num_known = num_assigned + num_unassigned;

        if num_known == 0 {
            return all_block_ids;
        }

        assert!(
            all_block_ids.len() >= num_known,
            "all_block_ids length ({}) < known blocks (assigned={} + unassigned={})",
            all_block_ids.len(),
            num_assigned,
            num_unassigned
        );

        // Validate assigned prefix
        for (i, (assigned, provided)) in self
            .assigned()
            .iter()
            .map(|b| b.block_id())
            .zip(all_block_ids.iter())
            .enumerate()
        {
            assert_eq!(
                assigned, *provided,
                "Assigned block ID mismatch at index {}: {} != {}",
                i, assigned, provided
            );
        }

        // Validate unassigned portion
        for (i, (unassigned, provided)) in self
            .unassigned()
            .iter()
            .map(|b| b.block_id())
            .zip(all_block_ids.iter().skip(num_assigned))
            .enumerate()
        {
            assert_eq!(
                unassigned, *provided,
                "Unassigned block ID mismatch at index {}: {} != {}",
                i, unassigned, provided
            );
        }

        all_block_ids.into_iter().skip(num_known).collect()
    }

    /// Get block mappings ready for offload based on token evaluation progress.
    ///
    /// Returns `(BlockId, SequenceHash)` pairs for blocks that will complete
    /// after scheduling `num_scheduled_tokens`.
    ///
    /// # Arguments
    ///
    /// * `evaluated_blocks` - Number of blocks already evaluated for offload
    /// * `evaluated_tokens` - Number of tokens already evaluated
    /// * `num_scheduled_tokens` - Tokens being scheduled this iteration
    /// * `block_size` - Number of tokens per block
    ///
    /// # Returns
    ///
    /// Block mappings for newly completed blocks.
    fn get_next_block_mappings(
        &self,
        evaluated_blocks: usize,
        evaluated_tokens: usize,
        num_scheduled_tokens: usize,
        block_size: usize,
    ) -> Vec<(BlockId, SequenceHash)> {
        let num_blocks_after_evaluation = (evaluated_tokens + num_scheduled_tokens) / block_size;
        let new_blocks_to_evaluate = num_blocks_after_evaluation.saturating_sub(evaluated_blocks);

        self.assigned()
            .iter()
            .skip(evaluated_blocks)
            .take(new_blocks_to_evaluate)
            .map(|b| (b.block_id(), b.sequence_hash()))
            .collect()
    }
}

// Blanket implementation: any type implementing Storage gets Ops for free
impl<T: BlockAssignmentStorage> BlockAssignmentOps for T {}

// ============================================================================
// Convenience implementations for common types
// ============================================================================

/// Implementation of `UnassignedBlock` for raw `BlockId`.
///
/// Used by the connector which stores only IDs (vLLM owns the actual blocks).
impl UnassignedBlock for BlockId {
    fn block_id(&self) -> BlockId {
        *self
    }
}

// ============================================================================
// Implementations for RAII block types (Scheduler)
// ============================================================================

use crate::v2::logical::blocks::{BlockMetadata, ImmutableBlock, MutableBlock};

/// Implementation of `UnassignedBlock` for `MutableBlock<T>`.
///
/// MutableBlocks are blocks in Reset state waiting to be populated with KV data.
impl<T: BlockMetadata> UnassignedBlock for MutableBlock<T> {
    fn block_id(&self) -> BlockId {
        self.block_id()
    }
}

/// Implementation of `AssignedBlock` for `ImmutableBlock<T>`.
///
/// ImmutableBlocks are registered blocks that already have their KV data
/// computed and their sequence hash assigned.
impl<T: BlockMetadata> AssignedBlock for ImmutableBlock<T> {
    fn block_id(&self) -> BlockId {
        self.block_id()
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash()
    }
}

/// Assigned block type for connector: stores (SequenceHash, BlockId) tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssignedBlockId {
    sequence_hash: SequenceHash,
    block_id: BlockId,
}

impl AssignedBlockId {
    /// Create a new assigned block ID.
    pub fn new(sequence_hash: SequenceHash, block_id: BlockId) -> Self {
        Self {
            sequence_hash,
            block_id,
        }
    }
}

impl AssignedBlock for AssignedBlockId {
    fn block_id(&self) -> BlockId {
        self.block_id
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }
}

impl From<(BlockId, SequenceHash)> for AssignedBlockId {
    fn from((block_id, sequence_hash): (BlockId, SequenceHash)) -> Self {
        Self::new(sequence_hash, block_id)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test implementation using BlockId for both types
    #[derive(Debug, Default)]
    struct TestBlockAssignments {
        assigned: Vec<AssignedBlockId>,
        unassigned: Vec<BlockId>,
    }

    impl BlockAssignmentStorage for TestBlockAssignments {
        type Unassigned = BlockId;
        type Assigned = AssignedBlockId;

        fn assigned(&self) -> &[Self::Assigned] {
            &self.assigned
        }

        fn unassigned(&self) -> &[Self::Unassigned] {
            &self.unassigned
        }

        fn unassigned_mut(&mut self) -> &mut Vec<Self::Unassigned> {
            &mut self.unassigned
        }

        fn extend_assigned(&mut self, blocks: impl IntoIterator<Item = Self::Assigned>) {
            self.assigned.extend(blocks);
        }

        fn take_unassigned(&mut self) -> Vec<Self::Unassigned> {
            std::mem::take(&mut self.unassigned)
        }

        fn extend_unassigned(&mut self, blocks: impl IntoIterator<Item = Self::Unassigned>) {
            self.unassigned.extend(blocks);
        }

        fn clear(&mut self) {
            self.assigned.clear();
            self.unassigned.clear();
        }
    }

    /// Mock hash source for testing
    struct MockHashSource(SequenceHash);

    impl KvbmSequenceHashProvider for MockHashSource {
        fn kvbm_sequence_hash(&self) -> SequenceHash {
            self.0
        }
    }

    /// Create a test sequence hash.
    ///
    /// Uses PositionalLineageHash::new() with deterministic values based on index.
    fn make_test_hash(index: usize) -> SequenceHash {
        // Use index as both the sequence hash and position for simplicity
        let current_seq_hash = index as u64;
        let parent_seq_hash = if index == 0 {
            None
        } else {
            Some((index - 1) as u64)
        };
        SequenceHash::new(current_seq_hash, parent_seq_hash, index as u64)
    }

    fn make_hashes(count: usize) -> Vec<MockHashSource> {
        (0..count).map(|i| MockHashSource(make_test_hash(i))).collect()
    }

    #[test]
    fn test_empty_storage() {
        let storage = TestBlockAssignments::default();
        assert!(storage.is_empty());
        assert_eq!(storage.num_assigned(), 0);
        assert_eq!(storage.num_unassigned(), 0);
        assert_eq!(storage.total_blocks(), 0);
    }

    #[test]
    fn test_apply_new_blocks_exact_match() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(3);
        let blocks: Vec<BlockId> = vec![100, 200, 300];

        let range = storage.apply_new_blocks(blocks, &hashes);

        assert_eq!(range, 0..3);
        assert_eq!(storage.num_assigned(), 3);
        assert_eq!(storage.num_unassigned(), 0);

        // Verify assignments
        assert_eq!(storage.assigned[0].block_id(), 100);
        assert_eq!(storage.assigned[1].block_id(), 200);
        assert_eq!(storage.assigned[2].block_id(), 300);
    }

    #[test]
    fn test_apply_new_blocks_excess() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(2);
        let blocks: Vec<BlockId> = vec![100, 200, 300, 400];

        let range = storage.apply_new_blocks(blocks, &hashes);

        assert_eq!(range, 0..2);
        assert_eq!(storage.num_assigned(), 2);
        assert_eq!(storage.num_unassigned(), 2);
        assert_eq!(storage.unassigned, vec![300, 400]);
    }

    #[test]
    fn test_apply_new_blocks_incremental() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(4);

        // First batch: 2 blocks
        let range1 = storage.apply_new_blocks(vec![100, 200], &hashes);
        assert_eq!(range1, 0..2);
        assert_eq!(storage.num_assigned(), 2);

        // Second batch: 2 more blocks
        let range2 = storage.apply_new_blocks(vec![300, 400], &hashes);
        assert_eq!(range2, 2..4);
        assert_eq!(storage.num_assigned(), 4);
    }

    #[test]
    fn test_apply_new_blocks_uses_unassigned_first() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(2);

        // First call: 3 blocks for 2 hashes -> 1 excess
        storage.apply_new_blocks(vec![100, 200, 300], &hashes);
        assert_eq!(storage.num_assigned(), 2);
        assert_eq!(storage.unassigned, vec![300]);

        // Add more hashes
        let more_hashes = make_hashes(4);

        // Second call: 1 new block, but unassigned block (300) goes first
        let range = storage.apply_new_blocks(vec![400], &more_hashes);
        assert_eq!(range, 2..4);
        assert_eq!(storage.num_assigned(), 4);
        assert_eq!(storage.assigned[2].block_id(), 300); // From unassigned
        assert_eq!(storage.assigned[3].block_id(), 400); // New
    }

    #[test]
    fn test_filter_block_ids() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(3);
        storage.apply_new_blocks(vec![100, 200, 300, 400], &hashes);

        // 3 assigned, 1 unassigned
        let new_ids = storage.filter_block_ids(vec![100, 200, 300, 400, 500, 600]);
        assert_eq!(new_ids, vec![500, 600]);
    }

    #[test]
    #[should_panic(expected = "Assigned block ID mismatch")]
    fn test_filter_block_ids_mismatch_panics() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(2);
        storage.apply_new_blocks(vec![100, 200], &hashes);

        // Wrong prefix - should panic
        storage.filter_block_ids(vec![999, 200, 300]);
    }

    #[test]
    fn test_transition_with_success() {
        let mut storage = TestBlockAssignments::default();
        storage.extend_unassigned(vec![100, 200, 300]);

        let result: Result<usize, &str> = storage.transition_with(2, |blocks| {
            let assigned: Vec<AssignedBlockId> = blocks
                .into_iter()
                .enumerate()
                .map(|(i, id)| AssignedBlockId::new(make_test_hash(i), id))
                .collect();
            Ok(assigned)
        });

        assert_eq!(result, Ok(2));
        assert_eq!(storage.num_assigned(), 2);
        assert_eq!(storage.num_unassigned(), 1);
        assert_eq!(storage.unassigned[0], 300);
    }

    #[test]
    fn test_transition_with_failure() {
        let mut storage = TestBlockAssignments::default();
        storage.extend_unassigned(vec![100, 200, 300]);

        let result: Result<usize, &str> = storage.transition_with(2, |blocks| {
            // Fail and return blocks
            Err((blocks, "transition failed"))
        });

        assert_eq!(result, Err("transition failed"));
        // Blocks should be returned to unassigned
        assert_eq!(storage.num_assigned(), 0);
        assert_eq!(storage.num_unassigned(), 3);
    }

    #[test]
    fn test_get_next_block_mappings() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(4);
        storage.apply_new_blocks(vec![100, 200, 300, 400], &hashes);

        // Block size 16, 32 evaluated tokens = 2 blocks
        // 16 scheduled tokens = 1 more block
        let mappings = storage.get_next_block_mappings(2, 32, 16, 16);

        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].0, 300); // Block ID at index 2
    }

    #[test]
    fn test_all_block_ids() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(2);
        storage.apply_new_blocks(vec![100, 200, 300, 400], &hashes);

        let ids = storage.all_block_ids();
        assert_eq!(ids, vec![100, 200, 300, 400]);
    }

    #[test]
    fn test_clear() {
        let mut storage = TestBlockAssignments::default();
        let hashes = make_hashes(2);
        storage.apply_new_blocks(vec![100, 200, 300], &hashes);

        assert!(!storage.is_empty());
        storage.clear();
        assert!(storage.is_empty());
    }
}
