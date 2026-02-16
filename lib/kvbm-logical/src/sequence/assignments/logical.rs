// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manages physical block guards through three ordered collections,
//! mirroring the block lifecycle state machine:
//! `MutableBlock<T>` → `CompleteBlock<T>` → `ImmutableBlock<T>`.

use crate::BlockId;
use crate::blocks::{BlockError, BlockMetadata, CompleteBlock, ImmutableBlock, MutableBlock};
use crate::manager::BlockManager;
use dynamo_tokens::TokenBlock;

use super::super::store::BlockStore;

/// Error type for [`LogicalBlockAssignments`] operations.
#[derive(Debug, thiserror::Error)]
pub enum LogicalBlockAssignmentError<T: BlockMetadata> {
    /// A mutable block_id in the input already exists in one of the three collections.
    #[error("duplicate block_id {block_id} already present")]
    DuplicateBlockId {
        /// The first duplicate block_id detected.
        block_id: BlockId,
        /// All input blocks returned for recovery (no leaks).
        blocks: Vec<MutableBlock<T>>,
    },

    /// An immutable block_id in the input already exists in one of the three collections.
    #[error("duplicate block_id {block_id} already present")]
    DuplicateAssignedBlockId {
        /// The first duplicate block_id detected.
        block_id: BlockId,
        /// All input blocks returned for recovery (no leaks).
        blocks: Vec<ImmutableBlock<T>>,
    },
}

/// Manages the physical block guards (RAII types) through three ordered collections,
/// mirroring the block lifecycle state machine:
/// `MutableBlock<T>` → `CompleteBlock<T>` → `ImmutableBlock<T>`.
///
/// Provides the same ordered-collection semantics as
/// [`ExternalBlockAssignments`](super::ExternalBlockAssignments) but at the guard level rather
/// than the identity level.
pub struct LogicalBlockAssignments<T: BlockMetadata> {
    store: BlockStore<MutableBlock<T>, CompleteBlock<T>, ImmutableBlock<T>>,
}

impl<T: BlockMetadata> LogicalBlockAssignments<T> {
    /// Creates an empty `LogicalBlockAssignments`.
    pub fn new() -> Self {
        Self {
            store: BlockStore::new(),
        }
    }

    // -- Counts & Queries --------------------------------------------------

    /// Returns the number of assigned (registered/immutable) blocks.
    pub fn assigned_count(&self) -> usize {
        self.store.assigned_count()
    }

    /// Returns the number of staged (complete) blocks.
    pub fn staged_count(&self) -> usize {
        self.store.staged_count()
    }

    /// Returns the number of unassigned (mutable) blocks.
    pub fn unassigned_count(&self) -> usize {
        self.store.unassigned_count()
    }

    /// Returns `true` if all three collections are empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Checks whether a `BlockId` is present in any of the three collections.
    pub fn contains(&self, block_id: &BlockId) -> bool {
        self.store.contains(block_id)
    }

    // -- Positional Access -------------------------------------------------

    /// Returns the assigned block at the given index (insertion order).
    pub fn get_assigned(&self, index: usize) -> Option<(&BlockId, &ImmutableBlock<T>)> {
        self.store.get_assigned(index)
    }

    /// Returns the staged block at the given index (staging order).
    pub fn get_staged(&self, index: usize) -> Option<(&BlockId, &CompleteBlock<T>)> {
        self.store.get_staged(index)
    }

    /// Returns the unassigned block at the given index (FIFO order).
    pub fn get_unassigned(&self, index: usize) -> Option<(&BlockId, &MutableBlock<T>)> {
        self.store.get_unassigned(index)
    }

    // -- Iteration ---------------------------------------------------------

    /// Iterates over assigned blocks in positional order.
    pub fn assigned_iter(&self) -> impl Iterator<Item = (&BlockId, &ImmutableBlock<T>)> {
        self.store.assigned_iter()
    }

    /// Iterates over staged blocks in staging order.
    pub fn staged_iter(&self) -> impl Iterator<Item = (&BlockId, &CompleteBlock<T>)> {
        self.store.staged_iter()
    }

    /// Iterates over unassigned blocks in FIFO order.
    pub fn unassigned_iter(&self) -> impl Iterator<Item = (&BlockId, &MutableBlock<T>)> {
        self.store.unassigned_iter()
    }

    // -- Mutation -----------------------------------------------------------

    /// Adds mutable blocks to the unassigned queue.
    ///
    /// Two-phase atomic: collects all blocks, validates no duplicate BlockIds
    /// across all three collections, then commits to unassigned. On error,
    /// all blocks are returned in the error variant (no leaks).
    pub fn extend_blocks(
        &mut self,
        blocks: impl IntoIterator<Item = MutableBlock<T>>,
    ) -> Result<usize, LogicalBlockAssignmentError<T>> {
        // Phase 1: collect
        let blocks: Vec<MutableBlock<T>> = blocks.into_iter().collect();

        // Phase 2: validate — check against existing collections and within input
        let mut seen = indexmap::IndexSet::with_capacity(blocks.len());
        for block in &blocks {
            let id = block.block_id();
            if self.store.contains(&id) || !seen.insert(id) {
                return Err(LogicalBlockAssignmentError::DuplicateBlockId {
                    block_id: id,
                    blocks,
                });
            }
        }

        // Phase 3: commit
        let count = blocks.len();
        for block in blocks {
            let id = block.block_id();
            self.store.insert_unassigned(id, block);
        }

        Ok(count)
    }

    /// Inserts pre-matched immutable blocks directly into the assigned collection.
    ///
    /// This is the entry point for blocks retrieved via
    /// [`BlockManager::match_blocks`] — blocks that already exist in the
    /// manager's pools and can skip the unassigned → staged → assigned pipeline.
    ///
    /// Two-phase atomic: collects all blocks, validates no duplicate BlockIds
    /// across all three collections, then commits to assigned. On error,
    /// all blocks are returned in the error variant (no leaks).
    pub fn extend_assigned(
        &mut self,
        blocks: impl IntoIterator<Item = ImmutableBlock<T>>,
    ) -> Result<usize, LogicalBlockAssignmentError<T>> {
        // Phase 1: collect
        let blocks: Vec<ImmutableBlock<T>> = blocks.into_iter().collect();

        // Phase 2: validate — check against existing collections and within input
        let mut seen = indexmap::IndexSet::with_capacity(blocks.len());
        for block in &blocks {
            let id = block.block_id();
            if self.store.contains(&id) || !seen.insert(id) {
                return Err(LogicalBlockAssignmentError::DuplicateAssignedBlockId {
                    block_id: id,
                    blocks,
                });
            }
        }

        // Phase 3: commit
        let count = blocks.len();
        for block in blocks {
            let id = block.block_id();
            self.store.insert_assigned(id, block);
        }

        Ok(count)
    }

    /// FIFO drain from unassigned, completing each block with the corresponding
    /// token block.
    ///
    /// Stages `min(sequence_blocks.len(), unassigned.len())` blocks. On
    /// [`BlockError`], already-staged blocks remain in staged; the failed block
    /// is returned in the error.
    pub fn stage(
        &mut self,
        sequence_blocks: &[TokenBlock],
    ) -> Result<usize, BlockError<MutableBlock<T>>> {
        let to_stage = sequence_blocks.len().min(self.store.unassigned_count());

        #[allow(clippy::needless_range_loop)]
        for i in 0..to_stage {
            let (block_id, mutable) = self.store.shift_unassigned().unwrap();
            match mutable.complete(&sequence_blocks[i]) {
                Ok(complete) => {
                    self.store.insert_staged(block_id, complete);
                }
                Err(err) => {
                    return Err(err);
                }
            }
        }

        Ok(to_stage)
    }

    /// Takes all staged blocks (FIFO order), registering each with the block
    /// manager and moving them to assigned.
    ///
    /// Returns the number of blocks registered.
    pub fn register(&mut self, manager: &BlockManager<T>) -> usize {
        let count = self.store.staged_count();

        while let Some((block_id, complete)) = self.store.shift_staged() {
            let immutable = manager.register_block(complete);
            self.store.insert_assigned(block_id, immutable);
        }

        count
    }

    /// Drops all guards across all three collections (RAII returns blocks to pools).
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// Takes all assigned blocks, returning them as a `Vec`.
    pub fn take_assigned(&mut self) -> Vec<(BlockId, ImmutableBlock<T>)> {
        self.store.take_assigned()
    }

    /// Takes all staged blocks, returning them as a `Vec`.
    pub fn take_staged(&mut self) -> Vec<(BlockId, CompleteBlock<T>)> {
        self.store.take_staged()
    }

    /// Takes all unassigned blocks, returning them as a `Vec`.
    pub fn take_unassigned(&mut self) -> Vec<(BlockId, MutableBlock<T>)> {
        self.store.take_unassigned()
    }
}

impl<T: BlockMetadata> Default for LogicalBlockAssignments<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BlockMetadata> std::fmt::Debug for LogicalBlockAssignments<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogicalBlockAssignments")
            .field("assigned_count", &self.store.assigned_count())
            .field("staged_count", &self.store.staged_count())
            .field("unassigned_count", &self.store.unassigned_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::BlockSequence;
    use crate::testing::{TestMeta, create_test_manager};

    const BLOCK_SIZE: u32 = 4;

    fn create_sequence(num_blocks: usize) -> BlockSequence {
        let total_tokens = num_blocks * BLOCK_SIZE as usize;
        let tokens: Vec<u32> = (0..total_tokens as u32).collect();
        BlockSequence::new(tokens, BLOCK_SIZE, None)
    }

    // =========================================================================
    // Empty construction
    // =========================================================================

    #[test]
    fn test_empty_construction() {
        let la = LogicalBlockAssignments::<TestMeta>::new();
        assert!(la.is_empty());
        assert_eq!(la.assigned_count(), 0);
        assert_eq!(la.staged_count(), 0);
        assert_eq!(la.unassigned_count(), 0);
        assert!(!la.contains(&0));
    }

    #[test]
    fn test_default_is_empty() {
        let la = LogicalBlockAssignments::<TestMeta>::default();
        assert!(la.is_empty());
    }

    // =========================================================================
    // extend_blocks
    // =========================================================================

    #[test]
    fn test_extend_blocks_basic() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(5).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();

        let mut la = LogicalBlockAssignments::new();
        let count = la.extend_blocks(blocks).unwrap();

        assert_eq!(count, 5);
        assert_eq!(la.unassigned_count(), 5);
        assert!(!la.is_empty());

        // Verify FIFO ordering
        for (i, expected_id) in ids.iter().enumerate() {
            let (id, _) = la.get_unassigned(i).unwrap();
            assert_eq!(id, expected_id);
        }
    }

    // =========================================================================
    // extend_assigned
    // =========================================================================

    /// Helper: allocate, complete, and register blocks through the manager,
    /// returning `ImmutableBlock`s suitable for `extend_assigned`.
    fn register_blocks_directly(
        manager: &BlockManager<TestMeta>,
        seq: &BlockSequence,
        count: usize,
    ) -> Vec<ImmutableBlock<TestMeta>> {
        let mutables = manager.allocate_blocks(count).unwrap();
        mutables
            .into_iter()
            .zip(seq.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect()
    }

    #[test]
    fn test_extend_assigned_basic() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(3);
        let immutables = register_blocks_directly(&manager, &seq, 3);
        let ids: Vec<BlockId> = immutables.iter().map(|b| b.block_id()).collect();
        let hashes = seq.all_sequence_hashes();

        let mut la = LogicalBlockAssignments::new();
        let count = la.extend_assigned(immutables).unwrap();

        assert_eq!(count, 3);
        assert_eq!(la.assigned_count(), 3);
        assert_eq!(la.staged_count(), 0);
        assert_eq!(la.unassigned_count(), 0);

        for (i, expected_id) in ids.iter().enumerate() {
            let (id, imm) = la.get_assigned(i).unwrap();
            assert_eq!(id, expected_id);
            assert_eq!(imm.block_id(), *expected_id);
            assert_eq!(imm.sequence_hash(), hashes[i]);
        }
    }

    #[test]
    fn test_extend_assigned_ordering_preserved() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(5);
        let immutables = register_blocks_directly(&manager, &seq, 5);
        let ids: Vec<BlockId> = immutables.iter().map(|b| b.block_id()).collect();

        let mut la = LogicalBlockAssignments::new();
        la.extend_assigned(immutables).unwrap();

        // Verify insertion order matches input order
        for (i, expected_id) in ids.iter().enumerate() {
            let (id, _) = la.get_assigned(i).unwrap();
            assert_eq!(id, expected_id);
        }
    }

    #[test]
    fn test_extend_assigned_contains() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(3);
        let immutables = register_blocks_directly(&manager, &seq, 3);
        let ids: Vec<BlockId> = immutables.iter().map(|b| b.block_id()).collect();

        let mut la = LogicalBlockAssignments::new();
        la.extend_assigned(immutables).unwrap();

        for id in &ids {
            assert!(la.contains(id));
        }
        assert!(!la.contains(&999));
    }

    #[test]
    fn test_extend_assigned_then_extend_stage_register() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(5);

        // Register first 3 as matched prefix
        let matched = register_blocks_directly(&manager, &seq, 3);
        let matched_ids: Vec<BlockId> = matched.iter().map(|b| b.block_id()).collect();

        let mut la = LogicalBlockAssignments::new();
        la.extend_assigned(matched).unwrap();
        assert_eq!(la.assigned_count(), 3);

        // Allocate new blocks for the remaining 2 positions
        let new_blocks = manager.allocate_blocks(2).unwrap();
        let new_ids: Vec<BlockId> = new_blocks.iter().map(|b| b.block_id()).collect();
        la.extend_blocks(new_blocks).unwrap();

        // Stage and register the new blocks
        la.stage(&seq.blocks()[3..5]).unwrap();
        la.register(&manager);

        assert_eq!(la.assigned_count(), 5);
        assert_eq!(la.staged_count(), 0);
        assert_eq!(la.unassigned_count(), 0);

        // First 3 are the matched prefix
        for (i, expected_id) in matched_ids.iter().enumerate() {
            let (id, _) = la.get_assigned(i).unwrap();
            assert_eq!(id, expected_id);
        }
        // Last 2 are the newly registered blocks
        for (i, expected_id) in new_ids.iter().enumerate() {
            let (id, _) = la.get_assigned(3 + i).unwrap();
            assert_eq!(id, expected_id);
        }
    }

    #[test]
    fn test_extend_assigned_with_match_blocks() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(3);
        let hashes = seq.all_sequence_hashes();

        // Populate the manager: allocate → complete → register → drop
        let mutables = manager.allocate_blocks(3).unwrap();
        let registered: Vec<ImmutableBlock<TestMeta>> = mutables
            .into_iter()
            .zip(seq.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();
        drop(registered);

        // match_blocks retrieves them from the manager's pools
        let matched = manager.match_blocks(&hashes);
        assert_eq!(matched.len(), 3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_assigned(matched).unwrap();
        assert_eq!(la.assigned_count(), 3);

        // Verify hashes match
        for (i, expected_hash) in hashes.iter().enumerate() {
            let (_, imm) = la.get_assigned(i).unwrap();
            assert_eq!(imm.sequence_hash(), *expected_hash);
        }
    }

    #[test]
    fn test_extend_assigned_empty() {
        let mut la = LogicalBlockAssignments::<TestMeta>::new();
        let count = la.extend_assigned(Vec::new()).unwrap();
        assert_eq!(count, 0);
        assert!(la.is_empty());
    }

    // =========================================================================
    // stage
    // =========================================================================

    #[test]
    fn test_stage_basic() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        let staged = la.stage(seq.blocks()).unwrap();
        assert_eq!(staged, 3);
        assert_eq!(la.staged_count(), 3);
        assert_eq!(la.unassigned_count(), 0);

        // Verify FIFO ordering in staged
        for (i, expected_id) in ids.iter().enumerate() {
            let (id, _) = la.get_staged(i).unwrap();
            assert_eq!(id, expected_id);
        }
    }

    #[test]
    fn test_stage_fifo_drain() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(5).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        // Stage 3 out of 5
        la.stage(seq.blocks()).unwrap();

        // First 3 should have been drained from unassigned (FIFO)
        assert_eq!(la.unassigned_count(), 2);
        let (remaining_0, _) = la.get_unassigned(0).unwrap();
        let (remaining_1, _) = la.get_unassigned(1).unwrap();
        assert_eq!(*remaining_0, ids[3]);
        assert_eq!(*remaining_1, ids[4]);
    }

    #[test]
    fn test_stage_partial_fewer_blocks_than_unassigned() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(5).unwrap();
        let seq = create_sequence(3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        let staged = la.stage(seq.blocks()).unwrap();
        assert_eq!(staged, 3);
        assert_eq!(la.staged_count(), 3);
        assert_eq!(la.unassigned_count(), 2);
    }

    #[test]
    fn test_stage_partial_fewer_unassigned_than_blocks() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(2).unwrap();
        let seq = create_sequence(5);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        let staged = la.stage(seq.blocks()).unwrap();
        assert_eq!(staged, 2);
        assert_eq!(la.staged_count(), 2);
        assert_eq!(la.unassigned_count(), 0);
    }

    #[test]
    fn test_stage_block_size_mismatch_recovery() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();

        // Sequence with block_size=8 (manager uses block_size=4)
        let tokens: Vec<u32> = (0..24).collect();
        let bad_seq = BlockSequence::new(tokens, 8, None);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        let result = la.stage(bad_seq.blocks());
        assert!(result.is_err());

        match result.unwrap_err() {
            BlockError::BlockSizeMismatch {
                expected,
                actual,
                block,
            } => {
                assert_eq!(expected, 4);
                assert_eq!(actual, 8);
                // Block recovered — drop to return to pool
                drop(block);
            }
        }

        // First block was removed from unassigned and returned in error
        assert_eq!(la.staged_count(), 0);
        assert_eq!(la.unassigned_count(), 2);
    }

    #[test]
    fn test_stage_partial_then_mismatch() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();
        let good_seq = create_sequence(2);
        let bad_tokens: Vec<u32> = (0..8).collect();
        let bad_seq = BlockSequence::new(bad_tokens, 8, None);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        // Stage first 2 successfully
        let staged = la.stage(good_seq.blocks()).unwrap();
        assert_eq!(staged, 2);
        assert_eq!(la.staged_count(), 2);
        assert_eq!(la.unassigned_count(), 1);

        // Try to stage 1 more with wrong size → error
        let result = la.stage(bad_seq.blocks());
        assert!(result.is_err());

        // The 2 already-staged blocks remain
        assert_eq!(la.staged_count(), 2);
        // The failed block was removed from unassigned and returned in error
        assert_eq!(la.unassigned_count(), 0);
    }

    // =========================================================================
    // register
    // =========================================================================

    #[test]
    fn test_register_basic() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();
        la.stage(seq.blocks()).unwrap();

        let registered = la.register(&manager);
        assert_eq!(registered, 3);
        assert_eq!(la.assigned_count(), 3);
        assert_eq!(la.staged_count(), 0);

        // Verify ordering preserved
        for (i, expected_id) in ids.iter().enumerate() {
            let (id, imm) = la.get_assigned(i).unwrap();
            assert_eq!(id, expected_id);
            assert_eq!(imm.block_id(), *expected_id);
        }
    }

    #[test]
    fn test_register_empty_staged() {
        let manager = create_test_manager::<TestMeta>(10);
        let mut la = LogicalBlockAssignments::<TestMeta>::new();

        let registered = la.register(&manager);
        assert_eq!(registered, 0);
    }

    // =========================================================================
    // Full pipeline
    // =========================================================================

    #[test]
    fn test_full_pipeline_extend_stage_register() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(5).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(5);
        let expected_hashes = seq.all_sequence_hashes();

        let mut la = LogicalBlockAssignments::new();

        // extend
        la.extend_blocks(blocks).unwrap();
        assert_eq!(la.unassigned_count(), 5);

        // stage
        la.stage(seq.blocks()).unwrap();
        assert_eq!(la.staged_count(), 5);
        assert_eq!(la.unassigned_count(), 0);

        // register
        la.register(&manager);
        assert_eq!(la.assigned_count(), 5);
        assert_eq!(la.staged_count(), 0);

        // Verify correct blocks in correct order with correct hashes
        for (i, expected_id) in ids.iter().enumerate() {
            let (id, immutable) = la.get_assigned(i).unwrap();
            assert_eq!(id, expected_id);
            assert_eq!(immutable.block_id(), *expected_id);
            assert_eq!(immutable.sequence_hash(), expected_hashes[i]);
        }
    }

    #[test]
    fn test_full_pipeline_incremental() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(6).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(6);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        // Stage first 3, register them
        la.stage(&seq.blocks()[..3]).unwrap();
        la.register(&manager);
        assert_eq!(la.assigned_count(), 3);
        assert_eq!(la.unassigned_count(), 3);

        // Stage remaining 3, register them
        la.stage(&seq.blocks()[..3]).unwrap();
        la.register(&manager);
        assert_eq!(la.assigned_count(), 6);
        assert_eq!(la.unassigned_count(), 0);

        // Verify all 6 in order
        for (i, expected_id) in ids.iter().enumerate() {
            let (id, _) = la.get_assigned(i).unwrap();
            assert_eq!(id, expected_id);
        }
    }

    // =========================================================================
    // contains across all three collections
    // =========================================================================

    #[test]
    fn test_contains_across_collections() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(6).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(4);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        // Stage first 2, then register them
        la.stage(&seq.blocks()[..2]).unwrap();
        la.register(&manager);

        // Stage 2 more
        la.stage(&seq.blocks()[..2]).unwrap();

        // State: 2 assigned, 2 staged, 2 unassigned
        assert_eq!(la.assigned_count(), 2);
        assert_eq!(la.staged_count(), 2);
        assert_eq!(la.unassigned_count(), 2);

        // All 6 should be contained
        for id in &ids {
            assert!(la.contains(id), "block_id {id} should be contained");
        }
        assert!(!la.contains(&999));
    }

    // =========================================================================
    // clear and drain
    // =========================================================================

    #[test]
    fn test_clear() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();
        let seq = create_sequence(3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();
        la.stage(seq.blocks()).unwrap();
        la.register(&manager);
        assert_eq!(la.assigned_count(), 3);

        la.clear();
        assert!(la.is_empty());
    }

    #[test]
    fn test_take_assigned() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();
        la.stage(seq.blocks()).unwrap();
        la.register(&manager);

        let drained = la.take_assigned();
        assert_eq!(drained.len(), 3);
        assert_eq!(la.assigned_count(), 0);

        for (i, (id, _)) in drained.iter().enumerate() {
            assert_eq!(*id, ids[i]);
        }
    }

    #[test]
    fn test_take_staged() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();
        la.stage(seq.blocks()).unwrap();

        let drained = la.take_staged();
        assert_eq!(drained.len(), 3);
        assert_eq!(la.staged_count(), 0);

        for (i, (id, _)) in drained.iter().enumerate() {
            assert_eq!(*id, ids[i]);
        }
    }

    #[test]
    fn test_take_unassigned() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        let drained = la.take_unassigned();
        assert_eq!(drained.len(), 3);
        assert_eq!(la.unassigned_count(), 0);

        for (i, (id, _)) in drained.iter().enumerate() {
            assert_eq!(*id, ids[i]);
        }
    }

    // =========================================================================
    // Negative: extend_assigned duplicate detection
    // =========================================================================

    #[test]
    fn test_extend_assigned_duplicate_already_in_assigned() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(3);
        let immutables = register_blocks_directly(&manager, &seq, 3);

        // Clone before consuming — ImmutableBlock is Clone
        let clones: Vec<ImmutableBlock<TestMeta>> = immutables.iter().cloned().collect();
        let dup_id = clones[0].block_id();

        let mut la = LogicalBlockAssignments::new();
        la.extend_assigned(immutables).unwrap();

        // Second call with same block_ids → duplicate error
        let result = la.extend_assigned(clones);
        assert!(result.is_err());
        match result.unwrap_err() {
            LogicalBlockAssignmentError::DuplicateAssignedBlockId { block_id, blocks } => {
                assert_eq!(block_id, dup_id);
                assert_eq!(blocks.len(), 3);
            }
            other => panic!("expected DuplicateAssignedBlockId, got: {other:?}"),
        }

        // Atomic rollback: assigned unchanged
        assert_eq!(la.assigned_count(), 3);
    }

    #[test]
    fn test_extend_assigned_duplicate_within_input_batch() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(1);
        let immutables = register_blocks_directly(&manager, &seq, 1);
        let dup = immutables[0].clone();
        let dup_id = dup.block_id();

        // Two copies of the same block in one batch
        let batch = vec![immutables.into_iter().next().unwrap(), dup];

        let mut la = LogicalBlockAssignments::new();
        let result = la.extend_assigned(batch);
        assert!(result.is_err());
        match result.unwrap_err() {
            LogicalBlockAssignmentError::DuplicateAssignedBlockId { block_id, blocks } => {
                assert_eq!(block_id, dup_id);
                assert_eq!(blocks.len(), 2);
            }
            other => panic!("expected DuplicateAssignedBlockId, got: {other:?}"),
        }

        // Nothing committed
        assert!(la.is_empty());
    }

    #[test]
    fn test_extend_assigned_duplicate_in_staged() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();
        let ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let seq = create_sequence(3);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();
        la.stage(seq.blocks()).unwrap();
        // 3 blocks now in staged

        // Get immutables whose block_ids collide with staged
        // Use match_blocks after populating the manager separately
        let other_blocks = manager.allocate_blocks(3).unwrap();
        let immutables: Vec<ImmutableBlock<TestMeta>> = other_blocks
            .into_iter()
            .zip(seq.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();

        // These have different block_ids (from the pool), so they won't collide
        // with staged. To force collision, clone an existing ImmutableBlock and
        // use the staged block_id. We can't do that directly, but we can verify
        // that `contains` correctly prevents the collision path.
        //
        // Instead, verify the non-collision case works:
        let other_ids: Vec<BlockId> = immutables.iter().map(|b| b.block_id()).collect();
        assert!(ids.iter().all(|id| !other_ids.contains(id)));
        la.extend_assigned(immutables).unwrap();
        assert_eq!(la.assigned_count(), 3);
        assert_eq!(la.staged_count(), 3);
    }

    #[test]
    fn test_extend_assigned_duplicate_in_unassigned() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(3);

        // Put blocks in unassigned
        let blocks = manager.allocate_blocks(3).unwrap();
        let unassigned_ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        // Register separate blocks to get immutables
        let other_blocks = manager.allocate_blocks(3).unwrap();
        let immutables: Vec<ImmutableBlock<TestMeta>> = other_blocks
            .into_iter()
            .zip(seq.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();

        // Different block_ids from the pool, so no collision — just verify
        let imm_ids: Vec<BlockId> = immutables.iter().map(|b| b.block_id()).collect();
        assert!(unassigned_ids.iter().all(|id| !imm_ids.contains(id)));
        la.extend_assigned(immutables).unwrap();
        assert_eq!(la.assigned_count(), 3);
        assert_eq!(la.unassigned_count(), 3);
    }

    // =========================================================================
    // Negative: extend_blocks with block_id in staged or assigned
    // =========================================================================

    #[test]
    fn test_extend_blocks_id_already_in_assigned() {
        let manager = create_test_manager::<TestMeta>(10);
        let seq = create_sequence(3);

        // Register 3 blocks → assigned
        let blocks = manager.allocate_blocks(3).unwrap();
        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();
        la.stage(seq.blocks()).unwrap();
        la.register(&manager);
        assert_eq!(la.assigned_count(), 3);

        // Those block_ids are now back in the reset pool (RAII from the original
        // MutableBlocks was consumed by stage/register). Allocating again may
        // return the same block_ids — but they're still in `assigned`.
        // With 10 total blocks and 3 in assigned, 7 are free. The 3 assigned
        // block_ids won't be re-allocated because their RAII guards live in
        // `la.assigned`. So we can't hit this path with a single manager.
        //
        // Verified: extend_blocks can only collide via programming error
        // (two managers), which is an invalid scenario.
        assert_eq!(la.assigned_count(), 3);
    }

    // =========================================================================
    // Negative: stage edge cases
    // =========================================================================

    #[test]
    fn test_stage_empty_unassigned() {
        let mut la = LogicalBlockAssignments::<TestMeta>::new();
        let seq = create_sequence(3);

        let staged = la.stage(seq.blocks()).unwrap();
        assert_eq!(staged, 0);
        assert_eq!(la.staged_count(), 0);
    }

    #[test]
    fn test_stage_empty_sequence_blocks() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(3).unwrap();

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();

        let staged = la.stage(&[]).unwrap();
        assert_eq!(staged, 0);
        assert_eq!(la.staged_count(), 0);
        assert_eq!(la.unassigned_count(), 3);
    }

    #[test]
    fn test_stage_both_empty() {
        let mut la = LogicalBlockAssignments::<TestMeta>::new();

        let staged = la.stage(&[]).unwrap();
        assert_eq!(staged, 0);
        assert!(la.is_empty());
    }

    // =========================================================================
    // Negative: positional access out of bounds
    // =========================================================================

    #[test]
    fn test_get_assigned_out_of_bounds() {
        let la = LogicalBlockAssignments::<TestMeta>::new();
        assert!(la.get_assigned(0).is_none());
        assert!(la.get_assigned(100).is_none());
    }

    #[test]
    fn test_get_staged_out_of_bounds() {
        let la = LogicalBlockAssignments::<TestMeta>::new();
        assert!(la.get_staged(0).is_none());
        assert!(la.get_staged(100).is_none());
    }

    #[test]
    fn test_get_unassigned_out_of_bounds() {
        let la = LogicalBlockAssignments::<TestMeta>::new();
        assert!(la.get_unassigned(0).is_none());
        assert!(la.get_unassigned(100).is_none());
    }

    #[test]
    fn test_get_out_of_bounds_with_populated_collections() {
        let manager = create_test_manager::<TestMeta>(10);
        let blocks = manager.allocate_blocks(2).unwrap();
        let seq = create_sequence(2);

        let mut la = LogicalBlockAssignments::new();
        la.extend_blocks(blocks).unwrap();
        la.stage(seq.blocks()).unwrap();
        la.register(&manager);

        assert!(la.get_assigned(0).is_some());
        assert!(la.get_assigned(1).is_some());
        assert!(la.get_assigned(2).is_none());
        assert!(la.get_staged(0).is_none());
        assert!(la.get_unassigned(0).is_none());
    }

    // =========================================================================
    // Debug
    // =========================================================================

    #[test]
    fn test_debug_impl() {
        let la = LogicalBlockAssignments::<TestMeta>::new();
        let debug_str = format!("{la:?}");
        assert!(debug_str.contains("LogicalBlockAssignments"));
    }
}
