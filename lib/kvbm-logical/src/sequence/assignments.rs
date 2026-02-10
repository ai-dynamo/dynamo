// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use crate::{BlockId, KvbmSequenceHashProvider, SequenceHash};
use dynamo_tokens::TokenBlock;

use super::BlockSequenceError;
use super::lifecycle_store::LifecycleStore;

/// Per-tier block_id tracking with an offset into the sequence.
///
/// Maintains an ordered mapping of `BlockId` → `SequenceHash` for assigned blocks,
/// a staging area for blocks whose hashes have been computed but not yet committed,
/// plus a FIFO queue of block_ids waiting for assignment. Index `i` in the assigned
/// map corresponds to sequence position `offset + i`.
///
/// The three-phase lifecycle is:
/// - **Unassigned** — block_ids queued for assignment (no hash yet).
/// - **Staged** — block_ids paired with their `SequenceHash` but not yet committed.
/// - **Assigned** — committed `BlockId → SequenceHash` pairs in positional order.
///
/// Multiple `BlockAssignments` instances can operate on the same `&[TokenBlock]` at
/// different offsets (multi-tier).
pub struct BlockAssignments {
    store: LifecycleStore<(), SequenceHash, SequenceHash>,

    /// Starting position in the sequence. Assignments begin at this position.
    offset: usize,
}

impl std::fmt::Debug for BlockAssignments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockAssignments")
            .field("assigned_count", &self.store.assigned_count())
            .field("staged_count", &self.store.staged_count())
            .field("unassigned_count", &self.store.unassigned_count())
            .field("offset", &self.offset)
            .finish()
    }
}

impl BlockAssignments {
    /// Creates a new `BlockAssignments` starting at the given offset.
    pub fn new(offset: usize) -> Self {
        Self {
            store: LifecycleStore::new(),
            offset,
        }
    }

    /// Returns the starting position in the sequence.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Checks whether a block_id is known (assigned, staged, or unassigned).
    pub fn contains(&self, block_id: &BlockId) -> bool {
        self.store.contains(block_id)
    }

    /// Positional access: returns `(BlockId, SequenceHash)` at the given index
    /// (relative to offset) in the assigned collection.
    pub fn get_assigned(&self, index: usize) -> Option<(BlockId, SequenceHash)> {
        self.store
            .get_assigned(index)
            .map(|(&id, &hash)| (id, hash))
    }

    /// Returns the number of assigned blocks.
    pub fn assigned_count(&self) -> usize {
        self.store.assigned_count()
    }

    /// Returns the number of staged blocks.
    pub fn staged_count(&self) -> usize {
        self.store.staged_count()
    }

    /// Returns the number of unassigned (pending) block_ids.
    pub fn unassigned_count(&self) -> usize {
        self.store.unassigned_count()
    }

    /// Positional access: returns `(BlockId, SequenceHash)` at the given index
    /// (relative to the start of staged) in the staged collection.
    pub fn get_staged(&self, index: usize) -> Option<(BlockId, SequenceHash)> {
        self.store
            .get_staged(index)
            .map(|(&id, &hash)| (id, hash))
    }

    /// Iterates over assigned blocks in positional order, yielding `(BlockId, SequenceHash)`.
    pub fn assigned_iter(&self) -> impl Iterator<Item = (BlockId, SequenceHash)> + '_ {
        self.store.assigned_iter().map(|(&id, &hash)| (id, hash))
    }

    /// Iterates over staged blocks in staging order, yielding `(BlockId, SequenceHash)`.
    pub fn staged_iter(&self) -> impl Iterator<Item = (BlockId, SequenceHash)> + '_ {
        self.store.staged_iter().map(|(&id, &hash)| (id, hash))
    }

    /// Iterates over unassigned block_ids in FIFO order.
    pub fn unassigned_iter(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.store.unassigned_iter().map(|(&id, _)| id)
    }

    /// Clears all assigned, staged, and unassigned blocks, preserving the offset.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// Takes all staged blocks, returning them as a `Vec`.
    pub fn take_staged(&mut self) -> Vec<(BlockId, SequenceHash)> {
        self.store.take_staged()
    }

    /// Returns the next sequence position to be assigned:
    /// `offset + assigned_count + staged_count`.
    pub fn next_position(&self) -> usize {
        self.offset + self.store.assigned_count() + self.store.staged_count()
    }

    /// Absolute position range of assigned blocks: `offset..offset + assigned_count`.
    pub fn assigned_positions(&self) -> Range<usize> {
        self.offset..self.offset + self.store.assigned_count()
    }

    /// Absolute position range of staged blocks:
    /// `offset + assigned_count .. offset + assigned_count + staged_count`.
    pub fn staged_positions(&self) -> Range<usize> {
        let start = self.offset + self.store.assigned_count();
        start..start + self.store.staged_count()
    }

    /// Get the assigned `(BlockId, SequenceHash)` at an absolute sequence position.
    ///
    /// Returns `None` if `abs_pos` is outside [`assigned_positions()`](Self::assigned_positions).
    pub fn get_at_position(&self, abs_pos: usize) -> Option<(BlockId, SequenceHash)> {
        let relative = abs_pos.checked_sub(self.offset)?;
        self.get_assigned(relative)
    }

    /// Absolute position range that pending (unassigned) blocks will occupy once
    /// flushed: `next_position()..next_position() + unassigned_count()`.
    pub fn pending_positions(&self) -> Range<usize> {
        let start = self.next_position();
        start..start + self.store.unassigned_count()
    }

    /// Get the pending `BlockId` at an absolute sequence position (FIFO order).
    ///
    /// Position `next_position()` maps to the first unassigned block,
    /// `next_position() + 1` to the second, etc.
    /// Returns `None` if `abs_pos` is outside [`pending_positions()`](Self::pending_positions).
    pub fn get_pending_at_position(&self, abs_pos: usize) -> Option<BlockId> {
        let start = self.next_position();
        let relative = abs_pos.checked_sub(start)?;
        self.store
            .get_unassigned(relative)
            .map(|(&id, _)| id)
    }

    /// Add new block_ids to the unassigned queue.
    ///
    /// `block_ids` is the **full, ordered** list of block IDs allocated to this
    /// assignment set. Known IDs (already in assigned, staged, or unassigned) must
    /// form a contiguous prefix and are silently skipped. New IDs are appended to
    /// the unassigned FIFO queue.
    ///
    /// This method does **not** assign blocks — call
    /// [`assign_pending`](Self::assign_pending) to pair unassigned IDs with
    /// available sequence blocks.
    ///
    /// # Block ID rules
    ///
    /// The list is partitioned into a **known prefix** and a **new suffix**:
    ///
    /// - **Known prefix** — IDs already present in `assigned`, `staged`, or
    ///   `unassigned`. These are silently skipped. They must appear contiguously
    ///   at the front of the list; interleaving a known ID after an unknown one
    ///   is an [`OrderingViolation`](BlockSequenceError::OrderingViolation).
    /// - **New suffix** — IDs not yet seen. These are appended (in order) to
    ///   the unassigned FIFO queue.
    ///
    /// # Algorithm (two-phase, atomic)
    ///
    /// 1. **Validate & collect** — iterate `block_ids`. Known IDs must form a
    ///    contiguous prefix (skip them). Unknown IDs are collected into a temp
    ///    buffer. If a known ID appears after an unknown one →
    ///    `OrderingViolation` error. No state is mutated until validation passes.
    /// 2. **Commit** — push all new IDs to the unassigned queue.
    pub fn extend_block_ids(
        &mut self,
        block_ids: impl IntoIterator<Item = BlockId>,
    ) -> Result<(), BlockSequenceError> {
        // Phase 1: Validate & collect
        let mut new_ids = Vec::new();
        let mut first_new_index: Option<usize> = None;

        for (i, id) in block_ids.into_iter().enumerate() {
            if self.contains(&id) {
                // Known ID — must come before any new IDs
                if let Some(first_new) = first_new_index {
                    return Err(BlockSequenceError::OrderingViolation {
                        known_id: id,
                        new_id: new_ids[0],
                        known_index: i,
                        first_new_index: first_new,
                    });
                }
                // Skip — already known
            } else {
                // Unknown ID — collect
                if first_new_index.is_none() {
                    first_new_index = Some(i);
                }
                new_ids.push(id);
            }
        }

        // Phase 2: Commit — no errors from here on
        for id in new_ids {
            self.store.insert_unassigned(id, ());
        }

        Ok(())
    }

    /// Inserts pre-matched `(BlockId, SequenceHash)` pairs directly into the
    /// assigned collection.
    ///
    /// This is the entry point for blocks whose hashes are already known (e.g.
    /// cache hits). Two-phase atomic: collects all items, validates no duplicate
    /// BlockIds across all three collections, then commits to assigned.
    pub fn extend_assigned(
        &mut self,
        items: impl IntoIterator<Item = (BlockId, SequenceHash)>,
    ) -> Result<usize, BlockSequenceError> {
        // Phase 1: collect
        let items: Vec<(BlockId, SequenceHash)> = items.into_iter().collect();

        // Phase 2: validate
        let mut seen = indexmap::IndexSet::with_capacity(items.len());
        for (id, _) in &items {
            if self.contains(id) || !seen.insert(*id) {
                return Err(BlockSequenceError::DuplicateBlockId { block_id: *id });
            }
        }

        // Phase 3: commit
        let count = items.len();
        for (id, hash) in items {
            self.store.insert_assigned(id, hash);
        }

        Ok(count)
    }

    /// FIFO drain from unassigned into staged, pairing each block_id with the
    /// sequence hash from the corresponding `TokenBlock`.
    ///
    /// Staging starts at `sequence_blocks[self.next_position()]` and proceeds
    /// forward, consuming one unassigned ID per available block. The loop stops
    /// when either the unassigned queue is empty or there are no more sequence
    /// blocks.
    ///
    /// Returns the range of newly staged indices (relative to the start of the
    /// staged collection before this call).
    ///
    /// Each staged pair is validated: the position embedded in the block's
    /// `kvbm_sequence_hash()` must equal the expected sequence index.
    /// A mismatch returns [`BlockSequenceError::PositionMismatch`].
    pub fn stage_pending(
        &mut self,
        sequence_blocks: &[TokenBlock],
    ) -> Result<Range<usize>, BlockSequenceError> {
        let staged_start_idx = self.store.staged_count();
        let start_pos = self.next_position();

        // How many sequence blocks are available starting from our next position?
        let available_blocks = sequence_blocks.len().saturating_sub(start_pos);

        // How many can we stage? Min of available blocks and unassigned count.
        let to_stage = available_blocks.min(self.store.unassigned_count());

        for i in 0..to_stage {
            let seq_pos = start_pos + i;
            let block = &sequence_blocks[seq_pos];
            let hash = block.kvbm_sequence_hash();

            // Validate position
            let actual_pos = hash.position();
            if actual_pos != seq_pos as u64 {
                let block_id = self.store.get_unassigned(0).map(|(&id, _)| id).unwrap();
                return Err(BlockSequenceError::PositionMismatch {
                    expected: seq_pos,
                    actual: actual_pos,
                    block_id,
                });
            }

            // Pop front of unassigned (FIFO) and insert into staged
            let (block_id, _) = self.store.shift_unassigned().unwrap();
            self.store.insert_staged(block_id, hash);
        }

        let staged_end_idx = self.store.staged_count();
        Ok(staged_start_idx..staged_end_idx)
    }

    /// Moves all staged blocks into assigned (infallible).
    ///
    /// Returns the range of newly assigned indices (relative to the start of
    /// the assigned collection before this call).
    pub fn commit_staged(&mut self) -> Range<usize> {
        let start_idx = self.store.assigned_count();

        while let Some((block_id, hash)) = self.store.shift_staged() {
            self.store.insert_assigned(block_id, hash);
        }

        let end_idx = self.store.assigned_count();
        start_idx..end_idx
    }

    /// Drain the unassigned FIFO queue into assigned, pairing each block_id
    /// with the next available `TokenBlock` starting at `next_position()`.
    ///
    /// This is a convenience method equivalent to calling
    /// [`stage_pending`](Self::stage_pending) followed by
    /// [`commit_staged`](Self::commit_staged).
    ///
    /// Returns the range of newly assigned indices (relative to offset).
    /// An empty range means no new assignments were made.
    pub fn assign_pending(
        &mut self,
        sequence_blocks: &[TokenBlock],
    ) -> Result<Range<usize>, BlockSequenceError> {
        self.stage_pending(sequence_blocks)?;
        Ok(self.commit_staged())
    }
}

/// Zip two [`BlockAssignments`] over their overlapping assigned positions.
///
/// For each absolute position where **both** `a` and `b` have assigned blocks,
/// yields `(position, a_block_id, b_block_id)`.
///
/// Iteration order: ascending position.
pub fn zip_assigned(
    a: &BlockAssignments,
    b: &BlockAssignments,
) -> Vec<(usize, BlockId, BlockId)> {
    let a_range = a.assigned_positions();
    let b_range = b.assigned_positions();
    let start = a_range.start.max(b_range.start);
    let end = a_range.end.min(b_range.end);

    let mut result = Vec::new();
    for pos in start..end {
        // Both lookups are guaranteed to succeed within the intersection range.
        let (a_id, _) = a.get_at_position(pos).unwrap();
        let (b_id, _) = b.get_at_position(pos).unwrap();
        result.push((pos, a_id, b_id));
    }
    result
}

/// Zip `src` assigned positions with `dst` pending positions.
///
/// For each absolute position where `src` has an assigned block and `dst`
/// has a pending (unassigned) block, yields `(position, src_block_id, dst_block_id)`.
///
/// This is the onboard/offload planning primitive: the result tells you
/// which source blocks to transfer into which destination blocks.
pub fn zip_assigned_pending(
    src: &BlockAssignments,
    dst: &BlockAssignments,
) -> Vec<(usize, BlockId, BlockId)> {
    let src_range = src.assigned_positions();
    let dst_range = dst.pending_positions();
    let start = src_range.start.max(dst_range.start);
    let end = src_range.end.min(dst_range.end);

    let mut result = Vec::new();
    for pos in start..end {
        // Both lookups are guaranteed to succeed within the intersection range.
        let (src_id, _) = src.get_at_position(pos).unwrap();
        let dst_id = dst.get_pending_at_position(pos).unwrap();
        result.push((pos, src_id, dst_id));
    }
    result
}
