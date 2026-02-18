// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../../docs/request_sequence.md")]

use crate::KvbmSequenceHashProvider;
use crate::blocks::{BlockMetadata, ImmutableBlock};
use crate::manager::BlockManager;
use crate::sequence::{BlockSequence, LogicalBlockAssignmentError, LogicalBlockAssignments};

use dynamo_tokens::Token;

/// Manages a request's block lifecycle through direct RAII integration with
/// [`BlockManager`], bypassing the `MoveBlock` signal protocol.
///
/// Composes [`BlockSequence`] (token data and hashing) and
/// [`LogicalBlockAssignments`] (RAII block guards) into a single unit that
/// handles:
///
/// - **Construction**: prefix matching, allocation, and registration of input blocks
/// - **Generation**: token-by-token extension with automatic block promotion
/// - **Preemption**: release all blocks, re-acquire later with potential cache hits
///
/// # Block lifecycle
///
/// The generation block (always at most one) lives as the single unassigned
/// entry in [`LogicalBlockAssignments`]. When a block boundary is crossed:
///
/// 1. `stage()` — the generation `MutableBlock` becomes a `CompleteBlock`
/// 2. `register()` — the `CompleteBlock` becomes an `ImmutableBlock`
/// 3. `allocate_blocks(1)` → `extend_blocks()` — a new generation block
pub struct RequestSequence<T: BlockMetadata> {
    sequence: BlockSequence,
    assignments: LogicalBlockAssignments<T>,
    generated_tokens: usize,
    max_output_tokens: usize,
    num_input_tokens: usize,
    prefix_matched_blocks: usize,
}

impl<T: BlockMetadata> RequestSequence<T> {
    // =====================================================================
    // Minimal constructor (no manager interaction)
    // =====================================================================

    /// Creates a `RequestSequence` with token data only. No blocks are
    /// allocated and no manager interaction occurs.
    ///
    /// The caller must use the individual methods ([`match_prefix`],
    /// [`add_matched_blocks`], [`allocate_blocks`],
    /// [`complete_and_register_pending`]) to search, allocate, and register
    /// blocks.
    ///
    /// [`match_prefix`]: Self::match_prefix
    /// [`add_matched_blocks`]: Self::add_matched_blocks
    /// [`allocate_blocks`]: Self::allocate_blocks
    /// [`complete_and_register_pending`]: Self::complete_and_register_pending
    pub fn new(tokens: Vec<Token>, max_output_tokens: usize, block_size: u32) -> Self {
        let num_input_tokens = tokens.len();
        let sequence = BlockSequence::new(tokens, block_size, None);
        let assignments = LogicalBlockAssignments::new();

        Self {
            sequence,
            assignments,
            generated_tokens: 0,
            max_output_tokens,
            num_input_tokens,
            prefix_matched_blocks: 0,
        }
    }

    // =====================================================================
    // Individual block operations
    // =====================================================================

    /// Search for prefix cache hits against the manager's pools.
    ///
    /// Returns matched [`ImmutableBlock`]s in sequence order. Pass the result
    /// to [`add_matched_blocks`](Self::add_matched_blocks).
    pub fn match_prefix(&self, manager: &BlockManager<T>) -> Vec<ImmutableBlock<T>> {
        let hashes = self.sequence.all_sequence_hashes();
        manager.match_blocks(&hashes)
    }

    /// Add prefix-matched immutable blocks as assigned.
    ///
    /// Updates the internal `prefix_matched_blocks` counter.
    /// Returns the number of blocks added.
    pub fn add_matched_blocks(
        &mut self,
        blocks: Vec<ImmutableBlock<T>>,
    ) -> Result<usize, LogicalBlockAssignmentError<T>> {
        let count = blocks.len();
        let start = self.assignments.assigned_count();
        let end = start + count;
        let sequence_blocks = self.sequence.blocks();

        assert!(
            end <= sequence_blocks.len(),
            "matched blocks exceed completed sequence blocks"
        );

        for (i, (block, seq_block)) in blocks.iter().zip(&sequence_blocks[start..end]).enumerate() {
            debug_assert_eq!(
                block.sequence_hash(),
                seq_block.kvbm_sequence_hash(),
                "matched block hash mismatch at position {}",
                start + i
            );
        }
        self.assignments.extend_assigned(blocks)?;
        self.prefix_matched_blocks = count;
        Ok(count)
    }

    /// Allocate mutable blocks from the manager and store as unassigned.
    ///
    /// Returns `false` if allocation fails (insufficient blocks).
    pub fn allocate_blocks(&mut self, count: usize, manager: &BlockManager<T>) -> bool {
        if count == 0 {
            return true;
        }
        let Some(new_blocks) = manager.allocate_blocks(count) else {
            return false;
        };
        self.assignments.extend_blocks(new_blocks).is_ok()
    }

    /// Stage and register all unassigned blocks that have corresponding
    /// completed token data in the sequence.
    ///
    /// Computes the offset from `assigned_count + staged_count` and stages
    /// all completed token blocks beyond that offset. This converts
    /// unassigned `MutableBlock`s → `CompleteBlock`s → `ImmutableBlock`s.
    ///
    /// A generation block (not yet filled) remains unassigned because it has
    /// no corresponding entry in `sequence.blocks()`.
    pub fn complete_and_register_pending(&mut self, manager: &BlockManager<T>) {
        let start = self.assignments.assigned_count() + self.assignments.staged_count();
        let completed = self.sequence.blocks().len();
        if start < completed {
            let blocks_slice = &self.sequence.blocks()[start..completed];
            self.assignments
                .stage(blocks_slice)
                .expect("staging should not fail (block sizes and counts match)");
            self.assignments.register(manager);
        }
    }

    // =====================================================================
    // Token-only append (no block lifecycle)
    // =====================================================================

    /// Append a generated token to the sequence. Increments `generated_tokens`.
    ///
    /// Returns `Some(block_index)` if a block boundary was crossed (the block
    /// at that index is now complete), `None` otherwise.
    ///
    /// Does **not** stage, register, or allocate — the caller handles block
    /// lifecycle via [`complete_and_register_pending`] and [`allocate_blocks`].
    ///
    /// [`complete_and_register_pending`]: Self::complete_and_register_pending
    /// [`allocate_blocks`]: Self::allocate_blocks
    ///
    /// # Panics
    ///
    /// Panics if `generated_tokens >= max_output_tokens`.
    pub fn append_token(&mut self, token: Token) -> Option<usize> {
        assert!(
            self.generated_tokens < self.max_output_tokens,
            "Cannot generate more tokens: reached max_output_tokens limit"
        );

        let completed_block = self
            .sequence
            .append_token(token)
            .expect("Token append failed");

        self.generated_tokens += 1;
        completed_block
    }

    /// Whether `generated_tokens >= max_output_tokens`.
    pub fn is_complete(&self) -> bool {
        self.generated_tokens >= self.max_output_tokens
    }

    // =====================================================================
    // Release / reacquire
    // =====================================================================

    /// Releases all block assignments (RAII returns them to pools).
    pub fn release(&mut self) {
        self.assignments.clear();
    }

    /// Re-acquires blocks from the manager after a release/preemption.
    ///
    /// Uses the sequence's current token state (input + generated) to
    /// match prefix blocks from pools and allocate the remainder.
    ///
    /// Returns `true` if all blocks were successfully acquired.
    pub fn reacquire(&mut self, manager: &BlockManager<T>) -> bool {
        debug_assert!(
            self.assignments.is_empty(),
            "reacquire called with existing assignments"
        );

        let completed_blocks = self.sequence.blocks().len();

        // Step 1: Prefix match
        let hashes = self.sequence.all_sequence_hashes();
        let matched = manager.match_blocks(&hashes);
        let matched_count = matched.len();

        if !matched.is_empty() && self.assignments.extend_assigned(matched).is_err() {
            self.assignments.clear();
            return false;
        }

        // Step 2: Allocate remaining complete blocks (gen block allocated by schedule_decode)
        let remaining_complete = completed_blocks - matched_count;
        let total = remaining_complete;

        if total > 0 {
            let Some(new_blocks) = manager.allocate_blocks(total) else {
                self.assignments.clear();
                return false;
            };
            if self.assignments.extend_blocks(new_blocks).is_err() {
                self.assignments.clear();
                return false;
            }
        }

        // Step 3: Stage and register remaining complete blocks
        if remaining_complete > 0 {
            let blocks_slice = &self.sequence.blocks()[matched_count..completed_blocks];
            if self.assignments.stage(blocks_slice).is_err() {
                self.assignments.clear();
                return false;
            }
            self.assignments.register(manager);
        }

        self.prefix_matched_blocks = matched_count;
        true
    }

    // =====================================================================
    // Accessors
    // =====================================================================

    /// Number of tokens generated so far.
    pub fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    /// Maximum number of output tokens.
    pub fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    /// Number of input tokens the sequence was created with.
    pub fn num_input_tokens(&self) -> usize {
        self.num_input_tokens
    }

    /// Total token count (input + generated).
    pub fn total_tokens(&self) -> usize {
        self.sequence.total_tokens()
    }

    /// Number of tokens remaining to be generated.
    pub fn remaining_tokens(&self) -> usize {
        self.max_output_tokens.saturating_sub(self.generated_tokens)
    }

    /// Number of completed blocks in the token sequence.
    pub fn num_blocks(&self) -> usize {
        self.sequence.blocks().len()
    }

    /// Number of complete token blocks in the sequence.
    /// Alias for `num_blocks()` with a more descriptive name for scheduling.
    pub fn complete_sequence_blocks(&self) -> usize {
        self.sequence.blocks().len()
    }

    /// Number of blocks currently assigned (registered or cache-matched).
    pub fn assigned_blocks(&self) -> usize {
        self.assignments.assigned_count()
    }

    /// Number of blocks currently staged (completed but not registered).
    pub fn staged_blocks(&self) -> usize {
        self.assignments.staged_count()
    }

    /// Number of unassigned blocks (the generation block, if any).
    pub fn unassigned_blocks(&self) -> usize {
        self.assignments.unassigned_count()
    }

    /// Number of blocks that were prefix-matched during construction or reacquire.
    pub fn prefix_matched_blocks(&self) -> usize {
        self.prefix_matched_blocks
    }

    /// Number of new (non-cached) tokens for prefill cost calculation.
    pub fn new_tokens_for_prefill(&self) -> usize {
        let cached_tokens = self.prefix_matched_blocks * self.sequence.block_size();
        self.total_tokens().saturating_sub(cached_tokens)
    }

    /// Reference to the underlying `BlockSequence`.
    pub fn sequence(&self) -> &BlockSequence {
        &self.sequence
    }

    /// Reference to the underlying `LogicalBlockAssignments`.
    pub fn assignments(&self) -> &LogicalBlockAssignments<T> {
        &self.assignments
    }

    /// Block size used by this sequence.
    pub fn block_size(&self) -> usize {
        self.sequence.block_size()
    }

    // =====================================================================
    // Crate-internal mutation accessors
    // =====================================================================

    /// Mutable access to assignments for higher-level wrappers.
    pub(crate) fn assignments_mut(&mut self) -> &mut LogicalBlockAssignments<T> {
        &mut self.assignments
    }

    /// Mutable access to the underlying `BlockSequence`.
    #[allow(dead_code)]
    pub(crate) fn sequence_mut(&mut self) -> &mut BlockSequence {
        &mut self.sequence
    }

    /// Bulk-increment the generated token counter (for speculative decode).
    #[allow(dead_code)]
    pub(crate) fn add_generated_tokens(&mut self, count: usize) {
        self.generated_tokens += count;
    }
}

impl<T: BlockMetadata> std::fmt::Debug for RequestSequence<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestSequence")
            .field("total_tokens", &self.sequence.total_tokens())
            .field("num_blocks", &self.sequence.blocks().len())
            .field("generated_tokens", &self.generated_tokens)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("num_input_tokens", &self.num_input_tokens)
            .field("prefix_matched_blocks", &self.prefix_matched_blocks)
            .field("assigned", &self.assignments.assigned_count())
            .field("staged", &self.assignments.staged_count())
            .field("unassigned", &self.assignments.unassigned_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestMeta, create_test_manager};

    const BLOCK_SIZE: u32 = 4;

    fn make_tokens(n: usize) -> Vec<Token> {
        (0..n as u32).collect()
    }

    /// Builds a prefilled `RequestSequence` with blocks allocated, staged,
    /// and registered — the same end state as the removed `with_manager()`.
    fn build_prefilled(
        tokens: Vec<Token>,
        max_output_tokens: usize,
        block_size: u32,
        manager: &BlockManager<TestMeta>,
    ) -> Option<RequestSequence<TestMeta>> {
        let mut seq = RequestSequence::new(tokens, max_output_tokens, block_size);
        let completed_blocks = seq.num_blocks();
        let matched = seq.match_prefix(manager);
        let matched_count = matched.len();
        if !matched.is_empty() {
            seq.add_matched_blocks(matched).ok()?;
        }
        let remaining_complete = completed_blocks - matched_count;
        let needs_generation = max_output_tokens > 0;
        let total_to_allocate = remaining_complete + usize::from(needs_generation);
        if !seq.allocate_blocks(total_to_allocate, manager) {
            return None;
        }
        seq.complete_and_register_pending(manager);
        Some(seq)
    }

    // =========================================================================
    // Construction
    // =========================================================================

    #[test]
    fn test_new_minimal_constructor() {
        let tokens = make_tokens(8);
        let seq = RequestSequence::<TestMeta>::new(tokens, 10, BLOCK_SIZE);

        assert_eq!(seq.num_input_tokens(), 8);
        assert_eq!(seq.total_tokens(), 8);
        assert_eq!(seq.num_blocks(), 2);
        assert_eq!(seq.generated_tokens(), 0);
        assert_eq!(seq.max_output_tokens(), 10);
        assert_eq!(seq.block_size(), BLOCK_SIZE as usize);

        assert_eq!(seq.assigned_blocks(), 0);
        assert_eq!(seq.staged_blocks(), 0);
        assert_eq!(seq.unassigned_blocks(), 0);
        assert_eq!(seq.prefix_matched_blocks(), 0);
    }

    #[test]
    fn test_build_prefilled_basic() {
        let manager = create_test_manager::<TestMeta>(20);
        let tokens = make_tokens(8); // 2 complete blocks
        let seq = build_prefilled(tokens, 10, BLOCK_SIZE, &manager).unwrap();

        assert_eq!(seq.num_input_tokens(), 8);
        assert_eq!(seq.total_tokens(), 8);
        assert_eq!(seq.num_blocks(), 2);
        assert_eq!(seq.generated_tokens(), 0);
        assert_eq!(seq.max_output_tokens(), 10);

        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.staged_blocks(), 0);
        assert_eq!(seq.unassigned_blocks(), 1);
    }

    #[test]
    fn test_build_prefilled_partial_tokens() {
        let manager = create_test_manager::<TestMeta>(20);
        let seq = build_prefilled(make_tokens(6), 10, BLOCK_SIZE, &manager).unwrap();

        assert_eq!(seq.num_blocks(), 1);
        assert_eq!(seq.assigned_blocks(), 1);
        assert_eq!(seq.unassigned_blocks(), 1);
    }

    #[test]
    fn test_build_prefilled_empty_tokens() {
        let manager = create_test_manager::<TestMeta>(20);
        let seq = build_prefilled(vec![], 10, BLOCK_SIZE, &manager).unwrap();

        assert_eq!(seq.num_blocks(), 0);
        assert_eq!(seq.assigned_blocks(), 0);
        assert_eq!(seq.unassigned_blocks(), 1);
    }

    #[test]
    fn test_build_prefilled_zero_max() {
        let manager = create_test_manager::<TestMeta>(20);
        let seq = build_prefilled(make_tokens(8), 0, BLOCK_SIZE, &manager).unwrap();

        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 0);
    }

    #[test]
    fn test_build_prefilled_allocation_failure() {
        let manager = create_test_manager::<TestMeta>(2);
        let result = build_prefilled(make_tokens(12), 10, BLOCK_SIZE, &manager);
        assert!(result.is_none());
        assert_eq!(manager.available_blocks(), 2);
    }

    // =========================================================================
    // Prefix matching
    // =========================================================================

    #[test]
    fn test_prefix_cache_hit() {
        let manager = create_test_manager::<TestMeta>(20);
        let tokens = make_tokens(8);

        // Populate the manager with blocks for these tokens
        let seq_for_populate = BlockSequence::new(tokens.clone(), BLOCK_SIZE, None);
        let mutables = manager.allocate_blocks(2).unwrap();
        let registered: Vec<_> = mutables
            .into_iter()
            .zip(seq_for_populate.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();
        drop(registered);

        let seq = build_prefilled(tokens, 10, BLOCK_SIZE, &manager).unwrap();
        assert_eq!(seq.prefix_matched_blocks(), 2);
    }

    #[test]
    fn test_partial_prefix_cache_hit() {
        let manager = create_test_manager::<TestMeta>(20);
        let tokens = make_tokens(12);

        let seq_for_populate = BlockSequence::new(tokens[..4].to_vec(), BLOCK_SIZE, None);
        let mutables = manager.allocate_blocks(1).unwrap();
        let registered: Vec<_> = mutables
            .into_iter()
            .zip(seq_for_populate.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();
        drop(registered);

        let seq = build_prefilled(tokens, 10, BLOCK_SIZE, &manager).unwrap();
        assert_eq!(seq.prefix_matched_blocks(), 1);
        assert_eq!(seq.assigned_blocks(), 3);
        assert_eq!(seq.unassigned_blocks(), 1);
    }

    #[test]
    #[should_panic(expected = "matched blocks exceed completed sequence blocks")]
    fn test_add_matched_blocks_panics_when_matched_exceeds_completed() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = RequestSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE);

        let source = BlockSequence::new(make_tokens(8), BLOCK_SIZE, None);
        let mutables = manager.allocate_blocks(2).unwrap();
        let matched: Vec<_> = mutables
            .into_iter()
            .zip(source.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();

        let _ = seq.add_matched_blocks(matched);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "matched block hash mismatch at position 0")]
    fn test_add_matched_blocks_panics_on_hash_mismatch() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = RequestSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE);

        let source = BlockSequence::new(vec![100, 101, 102, 103], BLOCK_SIZE, None);
        let mutable = manager
            .allocate_blocks(1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        let mismatched = manager.register_block(mutable.complete(&source.blocks()[0]).unwrap());

        let _ = seq.add_matched_blocks(vec![mismatched]);
    }

    #[test]
    fn test_new_tokens_for_prefill() {
        let manager = create_test_manager::<TestMeta>(20);
        let tokens = make_tokens(12);

        let seq_for_populate = BlockSequence::new(tokens[..4].to_vec(), BLOCK_SIZE, None);
        let mutables = manager.allocate_blocks(1).unwrap();
        let registered: Vec<_> = mutables
            .into_iter()
            .zip(seq_for_populate.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();
        drop(registered);

        let seq = build_prefilled(tokens, 10, BLOCK_SIZE, &manager).unwrap();
        assert_eq!(seq.prefix_matched_blocks(), 1);
        assert_eq!(seq.new_tokens_for_prefill(), 8);
    }

    // =========================================================================
    // Token append (individual ops)
    // =========================================================================

    #[test]
    fn test_append_token_no_boundary() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(5), 10, BLOCK_SIZE, &manager).unwrap();

        // 5 tokens = 1 complete + 1 partial (1 token). Append 1 → 2 partial tokens, no boundary.
        assert!(seq.append_token(100).is_none());
        assert_eq!(seq.generated_tokens(), 1);
        assert_eq!(seq.total_tokens(), 6);
    }

    #[test]
    fn test_append_token_crosses_boundary() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(7), 10, BLOCK_SIZE, &manager).unwrap();

        // 7 tokens = 1 complete + 3 partial. 1 more completes block 1.
        let block_idx = seq.append_token(100);
        assert!(block_idx.is_some());
        assert_eq!(seq.num_blocks(), 2);
    }

    #[test]
    #[should_panic(expected = "Cannot generate more tokens")]
    fn test_append_token_panics_after_max() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(4), 1, BLOCK_SIZE, &manager).unwrap();

        seq.append_token(100); // generated_tokens == 1 == max
        seq.append_token(101); // panics
    }

    #[test]
    fn test_is_complete_transitions() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(4), 3, BLOCK_SIZE, &manager).unwrap();

        assert!(!seq.is_complete());
        seq.append_token(100);
        assert!(!seq.is_complete());
        seq.append_token(101);
        assert!(!seq.is_complete());
        seq.append_token(102);
        assert!(seq.is_complete());
    }

    #[test]
    fn test_is_complete_zero_max() {
        let seq = RequestSequence::<TestMeta>::new(make_tokens(4), 0, BLOCK_SIZE);
        assert!(seq.is_complete());
    }

    // =========================================================================
    // Modular decode loop
    // =========================================================================

    #[test]
    fn test_modular_decode_loop() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(4), 8, BLOCK_SIZE, &manager).unwrap();

        // Generate 4 tokens → complete a block
        for i in 0..3 {
            assert!(seq.append_token(100 + i).is_none());
        }
        let block_idx = seq.append_token(103);
        assert!(block_idx.is_some());
        assert!(!seq.is_complete());

        seq.complete_and_register_pending(&manager);
        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 0);
        assert!(seq.allocate_blocks(1, &manager));
        assert_eq!(seq.unassigned_blocks(), 1);

        // Generate 4 more → complete + max
        for i in 0..3 {
            assert!(seq.append_token(200 + i).is_none());
        }
        assert!(seq.append_token(203).is_some());
        assert!(seq.is_complete());

        seq.complete_and_register_pending(&manager);
        assert_eq!(seq.assigned_blocks(), 3);
        seq.release();
        assert_eq!(seq.assigned_blocks(), 0);
    }

    #[test]
    fn test_modular_prefill_with_cache() {
        let manager = create_test_manager::<TestMeta>(20);
        let tokens = make_tokens(8);

        // Populate cache
        let seq_for_populate = BlockSequence::new(tokens[..4].to_vec(), BLOCK_SIZE, None);
        let mutables = manager.allocate_blocks(1).unwrap();
        let registered: Vec<_> = mutables
            .into_iter()
            .zip(seq_for_populate.blocks().iter())
            .map(|(m, tb)| manager.register_block(m.complete(tb).unwrap()))
            .collect();
        drop(registered);

        let mut seq = RequestSequence::<TestMeta>::new(tokens, 10, BLOCK_SIZE);
        let matched = seq.match_prefix(&manager);
        assert_eq!(matched.len(), 1);
        seq.add_matched_blocks(matched).unwrap();
        assert_eq!(seq.prefix_matched_blocks(), 1);

        let remaining = seq.num_blocks() - 1;
        assert!(seq.allocate_blocks(remaining + 1, &manager));
        seq.complete_and_register_pending(&manager);

        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 1);
    }

    // =========================================================================
    // Allocate blocks
    // =========================================================================

    #[test]
    fn test_allocate_blocks_failure() {
        let manager = create_test_manager::<TestMeta>(2);
        let mut seq = RequestSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE);

        assert!(!seq.allocate_blocks(3, &manager));
        assert!(seq.allocate_blocks(2, &manager));
        assert_eq!(seq.unassigned_blocks(), 2);
    }

    #[test]
    fn test_allocate_blocks_zero() {
        let manager = create_test_manager::<TestMeta>(2);
        let mut seq = RequestSequence::<TestMeta>::new(make_tokens(4), 10, BLOCK_SIZE);

        assert!(seq.allocate_blocks(0, &manager));
        assert_eq!(seq.unassigned_blocks(), 0);
    }

    // =========================================================================
    // Release / reacquire
    // =========================================================================

    #[test]
    fn test_release() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(8), 10, BLOCK_SIZE, &manager).unwrap();

        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 1);
        let available_before = manager.available_blocks();

        seq.release();

        assert_eq!(seq.assigned_blocks(), 0);
        assert_eq!(seq.staged_blocks(), 0);
        assert_eq!(seq.unassigned_blocks(), 0);
        assert!(manager.available_blocks() > available_before);
    }

    #[test]
    fn test_release_idempotent() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(8), 10, BLOCK_SIZE, &manager).unwrap();

        seq.release();
        seq.release();
        assert_eq!(seq.assigned_blocks(), 0);
    }

    #[test]
    fn test_release_returns_blocks_to_pools() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(8), 10, BLOCK_SIZE, &manager).unwrap();

        assert_eq!(manager.available_blocks(), 17);
        seq.release();
        assert_eq!(manager.available_blocks(), 20);
    }

    #[test]
    fn test_reacquire_basic() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(8), 10, BLOCK_SIZE, &manager).unwrap();

        seq.append_token(100);
        seq.append_token(101);

        seq.release();
        assert!(seq.reacquire(&manager));
        assert_eq!(seq.assigned_blocks(), 2);
        assert_eq!(seq.unassigned_blocks(), 0); // no gen block from reacquire
        assert_eq!(seq.generated_tokens(), 2);
    }

    #[test]
    fn test_reacquire_with_cache_hits() {
        let manager = create_test_manager::<TestMeta>(20);
        let mut seq = build_prefilled(make_tokens(8), 10, BLOCK_SIZE, &manager).unwrap();

        seq.release();
        assert!(seq.reacquire(&manager));
        assert_eq!(seq.prefix_matched_blocks(), 2);
    }

    #[test]
    fn test_reacquire_cleans_up_on_failure() {
        let manager = create_test_manager::<TestMeta>(4);
        let mut seq = build_prefilled(make_tokens(4), 10, BLOCK_SIZE, &manager).unwrap();

        seq.release();
        let _all = manager.allocate_blocks(4).unwrap();

        assert!(!seq.reacquire(&manager));
        assert_eq!(seq.assigned_blocks(), 0);
        assert_eq!(seq.unassigned_blocks(), 0);
    }

    // =========================================================================
    // RAII
    // =========================================================================

    #[test]
    fn test_blocks_returned_on_drop() {
        let manager = create_test_manager::<TestMeta>(20);
        {
            let _seq = build_prefilled(make_tokens(8), 10, BLOCK_SIZE, &manager).unwrap();
            assert_eq!(manager.available_blocks(), 17);
        }
        assert_eq!(manager.available_blocks(), 20);
    }

    // =========================================================================
    // Debug
    // =========================================================================

    #[test]
    fn test_debug_impl() {
        let manager = create_test_manager::<TestMeta>(20);
        let seq = build_prefilled(make_tokens(8), 10, BLOCK_SIZE, &manager).unwrap();

        let debug_str = format!("{seq:?}");
        assert!(debug_str.contains("RequestSequence"));
        assert!(debug_str.contains("total_tokens"));
    }
}
