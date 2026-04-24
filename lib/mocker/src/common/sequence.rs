// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::common::protocols::MoveBlock;
use derive_getters::Getters;
use dynamo_tokens::blocks::UniqueBlock;
use dynamo_tokens::{PositionalLineageHash, TokenBlockSequence, Tokens};
use rand::random;
use validator::Validate;

/// Create unique blocks, block hashes, and positional-lineage hashes from a
/// [`TokenBlockSequence`].
fn create_sequence_cache(
    tokens: &TokenBlockSequence,
    block_size: usize,
    enable_prefix_caching: bool,
) -> (Vec<UniqueBlock>, Vec<u64>, Vec<PositionalLineageHash>) {
    let mut unique_blocks = Vec::with_capacity(tokens.blocks().len() + 1);
    let mut block_hashes = Vec::with_capacity(tokens.blocks().len());
    let mut plhs = Vec::with_capacity(tokens.blocks().len());

    for (pos, block) in tokens.blocks().iter().enumerate() {
        block_hashes.push(block.block_hash());
        if enable_prefix_caching {
            unique_blocks.push(UniqueBlock::FullBlock(block.sequence_hash()));
            plhs.push(block.positional_lineage_hash());
        } else {
            unique_blocks.push(UniqueBlock::FullBlock(random::<u64>()));
            plhs.push(PositionalLineageHash::new(
                random::<u64>(),
                None,
                pos as u64,
            ));
        }
    }

    // Only push the partial block if tokens count isn't a multiple of block_size
    if !tokens.total_tokens().is_multiple_of(block_size) {
        unique_blocks.push(UniqueBlock::default());
    }
    (unique_blocks, block_hashes, plhs)
}

/// A sequence that is actively being built, with the ability to add tokens and commit to hashes
/// TODO: reuse tokens
#[derive(Debug, Getters, Validate)]
pub struct ActiveSequence {
    unique_blocks: Vec<UniqueBlock>,
    block_hashes: Vec<u64>,
    plhs: Vec<PositionalLineageHash>,

    tokens: TokenBlockSequence,

    #[getter(copy)]
    #[validate(range(min = 2))]
    block_size: usize,

    #[getter(copy)]
    max_output_tokens: usize,

    #[getter(copy)]
    generated_tokens: usize,

    #[getter(copy)]
    num_input_tokens: usize,

    #[getter(copy)]
    num_allocated_tokens: usize,

    #[getter(copy)]
    enable_prefix_caching: bool,

    #[getter(copy)]
    emit_token_ids: bool,
}

impl ActiveSequence {
    /// Create a new ActiveSequence instance with the provided tokens
    pub fn new(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        enable_prefix_caching: bool,
        emit_token_ids: bool,
    ) -> Self {
        let block_size = block_size.unwrap_or(64);
        let num_input_tokens = tokens.len();

        let tokens = Tokens::from(tokens).into_sequence(block_size as u32, Some(1337));
        let (unique_blocks, block_hashes, plhs) =
            create_sequence_cache(&tokens, block_size, enable_prefix_caching);

        let seq = Self {
            unique_blocks,
            block_hashes,
            plhs,
            tokens,
            block_size,
            max_output_tokens,
            generated_tokens: 0,
            num_input_tokens,
            num_allocated_tokens: 0,
            enable_prefix_caching,
            emit_token_ids,
        };
        seq.validate().expect("invalid ActiveSequence");
        seq
    }

    pub fn extra_tokens(&self) -> u32 {
        (self.len() % self.block_size) as u32
    }

    pub fn len(&self) -> usize {
        self.tokens.total_tokens()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.total_tokens() == 0
    }

    /// Build a `MoveBlock::Use` signal for blocks up to `cumulative_tokens`
    /// without updating internal state. Returns `None` if no new blocks are needed.
    /// Call `commit_allocation` after the signal is successfully processed.
    pub fn prepare_allocation(&self, cumulative_tokens: usize) -> Option<MoveBlock> {
        let prev_blocks = self
            .num_allocated_tokens
            .div_ceil(self.block_size)
            .min(self.unique_blocks.len());
        let target_blocks = cumulative_tokens
            .div_ceil(self.block_size)
            .min(self.unique_blocks.len());
        if target_blocks <= prev_blocks {
            return None;
        }

        let range = prev_blocks..target_blocks;
        let blocks = self.unique_blocks[range.clone()].to_vec();

        let hash_start = prev_blocks.min(self.block_hashes.len());
        let hash_end = target_blocks.min(self.block_hashes.len());
        let hashes = self.block_hashes[hash_start..hash_end].to_vec();
        // Cached per-sequence PLHs (stable across calls).
        let plhs = self.plhs[hash_start..hash_end].to_vec();

        let token_ids = if self.emit_token_ids && hash_start < hash_end {
            Some(
                self.tokens.blocks()[hash_start..hash_end]
                    .iter()
                    .map(|b| b.tokens().to_vec())
                    .collect(),
            )
        } else {
            None
        };

        let parent = if prev_blocks > 0 {
            Some(self.unique_blocks[prev_blocks - 1].clone())
        } else {
            None
        };
        Some(MoveBlock::Use(blocks, hashes, plhs, token_ids, parent))
    }

    /// Positional lineage hashes for all fully-tokenised blocks in the sequence.
    /// Mirrors `block_hashes()` but returns the PLH identity used by kvbm-logical.
    pub fn positional_lineage_hashes(&self) -> &[PositionalLineageHash] {
        &self.plhs
    }

    /// Commit a successful allocation by advancing `num_allocated_tokens`.
    pub fn commit_allocation(&mut self, cumulative_tokens: usize) {
        self.num_allocated_tokens = cumulative_tokens;
    }

    /// Prepare + commit in one call (convenience for paths where failure is impossible).
    pub fn allocate_blocks_for_chunk(&mut self, cumulative_tokens: usize) -> Option<MoveBlock> {
        let signal = self.prepare_allocation(cumulative_tokens);
        self.commit_allocation(cumulative_tokens);
        signal
    }

    /// Allocate all remaining blocks at once (backward compat).
    pub fn take_creation_signal(&mut self) -> Option<MoveBlock> {
        self.allocate_blocks_for_chunk(self.len())
    }

    /// Create a new ActiveSequence instance and return the creation signal
    pub fn new_with_signal(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        enable_prefix_caching: bool,
    ) -> (Self, Option<MoveBlock>) {
        let mut sequence = Self::new(
            tokens,
            max_output_tokens,
            block_size,
            enable_prefix_caching,
            false,
        );
        let signal = sequence.take_creation_signal();
        (sequence, signal)
    }

    /// Push a token to the sequence
    pub fn push(&mut self, token: u32) -> Option<Vec<MoveBlock>> {
        self.tokens.append(token).expect("Token push failed.");
        self.generated_tokens += 1;

        if self.len() % self.block_size != 1 {
            return None;
        }

        // Add a partial block for the first token in a new partial sequence
        // Send Use signal (to allocate space for this new generation block)
        let mut signals = Vec::new();

        // Replace last partial block with full block if it exists
        if let Some(UniqueBlock::PartialBlock(uuid)) = self.unique_blocks.last().cloned() {
            let last_complete = self.tokens.last_complete_block().unwrap();
            let last_seq_hash = if self.enable_prefix_caching {
                last_complete.sequence_hash()
            } else {
                random::<u64>()
            };
            let last_block_hash = last_complete.block_hash();
            // Same randomization story as `last_seq_hash`: with prefix caching off,
            // two identical prompts must not share blocks, so the PLH we promote
            // with must also be unique — otherwise `process_promote`'s
            // `match_blocks(&[plh])` lookup would reuse another request's block.
            let last_plh = if self.enable_prefix_caching {
                last_complete.positional_lineage_hash()
            } else {
                PositionalLineageHash::new(random::<u64>(), None, self.block_hashes.len() as u64)
            };
            let promote_token_ids = if self.emit_token_ids {
                Some(last_complete.tokens().to_vec())
            } else {
                None
            };
            self.block_hashes.push(last_block_hash);
            self.plhs.push(last_plh);
            self.unique_blocks.pop();

            // After pop, the last element is the parent block
            let second_to_last_hash = self.unique_blocks.last().map(|block| match block {
                UniqueBlock::FullBlock(hash) => *hash,
                UniqueBlock::PartialBlock(_) => panic!("Cannot have a partial block as parent"),
            });

            self.unique_blocks
                .push(UniqueBlock::FullBlock(last_seq_hash));
            signals.push(MoveBlock::Promote(
                uuid,
                last_seq_hash,
                second_to_last_hash,
                last_block_hash,
                last_plh,
                promote_token_ids,
            ));
        }

        let new_partial_block = UniqueBlock::default();
        self.unique_blocks.push(new_partial_block.clone());
        signals.push(MoveBlock::Use(
            vec![new_partial_block],
            vec![],
            vec![],
            None,
            None,
        ));
        Some(signals)
    }

    /// Generate a random token, push it to the sequence, and increment generation count.
    ///
    /// This function:
    /// - Generates a random token and adds it to the current sequence
    /// - Acquires a new partial block if needed or promotes an existing partial block to a full block
    /// - Returns appropriate signals for the KvManager to process
    ///
    /// # Panics
    ///
    /// Calling this function when max_output_tokens has already been reached will cause a panic.
    /// Always check `generated_tokens < max_output_tokens` before calling this method.
    pub fn generate(&mut self) -> Vec<MoveBlock> {
        // Assert that we haven't reached the maximum output tokens
        assert!(
            self.generated_tokens < self.max_output_tokens,
            "Cannot generate more tokens: reached max_output_tokens limit"
        );

        // Generate a random token
        let token = random::<u32>();

        // Collect signals
        let mut signals = Vec::new();

        // Push the token to the sequence and collect any signals
        if let Some(move_blocks) = self.push(token) {
            signals.extend(move_blocks);
        }

        // Check if we've reached the limit after pushing
        if self.generated_tokens != self.max_output_tokens {
            return signals;
        }

        // Free all blocks when we reach max tokens
        signals.extend(self.free_signal_for_tokens(self.len()));
        signals
    }

    fn free_signal_for_tokens(&self, active_tokens: usize) -> Vec<MoveBlock> {
        let active_blocks = active_tokens
            .div_ceil(self.block_size)
            .min(self.unique_blocks.len());
        self.unique_blocks[..active_blocks]
            .iter()
            .rev()
            .map(|block| match block {
                UniqueBlock::PartialBlock(uuid) => {
                    MoveBlock::Deref(vec![UniqueBlock::PartialBlock(*uuid)])
                }
                UniqueBlock::FullBlock(hash) => {
                    MoveBlock::Deref(vec![UniqueBlock::FullBlock(*hash)])
                }
            })
            .collect()
    }

    /// Free the currently active allocation footprint.
    pub fn free_signal(&self) -> Vec<MoveBlock> {
        self.free_signal_for_tokens(self.num_allocated_tokens)
    }

    /// Move the request to a preempted state and return the free signals from freeing current blocks.
    /// Upon preemption, the sequence retains the tokens generated during the decode phase (if any).
    /// Resets `num_allocated_tokens` so re-admission will re-allocate from scratch.
    pub fn reset_with_signal(&mut self) -> Vec<MoveBlock> {
        let free_signal = self.free_signal();
        self.num_allocated_tokens = 0;
        free_signal
    }

    /// Pops the last token in the sequence.
    ///
    /// This is only used to undo a freshly generated decode token after a failed
    /// allocation/preemption path. Under that invariant, the token being removed
    /// must be in the current partial block, so we only need to drop the trailing
    /// partial `UniqueBlock` when the sequence length returns to an exact block
    /// boundary. Using this to unwind arbitrary prompt history would be incorrect.
    pub fn pop(&mut self) {
        self.tokens.pop();
        self.generated_tokens = self.generated_tokens.saturating_sub(1);

        // Reverts to the last full block
        if self.tokens.total_tokens().is_multiple_of(self.block_size) {
            self.unique_blocks.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_hashes_from_tokens(seq: &ActiveSequence) -> Vec<u64> {
        seq.tokens
            .blocks()
            .iter()
            .map(|block| block.block_hash())
            .collect()
    }

    fn assert_cached_hashes_match_promoted_blocks(seq: &ActiveSequence) {
        let num_full_unique_blocks = seq
            .unique_blocks()
            .iter()
            .filter(|block| matches!(block, UniqueBlock::FullBlock(_)))
            .count();
        assert_eq!(
            seq.block_hashes().as_slice(),
            &block_hashes_from_tokens(seq)[..num_full_unique_blocks],
            "cached block hashes should match the promoted full blocks"
        );
    }

    fn assert_use_signal(
        signal: &MoveBlock,
        expected_blocks: &[UniqueBlock],
        expected_hashes: &[u64],
    ) {
        match signal {
            MoveBlock::Use(blocks, hashes, ..) => {
                assert_eq!(blocks, expected_blocks);
                assert_eq!(hashes, expected_hashes);
            }
            _ => panic!("Expected MoveBlock::Use"),
        }
    }

    fn assert_single_partial_use(signal: &MoveBlock) {
        match signal {
            MoveBlock::Use(blocks, hashes, ..) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
                assert!(hashes.is_empty());
            }
            _ => panic!("Expected MoveBlock::Use with a single partial block"),
        }
    }

    fn assert_promote_parent(signal: &MoveBlock, expected_parent: Option<u64>) {
        match signal {
            MoveBlock::Promote(_, _, parent_hash, _hash, ..) => {
                assert_eq!(*parent_hash, expected_parent);
            }
            _ => panic!("Expected MoveBlock::Promote"),
        }
    }

    fn assert_deref_partial(signal: &MoveBlock) {
        match signal {
            MoveBlock::Deref(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected MoveBlock::Deref for partial block"),
        }
    }

    fn assert_deref_full(signal: &MoveBlock) {
        match signal {
            MoveBlock::Deref(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::FullBlock(_)));
            }
            _ => panic!("Expected MoveBlock::Deref for full block"),
        }
    }

    #[test]
    fn test_new_with_signal_creates_initial_partial_block() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (seq, signal) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);

        assert_eq!(seq.num_input_tokens(), 15);
        assert_eq!(seq.len(), 15);
        assert_single_partial_use(signal.as_ref().expect("Expected initial Use signal"));
    }

    #[test]
    fn test_push_across_block_boundary_promotes_and_allocates_partial() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq, _) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);

        let signal_15 = seq.push(15);
        assert!(
            signal_15.is_none(),
            "Completing a block should not trigger signals"
        );

        let signal_16 = seq.push(16).expect("Expected boundary crossing signals");
        assert_eq!(signal_16.len(), 2);
        assert_promote_parent(&signal_16[0], None);
        assert_single_partial_use(&signal_16[1]);

        assert_eq!(
            seq.unique_blocks().len(),
            2,
            "sequence should have one full block and one partial block"
        );
        assert_eq!(
            seq.len() % seq.block_size(),
            1,
            "sequence should have one token in the new partial block"
        );
    }

    #[test]
    fn test_equivalent_histories_preserve_full_block_identity() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq1, _) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);
        seq1.push(15);
        seq1.push(16);

        let extended_tokens: Vec<u32> = (0..16).collect();
        let (mut seq2, _) = ActiveSequence::new_with_signal(extended_tokens, 100, Some(16), true);
        seq2.push(16);
        seq2.pop();
        seq2.push(16);

        assert_eq!(seq1.unique_blocks()[0], seq2.unique_blocks()[0]);
        assert_ne!(seq1.unique_blocks()[1], seq2.unique_blocks()[1]);
    }

    #[test]
    fn test_promote_uses_previous_full_block_as_parent() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq, _) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);
        seq.push(15);
        seq.push(16);

        seq.push(17);
        seq.pop();
        seq.pop();
        seq.push(16);

        let extended_tokens: Vec<u32> = (0..16).collect();
        let (mut seq_equiv, _) =
            ActiveSequence::new_with_signal(extended_tokens, 100, Some(16), true);
        seq_equiv.push(16);
        seq_equiv.pop();
        seq_equiv.push(16);
        for token in 17..33 {
            seq.push(token);
            seq_equiv.push(token);
        }

        assert_eq!(
            &seq.unique_blocks()[0..2],
            &seq_equiv.unique_blocks()[0..2],
            "first two full blocks should remain identical"
        );

        for token in 33..48 {
            seq.push(token);
        }

        let signal = seq
            .push(48)
            .expect("Expected promote when opening next partial");

        let UniqueBlock::FullBlock(expected_hash) = seq.unique_blocks()[1] else {
            panic!("unique_blocks[1] should be a full block");
        };
        assert_promote_parent(&signal[0], Some(expected_hash));
        assert_single_partial_use(&signal[1]);
    }

    #[test]
    fn test_reset_with_signal_frees_blocks_and_resets_allocation() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq, _) = ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);
        seq.push(15);
        seq.push(16);
        seq.commit_allocation(seq.len());

        let free_signals = seq.reset_with_signal();

        assert!(!free_signals.is_empty());
        assert_eq!(seq.num_allocated_tokens(), 0);
        assert_eq!(seq.generated_tokens(), 2);
    }

    #[test]
    fn test_active_sequence_generate_signals() {
        // Create a sequence with block size 16, max_output_tokens 4, initialized with tokens [0..14)
        let initial_tokens: Vec<u32> = (0..14).collect();
        let (mut seq, signal) = ActiveSequence::new_with_signal(initial_tokens, 5, Some(16), true);

        // Initial signal - should have received a Use signal for the partial block
        assert_single_partial_use(signal.as_ref().expect("Expected initial Use signal"));

        // Generate first two tokens - should not trigger new signals
        seq.generate();
        let signals_first = seq.generate();
        assert_eq!(signals_first.len(), 0);

        // Generate third token - this fills the block and should trigger both Promote and Use signals
        let signals_second = seq.generate();
        assert_eq!(signals_second.len(), 2);

        // First signal should be Promote
        assert_promote_parent(&signals_second[0], None);

        // Second signal should be Use for new partial block
        assert_single_partial_use(&signals_second[1]);

        // Generate fourth token - should not trigger new signals as it's adding to partial block
        let signals_third = seq.generate();
        assert_eq!(signals_third.len(), 0);

        // Generate last token - we reach max_output_tokens, should trigger Deref signals
        let signals_last = seq.generate();
        assert_eq!(signals_last.len(), 2);

        // First signal should be Deref for the partial block
        assert_deref_partial(&signals_last[0]);

        // Second signal should be Deref for the full block
        assert_deref_full(&signals_last[1]);
    }

    #[test]
    fn test_prepare_allocation_slices_full_and_partial_blocks() {
        let tokens: Vec<u32> = (0..10).collect();
        let seq = ActiveSequence::new(tokens, 4, Some(4), true, false);

        let first = seq.prepare_allocation(4).unwrap();
        assert_use_signal(
            &first,
            &seq.unique_blocks()[0..1],
            &seq.block_hashes()[0..1],
        );

        let second = seq.prepare_allocation(8).unwrap();
        assert_use_signal(
            &second,
            &seq.unique_blocks()[0..2],
            &seq.block_hashes()[0..2],
        );

        let third = seq.prepare_allocation(10).unwrap();
        assert_use_signal(
            &third,
            &seq.unique_blocks()[0..3],
            &seq.block_hashes()[0..2],
        );
    }

    #[test]
    fn test_prepare_allocation_is_stable_until_commit() {
        let tokens: Vec<u32> = (0..10).collect();
        let mut seq = ActiveSequence::new(tokens, 4, Some(4), true, false);

        let first = seq.prepare_allocation(4).unwrap();
        let second = seq.prepare_allocation(4).unwrap();
        assert_eq!(first, second);

        seq.commit_allocation(4);
        let next = seq.prepare_allocation(8).unwrap();
        assert_use_signal(&next, &seq.unique_blocks()[1..2], &seq.block_hashes()[1..2]);
    }

    #[test]
    fn test_block_hash_cache_stays_in_sync_after_promote_and_pop() {
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq, _) = ActiveSequence::new_with_signal(initial_tokens, 4, Some(16), true);

        assert_cached_hashes_match_promoted_blocks(&seq);

        seq.push(15);
        assert_cached_hashes_match_promoted_blocks(&seq);

        let promote_signals = seq.push(16).unwrap();
        assert_eq!(promote_signals.len(), 2);
        assert_cached_hashes_match_promoted_blocks(&seq);

        // `pop()` is only valid for undoing a freshly generated token from the
        // current partial block; this is the replay/preemption path we rely on.
        seq.pop();
        assert_cached_hashes_match_promoted_blocks(&seq);
    }
}
