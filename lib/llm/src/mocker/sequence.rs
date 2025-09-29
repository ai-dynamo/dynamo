// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::mocker::protocols::MoveBlock;
use crate::tokens::blocks::UniqueBlock;
use crate::tokens::{TokenBlockSequence, Tokens};
use derive_getters::Getters;
use rand::random;
use uuid;

/// Create unique blocks from a TokenBlockSequence
fn create_unique_blocks_from_sequence(
    tokens: &TokenBlockSequence,
    uuid: Option<uuid::Uuid>,
    block_size: usize,
    enable_prefix_caching: bool,
) -> Vec<UniqueBlock> {
    let mut unique_blocks: Vec<UniqueBlock> = tokens
        .blocks()
        .iter()
        .map(|block| {
            if enable_prefix_caching {
                UniqueBlock::FullBlock(block.sequence_hash())
            } else {
                UniqueBlock::FullBlock(random::<u64>())
            }
        })
        .collect();

    // Only push the partial block if tokens count isn't a multiple of block_size
    if !tokens.total_tokens().is_multiple_of(block_size) {
        unique_blocks.push(match uuid {
            Some(uuid) => UniqueBlock::PartialBlock(uuid),
            None => UniqueBlock::default(),
        });
    }
    unique_blocks
}

/// A sequence that is actively being built, with the ability to add tokens and commit to hashes
/// TODO: reuse tokens
#[derive(Debug, Getters)]
pub struct ActiveSequence {
    unique_blocks: Vec<UniqueBlock>,

    tokens: TokenBlockSequence,

    #[getter(copy)]
    block_size: usize,

    #[getter(copy)]
    max_output_tokens: usize,

    #[getter(copy)]
    generated_tokens: usize,

    #[getter(copy)]
    already_generated_tokens: usize,

    #[getter(copy)]
    num_input_tokens: usize,

    creation_signal: Option<MoveBlock>,

    #[getter(copy)]
    enable_prefix_caching: bool,
}

impl ActiveSequence {
    /// Create a new ActiveSequence instance with the provided tokens
    pub fn new(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        enable_prefix_caching: bool,
    ) -> Self {
        let block_size = block_size.unwrap_or(64);
        assert!(block_size > 1, "block_size must be greater than 1");
        let num_input_tokens = tokens.len();

        let tokens = Tokens::from(tokens).into_sequence(block_size as u32, Some(1337));
        let unique_blocks =
            create_unique_blocks_from_sequence(&tokens, None, block_size, enable_prefix_caching);
        let creation_signal = Some(MoveBlock::Use(unique_blocks.clone()));

        Self {
            unique_blocks,
            tokens,
            block_size,
            max_output_tokens,
            generated_tokens: 0,
            already_generated_tokens: 0,
            num_input_tokens,
            creation_signal,
            enable_prefix_caching,
        }
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

    pub fn take_creation_signal(&mut self) -> Option<MoveBlock> {
        self.creation_signal.take()
    }

    pub fn block_hashes(&self) -> Vec<u64> {
        self.tokens
            .blocks()
            .iter()
            .map(|block| block.block_hash())
            .collect()
    }

    /// Create a new ActiveSequence instance and return the creation signal
    pub fn new_with_signal(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        enable_prefix_caching: bool,
    ) -> (Self, Option<MoveBlock>) {
        let mut sequence = Self::new(tokens, max_output_tokens, block_size, enable_prefix_caching);
        let signal = sequence.take_creation_signal();
        (sequence, signal)
    }

    /// Get the parent hash from the second-to-last block if it exists and is a FullBlock
    fn get_parent_hash(&self) -> Option<u64> {
        if self.unique_blocks.len() < 2 {
            return None;
        }
        match &self.unique_blocks[self.unique_blocks.len() - 2] {
            UniqueBlock::FullBlock(hash) => Some(*hash),
            _ => panic!("Cannot have a partial block as parent"),
        }
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
            let last_block_hash = if self.enable_prefix_caching {
                self.tokens.last_complete_block().unwrap().sequence_hash()
            } else {
                random::<u64>()
            };
            self.unique_blocks.pop();
            self.unique_blocks
                .push(UniqueBlock::FullBlock(last_block_hash));
            signals.push(MoveBlock::Promote(
                uuid,
                last_block_hash,
                self.get_parent_hash(),
            ));
        }

        let new_partial_block = UniqueBlock::default();
        self.unique_blocks.push(new_partial_block.clone());
        signals.push(MoveBlock::Use(vec![new_partial_block]));
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
        signals.extend(self.free_signal());
        signals
    }

    /// Free all blocks, generating appropriate signals for each block type
    pub fn free_signal(&self) -> Vec<MoveBlock> {
        self.unique_blocks
            .iter()
            .rev()
            .map(|block| match block {
                UniqueBlock::PartialBlock(uuid) => {
                    MoveBlock::Destroy(vec![UniqueBlock::PartialBlock(*uuid)])
                }
                UniqueBlock::FullBlock(hash) => {
                    MoveBlock::Deref(vec![UniqueBlock::FullBlock(*hash)])
                }
            })
            .collect()
    }

    /// Reset the sequence to its initial state and return the free signals from freeing current blocks
    pub fn reset_with_signal(&mut self) -> Vec<MoveBlock> {
        let free_signal = self.free_signal();

        self.tokens.truncate(self.num_input_tokens).unwrap();
        self.unique_blocks = create_unique_blocks_from_sequence(
            &self.tokens,
            None,
            self.block_size,
            self.enable_prefix_caching,
        );
        self.already_generated_tokens = self.generated_tokens.max(self.already_generated_tokens);
        self.generated_tokens = 0;
        self.creation_signal = Some(MoveBlock::Use(self.unique_blocks.clone()));

        free_signal
    }

    /// Pops last token in the sequence.
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

    #[test]
    fn test_active_sequence_push() {
        // Create a sequence with block size 16 initialized with tokens [0..15]
        let initial_tokens: Vec<u32> = (0..15).collect();
        let (mut seq1, signal1) =
            ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), true);
        assert_eq!(seq1.num_input_tokens(), 15);
        assert_eq!(seq1.len(), 15);

        // Check that we got a Use signal
        assert!(signal1.is_some());
        match &signal1 {
            Some(MoveBlock::Use(blocks)) => {
                assert_eq!(blocks.len(), 1);
            }
            _ => panic!("Expected Use signal"),
        }

        // Push token 15 which should complete the block (no signals yet)
        let signal_15 = seq1.push(15);
        assert!(
            signal_15.is_none(),
            "Completing a block should not trigger signals"
        );

        // Push token 16 which should trigger both Promote and Use signals
        let signal_16 = seq1.push(16);
        assert!(signal_16.is_some());
        let signal_16 = signal_16.unwrap();
        assert_eq!(signal_16.len(), 2);

        // First signal should be Promote for the previous block
        match &signal_16[0] {
            MoveBlock::Promote(_, _, parent_hash) => {
                assert_eq!(*parent_hash, None);
            }
            _ => panic!("Expected Promote signal as second signal"),
        }

        // Second signal should be Use for new partial block
        match &signal_16[1] {
            MoveBlock::Use(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Use signal as first signal"),
        }

        // Verify state after pushing tokens
        assert_eq!(seq1.unique_blocks().len(), 2); // One full block and one partial block
        assert_eq!(seq1.len(), 17);
        assert_eq!(seq1.len() % seq1.block_size(), 1);

        // Create another sequence with block size 16 initialized with tokens [0..17]
        let extended_tokens: Vec<u32> = (0..16).collect();
        let (mut seq2, _) = ActiveSequence::new_with_signal(extended_tokens, 100, Some(16), true);
        seq2.push(16);
        seq2.pop();
        seq2.push(16);

        // Simplified assertions
        assert_eq!(
            seq1.unique_blocks()[0],
            seq2.unique_blocks()[0],
            "First blocks should be the same"
        );

        assert_ne!(
            seq1.unique_blocks()[1],
            seq2.unique_blocks()[1],
            "Second blocks should be different"
        );

        // Reset partial block on seq1 and push back token 16
        seq1.push(17);
        seq1.pop();
        seq1.pop();
        seq1.push(16);

        // Now push tokens 17..32 to both sequences
        for token in 17..33 {
            seq1.push(token);
            seq2.push(token);
        }

        // Both sequences should now have 2 blocks:
        // 1. FullBlock for tokens 0-15
        // 2. FullBlock for tokens 16-31
        // 3. No partial block since there are no remaining tokens
        assert_eq!(
            seq1.unique_blocks().len(),
            3,
            "seq1 should have exactly 3 blocks"
        );
        assert_eq!(
            seq2.unique_blocks().len(),
            3,
            "seq2 should have exactly 3 blocks"
        );
        assert_eq!(
            seq1.len() % seq1.block_size(),
            1,
            "seq1 should have 1 partial token"
        );
        assert_eq!(
            seq2.len() % seq2.block_size(),
            1,
            "seq2 should have 1 partial token"
        );

        // Verify that both sequences have identical blocks up to the second position
        assert_eq!(
            &seq1.unique_blocks()[0..2],
            &seq2.unique_blocks()[0..2],
            "First two blocks should be identical"
        );

        // Push tokens 34..47 to seq1
        for token in 33..48 {
            seq1.push(token);
        }

        // Push token 48 and get the signal - this completes the block and triggers signals
        let signal = seq1.push(48);
        let signal = signal.unwrap();

        // Check that signal[0] is promote
        match &signal[0] {
            MoveBlock::Promote(_, _, parent_hash) => {
                // Check that the parent_hash matches unique_blocks[1], which should be a full block
                if let UniqueBlock::FullBlock(expected_hash) = seq1.unique_blocks()[1] {
                    assert_eq!(
                        *parent_hash,
                        Some(expected_hash),
                        "Parent hash should match unique_blocks[1]"
                    );
                } else {
                    panic!("unique_blocks[1] should be a full block");
                }
            }
            _ => panic!("Expected Promote signal as first signal"),
        }

        // Reset seq1 and check that it equals the original clone
        let free_signals = seq1.reset_with_signal();

        // 49 - 15 generated tokens
        assert_eq!(seq1.already_generated_tokens, 34);

        // Verify the reset signals include proper cleanup events
        assert!(!free_signals.is_empty());
    }

    #[test]
    fn test_active_sequence_generate_signals() {
        // Create a sequence with block size 16, max_output_tokens 4, initialized with tokens [0..14)
        let initial_tokens: Vec<u32> = (0..14).collect();
        let (mut seq, signal) = ActiveSequence::new_with_signal(initial_tokens, 5, Some(16), true);

        // Initial signal - should have received a Use signal for the partial block
        assert!(signal.is_some());
        match signal {
            Some(MoveBlock::Use(blocks)) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Use signal for the initial partial block"),
        }

        // Generate first two tokens - should not trigger new signals
        seq.generate();
        let signals_first = seq.generate();
        assert_eq!(signals_first.len(), 0);

        // Generate third token - this fills the block and should trigger both Promote and Use signals
        let signals_second = seq.generate();
        assert_eq!(signals_second.len(), 2);

        // First signal should be Promote
        match &signals_second[0] {
            MoveBlock::Promote(_, _, parent_hash) => {
                assert_eq!(*parent_hash, None);
            }
            _ => panic!("Expected Promote signal as first signal after second token"),
        }

        // Second signal should be Use for new partial block
        match &signals_second[1] {
            MoveBlock::Use(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Use signal as second signal after second token"),
        }

        // Generate fourth token - should not trigger new signals as it's adding to partial block
        let signals_third = seq.generate();
        assert_eq!(signals_third.len(), 0);

        // Generate last token - we reach max_output_tokens, should trigger Destroy and Deref signals
        let signals_last = seq.generate();
        assert_eq!(signals_last.len(), 2);

        // First signal should be Destroy for the partial block
        match &signals_last[0] {
            MoveBlock::Destroy(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Destroy signal for partial block after fourth token"),
        }

        // Second signal should be Deref for the full block
        match &signals_last[1] {
            MoveBlock::Deref(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::FullBlock(_)));
            }
            _ => panic!("Expected Deref signal for full block after fourth token"),
        }
    }
}
