// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::mocker::protocols::{MoveBlock, UniqueBlock};
use crate::mocker::tokens::{compute_block_hash_for_seq, compute_seq_hash_for_blocks};
use rand::random;
use std::cmp::PartialEq;

/// A sequence that is actively being built, with the ability to add tokens and commit to hashes
#[derive(Debug, Clone)]
pub struct ActiveSequence {
    unique_blocks: Vec<UniqueBlock>,
    tokens: Vec<u32>,
    block_size: usize,
    chunk_size: usize,
    max_output_tokens: usize,
    generated_tokens: usize,
    num_input_tokens: usize,
    creation_signal: Option<MoveBlock>,
}

impl PartialEq for ActiveSequence {
    fn eq(&self, other: &Self) -> bool {
        // Check if basic fields match
        if self.tokens != other.tokens
            || self.block_size != other.block_size
            || self.chunk_size != other.chunk_size
            || self.max_output_tokens != other.max_output_tokens
            || self.num_input_tokens != other.num_input_tokens
            || self.generated_tokens != other.generated_tokens
        {
            return false;
        }

        // Check if both have the same number of blocks
        if self.unique_blocks.len() != other.unique_blocks.len() {
            return false;
        }

        // Compare blocks - we care about block type and hash equality for FullBlocks
        for (self_block, other_block) in self.unique_blocks.iter().zip(other.unique_blocks.iter()) {
            match (self_block, other_block) {
                (UniqueBlock::FullBlock(self_hash), UniqueBlock::FullBlock(other_hash)) => {
                    if self_hash != other_hash {
                        return false;
                    }
                }
                (UniqueBlock::PartialBlock(_), UniqueBlock::PartialBlock(_)) => {
                    // Both are PartialBlocks, we don't need to compare UUIDs
                    continue;
                }
                _ => {
                    // One is FullBlock and one is PartialBlock
                    return false;
                }
            }
        }

        true
    }
}

impl ActiveSequence {
    /// Create a new ActiveSequence instance with the provided tokens
    pub fn new(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        chunk_size: Option<usize>,
    ) -> Self {
        let block_size = block_size.unwrap_or(64);
        assert!(block_size > 1, "block_size must be greater than 1");
        let chunk_size = chunk_size.unwrap_or(256);
        let num_input_tokens = tokens.len();

        let mut unique_blocks = Vec::new();
        let mut signal = None;

        if !tokens.is_empty() {
            if tokens.len() >= block_size {
                // We have at least one complete block, process it
                let complete_blocks_len = (tokens.len() / block_size) * block_size;

                // Process complete blocks to get local block hashes
                let local_block_hashes =
                    compute_block_hash_for_seq(&tokens[0..complete_blocks_len], block_size);

                // Compute global hashes using rolling hash
                let global_hashes = compute_seq_hash_for_blocks(&local_block_hashes, None);

                // Convert global hashes to FullBlock variants
                for &hash in &global_hashes {
                    unique_blocks.push(UniqueBlock::FullBlock(hash));
                }
            }

            // Add a PartialBlock if there are remaining tokens
            if tokens.len() % block_size != 0 {
                unique_blocks.push(UniqueBlock::default()); // Creates a PartialBlock with a new UUID
            }

            // Create signal if we have blocks
            if !unique_blocks.is_empty() {
                signal = Some(MoveBlock::Use(unique_blocks.clone(), None));
            }
        }

        Self {
            unique_blocks,
            tokens,
            block_size,
            chunk_size,
            max_output_tokens,
            generated_tokens: 0,
            num_input_tokens,
            creation_signal: signal,
        }
    }

    /// Returns a reference to the unique blocks
    pub fn unique_blocks(&self) -> &Vec<UniqueBlock> {
        &self.unique_blocks
    }

    /// Returns a reference to the tokens
    pub fn tokens(&self) -> &Vec<u32> {
        &self.tokens
    }

    /// Returns the block size
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the chunk size
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Returns the maximum output tokens
    pub fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }

    /// Returns the number of generated tokens
    pub fn generated_tokens(&self) -> usize {
        self.generated_tokens
    }

    /// Returns the number of input tokens
    pub fn num_input_tokens(&self) -> usize {
        self.num_input_tokens
    }

    /// Returns a reference to the creation signal
    pub fn creation_signal(&self) -> &Option<MoveBlock> {
        &self.creation_signal
    }

    /// Create a new ActiveSequence instance and return the creation signal
    pub fn new_with_signal(
        tokens: Vec<u32>,
        max_output_tokens: usize,
        block_size: Option<usize>,
        chunk_size: Option<usize>,
    ) -> (Self, Option<MoveBlock>) {
        let mut sequence = Self::new(tokens, max_output_tokens, block_size, chunk_size);
        let signal = sequence.creation_signal.take();
        (sequence, signal)
    }

    /// Push a token to the sequence
    pub fn push(&mut self, token: u32) -> Option<MoveBlock> {
        self.tokens.push(token);
        self.generated_tokens += 1;
        let mut signal = None;

        // Add a partial block if this is the first token in a new partial sequence
        let remainder = self.tokens.len() % self.block_size;
        if remainder == 1 && self.tokens.len() > 1 {
            // Assert that we need a new partial block (empty or last block is full)
            assert!(
                self.unique_blocks.is_empty()
                    || matches!(self.unique_blocks.last(), Some(UniqueBlock::FullBlock(_))),
                "Expected empty blocks or last block to be a FullBlock"
            );

            let partial_block = UniqueBlock::default();
            self.unique_blocks.push(partial_block.clone());

            // Return Use signal for the new partial block
            signal = Some(MoveBlock::Use(vec![partial_block], None));

            return signal;
        }

        // Not enough tokens to complete a block
        if remainder != 0 {
            return None;
        }

        // At this point we have exactly completed a block
        // Get the latest block's worth of tokens
        let start_idx = self.tokens.len() - self.block_size;
        let block_tokens = &self.tokens[start_idx..];

        // Compute local block hash for the tokens
        let local_hash = compute_block_hash_for_seq(block_tokens, self.block_size);
        assert_eq!(
            local_hash.len(),
            1,
            "Expected local_hash to have exactly 1 value"
        );

        // Get the parent hash (the last full block if exists, otherwise None)
        let parent_hash = self
            .unique_blocks
            .iter()
            .rev()
            .find_map(|block| match block {
                UniqueBlock::FullBlock(hash) => Some(*hash),
                _ => None,
            });

        // Compute new global hash by rolling the local hash with the parent
        let new_global_hash = compute_seq_hash_for_blocks(&local_hash, parent_hash);

        // Ensure the last block is a partial block
        match self.unique_blocks.last() {
            Some(UniqueBlock::PartialBlock(uuid)) => {
                let uuid_copy = *uuid;

                // Replace the partial block with the full block
                self.unique_blocks.pop();

                if let Some(hash) = new_global_hash.first() {
                    // Add the new full block
                    self.unique_blocks.push(UniqueBlock::FullBlock(*hash));

                    // Return Promote signal
                    signal = Some(MoveBlock::Promote(uuid_copy, *hash));
                }
            }
            _ => panic!("Expected last block to be a PartialBlock"),
        }

        signal
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

        // Push the token to the sequence and collect any signal
        if let Some(signal) = self.push(token) {
            signals.push(signal);
        }

        // Check if we've reached the limit after pushing
        if self.generated_tokens != self.max_output_tokens {
            return signals;
        }

        // Free all blocks when we reach max tokens
        self.free_signal()
    }

    /// Free all blocks, generating appropriate signals for each block type
    fn free_signal(&self) -> Vec<MoveBlock> {
        let mut signals = Vec::new();

        // Collect blocks to deref based on type
        match self.unique_blocks.last() {
            Some(UniqueBlock::PartialBlock(uuid)) => {
                // All blocks except the last are full blocks, last is partial
                let full = self.unique_blocks[..self.unique_blocks.len() - 1].to_vec();
                let partial = vec![UniqueBlock::PartialBlock(*uuid)];

                // Add Destroy event for partial block first if it exists
                if !partial.is_empty() {
                    signals.push(MoveBlock::Destroy(partial));
                }

                // Then add Deref event for full blocks
                if !full.is_empty() {
                    signals.push(MoveBlock::Deref(full));
                }
            }
            _ => {
                // All blocks are full blocks
                if !self.unique_blocks.is_empty() {
                    signals.push(MoveBlock::Deref(self.unique_blocks.clone()));
                }
            }
        }

        signals
    }

    /// Reset the sequence to its initial state and return the reset sequence and signals from freeing current blocks
    ///
    /// This function:
    /// - Creates a new ActiveSequence with the original input tokens
    /// - Frees all current blocks, generating appropriate signals
    /// - Returns the new sequence and signals for the KvManager to process
    pub fn reset_with_signal(&self) -> (Self, Vec<MoveBlock>) {
        // Create a new sequence with the original input tokens
        let new_sequence = Self::new(
            self.tokens[0..self.num_input_tokens].to_vec(),
            self.max_output_tokens,
            Some(self.block_size),
            Some(self.chunk_size),
        );

        // Free current blocks and collect signals
        let signals = self.free_signal();

        (new_sequence, signals)
    }

    /// Resets the last partial block in the sequence.
    ///
    /// This function:
    /// - Asserts that the last block is a PartialBlock
    /// - Removes the last PartialBlock
    /// - Decrements the generated_tokens counter by the number of tokens in the partial block
    /// - Removes the corresponding tokens from the end of the sequence
    ///
    /// # Panics
    ///
    /// Panics if the last block is not a PartialBlock.
    pub fn partial_reset(&mut self) {
        // Assert that the last block is a PartialBlock
        match self.unique_blocks.last() {
            Some(UniqueBlock::PartialBlock(_)) => {
                // Calculate how many tokens are in the partial block
                let partial_block_size = self.tokens.len() % self.block_size;
                if partial_block_size == 0 {
                    // This shouldn't happen - if tokens.len() is a multiple of block_size,
                    // we should have no partial block
                    panic!("Expected tokens.len() to not be a multiple of block_size when a PartialBlock exists");
                }

                // Remove the last partial_block_size tokens
                self.tokens.truncate(self.tokens.len() - partial_block_size);

                // Decrement the generated tokens counter
                self.generated_tokens = self.generated_tokens.saturating_sub(partial_block_size);

                // Remove the PartialBlock
                self.unique_blocks.pop();
            }
            _ => panic!("Expected last block to be a PartialBlock"),
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
            ActiveSequence::new_with_signal(initial_tokens, 100, Some(16), Some(256));
        assert_eq!(seq1.num_input_tokens(), 15);

        // Clone seq1 before making any changes
        let seq1_clone = seq1.clone();

        // Check that we got a Use signal
        assert!(signal1.is_some());
        match &signal1 {
            Some(MoveBlock::Use(blocks, _)) => {
                assert_eq!(blocks.len(), 1);
            }
            _ => panic!("Expected Use signal"),
        }

        // Push tokens 15 and 16
        let signal_15 = seq1.push(15);
        let signal_16 = seq1.push(16);

        // Check signals
        assert!(signal_15.is_some());
        assert!(signal_16.is_some());
        if let Some(MoveBlock::Promote(_, _)) = signal_15 {
            // Expected behavior
        } else {
            panic!("Expected Promote signal for signal_15");
        }
        if let Some(MoveBlock::Use(_, _)) = signal_16 {
            // Expected behavior
        } else {
            panic!("Expected Use signal for signal_16");
        }

        // Verify state after pushing tokens
        assert_eq!(seq1.unique_blocks().len(), 2); // One full block and one partial block
        assert_eq!(seq1.tokens().len(), 17);
        assert_eq!(seq1.tokens().len() % seq1.block_size(), 1);

        // Create another sequence with block size 16 initialized with tokens [0..17]
        let extended_tokens: Vec<u32> = (0..17).collect();
        let (mut seq2, _) =
            ActiveSequence::new_with_signal(extended_tokens, 100, Some(16), Some(256));

        // Assert that the first block (full block) has the same hash in both sequences
        match (&seq1.unique_blocks()[0], &seq2.unique_blocks()[0]) {
            (UniqueBlock::FullBlock(hash1), UniqueBlock::FullBlock(hash2)) => {
                assert_eq!(hash1, hash2, "First blocks should have the same hash");
            }
            _ => panic!("Expected FullBlock for the first blocks"),
        }

        // Assert that the second blocks are different (both are partial blocks with different UUIDs)
        assert_ne!(
            seq1.unique_blocks()[1],
            seq2.unique_blocks()[1],
            "Second blocks should be different"
        );

        // Verify types of second blocks
        match (&seq1.unique_blocks()[1], &seq2.unique_blocks()[1]) {
            (UniqueBlock::PartialBlock(_), UniqueBlock::PartialBlock(_)) => {
                // Both are partial blocks but should have different UUIDs
            }
            _ => panic!("Expected PartialBlock for the second blocks"),
        }

        // Reset partial block on seq1 and push back token 16
        seq1.partial_reset();
        seq1.push(16);

        // Now push tokens 17..32 to both sequences
        for token in 17..32 {
            seq1.push(token);
            seq2.push(token);
        }

        // Both sequences should now have 2 blocks:
        // 1. FullBlock for tokens 0-15
        // 2. FullBlock for tokens 16-31
        // 3. No partial block since there are no remaining tokens
        assert_eq!(
            seq1.unique_blocks().len(),
            2,
            "seq1 should have exactly 2 blocks"
        );
        assert_eq!(
            seq2.unique_blocks().len(),
            2,
            "seq2 should have exactly 2 blocks"
        );
        assert_eq!(
            seq1.tokens().len() % seq1.block_size(),
            0,
            "seq1 should have no partial tokens"
        );
        assert_eq!(
            seq2.tokens().len() % seq2.block_size(),
            0,
            "seq2 should have no partial tokens"
        );

        // Verify that both sequences now have identical blocks
        assert_eq!(
            seq1.unique_blocks(),
            seq2.unique_blocks(),
            "After pushing tokens 17-31, both sequences should have identical blocks"
        );

        // Verify both blocks in detail
        match (&seq1.unique_blocks()[0], &seq2.unique_blocks()[0]) {
            (UniqueBlock::FullBlock(hash1), UniqueBlock::FullBlock(hash2)) => {
                assert_eq!(hash1, hash2, "First blocks should have the same hash");
            }
            _ => panic!("Expected FullBlock for the first blocks"),
        }

        match (&seq1.unique_blocks()[1], &seq2.unique_blocks()[1]) {
            (UniqueBlock::FullBlock(hash1), UniqueBlock::FullBlock(hash2)) => {
                assert_eq!(hash1, hash2, "Second blocks should have the same hash");
            }
            _ => panic!("Expected FullBlock for the second blocks"),
        }

        // Reset seq1 and check that it equals the original clone
        let (reset_seq, reset_signals) = seq1.reset_with_signal();

        // Verify the reset signals include proper cleanup events
        assert!(!reset_signals.is_empty());

        // Check that the reset sequence equals the original clone
        assert_eq!(
            reset_seq, seq1_clone,
            "Reset sequence should equal the original clone"
        );
    }

    #[test]
    fn test_active_sequence_generate_signals() {
        // Create a sequence with block size 16, max_output_tokens 4, initialized with tokens [0..14)
        let initial_tokens: Vec<u32> = (0..14).collect();
        let (mut seq, signal) =
            ActiveSequence::new_with_signal(initial_tokens, 4, Some(16), Some(256));

        // Initial signal - should have received a Use signal for the partial block
        assert!(signal.is_some());
        match signal {
            Some(MoveBlock::Use(blocks, _)) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Use signal for the initial partial block"),
        }

        // Generate first token - should not trigger new signals
        let signals_first = seq.generate();
        assert_eq!(signals_first.len(), 0);

        // Generate second token - this fills the block and should trigger a Promote signal
        let signals_second = seq.generate();
        assert_eq!(signals_second.len(), 1);
        match &signals_second[0] {
            MoveBlock::Promote(uuid, hash) => {
                // The uuid and hash values are generated dynamically, so we just check the event type
                let _ = uuid;
                let _ = hash;
            }
            _ => panic!("Expected Promote signal after second token"),
        }

        // Generate third token - should trigger a Use signal for the new partial block
        let signals_third = seq.generate();
        assert_eq!(signals_third.len(), 1);
        match &signals_third[0] {
            MoveBlock::Use(blocks, _) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Use signal for new partial block after third token"),
        }

        // Generate fourth token - we reach max_output_tokens, should trigger Destroy and Deref signals
        let signals_fourth = seq.generate();
        assert_eq!(signals_fourth.len(), 2);

        // First signal should be Destroy for the partial block
        match &signals_fourth[0] {
            MoveBlock::Destroy(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Destroy signal for partial block after fourth token"),
        }

        // Second signal should be Deref for the full block
        match &signals_fourth[1] {
            MoveBlock::Deref(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::FullBlock(_)));
            }
            _ => panic!("Expected Deref signal for full block after fourth token"),
        }
    }
}
