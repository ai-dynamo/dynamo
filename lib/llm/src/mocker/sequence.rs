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
use tokio::sync::mpsc::Sender;

/// A sequence that is actively being built, with the ability to add tokens and commit to hashes
#[derive(Debug, Clone)]
pub struct ActiveSequence {
    pub unique_blocks: Vec<UniqueBlock>,
    pub partial_tokens: Vec<u32>,
    pub block_size: usize,
    pub chunk_size: usize,
    pub max_output_tokens: usize,
    pub generated_tokens: usize,
    move_block_tx: Option<Sender<MoveBlock>>,
}

impl PartialEq for ActiveSequence {
    fn eq(&self, other: &Self) -> bool {
        self.unique_blocks == other.unique_blocks
            && self.partial_tokens == other.partial_tokens
            && self.block_size == other.block_size
            && self.chunk_size == other.chunk_size
            && self.max_output_tokens == other.max_output_tokens
        // move_block_tx is intentionally not compared
    }
}

impl ActiveSequence {
    /// Create a new ActiveSequence instance with the provided tokens
    pub fn new(
        tokens: Vec<u32>,
        block_size: Option<usize>,
        chunk_size: Option<usize>,
        max_output_tokens: usize,
        move_block_tx: Option<Sender<MoveBlock>>,
    ) -> Self {
        let block_size = block_size.unwrap_or(64);
        assert!(block_size > 1, "block_size must be greater than 1");
        let chunk_size = chunk_size.unwrap_or(256);

        let mut unique_blocks = Vec::new();
        let mut partial_tokens = Vec::new();

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

                // Get remaining tokens that don't form a complete block
                partial_tokens = tokens[complete_blocks_len..].to_vec();
            } else {
                // Not enough tokens for a full block, just store them as partial tokens
                partial_tokens = tokens;
            }

            // Add a PartialBlock if there are remaining tokens
            if !partial_tokens.is_empty() {
                unique_blocks.push(UniqueBlock::default()); // Creates a PartialBlock with a new UUID
            }
        }

        // Only process blocks and send event if event_tx is provided
        if let Some(tx) = &move_block_tx {
            // Send Use event if we have blocks
            if !unique_blocks.is_empty() {
                let _ = tx.try_send(MoveBlock::Use(unique_blocks.clone(), None));
            }
        }

        Self {
            unique_blocks,
            partial_tokens,
            block_size,
            chunk_size,
            max_output_tokens,
            generated_tokens: 0,
            move_block_tx,
        }
    }

    /// Push a token to the sequence
    pub fn push(&mut self, token: u32) {
        self.partial_tokens.push(token);

        // Add a partial block if this is the first token in a new partial sequence
        if self.partial_tokens.len() == 1 {
            // Assert that we need a new partial block (empty or last block is full)
            assert!(
                self.unique_blocks.is_empty()
                    || matches!(self.unique_blocks.last(), Some(UniqueBlock::FullBlock(_))),
                "Expected empty blocks or last block to be a FullBlock"
            );

            let partial_block = UniqueBlock::default();
            self.unique_blocks.push(partial_block.clone());

            // Send Use event for the new partial block
            if let Some(tx) = &self.move_block_tx {
                let _ = tx.try_send(MoveBlock::Use(vec![partial_block], None));
            }
        }

        // Not enough tokens for a complete block
        if self.partial_tokens.len() < self.block_size {
            return;
        }

        // At this point we should have exactly one block's worth of tokens
        assert_eq!(self.partial_tokens.len(), self.block_size);

        // Compute local block hash for the tokens
        let local_hash = compute_block_hash_for_seq(&self.partial_tokens, self.block_size);

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

                    // Send Promote event
                    if let Some(tx) = &self.move_block_tx {
                        let _ = tx.try_send(MoveBlock::Promote(uuid_copy, *hash));
                    }
                }
            }
            _ => panic!("Expected last block to be a PartialBlock"),
        }

        // Clear partial tokens since we've consumed them all
        self.partial_tokens.clear();
    }

    /// Generate a random token, push it to the sequence, and increment generation count.
    ///
    /// This function:
    /// - Generates a random token and adds it to the current sequence
    /// - Acquires a new partial block if needed or promotes an existing partial block to a full block
    /// - Manages block references by sending appropriate events to the KvManager
    /// - Automatically sends destroy and deref signals when max_output_tokens is reached
    ///
    /// # Panics
    ///
    /// Calling this function when max_output_tokens has already been reached will cause a panic.
    /// Always check `generated_tokens < max_output_tokens` before calling this method.
    pub fn generate(&mut self) {
        // Assert that we haven't reached the maximum output tokens
        assert!(
            self.generated_tokens < self.max_output_tokens,
            "Cannot generate more tokens: reached max_output_tokens limit"
        );

        // Generate a random token
        let token = random::<u32>();

        // Push the token to the sequence
        self.push(token);

        // Increment the generated tokens counter
        self.generated_tokens += 1;

        // Check if we've reached the limit after pushing
        if self.generated_tokens == self.max_output_tokens {
            if let Some(tx) = &self.move_block_tx {
                // Collect blocks to deref based on type
                let (full_blocks, partial_blocks) = match self.unique_blocks.last() {
                    Some(UniqueBlock::PartialBlock(uuid)) => {
                        // All blocks except the last are full blocks, last is partial
                        let full = self.unique_blocks[..self.unique_blocks.len() - 1].to_vec();
                        (full, vec![UniqueBlock::PartialBlock(*uuid)])
                    }
                    _ => {
                        // All blocks are full blocks
                        (self.unique_blocks.clone(), Vec::new())
                    }
                };

                // Send Destroy event for partial block first if it exists
                if !partial_blocks.is_empty() {
                    let _ = tx.try_send(MoveBlock::Destroy(partial_blocks));
                }

                // Then send Deref event for full blocks
                if !full_blocks.is_empty() {
                    let _ = tx.try_send(MoveBlock::Deref(full_blocks));
                }
            }
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
        let mut seq1 = ActiveSequence::new(initial_tokens, Some(16), Some(256), 100, None);

        // Push tokens 15 and 16
        seq1.push(15);
        seq1.push(16);

        // Verify state after pushing tokens
        assert_eq!(seq1.unique_blocks.len(), 2); // One full block and one partial block
        assert_eq!(seq1.partial_tokens.len(), 1);

        // Create another sequence with block size 16 initialized with tokens [0..17]
        let extended_tokens: Vec<u32> = (0..17).collect();
        let mut seq2 = ActiveSequence::new(extended_tokens, Some(16), Some(256), 100, None);

        // Assert that the first block (full block) has the same hash in both sequences
        match (&seq1.unique_blocks[0], &seq2.unique_blocks[0]) {
            (UniqueBlock::FullBlock(hash1), UniqueBlock::FullBlock(hash2)) => {
                assert_eq!(hash1, hash2, "First blocks should have the same hash");
            }
            _ => panic!("Expected FullBlock for the first blocks"),
        }

        // Assert that the second blocks are different (both are partial blocks with different UUIDs)
        assert_ne!(
            seq1.unique_blocks[1], seq2.unique_blocks[1],
            "Second blocks should be different"
        );

        // Verify types of second blocks
        match (&seq1.unique_blocks[1], &seq2.unique_blocks[1]) {
            (UniqueBlock::PartialBlock(_), UniqueBlock::PartialBlock(_)) => {
                // Both are partial blocks but should have different UUIDs
            }
            _ => panic!("Expected PartialBlock for the second blocks"),
        }

        // Now push tokens 17..32 to both sequences
        for token in 17..32 {
            seq1.push(token);
            seq2.push(token);
        }

        // Both sequences should now have 3 blocks:
        // 1. FullBlock for tokens 0-15
        // 2. FullBlock for tokens 16-31
        // 3. No partial block since there are no remaining tokens
        assert_eq!(
            seq1.unique_blocks.len(),
            2,
            "seq1 should have exactly 2 blocks"
        );
        assert_eq!(
            seq2.unique_blocks.len(),
            2,
            "seq2 should have exactly 2 blocks"
        );
        assert_eq!(
            seq1.partial_tokens.len(),
            0,
            "seq1 should have no partial tokens"
        );
        assert_eq!(
            seq2.partial_tokens.len(),
            0,
            "seq2 should have no partial tokens"
        );

        // Verify that both sequences now have identical blocks
        assert_eq!(
            seq1.unique_blocks, seq2.unique_blocks,
            "After pushing tokens 17-31, both sequences should have identical blocks"
        );

        // Verify both blocks in detail
        match (&seq1.unique_blocks[0], &seq2.unique_blocks[0]) {
            (UniqueBlock::FullBlock(hash1), UniqueBlock::FullBlock(hash2)) => {
                assert_eq!(hash1, hash2, "First blocks should have the same hash");
            }
            _ => panic!("Expected FullBlock for the first blocks"),
        }

        match (&seq1.unique_blocks[1], &seq2.unique_blocks[1]) {
            (UniqueBlock::FullBlock(hash1), UniqueBlock::FullBlock(hash2)) => {
                assert_eq!(hash1, hash2, "Second blocks should have the same hash");
            }
            _ => panic!("Expected FullBlock for the second blocks"),
        }
    }

    #[test]
    fn test_active_sequence_generate_events() {
        use std::collections::VecDeque;
        use tokio::sync::mpsc;

        // Create a channel to receive block events
        let (tx, mut rx) = mpsc::channel(10);

        // Create a sequence with block size 16, max_output_tokens 4, initialized with tokens [0..15)
        let initial_tokens: Vec<u32> = (0..14).collect();
        let mut seq = ActiveSequence::new(
            initial_tokens,
            Some(16),
            Some(256),
            4, // max of 4 tokens total
            Some(tx),
        );

        // Helper to collect events from the channel
        let collect_events = |rx: &mut mpsc::Receiver<MoveBlock>| {
            let mut events = VecDeque::new();
            while let Ok(event) = rx.try_recv() {
                events.push_back(event);
            }
            events
        };

        // Initial event - should have received a Use event for the partial block
        let initial_events = collect_events(&mut rx);
        assert_eq!(initial_events.len(), 1);
        match &initial_events[0] {
            MoveBlock::Use(blocks, _) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Use event for the initial partial block"),
        }

        // Generate first token - should not trigger new events
        seq.generate();
        assert!(
            rx.try_recv().is_err(),
            "No events should be triggered after first token"
        );

        // Generate second token - this fills the block and should trigger a Promote event
        seq.generate();
        let events_after_second = collect_events(&mut rx);
        assert_eq!(events_after_second.len(), 1);
        match &events_after_second[0] {
            MoveBlock::Promote(uuid, hash) => {
                // The uuid and hash values are generated dynamically, so we just check the event type
                let _ = uuid;
                let _ = hash;
            }
            _ => panic!("Expected Promote event after second token"),
        }

        // Generate third token - should trigger a Use event for the new partial block
        seq.generate();
        let events_after_third = collect_events(&mut rx);
        assert_eq!(events_after_third.len(), 1);
        match &events_after_third[0] {
            MoveBlock::Use(blocks, _) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Use event for new partial block after third token"),
        }

        // Generate fourth token - we reach max_output_tokens, should trigger Destroy and Deref events
        seq.generate();
        let events_after_fourth = collect_events(&mut rx);
        assert_eq!(events_after_fourth.len(), 2);

        // First event should be Destroy for the partial block
        match &events_after_fourth[0] {
            MoveBlock::Destroy(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::PartialBlock(_)));
            }
            _ => panic!("Expected Destroy event for partial block after fourth token"),
        }

        // Second event should be Deref for the full block
        match &events_after_fourth[1] {
            MoveBlock::Deref(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert!(matches!(blocks[0], UniqueBlock::FullBlock(_)));
            }
            _ => panic!("Expected Deref event for full block after fourth token"),
        }
    }
}
