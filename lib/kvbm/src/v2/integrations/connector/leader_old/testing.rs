// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test helpers for ConnectorLeader development and validation.

use dynamo_tokens::{SequenceHash, TokenBlockSequence, Tokens};

use crate::v2::integrations::connector::{Slot, SlotCore};

/// Generate a sample TokenBlockSequence for testing.
///
/// Creates a sequence with consecutive token IDs starting from 0.
///
/// # Arguments
/// * `num_blocks` - Number of complete blocks to generate
/// * `block_size` - Tokens per block
///
/// # Returns
/// TokenBlockSequence with `num_blocks * block_size` tokens
pub fn sample_sequence(num_blocks: usize, block_size: usize) -> TokenBlockSequence {
    let total_tokens = num_blocks * block_size;
    let tokens: Vec<i32> = (0..total_tokens as i32).collect();
    Tokens::from(tokens).into_sequence(block_size as u32, Some(42))
}

/// Create a slot with sample data for testing.
///
/// # Arguments
/// * `request_id` - Request identifier
/// * `num_blocks` - Number of blocks in the sequence
/// * `block_size` - Tokens per block
///
/// # Returns
/// Slot initialized with sample TokenBlockSequence
pub fn create_test_slot(request_id: &str, num_blocks: usize, block_size: usize) -> Slot {
    let sequence = sample_sequence(num_blocks, block_size);
    let core = SlotCore::new(request_id.to_string(), sequence, block_size);
    Slot::new(core)
}

/// Extract sequence hashes from a slot.
///
/// Useful for querying block managers or validating slot state.
///
/// # Arguments
/// * `slot` - The slot to extract hashes from
///
/// # Returns
/// Vector of SequenceHash values, one per complete block
pub fn extract_sequence_hashes(slot: &Slot) -> Vec<SequenceHash> {
    slot.core()
        .sequence()
        .blocks()
        .iter()
        .map(|b| b.sequence_hash())
        .collect()
}

/// Extract a range of sequence hashes from a slot.
///
/// # Arguments
/// * `slot` - The slot to extract hashes from
/// * `start_block` - Starting block index (inclusive)
/// * `end_block` - Ending block index (exclusive), or None for all remaining
///
/// # Returns
/// Vector of SequenceHash values for the specified range
pub fn extract_sequence_hashes_range(
    slot: &Slot,
    start_block: usize,
    end_block: Option<usize>,
) -> Vec<SequenceHash> {
    let blocks = slot.core().sequence().blocks();
    let end = end_block.unwrap_or(blocks.len());

    blocks[start_block..end]
        .iter()
        .map(|b| b.sequence_hash())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_sequence_creates_correct_size() {
        let sequence = sample_sequence(10, 16);
        assert_eq!(sequence.blocks().len(), 10);
        // Each block should have block_size tokens
        for block in sequence.blocks() {
            assert_eq!(block.block_size(), 16);
        }
    }

    #[test]
    fn test_create_test_slot() {
        let slot = create_test_slot("test-req", 5, 16);
        assert_eq!(slot.core().request_id(), "test-req");
        assert_eq!(slot.core().sequence().blocks().len(), 5);
        assert_eq!(slot.core().block_size(), 16);
    }

    #[test]
    fn test_extract_sequence_hashes() {
        let slot = create_test_slot("test-req", 8, 16);
        let hashes = extract_sequence_hashes(&slot);
        assert_eq!(hashes.len(), 8);
    }

    #[test]
    fn test_extract_sequence_hashes_range() {
        let slot = create_test_slot("test-req", 10, 16);

        // Extract middle range
        let hashes = extract_sequence_hashes_range(&slot, 3, Some(7));
        assert_eq!(hashes.len(), 4);

        // Extract from start to end
        let all_hashes = extract_sequence_hashes_range(&slot, 0, None);
        assert_eq!(all_hashes.len(), 10);
    }
}
