// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token block creation utilities for testing.

use dynamo_tokens::{TokenBlock, TokenBlockSequence};

use crate::v2::logical::pools::SequenceHash;

/// Create a token block from a slice of tokens.
///
/// # Example
/// ```ignore
/// let tokens = vec![1, 2, 3, 4];
/// let block = create_token_block(&tokens);
/// ```
pub fn create_token_block(tokens: &[u32]) -> TokenBlock {
    let token_sequence = TokenBlockSequence::from_slice(tokens, tokens.len() as u32, Some(42));
    if let Some(block) = token_sequence.blocks().first() {
        block.clone()
    } else {
        let mut partial = token_sequence.into_parts().1;
        partial.commit().expect("Should be able to commit")
    }
}

/// Create a token block with sequential tokens starting from `start`.
///
/// # Arguments
/// * `start` - Starting token value
/// * `count` - Number of tokens to generate
///
/// # Example
/// ```ignore
/// let block = create_sequential_block(100, 4);  // [100, 101, 102, 103]
/// ```
pub fn create_sequential_block(start: u32, count: usize) -> TokenBlock {
    let tokens: Vec<u32> = (start..start + count as u32).collect();
    create_token_block(&tokens)
}

/// Create a token sequence with multiple blocks.
///
/// # Arguments
/// * `num_blocks` - Number of blocks to create
/// * `block_size` - Tokens per block
/// * `start_token` - Starting token value
///
/// # Returns
/// A TokenBlockSequence containing the requested blocks.
///
/// # Example
/// ```ignore
/// // Create 32 blocks of 4 tokens each, starting from token 0
/// let sequence = create_token_sequence(32, 4, 0);
/// assert_eq!(sequence.blocks().len(), 32);
/// ```
pub fn create_token_sequence(
    num_blocks: usize,
    block_size: usize,
    start_token: u32,
) -> TokenBlockSequence {
    let total_tokens = num_blocks * block_size;
    let tokens: Vec<u32> = (start_token..start_token + total_tokens as u32).collect();
    TokenBlockSequence::from_slice(&tokens, block_size as u32, Some(42))
}

/// Generate sequence hashes from a token sequence.
///
/// # Example
/// ```ignore
/// let sequence = create_token_sequence(10, 4, 0);
/// let hashes = generate_sequence_hashes(&sequence);
/// assert_eq!(hashes.len(), 10);
/// ```
pub fn generate_sequence_hashes(token_sequence: &TokenBlockSequence) -> Vec<SequenceHash> {
    token_sequence
        .blocks()
        .iter()
        .map(|block| block.positional_sequence_hash())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_token_block() {
        let tokens = vec![1, 2, 3, 4];
        let block = create_token_block(&tokens);
        assert_eq!(block.tokens().len(), 4);
    }

    #[test]
    fn test_create_sequential_block() {
        let block = create_sequential_block(100, 4);
        assert_eq!(block.tokens().len(), 4);
    }

    #[test]
    fn test_create_token_sequence() {
        let sequence = create_token_sequence(10, 4, 0);
        assert_eq!(sequence.blocks().len(), 10);

        // Verify first block starts at token 0
        let first_block = &sequence.blocks()[0];
        assert_eq!(first_block.tokens().len(), 4);
    }

    #[test]
    fn test_generate_sequence_hashes() {
        let sequence = create_token_sequence(5, 4, 100);
        let hashes = generate_sequence_hashes(&sequence);

        assert_eq!(hashes.len(), 5);

        // Verify hashes are unique
        let unique_hashes: std::collections::HashSet<_> = hashes.iter().collect();
        assert_eq!(unique_hashes.len(), 5);
    }
}
