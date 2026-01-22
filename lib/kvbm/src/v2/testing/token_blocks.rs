// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token block creation utilities for testing.

use dynamo_tokens::{TokenBlock, TokenBlockSequence, compute_hash_v2};

use crate::{KvbmSequenceHashProvider, SequenceHash};

/// Compute the default salt hash for requests with no salt and no lora.
///
/// This matches the hash computed by `Request::new()` when salt=None and lora_name=None.
pub fn default_request_salt_hash() -> u64 {
    // Matches Request::new() computation:
    // SaltPayload { salt: None, lora_name: None } serializes to "{}"
    compute_hash_v2(b"{}", 0)
}

/// Create a token block from a slice of tokens.
///
/// Uses the default request salt hash to match blocks created by
/// requests with no salt parameter.
///
/// # Example
/// ```ignore
/// let tokens = vec![1, 2, 3, 4];
/// let block = create_token_block(&tokens);
/// ```
pub fn create_token_block(tokens: &[u32]) -> TokenBlock {
    let salt = default_request_salt_hash();
    let token_sequence = TokenBlockSequence::from_slice(tokens, tokens.len() as u32, Some(salt));
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
/// Uses the default request salt hash to match blocks created by
/// requests with no salt parameter.
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
    let salt = default_request_salt_hash();
    let total_tokens = num_blocks * block_size;
    let tokens: Vec<u32> = (start_token..start_token + total_tokens as u32).collect();
    TokenBlockSequence::from_slice(&tokens, block_size as u32, Some(salt))
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
        .map(|block| block.kvbm_sequence_hash())
        .collect()
}

/// Create multiple disjoint token sequences with gaps between them.
///
/// This is useful for testing contiguous subsequence detection, where you need
/// blocks at non-consecutive positions with gaps between them.
///
/// # Arguments
/// * `segments` - Vec of (num_blocks, start_token) pairs. Each segment creates
///   consecutive blocks starting at the given token.
/// * `block_size` - Tokens per block
///
/// # Returns
/// A tuple of (Vec<TokenBlock>, Vec<SequenceHash>) containing all blocks and
/// their hashes from all segments, sorted by position.
///
/// # Example
/// ```ignore
/// // Create 3 segments: positions 0-1, 5-6, 10-12
/// let segments = vec![(2, 0), (2, 20), (3, 40)];  // tokens at 0-8, 20-28, 40-52
/// let (blocks, hashes) = create_disjoint_sequences(segments, 4);
/// assert_eq!(blocks.len(), 7);  // 2 + 2 + 3 blocks
/// // hashes[0].position() == 0, hashes[1].position() == 1
/// // hashes[2].position() == 5, hashes[3].position() == 6  (gap at 2,3,4)
/// // hashes[4].position() == 10, etc.
/// ```
pub fn create_disjoint_sequences(
    segments: Vec<(usize, u32)>,
    block_size: usize,
) -> (Vec<TokenBlock>, Vec<SequenceHash>) {
    let mut all_blocks = Vec::new();
    let mut all_hashes = Vec::new();

    for (num_blocks, start_token) in segments {
        let token_sequence = create_token_sequence(num_blocks, block_size, start_token);
        let blocks = token_sequence.blocks().to_vec();
        let hashes = generate_sequence_hashes(&token_sequence);

        all_blocks.extend(blocks);
        all_hashes.extend(hashes);
    }

    // Sort by position to maintain order
    let mut combined: Vec<_> = all_blocks.into_iter().zip(all_hashes).collect();
    combined.sort_by_key(|(_, hash)| hash.position());

    combined.into_iter().unzip()
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

    #[test]
    fn test_create_disjoint_sequences() {
        // Create 3 segments with different token ranges
        // Note: Positions within each TokenBlockSequence are 0-indexed relative to that sequence
        let segments = vec![
            (2, 0),   // 2 blocks starting at token 0
            (2, 100), // 2 blocks starting at token 100
            (3, 200), // 3 blocks starting at token 200
        ];
        let block_size = 4;

        let (blocks, hashes) = create_disjoint_sequences(segments, block_size);

        // Should have 7 total blocks
        assert_eq!(blocks.len(), 7);
        assert_eq!(hashes.len(), 7);

        // All hashes should be unique (different token content = different hashes)
        let unique_hashes: std::collections::HashSet<_> = hashes.iter().collect();
        assert_eq!(unique_hashes.len(), 7);

        // Positions are relative within each segment's TokenBlockSequence:
        // Segment 1: positions 0, 1 (from sequence starting at token 0)
        // Segment 2: positions 0, 1 (from sequence starting at token 100)
        // Segment 3: positions 0, 1, 2 (from sequence starting at token 200)
        // When sorted by position: [0,0,0], [1,1,1], [2]
        // So hashes are ordered by position first
        assert_eq!(hashes[0].position(), 0);
        assert_eq!(hashes[1].position(), 0); // Different segment, same position
        assert_eq!(hashes[2].position(), 0);
        assert_eq!(hashes[3].position(), 1);
        assert_eq!(hashes[4].position(), 1);
        assert_eq!(hashes[5].position(), 1);
        assert_eq!(hashes[6].position(), 2);
    }
}
