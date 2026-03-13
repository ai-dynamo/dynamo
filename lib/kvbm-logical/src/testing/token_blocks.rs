// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token block creation helpers for tests.

use dynamo_tokens::{TokenBlock, TokenBlockSequence, compute_hash_v2};

use crate::{KvbmSequenceHashProvider, SequenceHash};

use super::TEST_SALT;

/// Create a token block from a slice of tokens with standard test salt.
///
/// If the token count matches block_size, returns a complete block.
/// Otherwise attempts to commit a partial block.
pub fn create_test_token_block(tokens: &[u32], block_size: u32) -> TokenBlock {
    let sequence = TokenBlockSequence::from_slice(tokens, block_size, Some(TEST_SALT));
    if let Some(block) = sequence.blocks().first() {
        block.clone()
    } else {
        let mut partial = sequence.into_parts().1;
        partial.commit().expect("Should be able to commit")
    }
}

/// Create a token block with sequential tokens starting from `start`.
///
/// Generates tokens [start, start+1, ..., start+block_size-1].
pub fn create_iota_token_block(start: u32, block_size: u32) -> TokenBlock {
    let tokens: Vec<u32> = (start..start + block_size).collect();
    create_test_token_block(&tokens, block_size)
}

/// Generate a vector of sequential tokens.
pub fn sequential_tokens(start: u32, count: usize) -> Vec<u32> {
    (start..start + count as u32).collect()
}

/// Generate tokens for a given block ID (for unique but deterministic test data).
pub fn tokens_for_id(id: u64) -> Vec<u32> {
    vec![id as u32, (id + 1) as u32, (id + 2) as u32, (id + 3) as u32]
}

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

        // Positions are relative within each segment's TokenBlockSequence
        assert_eq!(hashes[0].position(), 0);
        assert_eq!(hashes[1].position(), 0);
        assert_eq!(hashes[2].position(), 0);
        assert_eq!(hashes[3].position(), 1);
        assert_eq!(hashes[4].position(), 1);
        assert_eq!(hashes[5].position(), 1);
        assert_eq!(hashes[6].position(), 2);
    }
}
