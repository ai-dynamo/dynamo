// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token block creation utilities for testing.
//!
//! Note: These are local implementations using workspace-local `dynamo-tokens`
//! types. When kvbm-logical moves to a workspace path dependency, these can
//! be replaced with re-exports from `kvbm_logical::testing`.

use crate::SequenceHash;
use kvbm_common::tokens::{TokenBlock, TokenBlockSequence, compute_hash_v2};
use kvbm_logical::KvbmSequenceHashProvider;

/// Compute the default salt hash for requests with no salt and no lora.
pub fn default_request_salt_hash() -> u64 {
    compute_hash_v2(b"{}", 0)
}

/// Create a token block from a slice of tokens.
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
pub fn create_sequential_block(start: u32, count: usize) -> TokenBlock {
    let tokens: Vec<u32> = (start..start + count as u32).collect();
    create_token_block(&tokens)
}

/// Create a token sequence with multiple blocks.
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

    let mut combined: Vec<_> = all_blocks.into_iter().zip(all_hashes).collect();
    combined.sort_by_key(|(_, hash)| hash.position());

    combined.into_iter().unzip()
}
