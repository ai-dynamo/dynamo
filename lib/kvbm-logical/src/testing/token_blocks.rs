// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token block creation helpers for tests.

use dynamo_tokens::{TokenBlock, TokenBlockSequence};

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
