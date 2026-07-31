// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BlockSequenceBuilder for creating `(BlockId, SequenceHash)` sequences
//! that exercise the lineage / radix-tree relationships between blocks.

use dynamo_tokens::TokenBlockSequence;

use super::config::DEFAULT_TEST_BLOCK_SIZE;
use crate::BlockId;
use crate::KvbmSequenceHashProvider;
use crate::pools::SequenceHash;

/// Builder for sequences of `(BlockId, SequenceHash)` pairs derived from a
/// realistic token sequence (so parent/child lineage relationships line up).
pub struct BlockSequenceBuilder {
    tokens: Vec<u32>,
    salt: Option<dynamo_tokens::SaltHash>,
    block_size: usize,
}

impl BlockSequenceBuilder {
    /// Create from a token sequence.
    pub fn from_tokens(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            salt: None,
            block_size: DEFAULT_TEST_BLOCK_SIZE,
        }
    }

    /// Set block size (must be called before building).
    pub fn with_block_size(mut self, size: usize) -> Self {
        use super::config::validate_test_block_size;
        assert!(
            validate_test_block_size(size),
            "Invalid block size: {}. Must be power of 2 between 1 and 1024",
            size
        );
        self.block_size = size;
        self
    }

    /// Set salt for the underlying TokenBlockSequence.
    pub fn with_salt(mut self, salt: dynamo_tokens::SaltHash) -> Self {
        self.salt = Some(salt);
        self
    }

    /// Build the sequence as `(BlockId, SequenceHash)` pairs. Block ids are
    /// assigned sequentially starting at 0.
    pub fn build(self) -> Vec<(BlockId, SequenceHash)> {
        assert_eq!(
            self.tokens.len() % self.block_size,
            0,
            "Token count {} must be divisible by block size {}",
            self.tokens.len(),
            self.block_size
        );
        let token_seq =
            TokenBlockSequence::from_slice(&self.tokens, self.block_size as u32, self.salt);
        token_seq
            .blocks()
            .iter()
            .enumerate()
            .map(|(idx, tb)| (idx as BlockId, tb.kvbm_sequence_hash()))
            .collect()
    }
}
