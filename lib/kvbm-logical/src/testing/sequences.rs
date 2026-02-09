// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BlockSequenceBuilder for creating sequences of test blocks.

use std::sync::Arc;

use dynamo_tokens::TokenBlockSequence;

use super::config::DEFAULT_TEST_BLOCK_SIZE;
use crate::BlockId;
use crate::blocks::{
    Block, BlockMetadata,
    state::{Registered, Staged},
};
use crate::pools::SequenceHash;
use crate::registry::BlockRegistry;

use super::blocks::TestBlockBuilder;

/// Builder for creating sequences of blocks with relationships.
///
/// Supports two modes:
/// - Individual: Build blocks with custom configuration
/// - TokenSequence: Build from a realistic token sequence using TokenBlockSequence
pub struct BlockSequenceBuilder<T: BlockMetadata> {
    mode: BuilderMode<T>,
    block_size: usize,
}

enum BuilderMode<T: BlockMetadata> {
    /// Build individual blocks with custom configuration
    Individual { blocks: Vec<TestBlockBuilder<T>> },
    /// Build from a token sequence (more realistic)
    TokenSequence { tokens: Vec<u32>, salt: Option<u64> },
}

impl<T: BlockMetadata + std::fmt::Debug> BlockSequenceBuilder<T> {
    /// Start a new sequence in individual mode.
    pub(crate) fn new() -> Self {
        Self {
            mode: BuilderMode::Individual { blocks: Vec::new() },
            block_size: DEFAULT_TEST_BLOCK_SIZE,
        }
    }

    /// Create from a token sequence (switches to TokenSequence mode).
    pub(crate) fn from_tokens(tokens: Vec<u32>) -> Self {
        Self {
            mode: BuilderMode::TokenSequence { tokens, salt: None },
            block_size: DEFAULT_TEST_BLOCK_SIZE,
        }
    }

    /// Set block size (must be called before building).
    pub(crate) fn with_block_size(mut self, size: usize) -> Self {
        use super::config::validate_test_block_size;
        assert!(
            validate_test_block_size(size),
            "Invalid block size: {}. Must be power of 2 between 1 and 1024",
            size
        );
        self.block_size = size;
        self
    }

    /// Set salt for token sequence mode.
    pub(crate) fn with_salt(mut self, salt: u64) -> Self {
        if let BuilderMode::TokenSequence { tokens, .. } = self.mode {
            self.mode = BuilderMode::TokenSequence {
                tokens,
                salt: Some(salt),
            };
        } else {
            panic!("with_salt() only valid in TokenSequence mode");
        }
        self
    }

    /// Add a block to the sequence (Individual mode only).
    pub(crate) fn add_block(mut self, id: BlockId) -> Self {
        if let BuilderMode::Individual { mut blocks } = self.mode {
            blocks.push(TestBlockBuilder::<T>::new(id).with_block_size(self.block_size));
            self.mode = BuilderMode::Individual { blocks };
        } else {
            panic!("add_block() only valid in Individual mode");
        }
        self
    }

    /// Add a block with specific configuration (Individual mode only).
    pub(crate) fn add_block_with<F>(mut self, id: BlockId, f: F) -> Self
    where
        F: FnOnce(TestBlockBuilder<T>) -> TestBlockBuilder<T>,
    {
        if let BuilderMode::Individual { mut blocks } = self.mode {
            let builder = f(TestBlockBuilder::<T>::new(id).with_block_size(self.block_size));
            blocks.push(builder);
            self.mode = BuilderMode::Individual { blocks };
        } else {
            panic!("add_block_with() only valid in Individual mode");
        }
        self
    }

    /// Build the sequence, returning registered blocks.
    pub(crate) fn build(self) -> Vec<(Block<T, Registered>, SequenceHash)> {
        let registry = Arc::new(BlockRegistry::new());
        let block_size = self.block_size;

        match self.mode {
            BuilderMode::Individual { blocks } => Self::build_individual_static(blocks, registry),
            BuilderMode::TokenSequence { tokens, salt } => {
                Self::build_from_token_sequence_static(tokens, salt, registry, block_size)
            }
        }
    }

    fn build_from_token_sequence_static(
        tokens: Vec<u32>,
        salt: Option<u64>,
        registry: Arc<BlockRegistry>,
        block_size: usize,
    ) -> Vec<(Block<T, Registered>, SequenceHash)> {
        // Validate token count is divisible by block size
        assert_eq!(
            tokens.len() % block_size,
            0,
            "Token count {} must be divisible by block size {}",
            tokens.len(),
            block_size
        );

        // Create TokenBlockSequence
        let token_seq = TokenBlockSequence::from_slice(&tokens, block_size as u32, salt);

        // Convert each TokenBlock to a registered test block
        let mut results = Vec::new();
        let token_blocks = token_seq.blocks();

        for (idx, token_block) in token_blocks.iter().enumerate() {
            let block_id = idx as BlockId;
            let staged: Block<T, Staged> = Block::new(block_id, block_size)
                .complete(token_block)
                .expect("Block size should match");

            let seq_hash = staged.sequence_hash();
            let handle = registry.register_sequence_hash(seq_hash);
            let registered = staged.register_with_handle(handle);

            results.push((registered, seq_hash));
        }

        results
    }

    fn build_individual_static(
        blocks: Vec<TestBlockBuilder<T>>,
        registry: Arc<BlockRegistry>,
    ) -> Vec<(Block<T, Registered>, SequenceHash)> {
        let mut results = Vec::new();

        for builder in blocks {
            let block = builder.build_registered_with_registry(&registry);
            results.push(block);
        }

        results
    }
}

impl<T: BlockMetadata + std::fmt::Debug> Default for BlockSequenceBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}
