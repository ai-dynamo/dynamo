// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities and fixtures for block pool testing.

use super::super::{
    blocks::{state::*, *},
    pools::{backends::*, *},
};

/// Common test data type used across all v2 block manager tests
#[derive(Debug, Clone, PartialEq)]
pub struct TestData {
    pub value: u64,
}

#[cfg(test)]
#[allow(unused, dead_code)]
pub(crate) mod fixtures {
    use super::*;

    use dynamo_tokens::{TokenBlock, TokenBlockSequence};
    use std::sync::Arc;

    pub fn create_reset_block(id: BlockId) -> Block<TestData, Reset> {
        Block::new(id, 4)
    }

    pub fn create_reset_blocks(count: usize) -> Vec<Block<TestData, Reset>> {
        (0..count as BlockId).map(|id| Block::new(id, 4)).collect()
    }

    pub fn create_token_block(tokens: &[u32], block_size: u32) -> TokenBlock {
        let sequence = TokenBlockSequence::from_slice(tokens, block_size, Some(42));
        if let Some(block) = sequence.blocks().first() {
            block.clone()
        } else {
            let mut partial = sequence.into_parts().1;
            partial.commit().expect("Should be able to commit")
        }
    }

    pub fn create_complete_block(id: BlockId, tokens: &[u32]) -> Block<TestData, Complete> {
        let token_block = create_token_block(tokens, 4);
        Block::new(id, 4)
            .complete(token_block)
            .expect("Block size should match")
    }

    pub fn create_registered_block(
        id: BlockId,
        tokens: &[u32],
    ) -> (Block<TestData, Registered>, SequenceHash) {
        let complete_block = create_complete_block(id, tokens);
        let seq_hash = complete_block.sequence_hash();
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(seq_hash);
        (complete_block.register(handle), seq_hash)
    }

    pub fn create_test_reset_pool(count: usize) -> ResetPool<TestData> {
        let blocks = create_reset_blocks(count);
        ResetPool::new(blocks, 4)
    }

    pub fn create_test_registered_pool() -> (InactivePool<TestData>, ResetPool<TestData>) {
        let reuse_policy = Box::new(FifoReusePolicy::new());
        let backend = Box::new(HashMapBackend::new(reuse_policy));
        let reset_blocks = create_reset_blocks(10);
        let reset_pool = ResetPool::new(reset_blocks, 4);
        let inactive_pool = InactivePool::new(backend, &reset_pool);
        (inactive_pool, reset_pool)
    }

    pub fn tokens_for_id(id: u64) -> Vec<u32> {
        vec![id as u32, (id + 1) as u32, (id + 2) as u32, (id + 3) as u32]
    }

    /// Enhanced test block builder for consistent block creation
    ///
    /// This builder provides a fluent API for creating test blocks with
    /// explicit configuration of all parameters, reducing the likelihood
    /// of block size mismatches in tests.
    pub struct TestBlockBuilder {
        id: BlockId,
        block_size: usize,
        tokens: Option<Vec<u32>>,
    }

    impl TestBlockBuilder {
        /// Create a new test block builder with the given ID
        ///
        /// Uses the default test block size (4) but allows override.
        pub fn new(id: BlockId) -> Self {
            use crate::v2::test_config::DEFAULT_TEST_BLOCK_SIZE;
            Self {
                id,
                block_size: DEFAULT_TEST_BLOCK_SIZE,
                tokens: None,
            }
        }

        /// Set the block size for this test block
        ///
        /// The block size must be a power of 2 between 1 and 1024.
        pub fn with_block_size(mut self, size: usize) -> Self {
            use crate::v2::test_config::validate_test_block_size;
            assert!(
                validate_test_block_size(size),
                "Invalid test block size: {}. Must be power of 2 between 1 and 1024",
                size
            );
            self.block_size = size;
            self
        }

        /// Set specific tokens for this test block
        ///
        /// If not specified, tokens will be auto-generated to match the block size.
        /// The length of the tokens vector should match the configured block size.
        pub fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
            self.tokens = Some(tokens);
            self
        }

        /// Fill with sequential tokens starting from the given value
        ///
        /// Generates tokens [start, start+1, start+2, ...] up to block_size.
        /// This is mutually exclusive with `with_tokens()` - the last one called wins.
        pub fn fill_iota(mut self, start: u32) -> Self {
            let tokens: Vec<u32> = (start..start + self.block_size as u32).collect();
            self.tokens = Some(tokens);
            self
        }

        /// Build a Reset state block
        pub fn build_reset(self) -> Block<TestData, Reset> {
            Block::new(self.id, self.block_size)
        }

        /// Build a Complete state block
        ///
        /// This will auto-generate tokens if none were provided, ensuring they
        /// match the configured block size.
        pub fn build_complete(self) -> Block<TestData, Complete> {
            use crate::v2::test_config::generate_test_tokens;

            // Auto-generate tokens if not provided
            let tokens = self
                .tokens
                .unwrap_or_else(|| generate_test_tokens(self.id as u32 * 100, self.block_size));

            // Validate token count matches block size
            assert_eq!(
                tokens.len(),
                self.block_size,
                "Token count {} doesn't match block size {}",
                tokens.len(),
                self.block_size
            );

            let token_block = create_token_block(&tokens, self.block_size as u32);
            Block::new(self.id, self.block_size)
                .complete(token_block)
                .expect("Block size should match token block size")
        }

        /// Build a Registered state block
        ///
        /// This creates a complete block and then registers it with a new registry.
        pub fn build_registered(self) -> (Block<TestData, Registered>, SequenceHash) {
            let complete_block = self.build_complete();
            let seq_hash = complete_block.sequence_hash();
            let registry = BlockRegistry::new();
            let handle = registry.register_sequence_hash(seq_hash);
            (complete_block.register(handle), seq_hash)
        }

        /// Build a Registered state block with a specific registry
        ///
        /// This creates a complete block and then registers it with the provided registry.
        pub fn build_registered_with_registry(
            self,
            registry: &BlockRegistry,
        ) -> (Block<TestData, Registered>, SequenceHash) {
            let complete_block = self.build_complete();
            let seq_hash = complete_block.sequence_hash();
            let handle = registry.register_sequence_hash(seq_hash);
            (complete_block.register(handle), seq_hash)
        }

        /// Get the configured block size
        pub fn block_size(&self) -> usize {
            self.block_size
        }

        /// Get the configured block ID
        pub fn block_id(&self) -> BlockId {
            self.id
        }
    }

    /// Builder for creating sequences of blocks with relationships
    ///
    /// Supports two modes:
    /// - Individual: Build blocks with custom configuration
    /// - TokenSequence: Build from a realistic token sequence using TokenBlockSequence
    pub struct BlockSequenceBuilder {
        mode: BuilderMode,
        registry: Option<Arc<BlockRegistry>>,
        block_size: usize,
    }

    enum BuilderMode {
        /// Build individual blocks with custom configuration
        Individual {
            blocks: Vec<TestBlockBuilder>,
            parent_hash: Option<u64>,
        },
        /// Build from a token sequence (more realistic)
        TokenSequence { tokens: Vec<u32>, salt: Option<u64> },
    }

    impl BlockSequenceBuilder {
        /// Start a new sequence in individual mode
        pub fn new() -> Self {
            use crate::v2::test_config::DEFAULT_TEST_BLOCK_SIZE;
            Self {
                mode: BuilderMode::Individual {
                    blocks: Vec::new(),
                    parent_hash: None,
                },
                registry: None,
                block_size: DEFAULT_TEST_BLOCK_SIZE,
            }
        }

        /// Create from a token sequence (switches to TokenSequence mode)
        pub fn from_tokens(tokens: Vec<u32>) -> Self {
            use crate::v2::test_config::DEFAULT_TEST_BLOCK_SIZE;
            Self {
                mode: BuilderMode::TokenSequence { tokens, salt: None },
                registry: None,
                block_size: DEFAULT_TEST_BLOCK_SIZE,
            }
        }

        /// Set block size (must be called before building)
        pub fn with_block_size(mut self, size: usize) -> Self {
            use crate::v2::test_config::validate_test_block_size;
            assert!(
                validate_test_block_size(size),
                "Invalid block size: {}. Must be power of 2 between 1 and 1024",
                size
            );
            self.block_size = size;
            self
        }

        /// Set salt for token sequence mode
        pub fn with_salt(mut self, salt: u64) -> Self {
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

        /// Use a specific registry (otherwise creates a new one)
        pub fn with_registry(mut self, registry: Arc<BlockRegistry>) -> Self {
            self.registry = Some(registry);
            self
        }

        /// Set parent hash (Individual mode only)
        pub fn with_parent(mut self, parent_hash: u64) -> Self {
            if let BuilderMode::Individual { blocks, .. } = self.mode {
                self.mode = BuilderMode::Individual {
                    blocks,
                    parent_hash: Some(parent_hash),
                };
            } else {
                panic!("with_parent() only valid in Individual mode");
            }
            self
        }

        /// Add a block to the sequence (Individual mode only)
        pub fn add_block(mut self, id: BlockId) -> Self {
            if let BuilderMode::Individual {
                mut blocks,
                parent_hash,
            } = self.mode
            {
                blocks.push(TestBlockBuilder::new(id).with_block_size(self.block_size));
                self.mode = BuilderMode::Individual {
                    blocks,
                    parent_hash,
                };
            } else {
                panic!("add_block() only valid in Individual mode");
            }
            self
        }

        /// Add a block with specific configuration (Individual mode only)
        pub fn add_block_with<F>(mut self, id: BlockId, f: F) -> Self
        where
            F: FnOnce(TestBlockBuilder) -> TestBlockBuilder,
        {
            if let BuilderMode::Individual {
                mut blocks,
                parent_hash,
            } = self.mode
            {
                let builder = f(TestBlockBuilder::new(id).with_block_size(self.block_size));
                blocks.push(builder);
                self.mode = BuilderMode::Individual {
                    blocks,
                    parent_hash,
                };
            } else {
                panic!("add_block_with() only valid in Individual mode");
            }
            self
        }

        /// Build the sequence, returning registered blocks
        pub fn build(self) -> Vec<(Block<TestData, Registered>, SequenceHash)> {
            let registry = self
                .registry
                .unwrap_or_else(|| Arc::new(BlockRegistry::new()));
            let block_size = self.block_size;

            match self.mode {
                BuilderMode::Individual {
                    blocks,
                    parent_hash,
                } => Self::build_individual_static(blocks, parent_hash, registry),
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
        ) -> Vec<(Block<TestData, Registered>, SequenceHash)> {
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
                let complete_block = Block::new(block_id, block_size)
                    .complete(token_block.clone())
                    .expect("Block size should match");

                let seq_hash = complete_block.sequence_hash();
                let handle = registry.register_sequence_hash(seq_hash);
                let registered = complete_block.register(handle);

                results.push((registered, seq_hash));
            }

            results
        }

        fn build_individual_static(
            blocks: Vec<TestBlockBuilder>,
            _parent_hash: Option<u64>, // TODO: Implement parent hash support
            registry: Arc<BlockRegistry>,
        ) -> Vec<(Block<TestData, Registered>, SequenceHash)> {
            let mut results = Vec::new();

            for builder in blocks {
                let block = builder.build_registered_with_registry(&registry);
                results.push(block);
            }

            results
        }
    }
}

#[cfg(test)]
use fixtures::*;

#[test]
fn test_fill_iota_default_block_size() {
    let block = TestBlockBuilder::new(1).fill_iota(100).build_complete();

    assert_eq!(block.block_id(), 1);
    assert_eq!(block.block_size(), 4);
}

#[test]
fn test_fill_iota_custom_block_size() {
    let block = TestBlockBuilder::new(2)
        .with_block_size(8)
        .fill_iota(200)
        .build_complete();

    assert_eq!(block.block_id(), 2);
    assert_eq!(block.block_size(), 8);
}

#[test]
fn test_with_tokens_overrides_fill_iota() {
    let custom_tokens = vec![99, 98, 97, 96];
    let block = TestBlockBuilder::new(3)
        .fill_iota(100) // This should be overridden
        .with_tokens(custom_tokens)
        .build_complete();

    assert_eq!(block.block_id(), 3);
    assert_eq!(block.block_size(), 4);
}

#[test]
fn test_fill_iota_overrides_with_tokens() {
    let block = TestBlockBuilder::new(4)
        .with_tokens(vec![1, 2, 3, 4]) // This should be overridden
        .fill_iota(500)
        .build_complete();

    assert_eq!(block.block_id(), 4);
    assert_eq!(block.block_size(), 4);
}

#[test]
fn test_block_sequence_from_tokens() {
    let tokens = vec![100, 101, 102, 103, 104, 105, 106, 107]; // 2 blocks of size 4
    let blocks = BlockSequenceBuilder::from_tokens(tokens)
        .with_block_size(4)
        .with_salt(42)
        .build();

    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].0.block_id(), 0);
    assert_eq!(blocks[1].0.block_id(), 1);
    assert_eq!(blocks[0].0.block_size(), 4);
    assert_eq!(blocks[1].0.block_size(), 4);
}

#[test]
fn test_block_sequence_individual_mode() {
    let blocks = BlockSequenceBuilder::new()
        .add_block_with(1, |b| b.fill_iota(100))
        .add_block_with(2, |b| b.fill_iota(200))
        .add_block(3)
        .build();

    assert_eq!(blocks.len(), 3);
    assert_eq!(blocks[0].0.block_id(), 1);
    assert_eq!(blocks[1].0.block_id(), 2);
    assert_eq!(blocks[2].0.block_id(), 3);
}

#[test]
#[should_panic(expected = "Token count 7 must be divisible by block size 4")]
fn test_block_sequence_invalid_token_count() {
    let tokens = vec![1, 2, 3, 4, 5, 6, 7]; // 7 tokens, not divisible by 4
    BlockSequenceBuilder::from_tokens(tokens)
        .with_block_size(4)
        .build();
}

#[test]
fn test_block_sequence_custom_block_size() {
    let tokens: Vec<u32> = (0..16).collect(); // 2 blocks of size 8
    let blocks = BlockSequenceBuilder::from_tokens(tokens)
        .with_block_size(8)
        .build();

    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].0.block_size(), 8);
    assert_eq!(blocks[1].0.block_size(), 8);
}
