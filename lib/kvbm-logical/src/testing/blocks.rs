// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block lifecycle helpers and TestBlockBuilder for tests.

use super::config::DEFAULT_TEST_BLOCK_SIZE;
use crate::BlockId;
use crate::blocks::{
    Block, BlockMetadata,
    state::{Registered, Reset, Staged},
};
use crate::pools::SequenceHash;
use crate::registry::BlockRegistry;

use super::token_blocks::create_test_token_block;

// ============================================================================
// Block lifecycle helpers
// ============================================================================

/// Create a staged (completed but not registered) block.
pub(crate) fn create_staged_block<T: BlockMetadata + std::fmt::Debug>(
    id: BlockId,
    tokens: &[u32],
) -> Block<T, Staged> {
    let token_block = create_test_token_block(tokens, tokens.len() as u32);
    let block: Block<T, Reset> = Block::new(id, tokens.len());
    block.complete(&token_block).expect("Should complete")
}

/// Create a registered block with a new ephemeral registry.
///
/// Returns both the registered block and its sequence hash.
pub(crate) fn create_registered_block<T: BlockMetadata + std::fmt::Debug>(
    id: BlockId,
    tokens: &[u32],
) -> (Block<T, Registered>, SequenceHash) {
    let staged = create_staged_block::<T>(id, tokens);
    let seq_hash = staged.sequence_hash();
    let registry = BlockRegistry::new();
    let handle = registry.register_sequence_hash(seq_hash);
    (staged.register_with_handle(handle), seq_hash)
}

/// Create a reset block with the given ID and block size.
pub(crate) fn create_reset_block<T: BlockMetadata>(
    id: BlockId,
    block_size: usize,
) -> Block<T, Reset> {
    Block::new(id, block_size)
}

/// Create multiple reset blocks with sequential IDs starting from 0.
pub(crate) fn create_reset_blocks<T: BlockMetadata>(
    count: usize,
    block_size: usize,
) -> Vec<Block<T, Reset>> {
    (0..count as BlockId)
        .map(|id| Block::new(id, block_size))
        .collect()
}

// ============================================================================
// Generic TestBlockBuilder<T>
// ============================================================================

/// Enhanced test block builder for consistent block creation.
///
/// Provides a fluent API for creating test blocks with explicit configuration
/// of all parameters, reducing the likelihood of block size mismatches in tests.
pub(crate) struct TestBlockBuilder<T: BlockMetadata> {
    id: BlockId,
    block_size: usize,
    tokens: Option<Vec<u32>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: BlockMetadata> TestBlockBuilder<T> {
    /// Create a new test block builder with the given ID.
    ///
    /// Uses the default test block size (4) but allows override.
    pub(crate) fn new(id: BlockId) -> Self {
        Self {
            id,
            block_size: DEFAULT_TEST_BLOCK_SIZE,
            tokens: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the block size for this test block.
    ///
    /// The block size must be a power of 2 between 1 and 1024.
    pub(crate) fn with_block_size(mut self, size: usize) -> Self {
        use super::config::validate_test_block_size;
        assert!(
            validate_test_block_size(size),
            "Invalid test block size: {}. Must be power of 2 between 1 and 1024",
            size
        );
        self.block_size = size;
        self
    }

    /// Set specific tokens for this test block.
    ///
    /// If not specified, tokens will be auto-generated to match the block size.
    /// The length of the tokens vector should match the configured block size.
    pub(crate) fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Fill with sequential tokens starting from the given value.
    ///
    /// Generates tokens [start, start+1, start+2, ...] up to block_size.
    /// This is mutually exclusive with `with_tokens()` - the last one called wins.
    pub(crate) fn fill_iota(mut self, start: u32) -> Self {
        let tokens: Vec<u32> = (start..start + self.block_size as u32).collect();
        self.tokens = Some(tokens);
        self
    }

    /// Build a Staged (Complete) state block.
    ///
    /// This will auto-generate tokens if none were provided, ensuring they
    /// match the configured block size.
    pub(crate) fn build_staged(self) -> Block<T, Staged>
    where
        T: std::fmt::Debug,
    {
        use super::config::generate_test_tokens;

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

        let token_block = create_test_token_block(&tokens, self.block_size as u32);
        Block::new(self.id, self.block_size)
            .complete(&token_block)
            .expect("Block size should match token block size")
    }

    /// Build a Registered state block with a specific registry.
    ///
    /// Creates a staged block and registers it with the provided registry.
    pub(crate) fn build_registered_with_registry(
        self,
        registry: &BlockRegistry,
    ) -> (Block<T, Registered>, SequenceHash)
    where
        T: std::fmt::Debug,
    {
        let staged = self.build_staged();
        let seq_hash = staged.sequence_hash();
        let handle = registry.register_sequence_hash(seq_hash);
        (staged.register_with_handle(handle), seq_hash)
    }
}
