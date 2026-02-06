// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test fixtures for `v2/logical/` module tests.
//!
//! This module consolidates duplicated test helpers across registry, pools, and manager tests.
//! It uses internal types like `Block<T, Staged>` so it must be behind `#[cfg(test)]`.
//!
//! Note: `v2/testing/` uses only the public API and remains separate.

use std::sync::Arc;

use derive_builder::Builder;
use dynamo_tokens::{TokenBlock, TokenBlockSequence};

use crate::test_config::DEFAULT_TEST_BLOCK_SIZE;
use crate::BlockId;

use crate::blocks::{
    Block, BlockMetadata,
    state::{Registered, Reset, Staged},
};
use crate::pools::{
    InactivePool, ResetPool, ReusePolicy, SequenceHash,
    backends::{FifoReusePolicy, HashMapBackend},
};
use crate::registry::{BlockRegistrationHandle, BlockRegistry};

// ============================================================================
// Canonical test metadata types
// ============================================================================

/// Primary test metadata type used across all v2 tests.
/// Replaces: `TestMetadata` (registry), `TestData` (pools), `TestBlockData` (manager)
#[derive(Debug, Clone, PartialEq)]
pub struct TestMeta {
    pub value: u64,
}

/// Multi-type registry test metadata A
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataA(pub u32);

/// Multi-type registry test metadata B
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataB(pub String);

/// Multi-type registry test metadata C (unit struct)
#[derive(Debug, Clone, PartialEq)]
pub struct MetadataC;

// ============================================================================
// Constants
// ============================================================================

/// Standard salt value for test token blocks.
/// Standardizes mixed salt values (42 in pools/manager, 1337 in registry).
pub const TEST_SALT: u64 = 42;

// ============================================================================
// Token block helpers
// ============================================================================

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

// ============================================================================
// Block lifecycle helpers
// ============================================================================

/// Create a staged (completed but not registered) block.
pub fn create_staged_block<T: BlockMetadata + std::fmt::Debug>(
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
pub fn create_registered_block<T: BlockMetadata + std::fmt::Debug>(
    id: BlockId,
    tokens: &[u32],
) -> (Block<T, Registered>, SequenceHash) {
    let staged = create_staged_block::<T>(id, tokens);
    let seq_hash = staged.sequence_hash();
    let registry = BlockRegistry::new();
    let handle = registry.register_sequence_hash(seq_hash);
    (staged.register_with_handle(handle), seq_hash)
}

/// Register a test block with a specific registry.
///
/// Returns the registered block, sequence hash, and registration handle.
pub fn register_test_block<T: BlockMetadata + std::fmt::Debug>(
    registry: &BlockRegistry,
    block_id: BlockId,
    tokens: &[u32],
) -> (Block<T, Registered>, SequenceHash, BlockRegistrationHandle) {
    let staged = create_staged_block::<T>(block_id, tokens);
    let seq_hash = staged.sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);
    (staged.register_with_handle(handle.clone()), seq_hash, handle)
}

/// Create a reset block with the given ID and block size.
pub fn create_reset_block<T: BlockMetadata>(id: BlockId, block_size: usize) -> Block<T, Reset> {
    Block::new(id, block_size)
}

/// Create multiple reset blocks with sequential IDs starting from 0.
pub fn create_reset_blocks<T: BlockMetadata>(count: usize, block_size: usize) -> Vec<Block<T, Reset>> {
    (0..count as BlockId)
        .map(|id| Block::new(id, block_size))
        .collect()
}

/// Generate tokens for a given block ID (for unique but deterministic test data).
pub fn tokens_for_id(id: u64) -> Vec<u32> {
    vec![id as u32, (id + 1) as u32, (id + 2) as u32, (id + 3) as u32]
}

// ============================================================================
// Pool setup with derive_builder
// ============================================================================

/// Configuration for setting up test pools.
#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct TestPoolSetup {
    #[builder(default = "10")]
    pub block_count: usize,

    #[builder(default = "4")]
    pub block_size: usize,

    #[builder(default = "Box::new(FifoReusePolicy::new())")]
    pub reuse_policy: Box<dyn ReusePolicy>,
}

impl TestPoolSetup {
    /// Build a reset pool with the configured settings.
    pub fn build_reset_pool<T: BlockMetadata>(&self) -> ResetPool<T> {
        let blocks = create_reset_blocks::<T>(self.block_count, self.block_size);
        ResetPool::new(blocks, self.block_size)
    }

    /// Build both inactive and reset pools with the configured settings.
    pub fn build_pools<T: BlockMetadata>(&self) -> (InactivePool<T>, ResetPool<T>) {
        let reset_pool = self.build_reset_pool::<T>();
        let reuse_policy = Box::new(FifoReusePolicy::new());
        let backend = Box::new(HashMapBackend::new(reuse_policy));
        let inactive_pool = InactivePool::new(backend, &reset_pool);
        (inactive_pool, reset_pool)
    }
}

// ============================================================================
// Generic TestBlockBuilder<T>
// ============================================================================

/// Enhanced test block builder for consistent block creation.
///
/// Provides a fluent API for creating test blocks with explicit configuration
/// of all parameters, reducing the likelihood of block size mismatches in tests.
pub struct TestBlockBuilder<T: BlockMetadata> {
    id: BlockId,
    block_size: usize,
    tokens: Option<Vec<u32>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: BlockMetadata> TestBlockBuilder<T> {
    /// Create a new test block builder with the given ID.
    ///
    /// Uses the default test block size (4) but allows override.
    pub fn new(id: BlockId) -> Self {
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
    pub fn with_block_size(mut self, size: usize) -> Self {
        use crate::test_config::validate_test_block_size;
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
    pub fn with_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.tokens = Some(tokens);
        self
    }

    /// Fill with sequential tokens starting from the given value.
    ///
    /// Generates tokens [start, start+1, start+2, ...] up to block_size.
    /// This is mutually exclusive with `with_tokens()` - the last one called wins.
    pub fn fill_iota(mut self, start: u32) -> Self {
        let tokens: Vec<u32> = (start..start + self.block_size as u32).collect();
        self.tokens = Some(tokens);
        self
    }

    /// Build a Reset state block.
    pub fn build_reset(self) -> Block<T, Reset> {
        Block::new(self.id, self.block_size)
    }

    /// Build a Staged (Complete) state block.
    ///
    /// This will auto-generate tokens if none were provided, ensuring they
    /// match the configured block size.
    pub fn build_staged(self) -> Block<T, Staged>
    where
        T: std::fmt::Debug,
    {
        use crate::test_config::generate_test_tokens;

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

    /// Build a Registered state block.
    ///
    /// Creates a staged block and registers it with a new ephemeral registry.
    pub fn build_registered(self) -> (Block<T, Registered>, SequenceHash)
    where
        T: std::fmt::Debug,
    {
        let staged = self.build_staged();
        let seq_hash = staged.sequence_hash();
        let registry = BlockRegistry::new();
        let handle = registry.register_sequence_hash(seq_hash);
        (staged.register_with_handle(handle), seq_hash)
    }

    /// Build a Registered state block with a specific registry.
    ///
    /// Creates a staged block and registers it with the provided registry.
    pub fn build_registered_with_registry(
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

    /// Get the configured block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the configured block ID.
    pub fn block_id(&self) -> BlockId {
        self.id
    }
}

// ============================================================================
// Generic BlockSequenceBuilder<T>
// ============================================================================

/// Builder for creating sequences of blocks with relationships.
///
/// Supports two modes:
/// - Individual: Build blocks with custom configuration
/// - TokenSequence: Build from a realistic token sequence using TokenBlockSequence
pub struct BlockSequenceBuilder<T: BlockMetadata> {
    mode: BuilderMode<T>,
    registry: Option<Arc<BlockRegistry>>,
    block_size: usize,
}

enum BuilderMode<T: BlockMetadata> {
    /// Build individual blocks with custom configuration
    Individual {
        blocks: Vec<TestBlockBuilder<T>>,
        parent_hash: Option<u64>,
    },
    /// Build from a token sequence (more realistic)
    TokenSequence { tokens: Vec<u32>, salt: Option<u64> },
}

impl<T: BlockMetadata + std::fmt::Debug> BlockSequenceBuilder<T> {
    /// Start a new sequence in individual mode.
    pub fn new() -> Self {
        Self {
            mode: BuilderMode::Individual {
                blocks: Vec::new(),
                parent_hash: None,
            },
            registry: None,
            block_size: DEFAULT_TEST_BLOCK_SIZE,
        }
    }

    /// Create from a token sequence (switches to TokenSequence mode).
    pub fn from_tokens(tokens: Vec<u32>) -> Self {
        Self {
            mode: BuilderMode::TokenSequence { tokens, salt: None },
            registry: None,
            block_size: DEFAULT_TEST_BLOCK_SIZE,
        }
    }

    /// Set block size (must be called before building).
    pub fn with_block_size(mut self, size: usize) -> Self {
        use crate::test_config::validate_test_block_size;
        assert!(
            validate_test_block_size(size),
            "Invalid block size: {}. Must be power of 2 between 1 and 1024",
            size
        );
        self.block_size = size;
        self
    }

    /// Set salt for token sequence mode.
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

    /// Use a specific registry (otherwise creates a new one).
    pub fn with_registry(mut self, registry: Arc<BlockRegistry>) -> Self {
        self.registry = Some(registry);
        self
    }

    /// Set parent hash (Individual mode only).
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

    /// Add a block to the sequence (Individual mode only).
    pub fn add_block(mut self, id: BlockId) -> Self {
        if let BuilderMode::Individual {
            mut blocks,
            parent_hash,
        } = self.mode
        {
            blocks.push(TestBlockBuilder::<T>::new(id).with_block_size(self.block_size));
            self.mode = BuilderMode::Individual {
                blocks,
                parent_hash,
            };
        } else {
            panic!("add_block() only valid in Individual mode");
        }
        self
    }

    /// Add a block with specific configuration (Individual mode only).
    pub fn add_block_with<F>(mut self, id: BlockId, f: F) -> Self
    where
        F: FnOnce(TestBlockBuilder<T>) -> TestBlockBuilder<T>,
    {
        if let BuilderMode::Individual {
            mut blocks,
            parent_hash,
        } = self.mode
        {
            let builder = f(TestBlockBuilder::<T>::new(id).with_block_size(self.block_size));
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

    /// Build the sequence, returning registered blocks.
    pub fn build(self) -> Vec<(Block<T, Registered>, SequenceHash)> {
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
        _parent_hash: Option<u64>, // TODO: Implement parent hash support
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

// ============================================================================
// Test manager helper
// ============================================================================

use crate::manager::{BlockManager, FrequencyTrackingCapacity};

/// Create a basic test manager with LRU backend.
pub fn create_test_manager<T: BlockMetadata>(block_count: usize) -> BlockManager<T> {
    let registry = BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
        .build();

    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(4) // Most tests use 4-token blocks
        .registry(registry)
        .with_lru_backend()
        .build()
        .expect("Should build manager")
}

/// Create a test manager with custom block size.
pub fn create_test_manager_with_block_size<T: BlockMetadata>(
    block_count: usize,
    block_size: usize,
) -> BlockManager<T> {
    let registry = BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
        .build();

    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(block_size)
        .registry(registry)
        .with_lru_backend()
        .build()
        .expect("Should build manager")
}
