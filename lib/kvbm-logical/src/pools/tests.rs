// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities and fixtures for block pool testing.

use super::super::{
    blocks::{state::*, *},
    pools::{backends::*, *},
    testing::{self, TestMeta},
};

/// Re-export TestMeta as TestData for backward compatibility
pub type TestData = TestMeta;

#[cfg(test)]
#[allow(unused, dead_code)]
pub(crate) mod fixtures {
    use super::*;

    use dynamo_tokens::TokenBlock;
    use std::sync::Arc;

    // Re-export from testing module with TestData specialization
    pub use super::testing::tokens_for_id;

    pub fn create_reset_block(id: BlockId) -> Block<TestData, Reset> {
        testing::create_reset_block::<TestData>(id, 4)
    }

    pub fn create_reset_blocks(count: usize) -> Vec<Block<TestData, Reset>> {
        testing::create_reset_blocks::<TestData>(count, 4)
    }

    pub fn create_token_block(tokens: &[u32], block_size: u32) -> TokenBlock {
        testing::create_test_token_block(tokens, block_size)
    }

    pub fn create_complete_block(id: BlockId, tokens: &[u32]) -> Block<TestData, Staged> {
        testing::create_staged_block::<TestData>(id, tokens)
    }

    pub fn create_registered_block(
        id: BlockId,
        tokens: &[u32],
    ) -> (Block<TestData, Registered>, SequenceHash) {
        testing::create_registered_block::<TestData>(id, tokens)
    }

    pub fn create_test_reset_pool(count: usize) -> ResetPool<TestData> {
        testing::TestPoolSetupBuilder::default()
            .block_count(count)
            .build()
            .unwrap()
            .build_reset_pool::<TestData>()
    }

    pub fn create_test_registered_pool() -> (InactivePool<TestData>, ResetPool<TestData>) {
        testing::TestPoolSetupBuilder::default()
            .build()
            .unwrap()
            .build_pools::<TestData>()
    }

    /// Type alias for TestBlockBuilder specialized to TestData
    pub type TestBlockBuilder = testing::TestBlockBuilder<TestData>;

    /// Type alias for BlockSequenceBuilder specialized to TestData
    pub type BlockSequenceBuilder = testing::BlockSequenceBuilder<TestData>;
}

#[cfg(test)]
use fixtures::*;

#[test]
fn test_fill_iota_default_block_size() {
    let block = TestBlockBuilder::new(1).fill_iota(100).build_staged();

    assert_eq!(block.block_id(), 1);
    assert_eq!(block.block_size(), 4);
}

#[test]
fn test_fill_iota_custom_block_size() {
    let block = TestBlockBuilder::new(2)
        .with_block_size(8)
        .fill_iota(200)
        .build_staged();

    assert_eq!(block.block_id(), 2);
    assert_eq!(block.block_size(), 8);
}

#[test]
fn test_with_tokens_overrides_fill_iota() {
    let custom_tokens = vec![99, 98, 97, 96];
    let block = TestBlockBuilder::new(3)
        .fill_iota(100) // This should be overridden
        .with_tokens(custom_tokens)
        .build_staged();

    assert_eq!(block.block_id(), 3);
    assert_eq!(block.block_size(), 4);
}

#[test]
fn test_fill_iota_overrides_with_tokens() {
    let block = TestBlockBuilder::new(4)
        .with_tokens(vec![1, 2, 3, 4]) // This should be overridden
        .fill_iota(500)
        .build_staged();

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
