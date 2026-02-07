// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities and fixtures for block pool testing.

use super::super::{
    blocks::{state::*, *},
    pools::*,
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

#[test]
fn test_mutable_block_complete_error_returns_block() {
    use crate::blocks::BlockError;

    let reset_pool = create_test_reset_pool(5);
    let mut mutable_blocks = reset_pool.allocate_blocks(1);
    let mutable_block = mutable_blocks.pop().unwrap();
    let original_block_id = mutable_block.block_id();

    // block_size is 4, but token block has 8 tokens
    let big_token_block = testing::create_test_token_block(&[1, 2, 3, 4, 5, 6, 7, 8], 8);

    let result = mutable_block.complete(&big_token_block);
    assert!(result.is_err());

    match result {
        Err(BlockError::BlockSizeMismatch {
            expected,
            actual,
            block: recovered_block,
        }) => {
            assert_eq!(expected, 4);
            assert_eq!(actual, 8);
            // Block is recoverable from the error
            assert_eq!(recovered_block.block_id(), original_block_id);
        }
        _ => panic!("Expected BlockSizeMismatch error"),
    }
}

#[test]
fn test_mutable_block_stage_and_debug() {
    let reset_pool = create_test_reset_pool(5);
    let mut mutable_blocks = reset_pool.allocate_blocks(1);
    let mutable_block = mutable_blocks.pop().unwrap();

    // Exercise Debug for MutableBlock
    let debug_str = format!("{:?}", mutable_block);
    assert!(debug_str.contains("MutableBlock"));

    // Exercise the `stage` method (bypass block_size check)
    let seq_hash = crate::KvbmSequenceHashProvider::kvbm_sequence_hash(
        &testing::create_test_token_block(&[10, 11, 12, 13], 4),
    );
    let complete_block = mutable_block
        .stage(seq_hash, 4)
        .expect("block size should match");
    assert_eq!(complete_block.sequence_hash(), seq_hash);
}

#[test]
fn test_complete_block_reset() {
    let reset_pool = create_test_reset_pool(5);
    let mut mutable_blocks = reset_pool.allocate_blocks(1);
    let mutable_block = mutable_blocks.pop().unwrap();
    let original_block_id = mutable_block.block_id();

    let token_block = create_token_block(&[10, 11, 12, 13], 4);
    let complete_block = mutable_block
        .complete(&token_block)
        .expect("Should complete");

    assert_eq!(complete_block.block_id(), original_block_id);

    // Reset the complete block back to a mutable block
    let reset_mutable = complete_block.reset();
    assert_eq!(reset_mutable.block_id(), original_block_id);
}

#[test]
fn test_immutable_block_downgrade_and_upgrade() {
    let manager = testing::create_test_manager::<TestData>(10);

    let token_block = testing::create_iota_token_block(100, 4);
    let seq_hash = crate::KvbmSequenceHashProvider::kvbm_sequence_hash(&token_block);

    let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
    let complete_block = mutable_blocks
        .into_iter()
        .next()
        .unwrap()
        .complete(&token_block)
        .expect("Should complete");

    let immutable_blocks = manager.register_blocks(vec![complete_block]);
    let immutable_block = immutable_blocks.into_iter().next().unwrap();

    // Check accessors
    assert_eq!(immutable_block.sequence_hash(), seq_hash);
    let _block_id = immutable_block.block_id();
    let _handle = immutable_block.registration_handle();
    assert!(immutable_block.use_count() >= 1);

    // Downgrade to WeakBlock
    let weak_block = immutable_block.downgrade();
    assert_eq!(weak_block.sequence_hash(), seq_hash);

    // Upgrade while original is alive — should succeed via direct Weak path
    let upgraded = weak_block
        .upgrade()
        .expect("Should upgrade while original alive");
    assert_eq!(upgraded.sequence_hash(), seq_hash);
    assert_eq!(upgraded.block_id(), immutable_block.block_id());
}

#[test]
fn test_weak_block_upgrade_via_upgrade_fn() {
    let manager = testing::create_test_manager::<TestData>(10);

    let token_block = testing::create_iota_token_block(200, 4);
    let seq_hash = crate::KvbmSequenceHashProvider::kvbm_sequence_hash(&token_block);

    // Create weak block, then drop the original
    let weak_block = {
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete");
        let immutable_blocks = manager.register_blocks(vec![complete_block]);
        let immutable_block = immutable_blocks.into_iter().next().unwrap();
        immutable_block.downgrade()
    }; // original dropped — block returns to inactive pool

    // Upgrade should succeed via the upgrade_fn path (finds block in inactive pool)
    let upgraded = weak_block
        .upgrade()
        .expect("upgrade should succeed via upgrade_fn");
    assert_eq!(upgraded.sequence_hash(), seq_hash);
}

#[test]
fn test_immutable_and_weak_block_debug() {
    let manager = testing::create_test_manager::<TestData>(10);

    let token_block = testing::create_iota_token_block(300, 4);
    let _seq_hash = crate::KvbmSequenceHashProvider::kvbm_sequence_hash(&token_block);

    let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
    let complete_block = mutable_blocks
        .into_iter()
        .next()
        .unwrap()
        .complete(&token_block)
        .expect("Should complete");

    let immutable_blocks = manager.register_blocks(vec![complete_block]);
    let immutable_block = immutable_blocks.into_iter().next().unwrap();

    // Exercise Debug for ImmutableBlock
    let debug_str = format!("{:?}", immutable_block);
    assert!(debug_str.contains("ImmutableBlock"));

    // Exercise Debug for WeakBlock
    let weak_block = immutable_block.downgrade();
    let weak_debug_str = format!("{:?}", weak_block);
    assert!(weak_debug_str.contains("WeakBlock"));
}

#[test]
fn test_weak_block_upgrade_fails_when_evicted() {
    let manager = testing::create_test_manager::<TestData>(10);

    let token_block = testing::create_test_token_block(&[999, 998, 997, 996], 4);

    let weak_block = {
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete");
        let immutable_blocks = manager.register_blocks(vec![complete_block]);
        let immutable_block = immutable_blocks.into_iter().next().unwrap();
        immutable_block.downgrade()
    };

    // Fill up the pool with other blocks to force eviction of original
    for i in 0..10 {
        let tokens = vec![1000 + i, 1001 + i, 1002 + i, 1003 + i];
        let tb = testing::create_test_token_block(&tokens, 4);
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&tb)
            .expect("Should complete");
        let _immutable = manager.register_blocks(vec![complete_block]);
    }

    // Upgrade should fail since original was evicted
    let result = weak_block.upgrade();
    assert!(result.is_none(), "Upgrade should fail after eviction");
}
