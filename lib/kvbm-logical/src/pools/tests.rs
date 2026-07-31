// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities and pool/guard integration tests.

use super::super::{
    blocks::*,
    pools::*,
    testing::{self, TestMeta},
};

/// Re-export TestMeta as TestData for backward compatibility with older tests.
pub type TestData = TestMeta;

#[cfg(test)]
#[allow(unused, dead_code)]
pub(crate) mod fixtures {
    use super::*;

    use std::sync::Arc;

    use dynamo_tokens::TokenBlock;

    pub use super::testing::tokens_for_id;

    pub fn create_token_block(tokens: &[u32], block_size: u32) -> TokenBlock {
        testing::create_test_token_block(tokens, block_size)
    }

    pub fn create_test_block_store(count: usize) -> Arc<BlockStore<TestData>> {
        testing::TestPoolSetupBuilder::default()
            .block_count(count)
            .build()
            .unwrap()
            .build_store::<TestData>()
    }
}

#[cfg(test)]
use fixtures::*;

#[test]
fn test_mutable_block_complete_error_returns_block() {
    let store = create_test_block_store(5);
    let mut mutable_blocks = store.allocate_reset_blocks(1);
    let mutable_block = mutable_blocks.pop().unwrap();
    let original_block_id = mutable_block.block_id();

    // block_size is 4, but token block has 8 tokens
    let big_token_block = testing::create_test_token_block(&[1, 2, 3, 4, 5, 6, 7, 8], 8);

    let result = mutable_block.complete(&big_token_block);
    assert!(result.is_err());

    let BlockError::BlockSizeMismatch {
        expected,
        actual,
        block: recovered,
    } = result.expect_err("Expected BlockSizeMismatch error");
    assert_eq!(expected, 4);
    assert_eq!(actual, 8);
    let recovered: MutableBlock<TestData> = recovered;
    assert_eq!(recovered.block_id(), original_block_id);
}

#[test]
fn test_mutable_block_stage_and_debug() {
    let store = create_test_block_store(5);
    let mut mutable_blocks = store.allocate_reset_blocks(1);
    let mutable_block = mutable_blocks.pop().unwrap();

    let debug_str = format!("{:?}", mutable_block);
    assert!(debug_str.contains("MutableBlock"));

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
    let store = create_test_block_store(5);
    let mut mutable_blocks = store.allocate_reset_blocks(1);
    let mutable_block = mutable_blocks.pop().unwrap();
    let original_block_id = mutable_block.block_id();

    let token_block = create_token_block(&[10, 11, 12, 13], 4);
    let complete_block = mutable_block
        .complete(&token_block)
        .expect("Should complete");

    assert_eq!(complete_block.block_id(), original_block_id);

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

    assert_eq!(immutable_block.sequence_hash(), seq_hash);
    let _block_id = immutable_block.block_id();
    let _handle = immutable_block.registration_handle();
    assert!(immutable_block.use_count() >= 1);

    let weak_block = immutable_block.downgrade();
    assert_eq!(weak_block.sequence_hash(), seq_hash);

    let upgraded = weak_block
        .upgrade()
        .expect("Should upgrade while original alive");
    assert_eq!(upgraded.sequence_hash(), seq_hash);
    assert_eq!(upgraded.block_id(), immutable_block.block_id());
}

#[test]
fn test_weak_block_upgrade_via_resurrection() {
    let manager = testing::create_test_manager::<TestData>(10);

    let token_block = testing::create_iota_token_block(200, 4);
    let seq_hash = crate::KvbmSequenceHashProvider::kvbm_sequence_hash(&token_block);

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

    let upgraded = weak_block
        .upgrade()
        .expect("upgrade should succeed via resurrection");
    assert_eq!(upgraded.sequence_hash(), seq_hash);
}

#[test]
fn test_immutable_and_weak_block_debug() {
    let manager = testing::create_test_manager::<TestData>(10);

    let token_block = testing::create_iota_token_block(300, 4);

    let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
    let complete_block = mutable_blocks
        .into_iter()
        .next()
        .unwrap()
        .complete(&token_block)
        .expect("Should complete");

    let immutable_blocks = manager.register_blocks(vec![complete_block]);
    let immutable_block = immutable_blocks.into_iter().next().unwrap();

    let debug_str = format!("{:?}", immutable_block);
    assert!(debug_str.contains("ImmutableBlock"));

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

    let result = weak_block.upgrade();
    assert!(result.is_none(), "Upgrade should fail after eviction");
}
