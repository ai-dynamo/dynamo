// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::KvbmSequenceHashProvider;
use crate::blocks::BlockError;
use crate::testing::{
    self, TestMeta, create_iota_token_block, create_test_token_block as testing_create_token_block,
};
use rstest::rstest;

// Type alias for backward compatibility
type TestBlockData = TestMeta;

/// Helper function to create a token block with specific data (local wrapper)
fn create_token_block(tokens: &[u32]) -> dynamo_tokens::TokenBlock {
    testing_create_token_block(tokens, tokens.len() as u32)
}

/// Helper function to create a token block using fill_iota pattern
fn create_test_token_block_from_iota(start: u32) -> dynamo_tokens::TokenBlock {
    create_iota_token_block(start, 4)
}

fn create_test_token_block_8_from_iota(start: u32) -> dynamo_tokens::TokenBlock {
    create_iota_token_block(start, 8)
}

/// Helper function to create a basic manager for testing
fn create_test_manager(block_count: usize) -> BlockManager<TestBlockData> {
    testing::create_test_manager::<TestBlockData>(block_count)
}

// ============================================================================
// BUILDER PATTERN TESTS
// ============================================================================

mod builder_tests {
    use super::*;

    #[test]
    fn test_builder_default() {
        let registry = BlockRegistry::new();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .registry(registry)
            .build()
            .expect("Should build with defaults");

        // Verify initial gauge
        let snap = manager.metrics().snapshot();
        assert_eq!(snap.reset_pool_size, 100);
        assert_eq!(snap.inactive_pool_size, 0);

        // Verify we can allocate blocks
        let blocks = manager.allocate_blocks(5);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 5);
    }

    #[test]
    fn test_builder_with_lru_backend() {
        let registry = BlockRegistry::new();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .registry(registry)
            .with_lru_backend()
            .build()
            .expect("Should build with LRU backend");

        // Verify we can allocate blocks
        let blocks = manager.allocate_blocks(10);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 10);
    }

    #[test]
    fn test_builder_with_multi_lru_backend() {
        let registry = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::Small.create_tracker())
            .build();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .registry(registry)
            .with_multi_lru_backend()
            .build()
            .expect("Should build with MultiLRU backend");

        // Verify we can allocate blocks
        let blocks = manager.allocate_blocks(8);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 8);
    }

    #[test]
    fn test_builder_with_custom_multi_lru_thresholds() {
        let registry = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker())
            .build();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .registry(registry)
            .with_multi_lru_backend_custom_thresholds(2, 6, 12)
            .build()
            .expect("Should build with custom thresholds");

        // Verify we can allocate blocks
        let blocks = manager.allocate_blocks(4);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 4);
    }

    #[test]
    fn test_builder_with_duplication_policy() {
        let registry = BlockRegistry::new();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(50)
            .registry(registry)
            .duplication_policy(BlockDuplicationPolicy::Reject)
            .with_lru_backend()
            .build()
            .expect("Should build with duplication policy");

        let blocks = manager.allocate_blocks(2);
        assert!(blocks.is_some());
        assert_eq!(blocks.unwrap().len(), 2);
    }

    #[test]
    fn test_builder_validation_zero_blocks() {
        let registry = BlockRegistry::new();
        let result = BlockManager::<TestBlockData>::builder()
            .block_count(0)
            .registry(registry)
            .build();

        assert!(result.is_err());
        if let Err(err) = result {
            assert!(
                err.to_string()
                    .contains("block_count must be greater than 0")
            );
        }
    }

    #[test]
    fn test_builder_validation_missing_block_count() {
        let registry = BlockRegistry::new();
        let result = BlockManager::<TestBlockData>::builder()
            .registry(registry)
            .with_lru_backend()
            .build();

        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("block_count is required"));
        }
    }

    #[test]
    fn test_builder_validation_missing_registry() {
        let result = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .with_lru_backend()
            .build();

        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("registry is required"));
        }
    }

    #[test]
    #[should_panic(expected = "must be <= 15")]
    fn test_builder_invalid_threshold_too_high() {
        BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .with_multi_lru_backend_custom_thresholds(2, 6, 20); // 20 > 15, should panic
    }

    #[test]
    #[should_panic(expected = "must be in ascending order")]
    fn test_builder_invalid_threshold_order() {
        BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .with_multi_lru_backend_custom_thresholds(6, 2, 10); // Not ascending, should panic
    }

    #[test]
    fn test_builder_multi_lru_requires_frequency_tracking() {
        let registry = BlockRegistry::new(); // No frequency tracking
        let result = BlockManager::<TestBlockData>::builder()
            .block_count(100)
            .registry(registry)
            .with_multi_lru_backend()
            .build();

        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("frequency tracking"));
        }
    }
}

// ============================================================================
// BLOCK ALLOCATION TESTS
// ============================================================================

mod allocation_tests {
    use super::*;

    #[test]
    fn test_allocate_single_block() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        let initial_available = manager.available_blocks();
        let initial_total = manager.total_blocks();
        assert_eq!(initial_available, 10);

        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 10);

        let blocks = manager.allocate_blocks(1).expect("Should allocate 1 block");
        assert_eq!(blocks.len(), 1);

        // Verify available blocks decreased
        assert_eq!(manager.available_blocks(), initial_available - 1);
        assert_eq!(manager.total_blocks(), initial_total);

        let snap = m.snapshot();
        assert_eq!(snap.allocations, 1);
        assert_eq!(snap.inflight_mutable, 1);
        assert_eq!(snap.reset_pool_size, 9);

        let block = blocks.into_iter().next().unwrap();
        // Verify block has a valid ID
        let _block_id = block.block_id();

        // Drop the block and verify it returns to pool
        drop(block);
        assert_eq!(manager.available_blocks(), initial_available);
        assert_eq!(manager.total_blocks(), initial_total);

        let snap = m.snapshot();
        assert_eq!(snap.inflight_mutable, 0);
        assert_eq!(snap.reset_pool_size, 10);
    }

    #[test]
    fn test_allocate_multiple_blocks() {
        let manager = create_test_manager(20);
        let m = manager.metrics();

        let initial_available = manager.available_blocks();
        let initial_total = manager.total_blocks();
        assert_eq!(initial_available, 20);

        let blocks = manager
            .allocate_blocks(5)
            .expect("Should allocate 5 blocks");
        assert_eq!(blocks.len(), 5);

        // Verify available blocks decreased correctly
        assert_eq!(manager.available_blocks(), initial_available - 5);
        assert_eq!(manager.total_blocks(), initial_total);

        let snap = m.snapshot();
        assert_eq!(snap.allocations, 5);
        assert_eq!(snap.inflight_mutable, 5);

        // Verify all blocks have unique IDs
        let mut block_ids = Vec::new();
        for block in blocks {
            let id = block.block_id();
            assert!(!block_ids.contains(&id), "Block IDs should be unique");
            block_ids.push(id);
        }

        // All blocks should return to pool automatically on drop
        assert_eq!(manager.available_blocks(), initial_available);
        assert_eq!(manager.total_blocks(), initial_total);

        let snap = m.snapshot();
        assert_eq!(snap.inflight_mutable, 0);
    }

    #[test]
    fn test_allocate_all_blocks() {
        let manager = create_test_manager(10);

        let blocks = manager
            .allocate_blocks(10)
            .expect("Should allocate all blocks");
        assert_eq!(blocks.len(), 10);
    }

    #[test]
    fn test_allocate_more_than_available() {
        let manager = create_test_manager(5);

        let result = manager.allocate_blocks(10);
        assert!(
            result.is_none(),
            "Should not allocate more blocks than available"
        );
    }

    #[test]
    fn test_allocate_zero_blocks() {
        let manager = create_test_manager(10);

        let blocks = manager
            .allocate_blocks(0)
            .expect("Should allocate 0 blocks");
        assert_eq!(blocks.len(), 0);
    }

    #[test]
    fn test_sequential_allocations() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        let total_blocks = manager.total_blocks();
        assert_eq!(manager.available_blocks(), total_blocks);
        assert_eq!(m.snapshot().reset_pool_size, 10);

        let blocks1 = manager.allocate_blocks(3).expect("First allocation");
        assert_eq!(blocks1.len(), 3);
        assert_eq!(manager.available_blocks(), total_blocks - 3);
        assert_eq!(m.snapshot().reset_pool_size, 7);

        let blocks2 = manager.allocate_blocks(4).expect("Second allocation");
        assert_eq!(blocks2.len(), 4);
        assert_eq!(manager.available_blocks(), total_blocks - 7);
        assert_eq!(m.snapshot().reset_pool_size, 3);

        let blocks3 = manager.allocate_blocks(3).expect("Third allocation");
        assert_eq!(blocks3.len(), 3);
        assert_eq!(manager.available_blocks(), 0);
        assert_eq!(m.snapshot().reset_pool_size, 0);

        let snap = m.snapshot();
        assert_eq!(snap.allocations, 10);
        assert_eq!(snap.inflight_mutable, 10);

        // Should have no blocks left
        let blocks4 = manager.allocate_blocks(1);
        assert!(blocks4.is_none(), "Should not have any blocks left");

        // Drop blocks in reverse order and verify counts
        drop(blocks3);
        assert_eq!(manager.available_blocks(), 3);
        assert_eq!(m.snapshot().reset_pool_size, 3);

        drop(blocks2);
        assert_eq!(manager.available_blocks(), 7);
        assert_eq!(m.snapshot().reset_pool_size, 7);

        drop(blocks1);
        assert_eq!(manager.available_blocks(), total_blocks);
        assert_eq!(manager.total_blocks(), total_blocks);

        let snap = m.snapshot();
        assert_eq!(snap.inflight_mutable, 0);
        assert_eq!(snap.reset_pool_size, 10);
    }
}

// ============================================================================
// BLOCK LIFECYCLE AND POOL RETURN TESTS
// ============================================================================

mod lifecycle_tests {
    use super::*;

    #[test]
    fn test_mutable_block_returns_to_reset_pool() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        let initial_available = manager.available_blocks();
        let initial_total = manager.total_blocks();
        assert_eq!(initial_available, 10);
        assert_eq!(initial_total, 10);

        {
            let blocks = manager
                .allocate_blocks(3)
                .expect("Should allocate 3 blocks");
            assert_eq!(blocks.len(), 3);

            // Available blocks should decrease
            assert_eq!(manager.available_blocks(), initial_available - 3);
            assert_eq!(manager.total_blocks(), initial_total); // Total never changes

            let snap = m.snapshot();
            assert_eq!(snap.inflight_mutable, 3);
            assert_eq!(snap.reset_pool_size, 7);
        } // MutableBlocks dropped here - should return to reset pool

        // Available blocks should return to original count
        assert_eq!(manager.available_blocks(), initial_available);
        assert_eq!(manager.total_blocks(), initial_total);

        let snap = m.snapshot();
        assert_eq!(snap.inflight_mutable, 0);
        assert_eq!(snap.reset_pool_size, 10);
    }

    #[test]
    fn test_complete_block_returns_to_reset_pool() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        let initial_available = manager.available_blocks();
        let initial_total = manager.total_blocks();

        {
            let mutable_blocks = manager.allocate_blocks(2).expect("Should allocate blocks");
            assert_eq!(manager.available_blocks(), initial_available - 2);

            let snap = m.snapshot();
            assert_eq!(snap.reset_pool_size, 8);

            // Note: create_token_block uses 3 tokens but block_size is 4,
            // so complete() returns Err(BlockSizeMismatch) for all blocks.
            let _complete_blocks: Vec<_> = mutable_blocks
                .into_iter()
                .enumerate()
                .map(|(i, block)| {
                    let tokens = vec![400 + i as u32, 401 + i as u32, 402 + i as u32];
                    let token_block = create_token_block(&tokens);
                    block.complete(&token_block)
                })
                .collect();

            // Blocks are still unavailable while in Complete state
            assert_eq!(manager.available_blocks(), initial_available - 2);

            let snap = m.snapshot();
            assert_eq!(snap.inflight_mutable, 2);
            assert_eq!(snap.stagings, 0);
            assert_eq!(snap.reset_pool_size, 8);
        } // CompleteBlocks dropped here - should return to reset pool

        // Available blocks should return to original count since blocks weren't registered
        assert_eq!(manager.available_blocks(), initial_available);
        assert_eq!(manager.total_blocks(), initial_total);

        let snap = m.snapshot();
        assert_eq!(snap.inflight_mutable, 0);
        assert_eq!(snap.reset_pool_size, 10);
    }

    #[test]
    fn test_registered_block_lifecycle() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        let initial_available = manager.available_blocks();
        let initial_total = manager.total_blocks();

        // Step 1: Allocate and complete blocks
        let token_block = create_test_token_block_from_iota(500);
        let seq_hash = token_block.kvbm_sequence_hash();

        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
        assert_eq!(manager.available_blocks(), initial_available - 1);

        let snap = m.snapshot();
        assert_eq!(snap.allocations, 1);
        assert_eq!(snap.inflight_mutable, 1);
        assert_eq!(snap.reset_pool_size, 9);
        assert_eq!(snap.inactive_pool_size, 0);

        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete block");

        // Still unavailable while in Complete state
        assert_eq!(manager.available_blocks(), initial_available - 1);

        let snap = m.snapshot();
        assert_eq!(snap.stagings, 1);
        assert_eq!(snap.inflight_mutable, 0);

        // Step 2: Register the block
        let immutable_blocks = manager.register_blocks(vec![complete_block]);
        assert_eq!(immutable_blocks.len(), 1);
        let immutable_block = immutable_blocks.into_iter().next().unwrap();

        // Block is still not available (it's now in active/inactive pools, not reset)
        assert_eq!(manager.available_blocks(), initial_available - 1);

        let snap = m.snapshot();
        assert_eq!(snap.registrations, 1);
        assert_eq!(snap.inflight_immutable, 1);
        assert_eq!(snap.reset_pool_size, 9);
        assert_eq!(snap.inactive_pool_size, 0);

        {
            // Step 3: Use the block and verify it can be matched
            let matched_blocks = manager.match_blocks(&[seq_hash]);
            assert_eq!(matched_blocks.len(), 1);
            assert_eq!(matched_blocks[0].sequence_hash(), seq_hash);

            // Still not available while being used
            assert_eq!(manager.available_blocks(), initial_available - 1);

            let snap = m.snapshot();
            assert_eq!(snap.match_hashes_requested, 1);
            assert_eq!(snap.match_blocks_returned, 1);
            assert_eq!(snap.inflight_immutable, 2);
        } // matched blocks dropped here

        let snap = m.snapshot();
        assert_eq!(snap.inflight_immutable, 1);

        // Step 4: Drop the original registered block → block moves to inactive
        drop(immutable_block);

        // Block should now be available again (moved to inactive pool when ref count reached 0)
        assert_eq!(manager.available_blocks(), initial_available);
        assert_eq!(manager.total_blocks(), initial_total);

        let snap = m.snapshot();
        assert_eq!(snap.inflight_immutable, 0);
        assert_eq!(snap.reset_pool_size, 9);
        assert_eq!(snap.inactive_pool_size, 1);

        // Step 5: Re-match from inactive pool → pulls block out
        {
            let re_matched = manager.match_blocks(&[seq_hash]);
            assert_eq!(re_matched.len(), 1);

            let snap = m.snapshot();
            assert_eq!(snap.inactive_pool_size, 0);
        } // re_matched dropped → block returns to inactive

        let snap = m.snapshot();
        assert_eq!(snap.inactive_pool_size, 1);
    }

    #[test]
    fn test_concurrent_allocation_and_return() {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(create_test_manager(20));
        let initial_total = manager.total_blocks();

        let handles: Vec<_> = (0..5)
            .map(|i| {
                let manager_clone = Arc::clone(&manager);
                thread::spawn(move || {
                    // Each thread allocates and drops some blocks
                    for j in 0..3 {
                        let blocks = manager_clone.allocate_blocks(2);
                        if let Some(blocks) = blocks {
                            // Complete one block
                            let token_block =
                                create_test_token_block_from_iota((600 + i * 10 + j) as u32);
                            let complete_block = blocks
                                .into_iter()
                                .next()
                                .unwrap()
                                .complete(&token_block)
                                .expect("Should complete block");

                            // Register and drop
                            let _immutable_blocks =
                                manager_clone.register_blocks(vec![complete_block]);
                            // blocks automatically dropped at end of scope
                        }
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // All blocks should eventually be available again
        assert_eq!(manager.total_blocks(), initial_total);
        // Available might be less than total if some blocks are in inactive pool,
        // but total should be preserved
    }

    #[test]
    fn test_full_block_lifecycle() {
        let manager = create_test_manager(10);
        let total_blocks = manager.total_blocks();
        assert_eq!(manager.available_blocks(), total_blocks);

        // Step 1: Allocate 5 blocks
        let mutable_blocks = manager
            .allocate_blocks(5)
            .expect("Should allocate 5 blocks");
        assert_eq!(manager.available_blocks(), total_blocks - 5);
        assert_eq!(manager.total_blocks(), total_blocks);

        // Step 2: Complete 3 blocks, drop 2 mutable blocks
        let mut mutable_blocks_iter = mutable_blocks.into_iter();
        let complete_blocks: Vec<_> = (0..3)
            .map(|i| {
                let block = mutable_blocks_iter.next().unwrap();
                let tokens = vec![
                    700 + i as u32,
                    701 + i as u32,
                    702 + i as u32,
                    703 + i as u32,
                ];
                let token_block = create_token_block(&tokens);
                block.complete(&token_block).expect("Should complete block")
            })
            .collect();
        let mutable_part: Vec<_> = mutable_blocks_iter.collect();

        drop(mutable_part); // Drop 2 mutable blocks

        // Should have 2 blocks returned to reset pool
        assert_eq!(manager.available_blocks(), total_blocks - 3);

        // Step 3: Register the 3 completed blocks
        let immutable_blocks = manager.register_blocks(complete_blocks);
        assert_eq!(immutable_blocks.len(), 3);

        // Still 3 blocks unavailable (now in active pool)
        assert_eq!(manager.available_blocks(), total_blocks - 3);

        // Step 4: Match and use one of the blocks
        let seq_hash = create_test_token_block_from_iota(700).kvbm_sequence_hash();
        let matched_blocks = manager.match_blocks(&[seq_hash]);
        assert_eq!(matched_blocks.len(), 1);

        // Step 5: Drop one registered block, keep others
        drop(immutable_blocks.into_iter().next());

        // Still have registered blocks in use, so available count depends on ref counting
        let available_after_drop = manager.available_blocks();
        assert!(available_after_drop >= total_blocks - 3);
        assert!(available_after_drop <= total_blocks);

        // Step 6: Drop everything
        drop(matched_blocks);

        // Eventually all blocks should be available again
        // (Some might be in inactive pool, but available_blocks counts both reset and inactive)
        assert_eq!(manager.total_blocks(), total_blocks);
        let final_available = manager.available_blocks();
        assert_eq!(final_available, total_blocks); // Allow for some blocks in inactive pool
    }
}

// ============================================================================
// BLOCK SIZE VALIDATION TESTS
// ============================================================================

mod block_size_tests {

    use super::*;

    #[test]
    fn test_default_block_size() {
        let manager = create_test_manager(10);
        assert_eq!(manager.block_size(), 4); // create_test_manager uses block_size(4)
    }

    #[test]
    fn test_custom_block_size() {
        let registry = BlockRegistry::new();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(32)
            .registry(registry)
            .build()
            .expect("Should build with custom block size");
        assert_eq!(manager.block_size(), 32);
    }

    #[test]
    fn test_block_size_validation_correct_size() {
        let manager = create_test_manager(10);
        let token_block = create_test_token_block_from_iota(100); // 4 tokens

        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
        let mutable_block = mutable_blocks.into_iter().next().unwrap();

        // Should succeed since token_block has exactly 4 tokens
        let result = mutable_block.complete(&token_block);
        assert!(result.is_ok());
    }

    #[test]
    fn test_block_size_validation_wrong_size() {
        // Create a manager expecting 8-token blocks
        let registry = BlockRegistry::new();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(8)
            .registry(registry)
            .with_lru_backend()
            .build()
            .expect("Should build manager");
        let token_block = create_test_token_block_from_iota(1); // 4 tokens, expected 8

        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
        let mutable_block = mutable_blocks.into_iter().next().unwrap();

        // Should fail since token_block has 4 tokens but manager expects 8
        let result = mutable_block.complete(&token_block);
        assert!(result.is_err());

        if let Err(BlockError::BlockSizeMismatch {
            expected,
            actual,
            block: _,
        }) = result
        {
            assert_eq!(expected, 8);
            assert_eq!(actual, 4);
        } else {
            panic!("Expected BlockSizeMismatch error");
        }
    }

    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(4)]
    #[case(8)]
    #[case(16)]
    #[case(32)]
    #[case(64)]
    #[case(128)]
    #[case(256)]
    #[case(512)]
    #[case(1024)]
    fn test_builder_block_size_power_of_two(#[case] size: usize) {
        let registry = BlockRegistry::new();
        let result = BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(size)
            .registry(registry)
            .build();
        assert!(result.is_ok(), "Block size {} should be valid", size);
    }

    #[test]
    #[should_panic(expected = "block_size must be a power of 2")]
    fn test_builder_block_size_not_power_of_two() {
        BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(15); // Not a power of 2
    }

    #[test]
    #[should_panic(expected = "block_size must be between 1 and 1024")]
    fn test_builder_block_size_too_large() {
        BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(2048); // Too large
    }

    #[test]
    #[should_panic(expected = "block_size must be between 1 and 1024")]
    fn test_builder_block_size_zero() {
        BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(0); // Zero is invalid
    }

    #[test]
    #[should_panic(expected = "block_size must be a power of 2")]
    fn test_builder_validation_invalid_block_size() {
        BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(7); // Not a power of 2, panics immediately
    }

    #[test]
    fn test_different_block_sizes() {
        // Test with block size 4
        let registry_4 = BlockRegistry::new();
        let manager_4 = BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(4)
            .registry(registry_4)
            .build()
            .expect("Should build with block size 4");

        let token_block_4 = create_test_token_block_from_iota(10); // 4 tokens
        let mutable_blocks = manager_4
            .allocate_blocks(1)
            .expect("Should allocate blocks");
        let result = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block_4);
        assert!(result.is_ok());

        // Test with block size 8
        let registry_8 = BlockRegistry::new();
        let manager_8 = BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(8)
            .registry(registry_8)
            .build()
            .expect("Should build with block size 8");

        let token_block_8 = create_test_token_block_8_from_iota(20); // 8 tokens
        let mutable_blocks = manager_8
            .allocate_blocks(1)
            .expect("Should allocate blocks");
        let result = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block_8);
        assert!(result.is_ok());
    }
}

// ============================================================================
// BLOCK REGISTRATION AND DEDUPLICATION TESTS
// ============================================================================

mod registration_tests {
    use super::*;

    #[test]
    fn test_register_single_block() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        let token_block = create_test_token_block_from_iota(150);
        let expected_hash = token_block.kvbm_sequence_hash();
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete block");

        let immutable_blocks = manager.register_blocks(vec![complete_block]);
        assert_eq!(immutable_blocks.len(), 1);

        let immutable_block = immutable_blocks.into_iter().next().unwrap();
        assert_eq!(immutable_block.sequence_hash(), expected_hash);

        let snap = m.snapshot();
        assert_eq!(snap.registrations, 1);
        assert_eq!(snap.stagings, 1);
    }

    #[test]
    fn test_register_multiple_blocks() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        let mut complete_blocks = Vec::new();
        let mut expected_hashes = Vec::new();

        for i in 0..3 {
            let tokens = vec![100 + i, 101 + i, 102 + i, 103 + i];
            let token_block = create_token_block(&tokens);
            expected_hashes.push(token_block.kvbm_sequence_hash());

            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            complete_blocks.push(complete_block);
        }

        let immutable_blocks = manager.register_blocks(complete_blocks);
        assert_eq!(immutable_blocks.len(), 3);

        for (i, immutable_block) in immutable_blocks.iter().enumerate() {
            assert_eq!(immutable_block.sequence_hash(), expected_hashes[i]);
        }

        let snap = m.snapshot();
        assert_eq!(snap.registrations, 3);
        assert_eq!(snap.stagings, 3);
    }

    #[rstest]
    #[case(BlockDuplicationPolicy::Allow, 200, "allow", false)]
    #[case(BlockDuplicationPolicy::Reject, 300, "reject", true)]
    fn test_deduplication_policy(
        #[case] policy: BlockDuplicationPolicy,
        #[case] iota_base: u32,
        #[case] policy_name: &str,
        #[case] expect_same_block_id: bool,
    ) {
        let registry = BlockRegistry::new();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(10)
            .block_size(4)
            .registry(registry)
            .duplication_policy(policy)
            .with_lru_backend()
            .build()
            .expect("Should build manager");

        let token_block = create_test_token_block_from_iota(iota_base);
        let seq_hash = token_block.kvbm_sequence_hash();

        // Register the same sequence hash twice
        let complete_block1 = {
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block")
        };

        let complete_block2 = {
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block")
        };

        let immutable_blocks1 = manager.register_blocks(vec![complete_block1]);
        let immutable_blocks2 = manager.register_blocks(vec![complete_block2]);

        assert_eq!(immutable_blocks1.len(), 1);
        assert_eq!(immutable_blocks2.len(), 1);

        // Both should have the same sequence hash
        assert_eq!(immutable_blocks1[0].sequence_hash(), seq_hash);
        assert_eq!(immutable_blocks2[0].sequence_hash(), seq_hash);

        // Check block IDs based on policy
        if expect_same_block_id {
            // Duplicates are rejected - same block ID
            assert_eq!(
                immutable_blocks1[0].block_id(),
                immutable_blocks2[0].block_id(),
                "With {} policy, duplicates should reuse the same block ID",
                policy_name
            );

            let snap = manager.metrics().snapshot();
            assert_eq!(snap.registration_dedup, 1);
        } else {
            // Duplicates are allowed - different block IDs
            assert_ne!(
                immutable_blocks1[0].block_id(),
                immutable_blocks2[0].block_id(),
                "With {} policy, duplicates should have different block IDs",
                policy_name
            );

            let snap = manager.metrics().snapshot();
            assert_eq!(snap.duplicate_blocks, 1);
        }
    }

    #[test]
    fn test_register_mutable_block_from_existing_reject_returns_block_to_reset_pool() {
        let registry = BlockRegistry::new();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(2)
            .block_size(4)
            .registry(registry)
            .duplication_policy(BlockDuplicationPolicy::Reject)
            .build()
            .expect("Should build manager");

        let blocks = manager
            .allocate_blocks(2)
            .expect("Should allocate two blocks");
        let mut iter = blocks.into_iter();
        let primary_mutable = iter.next().expect("Should have first block");
        let duplicate_mutable = iter.next().expect("Should have second block");

        let primary_id = primary_mutable.block_id();
        let duplicate_id = duplicate_mutable.block_id();

        let token_block = create_test_token_block_from_iota(42);
        let primary_complete = primary_mutable
            .complete(&token_block)
            .expect("Should complete primary block");

        let mut registered = manager.register_blocks(vec![primary_complete]);
        let primary_immutable = registered.pop().expect("Should register primary block");

        let duplicate_completed = duplicate_mutable
            .stage(primary_immutable.sequence_hash(), manager.block_size())
            .expect("block size should match");

        let result = manager.register_block(duplicate_completed);

        assert_eq!(
            result.block_id(),
            primary_id,
            "Should reuse existing primary when duplicates are rejected"
        );

        assert_eq!(
            manager.available_blocks(),
            1,
            "Rejected duplicate should be returned to the reset pool"
        );

        let mut returned_blocks = manager
            .allocate_blocks(1)
            .expect("Should allocate returned reset block");
        let returned_block = returned_blocks
            .pop()
            .expect("Should contain one returned block");

        assert_eq!(
            returned_block.block_id(),
            duplicate_id,
            "Returned block should be the rejected duplicate"
        );

        let snap = manager.metrics().snapshot();
        assert_eq!(snap.registrations, 2);
        assert_eq!(snap.registration_dedup, 1);
        // returned_block is still held, so reset pool is empty
        assert_eq!(snap.reset_pool_size, 0);

        // Drop returned_block → back to reset pool
        drop(returned_block);
        assert_eq!(manager.metrics().snapshot().reset_pool_size, 1);
    }
}

// ============================================================================
// BLOCK MATCHING TESTS
// ============================================================================

mod matching_tests {
    use super::*;

    #[test]
    fn test_match_no_blocks() {
        let manager = create_test_manager(10);

        let seq_hashes = vec![create_test_token_block_from_iota(400).kvbm_sequence_hash()];
        let matched_blocks = manager.match_blocks(&seq_hashes);
        assert_eq!(matched_blocks.len(), 0);
    }

    #[test]
    fn test_match_single_block() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        let token_block = create_test_token_block_from_iota(500);
        let seq_hash = token_block.kvbm_sequence_hash();

        // Register a block
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete block");
        let _immutable_blocks = manager.register_blocks(vec![complete_block]);

        // Try to match it
        let matched_blocks = manager.match_blocks(&[seq_hash]);
        assert_eq!(matched_blocks.len(), 1);
        assert_eq!(matched_blocks[0].sequence_hash(), seq_hash);

        let snap = m.snapshot();
        assert_eq!(snap.match_hashes_requested, 1);
        assert_eq!(snap.match_blocks_returned, 1);
    }

    #[test]
    fn test_match_multiple_blocks() {
        let manager = create_test_manager(10);

        let mut seq_hashes = Vec::new();

        // Register multiple blocks
        for i in 0..4 {
            let tokens = vec![600 + i, 601 + i, 602 + i, 603 + i];
            let token_block = create_token_block(&tokens);
            seq_hashes.push(token_block.kvbm_sequence_hash());

            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            let _immutable_blocks = manager.register_blocks(vec![complete_block]);
        }

        // Match all blocks
        let matched_blocks = manager.match_blocks(&seq_hashes);
        assert_eq!(matched_blocks.len(), 4);

        for (i, matched_block) in matched_blocks.iter().enumerate() {
            assert_eq!(matched_block.sequence_hash(), seq_hashes[i]);
        }

        let snap = manager.metrics().snapshot();
        assert_eq!(snap.match_hashes_requested, 4);
        assert_eq!(snap.match_blocks_returned, 4);
    }

    #[test]
    fn test_match_partial_blocks() {
        let manager = create_test_manager(10);

        let mut seq_hashes = Vec::new();

        // Register only some blocks
        for i in 0..3 {
            let tokens = vec![700 + i, 701 + i, 702 + i, 703 + i];
            let token_block = create_token_block(&tokens);
            seq_hashes.push(token_block.kvbm_sequence_hash());

            if i < 2 {
                // Only register first 2 blocks
                let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
                let complete_block = mutable_blocks
                    .into_iter()
                    .next()
                    .unwrap()
                    .complete(&token_block)
                    .expect("Should complete block");
                let _immutable_blocks = manager.register_blocks(vec![complete_block]);
            }
        }

        // Try to match all 3 - should only get 2
        let matched_blocks = manager.match_blocks(&seq_hashes);
        assert_eq!(matched_blocks.len(), 2);

        for matched_block in matched_blocks {
            assert!(seq_hashes[0..2].contains(&matched_block.sequence_hash()));
        }

        let snap = manager.metrics().snapshot();
        assert_eq!(snap.match_hashes_requested, 3);
        assert_eq!(snap.match_blocks_returned, 2);
    }

    #[test]
    fn test_match_blocks_returns_immutable_blocks() {
        let manager = create_test_manager(10);

        let token_block = create_test_token_block_from_iota(800);
        let seq_hash = token_block.kvbm_sequence_hash();

        // Register a block
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete block");
        let _immutable_blocks = manager.register_blocks(vec![complete_block]);

        // Match and verify it's an ImmutableBlock
        let matched_blocks = manager.match_blocks(&[seq_hash]);
        assert_eq!(matched_blocks.len(), 1);

        let immutable_block = &matched_blocks[0];
        assert_eq!(immutable_block.sequence_hash(), seq_hash);

        // Test that we can downgrade it
        let weak_block = immutable_block.downgrade();
        assert_eq!(weak_block.sequence_hash(), seq_hash);
    }
}

// ============================================================================
// IMMUTABLE BLOCK AND WEAK BLOCK TESTS
// ============================================================================

mod immutable_block_tests {
    use super::*;

    #[test]
    fn test_immutable_block_downgrade_upgrade() {
        let manager = create_test_manager(10);

        let token_block = create_test_token_block_from_iota(100);
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete block");

        let immutable_blocks = manager.register_blocks(vec![complete_block]);
        let immutable_block = immutable_blocks.into_iter().next().unwrap();

        // Test downgrade to WeakBlock
        let weak_block = immutable_block.downgrade();
        assert_eq!(weak_block.sequence_hash(), immutable_block.sequence_hash());

        // Test upgrade from WeakBlock
        let upgraded_block = weak_block.upgrade().expect("Should be able to upgrade");
        assert_eq!(
            upgraded_block.sequence_hash(),
            immutable_block.sequence_hash()
        );
        assert_eq!(upgraded_block.block_id(), immutable_block.block_id());
    }

    #[test]
    fn test_weak_block_upgrade_after_drop() {
        let manager = create_test_manager(10);

        let token_block = create_test_token_block_from_iota(200);
        let seq_hash = token_block.kvbm_sequence_hash();

        // Create a weak block
        let weak_block = {
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            let immutable_block = immutable_blocks.into_iter().next().unwrap();

            // Downgrade to weak
            immutable_block.downgrade()
        }; // immutable_block is dropped here

        // The upgrade function should still find the block through the pools
        let upgraded_block = weak_block.upgrade();

        // The result depends on whether the block is still in the pools
        if let Some(block) = upgraded_block {
            assert_eq!(block.sequence_hash(), seq_hash);
        }
    }

    #[test]
    fn test_weak_block_upgrade_nonexistent() {
        let manager = create_test_manager(10);

        let token_block = create_token_block(&[999, 998, 997, 996]); // Keep non-sequential for this test

        // Create an ImmutableBlock and immediately downgrade it
        let weak_block = {
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            let immutable_block = immutable_blocks.into_iter().next().unwrap();
            immutable_block.downgrade()
        };

        // Force eviction by filling up the pool with other blocks
        for i in 0..10 {
            let tokens = vec![1000 + i, 1001 + i, 1002 + i, 1003 + i];
            let token_block = create_token_block(&tokens);
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            let _immutable_blocks = manager.register_blocks(vec![complete_block]);
        }

        // Try to upgrade - might fail if the original block was evicted
        let upgraded_block = weak_block.upgrade();
        assert!(upgraded_block.is_none());
        // // This test just verifies that upgrade doesn't panic, result can be None
        // if let Some(block) = upgraded_block {
        //     assert_eq!(
        //         block.sequence_hash(),
        //         create_token_block(&[999, 998, 997, 996]).sequence_hash()
        //     );
        // }
    }

    #[test]
    fn test_multiple_weak_blocks_same_sequence() {
        let manager = create_test_manager(10);

        let token_block = create_test_token_block_from_iota(150);
        let seq_hash = token_block.kvbm_sequence_hash();

        // Create multiple weak blocks from the same immutable block
        let (weak1, weak2, weak3) = {
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            let immutable_block = immutable_blocks.into_iter().next().unwrap();

            let w1 = immutable_block.downgrade();
            let w2 = immutable_block.downgrade();
            let w3 = immutable_block.downgrade();
            (w1, w2, w3)
        };

        // All weak blocks should have the same sequence hash
        assert_eq!(weak1.sequence_hash(), seq_hash);
        assert_eq!(weak2.sequence_hash(), seq_hash);
        assert_eq!(weak3.sequence_hash(), seq_hash);

        // All should be able to upgrade
        let upgraded1 = weak1.upgrade().expect("Should upgrade");
        let upgraded2 = weak2.upgrade().expect("Should upgrade");
        let upgraded3 = weak3.upgrade().expect("Should upgrade");

        assert_eq!(upgraded1.sequence_hash(), seq_hash);
        assert_eq!(upgraded2.sequence_hash(), seq_hash);
        assert_eq!(upgraded3.sequence_hash(), seq_hash);
    }
}

// ============================================================================
// UPGRADE FUNCTION TESTS
// ============================================================================

mod upgrade_function_tests {
    use super::*;

    #[test]
    fn test_upgrade_function_finds_active_blocks() {
        let manager = create_test_manager(10);

        let token_block = create_test_token_block_from_iota(250);
        let seq_hash = token_block.kvbm_sequence_hash();

        // Register a block (this puts it in active pool initially)
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
        let complete_block = mutable_blocks
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete block");
        let immutable_blocks = manager.register_blocks(vec![complete_block]);
        let immutable_block = immutable_blocks.into_iter().next().unwrap();

        // Create a weak block and test upgrade
        let weak_block = immutable_block.downgrade();
        let upgraded = weak_block
            .upgrade()
            .expect("Should find block in active pool");
        assert_eq!(upgraded.sequence_hash(), seq_hash);
    }

    #[test]
    fn test_upgrade_function_finds_inactive_blocks() {
        let manager = create_test_manager(20);

        let token_block = create_test_token_block_from_iota(350);
        let seq_hash = token_block.kvbm_sequence_hash();

        // Register a block
        let weak_block = {
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            let immutable_blocks = manager.register_blocks(vec![complete_block]);
            let immutable_block = immutable_blocks.into_iter().next().unwrap();
            immutable_block.downgrade()
        };

        // Force the block to potentially move to inactive pool by creating many other blocks
        for i in 0..10 {
            let tokens = vec![400 + i, 401 + i, 402 + i, 403 + i];
            let token_block = create_token_block(&tokens);
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate blocks");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            let _immutable_blocks = manager.register_blocks(vec![complete_block]);
        }

        // Try to upgrade - should still find the original block
        let upgraded = weak_block.upgrade();
        if let Some(block) = upgraded {
            assert_eq!(block.sequence_hash(), seq_hash);
        }
    }
}

// ============================================================================
// ERROR HANDLING AND EDGE CASE TESTS
// ============================================================================

mod error_handling_tests {
    use super::*;

    #[test]
    fn test_allocation_exhaustion() {
        let manager = create_test_manager(3);

        // Allocate all blocks
        let blocks1 = manager
            .allocate_blocks(2)
            .expect("Should allocate 2 blocks");
        let blocks2 = manager.allocate_blocks(1).expect("Should allocate 1 block");

        // Try to allocate more - should fail
        let blocks3 = manager.allocate_blocks(1);
        assert!(
            blocks3.is_none(),
            "Should not be able to allocate when pool is empty"
        );

        // Drop some blocks and try again
        drop(blocks1);
        drop(blocks2);

        // Blocks should be returned to pool automatically
        let blocks4 = manager.allocate_blocks(1);
        assert!(
            blocks4.is_some(),
            "Should be able to allocate after blocks are returned"
        );
    }

    #[test]
    fn test_empty_sequence_matching() {
        let manager = create_test_manager(10);

        let matched_blocks = manager.match_blocks(&[]);
        assert_eq!(matched_blocks.len(), 0);
    }

    #[test]
    fn test_register_empty_block_list() {
        let manager = create_test_manager(10);

        let immutable_blocks = manager.register_blocks(vec![]);
        assert_eq!(immutable_blocks.len(), 0);
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_lifecycle_single_block() {
        let manager = create_test_manager(10);

        // 1. Allocate a mutable block
        let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
        let mutable_block = mutable_blocks.into_iter().next().unwrap();
        let block_id = mutable_block.block_id();

        // 2. Complete the block
        let token_block = create_test_token_block_from_iota(1);
        let seq_hash = token_block.kvbm_sequence_hash();
        let complete_block = mutable_block
            .complete(&token_block)
            .expect("Should complete block");

        assert_eq!(complete_block.block_id(), block_id);
        assert_eq!(complete_block.sequence_hash(), seq_hash);

        // 3. Register the block
        let immutable_blocks = manager.register_blocks(vec![complete_block]);
        let immutable_block = immutable_blocks.into_iter().next().unwrap();

        assert_eq!(immutable_block.block_id(), block_id);
        assert_eq!(immutable_block.sequence_hash(), seq_hash);

        // 4. Match the block
        let matched_blocks = manager.match_blocks(&[seq_hash]);
        assert_eq!(matched_blocks.len(), 1);
        assert_eq!(matched_blocks[0].sequence_hash(), seq_hash);

        // 5. Create weak reference and upgrade
        let weak_block = immutable_block.downgrade();
        let upgraded_block = weak_block.upgrade().expect("Should upgrade");
        assert_eq!(upgraded_block.sequence_hash(), seq_hash);
    }

    #[rstest]
    #[case("lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lru_backend())]
    #[case("multi_lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_multi_lru_backend())]
    fn test_multiple_blocks_different_backends(
        #[case] backend_name: &str,
        #[case] backend_builder: fn(
            BlockManagerConfigBuilder<TestBlockData>,
        ) -> BlockManagerConfigBuilder<TestBlockData>,
    ) {
        let registry = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
            .build();
        let manager = backend_builder(
            BlockManager::<TestBlockData>::builder()
                .block_count(20)
                .block_size(4)
                .registry(registry),
        )
        .build()
        .expect("Should build");

        // Allocate, complete, and register blocks using BlockSequenceBuilder
        let base = 1000; // Use fixed base since we only test one backend per test now
        let tokens: Vec<u32> = (base as u32..base as u32 + 20).collect(); // 5 blocks * 4 tokens each = 20 tokens

        let mut seq_hashes = Vec::new();
        let mut complete_blocks = Vec::new();

        // Create token blocks from sequence
        let token_blocks = {
            let token_seq = dynamo_tokens::TokenBlockSequence::from_slice(&tokens, 4, Some(42));
            token_seq.blocks().to_vec()
        };

        for token_block in token_blocks.iter() {
            let seq_hash = token_block.kvbm_sequence_hash();
            seq_hashes.push(seq_hash);

            // Allocate mutable block and complete it
            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(token_block)
                .expect("Should complete block");
            complete_blocks.push(complete_block);
        }

        // Register all blocks
        let _immutable_blocks = manager.register_blocks(complete_blocks);

        // Verify all blocks can be matched
        let matched_blocks = manager.match_blocks(&seq_hashes);
        assert_eq!(
            matched_blocks.len(),
            5,
            "Manager with {} backend should match all blocks",
            backend_name
        );
    }

    #[test]
    fn test_concurrent_allocation_simulation() {
        let manager = create_test_manager(50);

        // Simulate concurrent allocations by interleaving operations
        let mut all_blocks = Vec::new();
        let mut all_hashes = Vec::new();

        // Phase 1: Allocate and complete some blocks
        for i in 0..10 {
            let tokens = vec![2000 + i, 2001 + i, 2002 + i, 2003 + i];
            let token_block = create_token_block(&tokens);
            all_hashes.push(token_block.kvbm_sequence_hash());

            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            all_blocks.push(complete_block);
        }

        // Phase 2: Register half the blocks
        let mut remaining_blocks = all_blocks.split_off(5);
        let _immutable_blocks1 = manager.register_blocks(all_blocks);

        // Phase 3: Allocate more blocks while some are registered
        for i in 10..15 {
            let tokens = vec![2000 + i, 2001 + i, 2002 + i, 2003 + i];
            let token_block = create_token_block(&tokens);
            all_hashes.push(token_block.kvbm_sequence_hash());

            let mutable_blocks = manager.allocate_blocks(1).expect("Should allocate");
            let complete_block = mutable_blocks
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .expect("Should complete block");
            remaining_blocks.push(complete_block);
        }

        // Phase 4: Register remaining blocks
        let _immutable_blocks2 = manager.register_blocks(remaining_blocks);

        // Phase 5: Verify we can match all registered blocks
        let matched_blocks = manager.match_blocks(&all_hashes);
        assert_eq!(
            matched_blocks.len(),
            15,
            "Should match all registered blocks"
        );
    }

    #[test]
    fn test_shared_registry_across_managers() {
        // Create shared registry with frequency tracking
        let tracker = FrequencyTrackingCapacity::Medium.create_tracker();
        let registry = BlockRegistry::builder().frequency_tracker(tracker).build();

        #[derive(Clone, Debug)]
        struct G1;

        #[derive(Clone, Debug)]
        struct G2;

        // Create two managers with different metadata types and policies
        let manager1 = BlockManager::<G1>::builder()
            .block_count(100)
            .block_size(4)
            .registry(registry.clone())
            .duplication_policy(BlockDuplicationPolicy::Allow)
            .with_multi_lru_backend()
            .build()
            .expect("Should build manager1");

        let manager2 = BlockManager::<G2>::builder()
            .block_count(100)
            .block_size(4)
            .registry(registry.clone())
            .duplication_policy(BlockDuplicationPolicy::Reject)
            .with_multi_lru_backend()
            .build()
            .expect("Should build manager2");

        // Verify both managers work
        assert_eq!(manager1.total_blocks(), 100);
        assert_eq!(manager2.total_blocks(), 100);

        // Verify they share the same registry (frequency tracking works across both)
        let token_block = create_test_token_block_from_iota(3000);
        let seq_hash = token_block.kvbm_sequence_hash();

        // Register in manager1
        let mutable_blocks1 = manager1.allocate_blocks(1).expect("Should allocate");
        let complete_block1 = mutable_blocks1
            .into_iter()
            .next()
            .unwrap()
            .complete(&token_block)
            .expect("Should complete");
        let _immutable1 = manager1.register_blocks(vec![complete_block1]);

        // Both managers should see the registered block count in shared registry
        assert!(registry.is_registered(seq_hash));
    }
}

mod capacity_lifecycle_tests {
    use super::*;

    /// Build a BlockManager with any backend. Always includes frequency_tracker
    /// so MultiLRU works; LRU/Lineage ignore it.
    fn create_backend_manager(
        block_count: usize,
        backend_builder: fn(
            BlockManagerConfigBuilder<TestBlockData>,
        ) -> BlockManagerConfigBuilder<TestBlockData>,
    ) -> BlockManager<TestBlockData> {
        let registry = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
            .build();
        backend_builder(
            BlockManager::<TestBlockData>::builder()
                .block_count(block_count)
                .block_size(4)
                .registry(registry),
        )
        .build()
        .expect("Should build manager")
    }

    /// Allocate N, complete each with a unique token block, register all.
    /// Returns the ImmutableBlocks.
    fn allocate_complete_register_all(
        manager: &BlockManager<TestBlockData>,
        block_count: usize,
        iota_base: u32,
    ) -> Vec<ImmutableBlock<TestBlockData>> {
        let mutable = manager
            .allocate_blocks(block_count)
            .expect("allocate failed");
        let complete: Vec<_> = mutable
            .into_iter()
            .enumerate()
            .map(|(i, mb)| {
                let tb = create_iota_token_block(iota_base + (i as u32 * 4), 4);
                mb.complete(&tb).expect("complete failed")
            })
            .collect();
        manager.register_blocks(complete)
    }

    // ====================================================================
    // 1. Full capacity register and return to inactive
    // ====================================================================

    #[rstest]
    #[case("lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lru_backend())]
    #[case("multi_lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_multi_lru_backend())]
    #[case("lineage", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lineage_backend())]
    fn test_full_capacity_register_and_return_to_inactive(
        #[case] _backend_name: &str,
        #[case] backend_builder: fn(
            BlockManagerConfigBuilder<TestBlockData>,
        ) -> BlockManagerConfigBuilder<TestBlockData>,
    ) {
        let manager = create_backend_manager(32, backend_builder);

        // Allocate, complete, register all 32
        let immutable = allocate_complete_register_all(&manager, 32, 5000);
        assert_eq!(manager.inactive_pool.len(), 0);
        assert_eq!(manager.reset_pool.len(), 0);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.reset_pool_size, 0);
        assert_eq!(snap.inactive_pool_size, 0);

        // Drop all ImmutableBlocks → should all land in inactive pool
        drop(immutable);
        assert_eq!(manager.inactive_pool.len(), 32);
        assert_eq!(manager.reset_pool.len(), 0);

        // Check metrics
        let snap = manager.metrics.snapshot();
        assert_eq!(snap.allocations, 32);
        assert_eq!(snap.registrations, 32);
        assert_eq!(snap.inflight_immutable, 0);
        assert_eq!(snap.inflight_mutable, 0);
        assert_eq!(snap.inactive_pool_size, 32);
        assert_eq!(snap.reset_pool_size, 0);

        // Check totals
        assert_eq!(manager.available_blocks(), 32);
        assert_eq!(manager.total_blocks(), 32);
    }

    // ====================================================================
    // 2. Full capacity eviction cycle
    // ====================================================================

    #[rstest]
    #[case("lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lru_backend())]
    #[case("multi_lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_multi_lru_backend())]
    #[case("lineage", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lineage_backend())]
    fn test_full_capacity_eviction_cycle(
        #[case] _backend_name: &str,
        #[case] backend_builder: fn(
            BlockManagerConfigBuilder<TestBlockData>,
        ) -> BlockManagerConfigBuilder<TestBlockData>,
    ) {
        let manager = create_backend_manager(16, backend_builder);

        // Allocate, register all 16
        let immutable = allocate_complete_register_all(&manager, 16, 6000);
        assert_eq!(manager.reset_pool.len(), 0);
        assert_eq!(manager.inactive_pool.len(), 0);

        // Drop all → inactive pool
        drop(immutable);
        assert_eq!(manager.inactive_pool.len(), 16);
        assert_eq!(manager.reset_pool.len(), 0);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.inactive_pool_size, 16);
        assert_eq!(snap.reset_pool_size, 0);

        // Allocate 16 again (evicts from inactive)
        let mutable = manager.allocate_blocks(16).expect("second allocate failed");
        assert_eq!(manager.inactive_pool.len(), 0);
        assert_eq!(manager.reset_pool.len(), 0);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.inactive_pool_size, 0);
        assert_eq!(snap.reset_pool_size, 0);

        // Drop mutable blocks → reset pool
        drop(mutable);
        assert_eq!(manager.reset_pool.len(), 16);
        assert_eq!(manager.inactive_pool.len(), 0);

        // Check metrics
        let snap = manager.metrics.snapshot();
        assert_eq!(snap.evictions, 16);
        assert_eq!(snap.allocations, 32);
        assert_eq!(snap.reset_pool_size, 16);
        assert_eq!(snap.inactive_pool_size, 0);
    }

    // ====================================================================
    // 3. Mutable drops go to reset, not inactive
    // ====================================================================

    #[test]
    fn test_mutable_drops_go_to_reset_not_inactive() {
        let manager = create_backend_manager(16, |b| b.with_lru_backend());

        let mutable = manager.allocate_blocks(16).expect("allocate failed");
        assert_eq!(manager.reset_pool.len(), 0);
        assert_eq!(manager.inactive_pool.len(), 0);

        // Drop all mutable blocks → reset pool
        drop(mutable);
        assert_eq!(manager.reset_pool.len(), 16);
        assert_eq!(manager.inactive_pool.len(), 0);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.inflight_mutable, 0);
        assert_eq!(snap.registrations, 0);
    }

    // ====================================================================
    // 4. Complete drops go to reset, not inactive
    // ====================================================================

    #[test]
    fn test_complete_drops_go_to_reset_not_inactive() {
        let manager = create_backend_manager(16, |b| b.with_lru_backend());

        let mutable = manager.allocate_blocks(16).expect("allocate failed");
        let complete: Vec<_> = mutable
            .into_iter()
            .enumerate()
            .map(|(i, mb)| {
                let tb = create_iota_token_block(7000 + (i as u32 * 4), 4);
                mb.complete(&tb).expect("complete failed")
            })
            .collect();
        assert_eq!(manager.reset_pool.len(), 0);

        // Drop all CompleteBlocks (not registered) → reset pool
        drop(complete);
        assert_eq!(manager.reset_pool.len(), 16);
        assert_eq!(manager.inactive_pool.len(), 0);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.stagings, 16);
        assert_eq!(snap.registrations, 0);
    }

    // ====================================================================
    // 5. Mixed return paths
    // ====================================================================

    #[rstest]
    #[case("lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lru_backend())]
    #[case("multi_lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_multi_lru_backend())]
    #[case("lineage", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lineage_backend())]
    fn test_mixed_return_paths(
        #[case] _backend_name: &str,
        #[case] backend_builder: fn(
            BlockManagerConfigBuilder<TestBlockData>,
        ) -> BlockManagerConfigBuilder<TestBlockData>,
    ) {
        let manager = create_backend_manager(24, backend_builder);

        let mutable = manager.allocate_blocks(24).expect("allocate failed");
        let mut mutable_iter = mutable.into_iter();

        // Group A (8): drop as MutableBlocks
        {
            let group_a: Vec<_> = mutable_iter.by_ref().take(8).collect();
            drop(group_a);
        }
        assert_eq!(manager.reset_pool.len(), 8);
        assert_eq!(manager.metrics.snapshot().reset_pool_size, 8);

        // Group B (8): complete, drop as CompleteBlocks
        {
            let group_b: Vec<_> = mutable_iter
                .by_ref()
                .take(8)
                .enumerate()
                .map(|(i, mb)| {
                    let tb = create_iota_token_block(8000 + (i as u32 * 4), 4);
                    mb.complete(&tb).expect("complete failed")
                })
                .collect();
            drop(group_b);
        }
        assert_eq!(manager.reset_pool.len(), 16);
        assert_eq!(manager.metrics.snapshot().reset_pool_size, 16);

        // Group C (8): complete, register, hold ImmutableBlocks
        let group_c_complete: Vec<_> = mutable_iter
            .enumerate()
            .map(|(i, mb)| {
                let tb = create_iota_token_block(8100 + (i as u32 * 4), 4);
                mb.complete(&tb).expect("complete failed")
            })
            .collect();
        let group_c_immutable = manager.register_blocks(group_c_complete);
        assert_eq!(manager.inactive_pool.len(), 0);

        // Drop Group C → inactive pool
        drop(group_c_immutable);
        assert_eq!(manager.inactive_pool.len(), 8);
        assert_eq!(manager.reset_pool.len(), 16);

        // Check totals
        assert_eq!(manager.available_blocks(), 24);

        // Check metrics
        let snap = manager.metrics.snapshot();
        assert_eq!(snap.allocations, 24);
        assert_eq!(snap.stagings, 16); // Group B (8) + Group C (8)
        assert_eq!(snap.registrations, 8);
        assert_eq!(snap.inflight_mutable, 0);
        assert_eq!(snap.inflight_immutable, 0);
        assert_eq!(snap.inactive_pool_size, 8);
        assert_eq!(snap.reset_pool_size, 16);
    }

    // ====================================================================
    // 6. MultiLRU all cold blocks at capacity (regression)
    // ====================================================================

    #[test]
    fn test_multi_lru_all_cold_blocks_at_capacity() {
        let manager = create_backend_manager(64, |b| b.with_multi_lru_backend());

        // Allocate, register all 64 (no frequency touches → all cold)
        let immutable = allocate_complete_register_all(&manager, 64, 9000);

        // Drop all → all go to level 0 (cold). With old div_ceil(4)=16
        // per-level capacity this would panic at block 17.
        drop(immutable);
        assert_eq!(manager.inactive_pool.len(), 64);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.evictions, 0);
        assert_eq!(snap.allocations, 64);
    }

    // ====================================================================
    // 7. MultiLRU mixed frequency levels
    // ====================================================================

    #[test]
    fn test_multi_lru_mixed_frequency_levels() {
        // thresholds [3, 8, 15]: cold=0-2, warm=3-7, hot=8-14, very_hot=15
        let registry = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::default().create_tracker())
            .build();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(32)
            .block_size(4)
            .registry(registry)
            .with_multi_lru_backend()
            .build()
            .expect("Should build manager");

        // Allocate, register all 32
        let immutable = allocate_complete_register_all(&manager, 32, 10000);

        // Touch frequency tracker for different blocks to spread across levels
        let tracker = manager.block_registry().frequency_tracker().unwrap();
        for block in &immutable {
            let hash = block.sequence_hash();
            let idx = block.block_id();
            let touches = if idx < 8 {
                0 // cold: 0-7 untouched
            } else if idx < 16 {
                3 // warm: 8-15
            } else if idx < 24 {
                8 // hot: 16-23
            } else {
                15 // very hot: 24-31
            };
            for _ in 0..touches {
                tracker.touch(hash.as_u128());
            }
        }

        // Drop all → distributed across 4 levels
        drop(immutable);
        assert_eq!(manager.inactive_pool.len(), 32);

        // Allocate 32 again → evicts from all levels
        let mutable = manager.allocate_blocks(32).expect("eviction allocate");
        assert_eq!(manager.inactive_pool.len(), 0);
        drop(mutable);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.evictions, 32);
        assert_eq!(snap.allocations, 64);
    }

    // ====================================================================
    // 8. Double lifecycle cycle
    // ====================================================================

    #[rstest]
    #[case("lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lru_backend())]
    #[case("multi_lru", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_multi_lru_backend())]
    #[case("lineage", |b: BlockManagerConfigBuilder<TestBlockData>| b.with_lineage_backend())]
    fn test_double_lifecycle_cycle(
        #[case] _backend_name: &str,
        #[case] backend_builder: fn(
            BlockManagerConfigBuilder<TestBlockData>,
        ) -> BlockManagerConfigBuilder<TestBlockData>,
    ) {
        let manager = create_backend_manager(16, backend_builder);
        let m = &manager.metrics;

        // Cycle 1: allocate, register, drop → inactive
        {
            let immutable = allocate_complete_register_all(&manager, 16, 11000);
            drop(immutable);
        }
        assert_eq!(manager.inactive_pool.len(), 16);
        let snap = m.snapshot();
        assert_eq!(snap.inactive_pool_size, 16);
        assert_eq!(snap.reset_pool_size, 0);

        // Evict all: allocate from inactive, drop mutable → reset
        {
            let mutable = manager.allocate_blocks(16).expect("eviction allocate");

            let snap = m.snapshot();
            assert_eq!(snap.inactive_pool_size, 0);
            assert_eq!(snap.reset_pool_size, 0);

            drop(mutable);
        }
        assert_eq!(manager.reset_pool.len(), 16);
        assert_eq!(manager.inactive_pool.len(), 0);
        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 16);
        assert_eq!(snap.inactive_pool_size, 0);

        // Cycle 2: allocate, register (different tokens), drop → inactive
        {
            let immutable = allocate_complete_register_all(&manager, 16, 12000);
            drop(immutable);
        }
        assert_eq!(manager.inactive_pool.len(), 16);

        // Check metrics
        let snap = m.snapshot();
        assert_eq!(snap.allocations, 48);
        assert_eq!(snap.registrations, 32);
        assert_eq!(snap.evictions, 16);
        assert_eq!(snap.inactive_pool_size, 16);
        assert_eq!(snap.reset_pool_size, 0);

        // Check totals
        assert_eq!(manager.available_blocks(), 16);
        assert_eq!(manager.total_blocks(), 16);
    }
}

// ============================================================================
// SCAN MATCHES POOL SIZE GAUGE TESTS
// ============================================================================

mod scan_matches_tests {
    use super::*;

    #[test]
    fn test_scan_matches_with_pool_size_gauges() {
        let manager = create_test_manager(10);
        let m = manager.metrics();

        // Register 3 blocks with distinct hashes
        let mut seq_hashes = Vec::new();
        for i in 0..3 {
            let tb = create_iota_token_block(13000 + (i as u32 * 4), 4);
            seq_hashes.push(tb.kvbm_sequence_hash());

            let mutable = manager.allocate_blocks(1).expect("allocate");
            let complete = mutable
                .into_iter()
                .next()
                .unwrap()
                .complete(&tb)
                .expect("complete");
            let immutable = manager.register_blocks(vec![complete]);
            drop(immutable);
        }

        // All 3 should be in inactive pool
        assert_eq!(manager.inactive_pool.len(), 3);
        let snap = m.snapshot();
        assert_eq!(snap.inactive_pool_size, 3);
        assert_eq!(snap.reset_pool_size, 7);

        // scan_matches with 2 matching + 1 missing hash
        let missing_hash = create_iota_token_block(99000, 4).kvbm_sequence_hash();
        let scan_hashes = vec![seq_hashes[0], missing_hash, seq_hashes[2]];

        let found = manager.scan_matches(&scan_hashes, true);
        assert_eq!(found.len(), 2);

        // inactive_pool_size decreased by 2
        let snap = m.snapshot();
        assert_eq!(snap.inactive_pool_size, 1);

        // Drop scanned blocks → they return to inactive
        drop(found);

        let snap = m.snapshot();
        assert_eq!(snap.inactive_pool_size, 3);
    }
}
