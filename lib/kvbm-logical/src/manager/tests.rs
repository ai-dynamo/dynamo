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
    testing::create_test_manager(block_count)
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

        // One-element prefix matches use the first-hash fast path.
        let matched_blocks = manager.match_blocks(&[seq_hash]);
        assert_eq!(matched_blocks.len(), 1);
        assert_eq!(matched_blocks[0].sequence_hash(), seq_hash);

        let snap = m.snapshot();
        assert_eq!(snap.match_hashes_requested, 1);
        assert_eq!(snap.match_blocks_returned, 1);

        drop(matched_blocks);
        drop(_immutable_blocks);

        // The same API must also reactivate an inactive block.
        let reactivated = manager.match_blocks(&[seq_hash]);
        assert_eq!(reactivated.len(), 1);
        assert_eq!(reactivated[0].sequence_hash(), seq_hash);

        let snap = m.snapshot();
        assert_eq!(snap.match_hashes_requested, 2);
        assert_eq!(snap.match_blocks_returned, 2);
        assert_eq!(snap.inactive_pool_size, 0);
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
// SINGLE-LOCK match_blocks REWRITE TESTS
// ============================================================================

/// Coverage for the single-lock `match_prefix_locked_batch` rewrite of
/// `match_blocks`. The pre-rewrite code already resolved each hash as
/// active-or-inactive (via `find_active_match` → `acquire_for_hash`), so
/// these are regression guards: they must pass identically on the old and
/// new code, locking in that collapsing N store-lock cycles into one did
/// not change the prefix the walk returns.
mod single_lock_match_tests {
    use super::*;

    /// Allocate, complete, and register one independent block. Returns its
    /// `SequenceHash` and the strong `ImmutableBlock` handle (hold it to
    /// keep the block `Primary`/active; drop it to push it to `Inactive`).
    fn register_one(
        manager: &BlockManager<TestBlockData>,
        base: u32,
    ) -> (SequenceHash, ImmutableBlock<TestBlockData>) {
        let token_block = create_token_block(&[base, base + 1, base + 2, base + 3]);
        let seq_hash = token_block.kvbm_sequence_hash();
        let mutable = manager
            .allocate_blocks(1)
            .expect("allocate")
            .into_iter()
            .next()
            .unwrap();
        let complete = mutable.complete(&token_block).expect("complete");
        let immutable = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        (seq_hash, immutable)
    }

    /// `[A, A, I, I, A, A]` — a prefix mixing active and inactive blocks
    /// must return all 6. Each hash is resolved active-or-inactive under
    /// the single batched store-lock; an inactive hash in the middle does
    /// NOT truncate the prefix.
    #[test]
    fn match_mixed_active_inactive_prefix_returns_full_length() {
        let manager = create_test_manager(6);

        let mut hashes = Vec::new();
        let mut held = Vec::new();
        for i in 0..6u32 {
            let (h, imm) = register_one(&manager, 1_000 + i * 10);
            hashes.push(h);
            held.push(Some(imm));
        }
        // Drop the immutables for positions 2 and 3 → those fall to Inactive.
        held[2] = None;
        held[3] = None;

        let matched = manager.match_blocks(&hashes);
        assert_eq!(
            matched.len(),
            6,
            "mixed active/inactive prefix must not truncate"
        );
        for (i, block) in matched.iter().enumerate() {
            assert_eq!(block.sequence_hash(), hashes[i], "order preserved at {i}");
        }
    }

    /// `[I, A]` — leading inactive hash, trailing active. Returns 2.
    #[test]
    fn match_inactive_then_active() {
        let manager = create_test_manager(2);
        let (h0, imm0) = register_one(&manager, 2_000);
        let (h1, _imm1) = register_one(&manager, 2_010);
        drop(imm0); // h0 → Inactive; h1 stays active.

        let matched = manager.match_blocks(&[h0, h1]);
        assert_eq!(matched.len(), 2);
        assert_eq!(matched[0].sequence_hash(), h0);
        assert_eq!(matched[1].sequence_hash(), h1);
    }

    /// `[A, I, A]` — inactive hash sandwiched between active ones. Returns 3.
    #[test]
    fn match_active_inactive_active() {
        let manager = create_test_manager(3);
        let (h0, _imm0) = register_one(&manager, 3_000);
        let (h1, imm1) = register_one(&manager, 3_010);
        let (h2, _imm2) = register_one(&manager, 3_020);
        drop(imm1); // h1 → Inactive.

        let matched = manager.match_blocks(&[h0, h1, h2]);
        assert_eq!(matched.len(), 3);
        assert_eq!(matched[1].sequence_hash(), h1);
    }

    /// `[A, A, miss, A]` — an unregistered hash terminates the prefix even
    /// though a registered hash follows it. Returns 2.
    #[test]
    fn match_stops_at_first_total_miss() {
        let manager = create_test_manager(4);
        let (h0, _imm0) = register_one(&manager, 4_000);
        let (h1, _imm1) = register_one(&manager, 4_010);
        let (h3, _imm3) = register_one(&manager, 4_030);
        // `miss` is a hash for a block that was never registered.
        let miss = create_token_block(&[4_020, 4_021, 4_022, 4_023]).kvbm_sequence_hash();

        let matched = manager.match_blocks(&[h0, h1, miss, h3]);
        assert_eq!(
            matched.len(),
            2,
            "prefix terminates at the first total miss"
        );
        assert_eq!(matched[0].sequence_hash(), h0);
        assert_eq!(matched[1].sequence_hash(), h1);
    }

    /// Empty input is an allocation-free early return.
    #[test]
    fn match_empty_input_returns_empty() {
        let manager = create_test_manager(4);
        let _held = register_one(&manager, 5_000);
        assert!(manager.match_blocks(&[]).is_empty());
        assert_eq!(manager.metrics().snapshot().match_blocks_returned, 0);
    }

    /// The batched frequency-tracker touch fires exactly once per returned
    /// block across a mixed active/inactive prefix — N hits → N touches,
    /// no double-count, no miss.
    #[test]
    fn match_mixed_prefix_touches_once_per_returned_block() {
        let (manager, metered) = crate::testing::create_test_manager_metered::<TestBlockData>(5);

        let mut hashes = Vec::new();
        let mut held = Vec::new();
        for i in 0..5u32 {
            let (h, imm) = register_one(&manager, 6_000 + i * 10);
            hashes.push(h);
            held.push(Some(imm));
        }
        // Positions 1 and 3 → Inactive; 0, 2, 4 stay active.
        held[1] = None;
        held[3] = None;

        metered.reset();
        let matched = manager.match_blocks(&hashes);
        assert_eq!(matched.len(), 5);
        assert_eq!(
            metered.touches(),
            5,
            "match_blocks must touch the frequency tracker exactly once per \
             returned block (mix of active hits and inactive resurrections); \
             got {}",
            metered.touches()
        );
    }

    /// A whole-prefix total miss touches nothing and returns empty.
    #[test]
    fn match_total_miss_touches_nothing() {
        let (manager, metered) = crate::testing::create_test_manager_metered::<TestBlockData>(4);
        let _held = register_one(&manager, 7_000);
        let miss = create_token_block(&[9_000, 9_001, 9_002, 9_003]).kvbm_sequence_hash();

        metered.reset();
        assert!(manager.match_blocks(&[miss]).is_empty());
        assert_eq!(metered.touches(), 0, "a total miss must not touch");
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
        assert_eq!(manager.store.inactive_len(), 0);
        assert_eq!(manager.store.reset_len(), 0);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.reset_pool_size, 0);
        assert_eq!(snap.inactive_pool_size, 0);

        // Drop all ImmutableBlocks → should all land in inactive pool
        drop(immutable);
        assert_eq!(manager.store.inactive_len(), 32);
        assert_eq!(manager.store.reset_len(), 0);

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
        assert_eq!(manager.store.reset_len(), 0);
        assert_eq!(manager.store.inactive_len(), 0);

        // Drop all → inactive pool
        drop(immutable);
        assert_eq!(manager.store.inactive_len(), 16);
        assert_eq!(manager.store.reset_len(), 0);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.inactive_pool_size, 16);
        assert_eq!(snap.reset_pool_size, 0);

        // Allocate 16 again (evicts from inactive)
        let mutable = manager.allocate_blocks(16).expect("second allocate failed");
        assert_eq!(manager.store.inactive_len(), 0);
        assert_eq!(manager.store.reset_len(), 0);

        let snap = manager.metrics.snapshot();
        assert_eq!(snap.inactive_pool_size, 0);
        assert_eq!(snap.reset_pool_size, 0);

        // Drop mutable blocks → reset pool
        drop(mutable);
        assert_eq!(manager.store.reset_len(), 16);
        assert_eq!(manager.store.inactive_len(), 0);

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
        assert_eq!(manager.store.reset_len(), 0);
        assert_eq!(manager.store.inactive_len(), 0);

        // Drop all mutable blocks → reset pool
        drop(mutable);
        assert_eq!(manager.store.reset_len(), 16);
        assert_eq!(manager.store.inactive_len(), 0);

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
        assert_eq!(manager.store.reset_len(), 0);

        // Drop all CompleteBlocks (not registered) → reset pool
        drop(complete);
        assert_eq!(manager.store.reset_len(), 16);
        assert_eq!(manager.store.inactive_len(), 0);

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
        assert_eq!(manager.store.reset_len(), 8);
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
        assert_eq!(manager.store.reset_len(), 16);
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
        assert_eq!(manager.store.inactive_len(), 0);

        // Drop Group C → inactive pool
        drop(group_c_immutable);
        assert_eq!(manager.store.inactive_len(), 8);
        assert_eq!(manager.store.reset_len(), 16);

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
        assert_eq!(manager.store.inactive_len(), 64);

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
        assert_eq!(manager.store.inactive_len(), 32);

        // Allocate 32 again → evicts from all levels
        let mutable = manager.allocate_blocks(32).expect("eviction allocate");
        assert_eq!(manager.store.inactive_len(), 0);
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
        assert_eq!(manager.store.inactive_len(), 16);
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
        assert_eq!(manager.store.reset_len(), 16);
        assert_eq!(manager.store.inactive_len(), 0);
        let snap = m.snapshot();
        assert_eq!(snap.reset_pool_size, 16);
        assert_eq!(snap.inactive_pool_size, 0);

        // Cycle 2: allocate, register (different tokens), drop → inactive
        {
            let immutable = allocate_complete_register_all(&manager, 16, 12000);
            drop(immutable);
        }
        assert_eq!(manager.store.inactive_len(), 16);

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
        assert_eq!(manager.store.inactive_len(), 3);
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

// ============================================================================
// RACE-CONDITION REGRESSION TESTS
//
// These tests cover the three high-severity races that motivated the
// move of active-by-hash, slot Weak ownership, and presence refcounting
// into BlockStore. They stress concurrent code paths that were known to
// produce inconsistent state under the previous two-lock design.
// ============================================================================

mod race_regression_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    use crate::pools::BlockDuplicationPolicy;

    fn build_manager_with_policy(
        block_count: usize,
        policy: BlockDuplicationPolicy,
    ) -> BlockManager<TestBlockData> {
        let registry = BlockRegistry::new();
        BlockManager::<TestBlockData>::builder()
            .block_count(block_count)
            .block_size(4)
            .registry(registry)
            .duplication_policy(policy)
            .build()
            .expect("build manager")
    }

    /// Finding 1: an active-pool lookup that races with the last
    /// `ImmutableBlock` drop must either succeed (returning the existing
    /// or a freshly resurrected block) or miss cleanly. It must never
    /// produce a state where two slots claim the same `seq_hash` as
    /// Primary.
    #[test]
    fn active_lookup_concurrent_with_last_drop() {
        const ITERATIONS: usize = 200;
        let manager = Arc::new(create_test_manager(8));

        for i in 0..ITERATIONS {
            let token_block = create_test_token_block_from_iota((10_000 + i * 4) as u32);
            let seq_hash = token_block.kvbm_sequence_hash();

            // Stage and register one primary.
            let mutables = manager.allocate_blocks(1).unwrap();
            let complete = mutables
                .into_iter()
                .next()
                .unwrap()
                .complete(&token_block)
                .unwrap();
            let immutable = manager
                .register_blocks(vec![complete])
                .into_iter()
                .next()
                .unwrap();

            // Race: thread A drops the last strong reference; thread B
            // looks the hash up. Repeat with manager-shared barriers.
            let manager_a = Arc::clone(&manager);
            let manager_b = Arc::clone(&manager);
            let drop_thread = thread::spawn(move || {
                drop(immutable);
                // One additional touch to ensure release_primary runs.
                manager_a.available_blocks();
            });
            let lookup_thread = thread::spawn(move || manager_b.match_blocks(&[seq_hash]));

            drop_thread.join().unwrap();
            let matched = lookup_thread.join().unwrap();
            assert!(
                matched.len() <= 1,
                "double-primary detected for {seq_hash:?}"
            );

            // Drop the lookup result; the system must converge.
            drop(matched);

            // Total slots must always equal block_count.
            assert_eq!(manager.total_blocks(), 8);
        }
    }

    /// Finding 2: when a block is evicted from the inactive pool while
    /// another thread re-registers a fresh `CompleteBlock` for the same
    /// hash, `check_presence::<T>` must report `true` afterwards. The
    /// previous design lost the new presence marker to the trailing
    /// `mark_absent` from the eviction path.
    #[test]
    fn eviction_concurrent_with_reregister_preserves_presence() {
        const ITERATIONS: usize = 100;
        // 1 block so a fresh registration must evict the inactive entry.
        let manager = Arc::new(create_test_manager(1));

        for i in 0..ITERATIONS {
            let token_a = create_test_token_block_from_iota((20_000 + i * 8) as u32);
            let token_b = create_test_token_block_from_iota((20_000 + i * 8 + 4) as u32);
            let hash_a = token_a.kvbm_sequence_hash();
            let hash_b = token_b.kvbm_sequence_hash();

            // Register one block, drop it so it lands in inactive.
            let mut mutables = manager.allocate_blocks(1).unwrap();
            let complete = mutables.pop().unwrap().complete(&token_a).unwrap();
            let immutable = manager
                .register_blocks(vec![complete])
                .into_iter()
                .next()
                .unwrap();
            drop(immutable);

            // Now allocate (forcing eviction of hash_a) and concurrently
            // register hash_b (the freshly evicted slot will be re-used
            // for hash_b). With one slot total, the only path possible
            // is "register a fresh hash_b that displaces hash_a".
            let mut mutables = manager.allocate_blocks(1).unwrap();
            let complete_b = mutables.pop().unwrap().complete(&token_b).unwrap();
            let immutable_b = manager
                .register_blocks(vec![complete_b])
                .into_iter()
                .next()
                .unwrap();

            // hash_a must no longer be present (evicted), hash_b must be
            // present.
            let presence = manager
                .block_registry()
                .check_presence::<TestBlockData>(&[hash_a, hash_b]);
            assert_eq!(
                presence,
                vec![(hash_a, false), (hash_b, true)],
                "iteration {i}: presence after eviction-vs-register"
            );

            drop(immutable_b);
        }
    }

    /// Finding 3: dropping an allowed duplicate must not clear presence
    /// while the primary is still alive. With the old boolean-set
    /// presence_markers, `release_duplicate` unconditionally cleared
    /// presence even though a primary remained.
    #[test]
    fn duplicate_drop_preserves_presence() {
        let manager = build_manager_with_policy(4, BlockDuplicationPolicy::Allow);
        let token = create_test_token_block_from_iota(30_000);

        // Allocate two slots, complete both with the same hash. The
        // second registration becomes a duplicate.
        let mutables = manager.allocate_blocks(2).unwrap();
        let mut iter = mutables.into_iter();
        let complete_primary = iter.next().unwrap().complete(&token).unwrap();
        let complete_dup = iter.next().unwrap().complete(&token).unwrap();

        let primary = manager
            .register_blocks(vec![complete_primary])
            .into_iter()
            .next()
            .unwrap();
        let dup = manager
            .register_blocks(vec![complete_dup])
            .into_iter()
            .next()
            .unwrap();

        let handle = primary.registration_handle();
        assert!(handle.has_block::<TestBlockData>(), "before any drop");

        // Drop the duplicate: presence must remain true (primary still
        // alive).
        drop(dup);
        assert!(
            handle.has_block::<TestBlockData>(),
            "after duplicate drop, primary still alive"
        );

        // Drop the primary: it transitions to Inactive — still
        // presence-bearing.
        drop(primary);
        assert!(
            handle.has_block::<TestBlockData>(),
            "after primary drop, slot in Inactive still counts"
        );
    }

    /// Allocation atomicity: with `free + inactive == N`, two concurrent
    /// requests for `N` blocks each must not both succeed at the same
    /// time. We synchronize all threads at a barrier before allocating
    /// and have each successful thread hold its blocks until all
    /// threads have attempted, so no thread can refill the pool early.
    #[test]
    fn allocate_atomic_no_over_commit_under_contention() {
        use std::sync::Barrier;

        const TOTAL: usize = 16;
        const THREADS: usize = 8;
        let manager = Arc::new(create_test_manager(TOTAL));
        let granted = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(THREADS));
        let after_alloc = Arc::new(Barrier::new(THREADS));

        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let m = Arc::clone(&manager);
                let g = Arc::clone(&granted);
                let s = Arc::clone(&start);
                let a = Arc::clone(&after_alloc);
                thread::spawn(move || {
                    s.wait();
                    let blocks = m.allocate_blocks(TOTAL);
                    if let Some(b) = &blocks {
                        g.fetch_add(b.len(), Ordering::SeqCst);
                    }
                    // Hold (or not) until everyone has attempted; this
                    // prevents an early dropper from refilling the pool
                    // and giving a second thread a green light.
                    a.wait();
                    drop(blocks);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        // Exactly one thread (or zero, in pathological scheduling)
        // should have received TOTAL blocks. No double-allocation.
        let g = granted.load(Ordering::SeqCst);
        assert!(
            g == 0 || g == TOTAL,
            "concurrent over-commit detected: granted={g}, expected 0 or {TOTAL}"
        );
        assert_eq!(manager.available_blocks(), TOTAL);
    }
}

// ============================================================================
// LOCK-ORDER ENFORCEMENT (tracing-mutex DAG)
//
// Under `#[cfg(test)]` the crate's parking_lot::Mutex types are
// swapped for `tracing_mutex::parkinglot::Mutex`, which builds a
// global lock-acquisition DAG and panics on cycle detection. This
// runtime-enforces the documented `attachments → store` ordering
// throughout the test suite. The sentinel test below proves the DAG
// is wired and active — if `tracing-mutex` is ever silently dropped
// or downgraded, this test stops panicking and the
// `#[should_panic]` mismatch flags the regression.
// ============================================================================

mod lock_order_enforcement_tests {
    use tracing_mutex::parkinglot::Mutex;

    /// Sentinel: deliberately introduce a cycle (a→b followed by
    /// b→a) using two `tracing_mutex::parkinglot::Mutex` instances
    /// drawn from the same dependency-tracking type that the crate's
    /// real locks use under `#[cfg(test)]`. The DAG must detect the
    /// cycle and panic. Failing this test means lock-order
    /// enforcement is no longer in place crate-wide.
    #[test]
    #[should_panic(expected = "Found cycle in mutex dependency graph")]
    fn deliberate_inversion_is_caught_by_tracing_mutex_dag() {
        let a: Mutex<()> = Mutex::new(());
        let b: Mutex<()> = Mutex::new(());

        // Forward edge a → b.
        {
            let _ga = a.lock();
            let _gb = b.lock();
        }
        // Reverse edge b → a — DAG cycle; tracing-mutex panics here.
        let _gb = b.lock();
        let _ga = a.lock();
    }
}

// ============================================================================
// AUDIT-COUNTER COVERAGE TESTS
//
// These tests target normally-rare branches by asserting on the audit
// counters added to BlockPoolMetrics. They catch regressions that
// emergent-behavior assertions would miss (e.g. the multi-LRU
// double-touch slipped past behavior tests because TinyLFU is a sketch
// and `count()` is approximate; an exact `touches()` counter on a
// metered FrequencyTracker fails loudly the moment it doubles).
// ============================================================================

mod audit_counter_tests {
    use super::*;
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::thread;

    use crate::SequenceHash;
    use crate::blocks::BlockId;
    use crate::manager::FrequencyTrackingCapacity;
    use crate::pools::backends::MultiLruBackend;
    use crate::pools::store::InactiveIndex;
    use crate::testing::MeteredFrequencyTracker;

    fn build_manager_with_metered_multi_lru(
        block_count: usize,
    ) -> (BlockManager<TestBlockData>, Arc<MeteredFrequencyTracker>) {
        let metered =
            MeteredFrequencyTracker::with_tinylfu(FrequencyTrackingCapacity::default().size());
        let registry = BlockRegistry::builder()
            .frequency_tracker(metered.clone())
            .build();
        let mgr = BlockManager::<TestBlockData>::builder()
            .block_count(block_count)
            .block_size(4)
            .registry(registry)
            .with_multi_lru_backend()
            .build()
            .expect("build manager");
        (mgr, metered)
    }

    /// Exact-call counter test: a single `match_blocks([hash])` against
    /// a hash that lives in the inactive pool must increment the
    /// frequency tracker `touch()` exactly once. The previous design
    /// double-counted (registry touch + backend touch). This test would
    /// have failed loudly when the double-touch was introduced.
    #[test]
    fn match_inactive_block_touches_frequency_tracker_exactly_once() {
        let (manager, metered) = build_manager_with_metered_multi_lru(4);

        // Stage one block, register it, drop the immutable so it lands
        // in inactive.
        let token = create_test_token_block_from_iota(40_000);
        let hash = token.kvbm_sequence_hash();
        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let immutable = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        drop(immutable);

        // Reset counters to isolate the match() under test.
        metered.reset();

        let matched = manager.match_blocks(&[hash]);
        assert_eq!(matched.len(), 1);
        assert_eq!(
            metered.touches(),
            1,
            "match_blocks against an inactive hash must touch the frequency \
             tracker exactly once; got {} (likely a registry+backend \
             double-touch regression)",
            metered.touches()
        );
    }

    /// Same invariant for `scan_matches(touch=true)`.
    #[test]
    fn scan_inactive_block_touches_frequency_tracker_exactly_once() {
        let (manager, metered) = build_manager_with_metered_multi_lru(4);
        let token = create_test_token_block_from_iota(40_100);
        let hash = token.kvbm_sequence_hash();
        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let immutable = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        drop(immutable);

        metered.reset();
        let scanned = manager.scan_matches(&[hash], /*touch=*/ true);
        assert_eq!(scanned.len(), 1);
        assert_eq!(
            metered.touches(),
            1,
            "scan_matches(touch=true) against an inactive hash must touch \
             the frequency tracker exactly once; got {}",
            metered.touches()
        );
    }

    /// `scan_matches(touch=false)` must not touch.
    #[test]
    fn scan_inactive_block_with_touch_false_does_not_touch() {
        let (manager, metered) = build_manager_with_metered_multi_lru(4);
        let token = create_test_token_block_from_iota(40_200);
        let hash = token.kvbm_sequence_hash();
        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let immutable = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        drop(immutable);

        metered.reset();
        let scanned = manager.scan_matches(&[hash], /*touch=*/ false);
        assert_eq!(scanned.len(), 1);
        assert_eq!(metered.touches(), 0, "touch=false must not touch");
    }

    /// `match_blocks` against an *active* hash also touches exactly once.
    #[test]
    fn match_active_block_touches_frequency_tracker_exactly_once() {
        let (manager, metered) = build_manager_with_metered_multi_lru(4);
        let token = create_test_token_block_from_iota(40_300);
        let hash = token.kvbm_sequence_hash();
        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let _immutable = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        // Hold _immutable so the slot stays Primary (active).

        metered.reset();
        let matched = manager.match_blocks(&[hash]);
        assert_eq!(matched.len(), 1);
        assert_eq!(
            metered.touches(),
            1,
            "match_blocks on an active hash should touch exactly once"
        );
    }

    /// Deterministic eager-transition test using the test-only
    /// `pause_release_primary` hook on `BlockStore`. A held guard
    /// blocks every `release_primary` *before* it takes the store
    /// lock, so the test can:
    ///   1. Spawn a thread that drops the last `ImmutableBlock`. Its
    ///      `Inner::drop` enters `release_primary` and parks on the
    ///      gate — the `Arc` strong count is 0 but the slot is still
    ///      `Primary` with a now-dead `Weak`.
    ///   2. From the main thread call `match_blocks([hash])`. It takes
    ///      the store lock, sees `Primary` with a dead `Weak`, and
    ///      drives `eager_primary_to_inactive_locked` then resurrects
    ///      from the inactive pool.
    ///   3. Release the gate; the parked drop completes and observes
    ///      the slot is no longer `Primary` for its `self_ptr`, so it
    ///      no-ops via `release_primary_noop`.
    ///
    /// Both audit counters tick exactly once. No timing dependencies.
    #[test]
    fn eager_primary_to_inactive_is_deterministic_with_pause_hook() {
        let manager = Arc::new(create_test_manager(4));
        let token = create_test_token_block_from_iota(70_000);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let immutable = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();

        let store = manager.store_for_test();

        let snap_before = manager.metrics().snapshot();
        assert_eq!(snap_before.eager_primary_to_inactive_total, 0);
        assert_eq!(snap_before.release_primary_noop_total, 0);

        // Hold the gate. Spawn a thread that drops the immutable —
        // its Inner::drop will park inside release_primary after its
        // own Arc strong-count went to zero.
        let gate = store.pause_release_primary();
        let arrivals_before = store.release_primary_arrivals();
        let drop_t = thread::spawn(move || drop(immutable));

        // Spin (yielding) until the drop thread has entered
        // release_primary and is about to park on the gate. The
        // arrivals counter is bumped at the first instruction of
        // release_primary, so once it ticks we know the drop has
        // committed to the gate path; the gate we hold then forces
        // it to park before mutating any slot state. No sleep — no
        // scheduler dependency.
        while store.release_primary_arrivals() == arrivals_before {
            std::thread::yield_now();
        }

        // Now the slot is `Primary { weak: dead }`. A lookup should
        // drive the eager transition and resurrect under one lock.
        let matched = manager.match_blocks(&[hash]);
        assert_eq!(matched.len(), 1, "lookup must resurrect");
        let snap_mid = manager.metrics().snapshot();
        assert_eq!(
            snap_mid.eager_primary_to_inactive_total, 1,
            "eager-transition counter must tick exactly once"
        );

        // Release the gate FIRST so the parked drop_t can run. We
        // cannot drop `matched` while holding the gate — that would
        // re-enter `release_primary` on the current thread and
        // deadlock against our own held guard (parking_lot::Mutex is
        // non-reentrant).
        drop(gate);
        drop_t.join().unwrap();

        let snap_after_drop = manager.metrics().snapshot();
        assert_eq!(
            snap_after_drop.release_primary_noop_total, 1,
            "release-primary no-op must tick exactly once (the parked \
             drop saw a slot that no longer matched its self_ptr)"
        );

        // Now safely drop the resurrected matched. This goes through
        // a normal (non-noop) `release_primary`, which must NOT bump
        // the no-op counter again.
        drop(matched);
        let snap_final = manager.metrics().snapshot();
        assert_eq!(
            snap_final.release_primary_noop_total, 1,
            "no-op counter unchanged after normal drop"
        );
    }

    /// `allocate_atomic` rollback path: wire the manager against a fake
    /// `InactiveIndex` that lies about its `len()` so `allocate(n)`
    /// returns fewer than `n`. The defensive rollback in
    /// `BlockStore::allocate_atomic` must restore reset-pool order, leave
    /// inactive intact, and bump `allocate_atomic_rollback_total`.
    #[test]
    fn allocate_atomic_rollback_counter_ticks_with_under_allocating_backend() {
        // A backend that claims `len() == reported_len` but returns 0
        // pairs from `allocate()`. Built by wrapping a real
        // `MultiLruBackend` and overriding only `len()`.
        struct UnderAllocatingBackend {
            inner: MultiLruBackend,
            reported_len: usize,
        }
        impl InactiveIndex for UnderAllocatingBackend {
            fn find_matches(
                &mut self,
                hashes: &[SequenceHash],
                touch: bool,
            ) -> Vec<(SequenceHash, BlockId)> {
                self.inner.find_matches(hashes, touch)
            }
            fn scan_matches(
                &mut self,
                hashes: &[SequenceHash],
                touch: bool,
            ) -> Vec<(SequenceHash, BlockId)> {
                self.inner.scan_matches(hashes, touch)
            }
            fn allocate(&mut self, _count: usize) -> Vec<(SequenceHash, BlockId)> {
                Vec::new() // Under-allocates: returns 0 regardless.
            }
            fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
                self.inner.insert(seq_hash, block_id);
            }
            fn len(&self) -> usize {
                self.reported_len
            }
            fn has(&self, seq_hash: SequenceHash) -> bool {
                self.inner.has(seq_hash)
            }
            fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
                self.inner.take(seq_hash, block_id)
            }
        }

        // Build a BlockStore directly with the under-allocating backend.
        // We can't easily go through the BlockManager builder for this
        // because it picks its own backend; instead we hand-roll the
        // store and call `allocate_atomic` against it.
        use crate::metrics::BlockPoolMetrics;
        use crate::pools::store::BlockStore;

        let metrics = Arc::new(BlockPoolMetrics::new("test".to_string()));
        let tracker = FrequencyTrackingCapacity::default().create_tracker();
        let backend = UnderAllocatingBackend {
            inner: MultiLruBackend::new(NonZeroUsize::new(4).unwrap(), tracker),
            reported_len: 2, // lie: claims 2, allocate() returns 0
        };
        let store: Arc<BlockStore<TestBlockData>> =
            BlockStore::new(4, 4, Box::new(backend), metrics.clone(), false);

        // free.len() is 4 (all reset). Asking for 5 forces from_inactive=1
        // which trips into the allocate path → backend lies → rollback.
        let result = store.allocate_atomic(5);
        assert!(result.is_none(), "rollback must yield None");
        let snap = metrics.snapshot();
        assert_eq!(
            snap.allocate_atomic_rollback_total, 1,
            "rollback counter must tick exactly once"
        );
        // Reset pool must be intact (FIFO preserved).
        assert_eq!(store.reset_len(), 4, "all reset blocks restored");
    }

    /// Concurrency stress against `allocate_atomic` to confirm the
    /// rollback counter stays at zero with shipped backends (i.e. no
    /// false positives in production).
    #[test]
    fn allocate_atomic_rollback_counter_zero_with_real_backend() {
        const TOTAL: usize = 16;
        const THREADS: usize = 8;
        let manager = Arc::new(create_test_manager(TOTAL));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let m = Arc::clone(&manager);
                thread::spawn(move || {
                    let _ = m.allocate_blocks(TOTAL / 4);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let snap = manager.metrics().snapshot();
        assert_eq!(
            snap.allocate_atomic_rollback_total, 0,
            "real LRU backend must never trigger rollback"
        );
    }

    /// `BlockManager::reset_inactive_pool` lifecycle: drop registered
    /// blocks so they land in the inactive pool, then reset and verify
    /// (1) reset gauge fully restored, (2) inactive gauge zeroed,
    /// (3) registry presence cleared for the type, (4) `mark_absent`
    /// counters in `presence_markers` reach zero.
    #[test]
    fn reset_inactive_pool_drains_and_clears_presence() {
        const N: usize = 4;
        let manager = create_test_manager(N);
        let mut hashes = Vec::with_capacity(N);

        // Register N blocks, drop the immutables → all land in inactive.
        let mutables = manager.allocate_blocks(N).unwrap();
        let mut completes = Vec::with_capacity(N);
        for (i, mb) in mutables.into_iter().enumerate() {
            let tb = create_test_token_block_from_iota((80_000 + i * 4) as u32);
            hashes.push(tb.kvbm_sequence_hash());
            completes.push(mb.complete(&tb).unwrap());
        }
        let immutables = manager.register_blocks(completes);
        drop(immutables);

        // Pre-reset state: all in inactive, presence true.
        let snap = manager.metrics().snapshot();
        assert_eq!(snap.inactive_pool_size, N as i64);
        assert_eq!(snap.reset_pool_size, 0);
        let presence = manager
            .block_registry()
            .check_presence::<TestBlockData>(&hashes);
        assert!(
            presence.iter().all(|(_, p)| *p),
            "all hashes present pre-reset"
        );

        // Reset.
        manager.reset_inactive_pool().expect("reset succeeds");

        // Post-reset state.
        let snap = manager.metrics().snapshot();
        assert_eq!(
            snap.inactive_pool_size, 0,
            "inactive gauge zeroed after reset"
        );
        assert_eq!(snap.reset_pool_size, N as i64, "reset gauge fully restored");
        assert_eq!(manager.available_blocks(), N);
        let presence = manager
            .block_registry()
            .check_presence::<TestBlockData>(&hashes);
        assert!(
            presence.iter().all(|(_, p)| !*p),
            "all hashes absent post-reset"
        );
    }

    /// Partial under-allocation: backend reports `len=4`, returns 2
    /// pairs from `allocate(3)`. Rollback must reinsert those 2 into
    /// the inactive index (covering the loop body in the rollback
    /// path) and restore the reset queue. Counter ticks.
    #[test]
    fn allocate_atomic_rollback_handles_partial_under_allocation() {
        struct PartiallyUnderAllocatingBackend {
            inner: MultiLruBackend,
            reported_len: usize,
            return_count: usize,
        }
        impl InactiveIndex for PartiallyUnderAllocatingBackend {
            fn find_matches(
                &mut self,
                hashes: &[SequenceHash],
                touch: bool,
            ) -> Vec<(SequenceHash, BlockId)> {
                self.inner.find_matches(hashes, touch)
            }
            fn scan_matches(
                &mut self,
                hashes: &[SequenceHash],
                touch: bool,
            ) -> Vec<(SequenceHash, BlockId)> {
                self.inner.scan_matches(hashes, touch)
            }
            fn allocate(&mut self, _count: usize) -> Vec<(SequenceHash, BlockId)> {
                // Return only `return_count` items even if asked for more.
                self.inner.allocate(self.return_count)
            }
            fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId) {
                self.inner.insert(seq_hash, block_id);
            }
            fn len(&self) -> usize {
                self.reported_len
            }
            fn has(&self, seq_hash: SequenceHash) -> bool {
                self.inner.has(seq_hash)
            }
            fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool {
                self.inner.take(seq_hash, block_id)
            }
        }

        use crate::metrics::BlockPoolMetrics;
        use crate::pools::store::BlockStore;

        let metrics = Arc::new(BlockPoolMetrics::new("test".to_string()));
        let tracker = FrequencyTrackingCapacity::default().create_tracker();
        // Pre-populate the inner backend with 4 entries by inserting
        // synthetic (hash, block_id) pairs.
        let mut inner = MultiLruBackend::new(NonZeroUsize::new(4).unwrap(), tracker);
        for i in 0..4u32 {
            let h = create_test_token_block_from_iota(90_000 + i * 4).kvbm_sequence_hash();
            inner.insert(h, i as BlockId);
        }
        let backend = PartiallyUnderAllocatingBackend {
            inner,
            reported_len: 4, // claims 4
            return_count: 2, // returns only 2 from allocate()
        };
        // 4 reset slots; backend lies that it has 4 inactive entries.
        // Asking for 7 forces from_reset=4, from_inactive=3, then
        // backend.allocate(3) returns 2 → rollback fires and reinserts
        // those 2 partial pairs into the inactive index.
        let store: Arc<BlockStore<TestBlockData>> =
            BlockStore::new(4, 4, Box::new(backend), metrics.clone(), false);

        let result = store.allocate_atomic(7);
        assert!(result.is_none(), "rollback returns None");

        let snap = metrics.snapshot();
        assert_eq!(
            snap.allocate_atomic_rollback_total, 1,
            "rollback counter ticks exactly once"
        );
        // Reset queue restored to original 4 (FIFO via push_front in reverse).
        assert_eq!(store.reset_len(), 4);
        // The partial pairs were reinserted into the index — exercises
        // the `for (h, id) in evicted_pairs { inner.inactive.insert ... }`
        // loop body in the rollback path.
    }
}

// ============================================================================
// RESET-ON-RELEASE TESTS
// ============================================================================
//
// `ImmutableBlock::set_evict_on_reset(true)` and the store-wide
// `with_default_reset_on_release(true)` cause the last drop of a primary
// to bypass the inactive pool and return the slot straight to the
// reset/free list (matching `release_duplicate`).

mod reset_on_release_tests {
    use super::*;
    use crate::testing::create_test_manager_with_default_reset_on_release;

    /// Per-block opt-in: a flagged block goes straight to reset on its
    /// last drop. The inactive pool stays empty and the seq_hash is no
    /// longer registered or matchable.
    #[test]
    fn per_block_flag_bypasses_inactive_pool() {
        let manager = create_test_manager(4);
        let m = manager.metrics();
        let token = create_test_token_block_from_iota(50_000);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let immutable = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();

        immutable.set_evict_on_reset(true);

        let store = manager.store_for_test();
        assert_eq!(store.inactive_len(), 0);
        assert_eq!(store.reset_len(), 3);

        drop(immutable);

        assert_eq!(store.inactive_len(), 0, "inactive pool stays empty");
        assert_eq!(store.reset_len(), 4, "slot returned to free list");
        assert!(
            !store.has_inactive(hash),
            "hash not present in inactive index"
        );
        assert!(
            !manager.block_registry().is_registered(hash),
            "registry handle marked absent"
        );
        assert!(
            manager.match_blocks(&[hash]).is_empty(),
            "block cannot be matched after reset"
        );

        let snap = m.snapshot();
        assert_eq!(snap.inflight_immutable, 0);
        assert_eq!(snap.inactive_pool_size, 0);
        assert_eq!(snap.reset_pool_size, 4);
    }

    /// Flag does not prevent in-flight sharing. While clones exist, the
    /// block is matchable. Only the *last* drop honors the flag.
    #[test]
    fn flag_does_not_prevent_inflight_sharing() {
        let manager = create_test_manager(4);
        let token = create_test_token_block_from_iota(50_001);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let a = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        let b = a.clone();
        a.set_evict_on_reset(true);

        // Drop only `a`. `b` is still alive; slot stays Primary.
        drop(a);
        let store = manager.store_for_test();
        assert_eq!(store.inactive_len(), 0);
        assert_eq!(store.reset_len(), 3);

        // Lookup must still succeed — flag has no effect while clones live.
        let matched = manager.match_blocks(&[hash]);
        assert_eq!(matched.len(), 1);
        drop(matched);
        drop(b);

        // Now the last clone is gone — flag fires, slot resets.
        assert_eq!(store.inactive_len(), 0);
        assert_eq!(store.reset_len(), 4);
        assert!(!manager.block_registry().is_registered(hash));
    }

    /// The flag is shared across clones via the Arc-backed AtomicBool.
    /// Last setter wins.
    #[test]
    fn last_writer_wins_across_clones() {
        let manager = create_test_manager(4);
        let token = create_test_token_block_from_iota(50_002);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let a = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        let b = a.clone();

        a.set_evict_on_reset(true);
        b.set_evict_on_reset(false); // overrides — clones share one atomic

        drop(a);
        drop(b);

        let store = manager.store_for_test();
        // Last writer was `false`, so the block went to the inactive
        // pool, not the reset pool.
        assert_eq!(store.inactive_len(), 1);
        assert_eq!(store.reset_len(), 3);
        assert!(store.has_inactive(hash));
    }

    /// Resetting one block's slot does NOT poison a future registration
    /// of the same `BlockId`. The next holder starts with the store's
    /// default (`false` here).
    #[test]
    fn no_poisoning_across_registrations() {
        let manager = create_test_manager(1);
        let store = manager.store_for_test();

        // First registration: flag = true, reset on drop.
        let token1 = create_test_token_block_from_iota(50_003);
        let hash1 = token1.kvbm_sequence_hash();
        {
            let mutables = manager.allocate_blocks(1).unwrap();
            let complete = mutables
                .into_iter()
                .next()
                .unwrap()
                .complete(&token1)
                .unwrap();
            let imm = manager
                .register_blocks(vec![complete])
                .into_iter()
                .next()
                .unwrap();
            imm.set_evict_on_reset(true);
            drop(imm);
        }
        assert_eq!(store.reset_len(), 1, "first registration reset");
        assert!(!manager.block_registry().is_registered(hash1));

        // Second registration: same BlockId (only one available),
        // different hash. Must NOT inherit the previous flag.
        let token2 = create_test_token_block_from_iota(50_004);
        let hash2 = token2.kvbm_sequence_hash();
        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token2)
            .unwrap();
        let imm = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        drop(imm);

        // If poisoning happened, this would reset; with the default
        // (false), the block should land in the inactive pool.
        assert_eq!(
            store.inactive_len(),
            1,
            "second block went to inactive, flag NOT inherited"
        );
        assert_eq!(store.reset_len(), 0);
        assert!(store.has_inactive(hash2));
    }

    /// Store-wide default: every primary release bypasses inactive.
    #[test]
    fn store_wide_default_resets_on_release() {
        let manager = create_test_manager_with_default_reset_on_release::<TestBlockData>(4, true);
        let token = create_test_token_block_from_iota(50_005);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let imm = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        // No explicit set_evict_on_reset call — relying on the store default.
        drop(imm);

        let store = manager.store_for_test();
        assert_eq!(store.inactive_len(), 0);
        assert_eq!(store.reset_len(), 4);
        assert!(!manager.block_registry().is_registered(hash));
    }

    /// Sticky override: with store default = true, a holder opts out
    /// via `set_evict_on_reset(false)`. After Drop the block lands in
    /// the inactive pool (correct). A later `match_blocks` resurrects
    /// it; the resurrected `ImmutableBlock` inherits the *stored*
    /// override, NOT the store default. Dropping the resurrected clone
    /// must keep the block in the inactive pool, not reset it.
    ///
    /// Regression test for the Codex review finding:
    /// "Preserve per-block opt-out across inactive hits".
    #[test]
    fn opt_out_survives_inactive_resurrection() {
        let manager = create_test_manager_with_default_reset_on_release::<TestBlockData>(4, true);
        let token = create_test_token_block_from_iota(50_009);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let imm = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        imm.set_evict_on_reset(false); // opt out of store-wide default
        drop(imm);

        let store = manager.store_for_test();
        assert_eq!(store.inactive_len(), 1, "opt-out kept block in inactive");
        assert_eq!(store.reset_len(), 3);

        // Resurrect via the inactive cache.
        let resurrected = manager.match_blocks(&[hash]);
        assert_eq!(resurrected.len(), 1);
        assert_eq!(store.inactive_len(), 0, "resurrection drained inactive");

        // Drop the resurrected clone. The override stored in
        // SlotState::Inactive must have travelled into the new Inner —
        // so this drop also keeps the block in the inactive pool
        // instead of resetting per the store default.
        drop(resurrected);

        assert_eq!(
            store.inactive_len(),
            1,
            "override survived resurrection — block stayed in inactive"
        );
        assert_eq!(store.reset_len(), 3);
        assert!(store.has_inactive(hash));
        assert!(manager.block_registry().is_registered(hash));
    }

    /// Holder can opt out of the store-wide default for a specific block.
    #[test]
    fn per_block_can_opt_out_of_store_default() {
        let manager = create_test_manager_with_default_reset_on_release::<TestBlockData>(4, true);
        let token = create_test_token_block_from_iota(50_006);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let imm = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        imm.set_evict_on_reset(false); // overrides store default
        drop(imm);

        let store = manager.store_for_test();
        assert_eq!(store.inactive_len(), 1, "kept in inactive despite default");
        assert_eq!(store.reset_len(), 3);
        assert!(store.has_inactive(hash));
    }

    /// Eager-resurrection path (concurrent lookup observes Arc strong=0
    /// before Drop's `release_primary` fires) preserves the per-block
    /// override. The override lives in the store-owned per-slot atomic
    /// (not on the dropping Inner), so the eager `Primary → Inactive`
    /// transition rides over the race window without losing the flag.
    /// The resurrected block reads the same atomic on its own drop.
    ///
    /// Mirrors `eager_primary_to_inactive_is_deterministic_with_pause_hook`
    /// in `race_regression_tests`. Regression test for the review finding
    /// at https://github.com/ai-dynamo/dynamo/pull/9504#discussion_r3238515930.
    #[test]
    fn eager_resurrection_preserves_flag() {
        use std::sync::Arc;
        use std::thread;

        let manager = Arc::new(create_test_manager(4));
        let token = create_test_token_block_from_iota(50_007);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let immutable = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        immutable.set_evict_on_reset(true);

        let store = manager.store_for_test().clone();

        let gate = store.pause_release_primary();
        let arrivals_before = store.release_primary_arrivals();
        let drop_t = thread::spawn(move || drop(immutable));

        while store.release_primary_arrivals() == arrivals_before {
            std::thread::yield_now();
        }

        // Concurrent lookup drives the eager `Primary → Inactive`
        // transition. The dropping Inner had flag=true; the eager path
        // leaves the per-slot atomic untouched, so the override rides
        // over the race window.
        let matched = manager.match_blocks(&[hash]);
        assert_eq!(matched.len(), 1, "eager resurrection must succeed");

        let snap_mid = manager.metrics().snapshot();
        assert_eq!(
            snap_mid.eager_primary_to_inactive_total, 1,
            "eager transition fired"
        );

        drop(gate);
        drop_t.join().unwrap();

        // The parked drop saw a slot that no longer matched its self_ptr
        // and no-opped — its destination decision was preempted by the
        // eager path.
        let snap_after = manager.metrics().snapshot();
        assert_eq!(snap_after.release_primary_noop_total, 1);

        // The resurrected Inner reads the per-slot atomic on its own
        // drop. Because the original holder had set the override to
        // `true`, this drop must bypass the inactive pool and reset the
        // slot — proving the override survived the race window.
        drop(matched);

        let store = manager.store_for_test();
        assert_eq!(
            store.inactive_len(),
            0,
            "override preserved — block bypasses inactive on drop",
        );
        assert_eq!(store.reset_len(), 4);
        assert!(
            !manager.block_registry().is_registered(hash),
            "registry handle marked absent",
        );
    }

    /// Duplicate blocks always reset on drop regardless of the flag —
    /// this is the pre-existing `release_duplicate` behavior and the
    /// shared helper. The store-wide default does not change it.
    #[test]
    fn duplicate_release_still_resets_with_default_false() {
        // Use Allow policy so duplicates are created.
        let registry = crate::registry::BlockRegistry::builder()
            .frequency_tracker(
                crate::manager::FrequencyTrackingCapacity::default().create_tracker(),
            )
            .build();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(4)
            .block_size(4)
            .registry(registry)
            .with_lru_backend()
            .duplication_policy(BlockDuplicationPolicy::Allow)
            .build()
            .expect("Should build manager");

        let token = create_test_token_block_from_iota(50_008);
        let _hash = token.kvbm_sequence_hash();

        // Primary
        let m1 = manager.allocate_blocks(1).unwrap();
        let c1 = m1.into_iter().next().unwrap().complete(&token).unwrap();
        let primary = manager
            .register_blocks(vec![c1])
            .into_iter()
            .next()
            .unwrap();

        // Duplicate (same hash on a different BlockId)
        let m2 = manager.allocate_blocks(1).unwrap();
        let c2 = m2.into_iter().next().unwrap().complete(&token).unwrap();
        let dup = manager
            .register_blocks(vec![c2])
            .into_iter()
            .next()
            .unwrap();
        assert_eq!(dup.block_id(), 1);
        assert_ne!(primary.block_id(), dup.block_id());

        let store = manager.store_for_test();
        let reset_before = store.reset_len();
        let inactive_before = store.inactive_len();

        // Drop duplicate first — always resets, never enters inactive.
        drop(dup);
        assert_eq!(
            store.reset_len(),
            reset_before + 1,
            "duplicate drop returned to reset pool"
        );
        assert_eq!(
            store.inactive_len(),
            inactive_before,
            "duplicate never enters inactive"
        );

        // Primary drop (flag = false default) → goes to inactive normally.
        drop(primary);
        assert_eq!(store.inactive_len(), inactive_before + 1);
    }

    /// Override survives two full `Inactive → Primary → Inactive`
    /// hops. Extends `opt_out_survives_inactive_resurrection` (one hop)
    /// to guarantee the carry path in `acquire_for_hash` /
    /// `match_blocks` is idempotent across repeated cache hits.
    #[test]
    fn override_survives_multiple_resurrection_cycles() {
        let manager = create_test_manager_with_default_reset_on_release::<TestBlockData>(4, true);
        let token = create_test_token_block_from_iota(50_010);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let imm = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        imm.set_evict_on_reset(false); // opt out of store-wide default
        drop(imm);

        let store = manager.store_for_test();
        assert_eq!(store.inactive_len(), 1, "first drop landed in inactive");
        assert_eq!(store.reset_len(), 3);

        // First resurrection
        let r1 = manager.match_blocks(&[hash]);
        assert_eq!(r1.len(), 1);
        assert_eq!(store.inactive_len(), 0);
        drop(r1);
        assert_eq!(
            store.inactive_len(),
            1,
            "override survived first resurrection"
        );
        assert_eq!(store.reset_len(), 3);

        // Second resurrection — same path, must still honor the override.
        let r2 = manager.match_blocks(&[hash]);
        assert_eq!(r2.len(), 1);
        assert_eq!(store.inactive_len(), 0);
        drop(r2);
        assert_eq!(
            store.inactive_len(),
            1,
            "override survived second resurrection"
        );
        assert_eq!(store.reset_len(), 3);
        assert!(store.has_inactive(hash));
        assert!(manager.block_registry().is_registered(hash));
    }

    /// Calling `set_evict_on_reset` on a resurrected `ImmutableBlock`
    /// must take effect — the resurrected Inner's atomic is the
    /// authoritative one. Resurrect with an inherited `false`, then
    /// flip to `true` and drop: slot must reset, not stay in inactive.
    ///
    /// Guards against a regression where `new_primary_resurrected` is
    /// decoupled from the setter (e.g. wrong Arc, wrong field).
    #[test]
    fn set_evict_on_reset_on_resurrected_inner_takes_effect() {
        let manager = create_test_manager_with_default_reset_on_release::<TestBlockData>(4, true);
        let token = create_test_token_block_from_iota(50_011);
        let hash = token.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(1).unwrap();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let imm = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        imm.set_evict_on_reset(false);
        drop(imm);

        let store = manager.store_for_test();
        assert_eq!(store.inactive_len(), 1);

        // Resurrect: inherited override is `false`.
        let resurrected = manager.match_blocks(&[hash]);
        assert_eq!(resurrected.len(), 1);

        // Flip on the *resurrected* Inner. If the setter wires through
        // correctly, the next drop honors `true` and resets.
        resurrected[0].set_evict_on_reset(true);
        drop(resurrected);

        assert_eq!(
            store.inactive_len(),
            0,
            "resurrected-Inner setter took effect"
        );
        assert_eq!(store.reset_len(), 4, "slot returned to free list");
        assert!(!manager.block_registry().is_registered(hash));
    }

    /// Three blocks registered in a single `register_blocks` call must
    /// honor independent per-block flags. Guards against any future
    /// refactor that shares state across the registration loop.
    #[test]
    fn mixed_flags_in_single_register_blocks_call() {
        let manager = create_test_manager(4);
        let t0 = create_test_token_block_from_iota(50_020);
        let t1 = create_test_token_block_from_iota(50_021);
        let t2 = create_test_token_block_from_iota(50_022);
        let h0 = t0.kvbm_sequence_hash();
        let h1 = t1.kvbm_sequence_hash();
        let h2 = t2.kvbm_sequence_hash();

        let mutables = manager.allocate_blocks(3).unwrap();
        let mut iter = mutables.into_iter();
        let c0 = iter.next().unwrap().complete(&t0).unwrap();
        let c1 = iter.next().unwrap().complete(&t1).unwrap();
        let c2 = iter.next().unwrap().complete(&t2).unwrap();

        let immutables = manager.register_blocks(vec![c0, c1, c2]);
        assert_eq!(immutables.len(), 3);
        immutables[0].set_evict_on_reset(true);
        // immutables[1] left at default (false)
        immutables[2].set_evict_on_reset(true);

        let store = manager.store_for_test();
        assert_eq!(store.inactive_len(), 0);
        assert_eq!(store.reset_len(), 1, "1 of 4 slots still free");

        drop(immutables);

        assert_eq!(
            store.inactive_len(),
            1,
            "only the default-flag block kept in inactive"
        );
        assert_eq!(
            store.reset_len(),
            3,
            "two flagged blocks reset + one originally free"
        );
        assert!(!store.has_inactive(h0), "flagged block 0 not in inactive");
        assert!(store.has_inactive(h1), "default-flag block 1 in inactive");
        assert!(!store.has_inactive(h2), "flagged block 2 not in inactive");
        assert!(!manager.block_registry().is_registered(h0));
        assert!(manager.block_registry().is_registered(h1));
        assert!(!manager.block_registry().is_registered(h2));
    }

    /// The per-block override must be discarded when the slot leaves
    /// `Inactive` via eviction back to `Mutable`. A subsequent fresh
    /// registration on the same `BlockId` must read the store default,
    /// not inherit the previous holder's override.
    ///
    /// Documented invariant from `pools/store.rs`: "The flag is
    /// discarded when the slot leaves `Inactive` via eviction
    /// (`Inactive → Mutable`); a future fresh registration on the same
    /// `BlockId` reads the store default again."
    #[test]
    fn override_discarded_on_eviction_back_to_mutable() {
        // block_count=1 so the second allocate must evict from inactive.
        let manager = create_test_manager_with_default_reset_on_release::<TestBlockData>(1, true);
        let store = manager.store_for_test().clone();

        // First registration: opt OUT of the store default → Inactive(false).
        let token1 = create_test_token_block_from_iota(50_030);
        let hash1 = token1.kvbm_sequence_hash();
        {
            let mutables = manager.allocate_blocks(1).unwrap();
            let complete = mutables
                .into_iter()
                .next()
                .unwrap()
                .complete(&token1)
                .unwrap();
            let imm = manager
                .register_blocks(vec![complete])
                .into_iter()
                .next()
                .unwrap();
            imm.set_evict_on_reset(false);
            drop(imm);
        }
        assert_eq!(store.inactive_len(), 1, "first drop kept in inactive");
        assert_eq!(store.reset_len(), 0);

        // Force eviction: only slot is in inactive, allocate must evict.
        let mutables = manager.allocate_blocks(1).unwrap();
        assert_eq!(store.inactive_len(), 0, "inactive evicted to mutable");
        assert!(!manager.block_registry().is_registered(hash1));

        // Re-register on the same BlockId with a fresh hash. The fresh
        // Inner is constructed via `new_primary` (not resurrected) and
        // must read the store default (true) — NOT inherit the discarded
        // `false` override.
        let token2 = create_test_token_block_from_iota(50_031);
        let hash2 = token2.kvbm_sequence_hash();
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token2)
            .unwrap();
        let imm = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        // No explicit `set_evict_on_reset` — fresh Inner must read the
        // store default (true) from `BlockStore::default_reset_on_release`.
        drop(imm);

        assert_eq!(
            store.reset_len(),
            1,
            "fresh Inner used store default — slot reset, not inactive"
        );
        assert_eq!(store.inactive_len(), 0);
        assert!(!store.has_inactive(hash2));
        assert!(!manager.block_registry().is_registered(hash2));
    }

    /// Calling `set_evict_on_reset(true)` on a *duplicate* must not
    /// affect its drop (always resets via `release_duplicate`) and must
    /// not affect the *primary*'s atomic — they live on separate Inners.
    /// Guards against a refactor that unifies primary/duplicate Inner
    /// state and silently makes the duplicate's flag observable on the
    /// primary.
    #[test]
    fn duplicate_set_evict_on_reset_is_noop() {
        let registry = crate::registry::BlockRegistry::builder()
            .frequency_tracker(
                crate::manager::FrequencyTrackingCapacity::default().create_tracker(),
            )
            .build();
        let manager = BlockManager::<TestBlockData>::builder()
            .block_count(4)
            .block_size(4)
            .registry(registry)
            .with_lru_backend()
            .duplication_policy(BlockDuplicationPolicy::Allow)
            .build()
            .expect("Should build manager");

        let token = create_test_token_block_from_iota(50_040);
        let hash = token.kvbm_sequence_hash();

        // Primary
        let m1 = manager.allocate_blocks(1).unwrap();
        let c1 = m1.into_iter().next().unwrap().complete(&token).unwrap();
        let primary = manager
            .register_blocks(vec![c1])
            .into_iter()
            .next()
            .unwrap();

        // Duplicate
        let m2 = manager.allocate_blocks(1).unwrap();
        let c2 = m2.into_iter().next().unwrap().complete(&token).unwrap();
        let dup = manager
            .register_blocks(vec![c2])
            .into_iter()
            .next()
            .unwrap();
        assert_ne!(primary.block_id(), dup.block_id());

        // Flag on the duplicate — must be a no-op at drop.
        dup.set_evict_on_reset(true);

        let store = manager.store_for_test();
        let reset_before = store.reset_len();
        let inactive_before = store.inactive_len();

        drop(dup);
        assert_eq!(
            store.reset_len(),
            reset_before + 1,
            "duplicate drop resets regardless of flag"
        );
        assert_eq!(
            store.inactive_len(),
            inactive_before,
            "duplicate drop never enters inactive"
        );

        // Primary must be entirely unaffected by the duplicate's flag:
        // still matchable, its own atomic still at default (false), so
        // dropping it lands in inactive.
        let matched = manager.match_blocks(&[hash]);
        assert_eq!(matched.len(), 1, "primary still matchable after dup drop");
        assert_eq!(matched[0].block_id(), primary.block_id());
        drop(matched);
        drop(primary);

        assert_eq!(
            store.inactive_len(),
            inactive_before + 1,
            "primary drop honors its own flag (default false → inactive), \
             unaffected by duplicate's flag"
        );
    }

    /// Pool gauges (`inflight_immutable`, `inactive_pool_size`,
    /// `reset_pool_size`) must stay coherent across the full lifecycle
    /// when the reset-on-release flag fires. Regression-protects the
    /// metric plumbing in `release_primary`.
    #[test]
    fn metric_gauges_stay_coherent_under_reset_flag() {
        let manager = create_test_manager(4);
        let m = manager.metrics();
        let token = create_test_token_block_from_iota(50_050);

        // Baseline: 4 free slots, nothing in flight.
        let s = m.snapshot();
        assert_eq!(s.inflight_immutable, 0);
        assert_eq!(s.inactive_pool_size, 0);
        assert_eq!(s.reset_pool_size, 4);

        // Allocate 1 mutable → reset pool decrements; no immutable yet.
        let mutables = manager.allocate_blocks(1).unwrap();
        let s = m.snapshot();
        assert_eq!(s.inflight_immutable, 0, "no immutable until register");
        assert_eq!(s.inactive_pool_size, 0);
        assert_eq!(s.reset_pool_size, 3);

        // Complete + register → 1 in flight.
        let complete = mutables
            .into_iter()
            .next()
            .unwrap()
            .complete(&token)
            .unwrap();
        let imm = manager
            .register_blocks(vec![complete])
            .into_iter()
            .next()
            .unwrap();
        let s = m.snapshot();
        assert_eq!(s.inflight_immutable, 1);
        assert_eq!(s.inactive_pool_size, 0);
        assert_eq!(s.reset_pool_size, 3);

        // Flip the flag — pure state change, no gauge movement.
        imm.set_evict_on_reset(true);
        let s = m.snapshot();
        assert_eq!(s.inflight_immutable, 1);
        assert_eq!(s.inactive_pool_size, 0);
        assert_eq!(s.reset_pool_size, 3);

        // Drop with flag=true → bypasses inactive, slot returns to reset.
        drop(imm);
        let s = m.snapshot();
        assert_eq!(s.inflight_immutable, 0, "drop decremented immutable");
        assert_eq!(s.inactive_pool_size, 0, "inactive gauge unchanged");
        assert_eq!(
            s.reset_pool_size, 4,
            "reset gauge incremented, not inactive"
        );
    }
}
