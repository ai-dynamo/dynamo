// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Property-based tests for block state machine.
//!
//! These tests verify critical invariants of the Block<T, State> type-state pattern
//! using property-based testing with proptest. The focus is on the state transitions
//! and block size validation logic.

use super::{super::blocks::*, tests::*, *};

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::testing::config::{
        COMMON_TEST_BLOCK_SIZES, constants, generate_test_tokens, validate_test_block_size,
    };

    use dynamo_tokens::{TokenBlock, TokenBlockSequence};
    use proptest::prelude::*;

    /// Helper function to create a TokenBlock from a token sequence
    fn create_token_block_from_sequence(tokens: &[u32]) -> Option<TokenBlock> {
        if tokens.is_empty() {
            return None;
        }

        let sequence = TokenBlockSequence::from_slice(tokens, tokens.len() as u32, Some(42));

        // If we have a complete block, return it
        if let Some(block) = sequence.blocks().first() {
            return Some(block.clone());
        }

        // Otherwise try to commit the partial block
        let (_, mut partial) = sequence.into_parts();
        partial.commit().ok()
    }

    proptest! {
        /// Property: Block size validation is consistent across all operations
        ///
        /// This test verifies that:
        /// 1. Blocks can only be completed when token count matches block size
        /// 2. Block size is preserved across all state transitions
        /// 3. Error handling returns the original block to prevent leaks
        #[test]
        fn prop_block_size_validation_consistency(
            block_size in prop::sample::select(COMMON_TEST_BLOCK_SIZES),
            token_count in 1usize..=128usize,
            block_id in any::<BlockId>(),
        ) {
            prop_assume!(validate_test_block_size(block_size));
            prop_assume!(token_count <= 128); // Keep token sequences reasonable

            let block = Block::<TestData, Reset>::new(block_id, block_size);
            prop_assert_eq!(block.block_size(), block_size);
            prop_assert_eq!(block.block_id(), block_id);

            let tokens = generate_test_tokens(100, token_count);

            // Only test if we can create a valid token block
            if let Some(token_block) = create_token_block_from_sequence(&tokens) {
                let actual_token_size = token_block.block_size();
                let result = block.complete(&token_block);

                if actual_token_size == block_size {
                    // Should succeed when sizes match
                    let complete_block = result.expect("Should complete when sizes match");
                    prop_assert_eq!(complete_block.block_size(), block_size);
                    prop_assert_eq!(complete_block.block_id(), block_id);
                } else {
                    // Should fail when sizes don't match
                    prop_assert!(matches!(result, Err(BlockError::BlockSizeMismatch { .. })),
                        "Expected BlockSizeMismatch error when sizes don't match");

                    // Verify error contains the original block
                    if let Err(BlockError::BlockSizeMismatch { expected, actual, block: returned_block }) = result {
                        prop_assert_eq!(expected, block_size);
                        prop_assert_eq!(actual, actual_token_size);
                        prop_assert_eq!(returned_block.block_size(), block_size);
                        prop_assert_eq!(returned_block.block_id(), block_id);
                    }
                }
            }
        }

        /// Property: Complete state transitions preserve all block properties
        ///
        /// This test verifies that going from Reset → Complete → Registered → Reset
        /// preserves the block ID and block size throughout all transitions.
        #[test]
        fn prop_state_transitions_preserve_properties(
            block_size in prop::sample::select(&[1usize, 4, 16, 64]),
            block_id in any::<BlockId>(),
            base_token in 0u32..1000u32,
        ) {
            prop_assume!(validate_test_block_size(block_size));

            // Start with Reset block
            let reset_block = Block::<TestData, Reset>::new(block_id, block_size);
            prop_assert_eq!(reset_block.block_size(), block_size);
            prop_assert_eq!(reset_block.block_id(), block_id);

            // Generate matching tokens and create token block
            let tokens = generate_test_tokens(base_token, block_size);
            let token_block = create_token_block_from_sequence(&tokens)
                .expect("Should be able to create token block");

            // Transition to Complete
            let complete_block = reset_block
                .complete(&token_block)
                .expect("Should complete with matching size");
            prop_assert_eq!(complete_block.block_size(), block_size);
            prop_assert_eq!(complete_block.block_id(), block_id);

            // Transition to Registered
            let registry = BlockRegistry::new();
            let seq_hash = complete_block.sequence_hash();
            let handle = registry.register_sequence_hash(seq_hash);
            let registered_block = complete_block.register_with_handle(handle);
            prop_assert_eq!(registered_block.block_size(), block_size);
            prop_assert_eq!(registered_block.block_id(), block_id);
            prop_assert_eq!(registered_block.sequence_hash(), seq_hash);

            // Transition back to Reset
            let reset_again = registered_block.reset();
            prop_assert_eq!(reset_again.block_size(), block_size);
            prop_assert_eq!(reset_again.block_id(), block_id);
        }

        /// Property: Block IDs are preserved but can be arbitrary values
        ///
        /// This test verifies that block IDs are not constrained by the block manager
        /// and are preserved through all operations.
        #[test]
        fn prop_block_id_preservation(
            block_id in any::<BlockId>(),
            block_size in prop::sample::select(&[constants::SMALL, constants::MEDIUM]),
        ) {
            let block = Block::<TestData, Reset>::new(block_id, block_size);
            prop_assert_eq!(block.block_id(), block_id);

            // Test with edge case block IDs
            let edge_cases = [0, 1, BlockId::MAX / 2, BlockId::MAX - 1, BlockId::MAX];
            for &test_id in &edge_cases {
                let edge_block = Block::<TestData, Reset>::new(test_id, block_size);
                prop_assert_eq!(edge_block.block_id(), test_id);
            }
        }

        // /// Property: Error handling prevents resource leaks
        // ///
        // /// This test verifies that when block completion fails, the original block
        // /// is returned in the error, preventing resource leaks.
        // #[test]
        // fn prop_error_handling_prevents_leaks(
        //     block_size in prop::sample::select(&[4usize, 8, 16]),
        //     wrong_token_count in prop::sample::select(&[1usize, 2, 3, 7, 9, 15, 17, 32]),
        //     block_id in 0u64..1000u64,
        // ) {
        //     prop_assume!(validate_test_block_size(block_size));
        //     prop_assume!(wrong_token_count != block_size); // Ensure mismatch
        //     prop_assume!(wrong_token_count <= 32); // Keep reasonable

        //     let original_block = Block::<TestData, Reset>::new(block_id, block_size);

        //     let tokens = generate_test_tokens(500, wrong_token_count);
        //     if let Some(token_block) = create_token_block_from_sequence(&tokens) {
        //         let result = original_block.complete(token_block);

        //         // Should always fail due to size mismatch
        //         prop_assert!(result.is_err());

        //         if let Err(BlockError::BlockSizeMismatch { expected, actual, block: returned_block }) = result {
        //             // Verify error details
        //             prop_assert_eq!(expected, block_size);
        //             prop_assert_ne!(actual, block_size); // Should be different

        //             // Verify returned block is equivalent to original
        //             prop_assert_eq!(returned_block.block_id(), block_id);
        //             prop_assert_eq!(returned_block.block_size(), block_size);

        //             // We can still use the returned block
        //             let correct_tokens = generate_test_tokens(600, block_size);
        //             if let Some(correct_token_block) = create_token_block_from_sequence(&correct_tokens) {
        //                 let success_result = returned_block.complete(correct_token_block);
        //                 prop_assert!(success_result.is_ok());
        //             }
        //         }
        //     }
        // }

        /// Property: Block size constraints are enforced at construction
        ///
        /// This test verifies that all valid block sizes work correctly,
        /// providing coverage across the full range of allowed values.
        #[test]
        fn prop_valid_block_sizes_work(
            block_size in prop::sample::select(&[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
            block_id in any::<BlockId>(),
        ) {
            // All these sizes should be valid
            prop_assert!(validate_test_block_size(block_size));

            // Should be able to create blocks with any valid size
            let block = Block::<TestData, Reset>::new(block_id, block_size);
            prop_assert_eq!(block.block_size(), block_size);

            // Should be able to complete with matching token count
            let tokens = generate_test_tokens(0, block_size);
            if let Some(token_block) = create_token_block_from_sequence(&tokens) {
                let result = block.complete(&token_block);
                prop_assert!(result.is_ok(), "Failed to complete block with size {}", block_size);
            }
        }
    }

    /// Additional focused property tests for specific scenarios
    mod focused_properties {
        use super::*;

        proptest! {
            /// Property: Sequence hash is deterministic for identical token sequences
            #[test]
            fn prop_sequence_hash_deterministic(
                tokens in prop::collection::vec(any::<u32>(), 4..=4), // Always 4 tokens
                block_id1 in any::<BlockId>(),
                block_id2 in any::<BlockId>(),
            ) {
                prop_assume!(block_id1 != block_id2); // Different block IDs

                if let Some(token_block1) = create_token_block_from_sequence(&tokens)
                    && let Some(token_block2) = create_token_block_from_sequence(&tokens) {
                        // Same tokens should produce same sequence hash regardless of block ID
                        let block1 = Block::<TestData, Reset>::new(block_id1, 4);
                        let block2 = Block::<TestData, Reset>::new(block_id2, 4);

                        let complete1 = block1.complete(&token_block1).expect("Should complete");
                        let complete2 = block2.complete(&token_block2).expect("Should complete");

                        prop_assert_eq!(complete1.sequence_hash(), complete2.sequence_hash());
                    }
            }

            /// Property: Different token sequences produce different hashes (with high probability)
            #[test]
            fn prop_different_tokens_different_hashes(
                tokens1 in prop::collection::vec(0u32..100u32, 4..=4),
                tokens2 in prop::collection::vec(100u32..200u32, 4..=4), // Different range
            ) {
                if let (Some(token_block1), Some(token_block2)) = (
                    create_token_block_from_sequence(&tokens1),
                    create_token_block_from_sequence(&tokens2)
                ) {
                    let block1 = Block::<TestData, Reset>::new(1, 4);
                    let block2 = Block::<TestData, Reset>::new(2, 4);

                    let complete1 = block1.complete(&token_block1).expect("Should complete");
                    let complete2 = block2.complete(&token_block2).expect("Should complete");

                    // Different token sequences should produce different hashes
                    // (This is probabilistic but very likely with our token ranges)
                    if tokens1 != tokens2 {
                        prop_assert_ne!(complete1.sequence_hash(), complete2.sequence_hash());
                    }
                }
            }
        }
    }
}
