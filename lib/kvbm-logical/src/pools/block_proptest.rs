// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Property-based tests for guard state transitions through the unified store.

use super::tests::*;
use crate::blocks::*;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::testing::config::{
        COMMON_TEST_BLOCK_SIZES, generate_test_tokens, validate_test_block_size,
    };
    use crate::testing::{TestPoolSetupBuilder, create_test_token_block};

    use proptest::prelude::*;

    proptest! {
        /// Property: complete() on a token block of mismatched size returns
        /// `BlockSizeMismatch` with the original MutableBlock recoverable.
        #[test]
        fn prop_complete_mismatch_returns_block(
            block_size in prop::sample::select(COMMON_TEST_BLOCK_SIZES),
            wrong_size in prop::sample::select(&[1usize, 2, 8, 16, 32]),
        ) {
            prop_assume!(validate_test_block_size(block_size));
            prop_assume!(wrong_size != block_size);
            prop_assume!(validate_test_block_size(wrong_size));

            let store = TestPoolSetupBuilder::default()
                .block_count(2)
                .block_size(block_size)
                .build()
                .unwrap()
                .build_store::<TestData>();

            let mut blocks = store.allocate_reset_blocks(1);
            let mutable = blocks.pop().unwrap();
            let original_id = mutable.block_id();

            let tokens = generate_test_tokens(100, wrong_size);
            let tb = create_test_token_block(&tokens, wrong_size as u32);
            let result = mutable.complete(&tb);

            // Exhaustive match: if `BlockError` grows a new variant, this
            // fails to compile rather than silently passing the property.
            match result {
                Ok(_) => prop_assert!(false, "expected BlockSizeMismatch, got Ok"),
                Err(BlockError::BlockSizeMismatch { expected, actual, block: recovered }) => {
                    prop_assert_eq!(expected, block_size);
                    prop_assert_eq!(actual, wrong_size);
                    let recovered: MutableBlock<TestData> = recovered;
                    prop_assert_eq!(recovered.block_id(), original_id);
                    prop_assert_eq!(recovered.block_size(), block_size);
                }
            }
        }

        /// Property: block_id and block_size are preserved through
        /// Reset → Staged → Reset round-trips.
        #[test]
        fn prop_state_transitions_preserve_properties(
            block_size in prop::sample::select(&[1usize, 4, 16, 64]),
            base_token in 0u32..1000u32,
        ) {
            prop_assume!(validate_test_block_size(block_size));

            let store = TestPoolSetupBuilder::default()
                .block_count(2)
                .block_size(block_size)
                .build()
                .unwrap()
                .build_store::<TestData>();

            let mut blocks = store.allocate_reset_blocks(1);
            let mutable = blocks.pop().unwrap();
            let id = mutable.block_id();
            prop_assert_eq!(mutable.block_size(), block_size);

            let tokens = generate_test_tokens(base_token, block_size);
            let tb = create_test_token_block(&tokens, block_size as u32);
            let complete = mutable.complete(&tb).expect("complete should succeed");
            prop_assert_eq!(complete.block_id(), id);
            prop_assert_eq!(complete.block_size(), block_size);
            let seq_hash = complete.sequence_hash();

            let reset_again = complete.reset();
            prop_assert_eq!(reset_again.block_id(), id);
            prop_assert_eq!(reset_again.block_size(), block_size);

            // Re-stage and ensure the same hash recomputes deterministically.
            let tb2 = create_test_token_block(
                &generate_test_tokens(base_token, block_size),
                block_size as u32,
            );
            let staged = reset_again.complete(&tb2).expect("complete should succeed");
            prop_assert_eq!(staged.sequence_hash(), seq_hash);
        }

        /// Property: every common block size yields a usable store and a
        /// complete-able guard.
        #[test]
        fn prop_valid_block_sizes_work(
            block_size in prop::sample::select(&[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        ) {
            prop_assert!(validate_test_block_size(block_size));

            let store = TestPoolSetupBuilder::default()
                .block_count(1)
                .block_size(block_size)
                .build()
                .unwrap()
                .build_store::<TestData>();

            let mut blocks = store.allocate_reset_blocks(1);
            let mutable = blocks.pop().unwrap();
            prop_assert_eq!(mutable.block_size(), block_size);

            let tokens = generate_test_tokens(0, block_size);
            let tb = create_test_token_block(&tokens, block_size as u32);
            prop_assert!(mutable.complete(&tb).is_ok());
        }
    }

    mod focused_properties {
        use super::*;

        proptest! {
            /// Property: identical token sequences hash identically regardless
            /// of which slot they're staged through.
            #[test]
            fn prop_sequence_hash_deterministic(
                tokens in prop::collection::vec(any::<u32>(), 4..=4),
            ) {
                let store = TestPoolSetupBuilder::default()
                    .block_count(2)
                    .block_size(4)
                    .build()
                    .unwrap()
                    .build_store::<TestData>();

                let mut allocated = store.allocate_reset_blocks(2);
                let m2 = allocated.pop().unwrap();
                let m1 = allocated.pop().unwrap();

                let tb1 = create_test_token_block(&tokens, 4);
                let tb2 = create_test_token_block(&tokens, 4);

                let c1 = m1.complete(&tb1).expect("complete");
                let c2 = m2.complete(&tb2).expect("complete");

                prop_assert_eq!(c1.sequence_hash(), c2.sequence_hash());
            }

            /// Property: distinct token sequences produce distinct hashes.
            #[test]
            fn prop_different_tokens_different_hashes(
                tokens1 in prop::collection::vec(0u32..100u32, 4..=4),
                tokens2 in prop::collection::vec(100u32..200u32, 4..=4),
            ) {
                prop_assume!(tokens1 != tokens2);

                let store = TestPoolSetupBuilder::default()
                    .block_count(2)
                    .block_size(4)
                    .build()
                    .unwrap()
                    .build_store::<TestData>();

                let mut allocated = store.allocate_reset_blocks(2);
                let m2 = allocated.pop().unwrap();
                let m1 = allocated.pop().unwrap();

                let tb1 = create_test_token_block(&tokens1, 4);
                let tb2 = create_test_token_block(&tokens2, 4);

                let c1 = m1.complete(&tb1).expect("complete");
                let c2 = m2.complete(&tb2).expect("complete");

                prop_assert_ne!(c1.sequence_hash(), c2.sequence_hash());
            }
        }
    }
}
