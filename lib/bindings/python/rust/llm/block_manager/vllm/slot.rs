// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub enum SlotPosition {
    /// The current position in the sequence representing all tokens that have been computed.
    Computed,

    /// The number of tokens that were ini
    Prefill,

    /// If the compute position is less than the prefill position, this will be the Prefill position;
    /// otherwise, it will be the Computed position
    All,
}

pub struct Slot<S: Storage> {
    /// Current position in the sequence of tokens that have been computed.
    /// When the slot is initialized, we populate the sequence with the prefill tokens.
    /// However, those tokens are not yet prefilled, so they are not yet represented
    /// in the sequence_position.
    computed_position: usize,

    /// The number of tokens that were initially prefilled.
    prefill_position: usize,

    /// The sequence of token blocks
    sequence: TokenBlockSequence,

    /// The immutable blocks
    immutable: Vec<ImmutableBlock<S, BasicMetadata>>,

    /// The mutable blocks
    mutable: VecDeque<MutableBlock<S, BasicMetadata>>,
}

impl<S: Storage> Slot<S> {
    /// Creates a new slot.
    pub fn new(tokens: Tokens, block_size: usize, salt_hash: SaltHash) -> Self {
        let sequence = TokenBlockSequence::new(tokens, block_size, Some(salt_hash));
        let prefill_position = sequence.total_tokens();

        Self {
            computed_position: 0,
            prefill_position,
            sequence,
            immutable: Vec::new(),
            mutable: VecDeque::new(),
        }
    }

    /// Updates the sequence with the given tokens.
    /// These tokens will advance the computed sequence position.
    pub fn apply_computed_tokens(
        &mut self,
        tokens_to_append: Vec<u32>,
        block_pool: &BlockPool<S, BasicMetadata>,
    ) -> Result<(), SlotError> {
        if tokens_to_append.is_empty() {
            return Ok(());
        }

        // assert that the number of tokens to apply is less than the number of tokens that can be applied to the
        // current collection of mutable blocks
        debug_assert!(
            tokens_to_append.len()
                < self.computed_position % self.sequence.block_size() + self.mutable.len()
        );

        // if we are still prefilling, we don't extend the sequence, but verify the tokens match what is already present.
        if self.computed_position < self.prefill_position {
            tracing::debug!("applying {} prefill tokens", tokens_to_append.len());
            debug_assert_eq!(
                self.sequence
                    .tokens_at(
                        self.computed_position..self.computed_position + tokens_to_append.len()
                    )
                    .as_ref(),
                &tokens_to_append,
            );
            self.computed_position += tokens_to_append.len();
        } else {
            tracing::debug!("applying {} tokens", tokens_to_append.len());
            // if we are not prefilling, we extend the sequence and advance the sequence position.
            // first advance the sequence, then the position -- this covers the case where the extend fails.
            let count = tokens_to_append.len();
            self.sequence
                .extend(tokens_to_append.into())
                .map_err(|e| SlotError::from_str(&format!("failed to extend sequence: {:?}", e)))?;
            self.computed_position += count;
        }

        // determine if we need to register any blocks
        // if the number of blocks for the computed position is greater than the number of immutable blocks,
        // then we have to transition one or more of the mutable blocks to immutable.
        let num_blocks_to_register =
            (self.computed_position / self.sequence.block_size()) - self.immutable.len();
        debug_assert!(num_blocks_to_register <= self.mutable.len());

        if num_blocks_to_register == 0 {
            tracing::debug!("no blocks to register");
            return Ok(());
        }

        let mut blocks_to_register = Vec::new();
        tracing::debug!("registering {} blocks", num_blocks_to_register);

        // create an iterator over the mutable blocks zipped with the token blocks
        let zipped_blocks = self
            .mutable
            .drain(0..num_blocks_to_register)
            .zip(self.sequence.blocks().iter().skip(self.immutable.len()));

        // apply the token blocks to the mutable blocks
        for (mut mutable_block, token_block) in zipped_blocks {
            mutable_block
                .state_mut()
                .apply_token_block(token_block.clone())
                .map_err(|e| {
                    SlotError::from_str(&format!("failed to apply token block: {:?}", e))
                })?;

            blocks_to_register.push(mutable_block);
        }

        // register the mutable blocks and extend the slot's immutable blocks
        let immutable_blocks = block_pool
            .register_blocks_blocking(blocks_to_register)
            .map_err(|e| SlotError::from_str(&format!("failed to register blocks: {:?}", e)))?;

        self.immutable.extend(immutable_blocks);

        Ok(())
    }

    /// Apply computed/cached blocks to the slot.
    pub fn apply_computed_blocks(
        &mut self,
        computed_blocks: Vec<ImmutableBlock<S, BasicMetadata>>,
    ) -> Result<(), SlotError> {
        assert!(self.mutable.is_empty());

        // create an iterator over the mutable blocks zipped with the token blocks
        let zipped_blocks = self
            .sequence
            .blocks()
            .iter()
            .skip(self.immutable.len())
            .zip(computed_blocks.into_iter());

        // validate the sequence hashes of the incoming immutable computed blocks
        // against the sequence hashes of blocks in the sequence.
        for (sequence_block, computed_block) in zipped_blocks {
            if sequence_block.sequence_hash() != computed_block.sequence_hash() {
                return Err(SlotError::from_str("computed block sequence hash mismatch"));
            }
            self.computed_position += sequence_block.block_size();
            self.immutable.push(computed_block);
        }

        Ok(())
    }

    /// Allocates space for the given number of new tokens.
    ///
    /// Returns None if unable to allocate new blocks,
    /// otherwise returns the block ids of the new blocks.
    ///
    /// An empty vector is returned if no new blocks are required.
    pub fn allocate_blocks(
        &mut self,
        num_new_tokens: usize,
        block_pool: &BlockPool<S, BasicMetadata>,
    ) -> Option<Vec<BlockId>> {
        let total_num_blocks = div_ceil(
            self.computed_position + num_new_tokens,
            self.sequence.block_size(),
        );

        let num_new_blocks = total_num_blocks - (self.immutable.len() + self.mutable.len());

        if num_new_blocks == 0 {
            return Some(Vec::new());
        }

        let new_blocks = block_pool.allocate_blocks_blocking(num_new_blocks).ok();

        match new_blocks {
            Some(new_blocks) => {
                let block_ids = new_blocks.iter().map(|b| b.block_id()).collect();
                self.mutable.extend(new_blocks.into_iter());
                Some(block_ids)
            }
            None => None,
        }
    }

    /// Number of tokens in the requested position.
    pub fn num_tokens(&self, position: SlotPosition) -> usize {
        match position {
            SlotPosition::Computed => self.computed_position,
            SlotPosition::Prefill => self.prefill_position,
            SlotPosition::All => self.sequence.total_tokens(),
        }
    }

    /// Sequence hashes for the requested position.
    pub fn sequence_hashes(&self, position: SlotPosition) -> Vec<SequenceHash> {
        match position {
            SlotPosition::Computed => {
                debug_assert!(self.computed_position <= self.sequence.total_tokens());
                self.sequence.blocks()[0..self.computed_position]
                    .iter()
                    .map(|b| b.sequence_hash())
                    .collect()
            }
            SlotPosition::Prefill => {
                assert!(self.prefill_position <= self.sequence.total_tokens());
                self.sequence.blocks()[0..self.prefill_position]
                    .iter()
                    .map(|b| b.sequence_hash())
                    .collect()
            }
            SlotPosition::All => self
                .sequence
                .blocks()
                .iter()
                .map(|b| b.sequence_hash())
                .collect(),
        }
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use dynamo_llm::block_manager::{
//         block::{BasicMetadata, BlockExt, Blocks, ImmutableBlock, MutableBlock},
//         layout::{tests::setup_layout, FullyContiguous},
//         pool::BlockPool,
//         storage::tests::{NullDeviceAllocator, NullDeviceStorage},
//         Storage,
//     };
//     use dynamo_llm::tokens::{SaltHash, SequenceHash, Tokens};
//     use rstest::*;
//     use std::collections::VecDeque;

//     const BLOCK_SIZE: usize = 4;
//     const SALT_HASH: SaltHash = 12345;

//     // Test fixture providing a pre-configured block pool for testing
//     struct TestFixture {
//         pool: BlockPool<NullDeviceStorage, BasicMetadata>,
//         _runtime: tokio::runtime::Runtime,
//     }

//     impl TestFixture {
//         fn new() -> Self {
//             let layout = setup_layout(None).unwrap();
//             let blocks = Blocks::<_, BasicMetadata>::new(layout, 42, 0)
//                 .unwrap()
//                 .into_blocks()
//                 .unwrap();

//             let runtime = tokio::runtime::Runtime::new().unwrap();
//             let pool = BlockPool::builder()
//                 .blocks(blocks)
//                 .async_runtime(runtime.handle().clone())
//                 .build()
//                 .unwrap();

//             Self {
//                 pool,
//                 _runtime: runtime,
//             }
//         }
//     }

//     // Helper function to create a slot with a given token sequence
//     fn create_slot_with_tokens(tokens: Vec<u32>) -> Slot<NullDeviceStorage> {
//         let token_sequence = Tokens::from(tokens);
//         Slot::new(token_sequence, BLOCK_SIZE, SALT_HASH)
//     }

//     // Helper function to allocate blocks for a slot
//     fn allocate_blocks_for_slot(
//         slot: &mut Slot<NullDeviceStorage>,
//         num_tokens: usize,
//         pool: &BlockPool<NullDeviceStorage, BasicMetadata>,
//     ) -> Option<Vec<BlockId>> {
//         slot.allocate_blocks(num_tokens, pool)
//     }

//     // Test chunked prefill scenarios with parameterized test cases
//     #[rstest]
//     #[case::aligned_chunks(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]], true)]
//     #[case::unaligned_chunks(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![vec![1, 2], vec![3, 4, 5, 6], vec![7, 8]], true)]
//     #[case::single_token_chunks(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![vec![1], vec![2], vec![3], vec![4], vec![5], vec![6], vec![7], vec![8]], true)]
//     #[case::incorrect_tokens(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![vec![1, 2, 3, 9]], false)] // Should fail on incorrect token
//     #[case::oversized_chunk(vec![1, 2, 3, 4], vec![vec![1, 2, 3, 4, 5]], false)] // Should fail on too many tokens
//     fn test_chunked_prefill(
//         #[case] initial_tokens: Vec<u32>,
//         #[case] chunks: Vec<Vec<u32>>,
//         #[case] should_succeed: bool,
//     ) {
//         let fixture = TestFixture::new();
//         let mut slot = create_slot_with_tokens(initial_tokens.clone());

//         // Verify initial state
//         assert_eq!(slot.num_tokens(SlotPosition::Prefill), initial_tokens.len());
//         assert_eq!(slot.num_tokens(SlotPosition::Computed), 0);
//         assert_eq!(slot.num_tokens(SlotPosition::All), initial_tokens.len());

//         // Allocate blocks before applying tokens (required by the system)
//         let total_tokens_needed = initial_tokens.len();
//         let allocated_blocks =
//             allocate_blocks_for_slot(&mut slot, total_tokens_needed, &fixture.pool);
//         assert!(allocated_blocks.is_some(), "Failed to allocate blocks");

//         // Apply chunks sequentially
//         let mut total_computed = 0;
//         for (i, chunk) in chunks.iter().enumerate() {
//             let result = slot.apply_computed_tokens(chunk.clone(), &fixture.pool);

//             if should_succeed {
//                 assert!(result.is_ok(), "Chunk {} failed: {:?}", i, result.err());
//                 total_computed += chunk.len();
//                 assert_eq!(slot.num_tokens(SlotPosition::Computed), total_computed);

//                 // Check that we're still within prefill bounds or have completed prefill
//                 if total_computed <= initial_tokens.len() {
//                     assert!(slot.computed_position <= slot.prefill_position);
//                 }
//             } else {
//                 // For negative test cases, expect failure
//                 if result.is_err() {
//                     return; // Test passed - expected failure occurred
//                 }
//             }
//         }

//         if should_succeed {
//             // After successful chunked prefill, computed position should match prefill position
//             assert_eq!(slot.computed_position, slot.prefill_position);
//         }
//     }

//     // Test standard decode scenarios
//     #[rstest]
//     #[case::single_decode_token(vec![1, 2, 3, 4], 1)]
//     #[case::multiple_decode_tokens(vec![1, 2, 3, 4], 5)]
//     fn test_standard_decode(#[case] initial_tokens: Vec<u32>, #[case] num_decode_tokens: usize) {
//         let fixture = TestFixture::new();
//         let mut slot = create_slot_with_tokens(initial_tokens.clone());

//         // Complete prefill first
//         let allocated_blocks =
//             allocate_blocks_for_slot(&mut slot, initial_tokens.len(), &fixture.pool);
//         assert!(allocated_blocks.is_some());

//         let result = slot.apply_computed_tokens(initial_tokens.clone(), &fixture.pool);
//         assert!(result.is_ok(), "Prefill failed: {:?}", result.err());

//         // Now we're in decode mode - allocate space for one additional token at a time
//         for i in 0..num_decode_tokens {
//             let decode_token = 100 + i as u32; // Use distinct tokens for decode

//             // Allocate space for the new token
//             let allocated_blocks = allocate_blocks_for_slot(&mut slot, 1, &fixture.pool);
//             assert!(
//                 allocated_blocks.is_some(),
//                 "Failed to allocate block for decode token {}",
//                 i
//             );

//             // Apply the decode token
//             let result = slot.apply_computed_tokens(vec![decode_token], &fixture.pool);
//             assert!(
//                 result.is_ok(),
//                 "Decode token {} failed: {:?}",
//                 i,
//                 result.err()
//             );

//             // Verify state
//             let expected_total = initial_tokens.len() + i + 1;
//             assert_eq!(slot.num_tokens(SlotPosition::Computed), expected_total);
//             assert_eq!(slot.num_tokens(SlotPosition::All), expected_total);
//         }
//     }

//     // Test speculative decode scenarios
//     #[rstest]
//     #[case::speculate_2_tokens(vec![1, 2, 3, 4], 2, 2)] // Allocate 2, apply 2
//     #[case::speculate_3_tokens(vec![1, 2, 3, 4], 3, 3)] // Allocate 3, apply 3
//     #[case::partial_speculation(vec![1, 2, 3, 4], 4, 2)] // Allocate 4, apply only 2
//     #[case::over_allocation(vec![1, 2, 3, 4], 5, 3)] // Allocate 5, apply only 3
//     fn test_speculative_decode(
//         #[case] initial_tokens: Vec<u32>,
//         #[case] num_tokens_to_allocate: usize,
//         #[case] num_tokens_to_apply: usize,
//     ) {
//         let fixture = TestFixture::new();
//         let mut slot = create_slot_with_tokens(initial_tokens.clone());

//         // Complete prefill first
//         let allocated_blocks =
//             allocate_blocks_for_slot(&mut slot, initial_tokens.len(), &fixture.pool);
//         assert!(allocated_blocks.is_some());

//         let result = slot.apply_computed_tokens(initial_tokens.clone(), &fixture.pool);
//         assert!(result.is_ok(), "Prefill failed: {:?}", result.err());

//         // Allocate space for speculative tokens
//         let allocated_blocks =
//             allocate_blocks_for_slot(&mut slot, num_tokens_to_allocate, &fixture.pool);
//         assert!(
//             allocated_blocks.is_some(),
//             "Failed to allocate speculative blocks"
//         );

//         // Generate speculative tokens
//         let speculative_tokens: Vec<u32> = (200..200 + num_tokens_to_apply as u32).collect();

//         // Apply the speculative tokens
//         let result = slot.apply_computed_tokens(speculative_tokens, &fixture.pool);
//         assert!(
//             result.is_ok(),
//             "Speculative decode failed: {:?}",
//             result.err()
//         );

//         // Verify state
//         let expected_total = initial_tokens.len() + num_tokens_to_apply;
//         assert_eq!(slot.num_tokens(SlotPosition::Computed), expected_total);
//         assert_eq!(slot.num_tokens(SlotPosition::All), expected_total);
//     }

//     // Test edge cases
//     #[rstest]
//     #[case::empty_token_application(vec![1, 2, 3, 4], vec![])] // Apply empty token list
//     #[case::single_token_sequence(vec![42], vec![42])] // Single token prefill
//     fn test_edge_cases(#[case] initial_tokens: Vec<u32>, #[case] tokens_to_apply: Vec<u32>) {
//         let fixture = TestFixture::new();
//         let mut slot = create_slot_with_tokens(initial_tokens.clone());

//         if !initial_tokens.is_empty() {
//             let allocated_blocks =
//                 allocate_blocks_for_slot(&mut slot, initial_tokens.len(), &fixture.pool);
//             assert!(allocated_blocks.is_some());
//         }

//         let result = slot.apply_computed_tokens(tokens_to_apply, &fixture.pool);
//         assert!(result.is_ok(), "Edge case failed: {:?}", result.err());
//     }

//     // Test sequence hash generation at different positions
//     #[test]
//     fn test_sequence_hashes() {
//         let fixture = TestFixture::new();
//         let initial_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8]; // 2 blocks worth
//         let mut slot = create_slot_with_tokens(initial_tokens.clone());

//         // Allocate blocks
//         let allocated_blocks =
//             allocate_blocks_for_slot(&mut slot, initial_tokens.len(), &fixture.pool);
//         assert!(allocated_blocks.is_some());

//         // Complete prefill
//         let result = slot.apply_computed_tokens(initial_tokens.clone(), &fixture.pool);
//         assert!(result.is_ok());

//         // Test sequence hashes at different positions
//         let prefill_hashes = slot.sequence_hashes(SlotPosition::Prefill);
//         let computed_hashes = slot.sequence_hashes(SlotPosition::Computed);
//         let all_hashes = slot.sequence_hashes(SlotPosition::All);

//         // After completing prefill, all should be equal
//         assert_eq!(prefill_hashes, computed_hashes);
//         assert_eq!(computed_hashes, all_hashes);
//         assert_eq!(prefill_hashes.len(), 2); // 8 tokens / 4 tokens per block = 2 blocks

//         // Add a decode token and check hashes again
//         let allocated_blocks = allocate_blocks_for_slot(&mut slot, 1, &fixture.pool);
//         assert!(allocated_blocks.is_some());

//         let result = slot.apply_computed_tokens(vec![99], &fixture.pool);
//         assert!(result.is_ok());

//         let new_computed_hashes = slot.sequence_hashes(SlotPosition::Computed);
//         let new_all_hashes = slot.sequence_hashes(SlotPosition::All);

//         // Prefill hashes should remain the same
//         assert_eq!(slot.sequence_hashes(SlotPosition::Prefill), prefill_hashes);
//         // But computed and all should now include the new token
//         assert_eq!(new_computed_hashes, new_all_hashes);
//         assert_eq!(new_computed_hashes.len(), 3); // Now we have 3 blocks worth of tokens
//     }

//     // Test block allocation scenarios
//     #[rstest]
//     #[case::exact_block_boundary(8, 0)] // Exactly 2 blocks needed, no additional
//     #[case::partial_block(6, 2)] // 1.5 blocks needed, allocate 2 more tokens
//     #[case::multiple_blocks(4, 8)] // Start with 1 block, allocate 2 more blocks worth
//     fn test_block_allocation(#[case] initial_tokens: usize, #[case] additional_tokens: usize) {
//         let fixture = TestFixture::new();
//         let tokens: Vec<u32> = (1..=initial_tokens as u32).collect();
//         let mut slot = create_slot_with_tokens(tokens.clone());

//         // Initial allocation
//         if initial_tokens > 0 {
//             let allocated_blocks =
//                 allocate_blocks_for_slot(&mut slot, initial_tokens, &fixture.pool);
//             assert!(allocated_blocks.is_some());

//             let result = slot.apply_computed_tokens(tokens, &fixture.pool);
//             assert!(result.is_ok());
//         }

//         // Additional allocation
//         if additional_tokens > 0 {
//             let allocated_blocks =
//                 allocate_blocks_for_slot(&mut slot, additional_tokens, &fixture.pool);
//             assert!(allocated_blocks.is_some());

//             let additional_token_vec: Vec<u32> = (100..100 + additional_tokens as u32).collect();
//             let result = slot.apply_computed_tokens(additional_token_vec, &fixture.pool);
//             assert!(result.is_ok());
//         }

//         let total_tokens = initial_tokens + additional_tokens;
//         assert_eq!(slot.num_tokens(SlotPosition::All), total_tokens);
//         assert_eq!(slot.num_tokens(SlotPosition::Computed), total_tokens);
//     }

//     // Test failure scenarios
//     #[test]
//     fn test_prefill_token_mismatch() {
//         let fixture = TestFixture::new();
//         let initial_tokens = vec![1, 2, 3, 4];
//         let mut slot = create_slot_with_tokens(initial_tokens.clone());

//         let allocated_blocks =
//             allocate_blocks_for_slot(&mut slot, initial_tokens.len(), &fixture.pool);
//         assert!(allocated_blocks.is_some());

//         // Try to apply wrong tokens during prefill
//         let wrong_tokens = vec![1, 2, 3, 5]; // Last token is wrong
//         let result = slot.apply_computed_tokens(wrong_tokens, &fixture.pool);

//         // This should panic in debug mode due to debug_assert_eq! in the actual code
//         // In release mode, it would likely succeed but with incorrect behavior
//         // The test here depends on the implementation detail that debug_assert_eq! is used
//     }
// }
