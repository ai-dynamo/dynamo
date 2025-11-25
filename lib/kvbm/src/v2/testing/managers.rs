// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! BlockManager testing utilities.

use anyhow::Result;

use crate::{
    logical::{
        blocks::{BlockMetadata, BlockRegistry, ImmutableBlock},
        manager::{BlockManager, FrequencyTrackingCapacity},
    },
    v2::logical::pools::SequenceHash,
};

use super::token_blocks;

/// Create a basic BlockManager for testing with standard configuration.
///
/// Uses:
/// - LRU backend
/// - Medium frequency tracking
/// - BlockDuplicationPolicy::Allow
///
/// # Arguments
/// * `block_count` - Number of blocks in the pool
/// * `block_size` - Tokens per block (must be power of 2, 1-1024)
///
/// # Example
/// ```ignore
/// let manager = create_test_manager::<G2>(100, 16);
/// let blocks = manager.allocate_blocks(10).unwrap();
/// ```
pub fn create_test_manager<T: BlockMetadata>(
    block_count: usize,
    block_size: usize,
) -> BlockManager<T> {
    let registry =
        BlockRegistry::with_frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker());

    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(block_size)
        .registry(registry)
        .with_lru_backend()
        .build()
        .expect("Should build test manager")
}

/// Populate a BlockManager with token blocks and return their sequence hashes.
///
/// This function:
/// 1. Allocates blocks from the manager
/// 2. Completes them with provided token blocks
/// 3. Registers them
/// 4. Drops the immutable blocks (returns to inactive pool)
///
/// # Returns
/// Vec of sequence hashes for the registered blocks (in order)
///
/// # Example
/// ```ignore
/// let manager = create_test_manager::<G2>(100, 4);
/// let token_seq = token_blocks::create_token_sequence(32, 4, 0);
/// let seq_hashes = populate_manager_with_blocks(&manager, token_seq.blocks())?;
/// assert_eq!(seq_hashes.len(), 32);
/// ```
pub fn populate_manager_with_blocks<T: BlockMetadata>(
    manager: &BlockManager<T>,
    token_blocks: &[dynamo_tokens::TokenBlock],
) -> Result<Vec<SequenceHash>> {
    let blocks = manager
        .allocate_blocks(token_blocks.len())
        .ok_or_else(|| anyhow::anyhow!("Failed to allocate {} blocks", token_blocks.len()))?;

    let complete_blocks: Vec<_> = blocks
        .into_iter()
        .zip(token_blocks.iter())
        .map(|(block, token_block)| {
            block
                .complete(token_block.clone())
                .map_err(|e| anyhow::anyhow!("Failed to complete block: {:?}", e))
        })
        .collect::<Result<Vec<_>>>()?;

    let seq_hashes: Vec<SequenceHash> = complete_blocks.iter().map(|b| b.sequence_hash()).collect();

    let immutable_blocks = manager.register_blocks(complete_blocks);

    // Drop immutable blocks - they return to inactive pool via RAII
    drop(immutable_blocks);

    Ok(seq_hashes)
}

/// Quick setup: create manager and populate with sequential token blocks.
///
/// # Arguments
/// * `block_count` - Number of blocks
/// * `block_size` - Tokens per block
/// * `start_token` - Starting token value for sequence
///
/// # Returns
/// (BlockManager, Vec<SequenceHash>)
///
/// # Example
/// ```ignore
/// let (manager, hashes) = create_and_populate_manager::<G2>(32, 4, 0)?;
/// assert_eq!(hashes.len(), 32);
/// assert_eq!(manager.available_blocks(), 32);  // All in inactive pool
/// ```
pub fn create_and_populate_manager<T: BlockMetadata>(
    block_count: usize,
    block_size: usize,
    start_token: u32,
) -> Result<(BlockManager<T>, Vec<SequenceHash>)> {
    let manager = create_test_manager(block_count, block_size);

    let token_sequence = token_blocks::create_token_sequence(block_count, block_size, start_token);
    let seq_hashes = populate_manager_with_blocks(&manager, token_sequence.blocks())?;

    Ok((manager, seq_hashes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct TestMetadata;

    #[test]
    fn test_create_test_manager() {
        let manager = create_test_manager::<TestMetadata>(100, 16);
        assert_eq!(manager.total_blocks(), 100);
        assert_eq!(manager.block_size(), 16);
        assert_eq!(manager.available_blocks(), 100);
    }

    #[test]
    fn test_populate_manager_with_blocks() {
        let manager = create_test_manager::<TestMetadata>(50, 4);
        let token_seq = token_blocks::create_token_sequence(10, 4, 0);

        let seq_hashes =
            populate_manager_with_blocks(&manager, token_seq.blocks()).expect("Should populate");

        assert_eq!(seq_hashes.len(), 10);
        // Blocks should be in inactive pool after population
        assert_eq!(manager.available_blocks(), 50);
    }

    #[test]
    fn test_create_and_populate_manager() {
        let (manager, hashes) =
            create_and_populate_manager::<TestMetadata>(32, 4, 100).expect("Should create");

        assert_eq!(hashes.len(), 32);
        assert_eq!(manager.total_blocks(), 32);
        assert_eq!(manager.available_blocks(), 32);

        // Verify blocks can be matched
        let matched = manager.match_blocks(&hashes);
        assert_eq!(matched.len(), 32);
    }
}
