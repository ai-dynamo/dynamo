// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trait defining the common interface for KV manager backends.

use crate::protocols::{MoveBlock, PrefillCost};
use crate::sequence::ActiveSequence;
use dynamo_tokens::PositionalLineageHash;
use dynamo_tokens::blocks::UniqueBlock;

/// Common interface implemented by each KV manager backend.
///
/// Required methods capture the per-backend logic (allocation, eviction, block
/// lifecycle), while default methods provide shared derived computations.
pub trait KvBackend {
    /// Process a `MoveBlock` instruction. Returns `false` when allocation fails
    /// (the scheduler uses this to trigger preemption).
    fn process(&mut self, event: &MoveBlock) -> bool;

    /// Total number of block slots available to this backend.
    fn max_capacity(&self) -> usize;

    /// Number of tokens per block.
    fn block_size(&self) -> usize;

    /// Number of blocks currently held by active requests.
    fn num_active_blocks(&self) -> usize;

    /// Number of blocks in the inactive (reclaimable) pool.
    fn num_inactive_blocks(&self) -> usize;

    /// Total blocks currently in use (active + inactive).
    fn current_capacity(&self) -> usize;

    /// Count how many of `blocks` are *not* present in any pool.
    fn probe_new_blocks(&self, blocks: &[UniqueBlock]) -> usize;

    /// Returns `true` when the block identified by `seq_hash` can be found in
    /// either the active or inactive pools. `plh` is provided for backends that
    /// need it for registry look-ups (e.g. kvbm-logical).
    fn is_block_cached(&self, seq_hash: u64, plh: Option<PositionalLineageHash>) -> bool;

    // ------------------------------------------------------------------
    // Default methods — shared logic across backends
    // ------------------------------------------------------------------

    /// Current capacity as a fraction of `max_capacity`.
    fn current_capacity_perc(&self) -> f64 {
        self.current_capacity() as f64 / self.max_capacity() as f64
    }

    /// Active blocks as a fraction of `max_capacity`.
    fn get_active_perc(&self) -> f64 {
        self.num_active_blocks() as f64 / self.max_capacity() as f64
    }

    /// Calculate the prefill cost for a sequence by finding the longest cached
    /// prefix and computing the number of new blocks/tokens required.
    fn get_prefill_cost(&self, sequence: &ActiveSequence) -> PrefillCost {
        let seq_blocks = sequence.unique_blocks();
        let plhs = sequence.positional_lineage_hashes();

        let mut overlap_blocks = 0;
        for (i, block) in seq_blocks.iter().enumerate() {
            match block {
                UniqueBlock::FullBlock(seq_hash) => {
                    let plh = plhs.get(i).copied();
                    if !self.is_block_cached(*seq_hash, plh) {
                        break;
                    }
                    overlap_blocks += 1;
                }
                UniqueBlock::PartialBlock(_) => {
                    // Partial blocks don't contribute to cache overlap
                    break;
                }
            }
        }

        let new_blocks = seq_blocks.len() - overlap_blocks;
        let cached_tokens = (overlap_blocks * self.block_size()).min(sequence.num_input_tokens());
        let new_tokens = sequence.num_input_tokens() - cached_tokens;

        PrefillCost {
            new_blocks,
            new_tokens,
        }
    }
}
