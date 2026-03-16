// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block pool RAII guards and allocation traits for thread-safe block management.
//!
//! This module provides:
//! - Type-safe RAII guards (MutableBlock, CompleteBlock, ImmutableBlock) for automatic resource cleanup
//! - ResetPool: Pool for mutable blocks in reset state
//! - InactivePool: Pool for inactive immutable registered blocks
//! - BlockRegistry: Global registry for block deduplication via weak references
//! - Pluggable allocation and reuse policies

mod active;
mod inactive;
mod reset;

#[cfg(test)]
pub mod tests;

#[cfg(test)]
mod block_proptest;

pub(crate) use active::ActivePool;
pub(crate) use inactive::backends;
pub(crate) use inactive::{InactivePool, InactivePoolBackend};
pub(crate) use reset::ResetPool;

// Re-export RAII guards from guards module
use crate::blocks::{
    Block, BlockId, BlockMetadata, ImmutableBlock, MutableBlock, PrimaryBlock, RegisteredBlock,
    state::{Registered, Reset},
};

pub(crate) use crate::SequenceHash;

pub(crate) trait BlockAllocator<T: BlockMetadata> {
    // fn new(blocks: Vec<Block<T, Reset>>) -> Arc<Self>
    // where
    //     Self: Sized;

    /// Insert a block into the pool
    fn insert(&mut self, block: Block<T, Reset>);

    /// Acquire the first block to be reused
    fn pop(&mut self) -> Option<Block<T, Reset>>;

    /// Get the number of available blocks
    fn len(&self) -> usize;

    /// Check if the pool is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[expect(dead_code)]
pub(crate) trait BlockMatcher<T: BlockMetadata> {
    fn find_match(&self, seq_hash: SequenceHash) -> Option<ImmutableBlock<T>>;
}

// Re-export block duplication policy
pub use crate::blocks::BlockDuplicationPolicy;

// Re-export reuse policy from inactive backends
pub use inactive::backends::{ReusePolicy, ReusePolicyError};

// Re-export the new RAII guard types - no need to re-export here since they're defined in this module

/// A block that is free and available for allocation
/// This block must be in a Registered state and have a valid sequence hash
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InactiveBlock {
    pub block_id: BlockId,
    pub seq_hash: SequenceHash,
}

// RegisteredPool implementation moved to registered.rs
