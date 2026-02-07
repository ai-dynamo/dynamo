// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reuse policies for determining block allocation priority.
//!
//! Different policies (FIFO, LRU, etc.) control which inactive registered
//! block should be allocated next when the reset pool is exhausted.

use super::{BlockId, InactiveBlock};

#[derive(Debug, thiserror::Error)]
pub enum ReusePolicyError {
    #[error("Block {0} already exists in free list")]
    BlockAlreadyExists(BlockId),

    #[error("Block {0} not found in free list")]
    BlockNotFound(BlockId),
}

/// Trait for managing a free list of blocks
///
/// Different implementations can provide different priority strategies
/// for selecting which block to allocate next.
pub trait ReusePolicy: Send + Sync + std::fmt::Debug {
    /// Insert a block into the free list
    ///
    /// The implementation will compute the priority key and manage the free list
    /// based on its specific strategy.
    fn insert(&mut self, inactive_block: InactiveBlock) -> Result<(), ReusePolicyError>;

    /// Remove a specific block from the free list
    fn remove(&mut self, block_id: BlockId) -> Result<(), ReusePolicyError>;

    /// Get the next free block based on the implementation's priority strategy
    ///
    /// Returns None if the free list is empty.
    /// The returned InactiveBlock contains both the block_id and seq_hash needed
    /// to look up the block in the InactivePool's HashMap.
    fn next_free(&mut self) -> Option<InactiveBlock>;

    /// Check if the free list is empty
    fn is_empty(&self) -> bool;

    /// Get the number of free blocks
    fn len(&self) -> usize;
}
