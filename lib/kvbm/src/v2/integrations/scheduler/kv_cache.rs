// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV cache manager for G1 (GPU) block allocation.
//!
//! This module provides a simplified interface for the scheduler to allocate
//! and track KV cache blocks. It wraps the BlockManager and handles the
//! block lifecycle (Mutable -> Complete -> Immutable).

use crate::G1;
use crate::v2::BlockId;
use crate::v2::logical::blocks::{CompleteBlock, ImmutableBlock, MutableBlock};
use crate::v2::logical::manager::BlockManager;
use dynamo_tokens::TokenBlock;

/// Manager for KV cache blocks on GPU (G1 tier).
///
/// This wraps the BlockManager<G1> and provides a simplified interface
/// for the scheduler to allocate and track blocks.
///
/// # Block Lifecycle
///
/// The underlying BlockManager has a complex RAII lifecycle:
/// 1. `allocate_blocks()` -> `MutableBlock` (block is reserved)
/// 2. `complete()` -> `CompleteBlock` (block has token data)
/// 3. `register_blocks()` -> `ImmutableBlock` (block is in cache)
///
/// For the scheduler's purposes, we simplify this by:
/// - Allocating "placeholder" blocks that reserve capacity
/// - Tracking block IDs for the scheduler output
/// - The actual token data is filled by the model forward pass
pub struct KVCacheManager {
    /// The underlying block manager for G1 blocks.
    block_manager: BlockManager<G1>,

    /// Block size in tokens.
    block_size: usize,

    /// Total number of blocks in the cache.
    total_blocks: usize,
}

impl KVCacheManager {
    /// Create a new KV cache manager wrapping the given block manager.
    pub fn new(block_manager: BlockManager<G1>, block_size: usize) -> Self {
        let total_blocks = block_manager.total_blocks();
        Self {
            block_manager,
            block_size,
            total_blocks,
        }
    }

    /// Get the block size in tokens.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get the total number of blocks in the cache.
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Get the number of free blocks available for allocation.
    pub fn free_blocks(&self) -> usize {
        self.block_manager.available_blocks()
    }

    /// Get the number of blocks currently in use.
    pub fn used_blocks(&self) -> usize {
        self.total_blocks.saturating_sub(self.free_blocks())
    }

    /// Get the cache usage as a fraction (0.0 to 1.0).
    pub fn usage(&self) -> f32 {
        if self.total_blocks == 0 {
            0.0
        } else {
            self.used_blocks() as f32 / self.total_blocks as f32
        }
    }

    /// Check if there are enough free blocks to allocate the requested amount.
    pub fn can_allocate(&self, num_blocks: usize) -> bool {
        self.free_blocks() >= num_blocks
    }

    /// Get the number of tokens that can be stored with the current free blocks.
    pub fn free_token_capacity(&self) -> usize {
        self.free_blocks() * self.block_size
    }

    /// Get the number of blocks needed to store the given number of tokens.
    pub fn blocks_needed(&self, num_tokens: usize) -> usize {
        (num_tokens + self.block_size - 1) / self.block_size
    }

    /// Get a reference to the underlying block manager.
    ///
    /// This allows advanced operations like prefix matching.
    pub fn block_manager(&self) -> &BlockManager<G1> {
        &self.block_manager
    }

    /// Allocate mutable blocks from the BlockManager.
    ///
    /// Returns `Some(blocks)` if allocation succeeds, `None` if there are
    /// not enough available blocks. The returned blocks are RAII guards
    /// that will return to the reset pool when dropped.
    ///
    /// # Arguments
    /// * `count` - Number of blocks to allocate
    ///
    /// # Returns
    /// * `Some(Vec<MutableBlock<G1>>)` - Successfully allocated blocks
    /// * `None` - Not enough blocks available
    pub fn allocate(&self, count: usize) -> Option<Vec<MutableBlock<G1>>> {
        self.block_manager.allocate_blocks(count)
    }

    /// Complete and register blocks after token data is available.
    ///
    /// This transitions blocks through the complete lifecycle:
    /// 1. MutableBlock + TokenBlock -> CompleteBlock
    /// 2. CompleteBlock -> ImmutableBlock (registered in cache)
    ///
    /// # Arguments
    /// * `blocks` - Mutable blocks to complete
    /// * `token_blocks` - Token data for each block (must be same length as blocks)
    ///
    /// # Returns
    /// * `Ok(Vec<ImmutableBlock<G1>>)` - Successfully registered blocks
    /// * `Err(Vec<MutableBlock<G1>>)` - Blocks returned on failure (e.g., size mismatch)
    ///
    /// # Panics
    /// Panics if `blocks.len() != token_blocks.len()`
    pub fn complete_and_register(
        &self,
        blocks: Vec<MutableBlock<G1>>,
        token_blocks: Vec<TokenBlock>,
    ) -> Result<Vec<ImmutableBlock<G1>>, Vec<MutableBlock<G1>>> {
        assert_eq!(
            blocks.len(),
            token_blocks.len(),
            "blocks and token_blocks must have same length"
        );

        // Complete all blocks
        let mut complete_blocks: Vec<CompleteBlock<G1>> = Vec::with_capacity(blocks.len());
        let mut failed_blocks: Vec<MutableBlock<G1>> = Vec::new();

        for (block, token_block) in blocks.into_iter().zip(token_blocks.into_iter()) {
            match block.complete(token_block) {
                Ok(complete) => complete_blocks.push(complete),
                Err(err) => {
                    // Extract the block from the error
                    match err {
                        crate::v2::logical::blocks::BlockError::BlockSizeMismatch { block, .. } => {
                            failed_blocks.push(block);
                        }
                    }
                }
            }
        }

        // If any blocks failed, return all remaining mutable blocks
        if !failed_blocks.is_empty() {
            // Drop complete_blocks (they will return to pool via RAII)
            drop(complete_blocks);
            return Err(failed_blocks);
        }

        // Register all complete blocks
        Ok(self.block_manager.register_blocks(complete_blocks))
    }
}

/// Allocated blocks for a request.
///
/// This struct holds the block IDs allocated for a request. The actual
/// ImmutableBlock objects are managed separately since their lifecycle
/// involves token data that comes from the model forward pass.
#[derive(Debug, Clone, Default)]
pub struct AllocatedBlocks {
    /// Block IDs allocated to this request.
    pub block_ids: Vec<BlockId>,
}

impl AllocatedBlocks {
    /// Create a new empty allocation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from a list of immutable blocks.
    pub fn from_blocks(blocks: &[ImmutableBlock<G1>]) -> Self {
        Self {
            block_ids: blocks.iter().map(|b| b.block_id()).collect(),
        }
    }

    /// Get the number of allocated blocks.
    pub fn len(&self) -> usize {
        self.block_ids.len()
    }

    /// Check if no blocks are allocated.
    pub fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }

    /// Extend with additional block IDs.
    pub fn extend(&mut self, block_ids: impl IntoIterator<Item = BlockId>) {
        self.block_ids.extend(block_ids);
    }

    /// Clear all allocations.
    pub fn clear(&mut self) {
        self.block_ids.clear();
    }
}

/// Per-request block state holding RAII blocks.
///
/// This struct manages the lifecycle of blocks for a single request,
/// holding both pending (mutable) blocks and registered (immutable) blocks.
///
/// # Block Lifecycle
///
/// 1. **Allocation**: Scheduler calls `KVCacheManager::allocate()` to get
///    `MutableBlock<G1>` and stores them in `pending`.
///
/// 2. **Registration**: After the forward pass computes token data,
///    `KVCacheManager::complete_and_register()` transitions pending blocks
///    to registered `ImmutableBlock<G1>`.
///
/// 3. **Cleanup**: When the request finishes, all blocks are dropped via RAII,
///    returning them to the appropriate pools.
#[derive(Default)]
pub struct RequestBlockState {
    /// Blocks that have been allocated but not yet completed (pending forward pass).
    /// These are MutableBlock<G1> that have been reserved for this request
    /// but don't yet contain token data.
    pending: Vec<MutableBlock<G1>>,

    /// Blocks that have been completed and registered in the cache.
    /// These contain token data and are in the active/inactive pools.
    registered: Vec<ImmutableBlock<G1>>,
}

impl RequestBlockState {
    /// Create a new empty block state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add pending (mutable) blocks.
    pub fn add_pending(&mut self, blocks: Vec<MutableBlock<G1>>) {
        self.pending.extend(blocks);
    }

    /// Add registered (immutable) blocks.
    pub fn add_registered(&mut self, blocks: Vec<ImmutableBlock<G1>>) {
        self.registered.extend(blocks);
    }

    /// Get the number of pending blocks.
    pub fn num_pending(&self) -> usize {
        self.pending.len()
    }

    /// Get the number of registered blocks.
    pub fn num_registered(&self) -> usize {
        self.registered.len()
    }

    /// Get the total number of blocks (pending + registered).
    pub fn total_blocks(&self) -> usize {
        self.pending.len() + self.registered.len()
    }

    /// Check if there are no blocks.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && self.registered.is_empty()
    }

    /// Take all pending blocks out of the state.
    ///
    /// This is used when transitioning pending blocks to registered after
    /// a forward pass completes.
    pub fn take_pending(&mut self) -> Vec<MutableBlock<G1>> {
        std::mem::take(&mut self.pending)
    }

    /// Get block IDs of all pending blocks.
    pub fn pending_block_ids(&self) -> Vec<BlockId> {
        self.pending.iter().map(|b| b.block_id()).collect()
    }

    /// Get block IDs of all registered blocks.
    pub fn registered_block_ids(&self) -> Vec<BlockId> {
        self.registered.iter().map(|b| b.block_id()).collect()
    }

    /// Get all block IDs (pending + registered).
    pub fn all_block_ids(&self) -> Vec<BlockId> {
        let mut ids = self.pending_block_ids();
        ids.extend(self.registered_block_ids());
        ids
    }

    /// Clear all blocks, returning them to pools via RAII.
    ///
    /// This is called when a request is preempted or finished.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.registered.clear();
    }
}

impl std::fmt::Debug for RequestBlockState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestBlockState")
            .field("pending", &self.pending.len())
            .field("registered", &self.registered.len())
            .field("pending_ids", &self.pending_block_ids())
            .field("registered_ids", &self.registered_block_ids())
            .finish()
    }
}

impl std::fmt::Debug for KVCacheManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KVCacheManager")
            .field("block_size", &self.block_size)
            .field("total_blocks", &self.total_blocks)
            .field("free_blocks", &self.free_blocks())
            .field("usage", &format!("{:.1}%", self.usage() * 100.0))
            .finish()
    }
}
