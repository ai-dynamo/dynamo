// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::block_manager::pool::{BlockPoolError, BlockPoolResult, ResetBlocksResponse};
use std::sync::{Arc, Mutex};

/// Direct access to the block pool state, bypassing the progress engine.
/// This provides synchronous access for performance-critical paths.
///
/// Note: This is a simplified initial implementation that provides basic
/// direct access without complex retry logic.
pub struct DirectAccess<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    state: Arc<Mutex<State<S, L, M>>>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Clone for DirectAccess<S, L, M> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> DirectAccess<S, L, M> {
    pub fn new(state: Arc<Mutex<State<S, L, M>>>) -> Self {
        Self { state }
    }

    /// Get a reference to the state - used for testing
    #[allow(dead_code)]
    pub(crate) fn state(&self) -> Arc<Mutex<State<S, L, M>>> {
        self.state.clone()
    }

    /// Allocate a set of blocks from the pool.
    pub fn allocate_blocks(&self, count: usize) -> BlockPoolResult<Vec<MutableBlock<S, L, M>>> {
        let mut state = self.state.lock().unwrap();
        state.allocate_blocks(count)
    }

    /// Add blocks to the inactive pool.
    pub fn add_blocks(&self, blocks: Vec<Block<S, L, M>>) {
        let mut state = self.state.lock().unwrap();
        state.inactive.add_blocks(blocks);
    }

    /// Internal helper to return blocks to the pool.
    fn return_blocks_internal(&self, blocks: Vec<Block<S, L, M>>) -> BlockPoolResult<()> {
        if blocks.is_empty() {
            return Ok(());
        }

        let mut state = self.state.lock().unwrap();
        for block in blocks {
            state.return_block(block);
        }

        Ok(())
    }

    /// Try to return mutable blocks to the pool.
    ///
    /// This method takes ownership of the MutableBlocks, extracts their inner Block,
    /// and returns them to the pool.
    pub fn try_return_mutable_blocks(
        &self,
        blocks: Vec<MutableBlock<S, L, M>>,
    ) -> BlockPoolResult<()> {
        if blocks.is_empty() {
            return Ok(());
        }

        let mut raw_blocks = Vec::with_capacity(blocks.len());
        for block in blocks {
            if let Some(raw_block) = block.try_take_block(private::PrivateToken) {
                raw_blocks.extend(raw_block);
            }
        }

        self.return_blocks_internal(raw_blocks)
    }

    /// Try to return immutable blocks to the pool.
    ///
    /// This method takes ownership of the ImmutableBlocks and attempts to return
    /// their inner blocks to the pool. Note that blocks may not be returnable if
    /// there are still other references to them.
    pub fn try_return_immutable_blocks(
        &self,
        blocks: Vec<ImmutableBlock<S, L, M>>,
    ) -> BlockPoolResult<()> {
        if blocks.is_empty() {
            return Ok(());
        }

        let mut raw_blocks = Vec::new();
        for block in blocks {
            if let Some(extracted_blocks) = block.try_take_block(private::PrivateToken) {
                raw_blocks.extend(extracted_blocks);
            }
        }

        self.return_blocks_internal(raw_blocks)
    }

    /// Get the current status of the block pool.
    pub fn status(&self) -> Result<BlockPoolStatus, BlockPoolError> {
        let state = self.state.lock().unwrap();
        Ok(state.status())
    }

    /// Reset the pool, returning all blocks to the inactive state.
    pub fn reset(&self) -> Result<(), BlockPoolError> {
        let mut state = self.state.lock().unwrap();
        state.inactive.reset()
    }

    /// Reset specific blocks by sequence hash.
    pub fn reset_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ResetBlocksResponse, BlockPoolError> {
        let mut state = self.state.lock().unwrap();
        Ok(state.try_reset_blocks(sequence_hashes))
    }
}
