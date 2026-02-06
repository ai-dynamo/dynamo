// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard for complete blocks

use super::{
    Block, BlockId, BlockMetadata, MutableBlock, ResetReturnFn, SequenceHash,
    state::Staged,
};

/// RAII guard for [`Block<T, Staged>`] that automatically returns to ResetPool on drop
pub struct CompleteBlock<T: BlockMetadata> {
    pub(crate) block: Option<Block<T, Staged>>,
    pub(crate) return_fn: ResetReturnFn<T>,
}

impl<T: BlockMetadata> CompleteBlock<T> {
    /// Create a new CompleteBlock
    pub(crate) fn new(block: Block<T, Staged>, return_fn: ResetReturnFn<T>) -> Self {
        Self {
            block: Some(block),
            return_fn,
        }
    }

    /// Get the block ID
    pub fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    /// Get sequence hash
    pub fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    /// Reset the block back to mutable state
    pub fn reset(mut self) -> MutableBlock<T> {
        let block = self.block.take().unwrap().reset();

        MutableBlock::new(block, self.return_fn.clone())
    }
}

impl<T: BlockMetadata> Drop for CompleteBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block.reset());
        }
    }
}
