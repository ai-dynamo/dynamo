// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard for mutable blocks in Reset state

use super::{
    Block, BlockError, BlockId, BlockMetadata, CompleteBlock, ResetReturnFn, SequenceHash,
    state::Reset,
};

use dynamo_tokens::TokenBlock;

/// RAII guard for [`Block<T, Reset>`] that automatically returns to ResetPool on drop
pub struct MutableBlock<T: BlockMetadata> {
    block: Option<Block<T, Reset>>,
    return_fn: ResetReturnFn<T>,
}

impl<T: BlockMetadata> MutableBlock<T> {
    /// Create a new MutableBlock in Reset state
    pub(crate) fn new(block: Block<T, Reset>, return_fn: ResetReturnFn<T>) -> Self {
        Self {
            block: Some(block),
            return_fn,
        }
    }

    /// Get the block ID
    pub fn block_id(&self) -> BlockId {
        self.block_ref().block_id()
    }

    /// Transition from Reset to Complete state with just a [`SequenceHash`]
    /// bypassing the check on block_size
    ///
    /// WARNING: This should only be used when the block size is known to be correct
    pub fn stage(mut self, seq_hash: SequenceHash) -> CompleteBlock<T> {
        CompleteBlock::new(self.take_block().stage(seq_hash), self.return_fn.clone())
    }

    /// Transition from Reset to Complete state
    pub fn complete(
        mut self,
        token_block: &TokenBlock,
    ) -> Result<CompleteBlock<T>, BlockError<MutableBlock<T>>> {
        let block = self.take_block();
        match block.complete(token_block) {
            Ok(complete_block) => Ok(CompleteBlock::new(complete_block, self.return_fn.clone())),
            Err(block_error) => {
                // Extract the block from the error and put it back in self
                match block_error {
                    BlockError::BlockSizeMismatch {
                        expected,
                        actual,
                        block,
                    } => {
                        self.block = Some(block);
                        Err(BlockError::BlockSizeMismatch {
                            expected,
                            actual,
                            block: self,
                        })
                    }
                }
            }
        }
    }

    pub(crate) fn into_parts(mut self) -> (Block<T, Reset>, ResetReturnFn<T>) {
        (self.take_block(), self.return_fn.clone())
    }

    #[inline(always)]
    fn take_block(&mut self) -> Block<T, Reset> {
        self.block.take().expect("MutableBlock missing block")
    }

    #[inline(always)]
    fn block_ref(&self) -> &Block<T, Reset> {
        self.block.as_ref().expect("MutableBlock missing block")
    }
}

impl<T: BlockMetadata> Drop for MutableBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block);
        }
    }
}

impl<T: BlockMetadata> std::fmt::Debug for MutableBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutableBlock")
            .field("block_id", &self.block.as_ref().map(|b| b.block_id()))
            .finish()
    }
}
