// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard for mutable blocks in Reset state

use std::sync::Arc;

use super::{Block, BlockError, BlockId, BlockMetadata, CompleteBlock, state::Reset};

use dynamo_tokens::TokenBlock;

/// RAII guard for [`Block<T, Reset>`] that automatically returns to ResetPool on drop
pub struct MutableBlock<T: BlockMetadata> {
    block: Option<Block<T, Reset>>,
    return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
}

impl<T: BlockMetadata> MutableBlock<T> {
    /// Create a new MutableBlock in Reset state
    pub(crate) fn new(
        block: Block<T, Reset>,
        return_fn: Arc<dyn Fn(Block<T, Reset>) + Send + Sync>,
    ) -> Self {
        Self {
            block: Some(block),
            return_fn,
        }
    }

    /// Get the block ID
    pub fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    /// Transition from Reset to Complete state
    pub fn complete(
        mut self,
        token_block: TokenBlock,
    ) -> Result<CompleteBlock<T>, BlockError<MutableBlock<T>>> {
        let block = self.block.take().unwrap();
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

    pub(crate) fn into_parts(
        mut self,
    ) -> (Block<T, Reset>, Arc<dyn Fn(Block<T, Reset>) + Send + Sync>) {
        let block = self.block.take().expect("MutableBlock missing block");
        (block, self.return_fn.clone())
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
            .field("block", &self.block.as_ref().map(|b| b.block_id()))
            .field("return_fn", &"<function>")
            .finish()
    }
}
