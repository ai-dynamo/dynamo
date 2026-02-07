// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard for mutable blocks in Reset state

use super::{
    Block, BlockError, BlockId, BlockMetadata, CompleteBlock, ResetReturnFn, SequenceHash,
    state::Reset,
};

use crate::metrics::BlockPoolMetrics;
use dynamo_tokens::TokenBlock;
use std::sync::Arc;

/// RAII guard for [`Block<T, Reset>`] that automatically returns to ResetPool on drop
pub struct MutableBlock<T: BlockMetadata> {
    block: Option<Block<T, Reset>>,
    return_fn: ResetReturnFn<T>,
    metrics: Option<Arc<BlockPoolMetrics>>,
}

impl<T: BlockMetadata> MutableBlock<T> {
    /// Create a new MutableBlock in Reset state
    pub(crate) fn new(
        block: Block<T, Reset>,
        return_fn: ResetReturnFn<T>,
        metrics: Option<Arc<BlockPoolMetrics>>,
    ) -> Self {
        if let Some(ref m) = metrics {
            m.inc_inflight_mutable();
        }
        Self {
            block: Some(block),
            return_fn,
            metrics,
        }
    }

    /// Get the block ID
    pub fn block_id(&self) -> BlockId {
        self.block_ref().block_id()
    }

    /// Transition from Reset to Complete state with just a [`SequenceHash`].
    ///
    /// Validates `block_size` against the inner block's size, returning
    /// `Err(BlockError::BlockSizeMismatch)` on mismatch (same as [`complete`](Self::complete)).
    pub fn stage(
        mut self,
        seq_hash: SequenceHash,
        block_size: usize,
    ) -> Result<CompleteBlock<T>, BlockError<MutableBlock<T>>> {
        let inner_size = self.block_ref().block_size();
        if block_size != inner_size {
            return Err(BlockError::BlockSizeMismatch {
                expected: inner_size,
                actual: block_size,
                block: self,
            });
        }
        if let Some(ref m) = self.metrics {
            m.inc_stagings();
        }
        Ok(CompleteBlock::new(
            self.take_block().stage(seq_hash),
            self.return_fn.clone(),
        ))
    }

    /// Transition from Reset to Complete state
    pub fn complete(
        mut self,
        token_block: &TokenBlock,
    ) -> Result<CompleteBlock<T>, BlockError<MutableBlock<T>>> {
        let block = self.take_block();
        match block.complete(token_block) {
            Ok(complete_block) => {
                if let Some(ref m) = self.metrics {
                    m.inc_stagings();
                }
                Ok(CompleteBlock::new(complete_block, self.return_fn.clone()))
            }
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
        if let Some(ref m) = self.metrics {
            m.dec_inflight_mutable();
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
