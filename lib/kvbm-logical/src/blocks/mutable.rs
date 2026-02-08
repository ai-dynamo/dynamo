// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard for a block in the **Reset** state.
//!
//! A [`MutableBlock`] is the entry point of the block lifecycle. It is
//! obtained from [`BlockManager::allocate_blocks`](crate::manager::BlockManager::allocate_blocks)
//! or by calling [`CompleteBlock::reset`](super::CompleteBlock::reset), and
//! can be advanced to a [`CompleteBlock`](super::CompleteBlock) via
//! [`stage`](MutableBlock::stage) or [`complete`](MutableBlock::complete).

use super::{
    Block, BlockError, BlockId, BlockMetadata, CompleteBlock, ResetReturnFn, SequenceHash,
    state::Reset,
};

use crate::metrics::BlockPoolMetrics;
use dynamo_tokens::TokenBlock;
use std::sync::Arc;

/// RAII guard for a block in the **Reset** state.
///
/// Wraps an internal `Block<T, Reset>` and guarantees that the block is
/// returned to the reset pool when the guard is dropped -- whether the
/// caller explicitly transitions it or simply lets it fall out of scope.
///
/// # Obtaining a `MutableBlock`
///
/// - [`BlockManager::allocate_blocks`](crate::manager::BlockManager::allocate_blocks)
///   -- pulls one or more blocks from the reset pool.
/// - [`CompleteBlock::reset`] -- undoes a staging operation, returning a
///   block to the Reset state (metrics are *not* carried over on this path).
///
/// # State transitions
///
/// - [`stage`](Self::stage) -- transitions to [`CompleteBlock`] using a
///   pre-computed [`SequenceHash`] and a block-size check.
/// - [`complete`](Self::complete) -- transitions to [`CompleteBlock`] by
///   extracting the hash from a [`TokenBlock`](dynamo_tokens::TokenBlock).
///
/// Both methods consume `self` and return the block inside
/// `Err(`[`BlockError`]`)` on size mismatch so it is never leaked.
///
/// # Drop behaviour
///
/// Dropping a `MutableBlock` returns the underlying block to the reset pool
/// and decrements the `inflight_mutable` metric gauge.
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

    /// Returns the [`BlockId`] assigned to this block.
    pub fn block_id(&self) -> BlockId {
        self.block_ref().block_id()
    }

    /// Transitions from **Reset** to **Staged**, producing a [`CompleteBlock`].
    ///
    /// The caller supplies a pre-computed [`SequenceHash`] and the expected
    /// `block_size`. If `block_size` does not match the block's fixed size
    /// the method returns `Err(`[`BlockError::BlockSizeMismatch`]`)` with the
    /// `MutableBlock` inside so the caller can recover it.
    ///
    /// Increments the `stagings` counter on success.
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

    /// Transitions from **Reset** to **Staged**, producing a [`CompleteBlock`].
    ///
    /// The [`SequenceHash`] is derived from the provided
    /// [`TokenBlock`](dynamo_tokens::TokenBlock). If the token block's size
    /// does not match the block's fixed size the method returns
    /// `Err(`[`BlockError::BlockSizeMismatch`]`)` with the `MutableBlock`
    /// inside so the caller can recover it.
    ///
    /// Increments the `stagings` counter on success.
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
