// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard for a block in the **Staged** state.
//!
//! A [`CompleteBlock`] represents a block that has been assigned a
//! [`SequenceHash`](super::SequenceHash) but has not yet been registered in
//! the [`BlockRegistry`](crate::registry::BlockRegistry). It is produced by
//! [`MutableBlock::stage`](super::MutableBlock::stage) or
//! [`MutableBlock::complete`](super::MutableBlock::complete) and can be
//! either registered via
//! [`BlockManager::register_block`](crate::manager::BlockManager::register_block)
//! or rolled back to a [`MutableBlock`](super::MutableBlock) with
//! [`reset`](CompleteBlock::reset).

use super::{
    Block, BlockId, BlockMetadata, MutableBlock, ResetReturnFn, SequenceHash, state::Staged,
};

/// RAII guard for a block in the **Staged** state.
///
/// Wraps an internal `Block<T, Staged>` -- a block that carries a
/// [`SequenceHash`](super::SequenceHash) and is ready for registration.
///
/// # Obtaining a `CompleteBlock`
///
/// - [`MutableBlock::stage`](super::MutableBlock::stage) -- from a
///   pre-computed [`SequenceHash`](super::SequenceHash).
/// - [`MutableBlock::complete`](super::MutableBlock::complete) -- by
///   extracting the hash from a [`TokenBlock`](dynamo_tokens::TokenBlock).
///
/// # State transitions
///
/// - Pass to [`BlockManager::register_block`](crate::manager::BlockManager::register_block)
///   to move the block into the **Registered** state and receive an
///   [`ImmutableBlock`](super::ImmutableBlock).
/// - Call [`reset`](Self::reset) to undo the staging and get a
///   [`MutableBlock`](super::MutableBlock) back (metrics are `None` on this
///   path).
///
/// # Drop behaviour
///
/// If the `CompleteBlock` is dropped without being consumed, the underlying
/// block is reset to the **Reset** state and returned to the reset pool.
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

    /// Returns the [`BlockId`] assigned to this block.
    pub fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    /// Returns the [`SequenceHash`](super::SequenceHash) that was assigned
    /// during staging.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    /// Undoes the staging transition, returning a [`MutableBlock`] in the
    /// **Reset** state.
    ///
    /// The returned `MutableBlock` does **not** carry metrics (they are set
    /// to `None`) because this is an undo/rollback path rather than a fresh
    /// allocation.
    pub fn reset(mut self) -> MutableBlock<T> {
        let block = self.block.take().unwrap().reset();

        MutableBlock::new(block, self.return_fn.clone(), None)
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
