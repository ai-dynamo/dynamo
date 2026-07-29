// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard for a block in the **Reset** state.
//!
//! A [`MutableBlock`] is the entry point of the block lifecycle. It is
//! obtained from
//! [`BlockManager::allocate_blocks`](crate::manager::BlockManager::allocate_blocks)
//! or by calling [`CompleteBlock::reset`](super::CompleteBlock::reset), and
//! can be advanced to a [`CompleteBlock`](super::CompleteBlock) via
//! [`stage`](MutableBlock::stage) or [`complete`](MutableBlock::complete).

use std::sync::Arc;

use dynamo_tokens::TokenBlock;

use crate::KvbmSequenceHashProvider;
use crate::blocks::{BlockError, BlockId, BlockMetadata, CompleteBlock, SequenceHash};
use crate::pools::BlockStore;

/// RAII guard for a block in the **Reset** state.
///
/// Holds an `Arc<BlockStore<T>>` and a `BlockId`; the slot at that id is in
/// `SlotState::Mutable` while this guard exists. Drop returns the slot to
/// the reset pool.
///
/// # Drop behaviour
///
/// Dropping a `MutableBlock` returns the slot to the reset pool. The store's
/// `release_mutable` updates the `inflight_mutable` gauge.
pub struct MutableBlock<T: BlockMetadata> {
    store: Arc<BlockStore<T>>,
    block_id: BlockId,
    block_size: usize,
    /// `false` once the guard has been consumed by a state-transition
    /// method (`stage` / `complete`); Drop becomes a no-op.
    armed: bool,
}

impl<T: BlockMetadata + Sync> MutableBlock<T> {
    /// Build a new `MutableBlock` for a slot the store has just transitioned
    /// to `Mutable`. The store has already incremented the `inflight_mutable`
    /// gauge as part of the transition.
    pub(crate) fn from_store(
        store: Arc<BlockStore<T>>,
        block_id: BlockId,
        block_size: usize,
    ) -> Self {
        Self {
            store,
            block_id,
            block_size,
            armed: true,
        }
    }

    /// Returns the [`BlockId`] assigned to this block.
    pub fn block_id(&self) -> BlockId {
        self.block_id
    }

    /// Returns the fixed block size of this block in tokens.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Transition from **Reset** to **Staged** with a pre-computed
    /// [`SequenceHash`] and an explicit `block_size` check.
    ///
    /// On size mismatch returns `Err(`[`BlockError::BlockSizeMismatch`]`)`
    /// containing this `MutableBlock` so the caller can recover it.
    pub fn stage(
        mut self,
        seq_hash: SequenceHash,
        block_size: usize,
    ) -> Result<CompleteBlock<T>, BlockError<MutableBlock<T>>> {
        if block_size != self.block_size {
            return Err(BlockError::BlockSizeMismatch {
                expected: self.block_size,
                actual: block_size,
                block: self,
            });
        }
        self.store.transition_to_staged(self.block_id, seq_hash);
        let id = self.block_id;
        let bsize = self.block_size;
        let store = self.store.clone();
        // Disarm so Drop is a no-op; the slot is now in Staged state.
        self.armed = false;
        drop(self);
        Ok(CompleteBlock::from_store(store, id, bsize, seq_hash))
    }

    /// Transition from **Reset** to **Staged** by extracting the
    /// [`SequenceHash`] from a [`TokenBlock`].
    pub fn complete(
        self,
        token_block: &TokenBlock,
    ) -> Result<CompleteBlock<T>, BlockError<MutableBlock<T>>> {
        let actual = token_block.block_size();
        if actual != self.block_size {
            return Err(BlockError::BlockSizeMismatch {
                expected: self.block_size,
                actual,
                block: self,
            });
        }
        let seq_hash = token_block.kvbm_sequence_hash();
        self.stage(seq_hash, actual)
    }
}

impl<T: BlockMetadata> Drop for MutableBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if self.armed {
            self.store.release_mutable(self.block_id);
        }
    }
}

impl<T: BlockMetadata> std::fmt::Debug for MutableBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutableBlock")
            .field("block_id", &self.block_id)
            .finish()
    }
}
