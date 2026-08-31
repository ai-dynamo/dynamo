// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guard for a block in the **Staged** state.
//!
//! A [`CompleteBlock`] holds a slot that has been assigned a
//! [`SequenceHash`] but has not yet been registered. It is produced by
//! [`MutableBlock::stage`](super::MutableBlock::stage) or
//! [`MutableBlock::complete`](super::MutableBlock::complete) and can be
//! either registered via
//! [`BlockManager::register_block`](crate::manager::BlockManager::register_block)
//! or rolled back to a [`MutableBlock`] with [`reset`](CompleteBlock::reset).

use std::sync::Arc;

use crate::blocks::{BlockId, BlockMetadata, MutableBlock, SequenceHash};
use crate::pools::BlockStore;

/// RAII guard for a block in the **Staged** state.
pub struct CompleteBlock<T: BlockMetadata> {
    store: Arc<BlockStore<T>>,
    block_id: BlockId,
    block_size: usize,
    seq_hash: SequenceHash,
    /// `false` once the guard has been consumed by `reset` or by
    /// `BlockManager::register_block`; Drop becomes a no-op.
    armed: bool,
}

impl<T: BlockMetadata + Sync> CompleteBlock<T> {
    pub(crate) fn from_store(
        store: Arc<BlockStore<T>>,
        block_id: BlockId,
        block_size: usize,
        seq_hash: SequenceHash,
    ) -> Self {
        Self {
            store,
            block_id,
            block_size,
            seq_hash,
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

    /// Returns the [`SequenceHash`] assigned during staging.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.seq_hash
    }

    /// Roll back to a [`MutableBlock`].
    pub fn reset(mut self) -> MutableBlock<T> {
        self.store.transition_back_to_mutable(self.block_id);
        self.armed = false;
        MutableBlock::from_store(self.store.clone(), self.block_id, self.block_size)
    }

    /// Disarm the guard so Drop is a no-op. Used by registration when
    /// the slot will be transitioned `Staged → Primary` / `Duplicate`
    /// under the store mutex; the caller takes responsibility for the
    /// slot's onward state.
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }

    /// Re-arm a previously disarmed guard so Drop releases the slot
    /// `Staged → Reset`. Used by the registration `Reject` path after
    /// the lookup commits to dropping the staged block.
    pub(crate) fn rearm(&mut self) {
        self.armed = true;
    }
}

impl<T: BlockMetadata> Drop for CompleteBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if self.armed {
            self.store.release_staged(self.block_id);
        }
    }
}

impl<T: BlockMetadata> std::fmt::Debug for CompleteBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompleteBlock")
            .field("block_id", &self.block_id)
            .field("seq_hash", &self.seq_hash)
            .finish()
    }
}
