// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guards for registered blocks (primary and duplicate)

use std::any::TypeId;
use std::sync::{Arc, Weak};

use super::{
    Block, BlockId, BlockMetadata, BlockRegistrationHandle, RegisteredBlock, RegisteredReturnFn,
    ResetReturnFn, SequenceHash, state::Registered,
};

/// Weak references to a block for resurrection during pool transitions.
///
/// Both fields are load-bearing:
/// - `raw_block`: catches blocks in the brief window between PrimaryBlock::drop
///   (takes Arc out of Option) and return_fn completing (inserts into pool)
/// - `primary_block`: fast path for upgrading to an existing PrimaryBlock
pub(crate) struct WeakBlockEntry<T: BlockMetadata + Sync> {
    pub(crate) raw_block: Weak<Block<T, Registered>>,
    pub(crate) primary_block: Weak<PrimaryBlock<T>>,
}

/// RAII guard for [`Block<T, Registered>`] that automatically returns to RegisteredPool on drop
pub(crate) struct PrimaryBlock<T: BlockMetadata> {
    pub(crate) block: Option<Arc<Block<T, Registered>>>,
    pub(crate) return_fn: RegisteredReturnFn<T>,
}

/// RAII guard for duplicate blocks that share the same sequence hash as a primary block
pub(crate) struct DuplicateBlock<T: BlockMetadata> {
    pub(crate) block: Option<Block<T, Registered>>,
    pub(crate) return_fn: ResetReturnFn<T>,
    pub(crate) _primary: Arc<PrimaryBlock<T>>,
}

impl<T: BlockMetadata + Sync> PrimaryBlock<T> {
    /// Create a PrimaryBlock and auto-store weak refs in the handle's AttachmentStore.
    /// Acquires the attachments lock internally.
    /// Used for 5 of 6 instantiation sites.
    pub(crate) fn new_attached(
        block: Arc<Block<T, Registered>>,
        return_fn: RegisteredReturnFn<T>,
    ) -> Arc<Self> {
        let primary = Self {
            block: Some(block),
            return_fn,
        };
        let primary_arc = Arc::new(primary);
        Self::store_weak_refs(&primary_arc);
        primary_arc
    }

    /// Create a PrimaryBlock WITHOUT storing weak refs.
    /// Used only when the caller already holds the attachments lock
    /// (find_block_as_primary, called from try_find_existing_block).
    /// Caller MUST call store_weak_refs() after dropping the lock.
    pub(crate) fn new_unattached(
        block: Arc<Block<T, Registered>>,
        return_fn: RegisteredReturnFn<T>,
    ) -> Arc<Self> {
        let primary = Self {
            block: Some(block),
            return_fn,
        };
        Arc::new(primary)
    }

    /// Store weak references (both raw block and primary) in the handle's
    /// AttachmentStore for block resurrection during pool transitions.
    ///
    /// SAFETY: Must not be called while the attachments lock is held.
    /// The raw_block weak ref enables resurrection during the brief window
    /// between PrimaryBlock::drop and return_fn completing.
    pub(crate) fn store_weak_refs(primary_arc: &Arc<Self>) {
        let block_ref = primary_arc.block.as_ref().unwrap();
        let handle = block_ref.registration_handle();
        let type_id = TypeId::of::<Weak<Block<T, Registered>>>();

        let raw_block = Arc::downgrade(block_ref);
        let primary_block = Arc::downgrade(primary_arc);

        let mut attachments = handle.inner.attachments.lock();
        attachments.weak_blocks.insert(
            type_id,
            Box::new(WeakBlockEntry {
                raw_block,
                primary_block,
            }),
        );
    }
}

impl<T: BlockMetadata> DuplicateBlock<T> {
    /// Create a new DuplicateBlock
    pub(crate) fn new(
        block: Block<T, Registered>,
        primary: Arc<PrimaryBlock<T>>,
        return_fn: ResetReturnFn<T>,
    ) -> Self {
        Self {
            block: Some(block),
            return_fn,
            _primary: primary,
        }
    }
}

impl<T: BlockMetadata> RegisteredBlock<T> for PrimaryBlock<T> {
    fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    fn registration_handle(&self) -> &BlockRegistrationHandle {
        self.block.as_ref().unwrap().registration_handle()
    }
}

impl<T: BlockMetadata> RegisteredBlock<T> for DuplicateBlock<T> {
    fn block_id(&self) -> BlockId {
        self.block.as_ref().unwrap().block_id()
    }

    fn sequence_hash(&self) -> SequenceHash {
        self.block.as_ref().unwrap().sequence_hash()
    }

    fn registration_handle(&self) -> &BlockRegistrationHandle {
        self.block.as_ref().unwrap().registration_handle()
    }
}

impl<T: BlockMetadata> Drop for PrimaryBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block);
        }
    }
}

impl<T: BlockMetadata> Drop for DuplicateBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(block) = self.block.take() {
            (self.return_fn)(block.reset());
        }
    }
}
