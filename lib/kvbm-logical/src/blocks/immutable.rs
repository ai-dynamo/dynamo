// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guards for blocks in the **Registered** state.
//!
//! [`ImmutableBlock`] is a cheap-to-clone strong handle backed by an
//! `Arc<ImmutableBlockInner<T>>`. The inner carries the slot's identity
//! ([`BlockId`], [`SequenceHash`], [`BlockRegistrationHandle`]) plus an
//! `is_primary` flag that decides the slot's drop transition:
//!
//! - **Primary** (`is_primary = true`): the canonical holder of a sequence
//!   hash. Drop of the last clone moves the slot to `Inactive` so it can
//!   be evicted later.
//! - **Duplicate** (`is_primary = false`): a second physical block sharing
//!   the same hash. It carries a strong [`Arc`] reference to the primary
//!   inner so the primary cannot be evicted while a duplicate exists. Drop
//!   of the last clone resets the slot via `mark_absent`.
//!
//! [`WeakBlock`] is a non-owning handle that can be upgraded back to an
//! [`ImmutableBlock`] either via `Weak::upgrade` (fast path) or by
//! resurrecting an evicted block from the store's inactive pool through
//! the registry (slow path).

use std::sync::{Arc, Weak};

use crate::ManagerId;
use crate::blocks::pin::{LifecyclePin, LifecyclePinRef};
use crate::blocks::{BlockId, BlockMetadata, BlockRegistrationHandle, SequenceHash};
use crate::pools::{BlockStore, store::upgrade_or_resurrect};

/// Internal owner of a registered slot. Every clone of an
/// [`ImmutableBlock`] shares an `Arc<ImmutableBlockInner<T>>`. When the
/// last `Arc` is dropped the slot transitions per `is_primary`.
pub(crate) struct ImmutableBlockInner<T: BlockMetadata> {
    store: Arc<BlockStore<T>>,
    block_id: BlockId,
    seq_hash: SequenceHash,
    handle: BlockRegistrationHandle,
    is_primary: bool,
    /// For duplicates, holds a strong reference to the primary's inner so
    /// the primary cannot transition to `Inactive` (and thus be evicted)
    /// while any duplicate is alive.
    _primary_keepalive: Option<Arc<ImmutableBlockInner<T>>>,
}

impl<T: BlockMetadata + Sync> ImmutableBlockInner<T> {
    pub(crate) fn new_primary(
        store: Arc<BlockStore<T>>,
        block_id: BlockId,
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            block_id,
            seq_hash,
            handle,
            is_primary: true,
            _primary_keepalive: None,
        })
    }

    pub(crate) fn new_duplicate(
        store: Arc<BlockStore<T>>,
        block_id: BlockId,
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
        primary: Arc<ImmutableBlockInner<T>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            store,
            block_id,
            seq_hash,
            handle,
            is_primary: false,
            _primary_keepalive: Some(primary),
        })
    }

    pub(crate) fn block_id(&self) -> BlockId {
        self.block_id
    }
}

impl<T: BlockMetadata + Sync> LifecyclePin for ImmutableBlockInner<T> {
    fn block_id(&self) -> BlockId {
        self.block_id
    }
    fn sequence_hash(&self) -> SequenceHash {
        self.seq_hash
    }
    fn manager_id(&self) -> ManagerId {
        self.store.id()
    }
    fn registration_handle(&self) -> BlockRegistrationHandle {
        self.handle.clone()
    }
}

impl<T: BlockMetadata> Drop for ImmutableBlockInner<T> {
    fn drop(&mut self) {
        // self_ptr identifies *this* Inner so the store can verify slot
        // identity before transitioning. If a concurrent
        // `acquire_for_hash` already eagerly completed the transition,
        // the store call is a no-op.
        let self_ptr = self as *const ImmutableBlockInner<T> as *const ();
        if self.is_primary {
            self.store.release_primary(self.block_id, self_ptr);
        } else {
            self.store.release_duplicate(self.block_id, self_ptr);
        }
    }
}

/// RAII guard for a block in the **Registered** state.
///
/// `Clone` increments an `Arc` and the `inflight_immutable` metric;
/// dropping a clone decrements the metric. Dropping the last strong
/// reference triggers the slot's transition to `Inactive` (primary) or
/// `Reset` (duplicate).
pub struct ImmutableBlock<T: BlockMetadata> {
    inner: Arc<ImmutableBlockInner<T>>,
}

impl<T: BlockMetadata + Sync> ImmutableBlock<T> {
    pub(crate) fn from_inner(inner: Arc<ImmutableBlockInner<T>>) -> Self {
        inner.store.metrics().inc_inflight_immutable();
        Self { inner }
    }

    /// Creates a [`WeakBlock`] that does not prevent the block from being
    /// evicted.
    pub fn downgrade(&self) -> WeakBlock<T> {
        WeakBlock {
            sequence_hash: self.inner.seq_hash,
            inner: Arc::downgrade(&self.inner),
            handle: self.inner.handle.clone(),
            store: self.inner.store.clone(),
        }
    }

    /// Returns the [`BlockId`] assigned to this block.
    pub fn block_id(&self) -> BlockId {
        self.inner.block_id
    }

    /// Returns the [`SequenceHash`] that identifies this block's content.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.inner.seq_hash
    }

    /// Returns a clone of the [`BlockRegistrationHandle`] for this block.
    pub fn registration_handle(&self) -> BlockRegistrationHandle {
        self.inner.handle.clone()
    }

    /// Returns the number of strong [`Arc`] references to the underlying
    /// inner.
    pub fn use_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Type-erased lifecycle pin for cross-policy use.
    ///
    /// Returns an [`LifecyclePinRef`] that:
    /// - Bumps the underlying `Arc<ImmutableBlockInner<T>>` once (the
    ///   only allocation cost is one `Arc::clone` — no `Box` and no new
    ///   heap node).
    /// - Keeps the slot alive (preventing the `Active → Inactive`
    ///   transition) while the pin is live, identical to holding an
    ///   `ImmutableBlock` clone.
    /// - Exposes `(manager_id, block_id, sequence_hash)` so callers that
    ///   stash a heterogeneous list of pins (different `T`s) can still
    ///   address each slot unambiguously at runtime.
    ///
    /// See [`crate::blocks::pin`] for the rationale.
    pub fn pin(&self) -> LifecyclePinRef {
        LifecyclePinRef::new(self.inner.clone() as Arc<dyn LifecyclePin>)
    }
}

impl<T: BlockMetadata + Sync> Clone for ImmutableBlock<T> {
    fn clone(&self) -> Self {
        self.inner.store.metrics().inc_inflight_immutable();
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: BlockMetadata> Drop for ImmutableBlock<T> {
    #[inline]
    fn drop(&mut self) {
        self.inner.store.metrics().dec_inflight_immutable();
    }
}

impl<T: BlockMetadata> std::fmt::Debug for ImmutableBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImmutableBlock")
            .field("block_id", &self.inner.block_id)
            .field("sequence_hash", &self.inner.seq_hash)
            .finish()
    }
}

/// Non-owning reference to a registered block.
///
/// Created via [`ImmutableBlock::downgrade`]. Cheap to clone. Calling
/// [`upgrade`](Self::upgrade) tries `Weak::upgrade` first (fast path) and
/// falls back to resurrecting the block from the store's inactive pool
/// via the registry (slow path).
pub struct WeakBlock<T: BlockMetadata> {
    sequence_hash: SequenceHash,
    inner: Weak<ImmutableBlockInner<T>>,
    handle: BlockRegistrationHandle,
    store: Arc<BlockStore<T>>,
}

impl<T: BlockMetadata + Sync> WeakBlock<T> {
    /// Attempt to upgrade this weak reference to a strong [`ImmutableBlock`].
    pub fn upgrade(&self) -> Option<ImmutableBlock<T>> {
        if let Some(strong) = self.inner.upgrade() {
            return Some(ImmutableBlock::from_inner(strong));
        }
        let inner = upgrade_or_resurrect::<T>(&self.handle, &self.store, false)?;
        Some(ImmutableBlock::from_inner(inner))
    }

    /// Returns the [`SequenceHash`] for the block this weak reference
    /// points to.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }
}

impl<T: BlockMetadata> Clone for WeakBlock<T> {
    fn clone(&self) -> Self {
        Self {
            sequence_hash: self.sequence_hash,
            inner: self.inner.clone(),
            handle: self.handle.clone(),
            store: self.store.clone(),
        }
    }
}

impl<T: BlockMetadata> std::fmt::Debug for WeakBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakBlock")
            .field("sequence_hash", &self.sequence_hash)
            .finish()
    }
}
