// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! RAII guards for blocks in the **Registered** state.
//!
//! [`ImmutableBlock`] is the strong, cloneable handle that keeps a registered
//! block alive. [`WeakBlock`] is its non-owning counterpart -- it does not
//! prevent the block from being evicted, but can be cheaply upgraded back to
//! an `ImmutableBlock` if the block is still present.

use super::{
    BlockId, BlockMetadata, BlockRegistrationHandle, RegisteredBlock, SequenceHash, UpgradeFn,
};

use crate::metrics::BlockPoolMetrics;
use std::sync::{Arc, Weak};

/// RAII guard for a block in the **Registered** state.
///
/// An `ImmutableBlock` is the primary handle through which callers interact
/// with registered blocks. It is reference-counted (`Clone`-able) and each
/// clone independently tracks the `inflight_immutable` metric gauge, so the
/// gauge reflects the total number of outstanding references across the
/// system.
///
/// # Obtaining an `ImmutableBlock`
///
/// - [`BlockManager::register_block`](crate::manager::BlockManager::register_block)
///   -- registers a [`CompleteBlock`](super::CompleteBlock) and returns an
///   `ImmutableBlock`.
/// - [`BlockManager::match_blocks`](crate::manager::BlockManager::match_blocks)
///   / [`BlockManager::scan_matches`](crate::manager::BlockManager::scan_matches)
///   -- look up already-registered blocks by [`SequenceHash`].
/// - [`WeakBlock::upgrade`] -- resurrects a weak reference if the block is
///   still alive.
///
/// # State transitions
///
/// - [`downgrade`](Self::downgrade) -- creates a [`WeakBlock`] that does not
///   keep the block alive.
///
/// # Clone behaviour
///
/// Cloning an `ImmutableBlock` increments `inflight_immutable`; dropping a
/// clone decrements it. The underlying registered block is shared via
/// `Arc`, so clones are cheap.
///
/// # Drop behaviour
///
/// Dropping the last strong reference (including internal pool references)
/// triggers the block's return to the inactive or reset pool. Every drop
/// decrements the `inflight_immutable` gauge.
pub struct ImmutableBlock<T: BlockMetadata> {
    block: Arc<dyn RegisteredBlock<T>>,
    upgrade_fn: UpgradeFn<T>,
    metrics: Option<Arc<BlockPoolMetrics>>,
}

/// Non-owning reference to a registered block.
///
/// A `WeakBlock` does not keep the underlying block alive -- if all
/// [`ImmutableBlock`] handles (and internal pool references) are dropped,
/// the block may be evicted and the weak reference will fail to upgrade.
///
/// Created via [`ImmutableBlock::downgrade`]. Cloneable and cheap to hold.
///
/// Call [`upgrade`](Self::upgrade) to attempt to recover a full
/// [`ImmutableBlock`].
#[derive(Clone)]
pub struct WeakBlock<T: BlockMetadata> {
    sequence_hash: SequenceHash,
    block: Weak<dyn RegisteredBlock<T>>,
    upgrade_fn: UpgradeFn<T>,
    metrics: Option<Arc<BlockPoolMetrics>>,
}

impl<T: BlockMetadata> ImmutableBlock<T> {
    /// Create a new ImmutableBlock with an upgrade function
    pub(crate) fn new(
        block: Arc<dyn RegisteredBlock<T>>,
        upgrade_fn: UpgradeFn<T>,
        metrics: Option<Arc<BlockPoolMetrics>>,
    ) -> Self {
        if let Some(ref m) = metrics {
            m.inc_inflight_immutable();
        }
        Self {
            block,
            upgrade_fn,
            metrics,
        }
    }

    /// Creates a [`WeakBlock`] that references the same registered block
    /// without preventing it from being evicted.
    pub fn downgrade(&self) -> WeakBlock<T> {
        WeakBlock {
            sequence_hash: self.sequence_hash(),
            block: Arc::downgrade(&self.block),
            upgrade_fn: self.upgrade_fn.clone(),
            metrics: self.metrics.clone(),
        }
    }

    /// Returns the [`BlockId`] assigned to this block.
    pub fn block_id(&self) -> BlockId {
        self.block.block_id()
    }

    /// Returns the [`SequenceHash`] that identifies this block's content.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.block.sequence_hash()
    }

    /// Returns a clone of the [`BlockRegistrationHandle`] for this block.
    pub fn registration_handle(&self) -> BlockRegistrationHandle {
        self.block.registration_handle().clone()
    }

    /// Returns the number of strong (`Arc`) references to the underlying
    /// registered block, including internal pool references.
    pub fn use_count(&self) -> usize {
        Arc::strong_count(&self.block)
    }
}

impl<T: BlockMetadata> Clone for ImmutableBlock<T> {
    fn clone(&self) -> Self {
        if let Some(ref m) = self.metrics {
            m.inc_inflight_immutable();
        }
        Self {
            block: self.block.clone(),
            upgrade_fn: self.upgrade_fn.clone(),
            metrics: self.metrics.clone(),
        }
    }
}

impl<T: BlockMetadata> Drop for ImmutableBlock<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(ref m) = self.metrics {
            m.dec_inflight_immutable();
        }
    }
}

impl<T: BlockMetadata> std::fmt::Debug for ImmutableBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImmutableBlock")
            .field("block_id", &self.block_id())
            .field("sequence_hash", &self.sequence_hash())
            .finish()
    }
}

impl<T: BlockMetadata> WeakBlock<T> {
    /// Attempts to upgrade this weak reference back to an [`ImmutableBlock`].
    ///
    /// Uses a two-phase strategy:
    /// 1. Tries a direct `Weak::upgrade` on the stored pointer (fast path).
    /// 2. Falls back to searching the
    ///    [`BlockRegistry`](crate::registry::BlockRegistry) by
    ///    [`SequenceHash`] in case the block was moved between pools.
    ///
    /// Returns `None` if the block has been fully evicted.
    pub fn upgrade(&self) -> Option<ImmutableBlock<T>> {
        // Fast path: direct weak upgrade
        if let Some(block) = self.block.upgrade() {
            return Some(ImmutableBlock::new(
                block,
                self.upgrade_fn.clone(),
                self.metrics.clone(),
            ));
        }

        // Slow path: search the registry by sequence hash
        if let Some(block) = (self.upgrade_fn)(self.sequence_hash) {
            return Some(ImmutableBlock::new(
                block,
                self.upgrade_fn.clone(),
                self.metrics.clone(),
            ));
        }

        None
    }

    /// Returns the [`SequenceHash`] for the block this weak reference
    /// points to.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.sequence_hash
    }
}

impl<T: BlockMetadata> std::fmt::Debug for WeakBlock<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeakBlock")
            .field("sequence_hash", &self.sequence_hash())
            .finish()
    }
}
