// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-mutex block store: unified bookkeeping for reset and inactive pools.
//!
//! `BlockStore<T>` owns the entire block bookkeeping for a single metadata
//! tier. A contiguous `Vec<BlockSlot>` indexed by `BlockId` is the source of
//! truth for every slot's state. The reset pool is a `VecDeque<BlockId>` over
//! slots in `Reset` state; the inactive pool's eviction order is owned by a
//! pluggable [`InactiveIndex`] backend over slots in `Inactive` state. All
//! transitions happen under one `parking_lot::Mutex`, so the dual-weak-ref
//! resurrection scheme that the previous design needed is no longer
//! necessary.
//!
//! # Lock ordering
//!
//! `BlockRegistrationHandle.attachments` (Mutex inside the registry) →
//! `BlockStore.inner` (Mutex). Never the reverse.

use std::any::TypeId;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::BlockId;
use crate::blocks::{BlockMetadata, ImmutableBlockInner, MutableBlock, SequenceHash};
use crate::metrics::BlockPoolMetrics;
use crate::registry::BlockRegistrationHandle;

/// Index trait for inactive-pool eviction backends. T-free: backends only
/// need `(SequenceHash, BlockId)` pairs.
#[allow(dead_code)]
pub(crate) trait InactiveIndex: Send + Sync {
    /// Find blocks for the given hashes in order, stopping on first miss.
    /// Removes matched entries from the index.
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId)>;

    /// Like `find_matches` but does not stop on miss.
    fn scan_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId)>;

    /// Pull `count` blocks for eviction in policy order.
    fn allocate(&mut self, count: usize) -> Vec<(SequenceHash, BlockId)>;

    /// Make `block_id` evictable under `seq_hash`.
    fn insert(&mut self, seq_hash: SequenceHash, block_id: BlockId);

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn has(&self, seq_hash: SequenceHash) -> bool;

    /// Remove a specific `block_id`/`seq_hash` pair if present.
    #[allow(dead_code)]
    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool;

    /// Drain the entire index.
    fn allocate_all(&mut self) -> Vec<(SequenceHash, BlockId)> {
        let n = self.len();
        self.allocate(n)
    }
}

/// State of an individual slot. The variant determines all drop transitions
/// and resurrection semantics. Tracked under the unified store mutex.
///
/// `seq_hash` fields on the non-`Primary` variants are kept for Debug
/// output and future invariant checks; only `Primary`'s `seq_hash` is
/// actually read (during the `release_primary` insert into the inactive
/// index).
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum SlotState {
    /// In the `free` list; available for allocation.
    Reset,
    /// Held by a `MutableBlock`. Drop → `Reset`.
    Mutable,
    /// Held by a `CompleteBlock`. Drop → `Reset`.
    Staged { seq_hash: SequenceHash },
    /// Held by an `ImmutableBlock` whose inner is the canonical primary.
    /// Drop of last clone → `Inactive`.
    Primary {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
    /// Held by an `ImmutableBlock` whose inner is a duplicate physical copy.
    /// Drop of last clone → `Reset` (with `mark_absent`).
    Duplicate {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
    /// Idle, evictable, registered. In the inactive index under `seq_hash`.
    Inactive {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
}

#[derive(Debug)]
pub(crate) struct BlockSlot {
    pub(crate) block_size: usize,
    pub(crate) state: SlotState,
}

/// Inner state of a `BlockStore` — protected by a single mutex.
pub(crate) struct BlockStoreInner {
    /// `slots[block_id]` — created at construction, never grows.
    slots: Vec<BlockSlot>,
    /// Free list (reset pool). FIFO.
    free: VecDeque<BlockId>,
    /// Inactive eviction index (T-free).
    inactive: Box<dyn InactiveIndex>,
}

/// Single-mutex bookkeeping store for the reset and inactive pools.
pub(crate) struct BlockStore<T: BlockMetadata> {
    inner: Arc<Mutex<BlockStoreInner>>,
    block_size: usize,
    total_blocks: usize,
    metrics: Option<Arc<BlockPoolMetrics>>,
    _marker: PhantomData<T>,
}

#[allow(dead_code)]
impl<T: BlockMetadata + Sync> BlockStore<T> {
    pub(crate) fn new(
        total_blocks: usize,
        block_size: usize,
        inactive: Box<dyn InactiveIndex>,
        metrics: Option<Arc<BlockPoolMetrics>>,
    ) -> Arc<Self> {
        let mut slots = Vec::with_capacity(total_blocks);
        let mut free = VecDeque::with_capacity(total_blocks);
        for i in 0..total_blocks {
            slots.push(BlockSlot {
                block_size,
                state: SlotState::Reset,
            });
            free.push_back(i);
        }
        let inner = Arc::new(Mutex::new(BlockStoreInner {
            slots,
            free,
            inactive,
        }));
        Arc::new(Self {
            inner,
            block_size,
            total_blocks,
            metrics,
            _marker: PhantomData,
        })
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub(crate) fn metrics(&self) -> Option<&Arc<BlockPoolMetrics>> {
        self.metrics.as_ref()
    }

    pub(crate) fn reset_len(&self) -> usize {
        self.inner.lock().free.len()
    }

    pub(crate) fn inactive_len(&self) -> usize {
        self.inner.lock().inactive.len()
    }

    pub(crate) fn has_inactive(&self, seq_hash: SequenceHash) -> bool {
        self.inner.lock().inactive.has(seq_hash)
    }

    pub(crate) fn slot_block_size(&self, block_id: BlockId) -> usize {
        self.inner.lock().slots[block_id].block_size
    }

    // ---------- guard construction ----------

    /// Allocate up to `count` MutableBlocks from the reset pool.
    pub(crate) fn allocate_reset_blocks(self: &Arc<Self>, count: usize) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.lock();
        let take = std::cmp::min(count, inner.free.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            let id = inner.free.pop_front().unwrap();
            inner.slots[id].state = SlotState::Mutable;
            if let Some(ref m) = self.metrics {
                m.dec_reset_pool_size();
            }
            let block_size = inner.slots[id].block_size;
            out.push(MutableBlock::from_store(
                self.clone(),
                id,
                block_size,
                self.metrics.clone(),
            ));
        }
        out
    }

    /// Evict up to `count` blocks from the inactive pool, transitioning each
    /// slot back to `Mutable`. Reports the evicted seq_hashes.
    pub(crate) fn evict_to_mutable(
        self: &Arc<Self>,
        count: usize,
    ) -> Option<(Vec<MutableBlock<T>>, Vec<SequenceHash>)> {
        if count == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let mut inner = self.inner.lock();
        if inner.inactive.len() < count {
            return None;
        }
        let evicted_pairs = inner.inactive.allocate(count);
        if evicted_pairs.len() != count {
            for (h, id) in evicted_pairs {
                inner.inactive.insert(h, id);
            }
            return None;
        }
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }

        let mut out = Vec::with_capacity(count);
        let mut evicted = Vec::with_capacity(count);
        let mut handles = Vec::with_capacity(count);
        for (seq_hash, block_id) in evicted_pairs {
            let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
            inner.slots[block_id].state = SlotState::Mutable;
            evicted.push(seq_hash);
            handles.push(handle);
            let block_size = inner.slots[block_id].block_size;
            out.push(MutableBlock::from_store(
                self.clone(),
                block_id,
                block_size,
                self.metrics.clone(),
            ));
        }
        drop(inner);
        // mark_absent::<T> takes the registry attachments lock — invoke
        // outside the store lock to honour the documented ordering.
        for h in handles {
            h.mark_absent::<T>();
        }
        Some((out, evicted))
    }

    /// Drain the inactive pool entirely into MutableBlocks.
    pub(crate) fn drain_inactive_to_mutable(self: &Arc<Self>) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.lock();
        let drained = inner.inactive.allocate_all();
        let count = drained.len();
        if let Some(ref m) = self.metrics {
            for _ in 0..count {
                m.dec_inactive_pool_size();
            }
        }
        let mut handles = Vec::with_capacity(count);
        let mut out = Vec::with_capacity(count);
        for (_seq_hash, block_id) in drained {
            let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
            inner.slots[block_id].state = SlotState::Mutable;
            handles.push(handle);
            let block_size = inner.slots[block_id].block_size;
            out.push(MutableBlock::from_store(
                self.clone(),
                block_id,
                block_size,
                self.metrics.clone(),
            ));
        }
        drop(inner);
        for h in handles {
            h.mark_absent::<T>();
        }
        out
    }

    /// Promote up to `hashes.len()` inactive slots to `Primary`, building
    /// fresh `ImmutableBlockInner`s. Stops on first miss.
    pub(crate) fn find_inactive_primaries(
        self: &Arc<Self>,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<Arc<ImmutableBlockInner<T>>> {
        let promoted = self.promote_inactive(hashes, touch, /*scan*/ false);
        promoted
            .into_iter()
            .map(|(seq_hash, block_id, handle)| {
                let inner_arc = ImmutableBlockInner::new_primary(
                    self.clone(),
                    block_id,
                    seq_hash,
                    handle.clone(),
                );
                store_weak_in_handle::<T>(&handle, &inner_arc);
                inner_arc
            })
            .collect()
    }

    /// Scan-style version of [`find_inactive_primaries`] — does not stop on miss.
    pub(crate) fn scan_inactive_primaries(
        self: &Arc<Self>,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<ImmutableBlockInner<T>>)> {
        let promoted = self.promote_inactive(hashes, touch, /*scan*/ true);
        promoted
            .into_iter()
            .map(|(seq_hash, block_id, handle)| {
                let inner_arc = ImmutableBlockInner::new_primary(
                    self.clone(),
                    block_id,
                    seq_hash,
                    handle.clone(),
                );
                store_weak_in_handle::<T>(&handle, &inner_arc);
                (seq_hash, inner_arc)
            })
            .collect()
    }

    /// Resurrect a single inactive block by hash, returning the new inner
    /// without storing its Weak in the registry attachments. The caller is
    /// responsible for calling [`store_weak_in_handle`] (or doing the
    /// equivalent) when it is safe to acquire the attachments lock.
    pub(crate) fn resurrect_inactive_no_attach(
        self: &Arc<Self>,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<(Arc<ImmutableBlockInner<T>>, BlockRegistrationHandle)> {
        let (block_id, handle) = {
            let mut inner = self.inner.lock();
            let mut matched = inner.inactive.find_matches(&[seq_hash], touch);
            let (_, block_id) = matched.pop()?;
            if let Some(ref m) = self.metrics {
                m.dec_inactive_pool_size();
            }
            let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
            inner.slots[block_id].state = SlotState::Primary {
                seq_hash,
                handle: handle.clone(),
            };
            (block_id, handle)
        };
        let inner_arc =
            ImmutableBlockInner::new_primary(self.clone(), block_id, seq_hash, handle.clone());
        Some((inner_arc, handle))
    }

    /// Common slot-transition core for find/scan inactive promotions.
    fn promote_inactive(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
        scan: bool,
    ) -> Vec<(SequenceHash, BlockId, BlockRegistrationHandle)> {
        let mut inner = self.inner.lock();
        let matched = if scan {
            inner.inactive.scan_matches(hashes, touch)
        } else {
            inner.inactive.find_matches(hashes, touch)
        };
        if let Some(ref m) = self.metrics {
            for _ in 0..matched.len() {
                m.dec_inactive_pool_size();
            }
        }
        matched
            .into_iter()
            .map(|(seq_hash, block_id)| {
                let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
                inner.slots[block_id].state = SlotState::Primary {
                    seq_hash,
                    handle: handle.clone(),
                };
                (seq_hash, block_id, handle)
            })
            .collect()
    }

    // ---------- guard transitions (called from guard methods / drops) ----------

    /// `Mutable` → `Reset` (MutableBlock dropped without a transition).
    pub(crate) fn release_mutable(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(inner.slots[block_id].state, SlotState::Mutable));
        inner.slots[block_id].state = SlotState::Reset;
        inner.free.push_back(block_id);
        if let Some(ref m) = self.metrics {
            m.inc_reset_pool_size();
        }
    }

    /// `Mutable` → `Staged` (MutableBlock::stage / ::complete).
    pub(crate) fn transition_to_staged(&self, block_id: BlockId, seq_hash: SequenceHash) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(inner.slots[block_id].state, SlotState::Mutable));
        inner.slots[block_id].state = SlotState::Staged { seq_hash };
    }

    /// `Staged` → `Mutable` (CompleteBlock::reset).
    pub(crate) fn transition_back_to_mutable(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(
            inner.slots[block_id].state,
            SlotState::Staged { .. }
        ));
        inner.slots[block_id].state = SlotState::Mutable;
    }

    /// `Staged` → `Reset` (CompleteBlock dropped without a transition).
    pub(crate) fn release_staged(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(
            inner.slots[block_id].state,
            SlotState::Staged { .. }
        ));
        inner.slots[block_id].state = SlotState::Reset;
        inner.free.push_back(block_id);
        if let Some(ref m) = self.metrics {
            m.inc_reset_pool_size();
        }
    }

    /// `Staged` → `Primary` (BlockManager::register_block, fresh primary).
    /// Calls `mark_present::<T>` on the handle and stores the Weak.
    pub(crate) fn transition_to_primary(
        self: &Arc<Self>,
        block_id: BlockId,
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    ) -> Arc<ImmutableBlockInner<T>> {
        // mark_present takes the attachments lock — invoke before locking
        // the store to honour ordering.
        handle.mark_present::<T>();
        {
            let mut inner = self.inner.lock();
            debug_assert!(matches!(
                inner.slots[block_id].state,
                SlotState::Staged { .. }
            ));
            inner.slots[block_id].state = SlotState::Primary {
                seq_hash,
                handle: handle.clone(),
            };
        }
        let inner_arc =
            ImmutableBlockInner::new_primary(self.clone(), block_id, seq_hash, handle.clone());
        store_weak_in_handle::<T>(&handle, &inner_arc);
        inner_arc
    }

    /// `Staged` → `Duplicate` (BlockManager::register_block, dup-of-existing).
    pub(crate) fn transition_to_duplicate(
        self: &Arc<Self>,
        block_id: BlockId,
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
        primary_keepalive: Arc<ImmutableBlockInner<T>>,
    ) -> Arc<ImmutableBlockInner<T>> {
        handle.mark_present::<T>();
        {
            let mut inner = self.inner.lock();
            debug_assert!(matches!(
                inner.slots[block_id].state,
                SlotState::Staged { .. }
            ));
            inner.slots[block_id].state = SlotState::Duplicate {
                seq_hash,
                handle: handle.clone(),
            };
        }
        ImmutableBlockInner::new_duplicate(
            self.clone(),
            block_id,
            seq_hash,
            handle,
            primary_keepalive,
        )
    }

    /// Drop transition for the last clone of a primary `ImmutableBlockInner`:
    /// `Primary` → `Inactive`.
    pub(crate) fn release_primary(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        let slot = &mut inner.slots[block_id];
        let (seq_hash, handle) = match &slot.state {
            SlotState::Primary { seq_hash, handle } => (*seq_hash, handle.clone()),
            other => panic!("release_primary: slot {block_id} was {other:?}"),
        };
        slot.state = SlotState::Inactive {
            seq_hash,
            handle: handle.clone(),
        };
        inner.inactive.insert(seq_hash, block_id);
        if let Some(ref m) = self.metrics {
            m.inc_inactive_pool_size();
        }
        tracing::trace!(?seq_hash, block_id, "Block stored in inactive pool");
    }

    /// Drop transition for the last clone of a duplicate `ImmutableBlockInner`:
    /// `Duplicate` → `Reset` (with `mark_absent::<T>`).
    pub(crate) fn release_duplicate(&self, block_id: BlockId) {
        let handle = {
            let mut inner = self.inner.lock();
            let slot = &mut inner.slots[block_id];
            let handle = match &slot.state {
                SlotState::Duplicate { handle, .. } => handle.clone(),
                other => panic!("release_duplicate: slot {block_id} was {other:?}"),
            };
            slot.state = SlotState::Reset;
            inner.free.push_back(block_id);
            if let Some(ref m) = self.metrics {
                m.inc_reset_pool_size();
            }
            handle
        };
        handle.mark_absent::<T>();
    }
}

impl<T: BlockMetadata> std::fmt::Debug for BlockStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockStore")
            .field("block_size", &self.block_size)
            .field("total_blocks", &self.total_blocks)
            .finish()
    }
}

// ---------- helpers ----------

/// Clone the [`BlockRegistrationHandle`] out of an `Inactive` slot. The
/// caller must overwrite `slot.state` before releasing the store lock.
fn take_inactive_handle(slot: &mut BlockSlot, block_id: BlockId) -> BlockRegistrationHandle {
    match &slot.state {
        SlotState::Inactive { handle, .. } => handle.clone(),
        other => panic!("expected Inactive state for slot {block_id}, got {other:?}"),
    }
}

/// Store a `Weak<ImmutableBlockInner<T>>` in the handle's `weak_blocks`
/// attachment under `TypeId::of::<Weak<ImmutableBlockInner<T>>>()`.
///
/// Acquires the attachments lock — caller must NOT hold it.
pub(crate) fn store_weak_in_handle<T: BlockMetadata + Sync>(
    handle: &BlockRegistrationHandle,
    inner: &Arc<ImmutableBlockInner<T>>,
) {
    let type_id = TypeId::of::<Weak<ImmutableBlockInner<T>>>();
    let weak = Arc::downgrade(inner);
    let mut attachments = handle.inner.attachments.lock();
    attachments.weak_blocks.insert(type_id, Box::new(weak));
}

/// Hold the attachments lock while attempting to upgrade a Weak<inner> and,
/// if that fails, resurrect from the inactive pool. Stores the new Weak
/// while still holding the attachments lock to serialize parallel
/// resurrections of the same hash.
///
/// Lock order: attachments → store. Caller must NOT hold either lock.
pub(crate) fn upgrade_or_resurrect<T: BlockMetadata + Sync>(
    handle: &BlockRegistrationHandle,
    store: &Arc<BlockStore<T>>,
) -> Option<Arc<ImmutableBlockInner<T>>> {
    let type_id = TypeId::of::<Weak<ImmutableBlockInner<T>>>();
    let mut attachments = handle.inner.attachments.lock();

    // Fast path: upgrade existing Weak.
    if let Some(weak) = attachments
        .weak_blocks
        .get(&type_id)
        .and_then(|w| w.downcast_ref::<Weak<ImmutableBlockInner<T>>>())
        .cloned()
        && let Some(strong) = weak.upgrade()
    {
        return Some(strong);
    }

    // Slow path: resurrect from inactive (still holding attachments).
    let (inner_arc, _h) = store.resurrect_inactive_no_attach(handle.seq_hash(), false)?;
    attachments
        .weak_blocks
        .insert(type_id, Box::new(Arc::downgrade(&inner_arc)));
    Some(inner_arc)
}
