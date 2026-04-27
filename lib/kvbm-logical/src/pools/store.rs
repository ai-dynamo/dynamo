// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-mutex block store: unified bookkeeping for reset and inactive pools.
//!
//! `BlockStore<T>` owns the entire block bookkeeping for a single metadata tier:
//! a contiguous `Vec<BlockSlot>` indexed by `BlockId`, plus the index
//! collections that implement reset-pool and inactive-pool semantics. All
//! transitions happen under one `parking_lot::Mutex`, eliminating the
//! cross-mutex race that the dual-weak-ref `WeakBlockEntry` resurrection
//! scheme was working around.
//!
//! # Lock ordering
//!
//! `BlockRegistrationHandle.attachments` (Mutex inside the registry) → `BlockStore.inner` (Mutex).
//! Never the reverse.
//!
//! # Status
//!
//! Skeleton only — types and slot-level transitions are defined here; wiring
//! into the guards and `BlockManager` happens in subsequent commits.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::BlockId;
use crate::blocks::{BlockMetadata, SequenceHash};
use crate::metrics::BlockPoolMetrics;
use crate::registry::BlockRegistrationHandle;

/// Index trait for inactive-pool eviction backends.
///
/// `T`-free replacement for the legacy `InactivePoolBackend<T>` — backends
/// only need `(SequenceHash, BlockId)` pairs; the slot's typed payload lives
/// in the `BlockStore`.
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

    /// Remove a specific `block_id`/`seq_hash` pair if present (used during
    /// resurrection: pulls a block out of the inactive index without
    /// disturbing eviction order).
    fn take(&mut self, seq_hash: SequenceHash, block_id: BlockId) -> bool;

    /// Drain the entire index.
    fn allocate_all(&mut self) -> Vec<(SequenceHash, BlockId)> {
        let n = self.len();
        self.allocate(n)
    }
}

/// State of an individual slot. The variant identifies which pool currently
/// "owns" the slot (or which guard kind has it checked out).
#[derive(Debug)]
pub(crate) enum SlotState {
    /// Slot index is in `free`; nothing else holds it.
    Reset,
    /// A guard owns this slot. The variant determines the drop transition.
    CheckedOut(CheckedOutKind),
    /// Slot index is in the inactive index, available for eviction or resurrection.
    Inactive {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
}

/// Sub-state for a checked-out slot — tells the store where the slot should
/// go when its guard is dropped.
#[derive(Debug)]
pub(crate) enum CheckedOutKind {
    /// Held by `MutableBlock`. Drop → return to `free`.
    Mutable,
    /// Held by `CompleteBlock`. Drop → reset and return to `free`.
    Staged { seq_hash: SequenceHash },
    /// Held by an `ImmutableBlock` whose inner is the canonical primary.
    /// Drop of last clone → transition to `Inactive`.
    Primary {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
    /// Held by an `ImmutableBlock` whose inner is a duplicate physical copy.
    /// Drop of last clone → `mark_absent` + return to `free`.
    Duplicate {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
}

#[derive(Debug)]
pub(crate) struct BlockSlot {
    pub(crate) block_size: usize,
    pub(crate) state: SlotState,
}

impl BlockSlot {
    fn new_reset(block_size: usize) -> Self {
        Self {
            block_size,
            state: SlotState::Reset,
        }
    }
}

/// Inner state of a `BlockStore` — protected by a single mutex.
pub(crate) struct BlockStoreInner {
    /// `slots[block_id as usize]` — created at construction, never grows.
    slots: Vec<BlockSlot>,
    /// Free list (reset pool). FIFO via `pop_front`/`push_back`.
    free: VecDeque<BlockId>,
    /// Inactive eviction index.
    inactive: Box<dyn InactiveIndex>,
}

/// Single-mutex bookkeeping store for the reset and inactive pools.
pub(crate) struct BlockStore<T: BlockMetadata> {
    inner: Arc<Mutex<BlockStoreInner>>,
    block_size: usize,
    metrics: Option<Arc<BlockPoolMetrics>>,
    _marker: PhantomData<T>,
}

impl<T: BlockMetadata> BlockStore<T> {
    /// Create a new store with `total_blocks` slots in the Reset state.
    pub(crate) fn new(
        total_blocks: usize,
        block_size: usize,
        inactive: Box<dyn InactiveIndex>,
        metrics: Option<Arc<BlockPoolMetrics>>,
    ) -> Self {
        let mut slots = Vec::with_capacity(total_blocks);
        let mut free = VecDeque::with_capacity(total_blocks);
        for i in 0..total_blocks {
            slots.push(BlockSlot::new_reset(block_size));
            free.push_back(i as BlockId);
        }
        if let Some(ref m) = metrics {
            for _ in 0..total_blocks {
                m.inc_reset_pool_size();
            }
        }
        Self {
            inner: Arc::new(Mutex::new(BlockStoreInner {
                slots,
                free,
                inactive,
            })),
            block_size,
            metrics,
            _marker: PhantomData,
        }
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn metrics(&self) -> Option<Arc<BlockPoolMetrics>> {
        self.metrics.clone()
    }

    /// Number of slots currently in the reset (free) list.
    pub(crate) fn reset_len(&self) -> usize {
        self.inner.lock().free.len()
    }

    /// Number of slots currently in the inactive index.
    pub(crate) fn inactive_len(&self) -> usize {
        self.inner.lock().inactive.len()
    }

    /// True if the inactive index has a block under this hash.
    pub(crate) fn has_inactive(&self, seq_hash: SequenceHash) -> bool {
        self.inner.lock().inactive.has(seq_hash)
    }

    // ---------- slot transitions ----------

    /// Pop up to `count` `BlockId`s from the reset pool, transitioning each
    /// slot into `CheckedOut(Mutable)`. Caller is responsible for wrapping
    /// the IDs in guards.
    pub(crate) fn checkout_reset(&self, count: usize) -> Vec<BlockId> {
        let mut inner = self.inner.lock();
        let take = std::cmp::min(count, inner.free.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            let id = inner.free.pop_front().unwrap();
            inner.slots[id as usize].state = SlotState::CheckedOut(CheckedOutKind::Mutable);
            if let Some(ref m) = self.metrics {
                m.dec_reset_pool_size();
            }
            out.push(id);
        }
        out
    }

    /// Return a slot from any `CheckedOut(*)` state to `Reset`. Pushes the
    /// id back onto the free list.
    pub(crate) fn return_to_reset(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(
            inner.slots[block_id as usize].state,
            SlotState::CheckedOut(_)
        ));
        inner.slots[block_id as usize].state = SlotState::Reset;
        inner.free.push_back(block_id);
        if let Some(ref m) = self.metrics {
            m.inc_reset_pool_size();
        }
    }

    /// Transition `block_id` from `CheckedOut(Mutable)` to `CheckedOut(Staged)`.
    pub(crate) fn mark_staged(&self, block_id: BlockId, seq_hash: SequenceHash) {
        let mut inner = self.inner.lock();
        let slot = &mut inner.slots[block_id as usize];
        debug_assert!(matches!(
            slot.state,
            SlotState::CheckedOut(CheckedOutKind::Mutable)
        ));
        slot.state = SlotState::CheckedOut(CheckedOutKind::Staged { seq_hash });
    }

    /// Transition `block_id` from `CheckedOut(Staged)` back to `CheckedOut(Mutable)`.
    pub(crate) fn unmark_staged(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        let slot = &mut inner.slots[block_id as usize];
        debug_assert!(matches!(
            slot.state,
            SlotState::CheckedOut(CheckedOutKind::Staged { .. })
        ));
        slot.state = SlotState::CheckedOut(CheckedOutKind::Mutable);
    }

    /// Transition `block_id` from `CheckedOut(Staged)` to
    /// `CheckedOut(Primary { .. })` and call `mark_present::<T>`. Returns
    /// the cached seq_hash.
    pub(crate) fn mark_primary(
        &self,
        block_id: BlockId,
        handle: BlockRegistrationHandle,
    ) -> SequenceHash {
        // mark_present uses the registry attachments lock — must happen
        // before we acquire the store lock to honour the documented order.
        handle.mark_present::<T>();
        let seq_hash = handle.seq_hash();
        let mut inner = self.inner.lock();
        let slot = &mut inner.slots[block_id as usize];
        debug_assert!(matches!(
            slot.state,
            SlotState::CheckedOut(CheckedOutKind::Staged { .. })
        ));
        slot.state = SlotState::CheckedOut(CheckedOutKind::Primary { seq_hash, handle });
        seq_hash
    }

    /// Transition `block_id` from `CheckedOut(Staged)` to
    /// `CheckedOut(Duplicate { .. })` and call `mark_present::<T>`.
    pub(crate) fn mark_duplicate(
        &self,
        block_id: BlockId,
        handle: BlockRegistrationHandle,
    ) -> SequenceHash {
        handle.mark_present::<T>();
        let seq_hash = handle.seq_hash();
        let mut inner = self.inner.lock();
        let slot = &mut inner.slots[block_id as usize];
        debug_assert!(matches!(
            slot.state,
            SlotState::CheckedOut(CheckedOutKind::Staged { .. })
        ));
        slot.state = SlotState::CheckedOut(CheckedOutKind::Duplicate { seq_hash, handle });
        seq_hash
    }

    /// Drop transition for the last clone of a primary `ImmutableBlockInner`:
    /// `CheckedOut(Primary)` → `Inactive` (block becomes evictable).
    pub(crate) fn primary_to_inactive(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        let slot = &mut inner.slots[block_id as usize];
        let (seq_hash, handle) = match std::mem::replace(&mut slot.state, SlotState::Reset) {
            SlotState::CheckedOut(CheckedOutKind::Primary { seq_hash, handle }) => {
                (seq_hash, handle)
            }
            other => panic!("primary_to_inactive: unexpected state {other:?}"),
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
    /// `CheckedOut(Duplicate)` → `Reset`. Calls `mark_absent::<T>` on the
    /// registration handle.
    pub(crate) fn duplicate_to_reset(&self, block_id: BlockId) {
        // Must extract handle before locking attachments to honour ordering.
        let handle = {
            let mut inner = self.inner.lock();
            let slot = &mut inner.slots[block_id as usize];
            let handle = match std::mem::replace(&mut slot.state, SlotState::Reset) {
                SlotState::CheckedOut(CheckedOutKind::Duplicate { handle, .. }) => handle,
                other => panic!("duplicate_to_reset: unexpected state {other:?}"),
            };
            inner.free.push_back(block_id);
            if let Some(ref m) = self.metrics {
                m.inc_reset_pool_size();
            }
            handle
        };
        handle.mark_absent::<T>();
    }

    // ---------- inactive-pool operations ----------

    /// Find blocks by hash (stop on first miss). Slots transition
    /// `Inactive` → `CheckedOut(Primary)`. Returns the matched
    /// `(seq_hash, block_id, handle)` triples; caller wraps in guards.
    pub(crate) fn find_inactive(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId, BlockRegistrationHandle)> {
        let mut inner = self.inner.lock();
        let matched = inner.inactive.find_matches(hashes, touch);
        let mut out = Vec::with_capacity(matched.len());
        for (seq_hash, block_id) in matched {
            let slot = &mut inner.slots[block_id as usize];
            let handle = match std::mem::replace(&mut slot.state, SlotState::Reset) {
                SlotState::Inactive { handle, .. } => handle,
                other => panic!("find_inactive: slot for {block_id} was {other:?}"),
            };
            slot.state = SlotState::CheckedOut(CheckedOutKind::Primary {
                seq_hash,
                handle: handle.clone(),
            });
            if let Some(ref m) = self.metrics {
                m.dec_inactive_pool_size();
            }
            out.push((seq_hash, block_id, handle));
        }
        out
    }

    /// Scan-style find — does not stop on miss.
    pub(crate) fn scan_inactive(
        &self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId, BlockRegistrationHandle)> {
        let mut inner = self.inner.lock();
        let matched = inner.inactive.scan_matches(hashes, touch);
        let mut out = Vec::with_capacity(matched.len());
        for (seq_hash, block_id) in matched {
            let slot = &mut inner.slots[block_id as usize];
            let handle = match std::mem::replace(&mut slot.state, SlotState::Reset) {
                SlotState::Inactive { handle, .. } => handle,
                other => panic!("scan_inactive: slot for {block_id} was {other:?}"),
            };
            slot.state = SlotState::CheckedOut(CheckedOutKind::Primary {
                seq_hash,
                handle: handle.clone(),
            });
            if let Some(ref m) = self.metrics {
                m.dec_inactive_pool_size();
            }
            out.push((seq_hash, block_id, handle));
        }
        out
    }

    /// Resurrect a single inactive block by hash, if present. Slot transitions
    /// `Inactive` → `CheckedOut(Primary)`. Returns `(block_id, handle)`.
    pub(crate) fn resurrect_primary(
        &self,
        seq_hash: SequenceHash,
    ) -> Option<(BlockId, BlockRegistrationHandle)> {
        let mut inner = self.inner.lock();
        let mut matched = inner.inactive.find_matches(&[seq_hash], false);
        let (hash, block_id) = matched.pop()?;
        debug_assert_eq!(hash, seq_hash);
        let slot = &mut inner.slots[block_id as usize];
        let handle = match std::mem::replace(&mut slot.state, SlotState::Reset) {
            SlotState::Inactive { handle, .. } => handle,
            other => panic!("resurrect_primary: slot for {block_id} was {other:?}"),
        };
        slot.state = SlotState::CheckedOut(CheckedOutKind::Primary {
            seq_hash,
            handle: handle.clone(),
        });
        if let Some(ref m) = self.metrics {
            m.dec_inactive_pool_size();
        }
        Some((block_id, handle))
    }

    /// Evict up to `count` blocks from the inactive pool, transitioning each
    /// to `CheckedOut(Mutable)`. Returns `(block_id, evicted_seq_hash)` pairs.
    /// Returns `None` if fewer than `count` are available.
    pub(crate) fn evict_for_reset(
        &self,
        count: usize,
    ) -> Option<Vec<(BlockId, SequenceHash)>> {
        if count == 0 {
            return Some(Vec::new());
        }
        let mut inner = self.inner.lock();
        if inner.inactive.len() < count {
            return None;
        }
        let evicted = inner.inactive.allocate(count);
        if evicted.len() != count {
            // Re-insert what we got and bail out.
            for (h, id) in evicted {
                inner.inactive.insert(h, id);
            }
            return None;
        }

        let mut handles = Vec::with_capacity(count);
        let mut out = Vec::with_capacity(count);
        for (seq_hash, block_id) in evicted {
            let slot = &mut inner.slots[block_id as usize];
            let handle = match std::mem::replace(&mut slot.state, SlotState::Reset) {
                SlotState::Inactive { handle, .. } => handle,
                other => panic!("evict_for_reset: slot for {block_id} was {other:?}"),
            };
            slot.state = SlotState::CheckedOut(CheckedOutKind::Mutable);
            if let Some(ref m) = self.metrics {
                m.dec_inactive_pool_size();
            }
            handles.push(handle);
            out.push((block_id, seq_hash));
        }
        drop(inner);
        // mark_absent calls take the attachments lock — outside the store lock.
        for h in handles {
            h.mark_absent::<T>();
        }
        Some(out)
    }

    /// Drain the inactive pool entirely, transitioning every slot to
    /// `CheckedOut(Mutable)`. Returns the list of `BlockId`s.
    pub(crate) fn drain_inactive_to_reset(&self) -> Vec<BlockId> {
        let mut inner = self.inner.lock();
        let drained = inner.inactive.allocate_all();

        let mut handles = Vec::with_capacity(drained.len());
        let mut ids = Vec::with_capacity(drained.len());
        for (_seq_hash, block_id) in drained {
            let slot = &mut inner.slots[block_id as usize];
            let handle = match std::mem::replace(&mut slot.state, SlotState::Reset) {
                SlotState::Inactive { handle, .. } => handle,
                other => panic!("drain_inactive_to_reset: slot for {block_id} was {other:?}"),
            };
            slot.state = SlotState::CheckedOut(CheckedOutKind::Mutable);
            if let Some(ref m) = self.metrics {
                m.dec_inactive_pool_size();
            }
            handles.push(handle);
            ids.push(block_id);
        }
        drop(inner);
        for h in handles {
            h.mark_absent::<T>();
        }
        ids
    }
}

impl<T: BlockMetadata> std::fmt::Debug for BlockStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockStore")
            .field("block_size", &self.block_size)
            .field("reset_len", &self.reset_len())
            .field("inactive_len", &self.inactive_len())
            .finish()
    }
}
