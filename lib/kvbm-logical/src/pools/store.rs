// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-mutex block store: unified bookkeeping for the reset, active,
//! and inactive pools.
//!
//! `BlockStore<T>` owns the entire block bookkeeping for a single
//! metadata tier. The unified mutex protects:
//!
//! - `slots: Vec<BlockSlot<T>>` — source of truth for every slot's state.
//! - `free: VecDeque<BlockId>` — reset pool (FIFO).
//! - `inactive: Box<dyn InactiveIndex>` — pluggable eviction-order index
//!   over slots in `Inactive` state.
//! - `active_by_hash: HashMap<SequenceHash, BlockId>` — primary block_id
//!   for each currently-registered hash.
//!
//! Active-pool lookup, slot transitions, and resurrection all happen
//! under one lock, so no across-lock gap can leave a hash unreachable
//! from both the active and inactive pools at the same time.
//!
//! # Lock ordering
//!
//! `BlockRegistrationHandle.attachments` (Mutex inside the registry) →
//! `BlockStore.inner` (Mutex). Never the reverse.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Weak};

// Under `#[cfg(test)]` use `tracing-mutex`'s parking_lot wrapper, which
// is API-identical to `parking_lot::Mutex` but builds a global
// lock-acquisition DAG and panics on order inversions or cycles. This
// turns the documented `attachments → store` ordering invariant into
// runtime-enforced behaviour during the test suite. In release/non-test
// builds the alias resolves to plain `parking_lot::Mutex` — zero cost.
#[cfg(not(test))]
use parking_lot::Mutex;
#[cfg(test)]
use tracing_mutex::parkinglot::Mutex;

use crate::BlockId;
use crate::blocks::{
    BlockDuplicationPolicy, BlockMetadata, CompleteBlock, ImmutableBlockInner, MutableBlock,
    SequenceHash,
};
use crate::metrics::BlockPoolMetrics;
use crate::registry::BlockRegistrationHandle;

/// Index trait for inactive-pool eviction backends. T-free: backends only
/// need `(SequenceHash, BlockId)` pairs.
pub(crate) trait InactiveIndex: Send + Sync {
    /// Find blocks for the given hashes in order, stopping on first miss.
    /// Removes matched entries from the index.
    fn find_matches(
        &mut self,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, BlockId)>;

    /// Find a single block matching `hash`. Default impl delegates to
    /// `find_matches`; backends override for an O(1) variant that
    /// avoids slice iteration on the single-block fast-path.
    fn find_match(&mut self, hash: SequenceHash, touch: bool) -> Option<(SequenceHash, BlockId)> {
        self.find_matches(&[hash], touch).into_iter().next()
    }

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

    #[allow(dead_code)]
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
/// `Primary`/`Duplicate` carry a `Weak<ImmutableBlockInner<T>>` so the
/// store can perform identity-checked drop transitions and serve active
/// lookups under the store mutex without consulting registry attachments.
#[allow(dead_code)]
pub(crate) enum SlotState<T: BlockMetadata> {
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
        inner: Weak<ImmutableBlockInner<T>>,
    },
    /// Held by an `ImmutableBlock` whose inner is a duplicate physical copy.
    /// Drop of last clone → `Reset` (with `mark_absent`).
    Duplicate {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
        inner: Weak<ImmutableBlockInner<T>>,
    },
    /// Idle, evictable, registered. In the inactive index under `seq_hash`.
    Inactive {
        seq_hash: SequenceHash,
        handle: BlockRegistrationHandle,
    },
}

impl<T: BlockMetadata> std::fmt::Debug for SlotState<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlotState::Reset => f.write_str("Reset"),
            SlotState::Mutable => f.write_str("Mutable"),
            SlotState::Staged { seq_hash } => f
                .debug_struct("Staged")
                .field("seq_hash", seq_hash)
                .finish(),
            SlotState::Primary { seq_hash, .. } => f
                .debug_struct("Primary")
                .field("seq_hash", seq_hash)
                .finish(),
            SlotState::Duplicate { seq_hash, .. } => f
                .debug_struct("Duplicate")
                .field("seq_hash", seq_hash)
                .finish(),
            SlotState::Inactive { seq_hash, .. } => f
                .debug_struct("Inactive")
                .field("seq_hash", seq_hash)
                .finish(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct BlockSlot<T: BlockMetadata> {
    pub(crate) block_size: usize,
    pub(crate) state: SlotState<T>,
}

/// Inner state of a `BlockStore` — protected by a single mutex.
pub(crate) struct BlockStoreInner<T: BlockMetadata> {
    /// `slots[block_id]` — created at construction, never grows.
    slots: Vec<BlockSlot<T>>,
    /// Free list (reset pool). FIFO.
    free: VecDeque<BlockId>,
    /// Inactive eviction index (T-free).
    inactive: Box<dyn InactiveIndex>,
    /// Primary `block_id` for each currently-registered sequence hash.
    /// Updated atomically with the slot's `Primary`/`Inactive` state.
    active_by_hash: HashMap<SequenceHash, BlockId>,
}

/// Single-mutex bookkeeping store for the reset, active, and inactive
/// pools.
pub(crate) struct BlockStore<T: BlockMetadata> {
    /// Stable, process-unique store identifier. Surfaced through
    /// `LifecyclePin::manager_id` so type-erased pins remain
    /// runtime-addressable to a unique physical pool.
    id: crate::ManagerId,
    inner: Mutex<BlockStoreInner<T>>,
    block_size: usize,
    total_blocks: usize,
    metrics: Arc<BlockPoolMetrics>,
    /// Test-only hook to deterministically widen the
    /// "Arc strong=0 but `release_primary` not yet run" race window.
    /// `release_primary` acquires this gate *before* taking the store
    /// mutex; while a test holds the gate via
    /// `pause_release_primary()`, every `release_primary` call parks
    /// here without ever inspecting slot state, leaving the slot in
    /// `Primary { weak: dead }` for a concurrent lookup to observe.
    /// Production builds elide the field entirely.
    #[cfg(test)]
    release_primary_gate: Mutex<()>,

    /// Test-only arrival counter. Incremented at the very first
    /// instruction of every `release_primary` call (before the gate
    /// is contended). Tests use it to signal "the dropping thread has
    /// reached `release_primary` and is about to park", replacing
    /// scheduler-dependent sleeps in deterministic race tests.
    #[cfg(test)]
    release_primary_arrivals: std::sync::atomic::AtomicU64,
}

#[allow(dead_code)]
impl<T: BlockMetadata + Sync> BlockStore<T> {
    pub(crate) fn new(
        total_blocks: usize,
        block_size: usize,
        inactive: Box<dyn InactiveIndex>,
        metrics: Arc<BlockPoolMetrics>,
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
        Arc::new(Self {
            id: crate::ManagerId::next(),
            inner: Mutex::new(BlockStoreInner {
                slots,
                free,
                inactive,
                active_by_hash: HashMap::new(),
            }),
            block_size,
            total_blocks,
            metrics,
            #[cfg(test)]
            release_primary_gate: Mutex::new(()),
            #[cfg(test)]
            release_primary_arrivals: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Stable, process-unique identifier of this store. See
    /// [`crate::ManagerId`].
    pub(crate) fn id(&self) -> crate::ManagerId {
        self.id
    }

    /// Test-only: acquire a guard that pauses every subsequent
    /// `release_primary` *before* it takes the store mutex, leaving
    /// the slot in `Primary { weak: dead }` so a concurrent lookup
    /// can drive the eager `Primary → Inactive` branch
    /// deterministically. Drop the returned guard to resume.
    #[cfg(test)]
    pub(crate) fn pause_release_primary(&self) -> tracing_mutex::parkinglot::MutexGuard<'_, ()> {
        self.release_primary_gate.lock()
    }

    /// Test-only: number of times `release_primary` has been entered
    /// since construction. Useful as a signal in race tests to wait
    /// for a drop thread to reach the gate without a sleep.
    #[cfg(test)]
    pub(crate) fn release_primary_arrivals(&self) -> u64 {
        self.release_primary_arrivals
            .load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn block_size(&self) -> usize {
        self.block_size
    }

    pub(crate) fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    pub(crate) fn metrics(&self) -> &Arc<BlockPoolMetrics> {
        &self.metrics
    }

    pub(crate) fn reset_len(&self) -> usize {
        self.inner.lock().free.len()
    }

    pub(crate) fn inactive_len(&self) -> usize {
        self.inner.lock().inactive.len()
    }

    /// Atomic snapshot of `reset_len + inactive_len` under a single store-lock
    /// acquisition. Reading the two pools separately can yield a count that
    /// never existed (e.g. a concurrent reset→inactive promotion observed
    /// twice, inflating the total above `total_blocks`).
    pub(crate) fn available_len(&self) -> usize {
        let inner = self.inner.lock();
        inner.free.len() + inner.inactive.len()
    }

    pub(crate) fn has_inactive(&self, seq_hash: SequenceHash) -> bool {
        self.inner.lock().inactive.has(seq_hash)
    }

    pub(crate) fn slot_block_size(&self, block_id: BlockId) -> usize {
        self.inner.lock().slots[block_id].block_size
    }

    // ---------- guard construction ----------

    /// Allocate up to `count` MutableBlocks from the reset pool only.
    /// Returns however many were available (no eviction).
    pub(crate) fn allocate_reset_blocks(self: &Arc<Self>, count: usize) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.lock();
        let take = std::cmp::min(count, inner.free.len());
        let mut out = Vec::with_capacity(take);
        for _ in 0..take {
            let id = inner.free.pop_front().unwrap();
            inner.slots[id].state = SlotState::Mutable;
            let block_size = inner.slots[id].block_size;
            out.push(MutableBlock::from_store(self.clone(), id, block_size));
        }
        self.metrics.dec_reset_pool_size_by(take as i64);
        self.metrics.inc_inflight_mutable_by(take as i64);
        self.metrics.inc_allocations(take as u64);
        self.metrics.inc_allocations_from_reset(take as u64);
        out
    }

    /// All-or-nothing allocation across the reset and inactive pools under a
    /// single store-mutex acquisition. Returns `None` iff
    /// `free.len() + inactive.len() < count`; otherwise drains `count`
    /// blocks (reset first, then inactive) and reports the evicted hashes.
    /// No partial commits, no put-backs.
    pub(crate) fn allocate_atomic(
        self: &Arc<Self>,
        count: usize,
    ) -> Option<(Vec<MutableBlock<T>>, Vec<SequenceHash>)> {
        if count == 0 {
            return Some((Vec::new(), Vec::new()));
        }
        let mut inner = self.inner.lock();
        if inner.free.len() + inner.inactive.len() < count {
            return None;
        }

        let from_reset = std::cmp::min(count, inner.free.len());
        let from_inactive = count - from_reset;

        // Stage decisions on raw `BlockId`s first; only commit slot
        // transitions (and construct MutableBlock guards) once we know
        // the inactive backend returned the requested count.
        let mut reset_ids: Vec<BlockId> = Vec::with_capacity(from_reset);
        for _ in 0..from_reset {
            reset_ids.push(inner.free.pop_front().unwrap());
        }
        let evicted_pairs = if from_inactive > 0 {
            inner.inactive.allocate(from_inactive)
        } else {
            Vec::new()
        };
        // Defensive runtime check: any backend that violates the
        // `len() >= n ⇒ allocate(n).len() == n` invariant must not
        // leave us partially committed. Roll back and return None.
        // Restore FIFO order by re-inserting the popped reset IDs at the
        // front in reverse — `pop_front` consumed `[a, b, c]`, so
        // `push_front` in reverse `[c, b, a]` re-prepends `a, b, c`.
        if evicted_pairs.len() != from_inactive {
            for (h, id) in evicted_pairs {
                inner.inactive.insert(h, id);
            }
            for id in reset_ids.into_iter().rev() {
                inner.free.push_front(id);
            }
            self.metrics.inc_allocate_atomic_rollback();
            return None;
        }

        // Commit. Past this point we cannot fail.
        let mut blocks = Vec::with_capacity(count);
        for id in reset_ids {
            inner.slots[id].state = SlotState::Mutable;
            let block_size = inner.slots[id].block_size;
            blocks.push(MutableBlock::from_store(self.clone(), id, block_size));
        }
        let mut evicted = Vec::with_capacity(from_inactive);
        let mut handles = Vec::with_capacity(from_inactive);
        for (seq_hash, block_id) in evicted_pairs {
            let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
            inner.slots[block_id].state = SlotState::Mutable;
            let block_size = inner.slots[block_id].block_size;
            blocks.push(MutableBlock::from_store(self.clone(), block_id, block_size));
            evicted.push(seq_hash);
            handles.push(handle);
        }

        self.metrics.dec_reset_pool_size_by(from_reset as i64);
        self.metrics.dec_inactive_pool_size_by(from_inactive as i64);
        self.metrics.inc_inflight_mutable_by(count as i64);
        self.metrics.inc_evictions(from_inactive as u64);
        self.metrics.inc_allocations(count as u64);
        self.metrics.inc_allocations_from_reset(from_reset as u64);

        drop(inner);
        // mark_absent::<T> takes the registry attachments lock — invoke
        // outside the store lock to honour the documented ordering.
        for h in handles {
            h.mark_absent::<T>();
        }
        Some((blocks, evicted))
    }

    /// Drain the inactive pool entirely into MutableBlocks.
    pub(crate) fn drain_inactive_to_mutable(self: &Arc<Self>) -> Vec<MutableBlock<T>> {
        let mut inner = self.inner.lock();
        let drained = inner.inactive.allocate_all();
        let count = drained.len();
        let mut handles = Vec::with_capacity(count);
        let mut out = Vec::with_capacity(count);
        for (_seq_hash, block_id) in drained {
            let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
            inner.slots[block_id].state = SlotState::Mutable;
            handles.push(handle);
            let block_size = inner.slots[block_id].block_size;
            out.push(MutableBlock::from_store(self.clone(), block_id, block_size));
        }
        self.metrics.dec_inactive_pool_size_by(count as i64);
        self.metrics.inc_inflight_mutable_by(count as i64);
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
        self.promote_inactive(hashes, touch, /*scan*/ false)
            .into_iter()
            .map(|(_, inner_arc)| inner_arc)
            .collect()
    }

    /// Scan-style version of [`find_inactive_primaries`] — does not stop on miss.
    pub(crate) fn scan_inactive_primaries(
        self: &Arc<Self>,
        hashes: &[SequenceHash],
        touch: bool,
    ) -> Vec<(SequenceHash, Arc<ImmutableBlockInner<T>>)> {
        self.promote_inactive(hashes, touch, /*scan*/ true)
    }

    /// Atomic active-or-inactive lookup by sequence hash. Replaces the
    /// previous `upgrade_or_resurrect` two-lock dance.
    pub(crate) fn acquire_for_hash(
        self: &Arc<Self>,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<ImmutableBlockInner<T>>> {
        let mut inner = self.inner.lock();
        self.acquire_for_hash_locked(&mut inner, seq_hash, touch)
    }

    /// Locked-form of [`acquire_for_hash`]. Walks one path under the
    /// caller's lock:
    /// 1. If `active_by_hash[seq_hash]` resolves to a Primary slot whose
    ///    `Weak` upgrades, return that strong `Arc`.
    /// 2. If the `Weak` is dead (last user is mid-drop), eagerly transition
    ///    `Primary → Inactive` ourselves, then fall through to (3).
    /// 3. If the inactive index has the hash, resurrect it.
    /// 4. Else `None`.
    fn acquire_for_hash_locked(
        self: &Arc<Self>,
        inner: &mut BlockStoreInner<T>,
        seq_hash: SequenceHash,
        touch: bool,
    ) -> Option<Arc<ImmutableBlockInner<T>>> {
        // (1) Active path.
        if let Some(&block_id) = inner.active_by_hash.get(&seq_hash) {
            let live: Option<Arc<ImmutableBlockInner<T>>> = match &inner.slots[block_id].state {
                SlotState::Primary { inner: weak, .. } => weak.upgrade(),
                other => panic!("active_by_hash[{seq_hash:?}] = {block_id} but slot is {other:?}"),
            };
            if let Some(arc) = live {
                return Some(arc);
            }
            // (2) Eager Primary → Inactive transition. The original
            // Inner::drop will see slot != Primary and no-op.
            self.eager_primary_to_inactive_locked(inner, seq_hash, block_id);
            // Fall through to inactive path.
        }

        // (3) Inactive path. Single-hash fast-path through the
        // backend-specific `find_match` override (O(1) for hashmap/lru
        // backends) instead of allocating a one-element slice + Vec.
        let block_id = inner.inactive.find_match(seq_hash, touch)?.1;
        self.metrics.dec_inactive_pool_size();
        let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
        let inner_arc =
            ImmutableBlockInner::new_primary(self.clone(), block_id, seq_hash, handle.clone());
        inner.slots[block_id].state = SlotState::Primary {
            seq_hash,
            handle,
            inner: Arc::downgrade(&inner_arc),
        };
        inner.active_by_hash.insert(seq_hash, block_id);
        Some(inner_arc)
    }

    /// Atomic registration of a [`CompleteBlock`]: lookup-then-transition
    /// under one store-mutex acquisition. Closes the register-vs-register
    /// race for the same sequence hash.
    ///
    /// On `BlockDuplicationPolicy::Allow` returns a duplicate-backed
    /// `Arc<ImmutableBlockInner<T>>`; on `Reject` returns the existing
    /// primary's `Arc` and lets the supplied `block` guard release its
    /// slot back to the reset pool.
    pub(crate) fn register_completed_block(
        self: &Arc<Self>,
        block: CompleteBlock<T>,
        handle: BlockRegistrationHandle,
        policy: BlockDuplicationPolicy,
    ) -> Arc<ImmutableBlockInner<T>> {
        let block_id = block.block_id();
        let seq_hash = block.sequence_hash();
        debug_assert_eq!(seq_hash, handle.seq_hash());

        // Disarm the guard up front so the slot stays in `Staged` state
        // when we transition; we re-arm only on the Reject path.
        let mut block = block;
        block.disarm();

        let mut inner = self.inner.lock();
        let existing = self.acquire_for_hash_locked(&mut inner, seq_hash, false);

        // Whether we added a new presence-bearing slot (Primary or
        // Duplicate). Reject does not, since the slot returns to Reset.
        let mut presence_added = false;

        let result = if let Some(existing_primary) = existing {
            assert_ne!(
                existing_primary.block_id(),
                block_id,
                "register_completed_block: collision with same block_id {block_id}"
            );
            match policy {
                BlockDuplicationPolicy::Allow => {
                    debug_assert!(matches!(
                        inner.slots[block_id].state,
                        SlotState::Staged { .. }
                    ));
                    let inner_arc = ImmutableBlockInner::new_duplicate(
                        self.clone(),
                        block_id,
                        seq_hash,
                        handle.clone(),
                        existing_primary,
                    );
                    inner.slots[block_id].state = SlotState::Duplicate {
                        seq_hash,
                        handle: handle.clone(),
                        inner: Arc::downgrade(&inner_arc),
                    };
                    self.metrics.inc_duplicate_blocks();
                    presence_added = true;
                    inner_arc
                }
                BlockDuplicationPolicy::Reject => {
                    self.metrics.inc_registration_dedup();
                    // Re-arm so the block guard's drop releases the slot
                    // (Staged → Reset) when it falls out of scope below.
                    block.rearm();
                    existing_primary
                }
            }
        } else {
            // Fresh primary.
            debug_assert!(matches!(
                inner.slots[block_id].state,
                SlotState::Staged { .. }
            ));
            let inner_arc =
                ImmutableBlockInner::new_primary(self.clone(), block_id, seq_hash, handle.clone());
            inner.slots[block_id].state = SlotState::Primary {
                seq_hash,
                handle: handle.clone(),
                inner: Arc::downgrade(&inner_arc),
            };
            inner.active_by_hash.insert(seq_hash, block_id);
            presence_added = true;
            inner_arc
        };

        drop(inner);

        // mark_present takes the attachments lock; lock-order
        // (attachments → store) is satisfied because the store lock has
        // already been released. Skip on Reject — no new presence-bearing
        // slot was created.
        if presence_added {
            handle.mark_present::<T>();
        }

        // Block guard drops here: armed=false on Allow/fresh paths
        // (slot already transitioned), armed=true on Reject (releases
        // Staged → Reset).
        drop(block);
        result
    }

    /// Internal helper: under the store lock, transition a Primary slot to
    /// Inactive without touching presence (the original Inner::drop's
    /// presence-side responsibilities are unchanged — it just no-ops the
    /// slot transition since we did it).
    fn eager_primary_to_inactive_locked(
        &self,
        inner: &mut BlockStoreInner<T>,
        seq_hash: SequenceHash,
        block_id: BlockId,
    ) {
        let slot = &mut inner.slots[block_id];
        let handle = match &slot.state {
            SlotState::Primary { handle, .. } => handle.clone(),
            other => panic!("eager_primary_to_inactive: slot {block_id} was {other:?}"),
        };
        slot.state = SlotState::Inactive { seq_hash, handle };
        inner.inactive.insert(seq_hash, block_id);
        inner.active_by_hash.remove(&seq_hash);
        self.metrics.inc_inactive_pool_size();
        self.metrics.inc_eager_primary_to_inactive();
        tracing::trace!(
            ?seq_hash,
            block_id,
            "Eager Primary → Inactive (lookup-driven)"
        );
    }

    /// Common slot-transition core for find/scan inactive promotions.
    /// Unlike the previous two-step version, this builds the
    /// `ImmutableBlockInner` and writes its `Weak` into the slot under the
    /// same lock acquisition.
    fn promote_inactive(
        self: &Arc<Self>,
        hashes: &[SequenceHash],
        touch: bool,
        scan: bool,
    ) -> Vec<(SequenceHash, Arc<ImmutableBlockInner<T>>)> {
        let mut inner = self.inner.lock();
        let matched: Vec<(SequenceHash, BlockId)> = if scan {
            inner.inactive.scan_matches(hashes, touch)
        } else {
            // First-hash fast-path: probe the head via the
            // backend-specific `find_match` override before allocating
            // the result Vec. Empty input or a head miss exits without
            // any allocation.
            let Some((&first_hash, rest)) = hashes.split_first() else {
                return Vec::new();
            };
            let Some(first_pair) = inner.inactive.find_match(first_hash, touch) else {
                return Vec::new();
            };
            let mut matched = Vec::with_capacity(hashes.len());
            matched.push(first_pair);
            if !rest.is_empty() {
                matched.extend(inner.inactive.find_matches(rest, touch));
            }
            matched
        };
        self.metrics.dec_inactive_pool_size_by(matched.len() as i64);
        matched
            .into_iter()
            .map(|(seq_hash, block_id)| {
                let handle = take_inactive_handle(&mut inner.slots[block_id], block_id);
                let inner_arc = ImmutableBlockInner::new_primary(
                    self.clone(),
                    block_id,
                    seq_hash,
                    handle.clone(),
                );
                inner.slots[block_id].state = SlotState::Primary {
                    seq_hash,
                    handle,
                    inner: Arc::downgrade(&inner_arc),
                };
                inner.active_by_hash.insert(seq_hash, block_id);
                (seq_hash, inner_arc)
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
        self.metrics.inc_reset_pool_size();
        self.metrics.dec_inflight_mutable();
    }

    /// `Mutable` → `Staged` (MutableBlock::stage / ::complete).
    pub(crate) fn transition_to_staged(&self, block_id: BlockId, seq_hash: SequenceHash) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(inner.slots[block_id].state, SlotState::Mutable));
        inner.slots[block_id].state = SlotState::Staged { seq_hash };
        self.metrics.dec_inflight_mutable();
        self.metrics.inc_stagings();
    }

    /// `Staged` → `Mutable` (CompleteBlock::reset).
    pub(crate) fn transition_back_to_mutable(&self, block_id: BlockId) {
        let mut inner = self.inner.lock();
        debug_assert!(matches!(
            inner.slots[block_id].state,
            SlotState::Staged { .. }
        ));
        inner.slots[block_id].state = SlotState::Mutable;
        self.metrics.inc_inflight_mutable();
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
        self.metrics.inc_reset_pool_size();
    }

    /// Drop transition for the last clone of a primary `ImmutableBlockInner`:
    /// `Primary` → `Inactive`, but only if the slot still references this
    /// specific Inner. If a concurrent `acquire_for_hash` already eagerly
    /// transitioned the slot, this is a no-op.
    pub(crate) fn release_primary(&self, block_id: BlockId, self_ptr: *const ()) {
        // Test-only deterministic race-window widening:
        //   1. Bump the arrival counter so a coordinating test can
        //      observe "the drop has entered release_primary" without
        //      a sleep.
        //   2. Acquire the gate. While a test holds it, this call
        //      parks here *before* the store mutex is touched, so the
        //      slot remains in `Primary { weak: dead }` and a
        //      concurrent lookup can drive the eager-transition path.
        #[cfg(test)]
        self.release_primary_arrivals
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        #[cfg(test)]
        let _gate = self.release_primary_gate.lock();
        let mut inner = self.inner.lock();
        let slot = &mut inner.slots[block_id];
        let (seq_hash, handle) = match &slot.state {
            SlotState::Primary {
                seq_hash,
                handle,
                inner: weak,
            } if weak.as_ptr() as *const () == self_ptr => (*seq_hash, handle.clone()),
            // Eager lookup-driven transition already ran, OR this slot has
            // since been resurrected to a different Inner. No-op.
            _ => {
                self.metrics.inc_release_primary_noop();
                return;
            }
        };
        slot.state = SlotState::Inactive { seq_hash, handle };
        inner.inactive.insert(seq_hash, block_id);
        inner.active_by_hash.remove(&seq_hash);
        self.metrics.inc_inactive_pool_size();
        tracing::trace!(?seq_hash, block_id, "Block stored in inactive pool");
    }

    /// Drop transition for the last clone of a duplicate `ImmutableBlockInner`:
    /// `Duplicate` → `Reset` (with `mark_absent::<T>`). Identity-checked.
    pub(crate) fn release_duplicate(&self, block_id: BlockId, self_ptr: *const ()) {
        let handle = {
            let mut inner = self.inner.lock();
            let slot = &mut inner.slots[block_id];
            let handle = match &slot.state {
                SlotState::Duplicate {
                    handle,
                    inner: weak,
                    ..
                } if weak.as_ptr() as *const () == self_ptr => handle.clone(),
                // Slot has moved on (this should not normally happen for
                // duplicates since they cannot be resurrected, but guard
                // defensively).
                _ => {
                    self.metrics.inc_release_duplicate_noop();
                    return;
                }
            };
            slot.state = SlotState::Reset;
            inner.free.push_back(block_id);
            self.metrics.inc_reset_pool_size();
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
fn take_inactive_handle<T: BlockMetadata>(
    slot: &mut BlockSlot<T>,
    block_id: BlockId,
) -> BlockRegistrationHandle {
    match &slot.state {
        SlotState::Inactive { handle, .. } => handle.clone(),
        other => panic!("expected Inactive state for slot {block_id}, got {other:?}"),
    }
}

/// Hash → strong `Arc<ImmutableBlockInner<T>>` lookup. Walks active
/// then inactive under one store-mutex acquisition. `touch` propagates
/// to the inactive resurrection path so frequency tracking observes
/// the hit even when the active path absorbs it.
pub(crate) fn upgrade_or_resurrect<T: BlockMetadata + Sync>(
    handle: &BlockRegistrationHandle,
    store: &Arc<BlockStore<T>>,
    touch: bool,
) -> Option<Arc<ImmutableBlockInner<T>>> {
    store.acquire_for_hash(handle.seq_hash(), touch)
}
