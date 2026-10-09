// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dense rank slots: CRTC's slot registry (`concurrent_radix_tree_compressed/coverage.rs`),
//! copied without its coverage types so CRTC's module stays untouched on this branch. A
//! merged design would share one module. The only addition is [`Slot::from_index`],
//! which this backend's bit walks need.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use crossbeam_utils::Backoff;
use parking_lot::{Condvar, Mutex};
use rustc_hash::FxHashMap;

use super::types::WorkerRemovalTarget;
use crate::protocols::{KvCacheEventError, WorkerWithDpRank};

const WORD_BITS: usize = u64::BITS as usize;
/// Slots are `u16`, so a registry hands out at most this many at once.
pub(super) const MAX_SLOTS: usize = 1 << u16::BITS;

/// Dense index of a rank in its tree's [`SlotRegistry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Slot(u16);

impl Slot {
    #[inline]
    pub(super) fn index(self) -> usize {
        usize::from(self.0)
    }

    /// The slot with dense index `index`, which must be below [`MAX_SLOTS`].
    #[inline]
    pub(super) fn from_index(index: usize) -> Self {
        debug_assert!(index < MAX_SLOTS);
        Self(index as u16)
    }

    #[inline]
    fn word_and_bit(self) -> (usize, u64) {
        (self.index() / WORD_BITS, 1 << (self.index() % WORD_BITS))
    }

    fn from_word_bit(word: usize, bit: u32) -> Self {
        Self((word * WORD_BITS + bit as usize) as u16)
    }
}

/// Slot ownership as published to readers and writers.
#[derive(Clone, Default)]
pub(super) struct SlotTable {
    /// The rank credited for each slot's bits; `None` once the slot is vacated.
    owners: Box<[Option<WorkerWithDpRank>]>,
    /// Ranks that may still set bits, and their slots.
    slots: FxHashMap<WorkerWithDpRank, Slot>,
}

impl SlotTable {
    #[inline]
    pub(super) fn owner(&self, slot: Slot) -> Option<WorkerWithDpRank> {
        self.owners.get(slot.index()).copied().flatten()
    }

    #[inline]
    pub(super) fn slot_of(&self, worker: WorkerWithDpRank) -> Option<Slot> {
        self.slots.get(&worker).copied()
    }

    /// Whether a slot of `target`'s ranks is unmapped but not yet released.
    fn has_unreleased(&self, target: WorkerRemovalTarget) -> bool {
        self.owners.iter().enumerate().any(|(index, owner)| {
            owner.is_some_and(|rank| {
                target.matches(rank)
                    && self.slots.get(&rank).map(|slot| slot.index()) != Some(index)
            })
        })
    }
}

/// Slots freed through the epoch and ready for reuse.
struct ReleasedSlots {
    words: Box<[AtomicU64]>,
}

impl ReleasedSlots {
    fn new() -> Self {
        Self {
            words: (0..MAX_SLOTS / WORD_BITS)
                .map(|_| AtomicU64::new(0))
                .collect(),
        }
    }

    fn release(&self, slot: Slot) {
        let (index, bit) = slot.word_and_bit();
        self.words[index].fetch_or(bit, Ordering::Release);
    }

    /// Claims the lowest released slot below `issued`. Callers hold the registry lock, so
    /// only releases race with this and a seen bit stays set until claimed.
    fn take_lowest(&self, issued: usize) -> Option<Slot> {
        let words = issued.div_ceil(WORD_BITS);
        self.words[..words]
            .iter()
            .enumerate()
            .find_map(|(index, word)| {
                let bits = word.load(Ordering::Acquire);
                (bits != 0).then(|| {
                    let bit = bits.trailing_zeros();
                    word.fetch_and(!(1 << bit), Ordering::Relaxed);
                    Slot::from_word_bit(index, bit)
                })
            })
    }

    #[cfg(test)]
    fn contains(&self, slot: Slot) -> bool {
        let (index, bit) = slot.word_and_bit();
        self.words[index].load(Ordering::Acquire) & bit != 0
    }
}

/// Maps ranks to dense slots.
///
/// Writers look a rank's slot up in the current [`SlotTable`] under the epoch guard that
/// covers their whole event, and readers map slots back to ranks through the table they
/// loaded at the start of their walk.
///
/// A slot is released in four steps. [`Self::unmap`] stops new events from resolving it.
/// [`wait_for_pinned_threads`] lets events that resolved it earlier finish, so none of
/// their bits can land behind the sweep. The caller then sweeps the slot out of the
/// reachable tree, and [`Self::release`] vacates the table entry and frees the slot
/// through the epoch, so a reader that still holds the old table never sees it reused.
/// A removal that finds its ranks already unmapped by another lane waits for that
/// release with [`Self::wait_for_release`].
pub(super) struct SlotRegistry {
    /// Never null; replaced tables are retired through the epoch.
    table: Atomic<SlotTable>,
    /// Slots `0..issued` have been handed out at least once. The lock also serializes
    /// table publications.
    issued: Mutex<usize>,
    /// Notified under `issued` whenever [`Self::release`] vacates slots.
    vacated: Condvar,
    released: Arc<ReleasedSlots>,
}

impl Default for SlotRegistry {
    fn default() -> Self {
        Self {
            table: Atomic::new(SlotTable::default()),
            issued: Mutex::new(0),
            vacated: Condvar::new(),
            released: Arc::new(ReleasedSlots::new()),
        }
    }
}

impl Drop for SlotRegistry {
    fn drop(&mut self) {
        // A table reference handed out by `table` borrows only its guard, so retire the
        // current table like a replaced one instead of freeing it under a pinned reader.
        let guard = epoch::pin();
        let current = self.table.swap(Shared::null(), Ordering::AcqRel, &guard);
        // SAFETY: the swap unlinked `current`, and nothing publishes into a dropped registry.
        unsafe { guard.defer_destroy(current) };
    }
}

impl SlotRegistry {
    pub(super) fn table<'g>(&self, guard: &'g Guard) -> &'g SlotTable {
        // SAFETY: the pointer is never null, and a replaced table is destroyed only after
        // every guard pinned before the replacement is dropped.
        unsafe { self.table.load(Ordering::Acquire, guard).deref() }
    }

    /// Publishes `next`. Callers hold `issued`, which serializes publications.
    fn publish(&self, next: SlotTable, guard: &Guard) {
        let previous = self.table.swap(Owned::new(next), Ordering::AcqRel, guard);
        // SAFETY: the swap unlinked `previous`; guards that loaded it keep it alive.
        unsafe { guard.defer_destroy(previous) };
    }

    /// Returns `worker`'s slot, allocating the lowest free one if it has none.
    ///
    /// `guard` must stay pinned for as long as the caller writes bits with the slot.
    pub(super) fn acquire(
        &self,
        worker: WorkerWithDpRank,
        guard: &Guard,
    ) -> Result<Slot, KvCacheEventError> {
        if let Some(slot) = self.table(guard).slot_of(worker) {
            return Ok(slot);
        }

        let mut issued = self.issued.lock();
        let table = self.table(guard);
        if let Some(slot) = table.slot_of(worker) {
            return Ok(slot);
        }
        let slot = match self.released.take_lowest(*issued) {
            Some(slot) => slot,
            None if *issued < MAX_SLOTS => {
                *issued += 1;
                Slot((*issued - 1) as u16)
            }
            None => return Err(KvCacheEventError::CapacityExhausted),
        };

        let mut owners = table.owners.to_vec();
        owners.resize(*issued, None);
        debug_assert!(owners[slot.index()].is_none());
        owners[slot.index()] = Some(worker);
        let mut slots = table.slots.clone();
        slots.insert(worker, slot);
        self.publish(
            SlotTable {
                owners: owners.into_boxed_slice(),
                slots,
            },
            guard,
        );
        Ok(slot)
    }

    /// Stops `target`'s ranks from resolving their slots and returns the slots, which
    /// readers keep crediting until [`Self::release`].
    pub(super) fn unmap(&self, target: WorkerRemovalTarget) -> Vec<Slot> {
        let guard = epoch::pin();
        let _issued = self.issued.lock();
        let table = self.table(&guard);
        let mut next = table.clone();
        next.slots.retain(|worker, _| !target.matches(*worker));
        if next.slots.len() == table.slots.len() {
            return Vec::new();
        }

        let unmapped = table
            .slots
            .iter()
            .filter(|(worker, _)| target.matches(**worker))
            .map(|(_, &slot)| slot)
            .collect();
        self.publish(next, &guard);
        unmapped
    }

    /// Vacates swept slots and frees them once every reader that could still map them to
    /// their old rank has unpinned.
    pub(super) fn release(&self, slots: Vec<Slot>) {
        if slots.is_empty() {
            return;
        }
        let guard = epoch::pin();
        {
            let _issued = self.issued.lock();
            let mut next = self.table(&guard).clone();
            for slot in &slots {
                debug_assert!(!next.slots.values().any(|mapped| mapped == slot));
                next.owners[slot.index()] = None;
            }
            self.publish(next, &guard);
            self.vacated.notify_all();
        }
        let released = self.released.clone();
        guard.defer(move || {
            for slot in slots {
                released.release(slot);
            }
        });
    }

    /// Blocks until every slot of `target`'s ranks that has been unmapped is also
    /// released, i.e. swept out of the tree. The caller must not be pinned: the remover
    /// it waits for first waits for pinned threads.
    pub(super) fn wait_for_release(&self, target: WorkerRemovalTarget) {
        assert!(
            !epoch::is_pinned(),
            "waiting for a slot release while pinned can deadlock"
        );
        // Publications happen under `issued`, so no release slips between check and wait.
        let mut issued = self.issued.lock();
        loop {
            let unreleased = self.table(&epoch::pin()).has_unreleased(target);
            if !unreleased {
                return;
            }
            self.vacated.wait(&mut issued);
        }
    }

    #[cfg(test)]
    pub(super) fn is_released(&self, slot: Slot) -> bool {
        self.released.contains(slot)
    }
}

/// Blocks until every thread that is pinned when this is called has unpinned at least
/// once. The caller must not be pinned.
pub(super) fn wait_for_pinned_threads() {
    assert!(
        !epoch::is_pinned(),
        "waiting for pinned threads while pinned never finishes"
    );
    let done = Arc::new(AtomicBool::new(false));
    {
        let guard = epoch::pin();
        let done = done.clone();
        guard.defer(move || done.store(true, Ordering::Release));
        guard.flush();
    }
    let backoff = Backoff::new();
    while !done.load(Ordering::Acquire) {
        // Each flush tries to advance the epoch and runs expired deferred functions.
        epoch::pin().flush();
        backoff.snooze();
    }
}

#[cfg(test)]
mod tests {
    use std::thread;

    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn release_now(registry: &SlotRegistry, target: WorkerRemovalTarget) -> Vec<Slot> {
        let slots = registry.unmap(target);
        registry.release(slots.clone());
        wait_for_released(registry, &slots);
        slots
    }

    fn wait_for_released(registry: &SlotRegistry, slots: &[Slot]) {
        while !slots.iter().all(|&slot| registry.is_released(slot)) {
            wait_for_pinned_threads();
        }
    }

    #[test]
    fn allocation_takes_the_lowest_free_slot() {
        let registry = SlotRegistry::default();
        let guard = epoch::pin();
        let slots: Vec<_> = (0..5)
            .map(|id| registry.acquire(worker(id), &guard).unwrap())
            .collect();
        assert_eq!(slots, (0..5).map(Slot).collect::<Vec<_>>());
        assert_eq!(registry.acquire(worker(3), &guard).unwrap(), Slot(3));
        drop(guard);

        release_now(&registry, WorkerRemovalTarget::WorkerId(3));
        release_now(&registry, WorkerRemovalTarget::WorkerId(1));

        let guard = epoch::pin();
        assert_eq!(registry.acquire(worker(10), &guard).unwrap(), Slot(1));
        assert_eq!(registry.acquire(worker(11), &guard).unwrap(), Slot(3));
        assert_eq!(registry.acquire(worker(12), &guard).unwrap(), Slot(5));
        let table = registry.table(&guard);
        assert_eq!(table.owner(Slot(1)), Some(worker(10)));
        assert_eq!(table.slot_of(worker(1)), None);
    }

    #[test]
    fn unmapped_rank_stays_credited_until_released_and_old_tables_stay_valid() {
        let registry = SlotRegistry::default();
        let guard = epoch::pin();
        let slot = registry.acquire(worker(1), &guard).unwrap();
        let before = registry.table(&guard);

        let claimed = registry.unmap(WorkerRemovalTarget::WorkerId(1));
        assert_eq!(claimed, vec![slot]);
        assert!(registry.unmap(WorkerRemovalTarget::WorkerId(1)).is_empty());
        let unmapped = registry.table(&guard);
        assert_eq!(unmapped.slot_of(worker(1)), None);
        assert_eq!(unmapped.owner(slot), Some(worker(1)));

        registry.release(claimed);
        assert_eq!(registry.table(&guard).owner(slot), None);

        // The tables loaded under this guard are untouched copies.
        assert_eq!(before.slot_of(worker(1)), Some(slot));
        assert_eq!(before.owner(slot), Some(worker(1)));
        assert_eq!(unmapped.owner(slot), Some(worker(1)));
    }

    #[test]
    fn released_slot_is_not_reused_while_a_reader_is_pinned() {
        let registry = Arc::new(SlotRegistry::default());
        let slot = registry.acquire(worker(1), &epoch::pin()).unwrap();

        let (pinned_tx, pinned_rx) = std::sync::mpsc::channel();
        let (unpin_tx, unpin_rx) = std::sync::mpsc::channel::<()>();
        let reader_registry = registry.clone();
        let reader = thread::spawn(move || {
            let guard = epoch::pin();
            let table = reader_registry.table(&guard);
            pinned_tx.send(()).unwrap();
            unpin_rx.recv().unwrap();
            // The table loaded before the release still credits the old rank.
            assert_eq!(table.owner(slot), Some(worker(1)));
        });
        pinned_rx.recv().unwrap();

        let claimed = registry.unmap(WorkerRemovalTarget::WorkerId(1));
        registry.release(claimed);
        for _ in 0..64 {
            epoch::pin().flush();
        }
        assert!(!registry.is_released(slot));
        let fresh = registry.acquire(worker(2), &epoch::pin()).unwrap();
        assert_ne!(fresh, slot);

        unpin_tx.send(()).unwrap();
        reader.join().unwrap();
        wait_for_released(&registry, &[slot]);
        assert_eq!(registry.acquire(worker(3), &epoch::pin()).unwrap(), slot);
    }

    #[test]
    fn exhausted_registry_fails_closed() {
        let registry = SlotRegistry::default();
        *registry.issued.lock() = MAX_SLOTS;
        let guard = epoch::pin();
        assert!(matches!(
            registry.acquire(worker(1), &guard),
            Err(KvCacheEventError::CapacityExhausted)
        ));
        assert_eq!(registry.table(&guard).slot_of(worker(1)), None);
    }

    #[test]
    fn wait_for_pinned_threads_waits_for_an_active_pin() {
        let (pinned_tx, pinned_rx) = std::sync::mpsc::channel();
        let unpinned = Arc::new(AtomicBool::new(false));
        let reader_unpinned = unpinned.clone();
        let reader = thread::spawn(move || {
            let guard = epoch::pin();
            pinned_tx.send(()).unwrap();
            thread::sleep(std::time::Duration::from_millis(50));
            reader_unpinned.store(true, Ordering::Release);
            drop(guard);
        });
        pinned_rx.recv().unwrap();
        wait_for_pinned_threads();
        assert!(unpinned.load(Ordering::Acquire));
        reader.join().unwrap();
    }
}
