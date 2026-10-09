// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dense rank slots (spec B 5.4), adapted from CRTC's `SlotRegistry`
//! (`concurrent_radix_tree_compressed/coverage.rs`) with one addition: `live_words`, the
//! number of coverage words that can hold an issued slot, so readers intersect only those.
//!
//! A slot is released in four steps: [`SlotRegistry::unmap`] stops new events from
//! resolving it, [`wait_for_pinned_threads`] lets events that resolved it earlier finish,
//! the caller sweeps the slot out of every reachable run, and [`SlotRegistry::release`]
//! vacates the table entry and frees the slot through the epoch, so a reader that still
//! holds the old table never sees the slot reused.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use crossbeam_utils::Backoff;
use parking_lot::{Condvar, Mutex};
use rustc_hash::FxHashMap;

use crate::protocols::{KvCacheEventError, WorkerId, WorkerWithDpRank};

const WORD_BITS: usize = u64::BITS as usize;
/// Slots are `u16`, so a registry hands out at most this many at once.
pub(crate) const MAX_SLOTS: usize = 1 << u16::BITS;

/// Dense index of a rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Slot(u16);

impl Slot {
    #[inline]
    pub(crate) fn from_index(index: usize) -> Self {
        debug_assert!(index < MAX_SLOTS);
        Self(index as u16)
    }

    #[inline]
    pub(crate) fn index(self) -> usize {
        usize::from(self.0)
    }

    /// `(word, bit)` of this slot in a coverage bitmap.
    #[inline]
    pub(crate) fn word_and_bit(self) -> (usize, u64) {
        (self.index() / WORD_BITS, 1 << (self.index() % WORD_BITS))
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RemovalTarget {
    Worker(WorkerId),
    Rank(WorkerWithDpRank),
}

impl RemovalTarget {
    pub(crate) fn matches(self, rank: WorkerWithDpRank) -> bool {
        match self {
            Self::Worker(worker_id) => rank.worker_id == worker_id,
            Self::Rank(target) => rank == target,
        }
    }
}

/// Slot ownership as published to readers and writers.
#[derive(Clone, Default)]
pub(crate) struct SlotTable {
    /// The rank credited for each slot's bits; `None` once the slot is vacated.
    owners: Box<[Option<WorkerWithDpRank>]>,
    /// Ranks that may still set bits, and their slots.
    slots: FxHashMap<WorkerWithDpRank, Slot>,
}

impl SlotTable {
    #[inline]
    pub(crate) fn owner(&self, slot: usize) -> Option<WorkerWithDpRank> {
        self.owners.get(slot).copied().flatten()
    }

    #[inline]
    pub(crate) fn slot_of(&self, rank: WorkerWithDpRank) -> Option<Slot> {
        self.slots.get(&rank).copied()
    }

    pub(crate) fn mapped(&self, target: RemovalTarget) -> Vec<WorkerWithDpRank> {
        self.slots
            .keys()
            .copied()
            .filter(|rank| target.matches(*rank))
            .collect()
    }

    fn has_unreleased(&self, target: RemovalTarget) -> bool {
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

    /// Claims the lowest released slot below `issued`. Callers hold the registry lock.
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
                    Slot::from_index(index * WORD_BITS + bit as usize)
                })
            })
    }

    #[cfg(test)]
    fn contains(&self, slot: Slot) -> bool {
        let (index, bit) = slot.word_and_bit();
        self.words[index].load(Ordering::Acquire) & bit != 0
    }
}

pub(crate) struct SlotRegistry {
    /// Never null; replaced tables are retired through the epoch.
    table: Atomic<SlotTable>,
    /// Slots `0..issued` have been handed out at least once. The lock also serializes
    /// table publications.
    issued: Mutex<usize>,
    /// Coverage words that can hold an issued slot.
    live_words: AtomicUsize,
    vacated: Condvar,
    released: Arc<ReleasedSlots>,
}

impl Default for SlotRegistry {
    fn default() -> Self {
        Self {
            table: Atomic::new(SlotTable::default()),
            issued: Mutex::new(0),
            live_words: AtomicUsize::new(0),
            vacated: Condvar::new(),
            released: Arc::new(ReleasedSlots::new()),
        }
    }
}

impl Drop for SlotRegistry {
    fn drop(&mut self) {
        let guard = epoch::pin();
        let current = self.table.swap(Shared::null(), Ordering::AcqRel, &guard);
        // SAFETY: the swap unlinked `current`, and nothing publishes into a dropped registry.
        unsafe { guard.defer_destroy(current) };
    }
}

impl SlotRegistry {
    pub(crate) fn table<'g>(&self, guard: &'g Guard) -> &'g SlotTable {
        // SAFETY: the pointer is never null, and a replaced table is destroyed only after
        // every guard pinned before the replacement is dropped.
        unsafe { self.table.load(Ordering::Acquire, guard).deref() }
    }

    /// Coverage words a reader needs to look at.
    #[inline]
    pub(crate) fn live_words(&self) -> usize {
        self.live_words.load(Ordering::Acquire)
    }

    fn publish(&self, next: SlotTable, guard: &Guard) {
        let previous = self.table.swap(Owned::new(next), Ordering::AcqRel, guard);
        // SAFETY: the swap unlinked `previous`; guards that loaded it keep it alive.
        unsafe { guard.defer_destroy(previous) };
    }

    /// Returns `rank`'s slot, allocating the lowest free one if it has none. `guard` must
    /// stay pinned for as long as the caller writes coverage with the slot.
    pub(crate) fn acquire(
        &self,
        rank: WorkerWithDpRank,
        guard: &Guard,
    ) -> Result<Slot, KvCacheEventError> {
        if let Some(slot) = self.table(guard).slot_of(rank) {
            return Ok(slot);
        }
        let mut issued = self.issued.lock();
        let table = self.table(guard);
        if let Some(slot) = table.slot_of(rank) {
            return Ok(slot);
        }
        let slot = match self.released.take_lowest(*issued) {
            Some(slot) => slot,
            None if *issued < MAX_SLOTS => {
                *issued += 1;
                self.live_words
                    .store(issued.div_ceil(WORD_BITS), Ordering::Release);
                Slot::from_index(*issued - 1)
            }
            None => return Err(KvCacheEventError::CapacityExhausted),
        };
        let mut owners = table.owners.to_vec();
        owners.resize(*issued, None);
        debug_assert!(owners[slot.index()].is_none());
        owners[slot.index()] = Some(rank);
        let mut slots = table.slots.clone();
        slots.insert(rank, slot);
        self.publish(
            SlotTable {
                owners: owners.into_boxed_slice(),
                slots,
            },
            guard,
        );
        Ok(slot)
    }

    /// Stops `target`'s ranks from resolving their slots and returns them, which readers
    /// keep crediting until [`Self::release`].
    pub(crate) fn unmap(&self, target: RemovalTarget) -> Vec<Slot> {
        let guard = epoch::pin();
        let _issued = self.issued.lock();
        let table = self.table(&guard);
        let mut next = table.clone();
        next.slots.retain(|rank, _| !target.matches(*rank));
        if next.slots.len() == table.slots.len() {
            return Vec::new();
        }
        let unmapped = table
            .slots
            .iter()
            .filter(|(rank, _)| target.matches(**rank))
            .map(|(_, &slot)| slot)
            .collect();
        self.publish(next, &guard);
        unmapped
    }

    /// Vacates swept slots and frees them once every reader that could still map them to
    /// their old rank has unpinned.
    pub(crate) fn release(&self, slots: Vec<Slot>) {
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

    /// Blocks until every unmapped slot of `target` is also released. The caller must not
    /// be pinned.
    pub(crate) fn wait_for_release(&self, target: RemovalTarget) {
        assert!(
            !epoch::is_pinned(),
            "waiting for a slot release while pinned can deadlock"
        );
        let mut issued = self.issued.lock();
        loop {
            if !self.table(&epoch::pin()).has_unreleased(target) {
                return;
            }
            self.vacated.wait(&mut issued);
        }
    }

    #[cfg(test)]
    pub(crate) fn is_released(&self, slot: Slot) -> bool {
        self.released.contains(slot)
    }

    #[cfg(test)]
    pub(crate) fn set_issued_for_test(&self, issued: usize) {
        *self.issued.lock() = issued;
        self.live_words
            .store(issued.div_ceil(WORD_BITS), Ordering::Release);
    }
}

/// Blocks until every thread pinned when this is called has unpinned at least once. The
/// caller must not be pinned.
pub(crate) fn wait_for_pinned_threads() {
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
        epoch::pin().flush();
        backoff.snooze();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rank(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    #[test]
    fn live_words_track_the_highest_issued_slot() {
        let registry = SlotRegistry::default();
        let guard = epoch::pin();
        assert_eq!(registry.live_words(), 0);
        registry.acquire(rank(0), &guard).unwrap();
        assert_eq!(registry.live_words(), 1);
        for id in 1..=64 {
            registry.acquire(rank(id), &guard).unwrap();
        }
        assert_eq!(registry.live_words(), 2);
        drop(guard);

        let slots = registry.unmap(RemovalTarget::Worker(64));
        registry.release(slots.clone());
        while !slots.iter().all(|&slot| registry.is_released(slot)) {
            wait_for_pinned_threads();
        }
        // A reused slot does not lower or raise the word count.
        let guard = epoch::pin();
        assert_eq!(registry.acquire(rank(100), &guard).unwrap(), Slot(64));
        assert_eq!(registry.live_words(), 2);
    }

    #[test]
    fn exhausted_registry_fails_closed() {
        let registry = SlotRegistry::default();
        registry.set_issued_for_test(MAX_SLOTS);
        let guard = epoch::pin();
        assert!(matches!(
            registry.acquire(rank(1), &guard),
            Err(KvCacheEventError::CapacityExhausted)
        ));
        assert_eq!(registry.live_words(), MAX_SLOTS / 64);
    }
}
