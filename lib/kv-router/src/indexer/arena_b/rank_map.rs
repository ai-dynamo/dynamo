// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A rank's block map (spec B 9): external hash to the position that holds the block,
//! with the run's generation (fix 5). Dynamo's `BlockLookup` design (Fibonacci home
//! slot, linear probing, at most 3/4 full, backward-shift deletion, batch prefetch)
//! specialised to 24-byte slots, where `run == 0` marks an empty slot.

use super::runs::RunId;
use crate::protocols::ExternalSequenceBlockHash;

const MIN_CAPACITY: usize = 16;
const PREFETCH_DISTANCE: usize = 8;

/// Where a rank's block lives: `(run, offset)`, valid while the run has `generation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) run: RunId,
    pub(crate) offset: u32,
    pub(crate) generation: u32,
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct MapSlot {
    key: u64,
    run: RunId,
    offset: u32,
    generation: u32,
    _spare: u32,
}

const _: () = assert!(std::mem::size_of::<MapSlot>() == 24);

impl MapSlot {
    #[inline]
    fn is_empty(&self) -> bool {
        self.run == 0
    }

    #[inline]
    fn entry(&self) -> Entry {
        Entry {
            run: self.run,
            offset: self.offset,
            generation: self.generation,
        }
    }
}

pub(crate) struct RankMap {
    /// Power-of-two length, or empty before the first insert.
    slots: Box<[MapSlot]>,
    len: usize,
    shift: u32,
}

impl Default for RankMap {
    fn default() -> Self {
        Self {
            slots: Box::default(),
            len: 0,
            shift: 64,
        }
    }
}

impl RankMap {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn home(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// The slot holding `key`, or the empty slot that ends its probe run.
    #[inline]
    fn probe(&self, key: u64) -> Result<usize, usize> {
        let mask = self.slots.len() - 1;
        let mut i = self.home(key);
        loop {
            let slot = &self.slots[i];
            if slot.is_empty() {
                return Err(i);
            }
            if slot.key == key {
                return Ok(i);
            }
            i = (i + 1) & mask;
        }
    }

    #[inline]
    pub(crate) fn get(&self, key: ExternalSequenceBlockHash) -> Option<Entry> {
        if self.len == 0 {
            return None;
        }
        self.probe(key.0).ok().map(|i| self.slots[i].entry())
    }

    fn reserve(&mut self, additional: usize) {
        let needed = self.len + additional;
        if needed * 4 <= self.slots.len() * 3 {
            return;
        }
        let mut capacity = self.slots.len().max(MIN_CAPACITY);
        while needed * 4 > capacity * 3 {
            capacity *= 2;
        }
        let old = std::mem::replace(
            &mut self.slots,
            vec![MapSlot::default(); capacity].into_boxed_slice(),
        );
        self.shift = 64 - capacity.trailing_zeros();
        let mask = capacity - 1;
        for slot in old.iter().filter(|slot| !slot.is_empty()) {
            let mut i = self.home(slot.key);
            while !self.slots[i].is_empty() {
                i = (i + 1) & mask;
            }
            self.slots[i] = *slot;
        }
    }

    /// Points `key` at `entry`. Returns the entry it replaced.
    pub(crate) fn insert(&mut self, key: ExternalSequenceBlockHash, entry: Entry) -> Option<Entry> {
        debug_assert_ne!(entry.run, 0);
        self.reserve(1);
        let slot = MapSlot {
            key: key.0,
            run: entry.run,
            offset: entry.offset,
            generation: entry.generation,
            _spare: 0,
        };
        match self.probe(key.0) {
            Ok(i) => Some(std::mem::replace(&mut self.slots[i], slot).entry()),
            Err(i) => {
                self.slots[i] = slot;
                self.len += 1;
                None
            }
        }
    }

    pub(crate) fn remove(&mut self, key: ExternalSequenceBlockHash) -> Option<Entry> {
        if self.len == 0 {
            return None;
        }
        let mut hole = self.probe(key.0).ok()?;
        let removed = std::mem::take(&mut self.slots[hole]).entry();
        self.len -= 1;
        // Backward-shift deletion.
        let mask = self.slots.len() - 1;
        let mut j = hole;
        loop {
            j = (j + 1) & mask;
            if self.slots[j].is_empty() {
                break;
            }
            let home = self.home(self.slots[j].key);
            if (j.wrapping_sub(home) & mask) >= (j.wrapping_sub(hole) & mask) {
                self.slots[hole] = std::mem::take(&mut self.slots[j]);
                hole = j;
            }
        }
        Some(removed)
    }

    #[inline]
    fn prefetch(&self, key: u64) {
        if self.slots.is_empty() {
            return;
        }
        let ptr = self
            .slots
            .as_ptr()
            .wrapping_add(self.home(key))
            .cast::<i8>();
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetching is a hint that never faults, and `ptr` points into `slots`.
        #[allow(unused_unsafe)]
        unsafe {
            std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = ptr;
    }

    /// Points each `keys[i]` at `(run, first_offset + i, generation)`, prefetching ahead. Calls
    /// `on_new` for each key that had no entry, and returns how many entries changed.
    pub(crate) fn insert_run(
        &mut self,
        keys: impl ExactSizeIterator<Item = ExternalSequenceBlockHash> + Clone,
        run: RunId,
        first_offset: u32,
        generation: u32,
        mut on_new: impl FnMut(ExternalSequenceBlockHash),
    ) -> usize {
        self.reserve(keys.len());
        let mut ahead = keys.clone();
        for key in ahead.by_ref().take(PREFETCH_DISTANCE) {
            self.prefetch(key.0);
        }
        let mut changed = 0;
        for (i, key) in keys.enumerate() {
            if let Some(next) = ahead.next() {
                self.prefetch(next.0);
            }
            let entry = Entry {
                run,
                offset: first_offset + i as u32,
                generation,
            };
            match self.probe(key.0) {
                Ok(slot) => {
                    let existing = &mut self.slots[slot];
                    if existing.entry() != entry {
                        existing.run = entry.run;
                        existing.offset = entry.offset;
                        existing.generation = entry.generation;
                        changed += 1;
                    }
                }
                Err(slot) => {
                    self.slots[slot] = MapSlot {
                        key: key.0,
                        run,
                        offset: entry.offset,
                        generation,
                        _spare: 0,
                    };
                    self.len += 1;
                    changed += 1;
                    on_new(key);
                }
            }
        }
        changed
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (ExternalSequenceBlockHash, Entry)> + '_ {
        self.slots
            .iter()
            .filter(|slot| !slot.is_empty())
            .map(|slot| (ExternalSequenceBlockHash(slot.key), slot.entry()))
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn bytes(&self) -> usize {
        self.slots.len() * std::mem::size_of::<MapSlot>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn key(k: u64) -> ExternalSequenceBlockHash {
        ExternalSequenceBlockHash(k)
    }

    fn entry(v: u64) -> Entry {
        Entry {
            run: 2 + (v % 1000) as u32,
            offset: (v / 1000) as u32,
            generation: (v % 7) as u32,
        }
    }

    /// Randomized differential test against `HashMap` (the `BlockLookup` test, for the
    /// 24-byte slot). A small key space forces long probe runs, wraparound, and backward
    /// shifts across them.
    #[test]
    fn matches_hash_map_under_random_operations() {
        for seed in 0..16u64 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut map = RankMap::default();
            let mut model = HashMap::new();
            let key_space = 8 + seed * 17;
            for step in 0..20_000u64 {
                let k = key(rng.u64(..key_space));
                match rng.u32(..10) {
                    0..=4 => assert_eq!(map.insert(k, entry(step)), model.insert(k, entry(step))),
                    5..=7 => assert_eq!(map.remove(k), model.remove(&k)),
                    _ => {
                        let keys: Vec<_> = (0..rng.usize(..12))
                            .map(|_| key(rng.u64(..key_space)))
                            .collect();
                        let mut unique = keys.clone();
                        unique.sort_unstable_by_key(|k| k.0);
                        unique.dedup();
                        let run = 2 + rng.u32(..5);
                        let mut fresh = Vec::new();
                        map.insert_run(unique.iter().copied(), run, 10, 3, |k| fresh.push(k));
                        let mut expected_fresh = Vec::new();
                        for (i, &k) in unique.iter().enumerate() {
                            let e = Entry {
                                run,
                                offset: 10 + i as u32,
                                generation: 3,
                            };
                            if model.insert(k, e).is_none() {
                                expected_fresh.push(k);
                            }
                        }
                        assert_eq!(fresh, expected_fresh);
                    }
                }
                assert_eq!(map.len(), model.len());
                assert_eq!(map.get(k), model.get(&k).copied());
            }
            let mut entries: Vec<_> = map
                .iter()
                .map(|(k, e)| (k.0, e.run, e.offset, e.generation))
                .collect();
            let mut expected: Vec<_> = model
                .into_iter()
                .map(|(k, e)| (k.0, e.run, e.offset, e.generation))
                .collect();
            entries.sort_unstable();
            expected.sort_unstable();
            assert_eq!(entries, expected);
        }
    }
}
