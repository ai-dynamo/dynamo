// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-worker map from sequence block hash to the node that holds the block.
//!
//! Event workers touch one entry per stored or removed block, and a single event often
//! carries tens to hundreds of blocks. Each slot keeps its key and value together and
//! probes linearly, so a lookup usually touches one cache line, and the batch operations
//! prefetch slots a few keys ahead to overlap those misses.

use crate::protocols::ExternalSequenceBlockHash;

const MIN_CAPACITY: usize = 16;
const PREFETCH_DISTANCE: usize = 8;

pub(crate) struct BlockLookup<V> {
    /// Power-of-two length, or empty before the first insert. At most three quarters
    /// full, so every probe run ends at an empty slot.
    slots: Box<[Option<(ExternalSequenceBlockHash, V)>]>,
    len: usize,
    shift: u32,
}

impl<V> Default for BlockLookup<V> {
    fn default() -> Self {
        Self {
            slots: Box::default(),
            len: 0,
            shift: 64,
        }
    }
}

impl<V> BlockLookup<V> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn home(&self, key: ExternalSequenceBlockHash) -> usize {
        // Fibonacci hashing spreads structured keys; sequence hashes are already uniform.
        (key.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// The slot holding `key`, or else the empty slot that ends its probe run.
    /// Requires allocated slots.
    fn probe(&self, key: ExternalSequenceBlockHash) -> Result<usize, usize> {
        let mask = self.mask();
        let mut i = self.home(key);
        loop {
            match &self.slots[i] {
                None => return Err(i),
                Some((k, _)) if *k == key => return Ok(i),
                Some(_) => i = (i + 1) & mask,
            }
        }
    }

    fn find(&self, key: ExternalSequenceBlockHash) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        self.probe(key).ok()
    }

    /// Stores `key` in `empty`, the slot that ended its probe run, growing first if the
    /// table would pass its load limit.
    fn insert_vacant(&mut self, empty: usize, key: ExternalSequenceBlockHash, value: V) {
        let empty = if (self.len + 1) * 4 > self.slots.len() * 3 {
            self.reserve(1);
            self.probe(key).unwrap_err()
        } else {
            empty
        };
        self.slots[empty] = Some((key, value));
        self.len += 1;
    }

    pub(crate) fn get(&self, key: &ExternalSequenceBlockHash) -> Option<&V> {
        let i = self.find(*key)?;
        self.slots[i].as_ref().map(|(_, value)| value)
    }

    fn get_mut(&mut self, key: ExternalSequenceBlockHash) -> Option<&mut V> {
        let i = self.find(key)?;
        self.slots[i].as_mut().map(|(_, value)| value)
    }

    pub(crate) fn contains_key(&self, key: &ExternalSequenceBlockHash) -> bool {
        self.find(*key).is_some()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&ExternalSequenceBlockHash, &V)> {
        self.slots
            .iter()
            .filter_map(|slot| slot.as_ref().map(|(key, value)| (key, value)))
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
        let old = std::mem::replace(&mut self.slots, (0..capacity).map(|_| None).collect());
        self.shift = 64 - capacity.trailing_zeros();
        let mask = capacity - 1;
        for (key, value) in old.into_vec().into_iter().flatten() {
            let mut i = self.home(key);
            while self.slots[i].is_some() {
                i = (i + 1) & mask;
            }
            self.slots[i] = Some((key, value));
        }
    }

    pub(crate) fn insert(&mut self, key: ExternalSequenceBlockHash, value: V) -> Option<V> {
        self.reserve(1);
        match self.probe(key) {
            Ok(i) => self.slots[i]
                .as_mut()
                .map(|(_, existing)| std::mem::replace(existing, value)),
            Err(empty) => {
                self.insert_vacant(empty, key, value);
                None
            }
        }
    }

    pub(crate) fn remove(&mut self, key: &ExternalSequenceBlockHash) -> Option<V> {
        let mut hole = self.find(*key)?;
        let (_, value) = self.slots[hole].take()?;
        self.len -= 1;

        // Backward-shift deletion: pull later entries of the probe run into the hole
        // whenever the hole lies between their home slot and their current slot.
        let mask = self.mask();
        let mut j = hole;
        loop {
            j = (j + 1) & mask;
            let Some((k, _)) = &self.slots[j] else {
                break;
            };
            let home = self.home(*k);
            if (j.wrapping_sub(home) & mask) >= (j.wrapping_sub(hole) & mask) {
                self.slots[hole] = self.slots[j].take();
                hole = j;
            }
        }
        Some(value)
    }

    #[inline]
    fn prefetch(&self, key: ExternalSequenceBlockHash) {
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
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `prfm` is a hint that never faults, and `ptr` points into `slots`.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) ptr, options(nostack, preserves_flags, readonly));
        }
    }

    fn for_each_prefetched<I>(
        &mut self,
        keys: I,
        mut op: impl FnMut(&mut Self, ExternalSequenceBlockHash),
    ) where
        I: Iterator<Item = ExternalSequenceBlockHash> + Clone,
    {
        let mut ahead = keys.clone();
        for key in ahead.by_ref().take(PREFETCH_DISTANCE) {
            self.prefetch(key);
        }
        for key in keys {
            if let Some(next) = ahead.next() {
                self.prefetch(next);
            }
            op(self, key);
        }
    }

    /// Removes every key, calling `on_removed` for each, present or not, with the value
    /// it held.
    pub(crate) fn remove_all<I>(
        &mut self,
        keys: I,
        mut on_removed: impl FnMut(ExternalSequenceBlockHash, Option<V>),
    ) where
        I: Iterator<Item = ExternalSequenceBlockHash> + Clone,
    {
        self.for_each_prefetched(keys, |lookup, key| {
            let removed = lookup.remove(&key);
            on_removed(key, removed);
        });
    }
}

impl<V: Copy + Eq> BlockLookup<V> {
    /// Points every key at `value`, passing each other value it overwrites to
    /// `on_replaced`. Returns the number of entries inserted or changed.
    pub(crate) fn upsert_all<I>(
        &mut self,
        keys: I,
        value: V,
        mut on_replaced: impl FnMut(V),
    ) -> usize
    where
        I: Iterator<Item = ExternalSequenceBlockHash> + Clone,
    {
        self.reserve(1);
        let mut changed = 0;
        self.for_each_prefetched(keys, |lookup, key| match lookup.probe(key) {
            Ok(i) => {
                if let Some((_, existing)) = lookup.slots[i].as_mut()
                    && *existing != value
                {
                    on_replaced(std::mem::replace(existing, value));
                    changed += 1;
                }
            }
            Err(empty) => {
                lookup.insert_vacant(empty, key, value);
                changed += 1;
            }
        });
        changed
    }

    /// Points keys holding `from` at `to`, leaving missing keys and keys holding anything
    /// else alone: lookup repair treats a missing or differently placed entry as
    /// meaningful state. Returns the number changed.
    pub(crate) fn redirect(
        &mut self,
        keys: impl IntoIterator<Item = ExternalSequenceBlockHash>,
        from: V,
        to: V,
    ) -> usize {
        if from == to {
            return 0;
        }
        let mut changed = 0;
        for key in keys {
            if let Some(existing) = self.get_mut(key)
                && *existing == from
            {
                *existing = to;
                changed += 1;
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn key(k: u64) -> ExternalSequenceBlockHash {
        ExternalSequenceBlockHash(k)
    }

    /// Randomized differential test against `HashMap`. A small key space forces long
    /// probe runs, wraparound, and backward shifts across them.
    #[test]
    fn matches_hash_map_under_random_operations() {
        for seed in 0..16u64 {
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let mut lookup = BlockLookup::default();
            let mut model = HashMap::new();
            let key_space = 8 + seed * 17;

            for step in 0..20_000u64 {
                let k = key(next() % key_space);
                match next() % 10 {
                    0..=3 => assert_eq!(lookup.insert(k, step), model.insert(k, step)),
                    4..=6 => assert_eq!(lookup.remove(&k), model.remove(&k)),
                    7 => {
                        let keys: Vec<_> = (0..(next() % 12))
                            .map(|_| key(next() % key_space))
                            .collect();
                        // Reuse a recent value so some entries already hold it.
                        let value = step.saturating_sub(next() % 4);
                        let mut replaced = Vec::new();
                        let changed =
                            lookup.upsert_all(keys.iter().copied(), value, |v| replaced.push(v));
                        let mut expected = 0;
                        let mut expected_replaced = Vec::new();
                        for &k in &keys {
                            match model.insert(k, value) {
                                Some(old) if old == value => {}
                                Some(old) => {
                                    expected_replaced.push(old);
                                    expected += 1;
                                }
                                None => expected += 1,
                            }
                        }
                        assert_eq!(changed, expected);
                        assert_eq!(replaced, expected_replaced);
                    }
                    8 => {
                        let keys: Vec<_> = (0..(next() % 12))
                            .map(|_| key(next() % key_space))
                            .collect();
                        let mut seen = Vec::new();
                        lookup.remove_all(keys.iter().copied(), |k, v| seen.push((k, v)));
                        let expected: Vec<_> =
                            keys.iter().map(|&k| (k, model.remove(&k))).collect();
                        assert_eq!(seen, expected);
                    }
                    _ => {
                        let keys: Vec<_> = (0..(next() % 12))
                            .map(|_| key(next() % key_space))
                            .collect();
                        // Redirect from a recent value so some entries match it.
                        let from = step.saturating_sub(next() % 4);
                        let changed = lookup.redirect(keys.iter().copied(), from, step);
                        let mut expected = 0;
                        for k in &keys {
                            if let Some(v) = model.get_mut(k)
                                && *v == from
                                && *v != step
                            {
                                *v = step;
                                expected += 1;
                            }
                        }
                        assert_eq!(changed, expected);
                    }
                }
                assert_eq!(lookup.len(), model.len());
                assert_eq!(lookup.get(&k), model.get(&k));
            }

            let mut entries: Vec<_> = lookup.iter().map(|(k, v)| (*k, *v)).collect();
            let mut expected: Vec<_> = model.into_iter().collect();
            entries.sort();
            expected.sort();
            assert_eq!(entries, expected);
        }
    }
}
