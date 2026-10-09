// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A rank's map from sequence block hash to the position that holds the block.
//!
//! This is CRTC's `BlockLookup` (Fibonacci home slot, linear probing, at most three
//! quarters full, backward-shift deletion, batch prefetch) specialised to 16-byte
//! [`BlockPos`] slots, with a batch insert that writes consecutive offsets of one run.

use super::types::BlockPos;
use crate::protocols::ExternalSequenceBlockHash;

const MIN_CAPACITY: usize = 16;
const PREFETCH_DISTANCE: usize = 8;

type Entry = Option<(ExternalSequenceBlockHash, BlockPos)>;

pub(super) struct RankMap {
    /// Power-of-two length, or empty before the first insert.
    slots: Box<[Entry]>,
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
    pub(super) fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    fn home(&self, key: ExternalSequenceBlockHash) -> usize {
        (key.0.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// The slot holding `key`, or else the empty slot that ends its probe run.
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

    pub(super) fn get(&self, key: ExternalSequenceBlockHash) -> Option<BlockPos> {
        if self.len == 0 {
            return None;
        }
        let i = self.probe(key).ok()?;
        self.slots[i].map(|(_, pos)| pos)
    }

    pub(super) fn contains_key(&self, key: ExternalSequenceBlockHash) -> bool {
        self.get(key).is_some()
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (ExternalSequenceBlockHash, BlockPos)> + '_ {
        self.slots.iter().filter_map(|slot| *slot)
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
        let old = std::mem::replace(&mut self.slots, vec![None; capacity].into_boxed_slice());
        self.shift = 64 - capacity.trailing_zeros();
        let mask = capacity - 1;
        for (key, pos) in old.into_vec().into_iter().flatten() {
            let mut i = self.home(key);
            while self.slots[i].is_some() {
                i = (i + 1) & mask;
            }
            self.slots[i] = Some((key, pos));
        }
    }

    /// Points `key` at `pos`. Returns the previous position.
    pub(super) fn insert(
        &mut self,
        key: ExternalSequenceBlockHash,
        pos: BlockPos,
    ) -> Option<BlockPos> {
        self.reserve(1);
        match self.probe(key) {
            Ok(i) => self.slots[i]
                .as_mut()
                .map(|(_, existing)| std::mem::replace(existing, pos)),
            Err(empty) => {
                self.slots[empty] = Some((key, pos));
                self.len += 1;
                None
            }
        }
    }

    pub(super) fn remove(&mut self, key: ExternalSequenceBlockHash) -> Option<BlockPos> {
        if self.len == 0 {
            return None;
        }
        let mut hole = self.probe(key).ok()?;
        let (_, pos) = self.slots[hole].take()?;
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
        Some(pos)
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
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = ptr;
    }

    /// Points the `i`-th key at `(run, from + i)`. Returns how many entries changed.
    pub(super) fn insert_run<I>(&mut self, keys: I, run: u32, from: u32) -> usize
    where
        I: ExactSizeIterator<Item = ExternalSequenceBlockHash> + Clone,
    {
        self.reserve(keys.len());
        let mut ahead = keys.clone();
        for key in ahead.by_ref().take(PREFETCH_DISTANCE) {
            self.prefetch(key);
        }
        let mut changed = 0;
        for (i, key) in keys.enumerate() {
            if let Some(next) = ahead.next() {
                self.prefetch(next);
            }
            let pos = BlockPos::new(run, from + i as u32);
            match self.probe(key) {
                Ok(slot) => {
                    if let Some((_, existing)) = self.slots[slot].as_mut()
                        && *existing != pos
                    {
                        *existing = pos;
                        changed += 1;
                    }
                }
                Err(empty) => {
                    self.slots[empty] = Some((key, pos));
                    self.len += 1;
                    changed += 1;
                }
            }
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn key(k: u64) -> ExternalSequenceBlockHash {
        ExternalSequenceBlockHash(k)
    }

    #[test]
    fn matches_hash_map_under_random_operations() {
        for seed in 0..16u64 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut map = RankMap::default();
            let mut model = HashMap::new();
            let key_space = 8 + seed * 17;
            for step in 0..20_000u32 {
                let k = key(rng.u64(..key_space));
                let pos = BlockPos::new(2 + step % 7, step);
                match rng.u32(..10) {
                    0..=3 => assert_eq!(map.insert(k, pos), model.insert(k, pos)),
                    4..=6 => assert_eq!(map.remove(k), model.remove(&k)),
                    _ => {
                        let keys: Vec<_> = (0..rng.usize(..12))
                            .map(|_| key(rng.u64(..key_space)))
                            .collect();
                        let run = 2 + step % 5;
                        let changed = map.insert_run(keys.iter().copied(), run, step);
                        let mut expected = 0;
                        for (i, &k) in keys.iter().enumerate() {
                            let pos = BlockPos::new(run, step + i as u32);
                            if model.insert(k, pos) != Some(pos) {
                                expected += 1;
                            }
                        }
                        assert_eq!(changed, expected);
                    }
                }
                assert_eq!(map.len(), model.len());
                assert_eq!(map.get(k), model.get(&k).copied());
            }
            let mut entries: Vec<_> = map.iter().map(|(k, p)| (k.0, p.run(), p.offset)).collect();
            let mut expected: Vec<_> = model
                .into_iter()
                .map(|(k, p)| (k.0, p.run(), p.offset))
                .collect();
            entries.sort_unstable();
            expected.sort_unstable();
            assert_eq!(entries, expected);
        }
    }
}
