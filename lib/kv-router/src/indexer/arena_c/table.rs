// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Child tables: open addressing with linear probing over arena words.
//!
//! ```text
//! word 0: slots: u32 | used: u32 << 32   // used = live + tombstones + claimed
//! word 1: live: u32
//! then slots x { key: u64, value: u64 }  // value = run id | generation << 32
//! ```
//!
//! A key is never [`EMPTY`] or [`TOMB`]. A value of [`EMPTY`] is a claim in flight and
//! [`TOMB`] an unlinked child. A slot's key is written once per table: a key appears in
//! at most one slot, and an insert of a key whose slot is a tombstone reuses that slot.
//!
//! Concurrency, as the backend uses it:
//! - Claims ([`Table::claim`]) run under the owning run's shared gate. They reserve
//!   budget in `used`, compare-and-swap an [`EMPTY`] key to theirs, then publish the
//!   value with `Release`.
//! - Locked inserts and rebuilds run under the exclusive gate (and the state write
//!   lock), so they never overlap a claim. A rebuild sizes the new table for three
//!   eighths load and drops every tombstone.
//! - Unlinks store [`TOMB`] into the value under the owning run's shared gate.
//! - Readers ([`Table::find`]) skip slots whose key differs, treat an unpublished value
//!   as absent and continue past tombstones.
//!
//! This file only uses `super::sync`, so the loom models compile it unchanged.

use super::sync::{AtomicU64, Ordering};

pub(super) const EMPTY: u64 = 0;
pub(super) const TOMB: u64 = u64::MAX;
pub(super) const HEADER_WORDS: usize = 2;
/// Smallest table; a run's first child table.
pub(super) const MIN_SLOTS: u32 = 4;

/// Words a table of `slots` slots occupies.
pub(super) const fn words_for(slots: u32) -> usize {
    HEADER_WORDS + 2 * slots as usize
}

/// Claims may fill a table to three quarters of its slots.
pub(super) const fn claim_budget(slots: u32) -> u32 {
    slots / 4 * 3
}

/// Smallest power-of-two slot count that keeps `live` at or below three eighths of the
/// slots, the size a rebuild picks.
pub(super) fn slots_for_live(live: u32) -> u32 {
    let mut slots = MIN_SLOTS;
    while u64::from(live) * 8 > u64::from(slots) * 3 {
        slots *= 2;
    }
    slots
}

#[inline]
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
fn offset_mix(offset: u32) -> u64 {
    mix(u64::from(offset) ^ 0x9E37_79B9_7F4A_7C15)
}

/// Forces the low bit on and the high bit off, so a key is never [`EMPTY`] or [`TOMB`].
#[inline]
fn force(key: u64) -> u64 {
    (key | 1) & !(1 << 63)
}

/// Key of the child at `offset` of its parent whose first local hash is `head`.
///
/// The offset and the head are mixed separately and combined with XOR, so a split can
/// move a child to a new offset with [`rekey`] without reading the child's head.
#[inline]
pub(super) fn child_key(offset: u32, head: u64) -> u64 {
    force(mix(head) ^ offset_mix(offset))
}

/// The key of the same child moved from offset `from` to offset `to`.
#[inline]
pub(super) fn rekey(key: u64, from: u32, to: u32) -> u64 {
    force(key ^ offset_mix(from) ^ offset_mix(to))
}

#[inline]
pub(super) fn pack(run: u32, generation: u32) -> u64 {
    u64::from(run) | (u64::from(generation) << 32)
}

#[inline]
pub(super) fn unpack_run(value: u64) -> u32 {
    value as u32
}

#[inline]
pub(super) fn unpack_generation(value: u64) -> u32 {
    (value >> 32) as u32
}

/// Outcome of a lock-free claim.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Claim {
    /// The slot is ours and its value is published.
    Claimed,
    /// A published child already has this key.
    Exists(u64),
    /// Another claim of this key is in flight; retry after it publishes.
    Busy,
    /// The claim budget is spent; take the locked path.
    Full,
}

/// Outcome of an insert under the exclusive gate.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Locked {
    Inserted,
    Exists(u64),
    Full,
}

/// A child table laid out over `words`.
#[derive(Clone, Copy)]
pub(super) struct Table<'a> {
    words: &'a [AtomicU64],
}

impl<'a> Table<'a> {
    /// `words` must be [`words_for`] of the table's slot count.
    #[inline]
    pub(super) fn new(words: &'a [AtomicU64]) -> Self {
        debug_assert!(words.len() >= HEADER_WORDS);
        Self { words }
    }

    /// Writes an empty table header. Every slot word must already be zero.
    pub(super) fn init(words: &[AtomicU64], slots: u32) {
        debug_assert!(slots.is_power_of_two() && slots >= MIN_SLOTS);
        debug_assert_eq!(words.len(), words_for(slots));
        words[0].store(u64::from(slots), Ordering::Relaxed);
        words[1].store(0, Ordering::Relaxed);
    }

    /// Slot count stored in a table's first word.
    #[inline]
    pub(super) fn slots_in(header: u64) -> u32 {
        header as u32
    }

    #[inline]
    pub(super) fn slots(&self) -> u32 {
        Self::slots_in(self.words[0].load(Ordering::Relaxed))
    }

    #[cfg(test)]
    pub(super) fn used(&self) -> u32 {
        (self.words[0].load(Ordering::Relaxed) >> 32) as u32
    }

    #[inline]
    pub(super) fn live(&self) -> u32 {
        self.words[1].load(Ordering::Relaxed) as u32
    }

    #[inline]
    fn key_word(&self, slot: usize) -> &'a AtomicU64 {
        &self.words[HEADER_WORDS + 2 * slot]
    }

    #[inline]
    fn value_word(&self, slot: usize) -> &'a AtomicU64 {
        &self.words[HEADER_WORDS + 2 * slot + 1]
    }

    #[inline]
    fn home(&self, key: u64) -> usize {
        ((key >> 1) as usize) & (self.slots() as usize - 1)
    }

    /// The published value for `key`, if any. Unpublished claims read as absent.
    #[inline]
    pub(super) fn find(&self, key: u64) -> Option<u64> {
        let slots = self.slots() as usize;
        let mask = slots - 1;
        let mut i = self.home(key);
        for _ in 0..slots {
            let k = self.key_word(i).load(Ordering::Acquire);
            if k == EMPTY {
                return None;
            }
            if k == key {
                let value = self.value_word(i).load(Ordering::Acquire);
                if value != EMPTY && value != TOMB {
                    return Some(value);
                }
            }
            i = (i + 1) & mask;
        }
        None
    }

    /// Reserves one unit of claim budget. Fails at three quarters of the slots.
    fn reserve(&self) -> bool {
        let header = &self.words[0];
        let mut current = header.load(Ordering::Relaxed);
        loop {
            let slots = Self::slots_in(current);
            let used = (current >> 32) as u32;
            if used + 1 > claim_budget(slots) {
                return false;
            }
            match header.compare_exchange_weak(
                current,
                current + (1 << 32),
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(seen) => current = seen,
            }
        }
    }

    fn unreserve(&self) {
        self.words[0].fetch_sub(1 << 32, Ordering::AcqRel);
    }

    /// Claims an [`EMPTY`] slot for `key` and publishes `value` there, unless a child
    /// with `key` is published or in flight. Callers hold the owning run's shared gate.
    pub(super) fn claim(&self, key: u64, value: u64) -> Claim {
        debug_assert!(key != EMPTY && key != TOMB);
        debug_assert!(value != EMPTY && value != TOMB);
        let slots = self.slots() as usize;
        let mask = slots - 1;
        let mut i = self.home(key);
        let mut reserved = false;
        let mut steps = 0;
        let outcome = loop {
            if steps == slots {
                break Claim::Full;
            }
            let key_word = self.key_word(i);
            let k = key_word.load(Ordering::Acquire);
            if k == EMPTY {
                if !reserved {
                    if !self.reserve() {
                        return Claim::Full;
                    }
                    reserved = true;
                }
                match key_word.compare_exchange(EMPTY, key, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        self.value_word(i).store(value, Ordering::Release);
                        self.words[1].fetch_add(1, Ordering::Relaxed);
                        return Claim::Claimed;
                    }
                    // Look at this slot again under the winner's key.
                    Err(_) => continue,
                }
            }
            if k == key {
                let value_word = self.value_word(i);
                let current = value_word.load(Ordering::Acquire);
                if current == EMPTY {
                    break Claim::Busy;
                }
                if current != TOMB {
                    break Claim::Exists(current);
                }
                // A tombstone of this very key: reuse it. The key never changes, so a
                // lock-free reader sees either the tombstone or the new child.
                match value_word.compare_exchange(TOMB, value, Ordering::AcqRel, Ordering::Acquire)
                {
                    Ok(_) => {
                        self.words[1].fetch_add(1, Ordering::Relaxed);
                        break Claim::Claimed;
                    }
                    Err(_) => continue,
                }
            }
            i = (i + 1) & mask;
            steps += 1;
        };
        if reserved {
            self.unreserve();
        }
        outcome
    }

    /// Inserts `key` with every claimer excluded: reuses a tombstone of the same key,
    /// else takes an [`EMPTY`] slot within the claim budget. Other keys' tombstones are
    /// never reused, so lock-free readers never see a slot change keys; past the budget
    /// the caller rebuilds, which purges tombstones.
    pub(super) fn insert_locked(&self, key: u64, value: u64) -> Locked {
        let slots = self.slots() as usize;
        let mask = slots - 1;
        let mut i = self.home(key);
        for _ in 0..slots {
            let k = self.key_word(i).load(Ordering::Relaxed);
            if k == EMPTY {
                if !self.reserve() {
                    return Locked::Full;
                }
                self.key_word(i).store(key, Ordering::Release);
                self.value_word(i).store(value, Ordering::Release);
                self.words[1].fetch_add(1, Ordering::Relaxed);
                return Locked::Inserted;
            }
            if k == key {
                let current = self.value_word(i).load(Ordering::Relaxed);
                if current != TOMB {
                    return Locked::Exists(current);
                }
                self.value_word(i).store(value, Ordering::Release);
                self.words[1].fetch_add(1, Ordering::Relaxed);
                return Locked::Inserted;
            }
            i = (i + 1) & mask;
        }
        Locked::Full
    }

    /// Tombstones the slot holding `key` with `value`. Finds it by probing for the key;
    /// never scans the table. Returns whether it was found.
    pub(super) fn unlink(&self, key: u64, value: u64) -> bool {
        let slots = self.slots() as usize;
        let mask = slots - 1;
        let mut i = self.home(key);
        for _ in 0..slots {
            let k = self.key_word(i).load(Ordering::Acquire);
            if k == EMPTY {
                return false;
            }
            if k == key
                && self
                    .value_word(i)
                    .compare_exchange(value, TOMB, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
            {
                self.words[1].fetch_sub(1, Ordering::Relaxed);
                return true;
            }
            i = (i + 1) & mask;
        }
        false
    }

    /// Published `(key, value)` pairs.
    pub(super) fn entries(&self) -> impl Iterator<Item = (u64, u64)> + 'a {
        let table = *self;
        (0..table.slots() as usize).filter_map(move |i| {
            let k = table.key_word(i).load(Ordering::Acquire);
            if k == EMPTY {
                return None;
            }
            let value = table.value_word(i).load(Ordering::Acquire);
            (value != EMPTY && value != TOMB).then_some((k, value))
        })
    }

    /// Slots whose value is [`TOMB`].
    #[cfg(any(test, feature = "bench"))]
    pub(super) fn tombstones(&self) -> usize {
        (0..self.slots() as usize)
            .filter(|&i| {
                self.key_word(i).load(Ordering::Relaxed) != EMPTY
                    && self.value_word(i).load(Ordering::Relaxed) == TOMB
            })
            .count()
    }

    /// Whether slot `slot` holds a key (live, claimed or tombstoned).
    #[cfg(test)]
    pub(super) fn occupied(&self, slot: usize) -> bool {
        self.key_word(slot).load(Ordering::Relaxed) != EMPTY
    }

    /// Longest probe run from any key's home slot to the key, for the shape report.
    #[cfg(any(test, feature = "bench"))]
    pub(super) fn max_probe(&self) -> usize {
        let mask = self.slots() as usize - 1;
        (0..self.slots() as usize)
            .filter_map(|i| {
                let k = self.key_word(i).load(Ordering::Relaxed);
                (k != EMPTY).then(|| (i.wrapping_sub(self.home(k))) & mask)
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_words(slots: u32) -> Vec<AtomicU64> {
        let words: Vec<_> = (0..words_for(slots)).map(|_| AtomicU64::new(0)).collect();
        Table::init(&words, slots);
        words
    }

    #[test]
    fn keys_avoid_sentinels_and_rekey_round_trips() {
        for head in [0, 1, u64::MAX, 0x8000_0000_0000_0000, 12345] {
            for offset in [0, 1, 7, 1 << 20] {
                let key = child_key(offset, head);
                assert_ne!(key, EMPTY);
                assert_ne!(key, TOMB);
                for to in [0, 3, 99] {
                    assert_eq!(rekey(key, offset, to), child_key(to, head));
                }
            }
        }
        // The same head at two offsets gets two keys.
        assert_ne!(child_key(1, 42), child_key(2, 42));
    }

    #[test]
    fn claims_stop_at_three_quarters_and_find_skips_tombstones() {
        let words = table_words(8);
        let table = Table::new(&words);
        let mut claimed = Vec::new();
        for head in 0..10u64 {
            let key = child_key(0, head);
            match table.claim(key, pack(head as u32 + 2, 1)) {
                Claim::Claimed => claimed.push(key),
                Claim::Full => break,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(claimed.len(), claim_budget(8) as usize);
        assert_eq!(table.used(), 6);
        assert_eq!(table.live(), 6);
        assert_eq!(
            table.claim(claimed[0], pack(99, 1)),
            Claim::Exists(pack(2, 1))
        );

        assert!(table.unlink(claimed[0], pack(2, 1)));
        assert!(!table.unlink(claimed[0], pack(2, 1)));
        assert_eq!(table.find(claimed[0]), None);
        assert_eq!(table.live(), 5);
        assert_eq!(table.used(), 6);
        assert_eq!(table.tombstones(), 1);
        for (i, &key) in claimed.iter().enumerate().skip(1) {
            assert_eq!(table.find(key), Some(pack(i as u32 + 2, 1)));
        }
        // The budget counts tombstones, so a new key cannot claim or insert; the same key
        // reuses its tombstone either way.
        assert_eq!(table.claim(child_key(0, 77), pack(77, 1)), Claim::Full);
        assert_eq!(
            table.insert_locked(child_key(0, 77), pack(77, 1)),
            Locked::Full
        );
        assert_eq!(table.claim(claimed[0], pack(77, 1)), Claim::Claimed);
        assert_eq!(table.tombstones(), 0);
        assert_eq!(table.used(), 6);
        assert_eq!(table.live(), 6);
        assert!(table.unlink(claimed[0], pack(77, 1)));
        assert_eq!(
            table.insert_locked(claimed[0], pack(78, 1)),
            Locked::Inserted
        );
        assert_eq!(table.find(claimed[0]), Some(pack(78, 1)));
    }

    #[test]
    fn rebuild_size_keeps_live_at_three_eighths() {
        assert_eq!(slots_for_live(0), MIN_SLOTS);
        assert_eq!(slots_for_live(1), 4);
        assert_eq!(slots_for_live(2), 8);
        assert_eq!(slots_for_live(3), 8);
        assert_eq!(slots_for_live(4), 16);
        assert_eq!(slots_for_live(3000), 8192);
    }

    #[test]
    fn in_flight_claim_reads_as_absent_and_busy() {
        let words = table_words(4);
        let table = Table::new(&words);
        let key = child_key(3, 9);
        // Simulate a claimer that has swapped the key in but not published yet.
        let i = table.home(key);
        assert!(table.reserve());
        words[HEADER_WORDS + 2 * i].store(key, Ordering::Release);
        assert_eq!(table.find(key), None);
        assert_eq!(table.claim(key, pack(5, 1)), Claim::Busy);
        words[HEADER_WORDS + 2 * i + 1].store(pack(4, 1), Ordering::Release);
        assert_eq!(table.find(key), Some(pack(4, 1)));
        assert_eq!(table.claim(key, pack(5, 1)), Claim::Exists(pack(4, 1)));
    }
}
