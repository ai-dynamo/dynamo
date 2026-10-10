// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The word arena (spec B 4.3): one `u32`-addressed space of `AtomicU64` words in
//! geometrically growing segments that are never freed while the index lives, so a stale
//! reader that follows a recycled address reads memory that is still allocated and fails
//! validation instead of faulting.
//!
//! Allocation is by size class with one lock-free free list per kind and class. Addresses
//! popped from a free list are fenced before their first write (A1).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam_queue::SegQueue;

use super::protocol::reuse_fence;
use crate::protocols::KvCacheEventError;

/// A word address; `0` is never handed out and means none.
pub(crate) type Addr = u32;
pub(crate) const NONE: Addr = 0;

/// Segment `k` holds `2^(FIRST_SEGMENT_BITS + k)` words.
const FIRST_SEGMENT_BITS: u32 = 16;
/// 16 segments cover addresses below `2^32 - 2^16`.
const SEGMENTS: usize = 16;

const fn segment_start(k: usize) -> u64 {
    (1u64 << (FIRST_SEGMENT_BITS + k as u32)) - (1u64 << FIRST_SEGMENT_BITS)
}

const fn segment_len(k: usize) -> u64 {
    1u64 << (FIRST_SEGMENT_BITS + k as u32)
}

const LIMIT: u64 = segment_start(SEGMENTS);

/// `(segment, index)` of an absolute word address below [`LIMIT`].
#[inline]
fn locate(addr: u64) -> (usize, usize) {
    let x = addr + (1u64 << FIRST_SEGMENT_BITS);
    let k = 63 - x.leading_zeros() - FIRST_SEGMENT_BITS;
    (
        k as usize,
        (x - (1u64 << (FIRST_SEGMENT_BITS + k))) as usize,
    )
}

const _: () = assert!(std::mem::align_of::<u64>() == std::mem::align_of::<AtomicU64>());
const _: () = assert!(std::mem::size_of::<u64>() == std::mem::size_of::<AtomicU64>());

/// `n` zeroed words, backed by `calloc`-style memory so untouched pages stay unmapped.
fn zeroed_words(n: usize) -> Box<[AtomicU64]> {
    let words: Box<[u64]> = vec![0u64; n].into_boxed_slice();
    // SAFETY: `AtomicU64` has the same size and bit validity as `u64`, and the assertions
    // above pin equal alignment, so the slice layout is unchanged.
    unsafe { Box::from_raw(Box::into_raw(words) as *mut [AtomicU64]) }
}

/// What an allocation holds; each kind has its own size classes and free lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Hash array: two header words, then `capacity` local and `capacity` ext words.
    Array,
    /// Child table: two header words, then `slots` key/value pairs.
    Children,
    /// Cutoff table: one header word, then `slots` packed entries.
    Cutoffs,
    /// Forwarding records: one header word, then `capacity` two-word records.
    Forwards,
    /// Coverage overflow chunk: a link word and four bit words.
    Chunk,
}

pub(crate) const ARRAY_CLASSES: usize = 71;
pub(crate) const CHILD_CLASSES: usize = 28;
pub(crate) const CUTOFF_CLASSES: usize = 6;
pub(crate) const FORWARD_CLASSES: usize = 19;
const CHUNK_CLASSES: usize = 1;
pub(crate) const CHUNK_WORDS_TOTAL: u32 = 5;

const fn kind_base(kind: Kind) -> usize {
    match kind {
        Kind::Array => 0,
        Kind::Children => ARRAY_CLASSES,
        Kind::Cutoffs => ARRAY_CLASSES + CHILD_CLASSES,
        Kind::Forwards => ARRAY_CLASSES + CHILD_CLASSES + CUTOFF_CLASSES,
        Kind::Chunk => ARRAY_CLASSES + CHILD_CLASSES + CUTOFF_CLASSES + FORWARD_CLASSES,
    }
}

const ALL_CLASSES: usize =
    ARRAY_CLASSES + CHILD_CLASSES + CUTOFF_CLASSES + FORWARD_CLASSES + CHUNK_CLASSES;

/// Hash-array capacity of class `class`: 2, 4, 8, then steps of 8 up to 128, then four
/// classes per octave, so rounding wastes at most a quarter.
pub(crate) const fn array_class_capacity(class: usize) -> u32 {
    match class {
        0 => 2,
        1 => 4,
        2 => 8,
        3..=17 => 8 * (class as u32 - 1),
        _ => {
            let octave = 7 + (class as u32 - 18) / 4;
            let quarter = (class as u32 - 18) % 4 + 1;
            (1 << octave) + quarter * (1 << (octave - 2))
        }
    }
}

/// The smallest array class holding `capacity` blocks.
pub(crate) fn array_class_for(capacity: u32) -> Option<usize> {
    let class = match capacity {
        0..=2 => 0,
        3..=4 => 1,
        5..=8 => 2,
        9..=128 => capacity.div_ceil(8) as usize + 1,
        _ => {
            let octave = 31 - (capacity - 1).leading_zeros();
            let step = 1u32 << (octave - 2);
            let quarter = (capacity - (1 << octave)).div_ceil(step);
            18 + (octave as usize - 7) * 4 + quarter as usize - 1
        }
    };
    (class < ARRAY_CLASSES).then_some(class)
}

/// Words of one allocation of `kind` in `class`.
pub(crate) const fn class_words(kind: Kind, class: usize) -> u32 {
    match kind {
        Kind::Array => 2 + 2 * array_class_capacity(class),
        Kind::Children => 2 + 2 * (4u32 << class),
        Kind::Cutoffs => 1 + (2u32 << class),
        Kind::Forwards => 1 + 2 * (1u32 << class),
        Kind::Chunk => CHUNK_WORDS_TOTAL,
    }
}

/// Byte totals for the memory report.
#[cfg(any(test, feature = "bench"))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ArenaBytes {
    pub(crate) reserved: u64,
    pub(crate) free_listed: u64,
    pub(crate) stranded: u64,
}

pub(crate) struct Arena {
    segments: [OnceLock<Box<[AtomicU64]>>; SEGMENTS],
    /// Next never-allocated word.
    cursor: AtomicU64,
    free: Box<[SegQueue<Addr>]>,
    /// Segment-tail words too small for any class.
    stranded: AtomicU64,
}

impl Default for Arena {
    fn default() -> Self {
        Self {
            segments: std::array::from_fn(|_| OnceLock::new()),
            cursor: AtomicU64::new(1),
            free: (0..ALL_CLASSES).map(|_| SegQueue::new()).collect(),
            stranded: AtomicU64::new(0),
        }
    }
}

impl Arena {
    /// The words `[addr, addr + len)`, if they lie in one allocated segment. Readers must
    /// treat `None` as a torn read: it never panics on a garbage address (R8).
    #[inline]
    pub(crate) fn slice(&self, addr: Addr, len: usize) -> Option<&[AtomicU64]> {
        if addr == NONE {
            return None;
        }
        let (k, index) = locate(u64::from(addr));
        let segment = self.segments.get(k)?.get()?;
        segment.get(index..index.checked_add(len)?)
    }

    /// One word, bounds-checked.
    #[inline]
    pub(crate) fn word(&self, addr: Addr) -> Option<&AtomicU64> {
        self.slice(addr, 1).map(|words| &words[0])
    }

    fn ensure_segment(&self, k: usize) -> &[AtomicU64] {
        self.segments[k].get_or_init(|| zeroed_words(segment_len(k) as usize))
    }

    /// Allocates one `kind` block of `class`, popping a free one first. Contents are
    /// whatever the previous owner left unless the block is fresh, in which case it is zero.
    pub(crate) fn alloc(&self, kind: Kind, class: usize) -> Result<Addr, KvCacheEventError> {
        if let Some(addr) = self.free[kind_base(kind) + class].pop() {
            reuse_fence();
            return Ok(addr);
        }
        self.bump(class_words(kind, class))
    }

    /// Allocates a hash array that holds at least `capacity` blocks. A freed array of the
    /// same class or up to one octave above is reused first. Returns the address and the
    /// class.
    pub(crate) fn alloc_array(&self, capacity: u32) -> Result<(Addr, usize), KvCacheEventError> {
        let class = array_class_for(capacity).ok_or(KvCacheEventError::CapacityExhausted)?;
        let limit = 2 * array_class_capacity(class);
        let mut candidate = class;
        while candidate < ARRAY_CLASSES && array_class_capacity(candidate) <= limit {
            if let Some(addr) = self.free[kind_base(Kind::Array) + candidate].pop() {
                reuse_fence();
                return Ok((addr, candidate));
            }
            candidate += 1;
        }
        Ok((self.bump(class_words(Kind::Array, class))?, class))
    }

    /// Returns a block to its free list. The caller holds no run lock (A2).
    pub(crate) fn free(&self, kind: Kind, class: usize, addr: Addr) {
        debug_assert_ne!(addr, NONE);
        self.free[kind_base(kind) + class].push(addr);
    }

    fn bump(&self, words: u32) -> Result<Addr, KvCacheEventError> {
        let words = u64::from(words);
        let mut cur = self.cursor.load(Ordering::Relaxed);
        loop {
            if cur >= LIMIT {
                return Err(KvCacheEventError::CapacityExhausted);
            }
            let (k, _) = locate(cur);
            let end = segment_start(k + 1);
            if cur + words <= end {
                match self.cursor.compare_exchange_weak(
                    cur,
                    cur + words,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        self.ensure_segment(k);
                        return Ok(cur as Addr);
                    }
                    Err(actual) => {
                        cur = actual;
                        continue;
                    }
                }
            }
            let Some(next) = (k + 1..SEGMENTS).find(|&next| segment_len(next) >= words) else {
                return Err(KvCacheEventError::CapacityExhausted);
            };
            let start = segment_start(next);
            match self.cursor.compare_exchange_weak(
                cur,
                start + words,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.recycle_tail(k, cur, end);
                    self.ensure_segment(next);
                    return Ok(start as Addr);
                }
                Err(actual) => cur = actual,
            }
        }
    }

    /// Puts the unused tail of segment `k` on the array free lists, largest class first.
    fn recycle_tail(&self, k: usize, mut from: u64, end: u64) {
        if self.segments[k].get().is_none() {
            // Nothing was ever handed out in this segment, so nothing was reserved.
            return;
        }
        for class in (0..ARRAY_CLASSES).rev() {
            let words = u64::from(class_words(Kind::Array, class));
            while from + words <= end {
                self.free(Kind::Array, class, from as Addr);
                from += words;
            }
        }
        self.stranded.fetch_add(end - from, Ordering::Relaxed);
    }

    /// Every free-listed block, as `(kind, class, addr)`, popped and pushed back. Only for
    /// quiescent checks.
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn free_blocks(&self) -> Vec<(Kind, usize, Addr)> {
        let kinds = [
            (Kind::Array, ARRAY_CLASSES),
            (Kind::Children, CHILD_CLASSES),
            (Kind::Cutoffs, CUTOFF_CLASSES),
            (Kind::Forwards, FORWARD_CLASSES),
            (Kind::Chunk, CHUNK_CLASSES),
        ];
        let mut blocks = Vec::new();
        for (kind, classes) in kinds {
            for class in 0..classes {
                let list = &self.free[kind_base(kind) + class];
                let mut popped = Vec::with_capacity(list.len());
                while let Some(addr) = list.pop() {
                    popped.push(addr);
                }
                for &addr in &popped {
                    blocks.push((kind, class, addr));
                    list.push(addr);
                }
            }
        }
        blocks
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn bytes(&self) -> ArenaBytes {
        let reserved_words: u64 = (0..SEGMENTS)
            .filter(|&k| self.segments[k].get().is_some())
            .map(segment_len)
            .sum();
        let mut free_words = 0u64;
        for (kind, classes) in [
            (Kind::Array, ARRAY_CLASSES),
            (Kind::Children, CHILD_CLASSES),
            (Kind::Cutoffs, CUTOFF_CLASSES),
            (Kind::Forwards, FORWARD_CLASSES),
            (Kind::Chunk, CHUNK_CLASSES),
        ] {
            for class in 0..classes {
                free_words += self.free[kind_base(kind) + class].len() as u64
                    * u64::from(class_words(kind, class));
            }
        }
        ArenaBytes {
            reserved: reserved_words * 8,
            free_listed: free_words * 8,
            stranded: self.stranded.load(Ordering::Relaxed) * 8,
        }
    }

    /// Words the bump allocator has passed in initialized segments, segment tails included,
    /// so `live + free-listed + stranded == bumped` at quiescence when nothing leaks.
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn bumped_words(&self) -> u64 {
        let cursor = self.cursor.load(Ordering::Relaxed).min(LIMIT);
        let mut total = 0;
        for k in 0..SEGMENTS {
            if self.segments[k].get().is_none() {
                continue;
            }
            let start = segment_start(k).max(1);
            let end = segment_start(k + 1).min(cursor.max(start));
            total += end - start;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locate_matches_segment_bounds() {
        assert_eq!(locate(0), (0, 0));
        assert_eq!(locate((1 << 16) - 1), (0, (1 << 16) - 1));
        assert_eq!(locate(1 << 16), (1, 0));
        assert_eq!(locate(segment_start(5) + 7), (5, 7));
        assert_eq!(locate(LIMIT - 1).0, SEGMENTS - 1);
        assert!(LIMIT <= u64::from(u32::MAX));
    }

    #[test]
    fn array_classes_round_up_by_at_most_a_quarter() {
        let mut previous = 0;
        for class in 0..ARRAY_CLASSES {
            let capacity = array_class_capacity(class);
            assert!(capacity > previous, "class {class} does not grow");
            assert_eq!(array_class_for(capacity), Some(class));
            assert_eq!(array_class_for(previous + 1), Some(class));
            // Steps of 8 up to 128, then four classes per octave: a quarter at most.
            let slack = capacity - previous - 1;
            if previous >= 128 {
                assert!(
                    slack * 4 <= capacity,
                    "class {class}: {previous}+1 rounds to {capacity}"
                );
            } else if previous >= 8 {
                assert!(
                    slack < 8,
                    "class {class}: {previous}+1 rounds to {capacity}"
                );
            }
            previous = capacity;
        }
        // The largest run's array fits, with the growth slack of `capacity_for`.
        let largest = super::super::runs::MAX_RUN_LEN + super::super::runs::MAX_RUN_LEN / 8;
        assert!(array_class_for(largest).is_some());
        assert!(
            u64::from(class_words(Kind::Array, ARRAY_CLASSES - 1)) <= segment_len(SEGMENTS - 1)
        );
        assert!(
            u64::from(class_words(Kind::Children, CHILD_CLASSES - 1)) <= segment_len(SEGMENTS - 1)
        );
    }

    #[test]
    fn allocations_never_span_segments_and_tails_are_recycled() {
        let arena = Arena::default();
        let big = ARRAY_CLASSES - 20;
        let words = class_words(Kind::Array, big) as usize;
        let mut seen = Vec::new();
        for _ in 0..40 {
            let addr = arena.alloc(Kind::Array, big).unwrap();
            assert!(
                arena.slice(addr, words).is_some(),
                "allocation spans a segment"
            );
            seen.push(addr);
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 40);
        // Tails went to free lists: small allocations now come from them.
        let bytes = arena.bytes();
        assert!(bytes.free_listed > 0 || bytes.stranded > 0);
        let blocks = arena.free_blocks();
        for (kind, class, addr) in blocks {
            let words = class_words(kind, class) as usize;
            assert!(arena.slice(addr, words).is_some());
        }
    }

    #[test]
    fn freed_blocks_are_reused_within_an_octave() {
        let arena = Arena::default();
        let (addr, class) = arena.alloc_array(100).unwrap();
        arena.free(Kind::Array, class, addr);
        // 60 rounds to class 64, whose octave reaches 128 >= class of 100 (104).
        let (again, again_class) = arena.alloc_array(60).unwrap();
        assert_eq!((again, again_class), (addr, class));
        let (fresh, _) = arena.alloc_array(10).unwrap();
        assert_ne!(fresh, addr);
    }

    #[test]
    fn garbage_addresses_read_as_absent() {
        let arena = Arena::default();
        assert!(arena.slice(NONE, 1).is_none());
        assert!(arena.slice(u32::MAX, 1).is_none());
        assert!(
            arena.slice(5, 1).is_none(),
            "segment 0 is not allocated yet"
        );
        arena.alloc(Kind::Chunk, 0).unwrap();
        assert!(arena.slice(5, 1).is_some());
        assert!(arena.slice((1 << 16) - 1, 2).is_none());
    }
}
