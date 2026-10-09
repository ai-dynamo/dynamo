// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The run slab and the word arena, plus epoch-deferred reuse of both.
//!
//! - The slab holds run headers in segments of `1024 * 2^k` headers that are never
//!   freed while the index lives; a run id maps to `(segment, index)` with leading-zero
//!   arithmetic.
//! - The arena is one `u32` address space of `AtomicU64` words in segments of
//!   `2^(16 + k)` words. An allocation never spans two segments; when the current
//!   segment cannot fit a request, its tail is stranded and allocation moves on. Word 0
//!   is never handed out, so address 0 means none.
//! - Allocations are rounded up to a size class and recycled through one lock-free free
//!   list per class. Nothing is ever returned to the OS.
//! - Every free goes through a lane's [`FreeBatch`], which hands it to
//!   `crossbeam-epoch`: an id or a word is reused only after every thread pinned before
//!   its release has unpinned.

use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering};

use crossbeam_queue::SegQueue;
use parking_lot::Mutex;

use super::run::RunHeader;
use crate::protocols::KvCacheEventError;

/// Words in arena segment 0; segment `k` holds `2^(SEG0_BITS + k)` words.
const SEG0_BITS: u32 = 16;
/// Enough segments to cover the whole `u32` address space.
const ARENA_SEGMENTS: usize = 17;
/// Headers in slab segment 0; segment `k` holds `SLAB0 * 2^k` headers.
const SLAB0: usize = 1024;
/// Enough segments to cover every `u32` run id.
const SLAB_SEGMENTS: usize = 23;
/// Even sizes up to 64 words, then four classes per octave.
pub(super) const CLASSES: usize = 136;
/// Items a lane collects before it defers them in one epoch closure.
const FREE_BATCH: usize = 64;

/// Size class index and rounded size, in words, of a `words`-word allocation.
pub(super) fn class_of(words: usize) -> Option<(u8, usize)> {
    let words = words.max(2);
    if words <= 64 {
        let rounded = words.next_multiple_of(2);
        return Some(((rounded / 2 - 1) as u8, rounded));
    }
    let octave = (usize::BITS - 1 - (words - 1).leading_zeros()) as usize;
    let step = 1usize << (octave - 2);
    let rounded = words.next_multiple_of(step);
    let class = 32 + 4 * (octave - 6) + (rounded / step - 5);
    (class < CLASSES).then_some((class as u8, rounded))
}

pub(super) fn class_words(class: u8) -> usize {
    let class = class as usize;
    if class < 32 {
        return 2 * (class + 1);
    }
    let octave = (class - 32) / 4 + 6;
    let step = 1usize << (octave - 2);
    (5 + (class - 32) % 4) * step
}

/// An arena allocation: its address and size class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Block {
    pub(super) addr: u32,
    pub(super) class: u8,
}

/// Something to reuse after an epoch grace period.
#[derive(Clone, Copy, Debug)]
pub(super) enum Free {
    Run(u32),
    Words(Block),
}

struct Bump {
    segment: usize,
    offset: usize,
}

pub(super) struct Arena {
    segments: [OnceLock<Box<[AtomicU64]>>; ARENA_SEGMENTS],
    bump: Mutex<Bump>,
    free: Box<[SegQueue<u32>]>,
    reserved_words: AtomicUsize,
    /// Words handed out and not yet freed, rounded to their classes.
    used_words: AtomicUsize,
    free_words: AtomicUsize,
    stranded_words: AtomicUsize,
}

#[inline]
fn segment_len(segment: usize) -> usize {
    1 << (SEG0_BITS as usize + segment)
}

#[inline]
fn segment_base(segment: usize) -> usize {
    (1 << SEG0_BITS) * ((1 << segment) - 1)
}

fn zeroed_words(len: usize) -> Box<[AtomicU64]> {
    let words = Box::<[AtomicU64]>::new_zeroed_slice(len);
    // SAFETY: an all-zero bit pattern is a valid `AtomicU64`.
    unsafe { words.assume_init() }
}

impl Arena {
    fn new() -> Self {
        Self {
            segments: std::array::from_fn(|_| OnceLock::new()),
            // Word 0 is never handed out, so address 0 can mean none.
            bump: Mutex::new(Bump {
                segment: 0,
                offset: 1,
            }),
            free: (0..CLASSES).map(|_| SegQueue::new()).collect(),
            reserved_words: AtomicUsize::new(0),
            used_words: AtomicUsize::new(0),
            free_words: AtomicUsize::new(0),
            stranded_words: AtomicUsize::new(1),
        }
    }

    #[inline]
    fn locate(addr: u32) -> (usize, usize) {
        let scaled = (addr as usize >> SEG0_BITS) + 1;
        let segment = (usize::BITS - 1 - scaled.leading_zeros()) as usize;
        (segment, addr as usize - segment_base(segment))
    }

    /// `len` words at `addr`, which must lie in one allocation.
    #[inline]
    pub(super) fn slice(&self, addr: u32, len: usize) -> &[AtomicU64] {
        let (segment, offset) = Self::locate(addr);
        let words = self.segments[segment]
            .get()
            .expect("arena addresses name initialized segments");
        &words[offset..offset + len]
    }

    #[inline]
    pub(super) fn word(&self, addr: u32) -> &AtomicU64 {
        let (segment, offset) = Self::locate(addr);
        &self.segments[segment]
            .get()
            .expect("arena addresses name initialized segments")[offset]
    }

    fn segment(&self, segment: usize) -> &[AtomicU64] {
        self.segments[segment].get_or_init(|| {
            self.reserved_words
                .fetch_add(segment_len(segment), Ordering::Relaxed);
            zeroed_words(segment_len(segment))
        })
    }

    /// Allocates at least `words` words. With `zero`, every word reads 0.
    pub(super) fn alloc(&self, words: usize, zero: bool) -> Result<Block, KvCacheEventError> {
        let (class, rounded) = class_of(words).ok_or(KvCacheEventError::CapacityExhausted)?;
        if let Some(addr) = self.free[class as usize].pop() {
            self.free_words.fetch_sub(rounded, Ordering::Relaxed);
            self.used_words.fetch_add(rounded, Ordering::Relaxed);
            if zero {
                for word in self.slice(addr, rounded) {
                    word.store(0, Ordering::Relaxed);
                }
            }
            return Ok(Block { addr, class });
        }

        let mut bump = self.bump.lock();
        while bump.offset + rounded > segment_len(bump.segment) {
            if bump.segment + 1 == ARENA_SEGMENTS {
                return Err(KvCacheEventError::CapacityExhausted);
            }
            let tail = segment_len(bump.segment) - bump.offset;
            self.stranded_words.fetch_add(tail, Ordering::Relaxed);
            bump.segment += 1;
            bump.offset = 0;
        }
        let global = segment_base(bump.segment) + bump.offset;
        let Ok(addr) = u32::try_from(global) else {
            return Err(KvCacheEventError::CapacityExhausted);
        };
        // Fresh segment words are zero already.
        self.segment(bump.segment);
        bump.offset += rounded;
        self.used_words.fetch_add(rounded, Ordering::Relaxed);
        Ok(Block { addr, class })
    }

    /// Returns `block` to its free list. Callers make sure nothing can still read it.
    pub(super) fn release(&self, block: Block) {
        let words = class_words(block.class);
        self.used_words.fetch_sub(words, Ordering::Relaxed);
        self.free_words.fetch_add(words, Ordering::Relaxed);
        self.free[block.class as usize].push(block.addr);
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn reserved_bytes(&self) -> usize {
        8 * self.reserved_words.load(Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn used_bytes(&self) -> usize {
        8 * self.used_words.load(Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn free_bytes(&self) -> usize {
        8 * self.free_words.load(Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn stranded_bytes(&self) -> usize {
        8 * self.stranded_words.load(Ordering::Relaxed)
    }

    /// Every free-listed block, popped and pushed back. Test-only: not atomic against
    /// concurrent frees.
    #[cfg(test)]
    pub(super) fn free_blocks(&self) -> Vec<Block> {
        let mut blocks = Vec::new();
        for (class, list) in self.free.iter().enumerate() {
            let mut addrs = Vec::new();
            while let Some(addr) = list.pop() {
                addrs.push(addr);
            }
            for &addr in &addrs {
                list.push(addr);
                blocks.push(Block {
                    addr,
                    class: class as u8,
                });
            }
        }
        blocks
    }
}

pub(super) struct Slab {
    segments: [OnceLock<Box<[RunHeader]>>; SLAB_SEGMENTS],
    next: AtomicU32,
    free: SegQueue<u32>,
    free_count: AtomicUsize,
}

impl Slab {
    fn new() -> Self {
        let slab = Self {
            segments: std::array::from_fn(|_| OnceLock::new()),
            // 0 is NONE and 1 is ROOT.
            next: AtomicU32::new(2),
            free: SegQueue::new(),
            free_count: AtomicUsize::new(0),
        };
        slab.segment(0);
        slab
    }

    #[inline]
    fn locate(id: u32) -> (usize, usize) {
        let scaled = id as usize / SLAB0 + 1;
        let segment = (usize::BITS - 1 - scaled.leading_zeros()) as usize;
        (segment, id as usize - SLAB0 * ((1 << segment) - 1))
    }

    fn segment(&self, segment: usize) -> &[RunHeader] {
        self.segments[segment].get_or_init(|| {
            (0..SLAB0 << segment)
                .map(|_| RunHeader::default())
                .collect()
        })
    }

    #[inline]
    pub(super) fn get(&self, id: u32) -> &RunHeader {
        let (segment, index) = Self::locate(id);
        &self.segments[segment]
            .get()
            .expect("run ids name initialized slab segments")[index]
    }

    /// A run id from the free list, or a fresh one.
    pub(super) fn alloc(&self) -> Result<u32, KvCacheEventError> {
        if let Some(id) = self.free.pop() {
            self.free_count.fetch_sub(1, Ordering::Relaxed);
            return Ok(id);
        }
        let id = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                (next < u32::MAX).then_some(next + 1)
            })
            .map_err(|_| KvCacheEventError::CapacityExhausted)?;
        let (segment, _) = Self::locate(id);
        if segment >= SLAB_SEGMENTS {
            return Err(KvCacheEventError::CapacityExhausted);
        }
        self.segment(segment);
        Ok(id)
    }

    pub(super) fn release(&self, id: u32) {
        self.free_count.fetch_add(1, Ordering::Relaxed);
        self.free.push(id);
    }

    /// Ids handed out at least once, ROOT included.
    pub(super) fn allocated(&self) -> u32 {
        self.next.load(Ordering::Relaxed) - 1
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn free_count(&self) -> usize {
        self.free_count.load(Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn reserved_bytes(&self) -> usize {
        let headers: usize = self
            .segments
            .iter()
            .filter_map(OnceLock::get)
            .map(|segment| segment.len())
            .sum();
        headers * std::mem::size_of::<RunHeader>()
    }

    #[cfg(test)]
    pub(super) fn free_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        while let Some(id) = self.free.pop() {
            ids.push(id);
        }
        for &id in &ids {
            self.free.push(id);
        }
        ids
    }
}

/// The storage deferred frees hand back to: shared by the index and every closure still
/// waiting in the epoch, so a closure can outlive the index.
pub(super) struct Storage {
    pub(super) arena: Arena,
    pub(super) slab: Slab,
    /// Words and runs deferred but not yet released, for the memory report.
    pending_words: AtomicI64,
    pending_runs: AtomicI64,
}

impl Storage {
    pub(super) fn new() -> Self {
        Self {
            arena: Arena::new(),
            slab: Slab::new(),
            pending_words: AtomicI64::new(0),
            pending_runs: AtomicI64::new(0),
        }
    }

    fn release_all(&self, items: Vec<Free>) {
        let mut words = 0i64;
        let mut runs = 0i64;
        for item in items {
            match item {
                Free::Run(id) => {
                    runs += 1;
                    self.slab.release(id);
                }
                Free::Words(block) => {
                    words += class_words(block.class) as i64;
                    self.arena.release(block);
                }
            }
        }
        self.pending_words.fetch_sub(words, Ordering::Relaxed);
        self.pending_runs.fetch_sub(runs, Ordering::Relaxed);
    }

    /// Bytes of arena words and run headers waiting out an epoch grace period.
    #[cfg(any(test, feature = "bench"))]
    pub(super) fn epoch_pending_bytes(&self) -> usize {
        let words = self.pending_words.load(Ordering::Relaxed).max(0) as usize;
        let runs = self.pending_runs.load(Ordering::Relaxed).max(0) as usize;
        8 * words + runs * std::mem::size_of::<RunHeader>()
    }
}

/// Frees one lane has collected but not yet handed to the epoch.
#[derive(Default)]
pub(super) struct FreeBatch {
    items: Vec<Free>,
}

impl FreeBatch {
    pub(super) fn push(&mut self, storage: &Arc<Storage>, item: Free) {
        self.items.push(item);
        if self.items.len() >= FREE_BATCH {
            self.flush(storage);
        }
    }

    /// Defers every collected item in one closure that owns an `Arc` of the storage.
    pub(super) fn flush(&mut self, storage: &Arc<Storage>) {
        if self.items.is_empty() {
            return;
        }
        let items = std::mem::take(&mut self.items);
        let (words, runs) = items.iter().fold((0i64, 0i64), |(w, r), item| match item {
            Free::Run(_) => (w, r + 1),
            Free::Words(block) => (w + class_words(block.class) as i64, r),
        });
        storage.pending_words.fetch_add(words, Ordering::Relaxed);
        storage.pending_runs.fetch_add(runs, Ordering::Relaxed);
        let storage = Arc::clone(storage);
        crossbeam_epoch::pin().defer(move || storage.release_all(items));
    }
}

impl Drop for FreeBatch {
    fn drop(&mut self) {
        debug_assert!(
            self.items.is_empty() || std::thread::panicking(),
            "a FreeBatch was dropped with {} unflushed items",
            self.items.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_round_up_by_at_most_a_quarter() {
        let mut previous = 0;
        for words in 1..100_000usize {
            let (class, rounded) = class_of(words).unwrap();
            assert!(rounded >= words);
            assert_eq!(class_words(class), rounded);
            assert!(class as usize >= previous);
            previous = class as usize;
            if words > 64 {
                assert!(rounded - words <= words / 4, "{words} -> {rounded}");
            }
        }
    }

    #[test]
    fn arena_addresses_map_into_segments_and_never_span_two() {
        let arena = Arena::new();
        let a = arena.alloc(4, true).unwrap();
        assert_eq!(a.addr, 1);
        let big = arena.alloc(segment_len(0) - 2, true).unwrap();
        // The big block did not fit after `a`, so the tail was stranded.
        assert_eq!(big.addr as usize, segment_base(1));
        assert!(arena.stranded_bytes() > 0);
        let words = arena.slice(big.addr, segment_len(0) - 2);
        assert!(words.iter().all(|w| w.load(Ordering::Relaxed) == 0));

        arena.word(a.addr).store(7, Ordering::Relaxed);
        arena.release(a);
        let again = arena.alloc(3, true).unwrap();
        assert_eq!(again, a);
        assert_eq!(arena.word(a.addr).load(Ordering::Relaxed), 0);
    }

    #[test]
    fn slab_ids_cross_segments() {
        for id in [2u32, 1023, 1024, 3071, 3072, 1 << 20, u32::MAX - 1] {
            let (segment, index) = Slab::locate(id);
            assert!(segment < SLAB_SEGMENTS, "{id}");
            assert!(index < SLAB0 << segment, "{id}");
        }
        let slab = Slab::new();
        let ids: Vec<_> = (0..3000).map(|_| slab.alloc().unwrap()).collect();
        assert_eq!(ids[0], 2);
        assert_eq!(*ids.last().unwrap(), 3001);
        slab.get(3001).len.store(5, Ordering::Relaxed);
        assert_eq!(slab.get(3001).len.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn deferred_frees_wait_for_pinned_threads() {
        let storage = Arc::new(Storage::new());
        let block = storage.arena.alloc(8, true).unwrap();
        let pin = crossbeam_epoch::pin();
        let mut batch = FreeBatch::default();
        batch.push(&storage, Free::Words(block));
        batch.flush(&storage);
        let other = {
            let storage = Arc::clone(&storage);
            std::thread::spawn(move || {
                for _ in 0..256 {
                    crossbeam_epoch::pin().flush();
                }
                storage.arena.free_blocks()
            })
        };
        assert!(!other.join().unwrap().contains(&block));
        drop(pin);
        while !storage.arena.free_blocks().contains(&block) {
            crossbeam_epoch::pin().flush();
        }
    }
}
