// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run headers and the arena structures a run owns: its window into a shared hash array,
//! whole-holder bits with overflow chunks, the cutoff (partial-holder) table and the
//! forwarding records a split leaves behind.
//!
//! Every field is an atomic so the storage code is data-race free whatever lock its
//! caller holds; the run's `gate` and `state` locks give the snapshots their meaning
//! (see the module docs).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use parking_lot::RwLock;

use super::arena::{Block, Free, FreeBatch, Storage, class_of};
use super::slots::Slot;
use crate::protocols::{KvCacheEventError, KvCacheStoredBlockData};

pub(super) const NONE: u32 = 0;
pub(super) const ROOT: u32 = 1;

/// Whole-holder words inline in every header, covering slots `0..128`.
pub(super) const INLINE_WORDS: usize = 2;
/// Bit words per overflow chunk, covering 256 slots.
const CHUNK_BIT_WORDS: usize = 4;
/// Header word plus bit words.
const CHUNK_WORDS: usize = 1 + CHUNK_BIT_WORDS;

/// A run that has been split. It never grows again.
pub(super) const SEALED: u32 = 1;
/// A run that has been unlinked. It has no holders and no children.
pub(super) const DEAD: u32 = 2;

/// Live partial holders a run may carry; one more forces a prefix-cap split.
pub(super) const PARTIAL_CAP: usize = 16;
const CUTOFF_MIN_CAP: usize = 4;

/// Maximum blocks per run; longer stores continue in end children.
pub(super) const MAX_RUN_LEN: usize = 1 << 18;

#[derive(Default)]
pub(super) struct RunHeader {
    /// Shape gate: shared for plans, claims and coverage changes; exclusive for anything
    /// that moves positions, bits between runs, or replaces the child table.
    pub(super) gate: RwLock<()>,
    /// Shared for reads of the window, coverage, cutoffs, forwards and child table
    /// pointer; exclusive while any of them change (bit sets excepted).
    pub(super) state: RwLock<()>,
    /// Bumped under the exclusive gate on every shape change; never odd-encoded.
    pub(super) version: AtomicU64,
    /// Bumped by every reincarnation of this id; never 0 once allocated.
    pub(super) generation: AtomicU32,
    pub(super) parent: AtomicU32,
    /// Absolute depth of position 0; a child's offset is `child.start - parent.start`.
    pub(super) start: AtomicU32,
    pub(super) len: AtomicU32,
    pub(super) array: AtomicU32,
    pub(super) base: AtomicU32,
    pub(super) children: AtomicU32,
    pub(super) cutoffs: AtomicU32,
    pub(super) forwards: AtomicU32,
    pub(super) overflow: AtomicU32,
    pub(super) flags: AtomicU32,
    pub(super) whole: [AtomicU64; INLINE_WORDS],
}

impl RunHeader {
    #[inline]
    pub(super) fn flag(&self, flag: u32) -> bool {
        self.flags.load(Ordering::Relaxed) & flag != 0
    }

    #[inline]
    pub(super) fn is_dead(&self) -> bool {
        self.flag(DEAD)
    }

    #[inline]
    pub(super) fn len(&self) -> u32 {
        self.len.load(Ordering::Relaxed)
    }

    #[inline]
    pub(super) fn bump_version(&self) {
        self.version.fetch_add(1, Ordering::Release);
    }
}

/// A run's positions, read under its state lock.
#[derive(Clone, Copy)]
pub(super) struct Window<'a> {
    pub(super) local: &'a [AtomicU64],
    pub(super) ext: &'a [AtomicU64],
}

impl Window<'_> {
    #[inline]
    pub(super) fn local(&self, offset: usize) -> u64 {
        self.local[offset].load(Ordering::Relaxed)
    }

    #[inline]
    pub(super) fn ext(&self, offset: usize) -> u64 {
        self.ext[offset].load(Ordering::Relaxed)
    }

    #[inline]
    pub(super) fn len(&self) -> usize {
        self.local.len()
    }
}

/// A forwarding record: positions at or past `at` moved to `suffix`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Forward {
    pub(super) at: u32,
    pub(super) suffix: u32,
    pub(super) generation: u32,
}

/// Capacity of a new hash array for `len` positions: `len + max(len / 8, 1)`.
fn capacity_for(len: usize) -> usize {
    len + (len / 8).max(1)
}

fn array_words(capacity: usize) -> usize {
    2 + 2 * capacity
}

/// Overflow chunk index and bit word within it of whole word `word`.
#[inline]
fn chunk_position(word: usize) -> (u32, usize) {
    let rest = word - INLINE_WORDS;
    ((rest / CHUNK_BIT_WORDS) as u32, rest % CHUNK_BIT_WORDS)
}

#[inline]
fn slot_word(slot: Slot) -> (usize, u64) {
    (slot.index() / 64, 1 << (slot.index() % 64))
}

/// Where the next overflow chunk is linked from.
enum Link<'a> {
    Head(&'a AtomicU32),
    Chunk(&'a AtomicU64),
}

impl Storage {
    #[inline]
    pub(super) fn run(&self, id: u32) -> &RunHeader {
        self.slab.get(id)
    }

    // ------------------------------------------------------------------
    // Hash arrays
    // ------------------------------------------------------------------

    /// Allocates an array holding `blocks`, with `refs = 1`.
    pub(super) fn new_array(
        &self,
        prefix: Option<Window<'_>>,
        blocks: &[KvCacheStoredBlockData],
    ) -> Result<u32, KvCacheEventError> {
        let keep = prefix.map_or(0, |window| window.len());
        let len = keep + blocks.len();
        let (_, words) =
            class_of(array_words(capacity_for(len))).ok_or(KvCacheEventError::CapacityExhausted)?;
        let block = self.arena.alloc(words, false)?;
        let capacity = (words - 2) / 2;
        let all = self.arena.slice(block.addr, words);
        all[0].store(len as u64 | ((capacity as u64) << 32), Ordering::Relaxed);
        all[1].store(1, Ordering::Relaxed);
        let (local, ext) = all[2..].split_at(capacity);
        let mut i = 0;
        if let Some(window) = prefix {
            for j in 0..keep {
                local[i].store(window.local(j), Ordering::Relaxed);
                ext[i].store(window.ext(j), Ordering::Relaxed);
                i += 1;
            }
        }
        for block in blocks {
            local[i].store(block.tokens_hash.0, Ordering::Relaxed);
            ext[i].store(block.block_hash.0, Ordering::Relaxed);
            i += 1;
        }
        Ok(block.addr)
    }

    fn array_block(&self, array: u32) -> Block {
        let capacity = (self.arena.word(array).load(Ordering::Relaxed) >> 32) as usize;
        let (class, _) = class_of(array_words(capacity)).expect("array class was allocated");
        Block { addr: array, class }
    }

    /// `(used, capacity)` of `array`.
    #[cfg(test)]
    pub(super) fn array_extent(&self, array: u32) -> (u32, u32) {
        let header = self.arena.word(array).load(Ordering::Relaxed);
        (header as u32, (header >> 32) as u32)
    }

    #[cfg(test)]
    pub(super) fn array_refs(&self, array: u32) -> u32 {
        self.arena.word(array + 1).load(Ordering::Relaxed) as u32
    }

    pub(super) fn retain_array(&self, array: u32) {
        self.arena.word(array + 1).fetch_add(1, Ordering::Relaxed);
    }

    /// Drops one reference; the last one frees the array after an epoch.
    pub(super) fn release_array(&self, array: u32, batch: &mut FreeBatch, storage: &Arc<Storage>) {
        if self.arena.word(array + 1).fetch_sub(1, Ordering::AcqRel) == 1 {
            batch.push(storage, Free::Words(self.array_block(array)));
        }
    }

    #[inline]
    pub(super) fn window(&self, run: &RunHeader) -> Window<'_> {
        let len = run.len.load(Ordering::Relaxed) as usize;
        let array = run.array.load(Ordering::Relaxed);
        if len == 0 || array == NONE {
            return Window {
                local: &[],
                ext: &[],
            };
        }
        let capacity = (self.arena.word(array).load(Ordering::Relaxed) >> 32) as usize;
        let base = run.base.load(Ordering::Relaxed) as usize;
        let words = self.arena.slice(array, array_words(capacity));
        Window {
            local: &words[2 + base..2 + base + len],
            ext: &words[2 + capacity + base..2 + capacity + base + len],
        }
    }

    /// Appends `blocks` to `run`, in place when its window ends at the array's `used`
    /// and the array has room, otherwise into a new array. Callers hold the exclusive
    /// gate and the state write lock.
    pub(super) fn append_blocks(
        &self,
        run: &RunHeader,
        blocks: &[KvCacheStoredBlockData],
        batch: &mut FreeBatch,
        storage: &Arc<Storage>,
    ) -> Result<(), KvCacheEventError> {
        let len = run.len();
        let array = run.array.load(Ordering::Relaxed);
        let base = run.base.load(Ordering::Relaxed);
        let header = self.arena.word(array);
        let current = header.load(Ordering::Relaxed);
        let (used, capacity) = (current as u32, (current >> 32) as u32);
        let k = blocks.len() as u32;
        let in_place = base + len == used
            && used + k <= capacity
            && header
                .compare_exchange(
                    current,
                    current + u64::from(k),
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok();
        if in_place {
            let words = self.arena.slice(array, array_words(capacity as usize));
            let (local, ext) = words[2..].split_at(capacity as usize);
            for (i, block) in blocks.iter().enumerate() {
                let at = (used as usize) + i;
                local[at].store(block.tokens_hash.0, Ordering::Relaxed);
                ext[at].store(block.block_hash.0, Ordering::Relaxed);
            }
        } else {
            let fresh = self.new_array(Some(self.window(run)), blocks)?;
            run.array.store(fresh, Ordering::Relaxed);
            run.base.store(0, Ordering::Relaxed);
            self.release_array(array, batch, storage);
        }
        run.len.store(len + k, Ordering::Release);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Whole holders
    // ------------------------------------------------------------------

    fn chunk_index_and_next(&self, chunk: u32) -> (u32, u32) {
        let header = self.arena.word(chunk).load(Ordering::Acquire);
        (header as u32, (header >> 32) as u32)
    }

    /// `(chunk index, chunk address)` for every overflow chunk, in link order.
    pub(super) fn chunks(&self, run: &RunHeader) -> impl Iterator<Item = (u32, u32)> + '_ {
        let first = run.overflow.load(Ordering::Acquire);
        std::iter::successors((first != NONE).then_some(first), move |&chunk| {
            let (_, next) = self.chunk_index_and_next(chunk);
            (next != NONE).then_some(next)
        })
        .map(move |chunk| (self.chunk_index_and_next(chunk).0, chunk))
    }

    #[inline]
    fn whole_word<'a>(&'a self, run: &'a RunHeader, word: usize) -> Option<&'a AtomicU64> {
        if word < INLINE_WORDS {
            return Some(&run.whole[word]);
        }
        let (index, within) = chunk_position(word);
        self.chunks(run)
            .find(|&(chunk_index, _)| chunk_index == index)
            .map(|(_, chunk)| self.arena.word(chunk + 1 + within as u32))
    }

    fn whole_word_or_install<'a>(
        &'a self,
        run: &'a RunHeader,
        word: usize,
    ) -> Result<&'a AtomicU64, KvCacheEventError> {
        if word < INLINE_WORDS {
            return Ok(&run.whole[word]);
        }
        let (index, within) = chunk_position(word);
        let mut link = Link::Head(&run.overflow);
        let mut spare: Option<Block> = None;
        loop {
            let next = match link {
                Link::Head(head) => head.load(Ordering::Acquire),
                Link::Chunk(header) => (header.load(Ordering::Acquire) >> 32) as u32,
            };
            if next != NONE {
                let (chunk_index, _) = self.chunk_index_and_next(next);
                if chunk_index == index {
                    if let Some(spare) = spare {
                        self.arena.release(spare);
                    }
                    return Ok(self.arena.word(next + 1 + within as u32));
                }
                link = Link::Chunk(self.arena.word(next));
                continue;
            }
            let block = match spare.take() {
                Some(block) => block,
                None => {
                    let block = self.arena.alloc(CHUNK_WORDS, true)?;
                    self.arena
                        .word(block.addr)
                        .store(u64::from(index), Ordering::Relaxed);
                    block
                }
            };
            let installed = match link {
                Link::Head(head) => head
                    .compare_exchange(NONE, block.addr, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok(),
                Link::Chunk(header) => {
                    let current = header.load(Ordering::Acquire);
                    (current >> 32) as u32 == NONE
                        && header
                            .compare_exchange(
                                current,
                                current | (u64::from(block.addr) << 32),
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            )
                            .is_ok()
                }
            };
            if installed {
                return Ok(self.arena.word(block.addr + 1 + within as u32));
            }
            // A racing installer linked another chunk first; keep walking with ours.
            spare = Some(block);
        }
    }

    #[inline]
    pub(super) fn whole_contains(&self, run: &RunHeader, slot: Slot) -> bool {
        let (word, bit) = slot_word(slot);
        self.whole_word(run, word)
            .is_some_and(|w| w.load(Ordering::Relaxed) & bit != 0)
    }

    /// Sets `slot`'s whole bit. Returns whether it was clear.
    pub(super) fn whole_set(&self, run: &RunHeader, slot: Slot) -> Result<bool, KvCacheEventError> {
        let (word, bit) = slot_word(slot);
        Ok(self
            .whole_word_or_install(run, word)?
            .fetch_or(bit, Ordering::Relaxed)
            & bit
            == 0)
    }

    /// Clears `slot`'s whole bit. Returns whether it was set.
    pub(super) fn whole_clear(&self, run: &RunHeader, slot: Slot) -> bool {
        let (word, bit) = slot_word(slot);
        self.whole_word(run, word)
            .is_some_and(|w| w.fetch_and(!bit, Ordering::Relaxed) & bit != 0)
    }

    /// Calls `f(word index, bits)` for every whole word, inline words first.
    #[inline]
    pub(super) fn for_each_whole_word(&self, run: &RunHeader, mut f: impl FnMut(usize, u64)) {
        for (index, word) in run.whole.iter().enumerate() {
            f(index, word.load(Ordering::Relaxed));
        }
        if run.overflow.load(Ordering::Relaxed) == NONE {
            return;
        }
        for (chunk_index, chunk) in self.chunks(run) {
            let words = self.arena.slice(chunk + 1, CHUNK_BIT_WORDS);
            for (within, word) in words.iter().enumerate() {
                f(
                    INLINE_WORDS + chunk_index as usize * CHUNK_BIT_WORDS + within,
                    word.load(Ordering::Relaxed),
                );
            }
        }
    }

    pub(super) fn whole_any(&self, run: &RunHeader) -> bool {
        let mut any = false;
        self.for_each_whole_word(run, |_, bits| any |= bits != 0);
        any
    }

    /// Whether `slot` is the only whole holder.
    pub(super) fn whole_sole(&self, run: &RunHeader, slot: Slot) -> bool {
        let (target, bit) = slot_word(slot);
        let mut sole = true;
        let mut seen = false;
        self.for_each_whole_word(run, |index, bits| {
            if index == target {
                seen |= bits & bit != 0;
                sole &= bits & !bit == 0;
            } else {
                sole &= bits == 0;
            }
        });
        sole && seen
    }

    pub(super) fn for_each_whole(&self, run: &RunHeader, mut f: impl FnMut(Slot)) {
        self.for_each_whole_word(run, |index, mut bits| {
            while bits != 0 {
                f(Slot::from_index(
                    index * 64 + bits.trailing_zeros() as usize,
                ));
                bits &= bits - 1;
            }
        });
    }

    /// Copies every whole bit of `from` onto `to`, which nothing else can reach yet.
    fn copy_whole(&self, from: &RunHeader, to: &RunHeader) -> Result<(), KvCacheEventError> {
        let mut failed = None;
        self.for_each_whole_word(from, |index, bits| {
            if bits == 0 || failed.is_some() {
                return;
            }
            match self.whole_word_or_install(to, index) {
                Ok(word) => {
                    word.fetch_or(bits, Ordering::Relaxed);
                }
                Err(error) => failed = Some(error),
            }
        });
        failed.map_or(Ok(()), Err)
    }

    fn free_chunks(&self, run: &RunHeader, mut free: impl FnMut(Block)) {
        let chunks: Vec<u32> = self.chunks(run).map(|(_, chunk)| chunk).collect();
        for chunk in chunks {
            let (class, _) = class_of(CHUNK_WORDS).expect("chunk class");
            free(Block { addr: chunk, class });
        }
    }

    // ------------------------------------------------------------------
    // Cutoffs: a dense table of `slot | cutoff << 32` words after a
    // `count | capacity << 32` header, changed only under the state write lock.
    // ------------------------------------------------------------------

    fn cutoff_header(&self, table: u32) -> (usize, usize) {
        let header = self.arena.word(table).load(Ordering::Relaxed);
        ((header as u32) as usize, (header >> 32) as usize)
    }

    pub(super) fn cutoff_count(&self, run: &RunHeader) -> usize {
        let table = run.cutoffs.load(Ordering::Relaxed);
        if table == NONE {
            return 0;
        }
        self.cutoff_header(table).0
    }

    /// `(slot, cutoff)` for every partial holder.
    #[inline]
    pub(super) fn cutoff_entries(&self, run: &RunHeader) -> impl Iterator<Item = (Slot, u32)> + '_ {
        let table = run.cutoffs.load(Ordering::Relaxed);
        let entries = if table == NONE {
            &[][..]
        } else {
            let (count, _) = self.cutoff_header(table);
            self.arena.slice(table + 1, count)
        };
        entries.iter().map(|entry| {
            let entry = entry.load(Ordering::Relaxed);
            (
                Slot::from_index(entry as u32 as usize),
                (entry >> 32) as u32,
            )
        })
    }

    pub(super) fn cutoff_of(&self, run: &RunHeader, slot: Slot) -> Option<u32> {
        self.cutoff_entries(run)
            .find(|&(entry, _)| entry == slot)
            .map(|(_, cutoff)| cutoff)
    }

    /// Sets `slot`'s cutoff to `cutoff`. Returns `Ok(false)` when that would exceed
    /// [`PARTIAL_CAP`] live entries, which calls for a prefix-cap split.
    pub(super) fn cutoff_set(
        &self,
        run: &RunHeader,
        slot: Slot,
        cutoff: u32,
        batch: &mut FreeBatch,
        storage: &Arc<Storage>,
    ) -> Result<bool, KvCacheEventError> {
        debug_assert!(cutoff > 0 && cutoff < run.len());
        let packed = slot.index() as u64 | (u64::from(cutoff) << 32);
        let table = run.cutoffs.load(Ordering::Relaxed);
        let (count, capacity) = if table == NONE {
            (0, 0)
        } else {
            self.cutoff_header(table)
        };
        if table != NONE {
            for entry in self.arena.slice(table + 1, count) {
                if entry.load(Ordering::Relaxed) as u32 as usize == slot.index() {
                    entry.store(packed, Ordering::Relaxed);
                    return Ok(true);
                }
            }
        }
        if count == PARTIAL_CAP {
            return Ok(false);
        }
        if count < capacity {
            self.arena
                .word(table + 1 + count as u32)
                .store(packed, Ordering::Relaxed);
            self.arena.word(table).store(
                (count + 1) as u64 | ((capacity as u64) << 32),
                Ordering::Relaxed,
            );
            return Ok(true);
        }
        let grown = (capacity * 2).clamp(CUTOFF_MIN_CAP.min(PARTIAL_CAP), PARTIAL_CAP);
        let block = self.arena.alloc(1 + grown, false)?;
        let words = self.arena.slice(block.addr, 1 + grown);
        if table != NONE {
            for (i, entry) in self.arena.slice(table + 1, count).iter().enumerate() {
                words[1 + i].store(entry.load(Ordering::Relaxed), Ordering::Relaxed);
            }
            batch.push(storage, Free::Words(self.cutoff_block(table)));
        }
        words[1 + count].store(packed, Ordering::Relaxed);
        words[0].store(
            (count + 1) as u64 | ((grown as u64) << 32),
            Ordering::Relaxed,
        );
        run.cutoffs.store(block.addr, Ordering::Release);
        Ok(true)
    }

    fn cutoff_block(&self, table: u32) -> Block {
        let (_, capacity) = self.cutoff_header(table);
        let (class, _) = class_of(1 + capacity).expect("cutoff class");
        Block { addr: table, class }
    }

    /// Drops `slot`'s entry. Frees the table when it empties.
    pub(super) fn cutoff_remove(
        &self,
        run: &RunHeader,
        slot: Slot,
        batch: &mut FreeBatch,
        storage: &Arc<Storage>,
    ) -> bool {
        let table = run.cutoffs.load(Ordering::Relaxed);
        if table == NONE {
            return false;
        }
        let (count, capacity) = self.cutoff_header(table);
        let entries = self.arena.slice(table + 1, count);
        let Some(index) = entries
            .iter()
            .position(|entry| entry.load(Ordering::Relaxed) as u32 as usize == slot.index())
        else {
            return false;
        };
        if count == 1 {
            run.cutoffs.store(NONE, Ordering::Release);
            batch.push(storage, Free::Words(self.cutoff_block(table)));
            return true;
        }
        let last = entries[count - 1].load(Ordering::Relaxed);
        entries[index].store(last, Ordering::Relaxed);
        self.arena.word(table).store(
            (count - 1) as u64 | ((capacity as u64) << 32),
            Ordering::Relaxed,
        );
        true
    }

    /// Replaces every cutoff with `entries`. Callers hold the state write lock.
    fn cutoff_replace(
        &self,
        run: &RunHeader,
        entries: &[(Slot, u32)],
        batch: &mut FreeBatch,
        storage: &Arc<Storage>,
    ) -> Result<(), KvCacheEventError> {
        let fresh = self.cutoff_table(entries)?;
        let old = run.cutoffs.swap(fresh, Ordering::AcqRel);
        if old != NONE {
            batch.push(storage, Free::Words(self.cutoff_block(old)));
        }
        Ok(())
    }

    fn cutoff_table(&self, entries: &[(Slot, u32)]) -> Result<u32, KvCacheEventError> {
        debug_assert!(entries.len() <= PARTIAL_CAP);
        if entries.is_empty() {
            return Ok(NONE);
        }
        let capacity = entries.len().next_power_of_two().max(CUTOFF_MIN_CAP);
        let block = self.arena.alloc(1 + capacity, false)?;
        let words = self.arena.slice(block.addr, 1 + capacity);
        words[0].store(
            entries.len() as u64 | ((capacity as u64) << 32),
            Ordering::Relaxed,
        );
        for (i, &(slot, cutoff)) in entries.iter().enumerate() {
            words[1 + i].store(
                slot.index() as u64 | (u64::from(cutoff) << 32),
                Ordering::Relaxed,
            );
        }
        Ok(block.addr)
    }

    /// `slot`'s holding: `len` when whole, its cutoff, or 0.
    #[inline]
    pub(super) fn held(&self, run: &RunHeader, slot: Slot) -> u32 {
        if self.whole_contains(run, slot) {
            run.len()
        } else {
            self.cutoff_of(run, slot).unwrap_or(0)
        }
    }

    pub(super) fn has_holders(&self, run: &RunHeader) -> bool {
        self.cutoff_count(run) > 0 || self.whole_any(run)
    }

    // ------------------------------------------------------------------
    // Forwarding records: `count | capacity << 32`, then two words per record,
    // `at | suffix << 32` and the suffix generation, in strictly decreasing `at`.
    // ------------------------------------------------------------------

    pub(super) fn forwards(&self, run: &RunHeader) -> Vec<Forward> {
        let table = run.forwards.load(Ordering::Relaxed);
        if table == NONE {
            return Vec::new();
        }
        let header = self.arena.word(table).load(Ordering::Relaxed);
        let count = header as u32 as usize;
        let words = self.arena.slice(table + 1, 2 * count);
        words
            .chunks_exact(2)
            .map(|record| {
                let first = record[0].load(Ordering::Relaxed);
                Forward {
                    at: first as u32,
                    suffix: (first >> 32) as u32,
                    generation: record[1].load(Ordering::Relaxed) as u32,
                }
            })
            .collect()
    }

    /// The record with the largest `at` at or below `offset`, which names where
    /// `offset` went.
    pub(super) fn forward_for(&self, run: &RunHeader, offset: u32) -> Option<Forward> {
        let table = run.forwards.load(Ordering::Relaxed);
        if table == NONE {
            return None;
        }
        let header = self.arena.word(table).load(Ordering::Relaxed);
        let count = header as u32 as usize;
        let words = self.arena.slice(table + 1, 2 * count);
        // Records are in decreasing `at`, so the first one at or below `offset` wins.
        words.chunks_exact(2).find_map(|record| {
            let first = record[0].load(Ordering::Relaxed);
            ((first as u32) <= offset).then(|| Forward {
                at: first as u32,
                suffix: (first >> 32) as u32,
                generation: record[1].load(Ordering::Relaxed) as u32,
            })
        })
    }

    /// Allocates the forward table `run` will have once `record` is appended. The caller
    /// publishes it with [`Self::forward_commit`] once nothing else can fail.
    fn forward_prepare(
        &self,
        run: &RunHeader,
        record: Forward,
    ) -> Result<Block, KvCacheEventError> {
        let mut records = self.forwards(run);
        debug_assert!(records.last().is_none_or(|last| last.at > record.at));
        records.push(record);
        let capacity = records.len().next_power_of_two();
        let words = 1 + 2 * capacity;
        let block = self.arena.alloc(words, false)?;
        let table = self.arena.slice(block.addr, words);
        table[0].store(
            records.len() as u64 | ((capacity as u64) << 32),
            Ordering::Relaxed,
        );
        for (i, record) in records.iter().enumerate() {
            table[1 + 2 * i].store(
                u64::from(record.at) | (u64::from(record.suffix) << 32),
                Ordering::Relaxed,
            );
            table[2 + 2 * i].store(u64::from(record.generation), Ordering::Relaxed);
        }
        Ok(block)
    }

    fn forward_block(&self, table: u32) -> Block {
        let capacity = (self.arena.word(table).load(Ordering::Relaxed) >> 32) as usize;
        let (class, _) = class_of(1 + 2 * capacity).expect("forward class");
        Block { addr: table, class }
    }

    fn forward_commit(
        &self,
        run: &RunHeader,
        block: Block,
        batch: &mut FreeBatch,
        storage: &Arc<Storage>,
    ) {
        let old = run.forwards.swap(block.addr, Ordering::AcqRel);
        if old != NONE {
            batch.push(storage, Free::Words(self.forward_block(old)));
        }
    }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    /// Gives `id` a new incarnation. Nothing else may reach it until a child table
    /// publishes it with `Release`.
    fn reincarnate(&self, id: u32, parent: u32, start: u32) -> u32 {
        let run = self.run(id);
        let mut generation = run.generation.load(Ordering::Relaxed).wrapping_add(1);
        if generation == 0 {
            generation = 1;
        }
        run.generation.store(generation, Ordering::Relaxed);
        run.parent.store(parent, Ordering::Relaxed);
        run.start.store(start, Ordering::Relaxed);
        run.len.store(0, Ordering::Relaxed);
        run.array.store(NONE, Ordering::Relaxed);
        run.base.store(0, Ordering::Relaxed);
        run.children.store(NONE, Ordering::Relaxed);
        run.cutoffs.store(NONE, Ordering::Relaxed);
        run.forwards.store(NONE, Ordering::Relaxed);
        run.overflow.store(NONE, Ordering::Relaxed);
        run.flags.store(0, Ordering::Relaxed);
        for word in &run.whole {
            word.store(0, Ordering::Relaxed);
        }
        run.bump_version();
        generation
    }

    /// Allocates a run holding `blocks` with `holder` as its whole holder. Returns its
    /// id and generation.
    pub(super) fn new_run(
        &self,
        parent: u32,
        start: u32,
        blocks: &[KvCacheStoredBlockData],
        holder: Slot,
    ) -> Result<(u32, u32), KvCacheEventError> {
        let id = self.slab.alloc()?;
        let array = match self.new_array(None, blocks) {
            Ok(array) => array,
            Err(error) => {
                self.slab.release(id);
                return Err(error);
            }
        };
        let generation = self.reincarnate(id, parent, start);
        let run = self.run(id);
        run.array.store(array, Ordering::Relaxed);
        run.len.store(blocks.len() as u32, Ordering::Relaxed);
        if let Err(error) = self.whole_set(run, holder) {
            self.discard_run(id);
            return Err(error);
        }
        Ok((id, generation))
    }

    /// Frees a run nothing has seen: its storage goes straight back to the free lists.
    pub(super) fn discard_run(&self, id: u32) {
        let run = self.run(id);
        run.flags.store(DEAD, Ordering::Relaxed);
        let array = run.array.swap(NONE, Ordering::Relaxed);
        if array != NONE && self.arena.word(array + 1).fetch_sub(1, Ordering::AcqRel) == 1 {
            self.arena.release(self.array_block(array));
        }
        for table in [&run.children, &run.cutoffs, &run.forwards] {
            debug_assert_eq!(table.load(Ordering::Relaxed), NONE);
        }
        self.free_chunks(run, |block| self.arena.release(block));
        run.overflow.store(NONE, Ordering::Relaxed);
        run.len.store(0, Ordering::Relaxed);
        self.slab.release(id);
    }

    /// Kills `id`: marks it dead and hands its id and storage to the epoch. Callers hold
    /// its exclusive gate and state write lock, and it has no holders and no children.
    pub(super) fn kill_run(&self, id: u32, batch: &mut FreeBatch, storage: &Arc<Storage>) {
        let run = self.run(id);
        debug_assert!(!self.has_holders(run));
        run.flags.fetch_or(DEAD, Ordering::Release);
        run.bump_version();
        let array = run.array.swap(NONE, Ordering::Relaxed);
        if array != NONE {
            self.release_array(array, batch, storage);
        }
        let children = run.children.swap(NONE, Ordering::Relaxed);
        if children != NONE {
            batch.push(storage, Free::Words(self.table_block(children)));
        }
        let cutoffs = run.cutoffs.swap(NONE, Ordering::Relaxed);
        if cutoffs != NONE {
            batch.push(storage, Free::Words(self.cutoff_block(cutoffs)));
        }
        let forwards = run.forwards.swap(NONE, Ordering::Relaxed);
        if forwards != NONE {
            batch.push(storage, Free::Words(self.forward_block(forwards)));
        }
        let mut chunks = Vec::new();
        self.free_chunks(run, |block| chunks.push(block));
        for block in chunks {
            batch.push(storage, Free::Words(block));
        }
        run.overflow.store(NONE, Ordering::Relaxed);
        run.len.store(0, Ordering::Relaxed);
        batch.push(storage, Free::Run(id));
    }

    pub(super) fn table_block(&self, table: u32) -> Block {
        let slots = super::table::Table::slots_in(self.arena.word(table).load(Ordering::Relaxed));
        let (class, _) =
            class_of(super::table::words_for(slots)).expect("child table class was allocated");
        Block { addr: table, class }
    }

    /// Splits `id` at `at`, `0 < at < len`, under its exclusive gate and state write
    /// lock. The suffix takes positions `[at, len)` (sharing the array), the whole bits,
    /// the cutoffs past `at` and the children past `at`; the prefix is sealed, promotes
    /// every partial holder that reaches `at`, and records where the suffix went.
    /// Every allocation happens before the prefix changes, so a capacity failure leaves
    /// the run as it was.
    pub(super) fn split_run(
        &self,
        id: u32,
        at: u32,
        batch: &mut FreeBatch,
        storage: &Arc<Storage>,
    ) -> Result<u32, KvCacheEventError> {
        use super::table::{Locked, Table, child_key, pack, rekey, slots_for_live, words_for};

        let run = self.run(id);
        let len = run.len();
        debug_assert!(at > 0 && at < len);
        let window = self.window(run);
        let start = run.start.load(Ordering::Relaxed);

        // Partition the cutoffs and children.
        let cutoffs: Vec<(Slot, u32)> = self.cutoff_entries(run).collect();
        let prefix_cutoffs: Vec<(Slot, u32)> =
            cutoffs.iter().copied().filter(|&(_, c)| c < at).collect();
        let promoted: Vec<Slot> = cutoffs
            .iter()
            .filter(|&&(_, c)| c >= at)
            .map(|&(slot, _)| slot)
            .collect();
        let suffix_cutoffs: Vec<(Slot, u32)> = cutoffs
            .iter()
            .filter(|&&(_, c)| c > at)
            .map(|&(slot, c)| (slot, c - at))
            .collect();
        let old_children = run.children.load(Ordering::Relaxed);
        let mut stay = Vec::new();
        let mut moved = Vec::new();
        if old_children != NONE {
            for (key, value) in self.table(old_children).entries() {
                let child = super::table::unpack_run(value);
                let offset = self.run(child).start.load(Ordering::Relaxed) - start;
                if offset > at {
                    moved.push((rekey(key, offset, offset - at), value, child));
                } else {
                    stay.push((key, value));
                }
            }
        }

        // The suffix, unpublished.
        let suffix = self.slab.alloc()?;
        let generation = self.reincarnate(suffix, id, start + at);
        let s = self.run(suffix);
        let array = run.array.load(Ordering::Relaxed);
        s.array.store(array, Ordering::Relaxed);
        s.base
            .store(run.base.load(Ordering::Relaxed) + at, Ordering::Relaxed);
        s.len.store(len - at, Ordering::Relaxed);
        self.retain_array(array);
        let mut prepared: Vec<Block> = Vec::new();
        let result = (|| {
            self.copy_whole(run, s)?;
            let cutoff_table = self.cutoff_table(&suffix_cutoffs)?;
            s.cutoffs.store(cutoff_table, Ordering::Relaxed);
            if !moved.is_empty() {
                let slots = slots_for_live(moved.len() as u32);
                let block = self.arena.alloc(words_for(slots), true)?;
                let words = self.arena.slice(block.addr, words_for(slots));
                Table::init(words, slots);
                let table = Table::new(words);
                for &(key, value, _) in &moved {
                    let inserted = table.insert_locked(key, value);
                    debug_assert_eq!(inserted, Locked::Inserted);
                }
                s.children.store(block.addr, Ordering::Relaxed);
            }
            // The prefix's new tables.
            let slots = slots_for_live(stay.len() as u32 + 1);
            let block = self.arena.alloc(words_for(slots), true)?;
            prepared.push(block);
            let words = self.arena.slice(block.addr, words_for(slots));
            Table::init(words, slots);
            let table = Table::new(words);
            for &(key, value) in &stay {
                table.insert_locked(key, value);
            }
            let inserted = table.insert_locked(
                child_key(at, window.local(at as usize)),
                pack(suffix, generation),
            );
            debug_assert_eq!(inserted, Locked::Inserted);
            let prefix_cutoff_table = self.cutoff_table(&prefix_cutoffs)?;
            if prefix_cutoff_table != NONE {
                prepared.push(self.cutoff_block(prefix_cutoff_table));
            }
            for &slot in &promoted {
                self.whole_word_or_install(run, slot_word(slot).0)?;
            }
            let forward = self.forward_prepare(
                run,
                Forward {
                    at,
                    suffix,
                    generation,
                },
            )?;
            prepared.push(forward);
            Ok::<_, KvCacheEventError>((block.addr, prefix_cutoff_table, forward))
        })();
        let (prefix_children, prefix_cutoffs_table, forward) = match result {
            Ok(tables) => tables,
            Err(error) => {
                for block in prepared {
                    self.arena.release(block);
                }
                let s_children = s.children.swap(NONE, Ordering::Relaxed);
                if s_children != NONE {
                    self.arena.release(self.table_block(s_children));
                }
                let s_cutoffs = s.cutoffs.swap(NONE, Ordering::Relaxed);
                if s_cutoffs != NONE {
                    self.arena.release(self.cutoff_block(s_cutoffs));
                }
                self.discard_run(suffix);
                return Err(error);
            }
        };

        // Commit: nothing below allocates.
        for &(_, _, child) in &moved {
            self.run(child).parent.store(suffix, Ordering::Release);
        }
        for &slot in &promoted {
            self.whole_set(run, slot)
                .expect("promoted slots' words were installed above");
        }
        let old_cutoffs = run.cutoffs.swap(prefix_cutoffs_table, Ordering::AcqRel);
        if old_cutoffs != NONE {
            batch.push(storage, Free::Words(self.cutoff_block(old_cutoffs)));
        }
        run.children.store(prefix_children, Ordering::Release);
        if old_children != NONE {
            batch.push(storage, Free::Words(self.table_block(old_children)));
        }
        self.forward_commit(run, forward, batch, storage);
        run.len.store(at, Ordering::Release);
        run.flags.fetch_or(SEALED, Ordering::Release);
        Ok(suffix)
    }

    #[inline]
    pub(super) fn table(&self, addr: u32) -> super::table::Table<'_> {
        let slots = super::table::Table::slots_in(self.arena.word(addr).load(Ordering::Relaxed));
        super::table::Table::new(self.arena.slice(addr, super::table::words_for(slots)))
    }

    /// The published child of `run` under `key`, read under its state lock (or, for
    /// ROOT, under the caller's pin).
    #[inline]
    pub(super) fn find_child(&self, run: &RunHeader, key: u64) -> Option<u32> {
        let table = run.children.load(Ordering::Acquire);
        if table == NONE {
            return None;
        }
        self.table(table).find(key).map(super::table::unpack_run)
    }

    /// Live children of `run`, as `(key, child id, generation)`.
    pub(super) fn children_of(&self, run: &RunHeader) -> Vec<(u64, u32, u32)> {
        let table = run.children.load(Ordering::Acquire);
        if table == NONE {
            return Vec::new();
        }
        self.table(table)
            .entries()
            .map(|(key, value)| {
                (
                    key,
                    super::table::unpack_run(value),
                    super::table::unpack_generation(value),
                )
            })
            .collect()
    }

    pub(super) fn has_live_children(&self, run: &RunHeader) -> bool {
        let table = run.children.load(Ordering::Acquire);
        table != NONE && self.table(table).live() > 0
    }

    /// Replaces `cutoffs` wholesale; used by the slot sweep.
    pub(super) fn cutoff_retain(
        &self,
        run: &RunHeader,
        mut keep: impl FnMut(Slot) -> bool,
        batch: &mut FreeBatch,
        storage: &Arc<Storage>,
    ) -> Result<bool, KvCacheEventError> {
        let entries: Vec<(Slot, u32)> = self.cutoff_entries(run).collect();
        let kept: Vec<(Slot, u32)> = entries
            .iter()
            .copied()
            .filter(|&(slot, _)| keep(slot))
            .collect();
        if kept.len() == entries.len() {
            return Ok(false);
        }
        self.cutoff_replace(run, &kept, batch, storage)?;
        Ok(true)
    }
}
