// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs (spec B 4): headers in a type-stable slab, hash arrays, coverage, cutoff tables,
//! child tables and forwarding records in the word arena, and the optimistic read attempt
//! every reader goes through (5.2, R1 to R8).
//!
//! Every header field is an atomic, so any read is data-race free. Writers change a run
//! only under its lock; lock-free child claims touch only the child table words and
//! `inflight`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crossbeam_queue::SegQueue;
use parking_lot::{Mutex, MutexGuard};

use super::arena::{self, Addr, Arena, Kind, NONE};
use super::protocol::{
    self, EMPTY, FLAG_DEAD, FLAG_POISONED, FLAG_SEALED, TOMB, VersionStep, read_begin,
    read_validate, reuse_fence,
};
use super::slots::Slot;
use crate::protocols::{ExternalSequenceBlockHash, KvCacheEventError, LocalBlockHash};

pub(crate) type RunId = u32;
pub(crate) const NO_RUN: RunId = 0;
pub(crate) const ROOT: RunId = 1;
/// The root's generation, fixed for the index's life.
pub(crate) const ROOT_GEN: u32 = 1;

/// Longest run; longer stores continue in end children.
pub(crate) const MAX_RUN_LEN: u32 = 1 << 18;
/// Whole-holder words stored inline in every header, covering slots `0..128`.
pub(crate) const INLINE_WORDS: usize = 2;
/// Words per coverage overflow chunk, covering 256 slots.
pub(crate) const CHUNK_WORDS: usize = 4;
/// Overflow chunks a run can have: `u16` slots, 256 per chunk.
const MAX_CHUNKS: usize = super::slots::MAX_SLOTS / 256;
/// Live partial-holder entries a run may carry before it splits at the median cutoff.
pub(crate) const PARTIAL_CAP: u32 = 16;
/// Forwarding hops a resolution may follow before it calls the entry stale.
pub(crate) const MAX_FORWARD_HOPS: usize = 64;
/// Failed optimistic attempts before a reader takes the run lock (R7).
pub(crate) const READ_RETRIES: u32 = 16;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
pub(crate) fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Key of the child that continues its parent's positions `[0, offset)` with `head`.
/// Never `EMPTY` or `TOMB`, and not the identity on `head`, so offsets do not collide.
#[inline]
pub(crate) fn child_key(offset: u32, head: LocalBlockHash) -> u64 {
    match mix64(head.0 ^ mix64(u64::from(offset).wrapping_add(GOLDEN))) {
        EMPTY => 1,
        TOMB => TOMB - 1,
        key => key,
    }
}

#[inline]
fn child_value(id: RunId, generation: u32) -> u64 {
    u64::from(id) | u64::from(generation) << 32
}

#[inline]
fn split_value(value: u64) -> (RunId, u32) {
    (value as u32, (value >> 32) as u32)
}

#[inline]
fn pack_cutoff(slot: Slot, cutoff: u32) -> u64 {
    debug_assert!(cutoff > 0);
    slot.index() as u64 | u64::from(cutoff) << 32
}

#[inline]
fn unpack_cutoff(entry: u64) -> Option<(usize, u32)> {
    (entry != EMPTY && entry != TOMB).then_some(((entry as u32) as usize, (entry >> 32) as u32))
}

/// Capacity a new hash array of `len` blocks gets: an eighth more, for appends.
#[inline]
pub(crate) fn capacity_for(len: u32) -> u32 {
    len + (len / 8).max(1)
}

// ----------------------------------------------------------------------------
// Headers and the slab
// ----------------------------------------------------------------------------

pub(crate) struct RunHeader {
    /// Even when stable, odd while a step is open.
    pub(crate) version: AtomicU64,
    /// Bumped by every reincarnation; its own word (fix 3).
    pub(crate) generation: AtomicU32,
    /// Run id of the parent; the root's parent is the root.
    pub(crate) parent: AtomicU32,
    /// Absolute depth of position 0.
    pub(crate) start: AtomicU32,
    /// Local hash of position 0, fixed for the incarnation, so a writer holding the parent
    /// can compute a child's key without reading the child's array or taking its lock.
    pub(crate) head: AtomicU64,
    pub(crate) len: AtomicU32,
    pub(crate) array: AtomicU32,
    pub(crate) base: AtomicU32,
    pub(crate) children: AtomicU32,
    pub(crate) cutoffs: AtomicU32,
    pub(crate) forwards: AtomicU32,
    pub(crate) overflow: AtomicU32,
    /// Lock-free child claims in progress.
    pub(crate) inflight: AtomicU32,
    pub(crate) flags: AtomicU32,
    pub(crate) whole: [AtomicU64; INLINE_WORDS],
    pub(crate) lock: Mutex<()>,
}

impl RunHeader {
    fn new() -> Self {
        Self {
            version: AtomicU64::new(0),
            generation: AtomicU32::new(0),
            parent: AtomicU32::new(NO_RUN),
            start: AtomicU32::new(0),
            head: AtomicU64::new(0),
            len: AtomicU32::new(0),
            array: AtomicU32::new(NONE),
            base: AtomicU32::new(0),
            children: AtomicU32::new(NONE),
            cutoffs: AtomicU32::new(NONE),
            forwards: AtomicU32::new(NONE),
            overflow: AtomicU32::new(NONE),
            inflight: AtomicU32::new(0),
            flags: AtomicU32::new(0),
            whole: Default::default(),
            lock: Mutex::new(()),
        }
    }

    /// The header fields as of `version` (R3). Only `len` and `overflow` are acquired:
    /// `len` for the stepless append, `overflow` because a chunk is published without a
    /// step.
    #[inline]
    fn snap(&self, version: u64) -> Snap {
        Snap {
            version,
            generation: self.generation.load(Ordering::Relaxed),
            flags: self.flags.load(Ordering::Relaxed),
            len: protocol::load_len(&self.len),
            array: self.array.load(Ordering::Relaxed),
            base: self.base.load(Ordering::Relaxed),
            children: self.children.load(Ordering::Relaxed),
            cutoffs: self.cutoffs.load(Ordering::Relaxed),
            forwards: self.forwards.load(Ordering::Relaxed),
            overflow: self.overflow.load(Ordering::Acquire),
            start: self.start.load(Ordering::Relaxed),
        }
    }

    #[inline]
    pub(crate) fn step(&self) -> VersionStep<'_> {
        VersionStep::open(&self.version, &self.flags)
    }

    #[inline]
    pub(crate) fn step_excluding_claims(&self) -> VersionStep<'_> {
        VersionStep::open_excluding_claims(&self.version, &self.flags, &self.inflight)
    }
}

/// A run's header fields from one attempt.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Snap {
    pub(crate) version: u64,
    pub(crate) generation: u32,
    pub(crate) flags: u32,
    pub(crate) len: u32,
    pub(crate) array: Addr,
    pub(crate) base: u32,
    pub(crate) children: Addr,
    pub(crate) cutoffs: Addr,
    pub(crate) forwards: Addr,
    pub(crate) overflow: Addr,
    pub(crate) start: u32,
}

impl Snap {
    #[inline]
    pub(crate) fn sealed(&self) -> bool {
        self.flags & FLAG_SEALED != 0
    }
}

const SLAB_FIRST_BITS: u32 = 10;
const SLAB_SEGMENTS: usize = 22;

#[inline]
fn slab_locate(id: RunId) -> (usize, usize) {
    let x = u64::from(id) + (1 << SLAB_FIRST_BITS);
    let k = 63 - x.leading_zeros() - SLAB_FIRST_BITS;
    (k as usize, (x - (1u64 << (SLAB_FIRST_BITS + k))) as usize)
}

/// Run headers in geometrically growing segments that are never freed while the index
/// lives (4.2). That type stability makes an optimistic read of a recycled header safe.
pub(crate) struct Slab {
    segments: [OnceLock<Box<[RunHeader]>>; SLAB_SEGMENTS],
    next: AtomicU32,
    free: SegQueue<RunId>,
}

impl Default for Slab {
    fn default() -> Self {
        Self {
            segments: std::array::from_fn(|_| OnceLock::new()),
            next: AtomicU32::new(ROOT + 1),
            free: SegQueue::new(),
        }
    }
}

impl Slab {
    #[inline]
    pub(crate) fn header(&self, id: RunId) -> Option<&RunHeader> {
        if id == NO_RUN {
            return None;
        }
        let (k, index) = slab_locate(id);
        self.segments.get(k)?.get()?.get(index)
    }

    fn ensure(&self, id: RunId) -> Option<&RunHeader> {
        let (k, index) = slab_locate(id);
        let segment = self.segments.get(k)?.get_or_init(|| {
            (0..1usize << (SLAB_FIRST_BITS as usize + k))
                .map(|_| RunHeader::new())
                .collect()
        });
        segment.get(index)
    }

    /// A recycled or fresh run id. Recycled ids are fenced before their first write (A1).
    fn alloc(&self) -> Result<RunId, KvCacheEventError> {
        if let Some(id) = self.free.pop() {
            reuse_fence();
            return Ok(id);
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        if id == u32::MAX || self.ensure(id).is_none() {
            self.next.store(u32::MAX, Ordering::Relaxed);
            return Err(KvCacheEventError::CapacityExhausted);
        }
        Ok(id)
    }

    pub(crate) fn issued(&self) -> u32 {
        self.next.load(Ordering::Relaxed).saturating_sub(ROOT + 1)
    }

    pub(crate) fn free_len(&self) -> usize {
        self.free.len()
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn free_ids(&self) -> Vec<RunId> {
        let mut ids = Vec::with_capacity(self.free.len());
        while let Some(id) = self.free.pop() {
            ids.push(id);
        }
        for &id in &ids {
            self.free.push(id);
        }
        ids
    }

    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn reserved_bytes(&self) -> u64 {
        (0..SLAB_SEGMENTS)
            .filter(|&k| self.segments[k].get().is_some())
            .map(|k| {
                (1u64 << (SLAB_FIRST_BITS as usize + k)) * std::mem::size_of::<RunHeader>() as u64
            })
            .sum()
    }
}

// ----------------------------------------------------------------------------
// Counters
// ----------------------------------------------------------------------------

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        /// Rare-event counters for the shape report. Hot paths never touch them.
        #[derive(Default)]
        pub(crate) struct Stats {
            $(pub(crate) $name: AtomicU64,)*
        }

        /// A snapshot of [`Stats`].
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
        pub struct StatsSnapshot {
            $(pub $name: u64,)*
        }

        impl Stats {
            pub(crate) fn snapshot(&self) -> StatsSnapshot {
                StatsSnapshot {
                    $($name: self.$name.load(Ordering::Relaxed),)*
                }
            }
        }
    };
}

counters!(
    splits_prefix_cap,
    unlinks,
    kills_aborted,
    h1_violations,
    ext_conflicts,
    reader_retries,
    reader_fallbacks,
    poisoned_seen,
    store_restarts,
    store_failures,
    replans,
    claims_exists,
    locked_inserts,
    table_rebuilds,
    appends_in_place,
    appends_realloc,
    steals,
    inline_fast_path,
    claim_check_failures,
    sweep_residue,
    cleanup_unlinks,
);

#[inline]
pub(crate) fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

// ----------------------------------------------------------------------------
// Deferred frees
// ----------------------------------------------------------------------------

/// Memory and ids a writer released under run locks, returned to the free lists only
/// after it drops them (A2).
#[derive(Default)]
pub(crate) struct Frees {
    blocks: Vec<(Kind, usize, Addr)>,
    ids: Vec<RunId>,
}

// ----------------------------------------------------------------------------
// Read results and locked handles
// ----------------------------------------------------------------------------

/// The outcome of a read attempt.
pub(crate) enum Read<T> {
    Ok(T),
    /// The run was recycled, killed, or poisoned: no credit from it.
    Gone,
}

/// A run held under its lock, with its generation checked.
pub(crate) struct Locked<'a> {
    pub(crate) h: &'a RunHeader,
    pub(crate) id: RunId,
    pub(crate) generation: u32,
    _guard: MutexGuard<'a, ()>,
}

impl Locked<'_> {
    /// Current fields; writers are excluded, so this needs no validation.
    #[inline]
    pub(crate) fn snap(&self) -> Snap {
        self.h.snap(self.h.version.load(Ordering::Relaxed))
    }
}

/// A child-table probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Probe {
    Found(RunId, u32),
    Absent,
    /// A garbage table: the attempt is torn.
    Torn,
}

/// The outcome of a lock-free claim.
pub(crate) enum Claim {
    Claimed,
    Exists(RunId, u32),
    /// The parent stepped or another claimer of the key has not published yet.
    Replan,
    /// The parent has no table, or the table is at its load limit.
    Locked,
}

/// The outcome of a locked insert.
pub(crate) enum Inserted {
    Claimed,
    Exists(RunId, u32),
    /// The planned offset no longer lies in the parent.
    Replan,
}

/// A run's hash columns, windowed to `[base, base + len)`.
#[derive(Clone, Copy)]
pub(crate) struct Columns<'a> {
    pub(crate) local: &'a [AtomicU64],
    pub(crate) ext: &'a [AtomicU64],
}

impl Columns<'_> {
    #[inline]
    pub(crate) fn local(&self, i: usize) -> LocalBlockHash {
        LocalBlockHash(self.local[i].load(Ordering::Relaxed))
    }

    #[inline]
    pub(crate) fn ext(&self, i: usize) -> ExternalSequenceBlockHash {
        ExternalSequenceBlockHash(self.ext[i].load(Ordering::Relaxed))
    }

    pub(crate) fn len(&self) -> usize {
        self.local.len()
    }
}

/// A partial holder's entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cutoff {
    pub(crate) slot: usize,
    pub(crate) cutoff: u32,
}

// ----------------------------------------------------------------------------
// The run store
// ----------------------------------------------------------------------------

/// Reader configuration, fixed at construction except for test hooks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReaderMode {
    /// Version-validated optimistic reads with a locked fallback (5.2).
    Optimistic,
    /// Every read under the run lock (the `B-lockread` ablation).
    Locked,
}

pub(crate) struct Runs {
    pub(crate) slab: Slab,
    pub(crate) arena: Arena,
    pub(crate) stats: Stats,
    reader: ReaderMode,
    read_retries: u32,
    #[cfg(test)]
    pub(crate) forced_failures: AtomicU32,
}

impl Runs {
    pub(crate) fn new(reader: ReaderMode, read_retries: u32) -> Self {
        let runs = Self {
            slab: Slab::default(),
            arena: Arena::default(),
            stats: Stats::default(),
            reader,
            read_retries: read_retries.max(1),
            #[cfg(test)]
            forced_failures: AtomicU32::new(0),
        };
        let root = runs
            .slab
            .ensure(ROOT)
            .expect("the first slab segment holds the root");
        {
            let _guard = root.lock.lock();
            let _step = root.step();
            root.generation.store(ROOT_GEN, Ordering::Relaxed);
            root.parent.store(ROOT, Ordering::Relaxed);
        }
        runs
    }

    #[inline]
    pub(crate) fn header(&self, id: RunId) -> Option<&RunHeader> {
        self.slab.header(id)
    }

    // ------------------------------------------------------------------
    // Reads
    // ------------------------------------------------------------------

    /// One read of run `id` at generation `generation` (R1 to R8): `f` loads what the caller
    /// needs into locals and returns `None` on a torn view (a bounds failure). The result
    /// is committed only after validation. After `read_retries` failed attempts the read
    /// runs under the run lock, and a run still odd under its lock is poisoned.
    #[inline]
    pub(crate) fn read<T>(
        &self,
        id: RunId,
        generation: u32,
        mut f: impl FnMut(&RunHeader, &Snap) -> Option<T>,
    ) -> Read<T> {
        let Some(h) = self.slab.header(id) else {
            return Read::Gone;
        };
        if self.reader == ReaderMode::Optimistic {
            let mut attempt = 0;
            loop {
                if let Some(v1) = read_begin(&h.version) {
                    let snap = h.snap(v1);
                    let alive = snap.generation == generation
                        && snap.flags & (FLAG_DEAD | FLAG_POISONED) == 0;
                    let out = if alive { f(h, &snap) } else { None };
                    #[cfg(test)]
                    let forced = self
                        .forced_failures
                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
                        .is_ok();
                    #[cfg(not(test))]
                    let forced = false;
                    if !forced && read_validate(&h.version, v1) {
                        match out {
                            Some(out) => return Read::Ok(out),
                            // Validated: the run really is gone, or its state is
                            // inconsistent, which only a bug can cause; no credit either way.
                            None => return Read::Gone,
                        }
                    }
                }
                attempt += 1;
                if attempt >= self.read_retries {
                    break;
                }
                bump(&self.stats.reader_retries);
                for _ in 0..1u32 << attempt.min(8) {
                    std::hint::spin_loop();
                }
            }
            bump(&self.stats.reader_fallbacks);
        }
        let _guard = h.lock.lock();
        let version = h.version.load(Ordering::Acquire);
        if version & 1 == 1 {
            bump(&self.stats.poisoned_seen);
            return Read::Gone;
        }
        let snap = h.snap(version);
        if snap.generation != generation || snap.flags & (FLAG_DEAD | FLAG_POISONED) != 0 {
            return Read::Gone;
        }
        match f(h, &snap) {
            Some(out) => Read::Ok(out),
            None => Read::Gone,
        }
    }

    // ------------------------------------------------------------------
    // Locks
    // ------------------------------------------------------------------

    /// Locks run `id` if it is still generation `generation` and neither dead nor poisoned.
    pub(crate) fn lock(&self, id: RunId, generation: u32) -> Option<Locked<'_>> {
        let h = self.slab.header(id)?;
        let guard = h.lock.lock();
        let alive = h.generation.load(Ordering::Relaxed) == generation
            && h.flags.load(Ordering::Relaxed) & (FLAG_DEAD | FLAG_POISONED) == 0
            && h.version.load(Ordering::Relaxed) & 1 == 0;
        alive.then_some(Locked {
            h,
            id,
            generation,
            _guard: guard,
        })
    }

    /// Locks run `id` at whatever generation it has, if it is alive.
    pub(crate) fn lock_current(&self, id: RunId) -> Option<Locked<'_>> {
        let h = self.slab.header(id)?;
        let guard = h.lock.lock();
        let generation = h.generation.load(Ordering::Relaxed);
        let alive = h.flags.load(Ordering::Relaxed) & (FLAG_DEAD | FLAG_POISONED) == 0
            && h.version.load(Ordering::Relaxed) & 1 == 0;
        alive.then_some(Locked {
            h,
            id,
            generation,
            _guard: guard,
        })
    }

    // ------------------------------------------------------------------
    // Hash arrays
    // ------------------------------------------------------------------

    /// The window `[base, base + len)` of `snap`'s array, or `None` if the fields are torn.
    #[inline]
    pub(crate) fn columns(&self, snap: &Snap) -> Option<Columns<'_>> {
        if snap.len == 0 {
            return Some(Columns {
                local: &[],
                ext: &[],
            });
        }
        let header = self.arena.word(snap.array)?.load(Ordering::Relaxed);
        let used = header as u32;
        let capacity = (header >> 32) as u32;
        let end = snap.base.checked_add(snap.len)?;
        if end > used || used > capacity {
            return None;
        }
        let words = self.arena.slice(snap.array, 2 + 2 * capacity as usize)?;
        let (base, end, capacity) = (snap.base as usize, end as usize, capacity as usize);
        Some(Columns {
            local: &words[2 + base..2 + end],
            ext: &words[2 + capacity + base..2 + capacity + end],
        })
    }

    /// A new array holding `blocks`, with one reference.
    pub(crate) fn new_array(
        &self,
        blocks: &[(LocalBlockHash, ExternalSequenceBlockHash)],
    ) -> Result<Addr, KvCacheEventError> {
        let len = u32::try_from(blocks.len()).map_err(|_| KvCacheEventError::CapacityExhausted)?;
        let (addr, class) = self.arena.alloc_array(capacity_for(len))?;
        let capacity = arena::array_class_capacity(class);
        let words = self
            .arena
            .slice(addr, arena::class_words(Kind::Array, class) as usize)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        for (i, &(local, ext)) in blocks.iter().enumerate() {
            words[2 + i].store(local.0, Ordering::Relaxed);
            words[2 + capacity as usize + i].store(ext.0, Ordering::Relaxed);
        }
        words[1].store(1, Ordering::Relaxed);
        words[0].store(
            u64::from(len) | u64::from(capacity) << 32,
            Ordering::Relaxed,
        );
        Ok(addr)
    }

    /// Adds a reference to `array` for a split suffix.
    pub(crate) fn retain_array(&self, array: Addr) {
        if let Some(word) = self.arena.word(array + 1) {
            word.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Drops a reference to `array`, freeing it with the last one.
    pub(crate) fn release_array(&self, array: Addr, frees: &mut Frees) {
        if array == NONE {
            return;
        }
        let (Some(header), Some(refs)) = (self.arena.word(array), self.arena.word(array + 1))
        else {
            return;
        };
        if refs.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let capacity = (header.load(Ordering::Relaxed) >> 32) as u32;
        if let Some(class) = arena::array_class_for(capacity) {
            frees.blocks.push((Kind::Array, class, array));
        }
    }

    /// Claims `[used, used + count)` of `array` for an in-place append by the run whose
    /// window ends at `used`. Returns false when the window does not end at `used` or the
    /// blocks do not fit.
    fn claim_room(&self, array: Addr, window_end: u32, count: u32) -> bool {
        let Some(header) = self.arena.word(array) else {
            return false;
        };
        let current = header.load(Ordering::Relaxed);
        let (used, capacity) = (current as u32, (current >> 32) as u32);
        if used != window_end || u64::from(used) + u64::from(count) > u64::from(capacity) {
            return false;
        }
        header
            .compare_exchange(
                current,
                u64::from(used + count) | u64::from(capacity) << 32,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    fn write_positions(
        &self,
        array: Addr,
        at: u32,
        blocks: &[(LocalBlockHash, ExternalSequenceBlockHash)],
    ) -> Option<()> {
        let capacity = (self.arena.word(array)?.load(Ordering::Relaxed) >> 32) as usize;
        let words = self.arena.slice(array, 2 + 2 * capacity)?;
        for (i, &(local, ext)) in blocks.iter().enumerate() {
            let pos = at as usize + i;
            words[2 + pos].store(local.0, Ordering::Relaxed);
            words[2 + capacity + pos].store(ext.0, Ordering::Relaxed);
        }
        Some(())
    }

    /// Appends `blocks` to the locked run, whose `len` is `snap.len`. In place when the
    /// window ends at the array's `used` and the blocks fit, otherwise by copying the
    /// window into a new array. `stepless` says the caller may publish without a step
    /// (W5: sole whole holder, no child table); otherwise the caller already holds one.
    pub(crate) fn append_locked(
        &self,
        locked: &Locked<'_>,
        snap: &Snap,
        blocks: &[(LocalBlockHash, ExternalSequenceBlockHash)],
        frees: &mut Frees,
    ) -> Result<bool, KvCacheEventError> {
        let count = blocks.len() as u32;
        let new_len = snap.len + count;
        if self.claim_room(snap.array, snap.base + snap.len, count) {
            self.write_positions(snap.array, snap.base + snap.len, blocks)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
            protocol::publish_len(&locked.h.len, new_len);
            bump(&self.stats.appends_in_place);
            return Ok(true);
        }
        let columns = self
            .columns(snap)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        let mut all = Vec::with_capacity(new_len as usize);
        all.extend((0..columns.len()).map(|i| (columns.local(i), columns.ext(i))));
        all.extend_from_slice(blocks);
        let array = self.new_array(&all)?;
        // A reallocation is always a step; the caller opened it.
        locked.h.array.store(array, Ordering::Relaxed);
        locked.h.base.store(0, Ordering::Relaxed);
        protocol::publish_len(&locked.h.len, new_len);
        self.release_array(snap.array, frees);
        bump(&self.stats.appends_realloc);
        Ok(false)
    }

    /// Whether an in-place append of `count` blocks would fit without a reallocation.
    pub(crate) fn has_room(&self, snap: &Snap, count: u32) -> bool {
        let Some(header) = self.arena.word(snap.array) else {
            return false;
        };
        let current = header.load(Ordering::Relaxed);
        let (used, capacity) = (current as u32, (current >> 32) as u32);
        used == snap.base + snap.len && u64::from(used) + u64::from(count) <= u64::from(capacity)
    }

    // ------------------------------------------------------------------
    // Coverage
    // ------------------------------------------------------------------

    /// Calls `f(word, bits)` for every whole word below `live_words`, inline words first,
    /// then chunks in slot order. `None` on a torn chunk list.
    #[inline]
    pub(crate) fn for_each_whole_word(
        &self,
        h: &RunHeader,
        snap: &Snap,
        live_words: usize,
        mut f: impl FnMut(usize, u64),
    ) -> Option<()> {
        for (word, bits) in h.whole.iter().enumerate().take(live_words) {
            f(word, bits.load(Ordering::Relaxed));
        }
        if live_words <= INLINE_WORDS {
            return Some(());
        }
        // Bounded even on a torn, cyclic list (R8): there are at most 256 chunks.
        let max_chunks = (live_words - INLINE_WORDS)
            .div_ceil(CHUNK_WORDS)
            .min(MAX_CHUNKS);
        let mut chunk = snap.overflow;
        let mut seen = 0;
        while chunk != NONE {
            seen += 1;
            if seen > max_chunks + 1 {
                return None;
            }
            let words = self.arena.slice(chunk, arena::CHUNK_WORDS_TOTAL as usize)?;
            let link = words[0].load(Ordering::Acquire);
            let first = INLINE_WORDS + (link >> 32) as usize * CHUNK_WORDS;
            if first >= live_words {
                break;
            }
            for (j, bits) in words[1..].iter().enumerate() {
                if first + j < live_words {
                    f(first + j, bits.load(Ordering::Relaxed));
                }
            }
            chunk = link as u32;
        }
        Some(())
    }

    /// The whole word `word` of a run, if it exists.
    pub(crate) fn whole_word<'a>(&'a self, h: &'a RunHeader, word: usize) -> Option<&'a AtomicU64> {
        if word < INLINE_WORDS {
            return Some(&h.whole[word]);
        }
        let index = ((word - INLINE_WORDS) / CHUNK_WORDS) as u64;
        let mut chunk = h.overflow.load(Ordering::Acquire);
        for _ in 0..=MAX_CHUNKS {
            if chunk == NONE {
                return None;
            }
            let words = self.arena.slice(chunk, arena::CHUNK_WORDS_TOTAL as usize)?;
            let link = words[0].load(Ordering::Acquire);
            match (link >> 32).cmp(&index) {
                std::cmp::Ordering::Equal => {
                    return Some(&words[1 + (word - INLINE_WORDS) % CHUNK_WORDS]);
                }
                std::cmp::Ordering::Greater => return None,
                std::cmp::Ordering::Less => chunk = link as u32,
            }
        }
        None
    }

    /// The whole word `word` of a locked run, installing its chunk if needed. A chunk is
    /// zeroed before it is linked, and linked with a release store, so readers that load
    /// the link with acquire see zeroes, never a previous owner's bits.
    pub(crate) fn whole_word_or_install<'a>(
        &'a self,
        locked: &Locked<'a>,
        word: usize,
    ) -> Result<&'a AtomicU64, KvCacheEventError> {
        if let Some(existing) = self.whole_word(locked.h, word) {
            return Ok(existing);
        }
        let index = ((word - INLINE_WORDS) / CHUNK_WORDS) as u64;
        // Find the link to insert after, keeping chunks in slot order.
        let mut prev: Option<&AtomicU64> = None;
        let mut next = locked.h.overflow.load(Ordering::Relaxed);
        while next != NONE {
            let words = self
                .arena
                .slice(next, arena::CHUNK_WORDS_TOTAL as usize)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
            let link = words[0].load(Ordering::Relaxed);
            if link >> 32 > index {
                break;
            }
            prev = Some(&words[0]);
            next = link as u32;
        }
        let chunk = self.arena.alloc(Kind::Chunk, 0)?;
        let words = self
            .arena
            .slice(chunk, arena::CHUNK_WORDS_TOTAL as usize)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        for bits in &words[1..] {
            bits.store(0, Ordering::Relaxed);
        }
        words[0].store(u64::from(next) | index << 32, Ordering::Relaxed);
        match prev {
            None => locked.h.overflow.store(chunk, Ordering::Release),
            Some(link) => {
                let prev_index = link.load(Ordering::Relaxed) >> 32;
                link.store(u64::from(chunk) | prev_index << 32, Ordering::Release);
            }
        }
        Ok(&words[1 + (word - INLINE_WORDS) % CHUNK_WORDS])
    }

    #[inline]
    pub(crate) fn has_whole(&self, h: &RunHeader, slot: Slot) -> bool {
        let (word, bit) = slot.word_and_bit();
        self.whole_word(h, word)
            .is_some_and(|w| w.load(Ordering::Relaxed) & bit != 0)
    }

    /// Every whole slot of a locked run.
    pub(crate) fn whole_slots(&self, locked: &Locked<'_>) -> Vec<usize> {
        let mut slots = Vec::new();
        let snap = locked.snap();
        self.for_each_whole_word(locked.h, &snap, usize::MAX >> 8, |word, mut bits| {
            while bits != 0 {
                slots.push(word * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        });
        slots
    }

    fn release_chunks(&self, first: Addr, frees: &mut Frees) {
        let mut chunk = first;
        for _ in 0..=MAX_CHUNKS {
            if chunk == NONE {
                return;
            }
            let Some(link) = self.arena.word(chunk) else {
                return;
            };
            let next = link.load(Ordering::Relaxed) as u32;
            frees.blocks.push((Kind::Chunk, 0, chunk));
            chunk = next;
        }
    }

    // ------------------------------------------------------------------
    // Cutoff tables
    // ------------------------------------------------------------------

    /// The entry words of a cutoff table, or `None` if torn.
    #[inline]
    pub(crate) fn cutoff_entries(&self, table: Addr) -> Option<&[AtomicU64]> {
        if table == NONE {
            return Some(&[]);
        }
        let slots = self.arena.word(table)?.load(Ordering::Relaxed) as u32 as usize;
        if !slots.is_power_of_two() || slots > 2 << (arena::CUTOFF_CLASSES - 1) {
            return None;
        }
        Some(&self.arena.slice(table, 1 + slots)?[1..])
    }

    /// Calls `f` for each live entry (R4), with the acquire load the promotion pairs with.
    #[inline]
    pub(crate) fn for_each_cutoff(&self, table: Addr, mut f: impl FnMut(Cutoff)) -> Option<()> {
        for entry in self.cutoff_entries(table)? {
            if let Some((slot, cutoff)) = unpack_cutoff(protocol::load_entry(entry)) {
                f(Cutoff { slot, cutoff });
            }
        }
        Some(())
    }

    /// The live entry count of a locked run's cutoff table.
    pub(crate) fn cutoff_live(&self, snap: &Snap) -> u32 {
        if snap.cutoffs == NONE {
            return 0;
        }
        self.arena
            .word(snap.cutoffs)
            .map_or(0, |w| (w.load(Ordering::Relaxed) >> 32) as u32)
    }

    /// `(index, cutoff)` of `slot`'s entry in a locked run.
    pub(crate) fn find_cutoff(&self, snap: &Snap, slot: Slot) -> Option<(usize, u32)> {
        self.cutoff_entries(snap.cutoffs)?
            .iter()
            .enumerate()
            .find_map(
                |(i, entry)| match unpack_cutoff(entry.load(Ordering::Relaxed)) {
                    Some((s, cutoff)) if s == slot.index() => Some((i, cutoff)),
                    _ => None,
                },
            )
    }

    /// Every live entry of a locked run.
    pub(crate) fn cutoffs(&self, snap: &Snap) -> Vec<Cutoff> {
        let mut out = Vec::new();
        self.for_each_cutoff(snap.cutoffs, |c| out.push(c));
        out
    }

    fn add_live(&self, table: Addr, delta: i64) {
        if let Some(word) = self.arena.word(table) {
            if delta >= 0 {
                word.fetch_add((delta as u64) << 32, Ordering::Relaxed);
            } else {
                word.fetch_sub(((-delta) as u64) << 32, Ordering::Relaxed);
            }
        }
    }

    /// Sets `slot`'s cutoff on a locked run: in place if it has an entry, else in an empty
    /// or tombstoned entry (W5). Grows the table when it is full, in a step unless the
    /// caller already holds one (`in_step`). The caller has checked [`PARTIAL_CAP`].
    pub(crate) fn set_cutoff(
        &self,
        locked: &Locked<'_>,
        slot: Slot,
        cutoff: u32,
        frees: &mut Frees,
        in_step: bool,
    ) -> Result<(), KvCacheEventError> {
        let snap = locked.snap();
        let packed = pack_cutoff(slot, cutoff);
        if let Some((i, _)) = self.find_cutoff(&snap, slot) {
            let entries = self
                .cutoff_entries(snap.cutoffs)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
            entries[i].store(packed, Ordering::Release);
            return Ok(());
        }
        if let Some(entries) = self.cutoff_entries(snap.cutoffs)
            && let Some(free) = entries.iter().find(|e| {
                let v = e.load(Ordering::Relaxed);
                v == EMPTY || v == TOMB
            })
        {
            free.store(packed, Ordering::Release);
            self.add_live(snap.cutoffs, 1);
            return Ok(());
        }
        // Grow: replacing the table pointer is a step (W4).
        let mut live = self.cutoffs(&snap);
        live.push(Cutoff {
            slot: slot.index(),
            cutoff,
        });
        let table = self.new_cutoff_table(&live)?;
        let step = (!in_step).then(|| locked.h.step());
        locked.h.cutoffs.store(table, Ordering::Relaxed);
        drop(step);
        self.release_cutoff_table(snap.cutoffs, frees);
        Ok(())
    }

    /// A new cutoff table holding `entries`, sized to the next power of two above them.
    pub(crate) fn new_cutoff_table(&self, entries: &[Cutoff]) -> Result<Addr, KvCacheEventError> {
        let needed = (entries.len() + 1).next_power_of_two().max(2);
        let class = (needed.trailing_zeros() - 1) as usize;
        if class >= arena::CUTOFF_CLASSES {
            return Err(KvCacheEventError::CapacityExhausted);
        }
        let slots = 2usize << class;
        let table = self.arena.alloc(Kind::Cutoffs, class)?;
        let words = self
            .arena
            .slice(table, 1 + slots)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        for (i, word) in words[1..].iter().enumerate() {
            let value = entries
                .get(i)
                .map_or(EMPTY, |e| pack_cutoff(Slot::from_index(e.slot), e.cutoff));
            word.store(value, Ordering::Relaxed);
        }
        words[0].store(
            slots as u64 | (entries.len() as u64) << 32,
            Ordering::Relaxed,
        );
        Ok(table)
    }

    pub(crate) fn release_cutoff_table(&self, table: Addr, frees: &mut Frees) {
        if table == NONE {
            return;
        }
        let Some(word) = self.arena.word(table) else {
            return;
        };
        let slots = word.load(Ordering::Relaxed) as u32;
        if slots.is_power_of_two() && slots >= 2 {
            frees
                .blocks
                .push((Kind::Cutoffs, (slots.trailing_zeros() - 1) as usize, table));
        }
    }

    /// Tombstones `slot`'s entry on a locked run (W5).
    pub(crate) fn remove_cutoff(&self, locked: &Locked<'_>, slot: Slot) -> bool {
        let snap = locked.snap();
        let Some((i, _)) = self.find_cutoff(&snap, slot) else {
            return false;
        };
        if let Some(entries) = self.cutoff_entries(snap.cutoffs) {
            entries[i].store(TOMB, Ordering::Release);
            self.add_live(snap.cutoffs, -1);
        }
        true
    }

    /// Promotes `slot` to a whole holder on a locked run: the bit, then the tombstone of
    /// its entry, if any (W5).
    pub(crate) fn promote(&self, locked: &Locked<'_>, slot: Slot) -> Result<(), KvCacheEventError> {
        let (word, bit) = slot.word_and_bit();
        let whole = self.whole_word_or_install(locked, word)?;
        let snap = locked.snap();
        match self.find_cutoff(&snap, slot) {
            Some((i, _)) => {
                let entries = self
                    .cutoff_entries(snap.cutoffs)
                    .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
                protocol::promote(whole, bit, &entries[i]);
                self.add_live(snap.cutoffs, -1);
            }
            None => {
                whole.fetch_or(bit, Ordering::Release);
            }
        }
        Ok(())
    }

    /// Clears `slot`'s whole bit on a locked run. Returns whether it was set.
    pub(crate) fn clear_whole(&self, locked: &Locked<'_>, slot: Slot) -> bool {
        let (word, bit) = slot.word_and_bit();
        self.whole_word(locked.h, word)
            .is_some_and(|w| w.fetch_and(!bit, Ordering::Release) & bit != 0)
    }

    /// What `slot` holds of a locked run: `len`, a cutoff, or `0`.
    pub(crate) fn held_locked(&self, locked: &Locked<'_>, snap: &Snap, slot: Slot) -> u32 {
        if self.has_whole(locked.h, slot) {
            return snap.len;
        }
        self.find_cutoff(snap, slot).map_or(0, |(_, cutoff)| cutoff)
    }

    /// Whether a locked run has neither whole nor partial holders.
    pub(crate) fn holderless(&self, locked: &Locked<'_>, snap: &Snap) -> bool {
        let mut any = false;
        self.for_each_whole_word(locked.h, snap, usize::MAX >> 8, |_, bits| any |= bits != 0);
        !any && self.cutoff_live(snap) == 0
    }

    // ------------------------------------------------------------------
    // Child tables
    // ------------------------------------------------------------------

    #[inline]
    fn table_words(&self, table: Addr) -> Option<(&[AtomicU64], usize)> {
        let slots = self.arena.word(table)?.load(Ordering::Relaxed) as u32 as usize;
        if !slots.is_power_of_two() || !(4..=4 << (arena::CHILD_CLASSES - 1)).contains(&slots) {
            return None;
        }
        Some((self.arena.slice(table, 2 + 2 * slots)?, slots))
    }

    #[inline]
    fn home(key: u64, slots: usize) -> usize {
        (key.wrapping_mul(GOLDEN) >> (64 - slots.trailing_zeros())) as usize
    }

    /// Probes `table` for `key` (readers and planners). An unpublished value reads as
    /// absent, and tombstones are skipped.
    #[inline]
    pub(crate) fn probe_child(&self, table: Addr, key: u64) -> Probe {
        if table == NONE {
            return Probe::Absent;
        }
        let Some((words, slots)) = self.table_words(table) else {
            return Probe::Torn;
        };
        let mask = slots - 1;
        let mut i = Self::home(key, slots);
        for _ in 0..slots {
            let k = words[2 + 2 * i].load(Ordering::Acquire);
            if k == EMPTY {
                return Probe::Absent;
            }
            if k == key {
                let value = words[3 + 2 * i].load(Ordering::Acquire);
                if value == EMPTY {
                    return Probe::Absent;
                }
                if value != TOMB {
                    let (id, generation) = split_value(value);
                    return Probe::Found(id, generation);
                }
            }
            i = (i + 1) & mask;
        }
        Probe::Absent
    }

    /// The live child count of a child table.
    pub(crate) fn child_live(&self, table: Addr) -> u32 {
        if table == NONE {
            return 0;
        }
        self.arena
            .word(table + 1)
            .map_or(0, |w| w.load(Ordering::Acquire) as u32)
    }

    /// `(used, slots)` of a child table.
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn child_load(&self, table: Addr) -> (u32, u32) {
        self.arena.word(table).map_or((0, 0), |w| {
            let v = w.load(Ordering::Relaxed);
            ((v >> 32) as u32, v as u32)
        })
    }

    /// Every published child of a table as `(key, id, generation)`.
    pub(crate) fn children_of(&self, table: Addr) -> Vec<(u64, RunId, u32)> {
        let Some((words, slots)) = self.table_words(table) else {
            return Vec::new();
        };
        (0..slots)
            .filter_map(|i| {
                let key = words[2 + 2 * i].load(Ordering::Acquire);
                let value = words[3 + 2 * i].load(Ordering::Acquire);
                (key != EMPTY && value != EMPTY && value != TOMB).then(|| {
                    let (id, generation) = split_value(value);
                    (key, id, generation)
                })
            })
            .collect()
    }

    /// Tombstone count of a table, for the shape report.
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn child_tombstones(&self, table: Addr) -> u32 {
        let Some((words, slots)) = self.table_words(table) else {
            return 0;
        };
        (0..slots)
            .filter(|&i| words[3 + 2 * i].load(Ordering::Relaxed) == TOMB)
            .count() as u32
    }

    /// Longest probe a miss would take, for the shape report.
    #[cfg(any(test, feature = "bench"))]
    pub(crate) fn child_max_probe(&self, table: Addr) -> u32 {
        let Some((words, slots)) = self.table_words(table) else {
            return 0;
        };
        let mask = slots - 1;
        let mut longest = 0;
        let mut run = 0;
        // Walk twice around so a run that wraps is measured whole.
        for i in 0..2 * slots {
            if words[2 + 2 * (i & mask)].load(Ordering::Relaxed) == EMPTY {
                run = 0;
            } else {
                run += 1;
                longest = longest.max(run);
            }
        }
        longest.min(slots as u32)
    }

    /// Lock-free claim of `key` in the parent's table for the unpublished child `child`
    /// (5.3). `snap` is the validated snapshot the caller planned from.
    pub(crate) fn claim_child(
        &self,
        parent: &RunHeader,
        snap: &Snap,
        key: u64,
        child: (RunId, u32),
    ) -> Claim {
        if snap.children == NONE {
            return Claim::Locked;
        }
        if !protocol::claim_enter(&parent.inflight, &parent.version, snap.version) {
            return Claim::Replan;
        }
        let outcome = self.claim_counted(snap.children, key, child);
        protocol::claim_exit(&parent.inflight);
        outcome
    }

    fn claim_counted(&self, table: Addr, key: u64, child: (RunId, u32)) -> Claim {
        let Some((words, slots)) = self.table_words(table) else {
            return Claim::Locked;
        };
        let header = &words[0];
        let mask = slots - 1;
        let mut reserved = false;
        let unreserve = |reserved: bool| {
            if reserved {
                header.fetch_sub(1 << 32, Ordering::Relaxed);
            }
        };
        let mut i = Self::home(key, slots);
        for _ in 0..slots {
            let slot_key = &words[2 + 2 * i];
            let value = &words[3 + 2 * i];
            let mut k = slot_key.load(Ordering::Acquire);
            if k == EMPTY {
                if !reserved {
                    // Reserve load budget: used + 1 <= 3/4 slots, else the locked path.
                    let mut current = header.load(Ordering::Relaxed);
                    loop {
                        let used = current >> 32;
                        if (used + 1) * 4 > slots as u64 * 3 {
                            return Claim::Locked;
                        }
                        match header.compare_exchange_weak(
                            current,
                            current + (1 << 32),
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(actual) => current = actual,
                        }
                    }
                    reserved = true;
                }
                match slot_key.compare_exchange(EMPTY, key, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        value.store(child_value(child.0, child.1), Ordering::Release);
                        words[1].fetch_add(1, Ordering::Relaxed);
                        return Claim::Claimed;
                    }
                    Err(winner) => k = winner,
                }
            }
            if k == key {
                let v = value.load(Ordering::Acquire);
                if v == EMPTY {
                    unreserve(reserved);
                    return Claim::Replan;
                }
                if v != TOMB {
                    unreserve(reserved);
                    let (id, generation) = split_value(v);
                    return Claim::Exists(id, generation);
                }
            }
            i = (i + 1) & mask;
        }
        unreserve(reserved);
        Claim::Locked
    }

    /// A new child table holding `children` (key, id, generation), sized so they fill at most
    /// 3/8 of it.
    pub(crate) fn new_child_table(
        &self,
        children: &[(u64, RunId, u32)],
    ) -> Result<Addr, KvCacheEventError> {
        let mut slots = 4usize;
        while children.len() * 8 > slots * 3 {
            slots *= 2;
        }
        let class = (slots.trailing_zeros() - 2) as usize;
        if class >= arena::CHILD_CLASSES {
            return Err(KvCacheEventError::CapacityExhausted);
        }
        let table = self.arena.alloc(Kind::Children, class)?;
        let words = self
            .arena
            .slice(table, 2 + 2 * slots)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        for word in &words[2..] {
            word.store(EMPTY, Ordering::Relaxed);
        }
        let mask = slots - 1;
        for &(key, id, generation) in children {
            let mut i = Self::home(key, slots);
            while words[2 + 2 * i].load(Ordering::Relaxed) != EMPTY {
                i = (i + 1) & mask;
            }
            words[2 + 2 * i].store(key, Ordering::Relaxed);
            words[3 + 2 * i].store(child_value(id, generation), Ordering::Relaxed);
        }
        let count = children.len() as u64;
        words[1].store(count, Ordering::Relaxed);
        words[0].store(slots as u64 | count << 32, Ordering::Relaxed);
        Ok(table)
    }

    pub(crate) fn release_child_table(&self, table: Addr, frees: &mut Frees) {
        if table == NONE {
            return;
        }
        let Some(word) = self.arena.word(table) else {
            return;
        };
        let slots = word.load(Ordering::Relaxed) as u32;
        if slots.is_power_of_two() && slots >= 4 {
            frees
                .blocks
                .push((Kind::Children, (slots.trailing_zeros() - 2) as usize, table));
        }
    }

    /// Locked insert of `key` into a locked parent (5.3): opens a step that excludes
    /// claims, re-probes, reuses the first tombstone on the probe chain or an empty slot,
    /// and rebuilds the table past its load limit. The old table is freed after unlock.
    pub(crate) fn insert_child_locked(
        &self,
        parent: &Locked<'_>,
        offset: u32,
        key: u64,
        child: (RunId, u32),
        frees: &mut Frees,
    ) -> Result<Inserted, KvCacheEventError> {
        let _step = parent.h.step_excluding_claims();
        let snap = parent.snap();
        if offset > snap.len || (offset == 0 && parent.id != ROOT) {
            return Ok(Inserted::Replan);
        }
        bump(&self.stats.locked_inserts);
        if let Probe::Found(id, generation) = self.probe_child(snap.children, key) {
            return Ok(Inserted::Exists(id, generation));
        }
        if snap.children != NONE
            && let Some((words, slots)) = self.table_words(snap.children)
        {
            let mask = slots - 1;
            let header = words[0].load(Ordering::Relaxed);
            let used = header >> 32;
            let mut i = Self::home(key, slots);
            for _ in 0..slots {
                let k = words[2 + 2 * i].load(Ordering::Relaxed);
                let v = words[3 + 2 * i].load(Ordering::Relaxed);
                if v == TOMB || (k == EMPTY && (used + 1) * 4 <= slots as u64 * 3) {
                    if k == EMPTY {
                        words[0].store(header + (1 << 32), Ordering::Relaxed);
                    }
                    words[2 + 2 * i].store(key, Ordering::Relaxed);
                    words[3 + 2 * i].store(child_value(child.0, child.1), Ordering::Release);
                    words[1].fetch_add(1, Ordering::Relaxed);
                    return Ok(Inserted::Claimed);
                }
                if k == EMPTY {
                    break;
                }
                i = (i + 1) & mask;
            }
        }
        // No table, or no room on the probe chain: rebuild, which purges tombstones.
        let mut children = self.children_of(snap.children);
        children.push((key, child.0, child.1));
        let table = self.new_child_table(&children)?;
        parent.h.children.store(table, Ordering::Relaxed);
        self.release_child_table(snap.children, frees);
        if snap.children != NONE {
            bump(&self.stats.table_rebuilds);
        }
        Ok(Inserted::Claimed)
    }

    /// Unlinks the child `(id, generation)` with `key` from a locked parent (fix 6): finds the
    /// slot by key and tombstones its value. `live` drops; `used` does not.
    pub(crate) fn unlink_child(
        &self,
        parent: &Locked<'_>,
        key: u64,
        id: RunId,
        generation: u32,
    ) -> bool {
        let snap = parent.snap();
        let Some((words, slots)) = self.table_words(snap.children) else {
            return false;
        };
        let mask = slots - 1;
        let target = child_value(id, generation);
        let mut i = Self::home(key, slots);
        for _ in 0..slots {
            let k = words[2 + 2 * i].load(Ordering::Relaxed);
            if k == EMPTY {
                return false;
            }
            if k == key && words[3 + 2 * i].load(Ordering::Relaxed) == target {
                words[3 + 2 * i].store(TOMB, Ordering::Release);
                words[1].fetch_sub(1, Ordering::Relaxed);
                return true;
            }
            i = (i + 1) & mask;
        }
        false
    }

    // ------------------------------------------------------------------
    // Forwarding records
    // ------------------------------------------------------------------

    /// The forwarding record that applies to position `offset`: the first, in decreasing
    /// `at`, with `at <= offset`. `None` on a torn table, `Some(None)` without one.
    #[inline]
    pub(crate) fn forward_for(
        &self,
        table: Addr,
        offset: u32,
    ) -> Option<Option<(u32, RunId, u32)>> {
        if table == NONE {
            return Some(None);
        }
        let header = self.arena.word(table)?.load(Ordering::Relaxed);
        let (count, capacity) = (header as u32 as usize, (header >> 32) as usize);
        if count > capacity || capacity > 1 << (arena::FORWARD_CLASSES - 1) {
            return None;
        }
        let words = self.arena.slice(table, 1 + 2 * capacity)?;
        for r in 0..count {
            let first = words[1 + 2 * r].load(Ordering::Relaxed);
            let at = first as u32;
            if at <= offset {
                let generation = words[2 + 2 * r].load(Ordering::Relaxed) as u32;
                return Some(Some((at, (first >> 32) as u32, generation)));
            }
        }
        Some(None)
    }

    /// Every record of a table as `(at, suffix, generation)`.
    pub(crate) fn forwards(&self, table: Addr) -> Vec<(u32, RunId, u32)> {
        if table == NONE {
            return Vec::new();
        }
        let Some(header) = self.arena.word(table).map(|w| w.load(Ordering::Relaxed)) else {
            return Vec::new();
        };
        let (count, capacity) = (header as u32 as usize, (header >> 32) as usize);
        let Some(words) = self.arena.slice(table, 1 + 2 * capacity) else {
            return Vec::new();
        };
        (0..count.min(capacity))
            .map(|r| {
                let first = words[1 + 2 * r].load(Ordering::Relaxed);
                let generation = words[2 + 2 * r].load(Ordering::Relaxed) as u32;
                (first as u32, (first >> 32) as u32, generation)
            })
            .collect()
    }

    /// Appends a record to a locked run, inside the caller's step.
    pub(crate) fn push_forward(
        &self,
        locked: &Locked<'_>,
        record: (u32, RunId, u32),
        frees: &mut Frees,
    ) -> Result<(), KvCacheEventError> {
        let table = locked.h.forwards.load(Ordering::Relaxed);
        let mut records = self.forwards(table);
        if table != NONE
            && let Some(header) = self.arena.word(table)
        {
            let current = header.load(Ordering::Relaxed);
            let (count, capacity) = (current as u32, (current >> 32) as u32);
            if count < capacity {
                let words = self
                    .arena
                    .slice(table, 1 + 2 * capacity as usize)
                    .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
                let r = count as usize;
                words[1 + 2 * r].store(
                    u64::from(record.0) | u64::from(record.1) << 32,
                    Ordering::Relaxed,
                );
                words[2 + 2 * r].store(u64::from(record.2), Ordering::Relaxed);
                header.store(current + 1, Ordering::Relaxed);
                return Ok(());
            }
        }
        records.push(record);
        let capacity = records.len().next_power_of_two();
        let class = capacity.trailing_zeros() as usize;
        if class >= arena::FORWARD_CLASSES {
            return Err(KvCacheEventError::CapacityExhausted);
        }
        let new = self.arena.alloc(Kind::Forwards, class)?;
        let words = self
            .arena
            .slice(new, 1 + 2 * capacity)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        for (r, &(at, suffix, generation)) in records.iter().enumerate() {
            words[1 + 2 * r].store(u64::from(at) | u64::from(suffix) << 32, Ordering::Relaxed);
            words[2 + 2 * r].store(u64::from(generation), Ordering::Relaxed);
        }
        words[0].store(
            records.len() as u64 | (capacity as u64) << 32,
            Ordering::Relaxed,
        );
        locked.h.forwards.store(new, Ordering::Relaxed);
        if table != NONE {
            frees.blocks.push((
                Kind::Forwards,
                (self.forwards_capacity(table).trailing_zeros()) as usize,
                table,
            ));
        }
        Ok(())
    }

    fn forwards_capacity(&self, table: Addr) -> u32 {
        self.arena
            .word(table)
            .map_or(1, |w| (w.load(Ordering::Relaxed) >> 32) as u32)
            .max(1)
    }

    // ------------------------------------------------------------------
    // Run lifecycle
    // ------------------------------------------------------------------

    /// Allocates and reincarnates a run (4.9): generation + 1 (skipping 0), every field
    /// reset, inside one step under the run lock. The run is unpublished and stable when
    /// this returns.
    pub(crate) fn create(
        &self,
        parent: RunId,
        start: u32,
        head: LocalBlockHash,
        array: Addr,
        base: u32,
        len: u32,
    ) -> Result<(RunId, u32), KvCacheEventError> {
        let id = self.slab.alloc()?;
        let h = self
            .slab
            .header(id)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        let _guard = h.lock.lock();
        let step = h.step();
        let generation = match h.generation.load(Ordering::Relaxed).wrapping_add(1) {
            0 => 1,
            generation => generation,
        };
        h.generation.store(generation, Ordering::Relaxed);
        h.flags.store(0, Ordering::Relaxed);
        h.parent.store(parent, Ordering::Relaxed);
        h.start.store(start, Ordering::Relaxed);
        h.head.store(head.0, Ordering::Relaxed);
        h.array.store(array, Ordering::Relaxed);
        h.base.store(base, Ordering::Relaxed);
        h.len.store(len, Ordering::Relaxed);
        h.children.store(NONE, Ordering::Relaxed);
        h.cutoffs.store(NONE, Ordering::Relaxed);
        h.forwards.store(NONE, Ordering::Relaxed);
        h.overflow.store(NONE, Ordering::Relaxed);
        for word in &h.whole {
            word.store(0, Ordering::Relaxed);
        }
        drop(step);
        Ok((id, generation))
    }

    /// Kills a locked run inside a step that waits out in-flight claims (4.9). A claim
    /// that landed meanwhile aborts the kill. Returns whether the run died; its memory and
    /// id go to `frees`.
    pub(crate) fn kill(&self, locked: &Locked<'_>, frees: &mut Frees) -> bool {
        let h = locked.h;
        let step = h.step_excluding_claims();
        let snap = locked.snap();
        if self.child_live(snap.children) > 0 {
            drop(step);
            bump(&self.stats.kills_aborted);
            return false;
        }
        h.flags.fetch_or(FLAG_DEAD, Ordering::Relaxed);
        h.len.store(0, Ordering::Relaxed);
        h.array.store(NONE, Ordering::Relaxed);
        h.children.store(NONE, Ordering::Relaxed);
        h.cutoffs.store(NONE, Ordering::Relaxed);
        h.forwards.store(NONE, Ordering::Relaxed);
        h.overflow.store(NONE, Ordering::Relaxed);
        for word in &h.whole {
            word.store(0, Ordering::Relaxed);
        }
        drop(step);
        self.release_array(snap.array, frees);
        self.release_child_table(snap.children, frees);
        self.release_cutoff_table(snap.cutoffs, frees);
        if snap.forwards != NONE {
            frees.blocks.push((
                Kind::Forwards,
                self.forwards_capacity(snap.forwards).trailing_zeros() as usize,
                snap.forwards,
            ));
        }
        self.release_chunks(snap.overflow, frees);
        frees.ids.push(locked.id);
        true
    }

    /// Discards a run that was created but never published.
    pub(crate) fn discard(&self, id: RunId, generation: u32, frees: &mut Frees) {
        if let Some(locked) = self.lock(id, generation) {
            self.kill(&locked, frees);
        }
    }

    /// Returns deferred frees to the free lists. The caller holds no run lock (A2).
    pub(crate) fn flush(&self, frees: &mut Frees) {
        for (kind, class, addr) in frees.blocks.drain(..) {
            self.arena.free(kind, class, addr);
        }
        for id in frees.ids.drain(..) {
            self.slab.free.push(id);
        }
    }

    /// Test hook: overwrites a run's generation (generation-wrap tests).
    #[cfg(test)]
    pub(crate) fn set_generation(&self, id: RunId, generation: u32) {
        if let Some(h) = self.slab.header(id) {
            let _guard = h.lock.lock();
            let _step = h.step();
            h.generation.store(generation, Ordering::Relaxed);
        }
    }

    /// Test hook: overwrites a run's version (version-wrap tests). `version` must be even.
    #[cfg(test)]
    pub(crate) fn set_version(&self, id: RunId, version: u64) {
        assert_eq!(version & 1, 0);
        if let Some(h) = self.slab.header(id) {
            let _guard = h.lock.lock();
            h.version.store(version, Ordering::Release);
        }
    }

    /// Test hook: leaves a step open on a run as if a panic had unwound through it.
    #[cfg(test)]
    pub(crate) fn poison(&self, id: RunId) {
        if let Some(h) = self.slab.header(id) {
            let _guard = h.lock.lock();
            h.step().unwind();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks(hashes: &[u64]) -> Vec<(LocalBlockHash, ExternalSequenceBlockHash)> {
        hashes
            .iter()
            .map(|&h| (LocalBlockHash(h), ExternalSequenceBlockHash(h + 1000)))
            .collect()
    }

    #[test]
    fn slab_ids_map_to_distinct_headers() {
        assert_eq!(slab_locate(0), (0, 0));
        assert_eq!(slab_locate(1023), (0, 1023));
        assert_eq!(slab_locate(1024), (1, 0));
        assert_eq!(slab_locate(u32::MAX - 1024).0, SLAB_SEGMENTS - 1);
        assert!(slab_locate(u32::MAX - 1023).0 >= SLAB_SEGMENTS);
    }

    #[test]
    fn child_keys_separate_offsets_and_avoid_sentinels() {
        let head = LocalBlockHash(42);
        assert_ne!(child_key(1, head), child_key(2, head));
        assert_ne!(child_key(0, head), head.0);
        for offset in 0..1000 {
            let key = child_key(offset, LocalBlockHash(offset as u64));
            assert_ne!(key, EMPTY);
            assert_ne!(key, TOMB);
        }
    }

    #[test]
    fn appends_reuse_room_then_reallocate_and_splits_share_arrays() {
        let runs = Runs::new(ReaderMode::Optimistic, READ_RETRIES);
        let array = runs.new_array(&blocks(&[1, 2, 3, 4, 5, 6, 7, 8])).unwrap();
        let (id, generation) = runs
            .create(ROOT, 0, LocalBlockHash(1), array, 0, 8)
            .unwrap();
        let mut frees = Frees::default();
        {
            let locked = runs.lock(id, generation).unwrap();
            let snap = locked.snap();
            // capacity_for(8) = 9 rounds to class 16.
            assert!(runs.has_room(&snap, 1));
            assert!(
                runs.append_locked(&locked, &snap, &blocks(&[9]), &mut frees)
                    .unwrap()
            );
            let snap = locked.snap();
            assert_eq!(snap.len, 9);
            assert_eq!(snap.array, array);
            let _step = locked.h.step();
            let in_place = runs
                .append_locked(
                    &locked,
                    &snap,
                    &blocks(&(10..40).collect::<Vec<_>>()),
                    &mut frees,
                )
                .unwrap();
            assert!(!in_place);
        }
        runs.flush(&mut frees);
        let Read::Ok(columns) = runs.read(id, generation, |_, snap| {
            let c = runs.columns(snap)?;
            Some((0..c.len()).map(|i| c.local(i).0).collect::<Vec<_>>())
        }) else {
            panic!("run is live");
        };
        assert_eq!(columns, (1..40).collect::<Vec<_>>());
        // The first array was released when the append reallocated.
        let free = runs.arena.free_blocks();
        assert!(
            free.iter()
                .any(|&(kind, _, addr)| kind == Kind::Array && addr == array)
        );
    }

    #[test]
    fn claims_stop_at_three_quarters_and_locked_inserts_reuse_tombstones() {
        let runs = Runs::new(ReaderMode::Optimistic, READ_RETRIES);
        let mut frees = Frees::default();
        let root = runs.lock(ROOT, ROOT_GEN).unwrap();
        // Create a 4-slot table with one child through the locked path.
        let key0 = child_key(0, LocalBlockHash(0));
        assert!(matches!(
            runs.insert_child_locked(&root, 0, key0, (7, 1), &mut frees)
                .unwrap(),
            Inserted::Claimed
        ));
        drop(root);
        let snap = |runs: &Runs| match runs.read(ROOT, ROOT_GEN, |_, s| Some(*s)) {
            Read::Ok(s) => s,
            Read::Gone => panic!("root is live"),
        };
        let h = runs.header(ROOT).unwrap();
        let mut claimed = 1;
        for head in 1..10u64 {
            let s = snap(&runs);
            match runs.claim_child(
                h,
                &s,
                child_key(0, LocalBlockHash(head)),
                (100 + head as u32, 1),
            ) {
                Claim::Claimed => claimed += 1,
                Claim::Locked => break,
                _ => panic!("unexpected claim outcome"),
            }
        }
        let s = snap(&runs);
        let (used, slots) = runs.child_load(s.children);
        assert_eq!(claimed, 3);
        assert!(used * 4 <= slots * 3);
        // Unlink one and re-insert under the lock: the tombstone is reused, `used` stays.
        let root = runs.lock(ROOT, ROOT_GEN).unwrap();
        assert!(runs.unlink_child(&root, key0, 7, 1));
        assert_eq!(runs.child_live(s.children), 2);
        let key = child_key(0, LocalBlockHash(55));
        assert!(matches!(
            runs.insert_child_locked(&root, 0, key, (55, 1), &mut frees)
                .unwrap(),
            Inserted::Claimed
        ));
        drop(root);
        let s2 = snap(&runs);
        // Either the tombstone was reused in place or the table was rebuilt.
        assert_eq!(runs.child_live(s2.children), 3);
        assert_eq!(runs.probe_child(s2.children, key), Probe::Found(55, 1));
        assert_eq!(runs.probe_child(s2.children, key0), Probe::Absent);
        runs.flush(&mut frees);
    }

    #[test]
    fn forwarding_records_pick_the_latest_applicable_split() {
        let runs = Runs::new(ReaderMode::Optimistic, READ_RETRIES);
        let array = runs.new_array(&blocks(&[1, 2, 3, 4, 5, 6])).unwrap();
        let (id, generation) = runs
            .create(ROOT, 0, LocalBlockHash(1), array, 0, 6)
            .unwrap();
        let mut frees = Frees::default();
        let locked = runs.lock(id, generation).unwrap();
        runs.push_forward(&locked, (4, 50, 3), &mut frees).unwrap();
        runs.push_forward(&locked, (2, 60, 9), &mut frees).unwrap();
        runs.push_forward(&locked, (1, 70, 2), &mut frees).unwrap();
        let table = locked.h.forwards.load(Ordering::Relaxed);
        assert_eq!(runs.forward_for(table, 5), Some(Some((4, 50, 3))));
        assert_eq!(runs.forward_for(table, 3), Some(Some((2, 60, 9))));
        assert_eq!(runs.forward_for(table, 1), Some(Some((1, 70, 2))));
        assert_eq!(runs.forward_for(table, 0), Some(None));
        drop(locked);
        runs.flush(&mut frees);
    }

    #[test]
    fn coverage_chunks_install_in_slot_order() {
        let runs = Runs::new(ReaderMode::Optimistic, READ_RETRIES);
        let array = runs.new_array(&blocks(&[1, 2])).unwrap();
        let (id, generation) = runs
            .create(ROOT, 0, LocalBlockHash(1), array, 0, 2)
            .unwrap();
        let locked = runs.lock(id, generation).unwrap();
        for slot in [5000usize, 300, 1, 70_000 % 65_536, 129] {
            runs.promote(&locked, Slot::from_index(slot)).unwrap();
        }
        let mut slots = runs.whole_slots(&locked);
        slots.sort_unstable();
        assert_eq!(slots, vec![1, 129, 300, 4464, 5000]);
        let snap = locked.snap();
        let mut words = Vec::new();
        runs.for_each_whole_word(locked.h, &snap, 2 + 4 * 20, |w, _| words.push(w));
        assert!(
            words.windows(2).all(|p| p[0] < p[1]),
            "chunks out of order: {words:?}"
        );
        assert!(runs.clear_whole(&locked, Slot::from_index(300)));
        assert!(!runs.has_whole(locked.h, Slot::from_index(300)));
    }
}
