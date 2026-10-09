// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arena index C: run-compressed arena storage under Dynamo's own concurrency model.
//!
//! Design ideas adapted from SMG's chain index (smg-project/smg#2814, Apache-2.0);
//! clean-room implementation. The storage ideas are SMG's: runs stored once in an
//! id-addressed arena as windows into shared hash arrays, children at any offset, splits
//! that share arrays and leave forwarding records, and per-rank block maps keyed by the
//! engine's hash. The concurrency is CRTC's, kept unchanged: `ThreadPoolIndexer`'s sticky
//! lanes, plan-then-validate writers with per-run shape gates, readers under the state
//! read lock, and `crossbeam-epoch` for every reuse.
//!
//! # Storage
//!
//! - Run headers live in a segmented slab and never move; a run is named by a `u32` id
//!   (`0` none, `1` ROOT). A header holds the window `[base, base + len)` into a hash
//!   array, the whole-holder bits (two words inline, 256-slot overflow chunks in the
//!   arena), and arena addresses of its child table, cutoff table and forwarding records.
//! - Every other structure lives in one `u32`-addressed arena of `AtomicU64` words with
//!   size-class free lists (`arena.rs`). Hash arrays keep the local and external hash
//!   columns of a chain once, with a reference count of the runs whose windows lie in it.
//! - A child hangs at any offset `o` of its parent, keyed by `(o, first local hash)` in
//!   an open-addressing table (`table.rs`). A store that diverges inside a run hangs a
//!   child there; nothing splits on divergence. A run splits only when its partial holders
//!   would exceed [`run::PARTIAL_CAP`], at their median cutoff: the suffix shares the
//!   array, the prefix is sealed and records where its tail went.
//! - Each lane keeps, per rank, a map from external hash to `(run, offset)` with 16-byte
//!   slots and no generation. An entry is resolved under the event's pin: a dead run
//!   means stale, an offset past a sealed run's end follows its forwarding records, and
//!   the stored external hash must equal the key (one hash per position, H1). Resolution
//!   rewrites a moved entry in place (path compression), so splits never touch maps.
//!
//! # Concurrency
//!
//! Each run has a shape `gate` and a `state` lock (both `parking_lot` `RwLock`s, kept
//! separate as in CRTC) and a `version` bumped under the exclusive gate on every shape
//! change.
//!
//! | Operation | Gate | State | Version |
//! |---|---|---|---|
//! | Lookup hop, non-root | none | read | none |
//! | Lookup hop, ROOT | none | none (epoch-published table) | none |
//! | Plan a store step | read | read | record |
//! | Claim a child slot (table exists) | read | none | no bump |
//! | Create, grow or rebuild a child table | write | write | validate; bump |
//! | Promote a rank to whole | read | write only to drop a stale cutoff | none |
//! | Raise or add a cutoff | read | write | none |
//! | Append (in place or by reallocation) | write | write | validate; re-probe end child; bump |
//! | Prefix-cap split | write | write | bump |
//! | Truncate a rank (removal) | write | upgradable read, upgraded to change a cutoff | none |
//! | Unlink a child | parent read; child `try_write` | child write | child: bump and `DEAD` |
//! | Dump a run | read | read | none |
//! | Slot sweep | write | write | none |
//!
//! - A writer holds at most one run's locks, except that an unlink holds the parent's
//!   shared gate while it try-locks the child. No thread waits on a run lock while it
//!   holds another run's exclusive gate.
//! - Claims under the shared gate never bump the version, so an append re-probes the
//!   end child it would shadow after validating, instead of relying on the version.
//! - Readers take one epoch pin per walk; stores and removals one per event. Every run
//!   id, array, table and chunk is reused only through a lane's `FreeBatch`, which defers
//!   to the epoch, so a pinned reader or writer never sees an id it read reincarnated.
//!   A reader that steps into a child being unlinked sees it live or `DEAD`; a `DEAD`
//!   run has no holders, so stopping there undercounts at most.
//! - Lookups may undercount during races and never overcount past a valid reachable
//!   prefix: a rank continues into a child only if it holds the parent's positions up to
//!   the child's offset, intersected on every hop.
//!
//! # Reclamation
//!
//! A removal or clear that leaves a run with no holders and no live children unlinks it
//! at once (and cascades upward). A failed try-lock or a moved parent puts the run on the
//! lane's pending list, retried while the lane is idle. A volume sweep runs when the
//! lanes' tallies say enough dead blocks are still linked, with CRTC's five-minute timer
//! as a backstop.
//!
//! Not supported: approximate-LRU and TTL pruning (`ApproximateLru` tasks reply
//! `Unsupported`), branch-sharding anchors, and the `HashLifecycle` delegate.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

use rustc_hash::FxHashMap;

use crate::cleanup::CleanupState;
use crate::protocols::*;

mod arena;
mod dump;
mod lookup;
mod rank_map;
mod reclaim;
mod remove;
mod run;
mod slots;
mod store;
mod sync;
mod sync_impl;
mod table;
mod types;

#[cfg(any(test, feature = "bench"))]
mod probe;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod harness_impl;

use arena::{FreeBatch, Storage};
use rank_map::RankMap;
use reclaim::{PendingUnlink, ReclaimShared, ReclaimTally};
use run::{NONE, ROOT};
use slots::{Slot, SlotRegistry};

/// Slots in ROOT's first child table.
const ROOT_SLOTS: u32 = 64;
/// Re-plans one store may take before it fails visibly.
const STORE_REPLANS: usize = 64;

/// Arena-backed [`SyncIndexer`](super::SyncIndexer) for [`ThreadPoolIndexer`](super::ThreadPoolIndexer).
pub struct ArenaIndexC {
    store: Arc<Storage>,
    slots: SlotRegistry,
    /// Slot-bit words a reader must look at: enough for every slot issued so far.
    live_words: AtomicUsize,
    cleanup: CleanupState,
    reclaim: ReclaimShared,
    counters: Counters,
    #[cfg(test)]
    hooks: probe::Hooks,
}

/// Shape counters for reports.
#[derive(Default)]
struct Counters {
    splits: AtomicU64,
    unlinks: AtomicU64,
    plan_retries: AtomicU64,
    h1_violations: AtomicU64,
    ext_conflicts: AtomicU64,
    volume_sweeps: AtomicU64,
    store_failures: AtomicU64,
    pending_unlinks: AtomicI64,
    max_forward_hops: AtomicU64,
    child_tables: AtomicI64,
}

/// One rank's state on its sticky lane: two integers per block, nothing shared.
#[derive(Default)]
pub(super) struct RankLookup {
    map: RankMap,
}

/// A lane's private state.
#[derive(Default)]
pub(super) struct CLane {
    ranks: FxHashMap<WorkerWithDpRank, RankLookup>,
    free: FreeBatch,
    pending: Vec<PendingUnlink>,
    tally: ReclaimTally,
}

impl CLane {
    pub(super) fn block_counts(&self) -> impl Iterator<Item = (WorkerWithDpRank, usize)> + '_ {
        self.ranks
            .iter()
            .map(|(rank, lookup)| (*rank, lookup.map.len()))
    }
}

impl Default for ArenaIndexC {
    fn default() -> Self {
        Self::new()
    }
}

impl ArenaIndexC {
    pub fn new() -> Self {
        let store = Arc::new(Storage::new());
        let block = store
            .arena
            .alloc(table::words_for(ROOT_SLOTS), true)
            .expect("the first arena segment fits ROOT's table");
        table::Table::init(
            store.arena.slice(block.addr, table::words_for(ROOT_SLOTS)),
            ROOT_SLOTS,
        );
        let root = store.run(ROOT);
        root.generation.store(1, Ordering::Relaxed);
        root.parent.store(ROOT, Ordering::Relaxed);
        root.children.store(block.addr, Ordering::Release);
        Self {
            store,
            slots: SlotRegistry::default(),
            live_words: AtomicUsize::new(run::INLINE_WORDS),
            cleanup: CleanupState::new(),
            reclaim: ReclaimShared::default(),
            counters: Counters::default(),
            #[cfg(test)]
            hooks: probe::Hooks::default(),
        }
    }

    /// Makes slot bits of `slot` visible to readers before any is written.
    #[inline]
    fn note_slot(&self, slot: Slot) {
        let words = slot.index() / 64 + 1;
        if words > self.live_words.load(Ordering::Relaxed) {
            self.live_words.fetch_max(words, Ordering::AcqRel);
        }
    }

    /// Hands a lane's collected frees to the epoch.
    fn flush_frees(&self, lane: &mut CLane) {
        lane.free.flush(&self.store);
    }

    pub(super) fn new_lane(&self) -> CLane {
        CLane::default()
    }
}

/// Length of the common prefix of `window.local[from..]` and `blocks`' local hashes.
#[inline]
fn common_prefix(window: run::Window<'_>, from: usize, blocks: &[KvCacheStoredBlockData]) -> usize {
    let avail = window.len().saturating_sub(from).min(blocks.len());
    (0..avail)
        .take_while(|&i| window.local(from + i) == blocks[i].tokens_hash.0)
        .count()
}

impl Drop for ArenaIndexC {
    fn drop(&mut self) {
        // Deferred frees own an `Arc` of the storage; nudge the epoch so they run soon.
        crossbeam_epoch::pin().flush();
    }
}
