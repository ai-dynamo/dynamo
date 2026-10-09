// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Arena index (campaign branch B): run-compressed chains in an id-addressed arena,
//! version-validated optimistic reads, rank-owned block maps, and a work-stealing lane pool.
//!
//! Design ideas adapted from SMG's chain index (smg-project/smg#2814, Apache-2.0);
//! clean-room implementation written from the campaign's spec B, not from SMG's source.
//! The adapted ideas are run-compressed storage in an id-addressed arena, windows into
//! shared hash arrays, children at any offset, splits that share arrays and leave
//! forwarding records, worker-owned block maps keyed by engine hash, and a pool that steals
//! whole workers between lanes. The spec's corrections of that design (generations in
//! their own word, release fences on steps and free-list reuse, poisoned steps, generation-
//! checked map entries, bounded child-table load, growable limits, hard scheduler checks,
//! bounded writer retries) are marked "fix N" where they appear.
//!
//! Layout:
//! - `runs`: run headers in a type-stable slab; hash arrays, coverage chunks, cutoff and
//!   child tables, and forwarding records in the word arena (`arena`); read attempts.
//! - `protocol`: the lock-free primitives, shared with the loom models in
//!   `lib/kv-router/arena-loom`.
//! - `lookup`: `find_matches` and the details variant.
//! - `write`: stores, removals, clears, rank removal, splits, reclamation.
//! - `pool`: the lane pool behind `SyncIndexer::worker`.
//! - `slots`: dense rank slots, adapted from CRTC's registry.
//!
//! Semantics match CRTC's (spec B 12), with two documented differences, both within the
//! no-overcount contract: a rank whose partial cutoff covers a divergence continues into
//! the child hanging there, and re-storing an evicted mid-chain block can make a
//! still-held descendant reachable again.

use std::sync::Arc;

use crate::cleanup::{CleanupGuard, CleanupState};
use crate::indexer::{
    AnchorTask, HashLifecycle, KvIndexerMetrics, MatchDetails, SyncIndexer, WorkerTask,
};
use crate::protocols::{
    KvCacheEventError, LocalBlockHash, OverlapScores, RouterEvent, WorkerWithDpRank,
};

mod arena;
mod dump;
mod lookup;
mod pool;
pub(crate) mod protocol;
mod rank_map;
mod runs;
mod slots;
pub(crate) mod sync;
mod write;

#[cfg(any(test, feature = "bench"))]
mod check;
#[cfg(any(test, feature = "bench"))]
pub use check::{MemoryReport, ShapeReport};

#[cfg(test)]
mod harness_impl;
#[cfg(test)]
mod tests;

pub use runs::{ReaderMode, StatsSnapshot};

/// Knobs and ablation switches (spec B 15).
#[derive(Clone, Copy, Debug)]
pub struct ArenaConfig {
    /// Lanes steal ready ranks from busy lanes; off is the `B-nosteal` ablation.
    pub steal: bool,
    /// A pumping lane applies an idle rank's task directly; off is `B-nofast`.
    pub inline_fast_path: bool,
    /// Optimistic or locked reads; `Locked` is `B-lockread`.
    pub reader: ReaderMode,
    /// Failed optimistic attempts before a read takes the run lock.
    pub read_retries: u32,
    /// Tasks a lane applies from one rank before it releases the rank.
    pub batch: usize,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            steal: true,
            inline_fast_path: true,
            reader: ReaderMode::Optimistic,
            read_retries: runs::READ_RETRIES,
            batch: 32,
        }
    }
}

/// The branch-B backend. Host it in a `ThreadPoolIndexer`.
pub struct ArenaIndex {
    runs: runs::Runs,
    slots: slots::SlotRegistry,
    pool: pool::Pool,
    config: ArenaConfig,
    cleanup: CleanupState,
    lifecycle: HashLifecycle,
}

impl Default for ArenaIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl ArenaIndex {
    pub fn new() -> Self {
        Self::with_config(ArenaConfig::default())
    }

    pub fn with_config(config: ArenaConfig) -> Self {
        Self {
            runs: runs::Runs::new(config.reader, config.read_retries),
            slots: slots::SlotRegistry::default(),
            pool: pool::Pool::default(),
            config: ArenaConfig {
                batch: config.batch.max(1),
                ..config
            },
            cleanup: CleanupState::new(),
            lifecycle: HashLifecycle::default(),
        }
    }

    pub fn new_with_delegate(delegate: Arc<dyn crate::indexer::KvIndexerDelegate>) -> Self {
        let mut index = Self::new();
        index.lifecycle = HashLifecycle::new(delegate);
        index
    }

    pub fn config(&self) -> ArenaConfig {
        self.config
    }

    /// Rare-event counters plus the pool's steal and fast-path counts.
    pub fn stats(&self) -> StatsSnapshot {
        let mut stats = self.runs.stats.snapshot();
        let (steals, inline) = self.pool.steals();
        stats.steals = steals;
        stats.inline_fast_path = inline;
        stats
    }

    /// Mailbox backlog of each lane's ranks when it pumped `SealObservation`. Mailboxes
    /// hide backlog from the channel length a bench reports, so add these to it.
    pub fn backlog_at_seal(&self) -> Vec<u64> {
        self.pool.backlog_at_seal()
    }

    /// Tasks fed to `lane`'s home ranks and not yet applied.
    pub fn pending_events(&self, lane: usize) -> u64 {
        self.pool.pending_events(lane)
    }

    /// Applies one event for its rank on the calling thread, outside any lane. The rank's
    /// cell is claimed like a lane would, so this is safe next to running lanes as long as
    /// no lane is fed the same rank.
    pub fn apply_event_inline(&self, event: RouterEvent) -> Result<(), KvCacheEventError> {
        let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let cell = self
            .pool
            .cells
            .entry(rank)
            .or_insert_with(|| Arc::new(pool::RankCell::new_detached(rank)))
            .clone();
        let mut data = cell.data.lock();
        let result = self.apply_event(&mut data, event, None);
        cell.blocks
            .store(data.map.len(), std::sync::atomic::Ordering::Relaxed);
        result
    }

    /// Removes a rank on the calling thread, as `apply_event_inline` applies events.
    pub fn remove_rank_inline(&self, rank: WorkerWithDpRank) {
        let Some(cell) = self.pool.cells.get(&rank).map(|cell| cell.clone()) else {
            return;
        };
        let mut data = cell.data.lock();
        self.remove_rank(&mut data);
        cell.blocks.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// The slot `rank` is mapped to, if any.
    pub fn rank_slot(&self, rank: WorkerWithDpRank) -> Option<usize> {
        let guard = crossbeam_epoch::pin();
        self.slots
            .table(&guard)
            .slot_of(rank)
            .map(|slot| slot.index())
    }
}

impl SyncIndexer for ArenaIndex {
    fn worker(
        &self,
        event_receiver: flume::Receiver<WorkerTask>,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> anyhow::Result<()> {
        self.run_lane(event_receiver, metrics)
    }

    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores {
        self.find_matches_impl(sequence, early_exit)
    }

    fn supports_routing_decision_pruning(&self) -> bool {
        false
    }

    fn apply_anchor(
        &self,
        _worker: WorkerWithDpRank,
        _anchor: AnchorTask,
    ) -> Result<(), KvCacheEventError> {
        Err(KvCacheEventError::InvalidBlockSequence)
    }

    fn try_schedule_cleanup(&self) -> bool {
        self.cleanup.try_schedule()
    }

    fn cancel_scheduled_cleanup(&self) {
        self.cleanup.cancel();
    }

    fn run_cleanup_task(&self) {
        let mut guard = CleanupGuard::new(&self.cleanup);
        let found = self.sweep_holderless();
        self.runs
            .stats
            .cleanup_unlinks
            .fetch_add(found, std::sync::atomic::Ordering::Relaxed);
        guard.mark_completed();
    }

    fn dump_events(&self) -> Option<Vec<RouterEvent>> {
        Some(self.dump_tree_as_events())
    }

    fn timing_report(&self) -> String {
        let stats = self.stats();
        let backlog = self.backlog_at_seal();
        format!(
            "ArenaIndex (branch B) counters:\n  steals = {}\n  inline fast path = {}\n  \
             reader retries = {}\n  reader fallbacks = {}\n  splits = {}\n  unlinks = {}\n  \
             store restarts = {}\n  replans = {}\n  claim check failures = {}\n  \
             h1 violations = {}\n  ext conflicts = {}\n  mailbox backlog at seal = {:?} (total {})",
            stats.steals,
            stats.inline_fast_path,
            stats.reader_retries,
            stats.reader_fallbacks,
            stats.splits_prefix_cap,
            stats.unlinks,
            stats.store_restarts,
            stats.replans,
            stats.claim_check_failures,
            stats.h1_violations,
            stats.ext_conflicts,
            backlog,
            backlog.iter().sum::<u64>(),
        )
    }

    fn node_count(&self) -> usize {
        self.runs.slab.issued() as usize - self.runs.slab.free_len()
    }
}

impl ArenaIndex {
    /// Scores plus last matched hashes, for callers that hold the concrete type.
    pub fn find_match_details_impl(
        &self,
        sequence: &[LocalBlockHash],
        early_exit: bool,
    ) -> MatchDetails {
        self.find_match_details(sequence, early_exit, false)
    }
}
