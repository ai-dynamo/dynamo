// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reclamation in three layers, cheapest first:
//!
//! 1. Eager unlink, by the writer that emptied a run: under the parent's shared gate it
//!    try-locks the run, re-checks that it is holder-less, childless and alive, marks it
//!    `DEAD`, tombstones the parent's slot by key and defers the frees; then it repeats
//!    on the parent.
//! 2. Pending retries: a failed try-lock or a moved parent queues the run on its lane,
//!    retried in chunks while the lane is idle.
//! 3. A volume sweep, scheduled when the lanes' tallies say enough dead blocks are still
//!    linked (or by CRTC's five-minute timer), which walks the tree and unlinks every
//!    holder-less leaf deepest-first.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use super::arena::FreeBatch;
use super::table::{child_key, pack};
use super::*;
use crate::cleanup::CleanupGuard;

/// Pending entries retried per idle chunk.
const PENDING_CHUNK: usize = 256;
/// Blocks of change a lane tallies before it publishes them.
const TALLY_FLUSH: u64 = 1024;
/// Dead linked blocks below which a volume sweep never triggers.
const DEAD_FLOOR: i64 = 65_536;
const MIN_SWEEP_GAP_MS: u64 = 100;

/// A run whose eager unlink could not run.
#[derive(Clone, Copy, Debug)]
pub(super) struct PendingUnlink {
    run: u32,
    generation: u32,
    blocks: u32,
}

/// A lane's unpublished changes to the shared reclaim totals.
#[derive(Default)]
pub(super) struct ReclaimTally {
    linked: i64,
    dead: i64,
    changes: u64,
}

impl ReclaimTally {
    pub(super) fn linked(&mut self, blocks: usize) {
        self.linked += blocks as i64;
        self.changes += blocks as u64;
    }

    fn unlinked(&mut self, blocks: u32) {
        self.linked -= i64::from(blocks);
        self.changes += u64::from(blocks);
    }

    fn dead(&mut self, blocks: i64) {
        self.dead += blocks;
        self.changes += blocks.unsigned_abs();
    }

    pub(super) fn maybe_flush(&mut self, shared: &ReclaimShared) {
        if self.changes >= TALLY_FLUSH {
            self.flush(shared);
        }
    }

    pub(super) fn flush(&mut self, shared: &ReclaimShared) {
        if self.linked != 0 {
            shared.linked.fetch_add(self.linked, Ordering::Relaxed);
        }
        if self.dead != 0 {
            shared.dead.fetch_add(self.dead, Ordering::Relaxed);
        }
        *self = Self::default();
    }
}

/// Estimates of linked and dead (holder-less, still linked) blocks, plus the volume
/// sweep's schedule.
pub(super) struct ReclaimShared {
    linked: AtomicI64,
    dead: AtomicI64,
    scheduled: AtomicBool,
    origin: Instant,
    last_sweep_end_ms: AtomicU64,
    last_sweep_ms: AtomicU64,
}

impl Default for ReclaimShared {
    fn default() -> Self {
        Self {
            linked: AtomicI64::new(0),
            dead: AtomicI64::new(0),
            scheduled: AtomicBool::new(false),
            origin: Instant::now(),
            last_sweep_end_ms: AtomicU64::new(0),
            last_sweep_ms: AtomicU64::new(0),
        }
    }
}

impl ReclaimShared {
    fn elapsed_ms(&self) -> u64 {
        self.origin.elapsed().as_millis() as u64
    }

    /// Whether dead blocks justify a sweep now: at least an eighth of the linked blocks
    /// and [`DEAD_FLOOR`], at most once per `max(100 ms, 10 x the last sweep)`.
    fn wants_volume_sweep(&self) -> bool {
        let dead = self.dead.load(Ordering::Relaxed);
        let linked = self.linked.load(Ordering::Relaxed);
        if dead < (linked / 8).max(DEAD_FLOOR) {
            return false;
        }
        let gap = (10 * self.last_sweep_ms.load(Ordering::Relaxed)).max(MIN_SWEEP_GAP_MS);
        self.elapsed_ms()
            .saturating_sub(self.last_sweep_end_ms.load(Ordering::Relaxed))
            >= gap
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn dead_blocks(&self) -> i64 {
        self.dead.load(Ordering::Relaxed)
    }

    #[cfg(any(test, feature = "bench"))]
    pub(super) fn linked_blocks(&self) -> i64 {
        self.linked.load(Ordering::Relaxed)
    }
}

impl ArenaIndexC {
    /// Unlinks `run_id` if it is still incarnation `generation` and holder-less and
    /// childless, then repeats on its parent. Queues it on `pending` when the child
    /// try-lock fails or the parent moved. Callers are pinned and hold no run lock.
    pub(super) fn try_unlink(
        &self,
        mut run_id: u32,
        mut generation: u32,
        free: &mut FreeBatch,
        pending: &mut Vec<PendingUnlink>,
        tally: &mut ReclaimTally,
    ) {
        let store = &self.store;
        loop {
            if run_id == ROOT {
                return;
            }
            let run = store.run(run_id);
            if run.generation.load(Ordering::Relaxed) != generation || run.is_dead() {
                return;
            }
            let parent_id = run.parent.load(Ordering::Acquire);
            let parent = store.run(parent_id);
            let parent_gate = parent.gate.read();
            // A split of the parent may have moved the run to a suffix.
            if run.parent.load(Ordering::Acquire) != parent_id || parent.is_dead() {
                drop(parent_gate);
                self.defer_unlink(run_id, generation, pending, tally);
                return;
            }
            let Some(gate) = self.child_try_write(run) else {
                drop(parent_gate);
                self.defer_unlink(run_id, generation, pending, tally);
                return;
            };
            let state = run.state.write();
            if run.is_dead()
                || run.generation.load(Ordering::Relaxed) != generation
                || store.has_holders(run)
                || store.has_live_children(run)
            {
                return;
            }
            let offset = run.start.load(Ordering::Relaxed) - parent.start.load(Ordering::Relaxed);
            let key = child_key(offset, store.window(run).local(0));
            let table = parent.children.load(Ordering::Acquire);
            if table == NONE || !store.table(table).unlink(key, pack(run_id, generation)) {
                drop((state, gate, parent_gate));
                self.defer_unlink(run_id, generation, pending, tally);
                return;
            }
            let blocks = run.len();
            store.kill_run(run_id, free, &self.store);
            self.counters.unlinks.fetch_add(1, Ordering::Relaxed);
            tally.unlinked(blocks);
            drop((state, gate, parent_gate));

            if parent_id == ROOT || store.has_holders(parent) || store.has_live_children(parent) {
                return;
            }
            run_id = parent_id;
            generation = parent.generation.load(Ordering::Relaxed);
        }
    }

    fn child_try_write<'a>(
        &self,
        run: &'a run::RunHeader,
    ) -> Option<parking_lot::RwLockWriteGuard<'a, ()>> {
        #[cfg(test)]
        if self.hooks.take_try_lock_failure() {
            return None;
        }
        run.gate.try_write()
    }

    fn defer_unlink(
        &self,
        run: u32,
        generation: u32,
        pending: &mut Vec<PendingUnlink>,
        tally: &mut ReclaimTally,
    ) {
        if pending
            .iter()
            .any(|entry| entry.run == run && entry.generation == generation)
        {
            return;
        }
        let blocks = self.store.run(run).len();
        pending.push(PendingUnlink {
            run,
            generation,
            blocks,
        });
        tally.dead(i64::from(blocks));
        self.counters
            .pending_unlinks
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Retries up to [`PENDING_CHUNK`] pending unlinks. Entries that fail again are
    /// queued again; the rest are dropped.
    pub(super) fn retry_pending(&self, lane: &mut CLane) -> bool {
        if lane.pending.is_empty() {
            return false;
        }
        let take = lane.pending.len().min(PENDING_CHUNK);
        let batch: Vec<PendingUnlink> = lane.pending.drain(..take).collect();
        let _guard = crossbeam_epoch::pin();
        for entry in batch {
            self.counters
                .pending_unlinks
                .fetch_sub(1, Ordering::Relaxed);
            lane.tally.dead(-i64::from(entry.blocks));
            self.try_unlink(
                entry.run,
                entry.generation,
                &mut lane.free,
                &mut lane.pending,
                &mut lane.tally,
            );
        }
        !lane.pending.is_empty()
    }

    /// Drops every pending unlink of `lane`; a later sweep finds those runs again.
    pub(super) fn forget_pending(&self, lane: &mut CLane) {
        let count = lane.pending.len() as i64;
        let blocks: i64 = lane
            .pending
            .iter()
            .map(|entry| i64::from(entry.blocks))
            .sum();
        lane.pending.clear();
        lane.tally.dead(-blocks);
        self.counters
            .pending_unlinks
            .fetch_sub(count, Ordering::Relaxed);
    }

    /// The volume sweep: walks from ROOT, records holder-less runs, and unlinks them
    /// deepest-first (each only if also childless). Replaces the tallied estimates with
    /// the totals the walk saw.
    pub(super) fn volume_sweep(&self, lane: &mut CLane) {
        let started = Instant::now();
        let store = &self.store;
        let mut guard = crossbeam_epoch::pin();
        let mut queue: Vec<(u32, u32, u32)> = store
            .children_of(store.run(ROOT))
            .into_iter()
            .map(|(_, child, generation)| (child, generation, 1))
            .collect();
        let mut holderless: Vec<(u32, u32, u32, u32)> = Vec::new();
        let mut linked = 0i64;
        let mut visited = 0usize;
        while let Some((run_id, generation, depth)) = queue.pop() {
            visited += 1;
            if visited.is_multiple_of(64) {
                guard.repin();
            }
            let run = store.run(run_id);
            let _state = run.state.read();
            if run.is_dead() || run.generation.load(Ordering::Relaxed) != generation {
                continue;
            }
            linked += i64::from(run.len());
            if !store.has_holders(run) {
                holderless.push((depth, run_id, generation, run.len()));
            }
            queue.extend(
                store
                    .children_of(run)
                    .into_iter()
                    .map(|(_, child, generation)| (child, generation, depth + 1)),
            );
        }
        holderless.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
        let unlinks_before = self.counters.unlinks.load(Ordering::Relaxed);
        for &(_, run_id, generation, _) in &holderless {
            self.try_unlink(
                run_id,
                generation,
                &mut lane.free,
                &mut lane.pending,
                &mut lane.tally,
            );
        }
        drop(guard);
        let dead: i64 = holderless
            .iter()
            .filter(|&&(_, run_id, generation, _)| {
                let run = store.run(run_id);
                !run.is_dead() && run.generation.load(Ordering::Relaxed) == generation
            })
            .map(|&(_, _, _, blocks)| i64::from(blocks))
            .sum();
        let unlinked = self.counters.unlinks.load(Ordering::Relaxed) - unlinks_before;
        lane.tally = ReclaimTally::default();
        self.reclaim.linked.store(linked, Ordering::Relaxed);
        self.reclaim.dead.store(dead, Ordering::Relaxed);
        self.counters.volume_sweeps.fetch_add(1, Ordering::Relaxed);
        let elapsed = started.elapsed().as_millis() as u64;
        self.reclaim.last_sweep_ms.store(elapsed, Ordering::Relaxed);
        self.reclaim
            .last_sweep_end_ms
            .store(self.reclaim.elapsed_ms(), Ordering::Relaxed);
        tracing::debug!(
            visited,
            holderless = holderless.len(),
            unlinked,
            elapsed_ms = elapsed,
            "arena-c volume sweep"
        );
    }

    pub(super) fn schedule_cleanup(&self) -> bool {
        if self.reclaim.wants_volume_sweep()
            && self
                .reclaim
                .scheduled
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
        {
            return true;
        }
        self.cleanup.try_schedule()
    }

    pub(super) fn cancel_cleanup(&self) {
        self.reclaim.scheduled.store(false, Ordering::Release);
        self.cleanup.cancel();
    }

    /// Runs a scheduled sweep on `lane`.
    pub(super) fn run_cleanup_on(&self, lane: &mut CLane) {
        let mut guard = CleanupGuard::new(&self.cleanup);
        self.volume_sweep(lane);
        guard.mark_completed();
        self.reclaim.scheduled.store(false, Ordering::Release);
        self.flush_frees(lane);
    }
}
