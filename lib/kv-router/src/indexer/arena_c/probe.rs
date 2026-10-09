// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test and bench hooks: shape and memory reports, quiescence, the structural checker
//! (S1 to S10 plus S4 both ways), and the C-only fault hooks.

use std::fmt;

#[cfg(test)]
use rustc_hash::{FxHashMap, FxHashSet};

use super::*;
#[cfg(test)]
use super::{
    run::{DEAD, PARTIAL_CAP, SEALED},
    store::Resolved,
    table::{child_key, claim_budget},
    types::BlockPos,
};

#[cfg(test)]
#[derive(Default)]
pub(super) struct Hooks {
    fail_try_locks: AtomicUsize,
    skip_path_compression: std::sync::atomic::AtomicBool,
}

#[cfg(test)]
impl Hooks {
    pub(super) fn take_try_lock_failure(&self) -> bool {
        self.fail_try_locks
            .fetch_update(Ordering::AcqRel, Ordering::Relaxed, |n| n.checked_sub(1))
            .is_ok()
    }

    pub(super) fn skip_path_compression(&self) -> bool {
        self.skip_path_compression.load(Ordering::Relaxed)
    }
}

/// Run and table counts from one walk of the tree, plus the event counters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShapeReport {
    pub runs_live: u64,
    pub runs_allocated: u64,
    pub runs_free: u64,
    pub linked_blocks: u64,
    pub dead_linked_blocks: u64,
    pub splits_prefix_cap: u64,
    pub unlinks: u64,
    pub child_tables: u64,
    pub child_slots: u64,
    pub child_tombstones: u64,
    pub max_child_probe: u64,
    pub h1_violations: u64,
    pub ext_conflicts: u64,
    pub max_forward_hops: u64,
    pub plan_retries: u64,
    pub store_failures: u64,
    pub pending_unlinks: i64,
    pub volume_sweeps: u64,
    /// The lanes' published estimates that schedule volume sweeps.
    pub tallied_linked_blocks: i64,
    pub tallied_dead_blocks: i64,
    pub arena_reserved_bytes: u64,
    pub arena_used_bytes: u64,
    pub arena_free_bytes: u64,
    pub arena_stranded_bytes: u64,
    pub slab_reserved_bytes: u64,
    pub epoch_pending_bytes: u64,
    // Fields that only branch B's seqlock and lane pool fill.
    pub reader_retries: u64,
    pub reader_fallbacks: u64,
    pub poisoned_runs: u64,
    pub steals: u64,
    pub inline_fast_path: u64,
    pub claim_check_failures: u64,
}

impl fmt::Display for ShapeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "arena-c shape:")?;
        writeln!(
            f,
            "  runs live/allocated/free = {}/{}/{}",
            self.runs_live, self.runs_allocated, self.runs_free
        )?;
        writeln!(
            f,
            "  linked blocks = {} (dead {})",
            self.linked_blocks, self.dead_linked_blocks
        )?;
        writeln!(
            f,
            "  prefix-cap splits = {}, unlinks = {}, pending unlinks = {}, volume sweeps = {}",
            self.splits_prefix_cap, self.unlinks, self.pending_unlinks, self.volume_sweeps
        )?;
        writeln!(
            f,
            "  child tables = {} ({} slots, {} tombstones, max probe {})",
            self.child_tables, self.child_slots, self.child_tombstones, self.max_child_probe
        )?;
        writeln!(
            f,
            "  plan retries = {}, store failures = {}, ext mismatches = {}, ext conflicts = {}, max forward hops = {}",
            self.plan_retries,
            self.store_failures,
            self.h1_violations,
            self.ext_conflicts,
            self.max_forward_hops
        )?;
        writeln!(
            f,
            "  tallied linked/dead blocks = {}/{}",
            self.tallied_linked_blocks, self.tallied_dead_blocks
        )?;
        write!(
            f,
            "  arena reserved/used/free/stranded = {}/{}/{}/{} B, slab = {} B, epoch pending = {} B",
            self.arena_reserved_bytes,
            self.arena_used_bytes,
            self.arena_free_bytes,
            self.arena_stranded_bytes,
            self.slab_reserved_bytes,
            self.epoch_pending_bytes
        )
    }
}

/// Bytes by component. `epoch_pending_bytes` is deferred memory not yet reusable.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub struct MemoryReport {
    pub array_bytes_live: u64,
    pub array_bytes_slack: u64,
    pub arena_used_bytes: u64,
    pub arena_free_bytes: u64,
    pub arena_reserved_bytes: u64,
    pub arena_stranded_bytes: u64,
    pub header_bytes: u64,
    pub slab_reserved_bytes: u64,
    pub child_table_bytes: u64,
    pub cutoff_table_bytes: u64,
    pub forward_bytes: u64,
    pub overflow_bytes: u64,
    pub map_bytes: u64,
    pub memberships: u64,
    pub distinct_blocks: u64,
    pub epoch_pending_bytes: u64,
}

impl ArenaIndexC {
    #[cfg(test)]
    pub(super) fn probe_fail_try_locks(&self, n: usize) {
        self.hooks.fail_try_locks.store(n, Ordering::Release);
    }

    #[cfg(test)]
    pub(super) fn probe_skip_path_compression(&self, skip: bool) {
        self.hooks
            .skip_path_compression
            .store(skip, Ordering::Relaxed);
    }

    /// Holds a pin, so nothing released meanwhile is reused until the guard drops.
    #[cfg(test)]
    pub(super) fn probe_hold_pin(&self) -> crossbeam_epoch::Guard {
        crossbeam_epoch::pin()
    }

    pub fn shape_report(&self) -> ShapeReport {
        let store = &self.store;
        let mut report = ShapeReport {
            runs_allocated: u64::from(store.slab.allocated()),
            runs_free: store.slab.free_count() as u64,
            splits_prefix_cap: self.counters.splits.load(Ordering::Relaxed),
            unlinks: self.counters.unlinks.load(Ordering::Relaxed),
            h1_violations: self.counters.h1_violations.load(Ordering::Relaxed),
            ext_conflicts: self.counters.ext_conflicts.load(Ordering::Relaxed),
            max_forward_hops: self.counters.max_forward_hops.load(Ordering::Relaxed),
            plan_retries: self.counters.plan_retries.load(Ordering::Relaxed),
            store_failures: self.counters.store_failures.load(Ordering::Relaxed),
            pending_unlinks: self.counters.pending_unlinks.load(Ordering::Relaxed),
            volume_sweeps: self.counters.volume_sweeps.load(Ordering::Relaxed),
            tallied_linked_blocks: self.reclaim.linked_blocks(),
            tallied_dead_blocks: self.reclaim.dead_blocks(),
            arena_reserved_bytes: store.arena.reserved_bytes() as u64,
            arena_used_bytes: store.arena.used_bytes() as u64,
            arena_free_bytes: store.arena.free_bytes() as u64,
            arena_stranded_bytes: store.arena.stranded_bytes() as u64,
            slab_reserved_bytes: store.slab.reserved_bytes() as u64,
            epoch_pending_bytes: store.epoch_pending_bytes() as u64,
            ..ShapeReport::default()
        };
        let _guard = crossbeam_epoch::pin();
        let mut tables = vec![store.run(ROOT).children.load(Ordering::Acquire)];
        for id in self.reachable_runs() {
            let run = store.run(id);
            let _state = run.state.read();
            if run.is_dead() {
                continue;
            }
            report.runs_live += 1;
            report.linked_blocks += u64::from(run.len());
            if !store.has_holders(run) {
                report.dead_linked_blocks += u64::from(run.len());
            }
            tables.push(run.children.load(Ordering::Relaxed));
        }
        for table in tables.into_iter().filter(|&t| t != NONE) {
            let table = store.table(table);
            report.child_tables += 1;
            report.child_slots += u64::from(table.slots());
            report.child_tombstones += table.tombstones() as u64;
            report.max_child_probe = report.max_child_probe.max(table.max_probe() as u64);
        }
        report
    }

    /// Bytes by component. Lane maps are counted from `lanes`.
    #[cfg(test)]
    pub(super) fn memory_report_for(&self, lanes: &[&CLane]) -> MemoryReport {
        let store = &self.store;
        let mut report = MemoryReport {
            arena_used_bytes: store.arena.used_bytes() as u64,
            arena_free_bytes: store.arena.free_bytes() as u64,
            arena_reserved_bytes: store.arena.reserved_bytes() as u64,
            arena_stranded_bytes: store.arena.stranded_bytes() as u64,
            slab_reserved_bytes: store.slab.reserved_bytes() as u64,
            epoch_pending_bytes: store.epoch_pending_bytes() as u64,
            ..MemoryReport::default()
        };
        let header = std::mem::size_of::<run::RunHeader>() as u64;
        let mut arrays: FxHashSet<u32> = FxHashSet::default();
        let _guard = crossbeam_epoch::pin();
        let ids = self.reachable_runs();
        report.header_bytes = header * (ids.len() as u64 + 1);
        for id in ids {
            let run = store.run(id);
            let _state = run.state.read();
            let array = run.array.load(Ordering::Relaxed);
            if array != NONE && arrays.insert(array) {
                let (used, capacity) = store.array_extent(array);
                report.array_bytes_live += 16 + 16 * u64::from(used);
                report.array_bytes_slack += 16 * u64::from(capacity - used);
                report.distinct_blocks += u64::from(used);
            }
            let children = run.children.load(Ordering::Relaxed);
            if children != NONE {
                report.child_table_bytes += 8 * store.table(children).slots() as u64 * 2 + 16;
            }
            report.cutoff_table_bytes += 8 * (1 + store.cutoff_count(run) as u64);
            report.forward_bytes += 16 * store.forwards(run).len() as u64;
            report.overflow_bytes += 40 * store.chunks(run).count() as u64;
            let len = u64::from(run.len());
            let mut members = 0u64;
            store.for_each_whole(run, |_| members += len);
            for (_, cutoff) in store.cutoff_entries(run) {
                members += u64::from(cutoff);
            }
            report.memberships += members;
        }
        for lane in lanes {
            for lookup in lane.ranks.values() {
                report.map_bytes += 16 * lookup.map.capacity() as u64;
            }
        }
        report
    }

    /// Finishes deferred work: pending unlinks, frees, and the epoch, so every deferred
    /// release has run when this returns.
    #[cfg(test)]
    pub(super) fn probe_quiesce(&self, lanes: &mut [&mut CLane]) {
        for lane in lanes.iter_mut() {
            let mut rounds = 0;
            while self.retry_pending(lane) && rounds < 1024 {
                rounds += 1;
            }
            self.flush_frees(lane);
            lane.tally.flush(&self.reclaim);
        }
        for _ in 0..4 {
            super::slots::wait_for_pinned_threads();
        }
        for _ in 0..128 {
            crossbeam_epoch::pin().flush();
        }
    }

    /// Checks S1 to S10 (C's versions) at quiescence. `lanes` must hold every rank's
    /// state. Returns the first violation.
    #[cfg(test)]
    pub(super) fn probe_check(&self, lanes: &[&CLane]) -> Result<(), String> {
        let store = &self.store;
        let guard = crossbeam_epoch::pin();
        let table = self.slots.table(&guard);
        let mut seen: FxHashSet<u32> = FxHashSet::default();
        let mut reachable_blocks: FxHashSet<u32> = FxHashSet::default();
        let mut array_refs: FxHashMap<u32, u32> = FxHashMap::default();
        // (rank, ext) -> (run, offset) for every held position.
        let mut held_positions: FxHashMap<(WorkerWithDpRank, u64), BlockPos> = FxHashMap::default();
        let root = store.run(ROOT);
        let mut queue: Vec<(u32, u32)> = Vec::new();
        let check_table = |parent_id: u32,
                           parent: &run::RunHeader,
                           queue: &mut Vec<(u32, u32)>,
                           reachable_blocks: &mut FxHashSet<u32>|
         -> Result<(), String> {
            let addr = parent.children.load(Ordering::Acquire);
            if addr == NONE {
                return Ok(());
            }
            reachable_blocks.insert(addr);
            let t = store.table(addr);
            if t.used() > claim_budget(t.slots()) {
                return Err(format!(
                    "S6: run {parent_id} table used {} of {}",
                    t.used(),
                    t.slots()
                ));
            }
            let entries: Vec<(u64, u64)> = t.entries().collect();
            if entries.len() as u32 != t.live() {
                return Err(format!(
                    "S6: run {parent_id} table live {} but {} published",
                    t.live(),
                    entries.len()
                ));
            }
            let mut keys = FxHashSet::default();
            let parent_window = store.window(parent);
            let parent_start = parent.start.load(Ordering::Relaxed);
            for (key, value) in entries {
                if !keys.insert(key) {
                    return Err(format!(
                        "S2: run {parent_id} has two live children under one key"
                    ));
                }
                let child_id = table::unpack_run(value);
                let child = store.run(child_id);
                if child.is_dead() {
                    return Err(format!("S10: run {parent_id} links dead child {child_id}"));
                }
                if child.generation.load(Ordering::Relaxed) != table::unpack_generation(value) {
                    return Err(format!("S1: child {child_id} generation mismatch"));
                }
                if child.parent.load(Ordering::Relaxed) != parent_id {
                    return Err(format!(
                        "S1: child {child_id} names parent {} but hangs under {parent_id}",
                        child.parent.load(Ordering::Relaxed)
                    ));
                }
                let offset = child
                    .start
                    .load(Ordering::Relaxed)
                    .wrapping_sub(parent_start);
                let valid_offset = if parent_id == ROOT {
                    offset == 0
                } else {
                    offset >= 1 && offset <= parent.len()
                };
                if !valid_offset {
                    return Err(format!(
                        "S1: child {child_id} at bad offset {offset} of {parent_id}"
                    ));
                }
                let head = store.window(child).local(0);
                if key != child_key(offset, head) {
                    return Err(format!("S1: child {child_id} under the wrong key"));
                }
                if (offset as usize) < parent_window.len()
                    && parent_window.local(offset as usize) == head
                {
                    return Err(format!(
                        "S2: child {child_id} repeats its parent {parent_id}'s own position {offset}"
                    ));
                }
                queue.push((child_id, parent_id));
            }
            Ok(())
        };
        check_table(ROOT, root, &mut queue, &mut reachable_blocks)?;
        while let Some((id, _parent)) = queue.pop() {
            if !seen.insert(id) {
                return Err(format!("S1: run {id} reachable twice"));
            }
            let run = store.run(id);
            let len = run.len();
            if len == 0 {
                return Err(format!("run {id} is empty"));
            }
            if run.flags.load(Ordering::Relaxed) & DEAD != 0 {
                return Err(format!("S10: dead run {id} reachable"));
            }
            // S7 and S5 bookkeeping.
            let array = run.array.load(Ordering::Relaxed);
            let (used, capacity) = store.array_extent(array);
            let base = run.base.load(Ordering::Relaxed);
            if used > capacity || base + len > used {
                return Err(format!(
                    "S7: run {id} window {base}+{len} outside used {used}/{capacity}"
                ));
            }
            *array_refs.entry(array).or_default() += 1;
            reachable_blocks.insert(array);
            // S3.
            let cutoffs: Vec<(Slot, u32)> = store.cutoff_entries(run).collect();
            if cutoffs.len() > PARTIAL_CAP {
                return Err(format!(
                    "S3: run {id} has {} partial holders",
                    cutoffs.len()
                ));
            }
            let mut cut_slots = FxHashSet::default();
            for &(slot, cutoff) in &cutoffs {
                if cutoff == 0 || cutoff >= len {
                    return Err(format!("S3: run {id} cutoff {cutoff} of {len}"));
                }
                if store.whole_contains(run, slot) {
                    return Err(format!(
                        "S3: slot {} whole and partial on {id}",
                        slot.index()
                    ));
                }
                if !cut_slots.insert(slot) {
                    return Err(format!("S3: slot {} has two cutoffs on {id}", slot.index()));
                }
            }
            if run.cutoffs.load(Ordering::Relaxed) != NONE {
                reachable_blocks.insert(run.cutoffs.load(Ordering::Relaxed));
            }
            // S8.
            let forwards = store.forwards(run);
            if !forwards.is_empty() {
                reachable_blocks.insert(run.forwards.load(Ordering::Relaxed));
                if run.flags.load(Ordering::Relaxed) & SEALED == 0 {
                    return Err(format!("S8: run {id} has forwards but is not sealed"));
                }
                if forwards.windows(2).any(|pair| pair[0].at <= pair[1].at) {
                    return Err(format!("S8: run {id} forwards not strictly decreasing"));
                }
                if forwards.last().is_some_and(|last| last.at != len) {
                    return Err(format!("S8: run {id} grew after its last split"));
                }
            }
            for (_, chunk) in store.chunks(run) {
                reachable_blocks.insert(chunk);
            }
            // S9.
            if !store.has_holders(run) && !store.has_live_children(run) {
                return Err(format!("S9: run {id} is holder-less and childless"));
            }
            // Held positions for S4.
            let window = store.window(run);
            let mut record = |slot: Slot, held: u32| -> Result<(), String> {
                let Some(rank) = table.owner(slot) else {
                    return Err(format!("slot {} on run {id} has no owner", slot.index()));
                };
                for p in 0..held {
                    held_positions.insert((rank, window.ext(p as usize)), BlockPos::new(id, p));
                }
                Ok(())
            };
            let mut failure = None;
            store.for_each_whole(run, |slot| {
                if let Err(error) = record(slot, len) {
                    failure.get_or_insert(error);
                }
            });
            if let Some(error) = failure {
                return Err(error);
            }
            for &(slot, cutoff) in &cutoffs {
                record(slot, cutoff)?;
            }
            check_table(id, run, &mut queue, &mut reachable_blocks)?;
        }

        // S5: array references and free lists.
        for (&array, &refs) in &array_refs {
            let actual = store.array_refs(array);
            if actual != refs {
                return Err(format!(
                    "S5: array {array} has refs {actual} for {refs} windows"
                ));
            }
        }
        for id in store.slab.free_ids() {
            if seen.contains(&id) {
                return Err(format!("S5: free-listed run {id} is reachable"));
            }
        }
        for block in store.arena.free_blocks() {
            if reachable_blocks.contains(&block.addr) {
                return Err(format!("S5: free-listed block {} is reachable", block.addr));
            }
        }

        // S4 forward: every entry resolves to a position its rank holds.
        let mut entries = 0usize;
        for lane in lanes {
            for (&rank, lookup) in &lane.ranks {
                let slot = table.slot_of(rank);
                for (key, entry) in lookup.map.iter() {
                    entries += 1;
                    match self.resolve(key, entry, slot) {
                        Resolved::At { pos, held } if pos.offset < held => {
                            if held_positions.get(&(rank, key.0)) != Some(&pos) {
                                return Err(format!(
                                    "S4: {rank:?} entry for {key:?} resolves to {pos:?}, not the held position"
                                ));
                            }
                        }
                        Resolved::At { pos, held } => {
                            return Err(format!(
                                "S4: {rank:?} entry for {key:?} resolves to {pos:?} which it holds only {held} of"
                            ));
                        }
                        Resolved::Stale => {
                            return Err(format!("S4: {rank:?} entry for {key:?} is stale"));
                        }
                    }
                }
            }
        }
        // S4 backward: every held position has an entry.
        let mut by_rank: FxHashMap<WorkerWithDpRank, &RankLookup> = FxHashMap::default();
        for lane in lanes {
            for (rank, lookup) in &lane.ranks {
                by_rank.insert(*rank, lookup);
            }
        }
        for &(rank, ext) in held_positions.keys() {
            let present = by_rank
                .get(&rank)
                .is_some_and(|lookup| lookup.map.contains_key(ExternalSequenceBlockHash(ext)));
            if !present {
                return Err(format!("S4: {rank:?} holds {ext:#x} without an entry"));
            }
        }
        if entries != held_positions.len() {
            return Err(format!(
                "S4: {entries} entries for {} held positions",
                held_positions.len()
            ));
        }
        Ok(())
    }
}
