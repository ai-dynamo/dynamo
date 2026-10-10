// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quiescent structural checks (spec B 11, S1 to S10, S4 both ways) and the memory and
//! shape reports (14). Test and bench builds only; every walk takes run locks.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::atomic::Ordering;

use rustc_hash::{FxHashMap, FxHashSet};

use super::ArenaIndex;
use super::arena::{self, Addr, Kind, NONE};
use super::protocol::{FLAG_POISONED, FLAG_SEALED};
use super::rank_map::Entry;
use super::runs::{Cutoff, NO_RUN, ROOT, ROOT_GEN, RunId, Snap, child_key};
use crate::protocols::{ExternalSequenceBlockHash, LocalBlockHash, WorkerWithDpRank};

/// Bytes by component (14). Arena memory is counted when reserved.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct MemoryReport {
    pub array_bytes_live: u64,
    pub array_bytes_slack: u64,
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
}

/// Tree shape and protocol counters (14).
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct ShapeReport {
    pub runs_live: u64,
    pub runs_allocated: u64,
    pub runs_free: u64,
    pub splits_prefix_cap: u64,
    pub unlinks: u64,
    pub child_tables: u64,
    pub child_slots: u64,
    pub child_tombstones: u64,
    pub max_child_probe: u64,
    pub h1_violations: u64,
    pub ext_conflicts: u64,
    pub reader_retries: u64,
    pub reader_fallbacks: u64,
    pub poisoned_runs: u64,
    pub store_restarts: u64,
    pub store_failures: u64,
    pub steals: u64,
    pub inline_fast_path: u64,
    pub claim_check_failures: u64,
    pub sweep_residue: u64,
}

/// One live run as the check sees it.
struct RunView {
    id: RunId,
    generation: u32,
    snap: Snap,
    local: Vec<LocalBlockHash>,
    ext: Vec<ExternalSequenceBlockHash>,
    whole: Vec<usize>,
    cuts: Vec<Cutoff>,
    children: Vec<(u64, RunId, u32)>,
    head: u64,
    parent: RunId,
}

impl ArenaIndex {
    /// Every live run reachable from the root, in BFS order, each read under its lock.
    fn live_runs(&self) -> Result<Vec<RunView>, String> {
        let mut out = Vec::new();
        let mut queue = VecDeque::from([(ROOT, ROOT_GEN)]);
        let mut seen = FxHashSet::default();
        while let Some((id, generation)) = queue.pop_front() {
            if !seen.insert(id) {
                return Err(format!("S1: run {id} is reachable twice"));
            }
            let h = self
                .runs
                .header(id)
                .ok_or(format!("run {id} has no header"))?;
            let Some(locked) = self.runs.lock(id, generation) else {
                return Err(format!(
                    "S1/S10: run {id} linked at generation {generation} is {} (version {}, flags {:#x})",
                    h.generation.load(Ordering::Acquire),
                    h.version.load(Ordering::Acquire),
                    h.flags.load(Ordering::Acquire),
                ));
            };
            if h.flags.load(Ordering::Relaxed) & FLAG_POISONED != 0 {
                return Err(format!("S10: run {id} is poisoned"));
            }
            let snap = locked.snap();
            let columns = self
                .runs
                .columns(&snap)
                .ok_or(format!("S7: run {id} has a window outside its array"))?;
            let view = RunView {
                id,
                generation,
                snap,
                local: (0..columns.len()).map(|i| columns.local(i)).collect(),
                ext: (0..columns.len()).map(|i| columns.ext(i)).collect(),
                whole: self.runs.whole_slots(&locked),
                cuts: self.runs.cutoffs(&snap),
                children: self.runs.children_of(snap.children),
                head: h.head.load(Ordering::Relaxed),
                parent: h.parent.load(Ordering::Relaxed),
            };
            queue.extend(
                view.children
                    .iter()
                    .map(|&(_, id, generation)| (id, generation)),
            );
            out.push(view);
        }
        Ok(out)
    }

    /// Checks S1 to S10, S4 in both directions, and that no run id or arena word leaks.
    /// Call only at quiescence.
    pub fn probe_check(&self) -> Result<(), String> {
        let runs = self.live_runs()?;
        let by_id: FxHashMap<RunId, usize> =
            runs.iter().enumerate().map(|(i, r)| (r.id, i)).collect();
        let guard = crossbeam_epoch::pin();
        let table = self.slots.table(&guard);
        let mut errors = String::new();
        let mut array_refs: FxHashMap<Addr, u64> = FxHashMap::default();
        let mut live_blocks: Vec<(Addr, u32)> = Vec::new();

        for run in &runs {
            let snap = &run.snap;
            if run.id != ROOT {
                // S7, and the array's reference count (S5).
                let header = self
                    .runs
                    .arena
                    .word(snap.array)
                    .map(|w| w.load(Ordering::Relaxed));
                let Some(header) = header else {
                    let _ = writeln!(errors, "S7: run {} has no array", run.id);
                    continue;
                };
                let (used, capacity) = (header as u32, (header >> 32) as u32);
                if snap.base + snap.len > used || used > capacity {
                    let _ = writeln!(errors, "S7: run {} window exceeds its array", run.id);
                }
                *array_refs.entry(snap.array).or_default() += 1;
                if run.local.first().map(|l| l.0) != Some(run.head) {
                    let _ = writeln!(errors, "run {} head disagrees with its array", run.id);
                }
                if snap.len == 0 {
                    let _ = writeln!(errors, "run {} is empty", run.id);
                }
            }
            // S3.
            let mut seen_cut = FxHashSet::default();
            for cut in &run.cuts {
                if !seen_cut.insert(cut.slot) {
                    let _ = writeln!(
                        errors,
                        "S3: run {} has two entries for slot {}",
                        run.id, cut.slot
                    );
                }
                if cut.cutoff == 0 || cut.cutoff >= snap.len {
                    let _ = writeln!(
                        errors,
                        "S3: run {} has cutoff {} of {}",
                        run.id, cut.cutoff, snap.len
                    );
                }
                if run.whole.contains(&cut.slot) {
                    let _ = writeln!(
                        errors,
                        "S3: run {} slot {} is whole and partial",
                        run.id, cut.slot
                    );
                }
            }
            if self.runs.cutoff_live(snap) as usize != run.cuts.len() {
                let _ = writeln!(errors, "run {} cutoff live count is off", run.id);
            }
            // S6.
            if snap.children != NONE {
                let (used, slots) = self.runs.child_load(snap.children);
                if used * 4 > slots * 3 {
                    let _ = writeln!(
                        errors,
                        "S6: run {} child table is {used}/{slots} used",
                        run.id
                    );
                }
                if self.runs.child_live(snap.children) as usize != run.children.len() {
                    let _ = writeln!(errors, "S6: run {} child live count is off", run.id);
                }
            }
            // S1, S2: child keys, offsets, parent links, unique paths.
            let mut keys = FxHashSet::default();
            for &(key, child_id, _) in &run.children {
                if !keys.insert(key) {
                    let _ = writeln!(errors, "S2: run {} links key {key:#x} twice", run.id);
                }
                let Some(&ci) = by_id.get(&child_id) else {
                    continue;
                };
                let child = &runs[ci];
                let offset = child.snap.start.wrapping_sub(snap.start);
                let valid = if run.id == ROOT {
                    offset == 0
                } else {
                    (1..=snap.len).contains(&offset)
                };
                if !valid {
                    let _ = writeln!(
                        errors,
                        "S1: run {} child {} at offset {offset}",
                        run.id, child_id
                    );
                    continue;
                }
                if child.parent != run.id {
                    let _ = writeln!(
                        errors,
                        "S1: child {} names parent {}, linked under {}",
                        child_id, child.parent, run.id
                    );
                }
                if child
                    .local
                    .first()
                    .is_none_or(|&head| key != child_key(offset, head))
                {
                    let _ = writeln!(errors, "S1: child {} key does not match its head", child_id);
                }
                if offset < snap.len && child.local.first() == run.local.get(offset as usize) {
                    let _ = writeln!(
                        errors,
                        "S2: child {} repeats its parent's position {offset}",
                        child_id
                    );
                }
            }
            // S8.
            if snap.flags & FLAG_SEALED != 0 {
                let records = self.runs.forwards(snap.forwards);
                if records.windows(2).any(|w| w[0].0 <= w[1].0) {
                    let _ = writeln!(
                        errors,
                        "S8: run {} forwarding records not decreasing",
                        run.id
                    );
                }
                for &(at, suffix, _) in &records {
                    if suffix != NO_RUN && at < snap.len {
                        let _ =
                            writeln!(errors, "S8: run {} forwards {at} below its length", run.id);
                    }
                }
            } else if snap.forwards != NONE {
                let _ = writeln!(errors, "S8: unsealed run {} has forwarding records", run.id);
            }
            // S9.
            if run.id != ROOT
                && run.whole.is_empty()
                && run.cuts.is_empty()
                && run.children.is_empty()
            {
                let _ = writeln!(errors, "S9: run {} is holder-less and childless", run.id);
            }
            // Live arena blocks, for the leak and overlap checks.
            if run.id != ROOT {
                let capacity = (self
                    .runs
                    .arena
                    .word(snap.array)
                    .map_or(0, |w| w.load(Ordering::Relaxed))
                    >> 32) as u32;
                if let Some(class) = arena::array_class_for(capacity) {
                    live_blocks.push((snap.array, arena::class_words(Kind::Array, class)));
                }
            }
            for (table, words) in self.table_blocks(snap) {
                live_blocks.push((table, words));
            }
        }

        // S5: reference counts match the live runs sharing each array.
        for (&array, &refs) in &array_refs {
            let stored = self
                .runs
                .arena
                .word(array + 1)
                .map_or(0, |w| w.load(Ordering::Relaxed));
            if stored != refs {
                let _ = writeln!(
                    errors,
                    "S5: array {array} has refs {stored}, {refs} live runs"
                );
            }
        }
        live_blocks.sort_unstable();
        live_blocks.dedup();

        // S5: no free-listed id or block is live, and nothing leaks.
        let free_ids: FxHashSet<RunId> = self.runs.slab.free_ids().into_iter().collect();
        for run in &runs {
            if free_ids.contains(&run.id) {
                let _ = writeln!(errors, "S5: live run {} is free-listed", run.id);
            }
        }
        let allocated = self.runs.slab.issued() as usize;
        if allocated != runs.len() - 1 + free_ids.len() {
            let _ = writeln!(
                errors,
                "leak: {allocated} run ids issued, {} live, {} free",
                runs.len() - 1,
                free_ids.len()
            );
        }
        let free_blocks = self.runs.arena.free_blocks();
        let mut intervals: Vec<(u64, u64, bool)> = live_blocks
            .iter()
            .map(|&(a, w)| (u64::from(a), u64::from(a) + u64::from(w), true))
            .chain(free_blocks.iter().map(|&(kind, class, a)| {
                (
                    u64::from(a),
                    u64::from(a) + u64::from(arena::class_words(kind, class)),
                    false,
                )
            }))
            .collect();
        intervals.sort_unstable();
        for pair in intervals.windows(2) {
            if pair[1].0 < pair[0].1 {
                let _ = writeln!(
                    errors,
                    "S5: arena blocks overlap: {:?} and {:?}",
                    pair[0], pair[1]
                );
                break;
            }
        }
        let live_words: u64 = live_blocks.iter().map(|&(_, w)| u64::from(w)).sum();
        let free_words: u64 = free_blocks
            .iter()
            .map(|&(kind, class, _)| u64::from(arena::class_words(kind, class)))
            .sum();
        let stranded = self.runs.arena.bytes().stranded / 8;
        let bumped = self.runs.arena.bumped_words();
        if live_words + free_words + stranded != bumped {
            let _ = writeln!(
                errors,
                "leak: arena bumped {bumped} words, {live_words} live + {free_words} free + {stranded} stranded"
            );
        }

        // S4, both ways.
        let mut maps: FxHashMap<WorkerWithDpRank, FxHashMap<ExternalSequenceBlockHash, Entry>> =
            FxHashMap::default();
        for cell in self.pool.cells.iter() {
            let Some(data) = cell.data.try_lock() else {
                let _ = writeln!(errors, "rank {:?} is running at quiescence", cell.rank);
                continue;
            };
            maps.insert(cell.rank, data.map.iter().collect());
        }
        let mut held: FxHashSet<(WorkerWithDpRank, RunId, u32)> = FxHashSet::default();
        for run in &runs {
            let holders = run
                .whole
                .iter()
                .map(|&slot| (slot, run.snap.len))
                .chain(run.cuts.iter().map(|c| (c.slot, c.cutoff)));
            for (slot, upto) in holders {
                let Some(rank) = table.owner(slot) else {
                    let _ = writeln!(errors, "S4: run {} credits vacated slot {slot}", run.id);
                    continue;
                };
                if table.slot_of(rank).map(|s| s.index()) != Some(slot) {
                    let _ = writeln!(
                        errors,
                        "S4: run {} credits unmapped slot {slot} of {rank:?}",
                        run.id
                    );
                }
                for p in 0..upto {
                    held.insert((rank, run.id, p));
                    let key = run.ext[p as usize];
                    let resolved = maps
                        .get(&rank)
                        .and_then(|entries| entries.get(&key).copied())
                        .and_then(|e| self.resolve(e, key));
                    if resolved
                        != Some(Entry {
                            run: run.id,
                            offset: p,
                            generation: run.generation,
                        })
                    {
                        let _ = writeln!(
                            errors,
                            "S4: {rank:?} holds run {} position {p} without an entry",
                            run.id
                        );
                        break;
                    }
                }
            }
        }
        for (rank, entries) in &maps {
            for (&key, &entry) in entries {
                match self.resolve(entry, key) {
                    Some(e) if held.contains(&(*rank, e.run, e.offset)) => {}
                    other => {
                        let _ = writeln!(
                            errors,
                            "S4: {rank:?} entry {key:?} -> {entry:?} resolves to {other:?}, not held"
                        );
                        break;
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// The table blocks a run owns, as `(addr, words)`.
    fn table_blocks(&self, snap: &Snap) -> Vec<(Addr, u32)> {
        let mut blocks = Vec::new();
        let arena = &self.runs.arena;
        if snap.children != NONE {
            let (_, slots) = self.runs.child_load(snap.children);
            blocks.push((snap.children, 2 + 2 * slots));
        }
        if snap.cutoffs != NONE {
            let slots = arena
                .word(snap.cutoffs)
                .map_or(0, |w| w.load(Ordering::Relaxed) as u32);
            blocks.push((snap.cutoffs, 1 + slots));
        }
        if snap.forwards != NONE {
            let capacity = arena
                .word(snap.forwards)
                .map_or(0, |w| (w.load(Ordering::Relaxed) >> 32) as u32);
            blocks.push((snap.forwards, 1 + 2 * capacity));
        }
        let mut chunk = snap.overflow;
        for _ in 0..=super::slots::MAX_SLOTS / 256 {
            if chunk == NONE {
                break;
            }
            blocks.push((chunk, arena::CHUNK_WORDS_TOTAL));
            chunk = arena
                .word(chunk)
                .map_or(NONE, |w| w.load(Ordering::Relaxed) as u32);
        }
        blocks
    }

    pub fn memory_report(&self) -> MemoryReport {
        let runs = self.live_runs().unwrap_or_default();
        let arena = self.runs.arena.bytes();
        let mut report = MemoryReport {
            arena_free_bytes: arena.free_listed,
            arena_reserved_bytes: arena.reserved,
            arena_stranded_bytes: arena.stranded,
            header_bytes: runs.len() as u64 * std::mem::size_of::<super::runs::RunHeader>() as u64,
            slab_reserved_bytes: self.runs.slab.reserved_bytes(),
            ..MemoryReport::default()
        };
        let mut arrays: FxHashMap<Addr, (u64, u64)> = FxHashMap::default();
        let mut distinct = 0u64;
        for run in &runs {
            if run.id == ROOT {
                continue;
            }
            distinct += u64::from(run.snap.len);
            report.memberships += run.whole.len() as u64 * u64::from(run.snap.len)
                + run.cuts.iter().map(|c| u64::from(c.cutoff)).sum::<u64>();
            let header = self
                .runs
                .arena
                .word(run.snap.array)
                .map_or(0, |w| w.load(Ordering::Relaxed));
            let entry = arrays.entry(run.snap.array).or_insert((0, 0));
            entry.0 = 16 + 16 * (header >> 32);
            entry.1 += 16 * u64::from(run.snap.len);
            for (table, words) in self.table_blocks(&run.snap) {
                let bytes = u64::from(words) * 8;
                if table == run.snap.children {
                    report.child_table_bytes += bytes;
                } else if table == run.snap.cutoffs {
                    report.cutoff_table_bytes += bytes;
                } else if table == run.snap.forwards {
                    report.forward_bytes += bytes;
                } else {
                    report.overflow_bytes += bytes;
                }
            }
        }
        if let Some(root) = runs.first() {
            for (table, words) in self.table_blocks(&root.snap) {
                if table == root.snap.children {
                    report.child_table_bytes += u64::from(words) * 8;
                }
            }
        }
        for (_, (total, live)) in arrays {
            report.array_bytes_live += live;
            report.array_bytes_slack += total.saturating_sub(live);
        }
        report.distinct_blocks = distinct;
        report.map_bytes = self
            .pool
            .cells
            .iter()
            .map(|cell| {
                cell.data
                    .try_lock()
                    .map_or(0, |data| data.map.bytes() as u64)
            })
            .sum();
        report
    }

    pub fn shape_report(&self) -> ShapeReport {
        let runs = self.live_runs().unwrap_or_default();
        let stats = self.stats();
        let mut report = ShapeReport {
            runs_live: runs.len() as u64 - 1,
            runs_allocated: u64::from(self.runs.slab.issued()),
            runs_free: self.runs.slab.free_len() as u64,
            splits_prefix_cap: stats.splits_prefix_cap,
            unlinks: stats.unlinks,
            h1_violations: stats.h1_violations,
            ext_conflicts: stats.ext_conflicts,
            reader_retries: stats.reader_retries,
            reader_fallbacks: stats.reader_fallbacks,
            poisoned_runs: stats.poisoned_seen,
            store_restarts: stats.store_restarts,
            store_failures: stats.store_failures,
            steals: stats.steals,
            inline_fast_path: stats.inline_fast_path,
            claim_check_failures: stats.claim_check_failures,
            sweep_residue: stats.sweep_residue,
            ..ShapeReport::default()
        };
        for run in &runs {
            if run.snap.children == NONE {
                continue;
            }
            let (_, slots) = self.runs.child_load(run.snap.children);
            report.child_tables += 1;
            report.child_slots += u64::from(slots);
            report.child_tombstones += u64::from(self.runs.child_tombstones(run.snap.children));
            report.max_child_probe = report
                .max_child_probe
                .max(u64::from(self.runs.child_max_probe(run.snap.children)));
        }
        report
    }

    /// Live runs, for the harness's structure size.
    pub fn live_run_count(&self) -> usize {
        self.live_runs().map_or(0, |runs| runs.len() - 1)
    }
}
