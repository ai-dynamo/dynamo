// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Stored` events, entry resolution, appends, child claims and prefix-cap splits.
//!
//! A store plans each step under the run's shared gate and state read lock. Coverage
//! changes and child claims commit under that same shared gate. Anything that needs the
//! exclusive gate (a first or grown child table, an append, a split) drops the shared
//! gate, takes the exclusive one, and validates the version it planned under; a changed
//! version sends the store back to planning. Re-plans are bounded by `STORE_REPLANS`.

use parking_lot::RwLockReadGuard;

use super::arena::Free;
use super::run::{MAX_RUN_LEN, PARTIAL_CAP, SEALED};
use super::table::{Claim, Locked, MIN_SLOTS, Table, child_key, pack, slots_for_live, words_for};
use super::types::BlockPos;
use super::*;
use crate::indexer::{EventWarningKind, PreBoundEventCounters};

/// A position: offset `offset` of run `run`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Pos {
    pub(super) run: u32,
    pub(super) offset: u32,
}

/// What one planning step decided.
enum Step {
    /// The rank now holds `count` blocks at `(run, from..)`.
    Placed {
        run: u32,
        from: u32,
        count: usize,
        changed: bool,
    },
    /// Continue at offset 0 of this child.
    Descend(u32),
    /// A split moved the position.
    Moved(Pos),
    /// The run died before the step could lock it.
    Dead,
    /// The plan went stale; plan again from the same position.
    Retry,
    /// A concurrent claim of the same key has not published yet.
    Busy,
}

/// What resolving a lane-map entry found.
pub(super) enum Resolved {
    Stale,
    /// The entry's position and the rank's holding in that run (0 when not asked).
    At {
        pos: BlockPos,
        held: u32,
    },
}

impl ArenaIndexC {
    /// Resolves `entry`, the rank's entry for `key`: a dead run means stale, an offset
    /// past a sealed run's end follows its forwarding records, and the stored external
    /// hash must equal `key`. With `slot`, also reads that rank's holding in the final
    /// run, under the same lock.
    pub(super) fn resolve(
        &self,
        key: ExternalSequenceBlockHash,
        entry: BlockPos,
        slot: Option<Slot>,
    ) -> Resolved {
        let store = &self.store;
        let mut run_id = entry.run();
        let mut offset = entry.offset;
        let mut hops = 0u64;
        loop {
            let run = store.run(run_id);
            let _state = run.state.read();
            if run.is_dead() {
                return Resolved::Stale;
            }
            if offset >= run.len() {
                if !run.flag(SEALED) {
                    return Resolved::Stale;
                }
                let Some(forward) = store.forward_for(run, offset) else {
                    return Resolved::Stale;
                };
                if store.run(forward.suffix).generation.load(Ordering::Relaxed)
                    != forward.generation
                {
                    return Resolved::Stale;
                }
                run_id = forward.suffix;
                offset -= forward.at;
                hops += 1;
                continue;
            }
            if store.window(run).ext(offset as usize) != key.0 {
                self.counters.h1_violations.fetch_add(1, Ordering::Relaxed);
                return Resolved::Stale;
            }
            if hops > 0 {
                self.counters
                    .max_forward_hops
                    .fetch_max(hops, Ordering::Relaxed);
            }
            return Resolved::At {
                pos: BlockPos::new(run_id, offset),
                held: slot.map_or(0, |slot| store.held(run, slot)),
            };
        }
    }

    /// Resolves the rank's entry for `key`, rewriting it if it moved. Drops it if stale.
    pub(super) fn resolve_entry(
        &self,
        lookup: &mut RankLookup,
        key: ExternalSequenceBlockHash,
        slot: Option<Slot>,
    ) -> Option<(BlockPos, u32)> {
        let entry = lookup.map.get(key)?;
        match self.resolve(key, entry, slot) {
            Resolved::Stale => {
                lookup.map.remove(key);
                None
            }
            Resolved::At { pos, held } => {
                if pos != entry && self.path_compression_enabled() {
                    lookup.map.insert(key, pos);
                }
                Some((pos, held))
            }
        }
    }

    #[cfg(test)]
    fn path_compression_enabled(&self) -> bool {
        !self.hooks.skip_path_compression()
    }

    #[cfg(not(test))]
    #[inline]
    fn path_compression_enabled(&self) -> bool {
        true
    }

    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn apply_stored(
        &self,
        lane: &mut CLane,
        rank: WorkerWithDpRank,
        op: KvCacheStoreData,
        id: u64,
        counters: Option<&PreBoundEventCounters>,
    ) -> Result<(), KvCacheEventError> {
        let guard = crossbeam_epoch::pin();
        let slot = self.slots.acquire(rank, &guard).inspect_err(|_| {
            tracing::warn!(
                worker_id = rank.worker_id.to_string(),
                dp_rank = rank.dp_rank,
                id,
                "No free coverage slot; skipping store operation"
            );
        })?;
        self.note_slot(slot);
        let CLane {
            ranks, free, tally, ..
        } = lane;
        let lookup = ranks.entry(rank).or_default();
        if op.blocks.is_empty() {
            return Ok(());
        }

        let start = match op.parent_hash {
            None => Pos {
                run: ROOT,
                offset: 0,
            },
            Some(parent) => {
                let Some((pos, held)) = self.resolve_entry(lookup, parent, Some(slot)) else {
                    tracing::debug!(
                        worker_id = rank.worker_id.to_string(),
                        dp_rank = rank.dp_rank,
                        id,
                        parent_hash = ?parent,
                        "Store parent not found"
                    );
                    return Err(KvCacheEventError::ParentBlockNotFound);
                };
                if held <= pos.offset {
                    // The entry outlived the rank's coverage of the parent.
                    lookup.map.remove(parent);
                    tracing::warn!(
                        worker_id = rank.worker_id.to_string(),
                        dp_rank = rank.dp_rank,
                        id,
                        parent_hash = ?parent,
                        "Store parent is not covered by the worker; rejecting store"
                    );
                    return Err(KvCacheEventError::ParentBlockNotFound);
                }
                Pos {
                    run: pos.run(),
                    offset: pos.offset + 1,
                }
            }
        };

        let changed = self.place(lookup, slot, start, &op.blocks, free, tally)?;
        if !changed && let Some(counters) = counters {
            counters.inc_warning(EventWarningKind::DuplicateStore);
        }
        Ok(())
    }

    /// Places `blocks` for `slot` starting at `at`, recording a map entry for each.
    /// Returns whether anything changed.
    fn place(
        &self,
        lookup: &mut RankLookup,
        slot: Slot,
        mut at: Pos,
        blocks: &[KvCacheStoredBlockData],
        free: &mut FreeBatch,
        tally: &mut ReclaimTally,
    ) -> Result<bool, KvCacheEventError> {
        let mut rest = blocks;
        let mut changed = false;
        // Where the store descended from, to re-plan if the child dies under it.
        let mut parent: Option<Pos> = None;
        let mut replans = 0usize;
        let mut busy = 0usize;
        while !rest.is_empty() {
            match self.store_step(at, slot, rest, free, tally)? {
                Step::Placed {
                    run,
                    from,
                    count,
                    changed: coverage_changed,
                } => {
                    changed |= coverage_changed;
                    let keys = rest[..count].iter().map(|block| block.block_hash);
                    changed |= lookup.map.insert_run(keys, run, from) > 0;
                    rest = &rest[count..];
                    at = Pos {
                        run,
                        offset: from + count as u32,
                    };
                    parent = None;
                }
                Step::Descend(child) => {
                    parent = Some(at);
                    at = Pos {
                        run: child,
                        offset: 0,
                    };
                }
                Step::Moved(pos) => at = pos,
                Step::Busy => {
                    // Bounded by the racing claimer, which publishes right after its CAS.
                    busy += 1;
                    if busy.is_multiple_of(64) {
                        std::thread::yield_now();
                    } else {
                        std::hint::spin_loop();
                    }
                    if busy > 1 << 20 {
                        return Err(self.store_failure("a racing child claim never published"));
                    }
                }
                step @ (Step::Dead | Step::Retry) => {
                    replans += 1;
                    self.counters.plan_retries.fetch_add(1, Ordering::Relaxed);
                    if replans > STORE_REPLANS {
                        return Err(self.store_failure("too many re-plans"));
                    }
                    if matches!(step, Step::Dead) {
                        let Some(from) = parent.take() else {
                            return Err(self.store_failure("a held run died"));
                        };
                        at = from;
                    }
                }
            }
        }
        Ok(changed)
    }

    fn store_failure(&self, reason: &'static str) -> KvCacheEventError {
        self.counters.store_failures.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(reason, "arena-c store failed");
        KvCacheEventError::IndexerInvariantViolation
    }

    /// Plans one step at `at` and commits it if the shared gate suffices.
    fn store_step(
        &self,
        at: Pos,
        slot: Slot,
        rest: &[KvCacheStoredBlockData],
        free: &mut FreeBatch,
        tally: &mut ReclaimTally,
    ) -> Result<Step, KvCacheEventError> {
        let store = &self.store;
        let run = store.run(at.run);
        let gate = run.gate.read();
        let state = run.state.read();
        if run.is_dead() {
            return Ok(Step::Dead);
        }
        let version = run.version.load(Ordering::Acquire);
        let len = run.len();
        let o = at.offset;
        if o > len {
            // A split moved our position. Follow the last position the rank holds,
            // `o - 1`: the suffix holding it is alive, while position `o` may lie in a
            // later suffix that has died since (nobody held it).
            let moved = run
                .flag(SEALED)
                .then(|| store.forward_for(run, o - 1))
                .flatten()
                .filter(|forward| {
                    store.run(forward.suffix).generation.load(Ordering::Relaxed)
                        == forward.generation
                });
            return Ok(match moved {
                Some(forward) => Step::Moved(Pos {
                    run: forward.suffix,
                    offset: o - forward.at,
                }),
                None => Step::Dead,
            });
        }

        let window = store.window(run);
        let m = common_prefix(window, o as usize, rest);
        if m > 0 {
            // H1: matching local hashes name the same external hashes. A disagreement is
            // counted; the rank's map stays keyed by its own hash.
            if (0..m).any(|i| window.ext(o as usize + i) != rest[i].block_hash.0) {
                self.counters.ext_conflicts.fetch_add(1, Ordering::Relaxed);
            }
            let target = o + m as u32;
            let held = store.held(run, slot);
            debug_assert!(held >= o, "store reached offset {o} holding only {held}");
            let placed = |changed| Step::Placed {
                run: at.run,
                from: o,
                count: m,
                changed,
            };
            if held >= target {
                return Ok(placed(false));
            }
            if target == len {
                // Promote: set the bit first so readers never miss the rank, then drop a
                // stale cutoff under the state write lock.
                let stale_cutoff = store.cutoff_of(run, slot).is_some();
                drop(state);
                store.whole_set(run, slot)?;
                if stale_cutoff {
                    let _state = run.state.write();
                    store.cutoff_remove(run, slot, free, &self.store);
                }
                return Ok(placed(true));
            }
            drop(state);
            let state = run.state.write();
            if store.cutoff_set(run, slot, target, free, &self.store)? {
                return Ok(placed(true));
            }
            drop(state);
            drop(gate);
            self.split_for_cap(at.run, free)?;
            return Ok(Step::Retry);
        }

        let key = child_key(o, rest[0].tokens_hash.0);
        if let Some(child) = store.find_child(run, key) {
            return Ok(Step::Descend(child));
        }
        let append = at.run != ROOT
            && o == len
            && !run.flag(SEALED)
            && len as usize + rest.len() <= MAX_RUN_LEN
            && store.whole_sole(run, slot);
        drop(state);
        if append {
            drop(gate);
            return self.append(at.run, version, slot, rest, free, tally);
        }
        self.create_child(at.run, gate, version, o, slot, rest, free, tally)
    }

    /// Appends `rest` to `run_id`, whose sole whole holder is `slot`, under the exclusive
    /// gate. Claims do not bump the version, so after validating it re-probes the end
    /// child the append would shadow and descends into it if one appeared.
    fn append(
        &self,
        run_id: u32,
        version: u64,
        slot: Slot,
        rest: &[KvCacheStoredBlockData],
        free: &mut FreeBatch,
        tally: &mut ReclaimTally,
    ) -> Result<Step, KvCacheEventError> {
        let store = &self.store;
        let run = store.run(run_id);
        let _gate = run.gate.write();
        let _state = run.state.write();
        if run.is_dead() {
            return Ok(Step::Dead);
        }
        if run.version.load(Ordering::Relaxed) != version {
            return Ok(Step::Retry);
        }
        let len = run.len();
        // A promotion under the shared gate may have added a whole holder since the plan.
        if !store.whole_sole(run, slot) || len as usize + rest.len() > MAX_RUN_LEN {
            return Ok(Step::Retry);
        }
        if let Some(child) = store.find_child(run, child_key(len, rest[0].tokens_hash.0)) {
            return Ok(Step::Descend(child));
        }
        store.append_blocks(run, rest, free, &self.store)?;
        run.bump_version();
        tally.linked(rest.len());
        Ok(Step::Placed {
            run: run_id,
            from: len,
            count: rest.len(),
            changed: true,
        })
    }

    /// Creates a run for `rest` and links it as `run_id`'s child at `o`: by a claim under
    /// the shared gate when the table has room, otherwise under the exclusive gate.
    #[allow(clippy::too_many_arguments)]
    fn create_child(
        &self,
        run_id: u32,
        gate: RwLockReadGuard<'_, ()>,
        version: u64,
        o: u32,
        slot: Slot,
        rest: &[KvCacheStoredBlockData],
        free: &mut FreeBatch,
        tally: &mut ReclaimTally,
    ) -> Result<Step, KvCacheEventError> {
        let store = &self.store;
        let run = store.run(run_id);
        let count = rest.len().min(MAX_RUN_LEN);
        let start = run.start.load(Ordering::Relaxed).saturating_add(o);
        let (child, generation) = store.new_run(run_id, start, &rest[..count], slot)?;
        let key = child_key(o, rest[0].tokens_hash.0);
        let value = pack(child, generation);
        let placed = Step::Placed {
            run: child,
            from: 0,
            count,
            changed: true,
        };

        let table = run.children.load(Ordering::Acquire);
        if table != NONE {
            match store.table(table).claim(key, value) {
                Claim::Claimed => {
                    tally.linked(count);
                    return Ok(placed);
                }
                Claim::Exists(existing) => {
                    store.discard_run(child);
                    return Ok(Step::Descend(table::unpack_run(existing)));
                }
                Claim::Busy => {
                    store.discard_run(child);
                    return Ok(Step::Busy);
                }
                Claim::Full => {}
            }
        }
        drop(gate);

        let _gate = run.gate.write();
        let _state = run.state.write();
        let outcome = if run.is_dead() {
            Some(Step::Dead)
        } else if run.version.load(Ordering::Relaxed) != version {
            Some(Step::Retry)
        } else {
            store.find_child(run, key).map(Step::Descend)
        };
        if let Some(step) = outcome {
            store.discard_run(child);
            return Ok(step);
        }
        if let Err(error) = self.insert_child_locked(run_id, key, value, free) {
            store.discard_run(child);
            return Err(error);
        }
        run.bump_version();
        tally.linked(count);
        Ok(placed)
    }

    /// Inserts a child under the exclusive gate and state write lock, creating the table,
    /// reusing a tombstone, or rebuilding at three eighths load. ROOT's table is read
    /// without locks, so it never reuses tombstones and is replaced through the epoch.
    fn insert_child_locked(
        &self,
        run_id: u32,
        key: u64,
        value: u64,
        free: &mut FreeBatch,
    ) -> Result<(), KvCacheEventError> {
        let store = &self.store;
        let run = store.run(run_id);
        let old = run.children.load(Ordering::Relaxed);
        if old != NONE {
            match store.table(old).insert_locked(key, value) {
                Locked::Inserted => return Ok(()),
                Locked::Exists(_) => return Err(KvCacheEventError::IndexerInvariantViolation),
                Locked::Full => {}
            }
        }
        let live = if old == NONE {
            0
        } else {
            store.table(old).live()
        };
        let floor = if run_id == ROOT {
            ROOT_SLOTS
        } else {
            MIN_SLOTS
        };
        let slots = slots_for_live(live + 1).max(floor);
        let block = store.arena.alloc(words_for(slots), true)?;
        let words = store.arena.slice(block.addr, words_for(slots));
        Table::init(words, slots);
        let fresh = Table::new(words);
        if old != NONE {
            for (k, v) in store.table(old).entries() {
                fresh.insert_locked(k, v);
            }
        }
        let inserted = fresh.insert_locked(key, value);
        debug_assert_eq!(inserted, Locked::Inserted);
        run.children.store(block.addr, Ordering::Release);
        if old == NONE {
            self.counters.child_tables.fetch_add(1, Ordering::Relaxed);
        } else {
            free.push(&self.store, Free::Words(store.table_block(old)));
        }
        Ok(())
    }

    /// Splits `run_id` at its median cutoff if it still carries [`PARTIAL_CAP`] partial
    /// holders.
    pub(super) fn split_for_cap(
        &self,
        run_id: u32,
        free: &mut FreeBatch,
    ) -> Result<(), KvCacheEventError> {
        let run = self.store.run(run_id);
        let _gate = run.gate.write();
        let _state = run.state.write();
        if run.is_dead() || self.store.cutoff_count(run) < PARTIAL_CAP {
            return Ok(());
        }
        self.split_at_median(run_id, free)
    }

    /// Splits at the median cutoff. Callers hold the exclusive gate and state write lock.
    pub(super) fn split_at_median(
        &self,
        run_id: u32,
        free: &mut FreeBatch,
    ) -> Result<(), KvCacheEventError> {
        let run = self.store.run(run_id);
        let mut cutoffs: Vec<u32> = self.store.cutoff_entries(run).map(|(_, c)| c).collect();
        if cutoffs.is_empty() {
            return Ok(());
        }
        cutoffs.sort_unstable();
        let at = cutoffs[cutoffs.len() / 2];
        self.store.split_run(run_id, at, free, &self.store)?;
        run.bump_version();
        self.counters.splits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
impl ArenaIndexC {
    /// Commits an append planned at `version` without planning again. Returns the child
    /// the end-child re-probe descended into, if any.
    pub(super) fn probe_append(
        &self,
        run: u32,
        version: u64,
        slot: Slot,
        blocks: &[KvCacheStoredBlockData],
        lane: &mut CLane,
    ) -> Result<Option<u32>, KvCacheEventError> {
        let step = self.append(run, version, slot, blocks, &mut lane.free, &mut lane.tally)?;
        Ok(match step {
            Step::Descend(child) => Some(child),
            Step::Placed { .. } => None,
            _ => panic!("append did not run"),
        })
    }
}

#[cfg(test)]
impl ArenaIndexC {
    /// Continues a store for `rank` at `(run, offset)`, as if a planning step had just
    /// placed the blocks before it there.
    pub(super) fn probe_place_at(
        &self,
        lane: &mut CLane,
        rank: WorkerWithDpRank,
        run: u32,
        offset: u32,
        blocks: &[KvCacheStoredBlockData],
    ) -> Result<bool, KvCacheEventError> {
        let slot = self
            .slots
            .table(&crossbeam_epoch::pin())
            .slot_of(rank)
            .expect("rank has a slot");
        let CLane {
            ranks, free, tally, ..
        } = lane;
        let lookup = ranks.entry(rank).or_default();
        self.place(lookup, slot, Pos { run, offset }, blocks, free, tally)
    }

    /// Splits `run` at `at` under its locks, as a prefix-cap split would.
    pub(super) fn probe_split(&self, run: u32, at: u32, lane: &mut CLane) {
        let header = self.store.run(run);
        let _gate = header.gate.write();
        let _state = header.state.write();
        self.store
            .split_run(run, at, &mut lane.free, &self.store)
            .unwrap();
        header.bump_version();
    }
}
