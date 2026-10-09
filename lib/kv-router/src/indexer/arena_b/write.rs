// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Writes (spec B 7): stores, removals, clears, rank removal, prefix-cap splits, and eager
//! reclamation (10). Every event runs on the lane that owns its rank's cell, with the
//! rank's map in hand; a run is locked only to change it, and planning reads validated
//! snapshots.

use std::collections::VecDeque;

use crossbeam_epoch::Guard;
use rustc_hash::{FxHashMap, FxHashSet};

use super::ArenaIndex;
use super::protocol::FLAG_SEALED;
use super::rank_map::{Entry, RankMap};
use super::runs::{
    Cutoff, Frees, INLINE_WORDS, Inserted, Locked, MAX_FORWARD_HOPS, MAX_RUN_LEN, NO_RUN,
    PARTIAL_CAP, Probe, ROOT, ROOT_GEN, Read, RunHeader, RunId, Snap, bump, child_key,
};
use super::slots::{RemovalTarget, Slot, wait_for_pinned_threads};
use crate::indexer::{EventKind, EventWarningKind, PreBoundEventCounters};
use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, KvCacheRemoveData,
    KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash, RouterEvent, WorkerWithDpRank,
};

/// Placement restarts a store may take before it fails visibly (fix 9).
pub(crate) const STORE_RESTARTS: u32 = 8;
/// Re-plans at one position before a re-plan counts as a restart.
const MAX_REPLANS: u32 = 16;

/// The state a rank's cell owns: its block map. Only the lane running the cell touches it.
pub(crate) struct RankState {
    pub(crate) rank: WorkerWithDpRank,
    pub(crate) map: RankMap,
    /// Whether the rank has stored since it was last cleared; removals for a rank without
    /// a map answer `BlockNotFound`, as CRTC's lane lookup does.
    pub(crate) has_map: bool,
}

impl RankState {
    pub(crate) fn new(rank: WorkerWithDpRank) -> Self {
        Self {
            rank,
            map: RankMap::default(),
            has_map: false,
        }
    }
}

/// A position: offset `off` of run `run` at generation `generation`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pos {
    run: RunId,
    generation: u32,
    off: u32,
}

const ROOT_POS: Pos = Pos {
    run: ROOT,
    generation: ROOT_GEN,
    off: 0,
};

/// The plan one validated attempt at a position produced.
enum Plan {
    /// The position moved past the run's end: follow the forwarding record.
    Forward(Option<(u32, RunId, u32)>),
    /// `m` positions from `off` match the next blocks; `conflict` when the next one
    /// matches by local hash only (an H1 violation).
    Match { m: u32, conflict: bool, held: u32 },
    /// The run diverges from the blocks at `off`, or ends there.
    Diverge {
        child: Probe,
        sole: bool,
        snap: Snap,
    },
}

enum Placed {
    Advanced(usize),
    Replan,
    Restart,
    /// An H1 conflict: stop placing, undercounting the rest.
    Stop,
}

/// Raising a holding.
enum Raise {
    Done(bool),
    /// The run split; the caller re-plans.
    Split,
}

fn pairs(blocks: &[KvCacheStoredBlockData]) -> Vec<(LocalBlockHash, ExternalSequenceBlockHash)> {
    blocks
        .iter()
        .map(|b| (b.tokens_hash, b.block_hash))
        .collect()
}

impl ArenaIndex {
    // ------------------------------------------------------------------
    // Dispatch
    // ------------------------------------------------------------------

    pub(crate) fn apply_event(
        &self,
        st: &mut RankState,
        event: RouterEvent,
        counters: Option<&PreBoundEventCounters>,
    ) -> Result<(), KvCacheEventError> {
        let id = event.event.event_id;
        debug_assert_eq!(
            st.rank,
            WorkerWithDpRank::new(event.worker_id, event.event.dp_rank)
        );
        let kind = EventKind::of(&event.event.data);
        let result = match event.event.data {
            KvCacheEventData::Stored(op) => {
                let guard = crossbeam_epoch::pin();
                self.apply_stored(st, op, id, counters, &guard)
            }
            KvCacheEventData::Removed(op) => {
                let guard = crossbeam_epoch::pin();
                self.apply_removed(st, op, &guard)
            }
            KvCacheEventData::Cleared => {
                self.clear_rank(st);
                self.slots.wait_for_release(RemovalTarget::Rank(st.rank));
                Ok(())
            }
        };
        if let Some(counters) = counters {
            counters.inc(kind, result);
        }
        result
    }

    // ------------------------------------------------------------------
    // Shared reads
    // ------------------------------------------------------------------

    /// Resolves a map entry for `key` (7.1): follows forwarding records for at most
    /// [`MAX_FORWARD_HOPS`] hops and checks H1 (`ext[offset] == key`). `None` means stale.
    pub(crate) fn resolve(&self, entry: Entry, key: ExternalSequenceBlockHash) -> Option<Entry> {
        self.resolve_held(entry, key, None).map(|(e, _)| e)
    }

    /// [`Self::resolve`], also reading what `slot` holds of the resolved run in the same
    /// validated attempt, so a split between the two cannot make a held parent look stale.
    fn resolve_held(
        &self,
        entry: Entry,
        key: ExternalSequenceBlockHash,
        slot: Option<Slot>,
    ) -> Option<(Entry, u32)> {
        enum Res {
            Here(ExternalSequenceBlockHash, u32),
            Forward(Option<(u32, RunId, u32)>),
            Stale,
        }
        let mut e = entry;
        for _ in 0..MAX_FORWARD_HOPS {
            let runs = &self.runs;
            let read = runs.read(e.run, e.generation, |h, snap| {
                if e.offset < snap.len {
                    let columns = runs.columns(snap)?;
                    let held = match slot {
                        Some(slot) => held_in_attempt(self, h, snap, slot)?,
                        None => 0,
                    };
                    Some(Res::Here(columns.ext(e.offset as usize), held))
                } else if snap.sealed() {
                    Some(Res::Forward(runs.forward_for(snap.forwards, e.offset)?))
                } else {
                    Some(Res::Stale)
                }
            });
            match read {
                Read::Ok(Res::Here(ext, held)) if ext == key => return Some((e, held)),
                Read::Ok(Res::Here(..)) => {
                    bump(&self.runs.stats.h1_violations);
                    return None;
                }
                Read::Ok(Res::Forward(Some((at, suffix, generation)))) if suffix != NO_RUN => {
                    e = Entry {
                        run: suffix,
                        offset: e.offset - at,
                        generation,
                    };
                }
                _ => return None,
            }
        }
        None
    }

    fn release_hash(&self, rank: WorkerWithDpRank, hash: ExternalSequenceBlockHash) {
        if self.lifecycle.is_enabled() {
            self.lifecycle.remove(rank, hash);
        }
    }

    fn forget(&self, st: &mut RankState, hash: ExternalSequenceBlockHash) {
        if st.map.remove(hash).is_some() {
            self.release_hash(st.rank, hash);
        }
    }

    // ------------------------------------------------------------------
    // Stored (7.2)
    // ------------------------------------------------------------------

    fn apply_stored(
        &self,
        st: &mut RankState,
        op: KvCacheStoreData,
        id: u64,
        counters: Option<&PreBoundEventCounters>,
        guard: &Guard,
    ) -> Result<(), KvCacheEventError> {
        let slot = self.slots.acquire(st.rank, guard).inspect_err(|_| {
            tracing::warn!(
                worker_id = st.rank.worker_id.to_string(),
                dp_rank = st.rank.dp_rank,
                id,
                "No free coverage slot; skipping store operation"
            );
        })?;
        st.has_map = true;
        if op.blocks.is_empty() {
            return Ok(());
        }
        let mut frees = Frees::default();
        let result = self
            .store_start(st, slot, op.parent_hash, id, op.blocks.len())
            .and_then(|start| self.place(st, slot, start, &op.blocks, op.parent_hash, &mut frees));
        self.runs.flush(&mut frees);
        let changed = result?;
        if !changed && let Some(counters) = counters {
            counters.inc_warning(EventWarningKind::DuplicateStore);
        }
        Ok(())
    }

    /// The position after the store's parent (7.2 step 2).
    fn store_start(
        &self,
        st: &mut RankState,
        slot: Slot,
        parent: Option<ExternalSequenceBlockHash>,
        id: u64,
        num_blocks: usize,
    ) -> Result<Pos, KvCacheEventError> {
        let Some(parent) = parent else {
            return Ok(ROOT_POS);
        };
        let Some(entry) = st.map.get(parent) else {
            tracing::warn!(
                worker_id = st.rank.worker_id.to_string(),
                dp_rank = st.rank.dp_rank,
                id,
                parent_hash = ?parent,
                num_blocks,
                "Failed to find parent block; skipping store operation"
            );
            return Err(KvCacheEventError::ParentBlockNotFound);
        };
        let resolved = self.resolve_held(entry, parent, Some(slot));
        match resolved {
            Some((e, held)) if held > e.offset => {
                if e != entry {
                    st.map.insert(parent, e);
                }
                Ok(Pos {
                    run: e.run,
                    generation: e.generation,
                    off: e.offset + 1,
                })
            }
            _ => {
                tracing::warn!(
                    worker_id = st.rank.worker_id.to_string(),
                    dp_rank = st.rank.dp_rank,
                    id,
                    parent_hash = ?parent,
                    "Stale parent: worker no longer holds parent_hash; rejecting store"
                );
                self.forget(st, parent);
                Err(KvCacheEventError::ParentBlockNotFound)
            }
        }
    }

    /// Places `blocks` from `pos`. Returns whether anything changed.
    fn place(
        &self,
        st: &mut RankState,
        slot: Slot,
        mut pos: Pos,
        blocks: &[KvCacheStoredBlockData],
        parent: Option<ExternalSequenceBlockHash>,
        frees: &mut Frees,
    ) -> Result<bool, KvCacheEventError> {
        let mut i = 0;
        let mut restarts = 0;
        let mut replans = 0;
        let mut changed = false;
        while i < blocks.len() {
            let placed = self.place_one(st, slot, &mut pos, &blocks[i..], &mut changed, frees)?;
            let restart = match placed {
                Placed::Advanced(n) => {
                    i += n;
                    replans = 0;
                    false
                }
                Placed::Replan => {
                    replans += 1;
                    bump(&self.runs.stats.replans);
                    replans > MAX_REPLANS
                }
                Placed::Restart => true,
                Placed::Stop => break,
            };
            if !restart {
                continue;
            }
            restarts += 1;
            replans = 0;
            bump(&self.runs.stats.store_restarts);
            if restarts > STORE_RESTARTS {
                bump(&self.runs.stats.store_failures);
                tracing::warn!(
                    worker_id = st.rank.worker_id.to_string(),
                    dp_rank = st.rank.dp_rank,
                    placed = i,
                    num_blocks = blocks.len(),
                    "Store placement restarted too often; failing the event"
                );
                return Err(KvCacheEventError::IndexerInvariantViolation);
            }
            // Restart from the last position the rank verifiably holds.
            pos = self.restart_pos(st, i, blocks, parent)?;
            // Most restarts meet a child an unlinker killed but has not tombstoned yet.
            // The unlinker holds the parent's lock across both, so taking it here waits
            // the window out instead of spinning through the restart budget.
            drop(self.runs.lock(pos.run, pos.generation));
            for _ in 0..restarts {
                std::thread::yield_now();
            }
        }
        Ok(changed)
    }

    fn restart_pos(
        &self,
        st: &RankState,
        placed: usize,
        blocks: &[KvCacheStoredBlockData],
        parent: Option<ExternalSequenceBlockHash>,
    ) -> Result<Pos, KvCacheEventError> {
        let key = match placed.checked_sub(1) {
            Some(last) => blocks[last].block_hash,
            None => match parent {
                None => return Ok(ROOT_POS),
                Some(parent) => parent,
            },
        };
        let e = st
            .map
            .get(key)
            .and_then(|entry| self.resolve(entry, key))
            .ok_or(KvCacheEventError::ParentBlockNotFound)?;
        Ok(Pos {
            run: e.run,
            generation: e.generation,
            off: e.offset + 1,
        })
    }

    fn plan(&self, pos: Pos, rest: &[KvCacheStoredBlockData], slot: Slot) -> Read<Plan> {
        let runs = &self.runs;
        runs.read(pos.run, pos.generation, |h, snap| {
            if pos.off > snap.len {
                if !snap.sealed() {
                    return Some(Plan::Forward(None));
                }
                return Some(Plan::Forward(runs.forward_for(snap.forwards, pos.off)?));
            }
            let columns = runs.columns(snap)?;
            let (len, off) = (snap.len as usize, pos.off as usize);
            let mut m = 0usize;
            let mut conflict = false;
            while off + m < len && m < rest.len() {
                if columns.local(off + m) != rest[m].tokens_hash {
                    break;
                }
                if columns.ext(off + m) != rest[m].block_hash {
                    conflict = true;
                    break;
                }
                m += 1;
            }
            if m > 0 || conflict {
                let held = held_in_attempt(self, h, snap, slot)?;
                return Some(Plan::Match {
                    m: m as u32,
                    conflict,
                    held,
                });
            }
            let child =
                match runs.probe_child(snap.children, child_key(pos.off, rest[0].tokens_hash)) {
                    Probe::Torn => return None,
                    probe => probe,
                };
            let sole = off == len
                && !snap.sealed()
                && pos.run != ROOT
                && sole_whole_in_attempt(self, h, snap, slot)?;
            Some(Plan::Diverge {
                child,
                sole,
                snap: *snap,
            })
        })
    }

    fn place_one(
        &self,
        st: &mut RankState,
        slot: Slot,
        pos: &mut Pos,
        rest: &[KvCacheStoredBlockData],
        changed: &mut bool,
        frees: &mut Frees,
    ) -> Result<Placed, KvCacheEventError> {
        let Read::Ok(plan) = self.plan(*pos, rest, slot) else {
            return Ok(Placed::Restart);
        };
        match plan {
            Plan::Forward(Some((at, suffix, generation))) if suffix != NO_RUN => {
                *pos = Pos {
                    run: suffix,
                    generation,
                    off: pos.off - at,
                };
                Ok(Placed::Replan)
            }
            Plan::Forward(_) => Ok(Placed::Restart),
            Plan::Match { m, conflict, held } => {
                if conflict {
                    bump(&self.runs.stats.ext_conflicts);
                }
                if m == 0 {
                    return Ok(Placed::Stop);
                }
                let target = pos.off + m;
                if held < target {
                    let Some(locked) = self.runs.lock(pos.run, pos.generation) else {
                        return Ok(Placed::Restart);
                    };
                    let snap = locked.snap();
                    if target > snap.len {
                        return Ok(Placed::Replan);
                    }
                    if self.runs.held_locked(&locked, &snap, slot) < pos.off {
                        // Contiguity broke: the rank no longer holds the positions before.
                        return Ok(Placed::Restart);
                    }
                    match self.raise_holding(&locked, slot, target, frees)? {
                        Raise::Split => return Ok(Placed::Replan),
                        Raise::Done(raised) => *changed |= raised,
                    }
                }
                let n = self.record(st, &rest[..m as usize], pos.run, pos.off, pos.generation);
                *changed |= n > 0;
                pos.off = target;
                Ok(if conflict {
                    Placed::Stop
                } else {
                    Placed::Advanced(m as usize)
                })
            }
            Plan::Diverge { child, sole, snap } => {
                if pos.off == 0 && pos.run != ROOT {
                    // The child slot that led here named a run whose head differs: a key
                    // collision or corruption. Fail visibly rather than nest offset 0.
                    return Err(KvCacheEventError::IndexerInvariantViolation);
                }
                match child {
                    Probe::Found(id, generation) => {
                        *pos = Pos {
                            run: id,
                            generation,
                            off: 0,
                        };
                        Ok(Placed::Advanced(0))
                    }
                    _ if sole && snap.len as usize + rest.len() <= MAX_RUN_LEN as usize => {
                        self.append(st, slot, pos, rest, changed, frees)
                    }
                    _ => self.hang_child(st, slot, pos, &snap, rest, changed, frees),
                }
            }
        }
    }

    /// Records map entries for `blocks` at `(run, off..)`; returns how many changed.
    fn record(
        &self,
        st: &mut RankState,
        blocks: &[KvCacheStoredBlockData],
        run: RunId,
        off: u32,
        generation: u32,
    ) -> usize {
        let rank = st.rank;
        let lifecycle = &self.lifecycle;
        st.map.insert_run(
            blocks.iter().map(|b| b.block_hash),
            run,
            off,
            generation,
            |hash| {
                if lifecycle.is_enabled() {
                    lifecycle.insert(rank, hash);
                }
            },
        )
    }

    /// Raises `slot`'s holding of a locked run to `target`, promoting it to whole at the
    /// run's end. Adding a partial entry past [`PARTIAL_CAP`] splits instead.
    fn raise_holding(
        &self,
        locked: &Locked<'_>,
        slot: Slot,
        target: u32,
        frees: &mut Frees,
    ) -> Result<Raise, KvCacheEventError> {
        let snap = locked.snap();
        let current = self.runs.held_locked(locked, &snap, slot);
        if current >= target {
            return Ok(Raise::Done(false));
        }
        if target == snap.len {
            self.runs.promote(locked, slot)?;
            return Ok(Raise::Done(true));
        }
        if current == 0 && self.runs.cutoff_live(&snap) >= PARTIAL_CAP {
            self.split_median(locked, frees)?;
            return Ok(Raise::Split);
        }
        self.runs.set_cutoff(locked, slot, target, frees, false)?;
        Ok(Raise::Done(true))
    }

    fn append(
        &self,
        st: &mut RankState,
        slot: Slot,
        pos: &mut Pos,
        rest: &[KvCacheStoredBlockData],
        changed: &mut bool,
        frees: &mut Frees,
    ) -> Result<Placed, KvCacheEventError> {
        let Some(locked) = self.runs.lock(pos.run, pos.generation) else {
            return Ok(Placed::Restart);
        };
        let snap = locked.snap();
        let count = rest.len() as u32;
        if snap.len != pos.off
            || snap.flags & FLAG_SEALED != 0
            || u64::from(snap.len) + u64::from(count) > u64::from(MAX_RUN_LEN)
            || !self.sole_whole_locked(&locked, slot)
        {
            return Ok(Placed::Replan);
        }
        let blocks = pairs(rest);
        if snap.children == super::arena::NONE && self.runs.has_room(&snap, count) {
            // W5: the sole whole holder, no child table, room in place.
            if self.runs.append_locked(&locked, &snap, &blocks, frees)? {
                drop(locked);
                return Ok(self.finish_append(st, pos, rest, changed));
            }
        }
        let step = if snap.children == super::arena::NONE {
            locked.h.step()
        } else {
            locked.h.step_excluding_claims()
        };
        // With a table, a claimer may have linked the end child the append would create.
        if let Probe::Found(id, generation) = self
            .runs
            .probe_child(snap.children, child_key(snap.len, rest[0].tokens_hash))
        {
            drop(step);
            *pos = Pos {
                run: id,
                generation,
                off: 0,
            };
            return Ok(Placed::Advanced(0));
        }
        let snap = locked.snap();
        self.runs.append_locked(&locked, &snap, &blocks, frees)?;
        drop(step);
        drop(locked);
        Ok(self.finish_append(st, pos, rest, changed))
    }

    fn finish_append(
        &self,
        st: &mut RankState,
        pos: &mut Pos,
        rest: &[KvCacheStoredBlockData],
        changed: &mut bool,
    ) -> Placed {
        self.record(st, rest, pos.run, pos.off, pos.generation);
        *changed = true;
        pos.off += rest.len() as u32;
        Placed::Advanced(rest.len())
    }

    /// Creates a run for up to [`MAX_RUN_LEN`] of the blocks with the rank as its whole
    /// holder, and claims it as the run's child at `pos.off` (5.3).
    #[allow(clippy::too_many_arguments)]
    fn hang_child(
        &self,
        st: &mut RankState,
        slot: Slot,
        pos: &mut Pos,
        snap: &Snap,
        rest: &[KvCacheStoredBlockData],
        changed: &mut bool,
        frees: &mut Frees,
    ) -> Result<Placed, KvCacheEventError> {
        let take = rest.len().min(MAX_RUN_LEN as usize);
        let blocks = &rest[..take];
        let array = self.runs.new_array(&pairs(blocks))?;
        let (child, child_gen) = match self.runs.create(
            pos.run,
            snap.start + pos.off,
            blocks[0].tokens_hash,
            array,
            0,
            take as u32,
        ) {
            Ok(created) => created,
            Err(error) => {
                self.runs.release_array(array, frees);
                return Err(error);
            }
        };
        {
            let locked = self
                .runs
                .lock(child, child_gen)
                .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
            self.runs.promote(&locked, slot)?;
        }
        let key = child_key(pos.off, blocks[0].tokens_hash);
        let parent = self
            .runs
            .header(pos.run)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        let outcome = match self.runs.claim_child(parent, snap, key, (child, child_gen)) {
            super::runs::Claim::Claimed => Inserted::Claimed,
            super::runs::Claim::Exists(id, generation) => Inserted::Exists(id, generation),
            super::runs::Claim::Replan => Inserted::Replan,
            super::runs::Claim::Locked => {
                let Some(locked) = self.runs.lock(pos.run, pos.generation) else {
                    self.runs.discard(child, child_gen, frees);
                    return Ok(Placed::Restart);
                };
                self.runs
                    .insert_child_locked(&locked, pos.off, key, (child, child_gen), frees)?
            }
        };
        if let Inserted::Exists(id, generation) = outcome {
            bump(&self.runs.stats.claims_exists);
            self.runs.discard(child, child_gen, frees);
            *pos = Pos {
                run: id,
                generation,
                off: 0,
            };
            return Ok(Placed::Advanced(0));
        }
        if matches!(outcome, Inserted::Replan) {
            self.runs.discard(child, child_gen, frees);
            return Ok(Placed::Replan);
        }
        self.record(st, blocks, child, 0, child_gen);
        *changed = true;
        *pos = Pos {
            run: child,
            generation: child_gen,
            off: take as u32,
        };
        Ok(Placed::Advanced(take))
    }

    fn sole_whole_locked(&self, locked: &Locked<'_>, slot: Slot) -> bool {
        let snap = locked.snap();
        sole_whole_in_attempt(self, locked.h, &snap, slot) == Some(true)
    }

    // ------------------------------------------------------------------
    // Splits (7.5)
    // ------------------------------------------------------------------

    /// Splits a locked run at its median cutoff, in one step that waits out claims
    /// because children move. Partial holders reaching the split point are promoted inside
    /// the step (fix 1). All allocation happens before the run changes.
    pub(crate) fn split_median(
        &self,
        locked: &Locked<'_>,
        frees: &mut Frees,
    ) -> Result<(), KvCacheEventError> {
        let runs = &self.runs;
        let step = locked.h.step_excluding_claims();
        let snap = locked.snap();
        let mut cuts = runs.cutoffs(&snap);
        cuts.sort_unstable_by_key(|c| c.cutoff);
        let Some(&Cutoff { cutoff: at, .. }) = cuts.get(cuts.len() / 2) else {
            return Ok(());
        };
        if at == 0 || at >= snap.len {
            return Ok(());
        }
        let columns = runs
            .columns(&snap)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        let split_head = columns.local(at as usize);

        // Children: those past `at` move to the suffix, rebased.
        let mut keep = Vec::new();
        let mut moved = Vec::new();
        for (key, id, generation) in runs.children_of(snap.children) {
            // A linked child cannot die while this thread holds its parent, so its start
            // and head are stable; reading them takes no child lock.
            let Some(child) = runs.header(id) else {
                keep.push((key, id, generation));
                continue;
            };
            let offset = child
                .start
                .load(std::sync::atomic::Ordering::Relaxed)
                .wrapping_sub(snap.start);
            if offset > at && offset <= snap.len {
                let head = LocalBlockHash(child.head.load(std::sync::atomic::Ordering::Relaxed));
                moved.push((child_key(offset - at, head), id, generation));
            } else {
                keep.push((key, id, generation));
            }
        }
        let whole = runs.whole_slots(locked);
        let suffix_cuts: Vec<Cutoff> = cuts
            .iter()
            .filter(|c| c.cutoff > at)
            .map(|c| Cutoff {
                slot: c.slot,
                cutoff: c.cutoff - at,
            })
            .collect();
        let needs_suffix = !whole.is_empty() || !suffix_cuts.is_empty() || !moved.is_empty();

        let suffix = if needs_suffix {
            runs.retain_array(snap.array);
            let created = runs.create(
                locked.id,
                snap.start + at,
                split_head,
                snap.array,
                snap.base + at,
                snap.len - at,
            );
            let (suffix, suffix_gen) = match created {
                Ok(created) => created,
                Err(error) => {
                    runs.release_array(snap.array, frees);
                    return Err(error);
                }
            };
            let built = (|| {
                let s = runs
                    .lock(suffix, suffix_gen)
                    .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
                for &slot in &whole {
                    runs.promote(&s, Slot::from_index(slot))?;
                }
                if !suffix_cuts.is_empty() {
                    let table = runs.new_cutoff_table(&suffix_cuts)?;
                    s.h.cutoffs
                        .store(table, std::sync::atomic::Ordering::Relaxed);
                }
                if !moved.is_empty() {
                    let table = runs.new_child_table(&moved)?;
                    s.h.children
                        .store(table, std::sync::atomic::Ordering::Relaxed);
                }
                Ok::<_, KvCacheEventError>(())
            })();
            if let Err(error) = built {
                runs.discard(suffix, suffix_gen, frees);
                return Err(error);
            }
            keep.push((child_key(at, split_head), suffix, suffix_gen));
            Some((suffix, suffix_gen))
        } else {
            None
        };
        let new_children = if keep.is_empty() {
            super::arena::NONE
        } else {
            match runs.new_child_table(&keep) {
                Ok(table) => table,
                Err(error) => {
                    if let Some((suffix, suffix_gen)) = suffix {
                        runs.discard(suffix, suffix_gen, frees);
                    }
                    return Err(error);
                }
            }
        };

        // Mutate the prefix inside the open step.
        for &(_, id, _) in &moved {
            if let (Some((suffix, _)), Some(child)) = (suffix, runs.header(id)) {
                child
                    .parent
                    .store(suffix, std::sync::atomic::Ordering::Release);
            }
        }
        for c in cuts.iter().filter(|c| c.cutoff >= at) {
            runs.promote(locked, Slot::from_index(c.slot))?;
        }
        locked
            .h
            .children
            .store(new_children, std::sync::atomic::Ordering::Relaxed);
        locked.h.len.store(at, std::sync::atomic::Ordering::Relaxed);
        locked
            .h
            .flags
            .fetch_or(FLAG_SEALED, std::sync::atomic::Ordering::Relaxed);
        let (suffix_id, suffix_gen) = suffix.unwrap_or((NO_RUN, 0));
        runs.push_forward(locked, (at, suffix_id, suffix_gen), frees)?;
        drop(step);
        runs.release_child_table(snap.children, frees);
        bump(&runs.stats.splits_prefix_cap);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Removed (7.3)
    // ------------------------------------------------------------------

    fn apply_removed(
        &self,
        st: &mut RankState,
        op: KvCacheRemoveData,
        guard: &Guard,
    ) -> Result<(), KvCacheEventError> {
        if !st.has_map {
            return Err(KvCacheEventError::BlockNotFound);
        }
        let Some(slot) = self.slots.table(guard).slot_of(st.rank) else {
            // The rank is being removed; its sweep drops its coverage.
            for &hash in &op.block_hashes {
                self.forget(st, hash);
            }
            return Ok(());
        };
        // Group the resolved positions by the run they resolve to.
        let mut groups: FxHashMap<(RunId, u32), Vec<u32>> = FxHashMap::default();
        for &hash in &op.block_hashes {
            let Some(entry) = st.map.get(hash) else {
                tracing::trace!(?hash, "Block not found during remove; skipping");
                continue;
            };
            if let Some(e) = self.resolve(entry, hash) {
                groups
                    .entry((e.run, e.generation))
                    .or_default()
                    .push(e.offset);
            }
            self.forget(st, hash);
        }
        let mut work: VecDeque<((RunId, u32), Vec<u32>)> = groups.into_iter().collect();
        let mut frees = Frees::default();
        let mut result = Ok(());
        let mut splits = 0;
        while let Some(((run, generation), offsets)) = work.pop_front() {
            match self.remove_group(st, slot, run, generation, offsets, &mut work, &mut frees) {
                Ok(true) => {
                    splits += 1;
                    if splits > 64 {
                        result = Err(KvCacheEventError::IndexerInvariantViolation);
                        break;
                    }
                }
                Ok(false) => {}
                Err(error) => {
                    result = Err(error);
                    break;
                }
            }
        }
        self.runs.flush(&mut frees);
        result
    }

    /// Truncates the rank's holding of one run at the lowest removed offset it holds.
    /// Returns true when the run had to split first and the group was re-queued.
    #[allow(clippy::too_many_arguments)]
    fn remove_group(
        &self,
        st: &mut RankState,
        slot: Slot,
        run: RunId,
        generation: u32,
        offsets: Vec<u32>,
        work: &mut VecDeque<((RunId, u32), Vec<u32>)>,
        frees: &mut Frees,
    ) -> Result<bool, KvCacheEventError> {
        let Some(locked) = self.runs.lock(run, generation) else {
            return Ok(false);
        };
        let snap = locked.snap();
        let mut here = Vec::with_capacity(offsets.len());
        for off in offsets {
            if off < snap.len {
                here.push(off);
                continue;
            }
            // A split landed since resolution: re-forward into the suffix's group.
            if let Some(Some((at, suffix, suffix_gen))) = self.runs.forward_for(snap.forwards, off)
                && suffix != NO_RUN
            {
                match work
                    .iter_mut()
                    .find(|(key, _)| *key == (suffix, suffix_gen))
                {
                    Some((_, group)) => group.push(off - at),
                    None => work.push_back(((suffix, suffix_gen), vec![off - at])),
                }
            }
        }
        let held = self.runs.held_locked(&locked, &snap, slot);
        let Some(cut) = here.iter().copied().filter(|&off| off < held).min() else {
            return Ok(false);
        };
        let whole = held == snap.len;
        if cut == 0 {
            if whole {
                self.runs.clear_whole(&locked, slot);
            } else {
                self.runs.remove_cutoff(&locked, slot);
            }
        } else if whole {
            if self.runs.cutoff_live(&snap) >= PARTIAL_CAP {
                self.split_median(&locked, frees)?;
                drop(locked);
                work.push_back(((run, generation), here));
                return Ok(true);
            }
            // Demoting a whole holder to a partial one changes two words: a step (W4).
            let step = locked.h.step();
            self.runs.set_cutoff(&locked, slot, cut, frees, true)?;
            self.runs.clear_whole(&locked, slot);
            drop(step);
        } else {
            // Lowering a cutoff in place is one word (W5).
            self.runs.set_cutoff(&locked, slot, cut, frees, false)?;
        }
        // Scrub entries that name the positions the rank no longer holds.
        let columns = self
            .runs
            .columns(&snap)
            .ok_or(KvCacheEventError::IndexerInvariantViolation)?;
        let mut indirect = Vec::new();
        for p in cut..held {
            let key = columns.ext(p as usize);
            match st.map.get(key) {
                Some(e) if e.run == run && e.generation == generation && e.offset == p => {
                    st.map.remove(key);
                    self.release_hash(st.rank, key);
                }
                Some(e) if e.run != run || e.generation != generation => indirect.push((key, e, p)),
                _ => {}
            }
        }
        let reclaim = self.reclaimable(&locked);
        drop(locked);
        // Entries naming an older position are resolved without a run lock held.
        for (key, entry, p) in indirect {
            if self.resolve(entry, key)
                == Some(Entry {
                    run,
                    offset: p,
                    generation,
                })
            {
                st.map.remove(key);
                self.release_hash(st.rank, key);
            }
        }
        if reclaim {
            self.reclaim(run, generation, frees);
        }
        Ok(false)
    }

    // ------------------------------------------------------------------
    // Cleared, rank and worker removal (7.4)
    // ------------------------------------------------------------------

    /// Drops every holding of the rank, found through its map (S4) and the forwarding
    /// records of the runs it names, then erases the map. The rank keeps its slot.
    pub(crate) fn clear_rank(&self, st: &mut RankState) {
        let mut frees = Frees::default();
        {
            let guard = crossbeam_epoch::pin();
            if let Some(slot) = self.slots.table(&guard).slot_of(st.rank) {
                let mut work: Vec<(RunId, u32)> =
                    st.map.iter().map(|(_, e)| (e.run, e.generation)).collect();
                work.sort_unstable();
                work.dedup();
                let mut seen: FxHashSet<(RunId, u32)> = FxHashSet::default();
                while let Some((run, generation)) = work.pop() {
                    if !seen.insert((run, generation)) {
                        continue;
                    }
                    let Some(locked) = self.runs.lock(run, generation) else {
                        continue;
                    };
                    let snap = locked.snap();
                    for (_, suffix, suffix_gen) in self.runs.forwards(snap.forwards) {
                        if suffix != NO_RUN {
                            work.push((suffix, suffix_gen));
                        }
                    }
                    let dropped = self.runs.clear_whole(&locked, slot)
                        | self.runs.remove_cutoff(&locked, slot);
                    let reclaim = dropped && self.reclaimable(&locked);
                    drop(locked);
                    if reclaim {
                        self.reclaim(run, generation, &mut frees);
                    }
                }
            }
        }
        if self.lifecycle.is_enabled() {
            for (hash, _) in st.map.iter() {
                self.lifecycle.remove(st.rank, hash);
            }
        }
        st.map = RankMap::default();
        st.has_map = false;
        self.runs.flush(&mut frees);
    }

    /// `RemoveWorkerDpRank`: the `Cleared` work, then CRTC's four-step slot release with a
    /// verification sweep.
    pub(crate) fn remove_rank(&self, st: &mut RankState) {
        self.clear_rank(st);
        self.release_slots(RemovalTarget::Rank(st.rank));
    }

    /// Unmaps `target`'s slots, waits for events that resolved them, sweeps them out of
    /// every reachable run, and releases them through the epoch.
    pub(crate) fn release_slots(&self, target: RemovalTarget) {
        let slots = self.slots.unmap(target);
        if !slots.is_empty() {
            wait_for_pinned_threads();
            let residue = self.sweep_slots(&slots);
            self.runs
                .stats
                .sweep_residue
                .fetch_add(residue, std::sync::atomic::Ordering::Relaxed);
            self.slots.release(slots);
        }
        self.slots.wait_for_release(target);
    }

    /// Clears `slots` from every run reachable from the root, each under its lock.
    /// Returns how many coverage entries it found (expected 0 after a clear).
    pub(crate) fn sweep_slots(&self, slots: &[Slot]) -> u64 {
        let mut residue = 0;
        let mut frees = Frees::default();
        let mut queue = VecDeque::from([(ROOT, ROOT_GEN)]);
        while let Some((run, generation)) = queue.pop_front() {
            let Some(locked) = self.runs.lock(run, generation) else {
                continue;
            };
            let snap = locked.snap();
            let mut dropped = false;
            for &slot in slots {
                if self.runs.clear_whole(&locked, slot) | self.runs.remove_cutoff(&locked, slot) {
                    residue += 1;
                    dropped = true;
                }
            }
            queue.extend(
                self.runs
                    .children_of(snap.children)
                    .into_iter()
                    .map(|(_, id, generation)| (id, generation)),
            );
            let reclaim = dropped && run != ROOT && self.reclaimable(&locked);
            drop(locked);
            if reclaim {
                self.reclaim(run, generation, &mut frees);
            }
        }
        self.runs.flush(&mut frees);
        residue
    }

    // ------------------------------------------------------------------
    // Reclamation (10)
    // ------------------------------------------------------------------

    fn reclaimable(&self, locked: &Locked<'_>) -> bool {
        let snap = locked.snap();
        locked.id != ROOT
            && snap.flags & super::protocol::FLAG_ANCHOR == 0
            && self.runs.child_live(snap.children) == 0
            && self.runs.holderless(locked, &snap)
    }

    /// Unlinks a holder-less, childless run and cascades upward. Holding the run's lock it
    /// takes the parent's (child, then parent), re-checks the parent link, kills the run
    /// (which a claim landing meanwhile aborts), and tombstones the parent's slot by key.
    pub(crate) fn reclaim(&self, mut run: RunId, mut generation: u32, frees: &mut Frees) {
        loop {
            let Some(child) = self.runs.lock(run, generation) else {
                return;
            };
            if !self.reclaimable(&child) {
                return;
            }
            let snap = child.snap();
            let Some(parent) = self.lock_parent(&child) else {
                return;
            };
            let parent_snap = parent.snap();
            let head = LocalBlockHash(child.h.head.load(std::sync::atomic::Ordering::Relaxed));
            let key = child_key(snap.start.wrapping_sub(parent_snap.start), head);
            if !self.runs.kill(&child, frees) {
                return;
            }
            if !self.runs.unlink_child(&parent, key, run, generation) {
                tracing::error!(
                    run,
                    generation,
                    "arena index: unlinked run was not in its parent's table"
                );
            }
            bump(&self.runs.stats.unlinks);
            let cascade = self
                .reclaimable(&parent)
                .then_some((parent.id, parent.generation));
            drop(parent);
            drop(child);
            let Some(next) = cascade else {
                return;
            };
            (run, generation) = next;
        }
    }

    /// Locks the parent of a locked child, re-checking the link: a split of the parent
    /// may have moved the child to the suffix meanwhile.
    fn lock_parent<'a>(&'a self, child: &Locked<'a>) -> Option<Locked<'a>> {
        for _ in 0..64 {
            let parent_id = child.h.parent.load(std::sync::atomic::Ordering::Acquire);
            let parent = self.runs.lock_current(parent_id)?;
            if child.h.parent.load(std::sync::atomic::Ordering::Acquire) == parent_id {
                return Some(parent);
            }
        }
        None
    }

    /// The verification sweep of the five-minute maintenance task (10): unlinks any
    /// holder-less leaf an earlier panic or bug left behind, and counts it.
    pub(crate) fn sweep_holderless(&self) -> u64 {
        let mut found = 0;
        let mut frees = Frees::default();
        let mut order = Vec::new();
        let mut queue = VecDeque::from([(ROOT, ROOT_GEN)]);
        while let Some((run, generation)) = queue.pop_front() {
            let Read::Ok(children) = self.runs.read(run, generation, |_, snap| {
                Some(self.runs.children_of(snap.children))
            }) else {
                continue;
            };
            order.push((run, generation));
            queue.extend(
                children
                    .into_iter()
                    .map(|(_, id, generation)| (id, generation)),
            );
        }
        for (run, generation) in order.into_iter().rev() {
            let reclaim = run != ROOT
                && self
                    .runs
                    .lock(run, generation)
                    .is_some_and(|locked| self.reclaimable(&locked));
            if reclaim {
                found += 1;
                self.reclaim(run, generation, &mut frees);
            }
        }
        self.runs.flush(&mut frees);
        found
    }
}

/// What `slot` holds, inside a read attempt: `len` if whole, else its largest entry.
fn held_in_attempt(index: &ArenaIndex, h: &RunHeader, snap: &Snap, slot: Slot) -> Option<u32> {
    let (word, bit) = slot.word_and_bit();
    let whole = |index: &ArenaIndex| -> Option<bool> {
        if word < INLINE_WORDS {
            return Some(h.whole[word].load(std::sync::atomic::Ordering::Relaxed) & bit != 0);
        }
        let mut found = false;
        index
            .runs
            .for_each_whole_word(h, snap, word + 1, |w, bits| {
                found |= w == word && bits & bit != 0
            })?;
        Some(found)
    };
    if whole(index)? {
        return Some(snap.len);
    }
    let mut cutoff = 0;
    index.runs.for_each_cutoff(snap.cutoffs, |c| {
        if c.slot == slot.index() {
            cutoff = cutoff.max(c.cutoff);
        }
    })?;
    if whole(index)? {
        return Some(snap.len);
    }
    Some(cutoff)
}

/// Whether `slot` is the run's only whole holder, inside a read attempt.
fn sole_whole_in_attempt(
    index: &ArenaIndex,
    h: &RunHeader,
    snap: &Snap,
    slot: Slot,
) -> Option<bool> {
    let (word, bit) = slot.word_and_bit();
    let mut mine = false;
    let mut others = false;
    index
        .runs
        .for_each_whole_word(h, snap, usize::MAX >> 8, |w, bits| {
            if w == word {
                mine |= bits & bit != 0;
                others |= bits & !bit != 0;
            } else {
                others |= bits != 0;
            }
        })?;
    Some(mine && !others)
}
