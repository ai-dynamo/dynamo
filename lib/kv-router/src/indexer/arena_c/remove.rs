// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Removed` and `Cleared` events, rank and worker removal, and the slot sweep.
//!
//! Removal resolves each evicted hash's own entry and groups the positions by the run
//! they resolve to, so a stale entry can never truncate a run the hash is not in (the
//! grouped-removal overcount CRTC needs a separate fix for). Clears are map-based: every
//! position a rank holds has an entry on its sticky lane (S4), so visiting the runs its
//! entries name, plus the suffixes their forwarding records reach, finds all of its
//! coverage.

use parking_lot::RwLockUpgradableReadGuard;
use rustc_hash::FxHashSet;

use super::slots::wait_for_pinned_threads;
use super::store::Resolved;
use super::types::{BlockPos, WorkerRemovalTarget};
use super::*;

/// Positions to truncate, grouped by run.
type Groups = Vec<(u32, Vec<u32>)>;

fn push_group(groups: &mut Groups, run: u32, offset: u32) {
    match groups.iter_mut().find(|(id, _)| *id == run) {
        Some((_, offsets)) => offsets.push(offset),
        None => groups.push((run, vec![offset])),
    }
}

impl ArenaIndexC {
    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn apply_removed(
        &self,
        lane: &mut CLane,
        rank: WorkerWithDpRank,
        op: KvCacheRemoveData,
        id: u64,
    ) -> Result<(), KvCacheEventError> {
        let CLane {
            ranks,
            free,
            pending,
            tally,
        } = lane;
        let Some(lookup) = ranks.get_mut(&rank) else {
            return Err(KvCacheEventError::BlockNotFound);
        };
        let guard = crossbeam_epoch::pin();
        let Some(slot) = self.slots.table(&guard).slot_of(rank) else {
            // The rank is being removed: the sweep that releases its slot drops its
            // coverage. Only the entries are left to scrub.
            for &hash in &op.block_hashes {
                lookup.map.remove(hash);
            }
            return Ok(());
        };

        let mut groups = Groups::new();
        for &hash in &op.block_hashes {
            let Some(entry) = lookup.map.remove(hash) else {
                tracing::debug!(
                    worker_id = rank.worker_id.to_string(),
                    dp_rank = rank.dp_rank,
                    id,
                    block_hash = ?hash,
                    "Block not found during remove; skipping"
                );
                continue;
            };
            if let Resolved::At { pos, .. } = self.resolve(hash, entry, None) {
                push_group(&mut groups, pos.run(), pos.offset);
            }
        }

        let mut empty = Vec::new();
        while let Some((run, offsets)) = groups.pop() {
            self.truncate(lookup, slot, run, offsets, &mut groups, free, &mut empty)?;
        }
        for (run, generation) in empty {
            self.try_unlink(run, generation, free, pending, tally);
        }
        drop(guard);
        Ok(())
    }

    /// Truncates `slot`'s holding in `run_id` at the lowest of `offsets` it holds, then
    /// scrubs the entries of the positions it no longer holds. Offsets a split moved are
    /// queued again under their suffix.
    #[allow(clippy::too_many_arguments)]
    fn truncate(
        &self,
        lookup: &mut RankLookup,
        slot: Slot,
        run_id: u32,
        mut offsets: Vec<u32>,
        groups: &mut Groups,
        free: &mut FreeBatch,
        empty: &mut Vec<(u32, u32)>,
    ) -> Result<(), KvCacheEventError> {
        let store = &self.store;
        let run = store.run(run_id);
        let gate = run.gate.write();
        let state = run.state.upgradable_read();
        if run.is_dead() {
            return Ok(());
        }
        let len = run.len();
        offsets.retain(|&offset| {
            if offset < len {
                return true;
            }
            if let Some(forward) = store.forward_for(run, offset)
                && store.run(forward.suffix).generation.load(Ordering::Relaxed)
                    == forward.generation
            {
                push_group(groups, forward.suffix, offset - forward.at);
            }
            false
        });
        let held = store.held(run, slot);
        let Some(cut) = offsets.iter().copied().filter(|&o| o < held).min() else {
            return Ok(());
        };
        let whole = store.whole_contains(run, slot);
        let window = store.window(run);
        let scrub: Vec<(u32, u64)> = (cut..held).map(|p| (p, window.ext(p as usize))).collect();

        if cut == 0 {
            if whole {
                // Dropping a whole bit needs no state write lock under the exclusive gate.
                store.whole_clear(run, slot);
            } else {
                let _state = RwLockUpgradableReadGuard::upgrade(state);
                store.cutoff_remove(run, slot, free, &self.store);
            }
        } else {
            let _state = RwLockUpgradableReadGuard::upgrade(state);
            if !store.cutoff_set(run, slot, cut, free, &self.store)? {
                // Demoting would pass the partial-holder cap: split, then redo.
                self.split_at_median(run_id, free)?;
                for offset in offsets {
                    push_group(groups, run_id, offset);
                }
                return Ok(());
            }
            // The cutoff is published before the bit clears.
            if whole {
                store.whole_clear(run, slot);
            }
        }
        if run_id != ROOT && !store.has_holders(run) && !store.has_live_children(run) {
            empty.push((run_id, run.generation.load(Ordering::Relaxed)));
        }
        drop(gate);

        for (p, key) in scrub {
            let key = ExternalSequenceBlockHash(key);
            let Some(entry) = lookup.map.get(key) else {
                continue;
            };
            if entry == BlockPos::new(run_id, p) {
                lookup.map.remove(key);
                continue;
            }
            // An entry naming another run is dropped unless it still resolves to a
            // position the rank holds, which only an H1 violation allows.
            match self.resolve(key, entry, Some(slot)) {
                Resolved::At { pos, held } if pos.offset < held => {}
                _ => {
                    lookup.map.remove(key);
                }
            }
        }
        Ok(())
    }

    /// Applies `Cleared` for `rank` on this lane: drops its coverage through its map and
    /// erases the map. The rank keeps its slot. Does not wait for removals on other lanes;
    /// callers follow with `wait_for_release` once unpinned.
    pub(super) fn clear_rank(&self, lane: &mut CLane, rank: WorkerWithDpRank) {
        let Some(lookup) = lane.ranks.remove(&rank) else {
            return;
        };
        let mut guard = crossbeam_epoch::pin();
        let Some(slot) = self.slots.table(&guard).slot_of(rank) else {
            return;
        };
        let mut queue: Vec<(u32, Option<u32>)> = lookup
            .map
            .iter()
            .map(|(_, pos)| pos.run())
            .collect::<FxHashSet<_>>()
            .into_iter()
            .map(|run| (run, None))
            .collect();
        drop(lookup);
        let mut visited = FxHashSet::default();
        let mut empty = Vec::new();
        let store = &self.store;
        let mut locked = 0usize;
        while let Some((run_id, generation)) = queue.pop() {
            locked += 1;
            if locked.is_multiple_of(64) {
                // Let the epoch advance. Once a removal on another lane has unmapped the
                // slot, its sweep clears the rest; never touch a slot it may recycle.
                guard.repin();
                if self.slots.table(&guard).slot_of(rank) != Some(slot) {
                    break;
                }
            }
            let run = store.run(run_id);
            let _gate = run.gate.write();
            let _state = run.state.write();
            let current = run.generation.load(Ordering::Relaxed);
            if run.is_dead() || generation.is_some_and(|g| g != current) {
                continue;
            }
            if !visited.insert((run_id, current)) {
                continue;
            }
            for forward in store.forwards(run) {
                queue.push((forward.suffix, Some(forward.generation)));
            }
            let mut changed = store.whole_clear(run, slot);
            changed |= store.cutoff_remove(run, slot, &mut lane.free, &self.store);
            if changed && run_id != ROOT && !store.has_holders(run) && !store.has_live_children(run)
            {
                empty.push((run_id, current));
            }
        }
        for (run, generation) in empty {
            self.try_unlink(
                run,
                generation,
                &mut lane.free,
                &mut lane.pending,
                &mut lane.tally,
            );
        }
        drop(guard);
    }

    /// `Cleared`: the map-based clear, then wait for any removal of this rank on another
    /// lane to finish its sweep.
    pub(super) fn apply_cleared(&self, lane: &mut CLane, rank: WorkerWithDpRank) {
        self.clear_rank(lane, rank);
        self.slots
            .wait_for_release(WorkerRemovalTarget::DpRank(rank));
    }

    /// Removes `target`'s ranks: map-based drops for the ranks on this lane and, with
    /// `sweep_tree`, CRTC's four-step slot release (unmap, wait for pinned threads, sweep
    /// every reachable run, release through the epoch). Returns once any removal of these
    /// ranks on another lane has finished too.
    pub(super) fn remove_ranks(
        &self,
        lane: &mut CLane,
        target: WorkerRemovalTarget,
        sweep_tree: bool,
    ) {
        let ranks: Vec<_> = lane
            .ranks
            .keys()
            .copied()
            .filter(|&rank| target.matches(rank))
            .collect();
        for rank in ranks {
            self.clear_rank(lane, rank);
        }
        if sweep_tree {
            let slots = self.slots.unmap(target);
            if !slots.is_empty() {
                // Events that resolved these slots before the unmap may still be writing
                // bits; let them finish so the sweep sees every bit.
                wait_for_pinned_threads();
                self.sweep_slots(&slots, lane);
                self.slots.release(slots);
            }
        }
        self.slots.wait_for_release(target);
    }

    /// Clears `slots` from every run reachable from ROOT. Repins every 64 runs and
    /// checks each queued run's generation under its lock, since an id read under an
    /// earlier pin may have been reused.
    pub(super) fn sweep_slots(&self, slots: &[Slot], lane: &mut CLane) {
        let store = &self.store;
        let mut guard = crossbeam_epoch::pin();
        let mut queue: Vec<(u32, u32)> = store
            .children_of(store.run(ROOT))
            .into_iter()
            .map(|(_, child, generation)| (child, generation))
            .collect();
        let mut empty = Vec::new();
        let mut visited = 0usize;
        while let Some((run_id, generation)) = queue.pop() {
            visited += 1;
            if visited.is_multiple_of(64) {
                guard.repin();
            }
            let run = store.run(run_id);
            let _gate = run.gate.write();
            let _state = run.state.write();
            if run.is_dead() || run.generation.load(Ordering::Relaxed) != generation {
                continue;
            }
            let mut changed = false;
            for &slot in slots {
                changed |= store.whole_clear(run, slot);
            }
            let swept = store.cutoff_retain(
                run,
                |slot| !slots.contains(&slot),
                &mut lane.free,
                &self.store,
            );
            changed |= swept.unwrap_or_else(|_| {
                // Rebuilding the table needs memory; fall back to removing entries in place.
                slots.iter().fold(false, |any, &slot| {
                    store.cutoff_remove(run, slot, &mut lane.free, &self.store) | any
                })
            });
            queue.extend(
                store
                    .children_of(run)
                    .into_iter()
                    .map(|(_, child, generation)| (child, generation)),
            );
            if changed && !store.has_holders(run) && !store.has_live_children(run) {
                empty.push((run_id, generation));
            }
        }
        for (run, generation) in empty {
            self.try_unlink(
                run,
                generation,
                &mut lane.free,
                &mut lane.pending,
                &mut lane.tally,
            );
        }
        drop(guard);
    }
}
