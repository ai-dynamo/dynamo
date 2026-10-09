// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clean-room reference for the indexer contract, written from Dynamo's event semantics
//! (`KvCacheEventData`, `KvIndexerInterface`) and not from any backend.
//!
//! Each rank holds a set of blocks keyed by sequence hash:
//! - `Stored { parent, blocks }` adds every block.
//! - `Removed { block_hashes }` drops exactly those blocks.
//! - `Cleared`, `remove_worker` and `remove_worker_dp_rank` drop all of a rank's blocks.
//!
//! A query's sequence hashes name one chain, so the most a backend may credit a rank with
//! is the longest query prefix whose blocks the rank holds: the held score. Scoring above
//! it credits a block the rank does not hold, an overcount.
//!
//! A backend may score lower where Dynamo semantics leave holes (see
//! `test_remove_mid_chain_block`): removing a block orphans the rank's blocks below it,
//! and re-storing only the removed block does not bring them back. The reachable set
//! models that: a stored block is reachable when the event's parent is (or the event
//! starts at the root), and removing a block drops every reachable block below it. The
//! reachable score is the least a hole-tolerant backend must report. Scores between the
//! two are hole undercounts; scores below the reachable score are unexplained.

use std::collections::BTreeMap;

use rustc_hash::{FxHashMap, FxHashSet};

use super::driver::Op;
use crate::protocols::{KvCacheEventData, WorkerWithDpRank};

/// One rank's blocks.
#[derive(Clone, Debug, Default)]
pub(super) struct RankState {
    /// Held blocks: sequence hash to parent sequence hash.
    held: FxHashMap<u64, Option<u64>>,
    /// Held blocks whose chain from the root was stored through reachable parents and
    /// never cut since. Always prefix-closed.
    reachable: FxHashSet<u64>,
}

impl RankState {
    pub(super) fn holds(&self, hash: u64) -> bool {
        self.held.contains_key(&hash)
    }

    pub(super) fn is_reachable(&self, hash: u64) -> bool {
        self.reachable.contains(&hash)
    }

    /// Adds `blocks` below `parent`. Returns whether the parent was reachable, that is,
    /// whether a backend has to accept the store.
    fn store(&mut self, parent: Option<u64>, blocks: &[u64]) -> bool {
        let reachable = parent.is_none_or(|p| self.reachable.contains(&p));
        let mut prev = parent;
        for &block in blocks {
            self.held.insert(block, prev);
            if reachable {
                self.reachable.insert(block);
            }
            prev = Some(block);
        }
        reachable
    }

    /// Drops `hashes`. Returns whether any of them was reachable.
    fn remove(&mut self, hashes: &[u64]) -> bool {
        let mut cut = false;
        for hash in hashes {
            self.held.remove(hash);
            cut |= self.reachable.remove(hash);
        }
        if cut {
            self.prune_orphans();
        }
        cut
    }

    fn clear(&mut self) {
        self.held.clear();
        self.reachable.clear();
    }

    /// Keeps a reachable block only while its whole chain to the root still is.
    fn prune_orphans(&mut self) {
        let mut intact: FxHashMap<u64, bool> = FxHashMap::default();
        let mut path = Vec::new();
        for &start in &self.reachable {
            let mut cur = start;
            let verdict = loop {
                if let Some(&known) = intact.get(&cur) {
                    break known;
                }
                if !self.reachable.contains(&cur) {
                    break false;
                }
                path.push(cur);
                // A reachable block is held, and its parent link names its chain.
                match self.held.get(&cur).copied().flatten() {
                    None => break true,
                    Some(parent) => cur = parent,
                }
            };
            for hash in path.drain(..) {
                intact.insert(hash, verdict);
            }
        }
        self.reachable.retain(|hash| intact[hash]);
    }

    /// `(held, reachable)` prefix lengths of `seqs`.
    pub(super) fn prefix(&self, seqs: &[u64]) -> (usize, usize) {
        let held = seqs
            .iter()
            .take_while(|h| self.held.contains_key(h))
            .count();
        let reachable = seqs
            .iter()
            .take_while(|h| self.reachable.contains(h))
            .count();
        (held, reachable)
    }
}

/// What a backend must do with an operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Expect {
    /// The operation must apply.
    Apply,
    /// A hole-tolerant backend may reject it (a store under an orphaned parent).
    MayReject,
}

/// Every rank's blocks under the event stream applied so far.
#[derive(Clone, Debug, Default)]
pub(super) struct Reference {
    ranks: BTreeMap<WorkerWithDpRank, RankState>,
}

impl Reference {
    pub(super) fn rank(&self, rank: WorkerWithDpRank) -> Option<&RankState> {
        self.ranks.get(&rank)
    }

    pub(super) fn ranks(&self) -> impl Iterator<Item = (&WorkerWithDpRank, &RankState)> {
        self.ranks.iter()
    }

    pub(super) fn apply(&mut self, op: &Op) -> Expect {
        match op {
            Op::Event(event) => {
                let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                let state = self.ranks.entry(rank).or_default();
                match &event.event.data {
                    KvCacheEventData::Stored(store) => {
                        let blocks: Vec<u64> = store
                            .blocks
                            .iter()
                            .map(|block| block.block_hash.0)
                            .collect();
                        if state.store(store.parent_hash.map(|hash| hash.0), &blocks) {
                            Expect::Apply
                        } else {
                            Expect::MayReject
                        }
                    }
                    KvCacheEventData::Removed(remove) => {
                        let hashes: Vec<u64> =
                            remove.block_hashes.iter().map(|hash| hash.0).collect();
                        if state.remove(&hashes) {
                            Expect::Apply
                        } else {
                            Expect::MayReject
                        }
                    }
                    KvCacheEventData::Cleared => {
                        state.clear();
                        Expect::Apply
                    }
                }
            }
            Op::RemoveRank(rank) => {
                if let Some(state) = self.ranks.get_mut(rank) {
                    state.clear();
                }
                Expect::Apply
            }
            Op::RemoveWorker(worker_id) => {
                for (_, state) in self
                    .ranks
                    .iter_mut()
                    .filter(|(rank, _)| rank.worker_id == *worker_id)
                {
                    state.clear();
                }
                Expect::Apply
            }
        }
    }

    /// Whether `op` is one an engine could emit now: stores chain from a held parent and
    /// removals name held blocks. The shrinker keeps only valid streams.
    pub(super) fn is_valid(&self, op: &Op) -> bool {
        let Op::Event(event) = op else {
            return true;
        };
        let rank = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let state = self.ranks.get(&rank);
        let holds = |hash: u64| state.is_some_and(|s| s.holds(hash));
        match &event.event.data {
            KvCacheEventData::Stored(store) => {
                !store.blocks.is_empty() && store.parent_hash.is_none_or(|p| holds(p.0))
            }
            KvCacheEventData::Removed(remove) => {
                !remove.block_hashes.is_empty()
                    && remove.block_hashes.iter().all(|hash| holds(hash.0))
            }
            KvCacheEventData::Cleared => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_mid_chain_block_orphans_the_rest() {
        let mut rank = RankState::default();
        assert!(rank.store(None, &[1, 2, 3, 4]));
        rank.remove(&[2]);
        assert_eq!(rank.prefix(&[1, 2, 3, 4]), (1, 1));

        // Re-storing the removed block makes it reachable, not the orphans below it.
        assert!(rank.store(Some(1), &[2]));
        assert_eq!(rank.prefix(&[1, 2, 3, 4]), (4, 2));

        // A store under an orphan holds the block without making it reachable.
        assert!(!rank.store(Some(4), &[5]));
        assert_eq!(rank.prefix(&[1, 2, 3, 4, 5]), (5, 2));

        // Re-storing the orphans through a reachable parent reconnects them.
        assert!(rank.store(Some(2), &[3, 4]));
        assert_eq!(rank.prefix(&[1, 2, 3, 4, 5]), (5, 4));
    }

    #[test]
    fn branches_survive_a_cut_on_a_sibling() {
        let mut rank = RankState::default();
        rank.store(None, &[1, 2, 3]);
        rank.store(Some(2), &[13]);
        rank.remove(&[3]);
        assert_eq!(rank.prefix(&[1, 2, 13]), (3, 3));
        rank.remove(&[1]);
        assert_eq!(rank.prefix(&[1, 2, 13]), (0, 0));
        assert!(rank.holds(13) && !rank.is_reachable(13));
    }
}
