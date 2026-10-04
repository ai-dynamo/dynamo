// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An event lane's block lookups and the node references they share.
//!
//! Every block entry names the node that holds its block. If each entry owned an `Arc`,
//! every stored or removed block would be an atomic read-modify-write on that node's
//! strong count, and shared prefix nodes are named from every lane at once. Instead the
//! lane holds one `Arc` per distinct node its entries name and counts those entries in
//! lane-local memory, so only the first entry naming a node and the last one leaving it
//! touch the node.
//!
//! Stale-leaf cleanup keeps working on strong counts: a node any entry of a lane names
//! carries exactly one extra strong reference from that lane.

use std::collections::hash_map::Entry;
use std::num::NonZeroU32;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use super::block_lookup::BlockLookup;
use super::types::SharedNode;
use crate::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

/// A lane-local name for a node the lane holds. Non-zero so a lookup slot holding a key
/// and an id stays 16 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct NodeId(NonZeroU32);

const _: () = assert!(size_of::<Option<(ExternalSequenceBlockHash, NodeId)>>() == 16);

impl NodeId {
    fn from_index(index: usize) -> Self {
        let id = u32::try_from(index + 1).expect("lane holds more than u32::MAX - 1 nodes");
        Self(NonZeroU32::new(id).expect("index + 1 is non-zero"))
    }

    fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

struct NodeHold {
    node: SharedNode,
    /// Block entries naming this node, across all of the lane's ranks.
    refs: usize,
}

/// One `Arc` per node the lane's block entries name, with the number of entries naming
/// it. Between `LaneLookup` operations a hold exists exactly while an entry names its
/// node. Node addresses identify holds: the hold's `Arc` keeps its node allocated, so no
/// other node can take the address while the mapping exists.
#[derive(Default)]
struct NodeHolds {
    holds: Vec<Option<NodeHold>>,
    free: Vec<NodeId>,
    ids: FxHashMap<usize, NodeId>,
}

fn address(node: &SharedNode) -> usize {
    Arc::as_ptr(node) as usize
}

impl NodeHolds {
    fn id_of(&self, node: &SharedNode) -> Option<NodeId> {
        self.ids.get(&address(node)).copied()
    }

    /// The id of `node`, holding it with no entries if it is not held yet. The caller
    /// settles the count with `add_refs` before its operation returns.
    fn hold(&mut self, node: &SharedNode) -> NodeId {
        let vacant = match self.ids.entry(address(node)) {
            Entry::Occupied(occupied) => return *occupied.get(),
            Entry::Vacant(vacant) => vacant,
        };
        let hold = Some(NodeHold {
            node: node.clone(),
            refs: 0,
        });
        let id = match self.free.pop() {
            Some(id) => {
                self.holds[id.index()] = hold;
                id
            }
            None => {
                self.holds.push(hold);
                NodeId::from_index(self.holds.len() - 1)
            }
        };
        *vacant.insert(id)
    }

    fn node(&self, id: NodeId) -> &SharedNode {
        &self.holds[id.index()]
            .as_ref()
            .expect("lookup entry names a released node id")
            .node
    }

    fn refs_mut(&mut self, id: NodeId) -> &mut usize {
        &mut self.holds[id.index()]
            .as_mut()
            .expect("lookup entry names a released node id")
            .refs
    }

    /// Counts `n` more entries naming `id`, releasing the hold if none name it.
    fn add_refs(&mut self, id: NodeId, n: usize) {
        let refs = self.refs_mut(id);
        *refs += n;
        if *refs == 0 {
            self.release(id);
        }
    }

    /// Counts `n` fewer entries naming `id`, releasing the hold once none name it.
    fn drop_refs(&mut self, id: NodeId, n: usize) {
        let refs = self.refs_mut(id);
        debug_assert!(*refs >= n, "dropping {n} of {refs} entries");
        *refs -= n;
        if *refs == 0 {
            self.release(id);
        }
    }

    fn release(&mut self, id: NodeId) {
        let hold = self.holds[id.index()]
            .take()
            .expect("released node id twice");
        self.ids.remove(&address(&hold.node));
        self.free.push(id);
    }
}

/// One event lane's block lookups: for each rank on the lane, the node holding each
/// stored block hash.
#[derive(Default)]
pub(super) struct LaneLookup {
    workers: FxHashMap<WorkerWithDpRank, BlockLookup<NodeId>>,
    nodes: NodeHolds,
}

impl LaneLookup {
    pub(super) fn contains_worker(&self, worker: WorkerWithDpRank) -> bool {
        self.workers.contains_key(&worker)
    }

    pub(super) fn add_worker(&mut self, worker: WorkerWithDpRank) {
        self.workers.entry(worker).or_default();
    }

    /// The lane's ranks, in the order `redirect` visits them.
    pub(super) fn workers(&self) -> impl Iterator<Item = WorkerWithDpRank> + '_ {
        self.workers.keys().copied()
    }

    pub(super) fn block_counts(&self) -> impl Iterator<Item = (WorkerWithDpRank, usize)> + '_ {
        self.workers
            .iter()
            .map(|(&worker, blocks)| (worker, blocks.len()))
    }

    pub(super) fn contains(
        &self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
    ) -> bool {
        self.workers
            .get(&worker)
            .is_some_and(|blocks| blocks.contains_key(&hash))
    }

    /// The node `worker`'s entry names for `hash`, without validating it. The entry can
    /// be stale after a cross-thread split; callers detect that in their own locked
    /// operation on the node and repair it.
    pub(super) fn node(
        &self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
    ) -> Option<SharedNode> {
        let &id = self.workers.get(&worker)?.get(&hash)?;
        Some(self.nodes.node(id).clone())
    }

    pub(super) fn names(&self, node: &SharedNode) -> bool {
        self.nodes.id_of(node).is_some()
    }

    /// Points `worker`'s entries for `hashes` at `node`. Returns the number of entries
    /// inserted or changed.
    pub(super) fn upsert_all<I>(
        &mut self,
        worker: WorkerWithDpRank,
        hashes: I,
        node: &SharedNode,
    ) -> usize
    where
        I: Iterator<Item = ExternalSequenceBlockHash> + Clone,
    {
        let Self { workers, nodes } = self;
        let id = nodes.hold(node);
        let changed = workers
            .entry(worker)
            .or_default()
            .upsert_all(hashes, id, |replaced| nodes.drop_refs(replaced, 1));
        nodes.add_refs(id, changed);
        changed
    }

    pub(super) fn insert(
        &mut self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
        node: &SharedNode,
    ) {
        let id = self.nodes.hold(node);
        let replaced = self.workers.entry(worker).or_default().insert(hash, id);
        if replaced == Some(id) {
            return;
        }
        self.nodes.add_refs(id, 1);
        if let Some(replaced) = replaced {
            self.nodes.drop_refs(replaced, 1);
        }
    }

    pub(super) fn remove(&mut self, worker: WorkerWithDpRank, hash: ExternalSequenceBlockHash) {
        let Some(id) = self
            .workers
            .get_mut(&worker)
            .and_then(|blocks| blocks.remove(&hash))
        else {
            return;
        };
        self.nodes.drop_refs(id, 1);
    }

    /// Removes `worker`'s entries for `hashes`, calling `on_removed` for each hash,
    /// present or not. Does nothing if the lane has no lookup for `worker`.
    pub(super) fn remove_all(
        &mut self,
        worker: WorkerWithDpRank,
        hashes: &[ExternalSequenceBlockHash],
        mut on_removed: impl FnMut(ExternalSequenceBlockHash),
    ) {
        let Some(blocks) = self.workers.get_mut(&worker) else {
            return;
        };
        let nodes = &mut self.nodes;
        blocks.remove_all(hashes.iter().copied(), |hash, removed| {
            if let Some(id) = removed {
                nodes.drop_refs(id, 1);
            }
            on_removed(hash);
        });
    }

    /// For each rank, in `workers` order, points its entries among `hashes(rank)` that
    /// still name `from` at `to`. Entries naming any other node stay put. Returns the
    /// number of entries changed.
    pub(super) fn redirect<I>(
        &mut self,
        from: &SharedNode,
        to: &SharedNode,
        mut hashes: impl FnMut(WorkerWithDpRank) -> I,
    ) -> usize
    where
        I: IntoIterator<Item = ExternalSequenceBlockHash>,
    {
        let Some(from_id) = self.nodes.id_of(from) else {
            return 0;
        };
        let to_id = self.nodes.hold(to);
        let mut changed = 0;
        for (&worker, blocks) in &mut self.workers {
            changed += blocks.redirect(hashes(worker), from_id, to_id);
        }
        self.nodes.add_refs(to_id, changed);
        self.nodes.drop_refs(from_id, changed);
        changed
    }

    /// Drops the lookups of every rank `target` selects, calling `on_removed` for each
    /// of their entries.
    pub(super) fn remove_workers(
        &mut self,
        mut target: impl FnMut(WorkerWithDpRank) -> bool,
        mut on_removed: impl FnMut(WorkerWithDpRank, ExternalSequenceBlockHash),
    ) {
        let nodes = &mut self.nodes;
        self.workers.retain(|&worker, blocks| {
            if !target(worker) {
                return true;
            }
            for (&hash, &id) in blocks.iter() {
                nodes.drop_refs(id, 1);
                on_removed(worker, hash);
            }
            false
        });
    }

    #[cfg(test)]
    pub(super) fn block_count(&self, worker: WorkerWithDpRank) -> Option<usize> {
        self.workers.get(&worker).map(BlockLookup::len)
    }

    /// Checks that the lane holds exactly one `Arc` per node its entries name, counts
    /// every entry naming it, and recycles exactly the ids it does not hold.
    #[cfg(test)]
    pub(super) fn assert_invariants(&self) {
        let mut refs: FxHashMap<NodeId, usize> = FxHashMap::default();
        for blocks in self.workers.values() {
            for (_, &id) in blocks.iter() {
                *refs.entry(id).or_default() += 1;
            }
        }

        let nodes = &self.nodes;
        let mut held = 0;
        let mut empty = Vec::new();
        for (index, hold) in nodes.holds.iter().enumerate() {
            let id = NodeId::from_index(index);
            let Some(hold) = hold else {
                empty.push(id.index());
                continue;
            };
            held += 1;
            assert_eq!(
                refs.remove(&id),
                Some(hold.refs),
                "hold {id:?} counts {} entries",
                hold.refs
            );
            assert_eq!(
                nodes.ids.get(&address(&hold.node)),
                Some(&id),
                "hold {id:?} is not mapped by its node's address"
            );
        }
        assert!(refs.is_empty(), "entries name released ids: {refs:?}");
        // With every held node mapped to its own id, equal sizes make the address map a
        // bijection, so no node is held twice.
        assert_eq!(nodes.ids.len(), held, "address map outlives its holds");

        let mut free: Vec<_> = nodes.free.iter().map(|id| id.index()).collect();
        free.sort_unstable();
        assert_eq!(free, empty, "free list differs from the released ids");
    }
}

#[cfg(test)]
mod tests {
    use super::super::node::Node;
    use super::*;
    use std::collections::HashMap;

    fn hash(k: u64) -> ExternalSequenceBlockHash {
        ExternalSequenceBlockHash(k)
    }

    /// Randomized differential test of every entry-changing operation against a map from
    /// (rank, hash) to a node. Besides the lane's own bookkeeping, it checks the contract
    /// stale-leaf cleanup relies on: each node's strong count is one for the test's pool
    /// plus one exactly while some entry names it. Nodes nothing names are replaced from
    /// time to time, so their freed addresses can come back as different nodes.
    #[test]
    fn matches_model_and_holds_one_arc_per_named_node() {
        for seed in 0..8u64 {
            let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            };
            let workers: Vec<_> = (0..3).map(|id| WorkerWithDpRank::new(id, 0)).collect();
            let mut pool: Vec<SharedNode> = (0..6).map(|_| Arc::new(Node::new())).collect();
            let mut lane = LaneLookup::default();
            let mut model: HashMap<(WorkerWithDpRank, ExternalSequenceBlockHash), usize> =
                HashMap::new();
            let key_space = 4 + seed * 5;

            for _ in 0..5_000 {
                let worker = workers[(next() % workers.len() as u64) as usize];
                let node = (next() % pool.len() as u64) as usize;
                let hashes: Vec<_> = (0..(next() % 8))
                    .map(|_| hash(next() % key_space))
                    .collect();
                match next() % 12 {
                    0..=3 => {
                        let changed = lane.upsert_all(worker, hashes.iter().copied(), &pool[node]);
                        let expected = hashes
                            .iter()
                            .filter(|&&h| model.insert((worker, h), node) != Some(node))
                            .count();
                        assert_eq!(changed, expected);
                    }
                    4 => {
                        let h = hash(next() % key_space);
                        lane.insert(worker, h, &pool[node]);
                        model.insert((worker, h), node);
                    }
                    5 => {
                        let h = hash(next() % key_space);
                        lane.remove(worker, h);
                        model.remove(&(worker, h));
                    }
                    6..=8 => {
                        let mut seen = Vec::new();
                        lane.remove_all(worker, &hashes, |h| seen.push(h));
                        if lane.contains_worker(worker) {
                            assert_eq!(seen, hashes);
                        }
                        for &h in &hashes {
                            model.remove(&(worker, h));
                        }
                    }
                    9 | 10 => {
                        let from = (next() % pool.len() as u64) as usize;
                        let changed = lane.redirect(&pool[from], &pool[node], |_| hashes.clone());
                        let mut expected = 0;
                        if from != node {
                            for &w in &workers {
                                for &h in &hashes {
                                    if let Some(entry) = model.get_mut(&(w, h))
                                        && *entry == from
                                    {
                                        *entry = node;
                                        expected += 1;
                                    }
                                }
                            }
                        }
                        assert_eq!(changed, expected);
                    }
                    _ => {
                        let mut removed = Vec::new();
                        lane.remove_workers(|w| w == worker, |w, h| removed.push((w, h)));
                        let mut expected: Vec<_> = model
                            .keys()
                            .filter(|(w, _)| *w == worker)
                            .copied()
                            .collect();
                        model.retain(|(w, _), _| *w != worker);
                        removed.sort_by_key(|(_, h)| h.0);
                        expected.sort_by_key(|(_, h)| h.0);
                        assert_eq!(removed, expected);
                    }
                }

                lane.assert_invariants();
                for &w in &workers {
                    for k in 0..key_space {
                        let expected = model.get(&(w, hash(k))).map(|&n| &pool[n]);
                        let got = lane.node(w, hash(k));
                        assert_eq!(got.is_some(), expected.is_some());
                        if let (Some(got), Some(expected)) = (got, expected) {
                            assert!(Arc::ptr_eq(&got, expected));
                        }
                    }
                }
                for (n, node) in pool.iter().enumerate() {
                    let named = model.values().any(|&m| m == n);
                    assert_eq!(Arc::strong_count(node), 1 + usize::from(named));
                    assert_eq!(lane.names(node), named);
                }

                let replace = (next() % pool.len() as u64) as usize;
                if !model.values().any(|&m| m == replace) {
                    pool[replace] = Arc::new(Node::new());
                }
            }

            drop(lane);
            assert!(pool.iter().all(|node| Arc::strong_count(node) == 1));
        }
    }
}
