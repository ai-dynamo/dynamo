// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use dashmap::DashMap;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::types::*;
use crate::protocols::*;

#[path = "node_state.rs"]
mod node_state;
use node_state::NodeState;

type NodeChildren = DashMap<LocalBlockHash, SharedNode, FxBuildHasher>;
pub(super) struct RemoveOutcome {
    pub(super) stale_hashes: Vec<ExternalSequenceBlockHash>,
}

fn record_last_matched_hash(
    last_matched_hashes: &mut Option<&mut FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>>,
    worker: WorkerWithDpRank,
    hash: ExternalSequenceBlockHash,
) {
    if let Some(last_matched_hashes) = last_matched_hashes.as_deref_mut() {
        last_matched_hashes.insert(worker, hash);
    }
}

/// A node in the concurrent radix tree.
///
/// The compressed edge and coverage state are protected separately from the
/// child map. The shape gate serializes operations that can move edge positions
/// or reparent children, while allowing ordinary child inserts to proceed under
/// a shared gate.
#[derive(Debug)]
pub(super) struct Node {
    shape_gate: RwLock<()>,
    shape_version: AtomicU64,
    /// Sticky logical-internal marker. Once true, this node is treated as
    /// internal even if cleanup removes all physical children later.
    internal: AtomicBool,
    state: RwLock<NodeState>,
    children: NodeChildren,
}

impl Node {
    pub(super) fn new() -> Self {
        Self::from_state_and_children(
            NodeState {
                edge: Vec::new(),
                edge_index: FxHashMap::default(),
                worker_cutoffs: FxHashMap::default(),
                full_edge_workers: FxHashSet::default(),
            },
            FxHashMap::default(),
        )
    }

    pub(super) fn from_blocks_for_worker(
        blocks: &[KvCacheStoredBlockData],
        worker: WorkerWithDpRank,
    ) -> Self {
        debug_assert!(!blocks.is_empty());

        let edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)> = blocks
            .iter()
            .map(|block| (block.tokens_hash, block.block_hash))
            .collect();
        let edge_index = NodeState::edge_index_for(&edge);
        let mut full_edge_workers = FxHashSet::with_capacity_and_hasher(1, FxBuildHasher);
        full_edge_workers.insert(worker);

        Self::from_state_and_children(
            NodeState {
                edge,
                edge_index,
                worker_cutoffs: FxHashMap::default(),
                full_edge_workers,
            },
            FxHashMap::default(),
        )
    }

    pub(super) fn from_anchor(
        anchor_local_hash: LocalBlockHash,
        anchor_id: ExternalSequenceBlockHash,
    ) -> Self {
        let edge = vec![(anchor_local_hash, anchor_id)];
        let edge_index = NodeState::edge_index_for(&edge);

        Self::from_state_and_children(
            NodeState {
                edge,
                edge_index,
                worker_cutoffs: FxHashMap::default(),
                full_edge_workers: FxHashSet::default(),
            },
            FxHashMap::default(),
        )
    }

    fn from_state_and_children(
        state: NodeState,
        children: FxHashMap<LocalBlockHash, SharedNode>,
    ) -> Self {
        let internal = !children.is_empty();
        let children_map = DashMap::with_hasher(FxBuildHasher);
        for (key, child) in children {
            children_map.insert(key, child);
        }

        Self {
            shape_gate: RwLock::new(()),
            shape_version: AtomicU64::new(0),
            internal: AtomicBool::new(internal),
            state: RwLock::new(state),
            children: children_map,
        }
    }

    fn with_shape_plan<R>(
        &self,
        plan: impl FnOnce(&NodeState, &NodeChildren, u64) -> R,
    ) -> Option<R> {
        let _gate = self.shape_gate.read();
        let shape_version = self.shape_version.load(Ordering::Acquire);
        let state = self.state.read();
        Some(plan(&state, &self.children, shape_version))
    }

    fn validate_shape_read<R>(&self, expected_version: u64, f: impl FnOnce() -> R) -> Option<R> {
        let _gate = self.shape_gate.read();
        if self.shape_version.load(Ordering::Acquire) != expected_version {
            return None;
        }
        Some(f())
    }

    fn apply_metadata_update<R>(
        &self,
        expected_version: u64,
        f: impl FnOnce(&mut NodeState) -> R,
    ) -> Option<R> {
        self.validate_shape_read(expected_version, || {
            let mut state = self.state.write();
            f(&mut state)
        })
    }

    fn apply_edge_shape_update<R>(
        &self,
        expected_version: u64,
        f: impl FnOnce(&mut NodeState, &NodeChildren) -> (R, bool),
    ) -> Option<R> {
        let _gate = self.shape_gate.write();
        if self.shape_version.load(Ordering::Acquire) != expected_version {
            return None;
        }

        let mut state = self.state.write();
        let (result, shape_changed) = f(&mut state, &self.children);
        if shape_changed {
            self.shape_version.fetch_add(1, Ordering::Release);
        }
        Some(result)
    }

    pub(super) fn take_children(&self) -> Vec<SharedNode> {
        let _gate = self.shape_gate.write();
        let children: Vec<_> = self
            .children
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        if !children.is_empty() {
            self.children.clear();
            self.shape_version.fetch_add(1, Ordering::Release);
        }
        children
    }

    pub(super) fn children_snapshot(&self) -> Vec<SharedNode> {
        let _gate = self.shape_gate.read();
        self.children
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    pub(super) fn child_edges_snapshot(&self) -> Vec<(LocalBlockHash, SharedNode)> {
        self.children
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    pub(super) fn child_snapshot(&self, local_hash: LocalBlockHash) -> Option<SharedNode> {
        self.children
            .get(&local_hash)
            .map(|entry| entry.value().clone())
    }

    pub(super) fn contains_edge_hash(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.state.read().edge_index.contains_key(&hash)
    }

    #[cfg(test)]
    pub(super) fn edge_len_for_test(&self) -> usize {
        self.state.read().edge.len()
    }

    #[cfg(test)]
    pub(super) fn edge_local_hashes_for_test(&self) -> Vec<u64> {
        self.state
            .read()
            .edge
            .iter()
            .map(|&(local_hash, _)| local_hash.0)
            .collect()
    }

    pub(super) fn promote_worker_to_full_edge(&self, worker: WorkerWithDpRank) -> bool {
        let _gate = self.shape_gate.read();
        self.state.write().promote_to_full(worker)
    }

    pub(super) fn drop_worker(&self, worker: WorkerWithDpRank) {
        let _gate = self.shape_gate.write();
        let should_clear_children = {
            let mut state = self.state.write();
            state.drop_worker(worker);
            state.full_edge_workers.is_empty()
        };
        self.clear_children_if_unreachable(should_clear_children);
    }

    fn clear_children_if_unreachable(&self, should_clear_children: bool) {
        if should_clear_children && !self.children.is_empty() {
            self.children.clear();
            self.shape_version.fetch_add(1, Ordering::Release);
        }
    }

    pub(super) fn live_children(&self) -> Vec<SharedNode> {
        let _gate = self.shape_gate.read();
        self.children
            .iter()
            .filter_map(|entry| {
                let child = entry.value();
                (child.state.read().has_any_workers() || !child.children.is_empty())
                    .then(|| child.clone())
            })
            .collect()
    }

    pub(super) fn dump_snapshot(&self) -> DumpNodeSnapshot {
        let _gate = self.shape_gate.read();
        let state = self.state.read();
        let live_children: Vec<_> = self
            .children
            .iter()
            .filter_map(|entry| {
                let child = entry.value();
                (child.state.read().has_any_workers() || !child.children.is_empty())
                    .then(|| child.clone())
            })
            .collect();

        let can_merge = state.worker_cutoffs.is_empty()
            && live_children.len() == 1
            && live_children[0].has_full_coverage_only_matching(&state.full_edge_workers);

        DumpNodeSnapshot {
            edge: state.edge.clone(),
            full_edge_workers: state.full_edge_workers.iter().copied().collect(),
            worker_cutoffs: state
                .worker_cutoffs
                .iter()
                .map(|(&worker, &cutoff)| (worker, cutoff))
                .collect(),
            live_children,
            has_any_workers: state.has_any_workers(),
            children_empty: self.children.is_empty(),
            can_merge,
        }
    }

    fn has_full_coverage_only_matching(&self, workers: &FxHashSet<WorkerWithDpRank>) -> bool {
        let state = self.state.read();
        state.worker_cutoffs.is_empty()
            && state.full_edge_workers == *workers
            && state.has_any_workers()
    }

    pub(super) fn lookup_entries_by_worker(
        &self,
    ) -> Vec<(WorkerWithDpRank, Vec<ExternalSequenceBlockHash>)> {
        let state = self.state.read();
        let mut entries = Vec::new();

        for &worker in &state.full_edge_workers {
            entries.push((worker, state.edge.iter().map(|&(_, hash)| hash).collect()));
        }
        for (&worker, &cutoff) in &state.worker_cutoffs {
            if cutoff > 0 {
                entries.push((
                    worker,
                    state.edge[..cutoff].iter().map(|&(_, hash)| hash).collect(),
                ));
            }
        }

        entries
    }

    pub(super) fn lookup_hashes_for_worker_repair(
        &self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
        direction: LookupRepairDirection,
    ) -> Vec<ExternalSequenceBlockHash> {
        let state = self.state.read();
        let Some(&pos) = state.edge_index.get(&hash) else {
            return Vec::new();
        };
        let cutoff = state.current_cutoff(worker).min(state.edge.len());
        let range = match direction {
            LookupRepairDirection::TowardTail => {
                if pos < cutoff {
                    pos..cutoff
                } else {
                    0..0
                }
            }
            LookupRepairDirection::TowardHead => 0..pos.min(cutoff),
        };

        state.edge[range].iter().map(|&(_, hash)| hash).collect()
    }

    pub(super) fn reject_uncovered_parent(
        &self,
        worker: WorkerWithDpRank,
        parent_hash: ExternalSequenceBlockHash,
    ) -> Option<UncoveredParent> {
        let _gate = self.shape_gate.read();
        let state = self.state.read();
        let &pos = state.edge_index.get(&parent_hash)?;
        if state.covers_pos(worker, pos) {
            return None;
        }
        Some(UncoveredParent {
            pos,
            cutoff: state.current_cutoff(worker),
        })
    }

    pub(super) fn plan_store_parent_edge(
        &self,
        parent_hash: ExternalSequenceBlockHash,
        blocks: &[KvCacheStoredBlockData],
    ) -> Option<ParentEdgePlan> {
        self.with_shape_plan(|state, _children, shape_version| {
            let &parent_pos = state.edge_index.get(&parent_hash)?;

            let action = if state.tail_hash_is(parent_hash) {
                ParentEdgePlanAction::InsertFromParent
            } else if state.suffix_matches_store(parent_pos, blocks) {
                ParentEdgePlanAction::ReuseExistingEdge {
                    cutoff: parent_pos + 1 + blocks.len(),
                }
            } else if !self.internal.load(Ordering::Acquire) {
                match state.store_starts_with_suffix(parent_pos, blocks) {
                    Some(append_start) => {
                        ParentEdgePlanAction::ReuseSuffixAndExtendLeaf { append_start }
                    }
                    None => ParentEdgePlanAction::Split {
                        split_pos: parent_pos + 1,
                    },
                }
            } else {
                ParentEdgePlanAction::Split {
                    split_pos: parent_pos + 1,
                }
            };

            Some(ParentEdgePlan {
                shape_version,
                action,
            })
        })
        .flatten()
    }

    pub(super) fn apply_store_parent_edge_plan(
        &self,
        worker: WorkerWithDpRank,
        plan: ParentEdgePlan,
        blocks: &[KvCacheStoredBlockData],
    ) -> ParentEdgeAction {
        match plan.action {
            ParentEdgePlanAction::InsertFromParent => self
                .validate_shape_read(plan.shape_version, || {
                    ParentEdgeAction::InsertFromParent(None)
                })
                .unwrap_or(ParentEdgeAction::Stale),
            ParentEdgePlanAction::ReuseExistingEdge { cutoff } => self
                .apply_metadata_update(plan.shape_version, |state| {
                    ParentEdgeAction::ReuseExistingEdge {
                        coverage_changed: state.cover_prefix_for_worker(worker, cutoff),
                    }
                })
                .unwrap_or(ParentEdgeAction::Stale),
            ParentEdgePlanAction::ReuseSuffixAndExtendLeaf { append_start } => self
                .apply_edge_shape_update(plan.shape_version, |state, _children| {
                    if !self.internal.load(Ordering::Acquire) {
                        state.append_blocks_to_leaf(worker, &blocks[append_start..]);
                        (
                            ParentEdgeAction::ReuseExistingEdge {
                                coverage_changed: true,
                            },
                            true,
                        )
                    } else {
                        (ParentEdgeAction::Stale, false)
                    }
                })
                .unwrap_or(ParentEdgeAction::Stale),
            ParentEdgePlanAction::Split { split_pos } => self
                .apply_edge_shape_update(plan.shape_version, |state, _children| {
                    (
                        ParentEdgeAction::InsertFromParent(Some(
                            self.split_at_locked(state, split_pos),
                        )),
                        true,
                    )
                })
                .unwrap_or(ParentEdgeAction::Stale),
        }
    }

    pub(super) fn scan_store_prefix(&self, blocks: &[KvCacheStoredBlockData]) -> ChildEdgeScan {
        loop {
            if let Some(scan) = self.with_shape_plan(|state, _children, shape_version| {
                let mut match_len = 0;
                let mut block_hash_mismatch = None;

                for (edge_elem, block) in state.edge.iter().zip(blocks.iter()) {
                    if edge_elem.0 != block.tokens_hash {
                        break;
                    }
                    if edge_elem.1 != block.block_hash && block_hash_mismatch.is_none() {
                        block_hash_mismatch = Some((block.block_hash, edge_elem.1));
                    }
                    match_len += 1;
                }

                ChildEdgeScan {
                    shape_version,
                    edge_len: state.edge.len(),
                    match_len,
                    block_hash_mismatch,
                }
            }) {
                return scan;
            }
        }
    }

    pub(super) fn cover_prefix_for_worker_with_version(
        &self,
        worker: WorkerWithDpRank,
        cutoff: usize,
        shape_version: u64,
    ) -> Option<bool> {
        self.apply_metadata_update(shape_version, |state| {
            state.cover_prefix_for_worker(worker, cutoff)
        })
    }

    pub(super) fn promote_to_full_with_version(
        &self,
        worker: WorkerWithDpRank,
        shape_version: u64,
    ) -> Option<bool> {
        self.apply_metadata_update(shape_version, |state| state.promote_to_full(worker))
    }

    pub(super) fn split_for_store_tail(
        &self,
        worker: WorkerWithDpRank,
        split_pos: usize,
        tail_first_local: LocalBlockHash,
        tail_node: SharedNode,
        shape_version: u64,
    ) -> SplitStoreOutcome {
        self.apply_edge_shape_update(shape_version, |state, children| {
            let split = self.split_at_locked(state, split_pos);
            state.promote_to_full(worker);
            children.insert(tail_first_local, tail_node.clone());
            (SplitStoreOutcome::Done { split, tail_node }, true)
        })
        .unwrap_or(SplitStoreOutcome::Stale)
    }

    pub(super) fn try_extend_leaf_with_version(
        &self,
        worker: WorkerWithDpRank,
        parent_hash: ExternalSequenceBlockHash,
        blocks: &[KvCacheStoredBlockData],
        shape_version: u64,
    ) -> Option<bool> {
        self.apply_edge_shape_update(shape_version, |state, _children| {
            if self.internal.load(Ordering::Acquire) || blocks.is_empty() || state.edge.is_empty() {
                return (false, false);
            }

            let old_len = state.edge.len();
            if !state.tail_hash_is(parent_hash) || !state.covers_pos(worker, old_len - 1) {
                return (false, false);
            }

            state.append_blocks_to_leaf(worker, blocks);
            (true, true)
        })
    }

    pub(super) fn child_lookup_plan(
        &self,
        last_ext_hash: Option<ExternalSequenceBlockHash>,
        first_local: LocalBlockHash,
    ) -> ParentChildPlan {
        self.with_shape_plan(|state, children, shape_version| {
            if let Some(hash) = last_ext_hash
                && !state.edge_index.contains_key(&hash)
            {
                return ParentChildPlan::StaleParent { hash };
            }

            if let Some(child) = children
                .get(&first_local)
                .map(|entry| entry.value().clone())
            {
                return ParentChildPlan::Descend(child);
            }

            ParentChildPlan::MissingChild { shape_version }
        })
        .unwrap_or(ParentChildPlan::Stale)
    }

    pub(super) fn insert_child_if_still_missing(
        &self,
        first_local: LocalBlockHash,
        child: SharedNode,
        shape_version: u64,
    ) -> InsertChildOutcome {
        {
            let _gate = self.shape_gate.read();
            if self.shape_version.load(Ordering::Acquire) != shape_version {
                return InsertChildOutcome::Stale;
            }
            if self.internal.load(Ordering::Acquire) {
                return match self.children.entry(first_local) {
                    dashmap::mapref::entry::Entry::Occupied(entry) => {
                        InsertChildOutcome::Existing(entry.get().clone())
                    }
                    dashmap::mapref::entry::Entry::Vacant(entry) => {
                        entry.insert(child.clone());
                        InsertChildOutcome::Inserted(child)
                    }
                };
            }
        }

        let _gate = self.shape_gate.write();
        if self.shape_version.load(Ordering::Acquire) != shape_version {
            return InsertChildOutcome::Stale;
        }
        match self.children.entry(first_local) {
            dashmap::mapref::entry::Entry::Occupied(entry) => {
                InsertChildOutcome::Existing(entry.get().clone())
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                entry.insert(child.clone());
                self.internal.store(true, Ordering::Release);
                self.shape_version.fetch_add(1, Ordering::Release);
                InsertChildOutcome::Inserted(child)
            }
        }
    }

    fn split_at_locked(&self, state: &mut NodeState, pos: usize) -> SplitLookupData {
        debug_assert!(
            pos > 0 && pos < state.edge.len(),
            "split position {pos} out of range for edge length {}",
            state.edge.len()
        );

        let suffix_edge = state.edge.split_off(pos);
        let suffix_first_local = suffix_edge[0].0;
        let prefix_len = pos;
        let suffix_edge_index = NodeState::edge_index_for(&suffix_edge);

        for &(_, hash) in &suffix_edge {
            state.edge_index.remove(&hash);
        }

        let mut suffix_full =
            FxHashSet::with_capacity_and_hasher(state.full_edge_workers.len(), FxBuildHasher);
        let mut suffix_cutoffs =
            FxHashMap::with_capacity_and_hasher(state.worker_cutoffs.len(), FxBuildHasher);
        let mut to_promote: Vec<WorkerWithDpRank> = Vec::new();

        for &worker in &state.full_edge_workers {
            suffix_full.insert(worker);
        }
        for (&worker, &cutoff) in &state.worker_cutoffs {
            if cutoff >= prefix_len {
                to_promote.push(worker);
                let suffix_cutoff = cutoff - prefix_len;
                if suffix_cutoff > 0 {
                    suffix_cutoffs.insert(worker, suffix_cutoff);
                }
            }
        }
        for worker in &to_promote {
            state.worker_cutoffs.remove(worker);
            state.full_edge_workers.insert(*worker);
        }

        let suffix_children: FxHashMap<_, _> = self
            .children
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect();
        self.children.clear();

        let suffix = Arc::new(Node::from_state_and_children(
            NodeState {
                edge: suffix_edge,
                edge_index: suffix_edge_index,
                worker_cutoffs: suffix_cutoffs,
                full_edge_workers: suffix_full,
            },
            suffix_children,
        ));
        self.children.insert(suffix_first_local, suffix.clone());
        self.internal.store(true, Ordering::Release);

        SplitLookupData { suffix }
    }

    pub(super) fn remove_worker_for_hash(
        &self,
        worker: WorkerWithDpRank,
        block_hash: ExternalSequenceBlockHash,
    ) -> Option<RemoveOutcome> {
        let _gate = self.shape_gate.write();
        let mut state = self.state.write();
        let pos = state.edge_index.get(&block_hash).copied()?;
        let outcome = state.remove_worker_at_pos(worker, pos, block_hash);
        let should_clear_children = state.full_edge_workers.is_empty();
        drop(state);
        self.clear_children_if_unreachable(should_clear_children);
        Some(outcome)
    }

    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn find_match_step<S: HashSequence>(
        &self,
        mut input: FindStepInput<'_, S>,
    ) -> FindStepOutcome {
        // NOTE: This read intentionally does not take shape_gate. A concurrent
        // split can make the edge snapshot and child lookup come from adjacent
        // tree shapes; find_matches tolerates that brief best-effort race.
        let state = self.state.read();
        let edge_len = state.edge.len();
        let walk_len = edge_len.min(input.sequence.len() - input.seq_pos);

        let mut edge_match_len = 1;
        for i in 1..walk_len {
            if state.edge[i].0 != input.sequence.at(input.seq_pos + i) {
                break;
            }
            edge_match_len += 1;
        }

        let edge_hash_at = |depth: usize| -> ExternalSequenceBlockHash {
            debug_assert!(depth > 0 && depth <= state.edge.len());
            state.edge[depth - 1].1
        };

        if input.first_node {
            *input.active = state.full_edge_workers.clone();
            for (&worker, &cutoff) in &state.worker_cutoffs {
                let contribution = cutoff.min(edge_match_len);
                if contribution > 0 {
                    input.scores.scores.insert(worker, contribution as u32);
                    record_last_matched_hash(
                        &mut input.last_matched_hashes,
                        worker,
                        edge_hash_at(contribution),
                    );
                }
            }
        } else {
            let has_partial = !state.worker_cutoffs.is_empty();
            if has_partial {
                input.active.retain(|worker| {
                    if state.full_edge_workers.contains(worker) {
                        true
                    } else if let Some(&cutoff) = state.worker_cutoffs.get(worker) {
                        let effective = cutoff.min(edge_match_len);
                        input
                            .scores
                            .scores
                            .insert(*worker, input.prev_depth + effective as u32);
                        if effective > 0 {
                            record_last_matched_hash(
                                &mut input.last_matched_hashes,
                                *worker,
                                edge_hash_at(effective),
                            );
                        } else if let Some(hash) = input.prev_edge_last_hash {
                            record_last_matched_hash(&mut input.last_matched_hashes, *worker, hash);
                        }
                        false
                    } else {
                        input.scores.scores.insert(*worker, input.prev_depth);
                        if let Some(hash) = input.prev_edge_last_hash {
                            record_last_matched_hash(&mut input.last_matched_hashes, *worker, hash);
                        }
                        false
                    }
                });
            } else if state.full_edge_workers.len() != input.active_count {
                input.active.retain(|worker| {
                    if state.full_edge_workers.contains(worker) {
                        true
                    } else {
                        input.scores.scores.insert(*worker, input.prev_depth);
                        if let Some(hash) = input.prev_edge_last_hash {
                            record_last_matched_hash(&mut input.last_matched_hashes, *worker, hash);
                        }
                        false
                    }
                });
            }
        }

        let active_count = input.active.len();
        let next_child = if edge_match_len == edge_len
            && active_count > 0
            && input.seq_pos + edge_match_len < input.sequence.len()
        {
            self.children
                .get(&input.sequence.at(input.seq_pos + edge_match_len))
                .map(|entry| entry.value().clone())
        } else {
            None
        };

        FindStepOutcome {
            edge_len,
            edge_match_len,
            active_count,
            next_child,
            prev_edge_last_hash: Some(state.edge[edge_match_len - 1].1),
        }
    }

    pub(super) fn remove_child_if_stale_leaf(&self, key: LocalBlockHash, child: &SharedNode) {
        let _parent_gate = self.shape_gate.write();
        let still_attached = self
            .children
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current.value(), child));
        if !still_attached {
            return;
        }

        let Some(_child_gate) = child.shape_gate.try_write() else {
            return;
        };
        if child.state.read().has_any_workers() || !child.children.is_empty() {
            return;
        }
        if Arc::strong_count(child) != 2 {
            return;
        }

        self.children.remove(&key);
        self.shape_version.fetch_add(1, Ordering::Release);
    }
}

pub(super) struct CleanupEdge {
    pub(super) parent: Weak<Node>,
    pub(super) key: LocalBlockHash,
    pub(super) child: Weak<Node>,
}
