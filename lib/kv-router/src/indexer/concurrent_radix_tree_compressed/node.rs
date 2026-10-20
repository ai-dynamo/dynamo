// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::{RwLock, RwLockUpgradableReadGuard};
use rustc_hash::FxHashMap;

use super::children::{ChildInsertResult, NodeChildren};
use super::state::CrtcNodeState;
use super::types::*;
use super::worker_set::WorkerSet;
use crate::protocols::*;

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
    // NOTE(perf): Consolidating the shape gate and version synchronization into
    // one lock regressed throughput. Re-profile before combining these fields.
    shape_gate: RwLock<()>,
    /// NOTE(concurrency): This is a post-commit validation token, not a seqlock.
    /// Node state and children do not share one immutable publication boundary.
    shape_version: AtomicU64,
    /// Sticky logical-internal marker. Once true, this node is treated as
    /// internal even if cleanup removes all physical children later.
    internal: AtomicBool,
    /// Set only for synthetic branch anchors, which live in `anchor_nodes` for the
    /// tree's lifetime and never change their single-block edge.
    anchor: bool,
    state: RwLock<CrtcNodeState>,
    children: NodeChildren,
}

impl Node {
    pub(super) fn new() -> Self {
        Self::from_state_and_children(CrtcNodeState::new(Vec::new()), FxHashMap::default())
    }

    /// A leaf for `blocks` that `worker` fully covers, attached under the node whose edge
    /// ends with `parent_tail` (`None` under the root).
    pub(super) fn from_blocks_for_worker(
        blocks: &[KvCacheStoredBlockData],
        worker: WorkerWithDpRank,
        parent_tail: Option<ExternalSequenceBlockHash>,
    ) -> Self {
        debug_assert!(!blocks.is_empty());
        let mut state = CrtcNodeState::for_blocks(blocks, worker);
        state.parent_tail = parent_tail.unwrap_or(ExternalSequenceBlockHash(0));
        Self::from_state_and_children(state, FxHashMap::default())
    }

    pub(super) fn from_anchor(
        anchor_local_hash: LocalBlockHash,
        anchor_id: ExternalSequenceBlockHash,
    ) -> Self {
        let mut node = Self::from_state_and_children(
            CrtcNodeState::new(vec![(anchor_local_hash, anchor_id)]),
            FxHashMap::default(),
        );
        node.anchor = true;
        node
    }

    fn from_state_and_children(
        state: CrtcNodeState,
        children: FxHashMap<LocalBlockHash, SharedNode>,
    ) -> Self {
        Self::from_state_and_node_children(state, NodeChildren::from_map(children))
    }

    fn from_state_and_node_children(state: CrtcNodeState, children: NodeChildren) -> Self {
        let internal = !children.is_empty();
        Self {
            shape_gate: RwLock::new(()),
            shape_version: AtomicU64::new(0),
            internal: AtomicBool::new(internal),
            anchor: false,
            state: RwLock::new(state),
            children,
        }
    }

    fn with_shape_plan<R>(&self, plan: impl FnOnce(&CrtcNodeState, &NodeChildren, u64) -> R) -> R {
        // NOTE(perf): Replacing these shape-gated reads with state-only snapshots
        // was neutral or regressive, and profiling did not identify the RwLock
        // as a hotspot. Re-profile before removing this shape read.
        let _gate = self.shape_gate.read();
        let shape_version = self.shape_version.load(Ordering::Acquire);
        let state = self.state.read();
        plan(&state, &self.children, shape_version)
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
        f: impl FnOnce(&mut CrtcNodeState) -> R,
    ) -> Option<R> {
        self.validate_shape_read(expected_version, || {
            let mut state = self.state.write();
            f(&mut state)
        })
    }

    fn apply_edge_shape_update<R>(
        &self,
        expected_version: u64,
        f: impl FnOnce(&mut CrtcNodeState, &NodeChildren) -> (R, bool),
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
        let children = self.children.values_snapshot();
        if self.children.clear() {
            self.shape_version.fetch_add(1, Ordering::Release);
        }
        children
    }

    #[cfg(test)]
    pub(super) fn children_snapshot(&self) -> Vec<SharedNode> {
        let _gate = self.shape_gate.read();
        self.children.values_snapshot()
    }

    pub(super) fn push_children_into(&self, queue: &mut VecDeque<SharedNode>) {
        let _gate = self.shape_gate.read();
        self.children.extend_values(queue);
    }

    #[cfg(test)]
    pub(super) fn child_edges_snapshot(&self) -> Vec<(LocalBlockHash, SharedNode)> {
        self.children.entries_snapshot()
    }

    pub(super) fn child_snapshot(&self, local_hash: LocalBlockHash) -> Option<SharedNode> {
        self.children.get(&local_hash)
    }

    pub(super) fn is_anchor(&self) -> bool {
        self.anchor
    }

    pub(super) fn contains_edge_hash(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.state.read().contains_hash(hash)
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
        // NOTE(perf): This path is anchor-only today. Removing its shape read
        // did not improve throughput; re-evaluate if non-anchor callers appear.
        let _gate = self.shape_gate.read();
        self.state.write().promote_to_full(worker)
    }

    pub(super) fn remove_target_and_snapshot_children(
        &self,
        target: WorkerRemovalTarget,
    ) -> Vec<SharedNode> {
        let _gate = self.shape_gate.write();
        let mut state = self.state.write();
        let old_full_len = state.full_edge_workers.len();
        let old_cutoff_len = state.worker_cutoffs.len();
        state
            .full_edge_workers
            .retain(|worker| !target.matches(*worker));
        state
            .worker_cutoffs
            .retain(|worker, _| !target.matches(*worker));
        let removed_worker = old_full_len != state.full_edge_workers.len()
            || old_cutoff_len != state.worker_cutoffs.len();
        let should_clear_children = removed_worker && state.full_edge_workers.is_empty();

        // A concurrent split is either visible in this snapshot or starts after
        // the target coverage is gone and therefore cannot copy it forward.
        let children = self.children.values_snapshot();
        drop(state);
        self.clear_children_if_unreachable(should_clear_children);
        children
    }

    fn clear_children_if_unreachable(&self, should_clear_children: bool) {
        if should_clear_children && self.children.clear() {
            self.shape_version.fetch_add(1, Ordering::Release);
        }
    }

    pub(super) fn live_children(&self) -> Vec<SharedNode> {
        let _gate = self.shape_gate.read();
        self.children
            .values_snapshot()
            .into_iter()
            .filter(|child| child.state.read().has_any_workers() || !child.children.is_empty())
            .collect()
    }

    pub(super) fn dump_snapshot(&self) -> DumpNodeSnapshot {
        let _gate = self.shape_gate.read();
        let state = self.state.read();
        let live_children: Vec<_> = self
            .children
            .values_snapshot()
            .into_iter()
            .filter(|child| child.state.read().has_any_workers() || !child.children.is_empty())
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

    fn has_full_coverage_only_matching(&self, workers: &WorkerSet) -> bool {
        let state = self.state.read();
        state.worker_cutoffs.is_empty()
            && state.full_edge_workers == *workers
            && state.has_any_workers()
    }

    /// The edge's hashes and how far each of `workers` covers them, read under one lock.
    pub(super) fn covered_prefixes(
        &self,
        workers: impl Iterator<Item = WorkerWithDpRank>,
    ) -> (Vec<ExternalSequenceBlockHash>, Vec<usize>) {
        let state = self.state.read();
        let cutoffs = workers.map(|worker| state.current_cutoff(worker)).collect();
        let hashes = state.edge.iter().map(|&(_, hash)| hash).collect();
        (hashes, cutoffs)
    }

    pub(super) fn lookup_hashes_for_worker_repair(
        &self,
        worker: WorkerWithDpRank,
        hash: ExternalSequenceBlockHash,
        direction: LookupRepairDirection,
    ) -> Vec<ExternalSequenceBlockHash> {
        let state = self.state.read();
        let Some(pos) = state.position(hash) else {
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

    pub(super) fn parent_coverage(
        &self,
        worker: WorkerWithDpRank,
        parent_hash: ExternalSequenceBlockHash,
    ) -> ParentCoverage {
        let _gate = self.shape_gate.read();
        let state = self.state.read();
        let Some(pos) = state.position(parent_hash) else {
            return ParentCoverage::Missing;
        };
        if state.covers_pos(worker, pos) {
            return ParentCoverage::Covered;
        }
        ParentCoverage::Uncovered(UncoveredParent {
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
            let parent_pos = state.position(parent_hash)?;

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
    }

    pub(super) fn apply_store_parent_edge_plan(
        &self,
        worker: WorkerWithDpRank,
        plan: ParentEdgePlan,
        blocks: &[KvCacheStoredBlockData],
    ) -> ParentEdgeAction {
        match plan.action {
            // NOTE(perf): Removing this validation did not produce a repeatable
            // benefit and regressed scaled cumulative workloads.
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
            // NOTE(perf): An additional sticky-internal rejection before this
            // commit did not improve throughput. The check inside the gate
            // closes the split race.
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
        self.with_shape_plan(|state, _children, shape_version| {
            let mut match_len = 0;
            for (edge_elem, block) in state.edge.iter().zip(blocks) {
                if edge_elem.0 != block.tokens_hash {
                    break;
                }
                match_len += 1;
            }

            ChildEdgeScan {
                shape_version,
                edge_len: state.edge.len(),
                match_len,
            }
        })
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
            let _ = children.insert(tail_first_local, tail_node.clone());
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
        if self.internal.load(Ordering::Acquire) {
            return Some(false);
        }

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
                && !state.tail_hash_is(hash)
            {
                if state.contains_hash(hash) {
                    return ParentChildPlan::InteriorParent { shape_version };
                }
                return ParentChildPlan::StaleParent { hash };
            }

            if let Some(child) = children.get(&first_local) {
                return ParentChildPlan::Descend(child);
            }

            ParentChildPlan::MissingChild { shape_version }
        })
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
                return match self.children.insert_if_absent(first_local, child) {
                    ChildInsertResult::Existing(child) => InsertChildOutcome::Existing(child),
                    ChildInsertResult::Inserted(child) => InsertChildOutcome::Inserted(child),
                };
            }
        }

        let _gate = self.shape_gate.write();
        if self.shape_version.load(Ordering::Acquire) != shape_version {
            return InsertChildOutcome::Stale;
        }
        match self.children.insert_if_absent(first_local, child) {
            ChildInsertResult::Existing(child) => InsertChildOutcome::Existing(child),
            ChildInsertResult::Inserted(child) => {
                self.internal.store(true, Ordering::Release);
                self.shape_version.fetch_add(1, Ordering::Release);
                InsertChildOutcome::Inserted(child)
            }
        }
    }

    fn split_at_locked(&self, state: &mut CrtcNodeState, pos: usize) -> SplitLookupData {
        let suffix_state = state.split_off_suffix(pos);
        let suffix_first_local = suffix_state.edge[0].0;
        let suffix_children = self.children.transfer_for_split();

        let suffix = Arc::new(Node::from_state_and_node_children(
            suffix_state,
            suffix_children,
        ));
        let _ = self.children.insert(suffix_first_local, suffix.clone());
        self.internal.store(true, Ordering::Release);

        SplitLookupData { suffix }
    }

    /// Removes `worker` from the leading run of `hashes` found in this edge, under one
    /// exclusive shape gate. Returns how many hashes the run consumed, the newly uncovered
    /// hashes and whether the removal took the node's last holder, or `None` if the first
    /// hash is no longer in this edge.
    pub(super) fn remove_worker_for_leading_hashes(
        &self,
        worker: WorkerWithDpRank,
        hashes: &[ExternalSequenceBlockHash],
    ) -> Option<(usize, Vec<ExternalSequenceBlockHash>, bool)> {
        // The exclusive gate keeps every other writer off this node, so the run is read
        // alongside readers, and only a coverage change upgrades to the write lock.
        let _gate = self.shape_gate.write();
        let state = self.state.upgradable_read();
        let mut min_match: Option<(usize, ExternalSequenceBlockHash)> = None;
        let mut consumed = 0;
        let mut last_pos = None;

        for &hash in hashes {
            let Some(pos) = state.position_near(hash, last_pos) else {
                break;
            };
            last_pos = Some(pos);
            if min_match.is_none_or(|(min_pos, _)| pos < min_pos) {
                min_match = Some((pos, hash));
            }
            consumed += 1;
        }

        let (pos, block_hash) = min_match?;
        let (stale_hashes, should_clear_children, became_dead) =
            if pos >= state.current_cutoff(worker) {
                let should_clear_children = state.full_edge_workers.is_empty();
                drop(state);
                (vec![block_hash], should_clear_children, false)
            } else {
                let mut state = RwLockUpgradableReadGuard::upgrade(state);
                let outcome = state.remove_worker_at_pos(worker, pos, block_hash);
                (
                    outcome.stale_hashes,
                    state.full_edge_workers.is_empty(),
                    !state.has_any_workers(),
                )
            };
        self.clear_children_if_unreachable(should_clear_children);
        Some((consumed, stale_hashes, became_dead))
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

        if let Some(block_hashes) = input.kv_transfer_chain.as_deref_mut() {
            block_hashes.extend(
                state
                    .edge
                    .iter()
                    .take(edge_match_len)
                    .map(|(_, block_hash)| *block_hash),
            );
        }

        if input.first_node {
            // Every scored worker is covered by the first node, so its coverage bounds
            // the result size; reserving avoids repeated rehash growth per query.
            let scored_bound = state.full_edge_workers.len() + state.worker_cutoffs.len();
            input.scores.scores.reserve(scored_bound);
            if let Some(last_matched_hashes) = input.last_matched_hashes.as_deref_mut() {
                last_matched_hashes.reserve(scored_bound);
            }
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
            } else if !input.active.same_members(&state.full_edge_workers) {
                // Matching size and fingerprint skip the intersection; see README
                // "Equal-set skip".
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
            } else {
                debug_assert_eq!(**input.active, *state.full_edge_workers);
            }
        }

        let active_count = input.active.len();
        let next_child = if edge_match_len == edge_len
            && active_count > 0
            && input.seq_pos + edge_match_len < input.sequence.len()
        {
            self.children
                .get(&input.sequence.at(input.seq_pos + edge_match_len))
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

    /// The child-map key and parent tail of a node that no worker covers and that has no
    /// children, or `None` if it is still live.
    pub(super) fn dead_leaf_key_and_tail(
        &self,
    ) -> Option<(LocalBlockHash, ExternalSequenceBlockHash)> {
        let state = self.state.read();
        if state.has_any_workers() || state.edge.is_empty() || !self.children.is_empty() {
            return None;
        }
        Some((state.edge[0].0, state.parent_tail))
    }

    pub(super) fn is_holderless(&self) -> bool {
        !self.state.read().has_any_workers()
    }

    pub(super) fn has_children(&self) -> bool {
        !self.children.is_empty()
    }

    pub(super) fn for_each_child(&self, visit: impl FnMut(LocalBlockHash, &SharedNode)) {
        self.children.for_each_child(visit);
    }

    pub(super) fn has_child(&self, key: LocalBlockHash, child: &SharedNode) -> bool {
        self.children
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(&current, child))
    }

    /// Unlinks every child in `batch` that is still attached, holder-less, childless and
    /// referenced by nothing else, under one exclusive gate and one version bump. Returns
    /// how many it unlinked, or `None` without waiting if the gate is taken.
    pub(super) fn try_remove_stale_children(
        &self,
        batch: &[(LocalBlockHash, Weak<Node>)],
    ) -> Option<usize> {
        let _gate = self.shape_gate.try_write()?;
        let mut removed = 0;
        for (key, child) in batch {
            let Some(child) = child.upgrade() else {
                continue;
            };
            if !self.has_child(*key, &child) {
                continue;
            }
            let Some(_child_gate) = child.shape_gate.try_write() else {
                continue;
            };
            if child.state.read().has_any_workers() || !child.children.is_empty() {
                continue;
            }
            // The parent's map and this upgrade. Any other reference is a lane lookup
            // that still names the child or a reader mid-walk.
            if Arc::strong_count(&child) != 2 {
                continue;
            }
            let _ = self.children.remove(key);
            removed += 1;
        }
        if removed > 0 {
            self.shape_version.fetch_add(1, Ordering::Release);
        }
        Some(removed)
    }
}

pub(super) struct CleanupEdge {
    pub(super) parent: Weak<Node>,
    pub(super) key: LocalBlockHash,
    pub(super) child: Weak<Node>,
}
