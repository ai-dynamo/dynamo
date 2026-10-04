// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::atomic::{self, AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use crossbeam_epoch::Guard;
use parking_lot::{RwLock, RwLockWriteGuard};
use rustc_hash::FxHashMap;

use super::children::{ChildInsertResult, NodeChildren};
use super::coverage::{FullCoverage, Slot, SlotSet, SlotTable};
use super::state::CrtcNodeState;
use super::types::*;
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
    /// Ranks covering the whole edge. Readers load the bits under the state read lock,
    /// so they see bits consistent with the edge. A bit is set under the shared shape
    /// gate (see `promote_full`) or with the state write lock held; a slot dropped from
    /// the whole edge is cleared under the exclusive gate. Splits and extensions, which
    /// move bits between nodes or demote them to cutoffs, hold the exclusive gate.
    full: FullCoverage,
    state: RwLock<CrtcNodeState>,
    /// Whether `state.cutoffs` is non-empty, republished whenever the state write lock
    /// is released, so a promote can skip the state lock when it has no cutoff to delete.
    has_cutoffs: AtomicBool,
    children: NodeChildren,
    /// Strong references to this node held by retired child-map snapshots that are
    /// awaiting epoch reclamation. Never larger than the actual number.
    retired_snapshot_refs: AtomicUsize,
}

impl Node {
    pub(super) fn new() -> Self {
        Self::childless(CrtcNodeState::new(Vec::new()), FullCoverage::default())
    }

    pub(super) fn from_blocks_for_slot(blocks: &[KvCacheStoredBlockData], slot: Slot) -> Self {
        debug_assert!(!blocks.is_empty());
        Self::childless(
            CrtcNodeState::for_blocks(blocks),
            FullCoverage::single(slot),
        )
    }

    pub(super) fn from_anchor(
        anchor_local_hash: LocalBlockHash,
        anchor_id: ExternalSequenceBlockHash,
    ) -> Self {
        Self::childless(
            CrtcNodeState::new(vec![(anchor_local_hash, anchor_id)]),
            FullCoverage::default(),
        )
    }

    fn childless(state: CrtcNodeState, full: FullCoverage) -> Self {
        Self::from_parts(state, full, NodeChildren::from_map(FxHashMap::default()))
    }

    fn from_parts(state: CrtcNodeState, full: FullCoverage, children: NodeChildren) -> Self {
        let internal = !children.is_empty();
        Self {
            shape_gate: RwLock::new(()),
            shape_version: AtomicU64::new(0),
            internal: AtomicBool::new(internal),
            full,
            has_cutoffs: AtomicBool::new(!state.cutoffs.is_empty()),
            state: RwLock::new(state),
            children,
            retired_snapshot_refs: AtomicUsize::new(0),
        }
    }

    fn write_state(&self) -> StateWriteGuard<'_> {
        StateWriteGuard {
            has_cutoffs: &self.has_cutoffs,
            state: self.state.write(),
        }
    }

    pub(super) fn note_retired_snapshot_ref(&self) {
        self.retired_snapshot_refs.fetch_add(1, Ordering::Release);
    }

    pub(super) fn release_retired_snapshot_ref(&self) {
        self.retired_snapshot_refs.fetch_sub(1, Ordering::Release);
    }

    #[cfg(test)]
    pub(super) fn retired_snapshot_refs_for_test(&self) -> usize {
        self.retired_snapshot_refs.load(Ordering::Acquire)
    }

    /// Strong references other than those held by retired child-map snapshots, or
    /// `None` if a snapshot was reclaimed during the read.
    ///
    /// Callers hold the parent's exclusive shape gate, so no snapshot holding this node
    /// can be retired meanwhile; the count can only fall. Equal reads before and after
    /// the strong count therefore bracket a strong count with no reclamation in between.
    fn live_strong_count(this: &SharedNode) -> Option<usize> {
        let retired = this.retired_snapshot_refs.load(Ordering::Acquire);
        let strong = Arc::strong_count(this);
        atomic::fence(Ordering::Acquire);
        (this.retired_snapshot_refs.load(Ordering::Acquire) == retired)
            .then(|| strong.saturating_sub(retired))
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
            let mut state = self.write_state();
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

        let mut state = self.write_state();
        let (result, shape_changed) = f(&mut state, &self.children);
        if shape_changed {
            self.shape_version.fetch_add(1, Ordering::Release);
        }
        Some(result)
    }

    /// Takes this node's child `Arc`s and frees its child map now instead of through the
    /// epoch, for a tree being dropped.
    ///
    /// # Safety
    ///
    /// No thread may still walk this node's children: the caller owns the tree exclusively.
    pub(super) unsafe fn take_children(&self) -> Vec<SharedNode> {
        // SAFETY: forwarded from the caller.
        unsafe { self.children.take_exclusive() }
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

    pub(super) fn child_edges_snapshot(&self) -> Vec<(LocalBlockHash, SharedNode)> {
        self.children.entries_snapshot()
    }

    #[cfg(test)]
    pub(super) fn child_snapshot(&self, local_hash: LocalBlockHash) -> Option<SharedNode> {
        self.children.get(&local_hash)
    }

    pub(super) fn child_ref<'g>(
        &'g self,
        local_hash: LocalBlockHash,
        guard: &'g Guard,
    ) -> Option<&'g Node> {
        self.children.get_ref(&local_hash, guard)
    }

    /// Takes this node's child `Arc`s so the graveyard can free a subtree iteratively.
    pub(super) fn into_child_arcs(self) -> Vec<SharedNode> {
        self.children.into_child_arcs()
    }

    /// Number of leading `hashes` present in this node's edge, read under one lock.
    pub(super) fn leading_edge_hash_count(&self, hashes: &[ExternalSequenceBlockHash]) -> usize {
        let state = self.state.read();
        hashes
            .iter()
            .take_while(|hash| state.edge_index.contains_key(hash))
            .count()
    }

    pub(super) fn contains_edge_hash(&self, hash: ExternalSequenceBlockHash) -> bool {
        self.state.read().edge_index.contains_key(&hash)
    }

    #[cfg(test)]
    pub(super) fn edge_len_for_test(&self) -> usize {
        self.state.read().edge.len()
    }

    /// Full-edge slots and cutoffs, read under one state lock.
    #[cfg(test)]
    pub(super) fn coverage_for_test(&self) -> (SlotSet, Vec<(Slot, usize)>) {
        let state = self.state.read();
        (self.full.snapshot(), state.cutoffs.iter().collect())
    }

    #[cfg(test)]
    pub(super) fn shape_version_for_test(&self) -> u64 {
        self.shape_version.load(Ordering::Acquire)
    }

    /// Replaces this node's coverage without any version change.
    #[cfg(test)]
    pub(super) fn set_coverage_for_test(&self, full: &[Slot], cutoffs: &[(Slot, usize)]) {
        let mut state = self.write_state();
        self.full.remove_all(&self.full.snapshot());
        for &slot in full {
            self.full.insert(slot);
        }
        state.cutoffs = Default::default();
        for &(slot, cutoff) in cutoffs {
            state.cutoffs.insert(slot, cutoff);
        }
    }

    #[cfg(test)]
    pub(super) fn attach_child_for_test(&self, child: SharedNode) {
        let key = child.state.read().edge[0].0;
        let _gate = self.shape_gate.write();
        self.children.insert(key, child);
        self.internal.store(true, Ordering::Release);
        self.shape_version.fetch_add(1, Ordering::Release);
    }

    /// Splits this node at `pos` as a store would, returning the suffix.
    #[cfg(test)]
    pub(super) fn split_for_test(&self, pos: usize) -> SharedNode {
        let version = self.shape_version.load(Ordering::Acquire);
        self.apply_edge_shape_update(version, |state, _children| {
            (self.split_at_locked(state, pos).suffix, true)
        })
        .expect("no concurrent shape change in a test")
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

    pub(super) fn promote_slot_to_full_edge(&self, slot: Slot) -> bool {
        // NOTE(perf): This path is anchor-only today. Removing its shape read
        // did not improve throughput; re-evaluate if non-anchor callers appear.
        self.promote_full(slot, None) == Some(true)
    }

    /// Marks `slot` as covering the whole edge, if the shape is still `expected_version`.
    /// Returns whether its coverage changed, or `None` if the shape moved.
    ///
    /// The bit is set under the shared gate alone: every update that moves bits or
    /// reads them to change the shape holds the exclusive gate, and readers already
    /// load bits under the state lock. The state write lock is taken only to delete
    /// a stale cutoff, after the bit is set, so readers never miss the slot.
    fn promote_full(&self, slot: Slot, expected_version: Option<u64>) -> Option<bool> {
        let _gate = self.shape_gate.read();
        // Validate before trusting a set bit: a split also sets it, when it promotes the
        // slot's cutoff on the prefix, and then leaves only a cutoff on the suffix.
        if expected_version
            .is_some_and(|version| self.shape_version.load(Ordering::Acquire) != version)
        {
            return None;
        }
        if self.full.contains(slot) || !self.full.insert(slot) {
            return Some(false);
        }
        if self.has_cutoffs.load(Ordering::Acquire) {
            self.write_state().cutoffs.remove(slot);
        }
        Some(true)
    }

    pub(super) fn remove_slots_and_snapshot_children(&self, slots: &SlotSet) -> Vec<SharedNode> {
        let _gate = self.shape_gate.write();
        let mut state = self.write_state();
        let old_cutoff_len = state.cutoffs.len();
        state.cutoffs.retain(|slot| !slots.contains(slot));
        let removed_full = self.full.remove_all(slots);
        let removed_worker = removed_full || old_cutoff_len != state.cutoffs.len();
        let should_clear_children = removed_worker && self.full.is_empty();

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

    fn has_any_workers(&self) -> bool {
        self.state.read().has_any_workers(&self.full)
    }

    pub(super) fn live_children(&self) -> Vec<SharedNode> {
        let _gate = self.shape_gate.read();
        self.children
            .values_snapshot()
            .into_iter()
            .filter(|child| child.has_any_workers() || !child.children.is_empty())
            .collect()
    }

    pub(super) fn dump_snapshot(&self, table: &SlotTable) -> DumpNodeSnapshot {
        let _gate = self.shape_gate.read();
        let state = self.state.read();
        let live_children: Vec<_> = self
            .children
            .values_snapshot()
            .into_iter()
            .filter(|child| child.has_any_workers() || !child.children.is_empty())
            .collect();

        let full = self.full.snapshot();
        let can_merge = state.cutoffs.is_empty()
            && live_children.len() == 1
            && live_children[0].has_full_coverage_only_matching(&full);

        DumpNodeSnapshot {
            edge: state.edge.clone(),
            full_edge_workers: full.iter().filter_map(|slot| table.owner(slot)).collect(),
            worker_cutoffs: state
                .cutoffs
                .iter()
                .filter_map(|(slot, cutoff)| Some((table.owner(slot)?, cutoff)))
                .collect(),
            live_children,
            has_any_workers: state.has_any_workers(&self.full),
            children_empty: self.children.is_empty(),
            can_merge,
        }
    }

    fn has_full_coverage_only_matching(&self, slots: &SlotSet) -> bool {
        let state = self.state.read();
        state.cutoffs.is_empty() && self.full.snapshot() == *slots && !slots.is_empty()
    }

    /// The edge's hashes and how far each slot in `slots` covers them, read under one lock.
    pub(super) fn covered_prefixes(
        &self,
        slots: impl Iterator<Item = Option<Slot>>,
    ) -> (Vec<ExternalSequenceBlockHash>, Vec<usize>) {
        let state = self.state.read();
        let cutoffs = slots
            .map(|slot| slot.map_or(0, |slot| state.current_cutoff(&self.full, slot)))
            .collect();
        let hashes = state.edge.iter().map(|&(_, hash)| hash).collect();
        (hashes, cutoffs)
    }

    pub(super) fn lookup_hashes_for_slot_repair(
        &self,
        slot: Slot,
        hash: ExternalSequenceBlockHash,
        direction: LookupRepairDirection,
    ) -> Vec<ExternalSequenceBlockHash> {
        let state = self.state.read();
        let Some(&pos) = state.edge_index.get(&hash) else {
            return Vec::new();
        };
        let cutoff = state.current_cutoff(&self.full, slot).min(state.edge.len());
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
        slot: Slot,
        parent_hash: ExternalSequenceBlockHash,
    ) -> Option<UncoveredParent> {
        let _gate = self.shape_gate.read();
        let state = self.state.read();
        let &pos = state.edge_index.get(&parent_hash)?;
        if state.covers_pos(&self.full, slot, pos) {
            return None;
        }
        Some(UncoveredParent {
            pos,
            cutoff: state.current_cutoff(&self.full, slot),
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
                let cutoff = parent_pos + 1 + blocks.len();
                ParentEdgePlanAction::ReuseExistingEdge {
                    cutoff,
                    covers_edge: cutoff >= state.edge.len(),
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
        slot: Slot,
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
            ParentEdgePlanAction::ReuseExistingEdge {
                covers_edge: true, ..
            } => self
                .promote_full(slot, Some(plan.shape_version))
                .map_or(ParentEdgeAction::Stale, |coverage_changed| {
                    ParentEdgeAction::ReuseExistingEdge { coverage_changed }
                }),
            ParentEdgePlanAction::ReuseExistingEdge { cutoff, .. } => self
                .cover_prefix_with_version(slot, cutoff, plan.shape_version)
                .map_or(ParentEdgeAction::Stale, |coverage_changed| {
                    ParentEdgeAction::ReuseExistingEdge { coverage_changed }
                }),
            // NOTE(perf): An additional sticky-internal rejection before this
            // commit did not improve throughput. The check inside the gate
            // closes the split race.
            ParentEdgePlanAction::ReuseSuffixAndExtendLeaf { append_start } => self
                .apply_edge_shape_update(plan.shape_version, |state, _children| {
                    if !self.internal.load(Ordering::Acquire) {
                        state.append_blocks_to_leaf(&self.full, slot, &blocks[append_start..]);
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

    pub(super) fn cover_prefix_with_version(
        &self,
        slot: Slot,
        cutoff: usize,
        shape_version: u64,
    ) -> Option<bool> {
        self.apply_metadata_update(shape_version, |state| {
            state.cover_prefix(&self.full, slot, cutoff)
        })
    }

    pub(super) fn promote_to_full_with_version(
        &self,
        slot: Slot,
        shape_version: u64,
    ) -> Option<bool> {
        self.promote_full(slot, Some(shape_version))
    }

    pub(super) fn split_for_store_tail(
        &self,
        slot: Slot,
        split_pos: usize,
        tail_first_local: LocalBlockHash,
        tail_node: SharedNode,
        shape_version: u64,
    ) -> SplitStoreOutcome {
        self.apply_edge_shape_update(shape_version, |state, children| {
            let split = self.split_at_locked(state, split_pos);
            state.promote_to_full(&self.full, slot);
            children.insert(tail_first_local, tail_node.clone());
            (SplitStoreOutcome::Done { split, tail_node }, true)
        })
        .unwrap_or(SplitStoreOutcome::Stale)
    }

    pub(super) fn try_extend_leaf_with_version(
        &self,
        slot: Slot,
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
            if !state.tail_hash_is(parent_hash) || !state.covers_pos(&self.full, slot, old_len - 1)
            {
                return (false, false);
            }

            state.append_blocks_to_leaf(&self.full, slot, blocks);
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
                if state.edge_index.contains_key(&hash) {
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
        // The suffix inherits this node's full coverage as it was before the split
        // promotes partial ranks that reach the split point.
        let suffix_full = FullCoverage::from_set(&self.full.snapshot());
        let suffix_state = state.split_off_suffix(&self.full, pos);
        let suffix_first_local = suffix_state.edge[0].0;
        let suffix_children = self.children.transfer_for_split();

        let suffix = Arc::new(Node::from_parts(suffix_state, suffix_full, suffix_children));
        self.children.insert(suffix_first_local, suffix.clone());
        self.internal.store(true, Ordering::Release);

        SplitLookupData { suffix }
    }

    pub(super) fn remove_worker_for_hashes(
        &self,
        slot: Slot,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Option<RemoveBatchOutcome> {
        let _gate = self.shape_gate.write();
        let mut state = self.write_state();
        let mut min_match = None;
        let mut unmatched_hashes = Vec::new();

        for &hash in block_hashes {
            match state.edge_index.get(&hash).copied() {
                Some(pos) => {
                    if min_match.is_none_or(|(min_pos, _)| pos < min_pos) {
                        min_match = Some((pos, hash));
                    }
                }
                None => unmatched_hashes.push(hash),
            }
        }

        let (pos, block_hash) = min_match?;
        let outcome = state.remove_worker_at_pos(&self.full, slot, pos, block_hash);
        let should_clear_children = self.full.is_empty();
        drop(state);
        self.clear_children_if_unreachable(should_clear_children);
        Some(RemoveBatchOutcome {
            stale_hashes: outcome.stale_hashes,
            unmatched_hashes,
        })
    }

    #[cfg_attr(feature = "profile", inline(never))]
    pub(super) fn find_match_step<'g, S: HashSequence>(
        &'g self,
        input: FindStepInput<'_, S>,
        guard: &'g Guard,
    ) -> FindStepOutcome<'g> {
        let FindStepInput {
            sequence,
            seq_pos,
            first_node,
            prev_depth,
            prev_edge_last_hash,
            table,
            active,
            scores,
            mut last_matched_hashes,
            kv_transfer_chain,
        } = input;

        // NOTE: This read intentionally does not take shape_gate. A concurrent
        // split can make the edge snapshot and child lookup come from adjacent
        // tree shapes; find_matches tolerates that brief best-effort race.
        let state = self.state.read();
        let edge_len = state.edge.len();
        let walk_len = edge_len.min(sequence.len() - seq_pos);

        let mut edge_match_len = 1;
        for i in 1..walk_len {
            if state.edge[i].0 != sequence.at(seq_pos + i) {
                break;
            }
            edge_match_len += 1;
        }

        let edge_hash_at = |depth: usize| -> ExternalSequenceBlockHash {
            debug_assert!(depth > 0 && depth <= state.edge.len());
            state.edge[depth - 1].1
        };

        if let Some(block_hashes) = kv_transfer_chain {
            block_hashes.extend(
                state
                    .edge
                    .iter()
                    .take(edge_match_len)
                    .map(|(_, block_hash)| *block_hash),
            );
        }

        if first_node {
            active.load(&self.full);
            // Every scored worker is covered by the first node, so its coverage bounds
            // the result size; reserving avoids repeated rehash growth per query.
            let scored_bound = active.count() + state.cutoffs.len();
            scores.scores.reserve(scored_bound);
            if let Some(last_matched_hashes) = last_matched_hashes.as_deref_mut() {
                last_matched_hashes.reserve(scored_bound);
            }
            for (slot, cutoff) in state.cutoffs.iter() {
                let contribution = cutoff.min(edge_match_len);
                if contribution == 0 {
                    continue;
                }
                let Some(worker) = table.owner(slot) else {
                    continue;
                };
                scores.scores.insert(worker, contribution as u32);
                record_last_matched_hash(
                    &mut last_matched_hashes,
                    worker,
                    edge_hash_at(contribution),
                );
            }
        } else {
            // Ranks that do not cover this whole edge stop here, scored by their cutoff,
            // if any, past the previous depth.
            active.intersect(&self.full, |slot| {
                let Some(worker) = table.owner(slot) else {
                    return;
                };
                let effective = state
                    .cutoffs
                    .get(slot)
                    .map_or(0, |cutoff| cutoff.min(edge_match_len));
                scores.scores.insert(worker, prev_depth + effective as u32);
                if effective > 0 {
                    record_last_matched_hash(
                        &mut last_matched_hashes,
                        worker,
                        edge_hash_at(effective),
                    );
                } else if let Some(hash) = prev_edge_last_hash {
                    record_last_matched_hash(&mut last_matched_hashes, worker, hash);
                }
            });
        }

        let active_count = active.count();
        let next_child = if edge_match_len == edge_len
            && active_count > 0
            && seq_pos + edge_match_len < sequence.len()
        {
            self.children
                .get_ref(&sequence.at(seq_pos + edge_match_len), guard)
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
        // Pin after the gate so a parked wait does not hold back epoch reclamation.
        let _guard = crossbeam_epoch::pin();
        let still_attached = self
            .children
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(&current, child));
        if !still_attached {
            return;
        }

        let Some(_child_gate) = child.shape_gate.try_write() else {
            return;
        };
        if child.has_any_workers() || !child.children.is_empty() {
            return;
        }
        // The parent map and the caller's candidate must hold the only live references;
        // any other holder is a writer that may still cover or extend this node. Readers
        // borrow children without a reference, so this can unlink a leaf a read is
        // standing on. That read sees no workers here and so cannot overcount, and the
        // epoch keeps the node allocated until it unpins.
        if Self::live_strong_count(child) != Some(2) {
            return;
        }

        self.children.remove(&key);
        self.shape_version.fetch_add(1, Ordering::Release);
    }
}

/// The state write lock. Releasing it republishes whether any cutoffs remain.
struct StateWriteGuard<'a> {
    has_cutoffs: &'a AtomicBool,
    state: RwLockWriteGuard<'a, CrtcNodeState>,
}

impl std::ops::Deref for StateWriteGuard<'_> {
    type Target = CrtcNodeState;

    fn deref(&self) -> &CrtcNodeState {
        &self.state
    }
}

impl std::ops::DerefMut for StateWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut CrtcNodeState {
        &mut self.state
    }
}

impl Drop for StateWriteGuard<'_> {
    fn drop(&mut self) {
        // Runs before the lock is released, so the flag never lags a reader of the state.
        self.has_cutoffs
            .store(!self.state.cutoffs.is_empty(), Ordering::Release);
    }
}

pub(super) struct CleanupEdge {
    pub(super) parent: Weak<Node>,
    pub(super) key: LocalBlockHash,
    pub(super) child: Weak<Node>,
}
