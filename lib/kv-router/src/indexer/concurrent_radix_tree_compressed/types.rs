// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::node::Node;
use crate::protocols::*;

/// Thread-safe shared reference to a Node.
pub(super) type SharedNode = Arc<Node>;

/// Per-worker block-hash -> node map.
///
/// Maps each `ExternalSequenceBlockHash` to the node whose `edge` contains it.
/// Position within the edge is resolved via `Node::edge_index` (O(1)) rather than
/// stored here, keeping the map compact and correct across concurrent splits.
pub(super) type WorkerLookup = FxHashMap<ExternalSequenceBlockHash, SharedNode>;

pub(super) struct MatchWalkResult {
    pub(super) active: FxHashSet<WorkerWithDpRank>,
    pub(super) matched_depth: u32,
    pub(super) prev_edge_last_hash: Option<ExternalSequenceBlockHash>,
}

// For short anchored reads this avoids a Vec allocation. For long suffixes,
// materializing once is faster than paying virtual-index branching throughout
// the radix walk.
pub(super) const MAX_NO_COPY_ANCHORED_SUFFIX_BLOCKS: usize = 32;

pub(super) trait HashSequence {
    fn len(&self) -> usize;
    fn at(&self, index: usize) -> LocalBlockHash;
}

pub(super) struct SliceHashSequence<'a>(pub(super) &'a [LocalBlockHash]);

impl HashSequence for SliceHashSequence<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn at(&self, index: usize) -> LocalBlockHash {
        self.0[index]
    }
}

pub(super) struct AnchoredHashSequence<'a> {
    pub(super) head: LocalBlockHash,
    pub(super) tail: &'a [LocalBlockHash],
}

impl HashSequence for AnchoredHashSequence<'_> {
    fn len(&self) -> usize {
        self.tail.len() + 1
    }

    fn at(&self, index: usize) -> LocalBlockHash {
        if index == 0 {
            self.head
        } else {
            self.tail[index - 1]
        }
    }
}

pub(super) struct DumpNodeSnapshot {
    pub(super) edge: Vec<(LocalBlockHash, ExternalSequenceBlockHash)>,
    pub(super) full_edge_workers: Vec<WorkerWithDpRank>,
    pub(super) worker_cutoffs: Vec<(WorkerWithDpRank, usize)>,
    pub(super) live_children: Vec<SharedNode>,
    pub(super) has_any_workers: bool,
    pub(super) children_empty: bool,
    pub(super) can_merge: bool,
}

pub(super) struct UncoveredParent {
    pub(super) pos: usize,
    pub(super) cutoff: usize,
}

pub(super) struct FindStepInput<'a, S: HashSequence> {
    pub(super) sequence: &'a S,
    pub(super) seq_pos: usize,
    pub(super) first_node: bool,
    pub(super) prev_depth: u32,
    pub(super) prev_edge_last_hash: Option<ExternalSequenceBlockHash>,
    pub(super) active: &'a mut FxHashSet<WorkerWithDpRank>,
    pub(super) active_count: usize,
    pub(super) scores: &'a mut OverlapScores,
    pub(super) last_matched_hashes:
        Option<&'a mut FxHashMap<WorkerWithDpRank, ExternalSequenceBlockHash>>,
}

pub(super) struct FindStepOutcome {
    pub(super) edge_len: usize,
    pub(super) edge_match_len: usize,
    pub(super) active_count: usize,
    pub(super) next_child: Option<SharedNode>,
    pub(super) prev_edge_last_hash: Option<ExternalSequenceBlockHash>,
}

/// Data returned by a split for deferred lookup updates.
pub(super) struct SplitLookupData {
    pub(super) suffix: SharedNode,
}

#[derive(Clone, Copy)]
pub(super) enum LookupRepairDirection {
    TowardTail,
    TowardHead,
}

pub(super) struct StoreInsertOutcome {
    pub(super) duplicate_store: bool,
}

pub(super) enum StoreParentResolution {
    InsertFrom {
        parent: SharedNode,
        parent_is_anchor: bool,
    },
    ReusedExistingEdge {
        node: SharedNode,
        coverage_changed: bool,
    },
}

pub(super) enum ParentEdgeAction {
    Stale,
    ReuseExistingEdge { coverage_changed: bool },
    InsertFromParent(Option<SplitLookupData>),
}

pub(super) struct ParentEdgePlan {
    pub(super) shape_version: u64,
    pub(super) action: ParentEdgePlanAction,
}

pub(super) enum ParentEdgePlanAction {
    InsertFromParent,
    ReuseExistingEdge { cutoff: usize },
    ReuseSuffixAndExtendLeaf { append_start: usize },
    Split { split_pos: usize },
}

pub(super) struct ChildEdgeScan {
    pub(super) shape_version: u64,
    pub(super) edge_len: usize,
    pub(super) match_len: usize,
    pub(super) block_hash_mismatch: Option<(ExternalSequenceBlockHash, ExternalSequenceBlockHash)>,
}

pub(super) enum ParentChildPlan {
    Stale,
    StaleParent { hash: ExternalSequenceBlockHash },
    Descend(SharedNode),
    MissingChild { shape_version: u64 },
}

pub(super) enum InsertChildOutcome {
    Stale,
    Existing(SharedNode),
    Inserted(SharedNode),
}

pub(super) enum SplitStoreOutcome {
    Stale,
    Done {
        split: SplitLookupData,
        tail_node: SharedNode,
    },
}

pub(super) enum StoreInsertStep {
    RetryParent {
        parent: SharedNode,
        parent_is_anchor: bool,
    },
    Descend(SharedNode),
    Done(StoreInsertOutcome),
}

pub(super) enum ChildInsertStep {
    Descend {
        edge_len: usize,
        last_ext_hash: ExternalSequenceBlockHash,
        duplicate_store: bool,
    },
    Done(StoreInsertOutcome),
}
