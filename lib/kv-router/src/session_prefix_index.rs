// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session lineage retained independently of physical cache eviction.

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};

use crate::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

/// Logical session identity as owned by this index.
pub type SessionId = String;

new_key_type! {
    /// Generational handle to a [`LogicalNode`] in the arena.
    pub struct NodeId;
}

/// One block and its liveness links in the logical session forest.
#[derive(Clone, Copy, Debug)]
pub struct LogicalNode {
    block_hash: ExternalSequenceBlockHash,
    parent: Option<NodeId>,
    frontier_refs: u32,
    child_count: u32,
}

impl LogicalNode {
    pub fn block_hash(&self) -> ExternalSequenceBlockHash {
        self.block_hash
    }

    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    pub fn frontier_refs(&self) -> u32 {
        self.frontier_refs
    }

    pub fn child_count(&self) -> u32 {
        self.child_count
    }
}

/// Errors surfaced by [`SessionPrefixIndexer`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SessionPrefixIndexError {
    /// A lineage query named an unknown anchor hash.
    #[error("unknown anchor block hash {0:?}")]
    UnknownAnchor(ExternalSequenceBlockHash),

    /// A block was attached beneath a conflicting parent.
    #[error("block {block:?} is already parented elsewhere")]
    ConflictingParent { block: ExternalSequenceBlockHash },

    /// A graft would create a parent cycle.
    #[error("block {block:?} would become its own ancestor")]
    CyclicParent { block: ExternalSequenceBlockHash },
}

/// Thread-safe session-aware logical prefix index.
#[derive(Debug, Default)]
pub struct SessionPrefixIndexer {
    state: RwLock<IndexState>,
}

#[derive(Debug, Default)]
struct SessionEntry {
    worker_frontiers: FxHashMap<WorkerWithDpRank, FxHashSet<NodeId>>,
}

#[derive(Debug, Default)]
struct IndexState {
    nodes: SlotMap<NodeId, LogicalNode>,
    hash_to_node: FxHashMap<ExternalSequenceBlockHash, NodeId>,
    session_to_worker_frontiers: HashMap<SessionId, SessionEntry>,
    worker_frontier_to_sessions: FxHashMap<WorkerWithDpRank, FxHashMap<NodeId, HashSet<SessionId>>>,
}

impl SessionPrefixIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_node_from_hash(&self, block_hash: ExternalSequenceBlockHash) -> Option<NodeId> {
        self.state.read().hash_to_node.get(&block_hash).copied()
    }

    pub fn get_node(&self, node_id: NodeId) -> Option<LogicalNode> {
        self.state.read().nodes.get(node_id).copied()
    }

    /// Returns unordered worker-qualified frontier nodes.
    pub fn get_session_frontiers(&self, session_id: &str) -> Vec<(WorkerWithDpRank, NodeId)> {
        self.state
            .read()
            .session_to_worker_frontiers
            .get(session_id)
            .map(|entry| {
                entry
                    .worker_frontiers
                    .iter()
                    .flat_map(|(&worker, frontiers)| {
                        frontiers.iter().map(move |&frontier| (worker, frontier))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns root-first chains for one worker, optionally truncated at an anchor.
    pub fn get_session_block_lineage(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        anchor_hash: Option<ExternalSequenceBlockHash>,
    ) -> Result<Vec<Vec<ExternalSequenceBlockHash>>, SessionPrefixIndexError> {
        let state = self.state.read();

        let anchor_node = match anchor_hash {
            Some(hash) => Some(
                state
                    .hash_to_node
                    .get(&hash)
                    .copied()
                    .ok_or(SessionPrefixIndexError::UnknownAnchor(hash))?,
            ),
            None => None,
        };

        let Some(frontiers) = state
            .session_to_worker_frontiers
            .get(session_id)
            .and_then(|entry| entry.worker_frontiers.get(&worker))
        else {
            return Ok(Vec::new());
        };

        let mut lineages = Vec::with_capacity(frontiers.len());
        for &frontier in frontiers {
            let path = state.path_to_root(frontier);
            let start = match anchor_node {
                Some(anchor) => match path.iter().position(|&node| node == anchor) {
                    Some(position) => position,
                    None => continue,
                },
                None => 0,
            };
            lineages.push(
                path[start..]
                    .iter()
                    .map(|&node| state.nodes[node].block_hash)
                    .collect(),
            );
        }
        Ok(lineages)
    }

    /// Records a route-time match and reports whether the frontier advanced.
    pub fn update_session_from_match(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        matched_hash: ExternalSequenceBlockHash,
    ) -> Result<bool, SessionPrefixIndexError> {
        let mut state = self.state.write();
        let node = state.resolve_or_insert_root(matched_hash);
        Ok(state.advance_frontier(session_id, worker, node))
    }

    /// Records a stored block chain and reports whether the frontier advanced.
    pub fn update_session_from_stored_blocks(
        &self,
        session_id: &str,
        worker: WorkerWithDpRank,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<bool, SessionPrefixIndexError> {
        if block_hashes.is_empty() {
            return Ok(false);
        }

        let mut state = self.state.write();
        // Reject the whole update before mutating the arena.
        state.validate_chain(parent_hash, block_hashes)?;

        let mut parent = parent_hash.map(|hash| state.resolve_or_insert_root(hash));
        for &block_hash in block_hashes {
            let node = match state.hash_to_node.get(&block_hash).copied() {
                Some(existing) => {
                    // Graft nodes previously known only as roots.
                    if let (None, Some(expected)) = (state.nodes[existing].parent, parent) {
                        state.nodes[existing].parent = Some(expected);
                        state.nodes[expected].child_count += 1;
                    }
                    existing
                }
                None => state.insert_node(block_hash, parent),
            };
            parent = Some(node);
        }

        let leaf = parent.expect("non-empty block chain always yields a node");
        Ok(state.advance_frontier(session_id, worker, leaf))
    }

    /// Recedes worker-local session frontiers affected by removed blocks.
    pub fn update_session_from_removed_blocks(
        &self,
        worker: WorkerWithDpRank,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> usize {
        if block_hashes.is_empty() {
            return 0;
        }

        self.state
            .write()
            .recede_removed_frontiers(worker, block_hashes)
    }

    /// Removes every session frontier associated with one worker rank.
    pub fn clear_worker_frontiers(&self, worker: WorkerWithDpRank) -> usize {
        self.state.write().clear_worker_frontiers(worker)
    }

    pub fn node_count(&self) -> usize {
        self.state.read().nodes.len()
    }

    pub fn session_count(&self) -> usize {
        self.state.read().session_to_worker_frontiers.len()
    }
}

impl IndexState {
    fn insert_node(
        &mut self,
        block_hash: ExternalSequenceBlockHash,
        parent: Option<NodeId>,
    ) -> NodeId {
        let node = self.nodes.insert(LogicalNode {
            block_hash,
            parent,
            frontier_refs: 0,
            child_count: 0,
        });
        self.hash_to_node.insert(block_hash, node);
        if let Some(parent) = parent {
            self.nodes[parent].child_count += 1;
        }
        node
    }

    // Reject conflicting parents and graft cycles before mutation.
    fn validate_chain(
        &self,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<(), SessionPrefixIndexError> {
        let mut dominators =
            FxHashSet::with_capacity_and_hasher(block_hashes.len(), Default::default());
        if let Some(parent_hash) = parent_hash {
            dominators.insert(parent_hash);
            if let Some(&parent_node) = self.hash_to_node.get(&parent_hash) {
                dominators.extend(
                    self.path_to_root(parent_node)
                        .into_iter()
                        .map(|node| self.nodes[node].block_hash),
                );
            }
        }

        let mut expected_parent = parent_hash;
        for &block_hash in block_hashes {
            if dominators.contains(&block_hash) {
                return Err(SessionPrefixIndexError::CyclicParent { block: block_hash });
            }
            if let Some(&existing) = self.hash_to_node.get(&block_hash)
                && let Some(recorded) = self.nodes[existing].parent
                && Some(self.nodes[recorded].block_hash) != expected_parent
            {
                return Err(SessionPrefixIndexError::ConflictingParent { block: block_hash });
            }
            dominators.insert(block_hash);
            expected_parent = Some(block_hash);
        }
        Ok(())
    }

    fn resolve_or_insert_root(&mut self, block_hash: ExternalSequenceBlockHash) -> NodeId {
        match self.hash_to_node.get(&block_hash).copied() {
            Some(node) => node,
            None => self.insert_node(block_hash, None),
        }
    }

    // Bound parent walks so corrupted cycles cannot hang while holding the lock.
    fn path_to_root(&self, tail: NodeId) -> Vec<NodeId> {
        let limit = self.nodes.len();
        let mut path = Vec::new();
        let mut current = Some(tail);
        while let Some(node) = current {
            if path.len() >= limit {
                debug_assert!(false, "parent cycle in session prefix index forest");
                tracing::error!("session prefix index parent walk exceeded the arena; truncating");
                break;
            }
            path.push(node);
            current = self.nodes[node].parent;
        }
        path.reverse();
        path
    }

    fn is_ancestor_or_self(&self, candidate: NodeId, node: NodeId) -> bool {
        let limit = self.nodes.len();
        let mut current = Some(node);
        let mut steps = 0usize;
        while let Some(walk) = current {
            if walk == candidate {
                return true;
            }
            steps += 1;
            if steps > limit {
                debug_assert!(false, "parent cycle in session prefix index forest");
                tracing::error!("session prefix index ancestry walk exceeded the arena; aborting");
                return false;
            }
            current = self.nodes[walk].parent;
        }
        false
    }

    // Keep only the deepest frontier on each worker-local chain.
    fn advance_frontier(
        &mut self,
        session_id: &str,
        worker: WorkerWithDpRank,
        node: NodeId,
    ) -> bool {
        let already_reached = self
            .session_to_worker_frontiers
            .get(session_id)
            .and_then(|entry| entry.worker_frontiers.get(&worker))
            .is_some_and(|frontiers| {
                frontiers.contains(&node)
                    || frontiers
                        .iter()
                        .any(|&frontier| self.is_ancestor_or_self(node, frontier))
            });
        if already_reached {
            return false;
        }

        let subsumed: Vec<NodeId> = self
            .session_to_worker_frontiers
            .get(session_id)
            .and_then(|entry| entry.worker_frontiers.get(&worker))
            .map(|frontiers| {
                frontiers
                    .iter()
                    .copied()
                    .filter(|&frontier| self.is_ancestor_or_self(frontier, node))
                    .collect()
            })
            .unwrap_or_default();

        for &frontier in &subsumed {
            self.remove_frontier_binding(session_id, worker, frontier);
        }
        self.add_frontier_binding(session_id, worker, node);
        true
    }

    fn add_frontier_binding(&mut self, session_id: &str, worker: WorkerWithDpRank, node: NodeId) {
        let inserted = self
            .session_to_worker_frontiers
            .entry(session_id.to_string())
            .or_default()
            .worker_frontiers
            .entry(worker)
            .or_default()
            .insert(node);
        if !inserted {
            return;
        }

        self.nodes[node].frontier_refs += 1;
        self.worker_frontier_to_sessions
            .entry(worker)
            .or_default()
            .entry(node)
            .or_default()
            .insert(session_id.to_string());
    }

    fn remove_frontier_binding(
        &mut self,
        session_id: &str,
        worker: WorkerWithDpRank,
        node: NodeId,
    ) -> bool {
        let removed = self
            .session_to_worker_frontiers
            .get_mut(session_id)
            .and_then(|entry| entry.worker_frontiers.get_mut(&worker))
            .is_some_and(|frontiers| frontiers.remove(&node));
        if !removed {
            return false;
        }

        self.nodes[node].frontier_refs -= 1;
        self.remove_reverse_frontier(worker, node, session_id);

        let remove_worker = self
            .session_to_worker_frontiers
            .get(session_id)
            .and_then(|entry| entry.worker_frontiers.get(&worker))
            .is_some_and(FxHashSet::is_empty);
        if remove_worker && let Some(entry) = self.session_to_worker_frontiers.get_mut(session_id) {
            entry.worker_frontiers.remove(&worker);
        }
        let remove_session = self
            .session_to_worker_frontiers
            .get(session_id)
            .is_some_and(|entry| entry.worker_frontiers.is_empty());
        if remove_session {
            self.session_to_worker_frontiers.remove(session_id);
        }
        true
    }

    fn remove_reverse_frontier(
        &mut self,
        worker: WorkerWithDpRank,
        node: NodeId,
        session_id: &str,
    ) {
        let remove_node = self
            .worker_frontier_to_sessions
            .get_mut(&worker)
            .and_then(|frontiers| frontiers.get_mut(&node))
            .is_some_and(|sessions| {
                sessions.remove(session_id);
                sessions.is_empty()
            });
        if remove_node && let Some(frontiers) = self.worker_frontier_to_sessions.get_mut(&worker) {
            frontiers.remove(&node);
        }
        let remove_worker = self
            .worker_frontier_to_sessions
            .get(&worker)
            .is_some_and(FxHashMap::is_empty);
        if remove_worker {
            self.worker_frontier_to_sessions.remove(&worker);
        }
    }

    fn recede_removed_frontiers(
        &mut self,
        worker: WorkerWithDpRank,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> usize {
        let removed_nodes: FxHashSet<NodeId> = block_hashes
            .iter()
            .filter_map(|block_hash| self.hash_to_node.get(block_hash).copied())
            .collect();
        if removed_nodes.is_empty() {
            return 0;
        }

        // Examine each frontier path; recede past the removed block and its descendants.
        let affected: Vec<(SessionId, NodeId, Option<NodeId>)> = self
            .worker_frontier_to_sessions
            .get(&worker)
            .map(|frontiers| {
                frontiers
                    .iter()
                    .filter_map(|(&frontier, sessions)| {
                        let path = self.path_to_root(frontier);
                        let shallowest_removed =
                            path.into_iter().find(|node| removed_nodes.contains(node))?;
                        let replacement = self.nodes[shallowest_removed].parent;
                        Some(
                            sessions
                                .iter()
                                .cloned()
                                .map(move |session_id| (session_id, frontier, replacement)),
                        )
                    })
                    .flatten()
                    .collect()
            })
            .unwrap_or_default();

        for (session_id, old_frontier, replacement) in &affected {
            self.remove_frontier_binding(session_id, worker, *old_frontier);
            if let Some(replacement) = *replacement {
                self.advance_frontier(session_id, worker, replacement);
            }
        }
        affected.len()
    }

    fn clear_worker_frontiers(&mut self, worker: WorkerWithDpRank) -> usize {
        let affected: Vec<(SessionId, NodeId)> = self
            .worker_frontier_to_sessions
            .get(&worker)
            .map(|frontiers| {
                frontiers
                    .iter()
                    .flat_map(|(&node, sessions)| {
                        sessions
                            .iter()
                            .cloned()
                            .map(move |session_id| (session_id, node))
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (session_id, node) in &affected {
            self.remove_frontier_binding(session_id, worker, *node);
        }
        affected.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_blocks;

    fn hashes(ids: Vec<u64>) -> Vec<ExternalSequenceBlockHash> {
        make_blocks(ids)
            .into_iter()
            .map(|block| block.block_hash)
            .collect()
    }

    fn worker(worker_id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, 0)
    }

    fn lineage_of(
        indexer: &SessionPrefixIndexer,
        session: &str,
    ) -> Vec<Vec<ExternalSequenceBlockHash>> {
        lineage_on_worker(indexer, session, worker(1))
    }

    fn lineage_on_worker(
        indexer: &SessionPrefixIndexer,
        session: &str,
        worker: WorkerWithDpRank,
    ) -> Vec<Vec<ExternalSequenceBlockHash>> {
        let mut lineages = indexer
            .get_session_block_lineage(session, worker, None)
            .expect("lineage query without an anchor cannot fail");
        lineages.sort();
        lineages
    }

    #[test]
    fn match_creates_lineage_and_repeat_is_idempotent() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        assert!(
            indexer
                .update_session_from_match("s1", worker(1), chain[0])
                .unwrap(),
            "first match must advance the frontier"
        );
        assert_eq!(indexer.node_count(), 1);
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);

        assert!(
            !indexer
                .update_session_from_match("s1", worker(1), chain[0])
                .unwrap(),
            "re-matching the same block is not an advance"
        );
        assert_eq!(
            indexer.node_count(),
            1,
            "a repeated match must not grow the arena"
        );
        assert_eq!(indexer.get_session_frontiers("s1").len(), 1);
    }

    #[test]
    fn deeper_match_replaces_the_shallower_frontier() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        assert_eq!(indexer.get_session_frontiers("s1").len(), 1);

        assert!(
            !indexer
                .update_session_from_match("s1", worker(1), chain[1])
                .unwrap(),
            "a match above the current frontier must not move it"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![chain.clone()]);

        assert!(
            indexer
                .update_session_from_match("s2", worker(1), chain[1])
                .unwrap()
        );
        assert_eq!(lineage_of(&indexer, "s2"), vec![chain[..2].to_vec()]);
    }

    #[test]
    fn stored_blocks_reuse_shared_prefix_nodes() {
        let trunk = hashes(vec![1, 2]);
        let branch = hashes(vec![3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", worker(1), Some(trunk[1]), &branch)
            .unwrap();

        assert_eq!(
            indexer.node_count(),
            3,
            "the shared trunk must be stored once, not per session"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![trunk.clone()]);
        assert_eq!(
            lineage_of(&indexer, "s2"),
            vec![vec![trunk[0], trunk[1], branch[0]]]
        );
    }

    #[test]
    fn stored_blocks_graft_onto_a_node_first_seen_as_a_match() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_match("s1", worker(1), chain[1])
            .unwrap();
        let child = indexer.get_node_from_hash(chain[1]).unwrap();
        assert_eq!(indexer.get_node(child).unwrap().parent(), None);

        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(chain[0]), &chain[1..])
            .unwrap();

        assert_eq!(
            indexer.get_node_from_hash(chain[1]),
            Some(child),
            "grafting must keep the original node handle valid"
        );
        assert_eq!(
            indexer.node_count(),
            2,
            "grafting must not leave a duplicate root behind"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![chain]);
    }

    #[test]
    fn conflicting_parent_is_rejected() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain[..2])
            .unwrap();

        let err = indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(chain[2]), &chain[1..2])
            .expect_err("re-parenting a known block violates the hash invariant");
        assert_eq!(
            err,
            SessionPrefixIndexError::ConflictingParent { block: chain[1] }
        );
    }

    #[test]
    fn branching_session_keeps_one_frontier_per_chain() {
        let trunk = hashes(vec![1]);
        let left = hashes(vec![2]);
        let right = hashes(vec![3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(trunk[0]), &left)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(trunk[0]), &right)
            .unwrap();

        assert_eq!(
            indexer.get_session_frontiers("s1").len(),
            2,
            "each branch keeps its own frontier"
        );
        assert_eq!(
            lineage_of(&indexer, "s1"),
            vec![vec![trunk[0], left[0]], vec![trunk[0], right[0]]]
        );
    }

    #[test]
    fn lineage_anchor_truncates_and_filters() {
        let trunk = hashes(vec![1, 2]);
        let left = hashes(vec![3]);
        let right = hashes(vec![4]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(trunk[1]), &left)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &right)
            .unwrap();

        let anchored = indexer
            .get_session_block_lineage("s1", worker(1), Some(trunk[1]))
            .unwrap();
        assert_eq!(
            anchored,
            vec![vec![trunk[1], left[0]]],
            "anchoring drops chains that miss the anchor and trims the rest"
        );
    }

    #[test]
    fn unknown_anchor_is_an_error_and_unknown_session_is_empty() {
        let chain = hashes(vec![1]);
        let missing = hashes(vec![99]);
        let indexer = SessionPrefixIndexer::new();
        indexer
            .update_session_from_match("s1", worker(1), chain[0])
            .unwrap();

        assert_eq!(
            indexer.get_session_block_lineage("s1", worker(1), Some(missing[0])),
            Err(SessionPrefixIndexError::UnknownAnchor(missing[0]))
        );
        assert_eq!(
            indexer.get_session_block_lineage("unrouted", worker(1), None),
            Ok(Vec::new()),
            "a session with no routed requests is empty, not an error"
        );
    }

    #[test]
    fn eviction_shaped_replay_does_not_lose_lineage() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();
        let before = lineage_of(&indexer, "s1");

        indexer
            .update_session_from_match("s1", worker(1), chain[0])
            .unwrap();

        assert_eq!(
            lineage_of(&indexer, "s1"),
            before,
            "re-matching a shallow block must not truncate the recorded lineage"
        );
        assert_eq!(indexer.node_count(), 2);
    }

    #[test]
    fn removal_recedes_only_the_affected_worker_frontier() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        for target in [worker(1), worker(2)] {
            indexer
                .update_session_from_stored_blocks("s1", target, None, &chain)
                .unwrap();
        }

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(2), &chain[1..]),
            1
        );
        assert_eq!(
            lineage_on_worker(&indexer, "s1", worker(1)),
            vec![chain.clone()]
        );
        assert_eq!(
            lineage_on_worker(&indexer, "s1", worker(2)),
            vec![vec![chain[0]]]
        );
    }

    #[test]
    fn batched_removal_recedes_all_sessions_and_is_idempotent() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        for session in ["s1", "s2"] {
            indexer
                .update_session_from_stored_blocks(session, worker(1), None, &chain)
                .unwrap();
        }

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[chain[2], chain[1]]),
            2
        );
        for session in ["s1", "s2"] {
            assert_eq!(lineage_of(&indexer, session), vec![vec![chain[0]]]);
        }
        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[chain[1], chain[2]]),
            0,
            "replaying the same removal must not recede the frontier again"
        );
    }

    #[test]
    fn interior_removal_recedes_before_later_frontier_removal() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[chain[1]]),
            1
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);

        assert_eq!(
            indexer.update_session_from_removed_blocks(worker(1), &[chain[2]]),
            0,
            "removing the old tail must not restore its already-removed parent"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);
    }

    #[test]
    fn clear_removes_only_one_workers_frontiers() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        for target in [worker(1), worker(2)] {
            indexer
                .update_session_from_stored_blocks("s1", target, None, &chain)
                .unwrap();
        }

        assert_eq!(indexer.clear_worker_frontiers(worker(2)), 1);
        assert!(lineage_on_worker(&indexer, "s1", worker(2)).is_empty());
        assert_eq!(lineage_on_worker(&indexer, "s1", worker(1)), vec![chain]);
        assert_eq!(indexer.node_count(), 2, "clear preserves logical topology");
    }

    #[test]
    fn a_chain_that_would_close_a_cycle_is_rejected() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &chain)
            .unwrap();

        let err = indexer
            .update_session_from_stored_blocks("s1", worker(1), Some(chain[2]), &chain[..1])
            .expect_err("grafting an ancestor under its own descendant must fail");
        assert!(
            matches!(
                err,
                SessionPrefixIndexError::CyclicParent { block } if block == chain[0]
            ),
            "expected CyclicParent for the offending block, got {err:?}"
        );

        assert_eq!(
            lineage_of(&indexer, "s1"),
            vec![chain.clone()],
            "a rejected chain must not half-apply"
        );
    }

    #[test]
    fn a_block_repeated_within_one_chain_is_rejected() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        let repeating = vec![chain[0], chain[1], chain[0]];
        let err = indexer
            .update_session_from_stored_blocks("s1", worker(1), None, &repeating)
            .expect_err("a chain that revisits its own block must fail");
        assert!(
            matches!(
                err,
                SessionPrefixIndexError::CyclicParent { block } if block == chain[0]
            ),
            "expected CyclicParent for the repeated block, got {err:?}"
        );
        assert!(
            lineage_of(&indexer, "s1").is_empty(),
            "a rejected chain must not create any nodes"
        );
    }
}
