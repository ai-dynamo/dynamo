// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session lineage retained independently of physical cache eviction.
//! Hash-keyed nodes and parent edges are shared across sessions, so a lineage may contain ancestors learned from another session and does not prove this session matched every block.
//!
//! Final-session removal, an LRU session cap, and a lineage-depth cap bound retention.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashMap},
};

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};

use crate::protocols::ExternalSequenceBlockHash;

/// Logical session identity as owned by this index.
pub type SessionId = String;

/// Default tracked-session ceiling before LRU eviction.
pub const DEFAULT_MAX_SESSIONS: usize = 16_384;

/// Maximum nodes retained in any session-frontier lineage.
pub const DEFAULT_MAX_SESSION_DEPTH: usize = 1_024;

new_key_type! {
    /// Generational handle to a [`LogicalNode`] in the arena.
    pub struct NodeId;
}

/// One block and its liveness links in the logical session forest.
#[derive(Clone, Copy, Debug)]
pub struct LogicalNode {
    block_hash: ExternalSequenceBlockHash,
    parent: Option<NodeId>,
    parent_edge_refs: u32,
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

// None distinguishes an untouched entry from touch sequence zero.
#[derive(Debug, Default)]
struct SessionEntry {
    frontiers: FxHashMap<NodeId, SessionFrontier>,
    last_touch: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SessionFrontier {
    retained_root: NodeId,
    len: usize,
}

#[derive(Debug)]
struct IndexState {
    nodes: SlotMap<NodeId, LogicalNode>,
    hash_to_node: FxHashMap<ExternalSequenceBlockHash, NodeId>,
    sessions: HashMap<SessionId, SessionEntry>,
    // Kept in lockstep with SessionEntry::last_touch.
    lru: BTreeMap<u64, SessionId>,
    next_touch: u64,
    max_sessions: usize,
    max_session_depth: usize,
}

impl Default for IndexState {
    fn default() -> Self {
        Self {
            nodes: SlotMap::default(),
            hash_to_node: FxHashMap::default(),
            sessions: HashMap::default(),
            lru: BTreeMap::default(),
            next_touch: 0,
            max_sessions: DEFAULT_MAX_SESSIONS,
            max_session_depth: DEFAULT_MAX_SESSION_DEPTH,
        }
    }
}

impl SessionPrefixIndexer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an index and clamps a zero session cap to one.
    pub fn with_max_sessions(max_sessions: usize) -> Self {
        Self::with_limits(max_sessions, DEFAULT_MAX_SESSION_DEPTH)
    }

    fn with_limits(max_sessions: usize, max_session_depth: usize) -> Self {
        Self {
            state: RwLock::new(IndexState {
                max_sessions: max_sessions.max(1),
                max_session_depth: max_session_depth.max(1),
                ..IndexState::default()
            }),
        }
    }

    pub fn max_sessions(&self) -> usize {
        self.state.read().max_sessions
    }

    pub fn get_node_from_hash(&self, block_hash: ExternalSequenceBlockHash) -> Option<NodeId> {
        self.state.read().hash_to_node.get(&block_hash).copied()
    }

    pub fn get_node(&self, node_id: NodeId) -> Option<LogicalNode> {
        self.state.read().nodes.get(node_id).copied()
    }

    /// Returns unordered frontier nodes, or an empty vector for an unknown session.
    pub fn get_session_frontiers(&self, session_id: &str) -> Vec<NodeId> {
        self.state
            .read()
            .sessions
            .get(session_id)
            .map(|entry| entry.frontiers.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Returns root-first chains, optionally filtered and truncated at an anchor.
    /// Hash-keyed nodes and parent edges are shared, so a chain may include ancestors learned from another session and does not prove this session matched every block.
    pub fn get_session_block_lineage(
        &self,
        session_id: &str,
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

        let Some(entry) = state.sessions.get(session_id) else {
            return Ok(Vec::new());
        };

        let mut lineages = Vec::with_capacity(entry.frontiers.len());
        for (&frontier, session_frontier) in &entry.frontiers {
            let path = state.path_from_retained_root(frontier, *session_frontier);
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
        matched_hash: ExternalSequenceBlockHash,
    ) -> Result<bool, SessionPrefixIndexError> {
        let mut state = self.state.write();
        let continuation = state.deepest_frontier(session_id);
        let mut attached = false;
        let node = match state.hash_to_node.get(&matched_hash).copied() {
            Some(node) => {
                let already_reached = state.session_has_reached(session_id, node);
                if let Some(parent) = continuation
                    && !already_reached
                    && state.nodes[node].parent.is_none()
                    && !state.is_ancestor_or_self(node, parent)
                {
                    state.nodes[node].parent = Some(parent);
                    state.nodes[parent].child_count = state.nodes[parent]
                        .child_count
                        .checked_add(1)
                        .expect("a parent child count must not overflow");
                    attached = true;
                }
                node
            }
            None => {
                attached = continuation.is_some();
                state.insert_node(matched_hash, continuation)
            }
        };
        let advanced = state.advance_match_frontier(session_id, node, continuation);
        if attached {
            state.detach_unretained_parent_edge(node);
        }
        Ok(advanced)
    }

    /// Records a stored block chain and reports whether the frontier advanced.
    /// KV events do not yet supply the session identity required by this API.
    pub fn update_session_from_stored_blocks(
        &self,
        session_id: &str,
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
        let mut attached_nodes = Vec::new();
        for &block_hash in block_hashes {
            let node = match state.hash_to_node.get(&block_hash).copied() {
                Some(existing) => {
                    // Graft nodes previously known only as roots.
                    if let (None, Some(expected)) = (state.nodes[existing].parent, parent) {
                        state.nodes[existing].parent = Some(expected);
                        state.nodes[expected].child_count = state.nodes[expected]
                            .child_count
                            .checked_add(1)
                            .expect("a parent child count must not overflow");
                        attached_nodes.push(existing);
                    }
                    existing
                }
                None => {
                    let node = state.insert_node(block_hash, parent);
                    if parent.is_some() {
                        attached_nodes.push(node);
                    }
                    node
                }
            };
            parent = Some(node);
        }

        let leaf = parent.expect("non-empty block chain always yields a node");
        let advanced = state.advance_frontier(session_id, leaf);
        if !attached_nodes.is_empty() {
            state.refresh_session_cutoffs(session_id);
        }
        for node in attached_nodes {
            state.detach_unretained_parent_edge(node);
        }
        Ok(advanced)
    }

    /// Removes a session and reclaims its unshared nodes.
    pub fn remove_session(&self, session_id: &str) -> bool {
        let mut state = self.state.write();
        state.drop_session(session_id)
    }

    pub fn node_count(&self) -> usize {
        self.state.read().nodes.len()
    }

    pub fn session_count(&self) -> usize {
        self.state.read().sessions.len()
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
            parent_edge_refs: 0,
            frontier_refs: 0,
            child_count: 0,
        });
        self.hash_to_node.insert(block_hash, node);
        if let Some(parent) = parent {
            self.nodes[parent].child_count = self.nodes[parent]
                .child_count
                .checked_add(1)
                .expect("a parent child count must not overflow");
        }
        node
    }

    // Reject conflicting parents and graft cycles before mutation.
    fn validate_chain(
        &self,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<(), SessionPrefixIndexError> {
        let mut dominators: FxHashSet<ExternalSequenceBlockHash> = FxHashSet::default();
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

    // Public removal and capacity eviction share this reclamation path.
    fn drop_session(&mut self, session_id: &str) -> bool {
        let Some(entry) = self.sessions.remove(session_id) else {
            return false;
        };
        if let Some(last_touch) = entry.last_touch {
            self.lru.remove(&last_touch);
        }
        for (frontier, session_frontier) in entry.frontiers {
            self.release_frontier(frontier, session_frontier);
        }
        true
    }

    fn touch_session(&mut self, session_id: &str) {
        let seq = self.next_touch;
        let Some(entry) = self.sessions.get_mut(session_id) else {
            return;
        };
        let previous = entry.last_touch.replace(seq);
        self.next_touch += 1;
        if let Some(previous) = previous {
            self.lru.remove(&previous);
        }
        self.lru.insert(seq, session_id.to_string());
    }

    // Enforce the cap after touching the newly recorded session.
    fn enforce_session_cap(&mut self) {
        while self.sessions.len() > self.max_sessions {
            let Some(victim) = self
                .lru
                .first_key_value()
                .map(|(_, session_id)| session_id.clone())
            else {
                // Avoid a hang if LRU bookkeeping is ever inconsistent.
                debug_assert!(false, "lru is empty while sessions is over capacity");
                break;
            };
            self.drop_session(&victim);
            tracing::debug!(
                session_id = %victim,
                max_sessions = self.max_sessions,
                "session prefix index evicted its least recently used session"
            );
        }
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

    fn path_from_retained_root(
        &self,
        frontier: NodeId,
        session_frontier: SessionFrontier,
    ) -> Vec<NodeId> {
        let mut path = Vec::with_capacity(session_frontier.len);
        let mut current = frontier;
        for _ in 0..session_frontier.len {
            path.push(current);
            if current == session_frontier.retained_root {
                break;
            }
            current = self.nodes[current]
                .parent
                .expect("a retained session interval must remain connected");
        }
        debug_assert_eq!(path.last(), Some(&session_frontier.retained_root));
        debug_assert_eq!(path.len(), session_frontier.len);
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

    fn session_has_reached(&self, session_id: &str, node: NodeId) -> bool {
        self.sessions.get(session_id).is_some_and(|entry| {
            entry.frontiers.iter().any(|(&frontier, session_frontier)| {
                self.session_frontier_contains(frontier, *session_frontier, node)
            })
        })
    }

    fn session_frontier_contains(
        &self,
        frontier: NodeId,
        session_frontier: SessionFrontier,
        candidate: NodeId,
    ) -> bool {
        let mut current = frontier;
        for _ in 0..session_frontier.len {
            if current == candidate {
                return true;
            }
            if current == session_frontier.retained_root {
                return false;
            }
            current = self.nodes[current]
                .parent
                .expect("a retained session interval must remain connected");
        }
        debug_assert!(false, "retained session root was not reached");
        false
    }

    fn retained_frontier(&self, frontier: NodeId) -> SessionFrontier {
        let limit = self.nodes.len();
        let mut retained_root = frontier;
        let mut len = 1usize;
        while len < self.max_session_depth {
            let Some(parent) = self.nodes[retained_root].parent else {
                break;
            };
            if len >= limit {
                debug_assert!(false, "parent cycle in session prefix index forest");
                tracing::error!("session prefix index parent walk exceeded the arena; truncating");
                break;
            }
            retained_root = parent;
            len += 1;
        }
        SessionFrontier { retained_root, len }
    }

    fn deepest_frontier(&self, session_id: &str) -> Option<NodeId> {
        self.sessions.get(session_id).and_then(|entry| {
            entry
                .frontiers
                .iter()
                .max_by_key(|(frontier, session_frontier)| {
                    (
                        session_frontier.len,
                        Reverse(self.nodes[**frontier].block_hash),
                    )
                })
                .map(|(&frontier, _)| frontier)
        })
    }

    fn advance_match_frontier(
        &mut self,
        session_id: &str,
        node: NodeId,
        continuation: Option<NodeId>,
    ) -> bool {
        let advanced = self.advance_frontier(session_id, node);
        if !advanced {
            return false;
        }

        let replaced = continuation.and_then(|continuation| {
            self.sessions
                .get_mut(session_id)
                .and_then(|entry| entry.frontiers.remove(&continuation))
                .map(|session_frontier| (continuation, session_frontier))
        });
        if let Some((frontier, session_frontier)) = replaced {
            self.release_frontier(frontier, session_frontier);
        }
        true
    }

    // Keep only the deepest frontier on each chain.
    fn advance_frontier(&mut self, session_id: &str, node: NodeId) -> bool {
        if self.session_has_reached(session_id, node) {
            // Repeated matches still refresh LRU order.
            self.touch_session(session_id);
            return false;
        }

        let subsumed: Vec<(NodeId, SessionFrontier)> = self
            .sessions
            .get(session_id)
            .map(|entry| {
                entry
                    .frontiers
                    .iter()
                    .filter(|(frontier, _)| self.is_ancestor_or_self(**frontier, node))
                    .map(|(&frontier, &session_frontier)| (frontier, session_frontier))
                    .collect()
            })
            .unwrap_or_default();

        let session_frontier = self.retained_frontier(node);
        self.acquire_frontier(node, session_frontier);

        let entry = self.sessions.entry(session_id.to_string()).or_default();
        for &(frontier, _) in &subsumed {
            entry.frontiers.remove(&frontier);
        }
        entry.frontiers.insert(node, session_frontier);

        for (frontier, session_frontier) in subsumed {
            self.release_frontier(frontier, session_frontier);
        }

        self.touch_session(session_id);
        self.enforce_session_cap();
        true
    }

    fn refresh_session_cutoffs(&mut self, session_id: &str) {
        let changes: Vec<(NodeId, SessionFrontier, SessionFrontier)> = self
            .sessions
            .get(session_id)
            .map(|entry| {
                entry
                    .frontiers
                    .iter()
                    .filter_map(|(&frontier, &old)| {
                        let new = self.retained_frontier(frontier);
                        (new != old).then_some((frontier, old, new))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if changes.is_empty() {
            return;
        }

        for &(frontier, _, new) in &changes {
            self.acquire_frontier(frontier, new);
        }
        let entry = self
            .sessions
            .get_mut(session_id)
            .expect("session cutoffs came from an existing session");
        for &(frontier, _, new) in &changes {
            entry.frontiers.insert(frontier, new);
        }
        for (frontier, old, _) in changes {
            self.release_frontier(frontier, old);
        }
    }

    fn acquire_frontier(&mut self, frontier: NodeId, session_frontier: SessionFrontier) {
        self.nodes[frontier].frontier_refs = self.nodes[frontier]
            .frontier_refs
            .checked_add(1)
            .expect("a frontier session count must not overflow");

        let mut child = frontier;
        for _ in 1..session_frontier.len {
            let parent = self.nodes[child]
                .parent
                .expect("a retained session interval must remain connected");
            self.nodes[child].parent_edge_refs = self.nodes[child]
                .parent_edge_refs
                .checked_add(1)
                .expect("a retained parent edge count must not overflow");
            child = parent;
        }
        debug_assert_eq!(child, session_frontier.retained_root);
    }

    fn release_frontier(&mut self, frontier: NodeId, session_frontier: SessionFrontier) {
        let mut child = frontier;
        let mut detached_edges = Vec::new();
        for _ in 1..session_frontier.len {
            let parent = self.nodes[child]
                .parent
                .expect("a retained session interval must remain connected");
            let edge_refs = &mut self.nodes[child].parent_edge_refs;
            *edge_refs = edge_refs
                .checked_sub(1)
                .expect("a released parent edge must have a retained interval");
            if *edge_refs == 0 {
                detached_edges.push(child);
            }
            child = parent;
        }
        debug_assert_eq!(child, session_frontier.retained_root);

        self.nodes[frontier].frontier_refs = self.nodes[frontier]
            .frontier_refs
            .checked_sub(1)
            .expect("a released frontier must have a session reference");

        let mut reclaim = Vec::with_capacity(detached_edges.len() + 1);
        reclaim.push(frontier);
        for child in detached_edges {
            if let Some(parent) = self.detach_parent_edge(child) {
                reclaim.push(parent);
            }
        }
        for node in reclaim {
            self.reclaim_unreferenced(Some(node));
        }
    }

    fn detach_unretained_parent_edge(&mut self, child: NodeId) {
        if !self.nodes.contains_key(child) || self.nodes[child].parent_edge_refs > 0 {
            return;
        }
        if let Some(parent) = self.detach_parent_edge(child) {
            self.reclaim_unreferenced(Some(parent));
        }
    }

    fn detach_parent_edge(&mut self, child: NodeId) -> Option<NodeId> {
        debug_assert_eq!(self.nodes[child].parent_edge_refs, 0);
        let parent = self.nodes[child].parent.take()?;
        self.nodes[parent].child_count = self.nodes[parent]
            .child_count
            .checked_sub(1)
            .expect("a detached parent edge must have a child reference");
        Some(parent)
    }

    fn reclaim_unreferenced(&mut self, mut current: Option<NodeId>) {
        while let Some(node) = current {
            if !self.nodes.contains_key(node) {
                break;
            }
            let entry = self.nodes[node];
            if entry.frontier_refs > 0 || entry.child_count > 0 || entry.parent_edge_refs > 0 {
                break;
            }
            self.nodes.remove(node);
            self.hash_to_node.remove(&entry.block_hash);
            if let Some(parent) = entry.parent {
                self.nodes[parent].child_count = self.nodes[parent]
                    .child_count
                    .checked_sub(1)
                    .expect("a reclaimed parent edge must have a child reference");
            }
            current = entry.parent;
        }
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

    fn lineage_of(
        indexer: &SessionPrefixIndexer,
        session: &str,
    ) -> Vec<Vec<ExternalSequenceBlockHash>> {
        let mut lineages = indexer
            .get_session_block_lineage(session, None)
            .expect("lineage query without an anchor cannot fail");
        lineages.sort();
        lineages
    }

    #[test]
    fn match_creates_lineage_and_repeat_is_idempotent() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        assert!(
            indexer.update_session_from_match("s1", chain[0]).unwrap(),
            "first match must advance the frontier"
        );
        assert_eq!(indexer.node_count(), 1);
        assert_eq!(lineage_of(&indexer, "s1"), vec![vec![chain[0]]]);

        assert!(
            !indexer.update_session_from_match("s1", chain[0]).unwrap(),
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
    fn sequential_matches_extend_one_lineage() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        for &block_hash in &chain {
            assert!(indexer.update_session_from_match("s1", block_hash).unwrap());
        }

        assert_eq!(indexer.get_session_frontiers("s1").len(), 1);
        assert_eq!(lineage_of(&indexer, "s1"), vec![chain]);
    }

    #[test]
    fn route_matches_replace_one_of_multiple_frontiers() {
        let trunk = hashes(vec![1]);
        let shallow = hashes(vec![2]);
        let deep = hashes(vec![3, 4]);
        let unrelated = hashes(vec![5, 6]);
        let next = hashes(vec![7]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", Some(trunk[0]), &shallow)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", Some(trunk[0]), &deep)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("other", None, &unrelated)
            .unwrap();

        indexer
            .update_session_from_match("s1", unrelated[1])
            .unwrap();
        indexer.update_session_from_match("s1", next[0]).unwrap();

        assert_eq!(indexer.get_session_frontiers("s1").len(), 2);
        assert_eq!(
            lineage_of(&indexer, "s1"),
            vec![
                vec![trunk[0], shallow[0]],
                vec![unrelated[0], unrelated[1], next[0]],
            ]
        );
    }

    #[test]
    fn deeper_match_replaces_the_shallower_frontier() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", None, &chain)
            .unwrap();
        assert_eq!(indexer.get_session_frontiers("s1").len(), 1);

        assert!(
            !indexer.update_session_from_match("s1", chain[1]).unwrap(),
            "a match above the current frontier must not move it"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![chain.clone()]);

        assert!(indexer.update_session_from_match("s2", chain[1]).unwrap());
        assert_eq!(lineage_of(&indexer, "s2"), vec![chain[..2].to_vec()]);
    }

    #[test]
    fn stored_blocks_reuse_shared_prefix_nodes() {
        let trunk = hashes(vec![1, 2]);
        let branch = hashes(vec![3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", Some(trunk[1]), &branch)
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

        indexer.update_session_from_match("s1", chain[1]).unwrap();
        let child = indexer.get_node_from_hash(chain[1]).unwrap();
        assert_eq!(indexer.get_node(child).unwrap().parent(), None);

        indexer
            .update_session_from_stored_blocks("s1", Some(chain[0]), &chain[1..])
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
            .update_session_from_stored_blocks("s1", None, &chain[..2])
            .unwrap();

        let err = indexer
            .update_session_from_stored_blocks("s1", Some(chain[2]), &chain[1..2])
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
            .update_session_from_stored_blocks("s1", None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", Some(trunk[0]), &left)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", Some(trunk[0]), &right)
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
            .update_session_from_stored_blocks("s1", None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", Some(trunk[1]), &left)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s1", None, &right)
            .unwrap();

        let anchored = indexer
            .get_session_block_lineage("s1", Some(trunk[1]))
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
        indexer.update_session_from_match("s1", chain[0]).unwrap();

        assert_eq!(
            indexer.get_session_block_lineage("s1", Some(missing[0])),
            Err(SessionPrefixIndexError::UnknownAnchor(missing[0]))
        );
        assert_eq!(
            indexer.get_session_block_lineage("unrouted", None),
            Ok(Vec::new()),
            "a session with no routed requests is empty, not an error"
        );
    }

    #[test]
    fn eviction_shaped_replay_does_not_lose_lineage() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", None, &chain)
            .unwrap();
        let before = lineage_of(&indexer, "s1");

        indexer.update_session_from_match("s1", chain[0]).unwrap();

        assert_eq!(
            lineage_of(&indexer, "s1"),
            before,
            "re-matching a shallow block must not truncate the recorded lineage"
        );
        assert_eq!(indexer.node_count(), 2);
    }

    #[test]
    fn removing_a_session_frees_only_its_exclusive_tail() {
        let trunk = hashes(vec![1, 2]);
        let tail = hashes(vec![3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", None, &trunk)
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", Some(trunk[1]), &tail)
            .unwrap();
        assert_eq!(indexer.node_count(), 3);

        assert!(indexer.remove_session("s2"));
        assert_eq!(
            indexer.node_count(),
            2,
            "only the tail s2 held exclusively is reclaimed"
        );
        assert_eq!(indexer.get_node_from_hash(tail[0]), None);
        assert_eq!(
            lineage_of(&indexer, "s1"),
            vec![trunk.clone()],
            "the surviving session keeps the shared trunk"
        );

        assert!(indexer.remove_session("s1"));
        assert_eq!(
            indexer.node_count(),
            0,
            "the last session out releases every slot"
        );
        assert_eq!(indexer.session_count(), 0);
        assert_eq!(indexer.get_node_from_hash(trunk[0]), None);

        assert!(
            !indexer.remove_session("s1"),
            "removing an unknown session reports that nothing was removed"
        );
    }

    #[test]
    fn stale_node_handles_do_not_alias_after_reuse() {
        let first = hashes(vec![1]);
        let second = hashes(vec![2]);
        let indexer = SessionPrefixIndexer::new();

        indexer.update_session_from_match("s1", first[0]).unwrap();
        let stale = indexer.get_node_from_hash(first[0]).unwrap();
        indexer.remove_session("s1");

        indexer.update_session_from_match("s2", second[0]).unwrap();
        let fresh = indexer.get_node_from_hash(second[0]).unwrap();

        assert_ne!(stale, fresh, "the generational key must not be reissued");
        assert!(
            indexer.get_node(stale).is_none(),
            "a handle to a removed node must not resolve to its replacement"
        );
    }

    #[test]
    fn passing_the_session_cap_evicts_the_least_recently_used_session() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::with_max_sessions(2);
        assert_eq!(indexer.max_sessions(), 2);

        indexer.update_session_from_match("s1", chain[0]).unwrap();
        indexer.update_session_from_match("s2", chain[1]).unwrap();
        indexer.update_session_from_match("s1", chain[1]).unwrap();

        indexer.update_session_from_match("s3", chain[2]).unwrap();

        assert!(
            lineage_of(&indexer, "s2").is_empty(),
            "the least recently touched session is the one evicted"
        );
        assert!(
            !lineage_of(&indexer, "s1").is_empty(),
            "a recently touched session survives the eviction"
        );
        assert!(
            !lineage_of(&indexer, "s3").is_empty(),
            "the session that triggered the eviction is retained"
        );
    }

    #[test]
    fn eviction_releases_the_evicted_session_arena_nodes() {
        let chain = hashes(vec![1, 2]);
        let indexer = SessionPrefixIndexer::with_max_sessions(1);

        indexer.update_session_from_match("s1", chain[0]).unwrap();
        let evicted = indexer.get_node_from_hash(chain[0]).unwrap();

        indexer.update_session_from_match("s2", chain[1]).unwrap();

        assert!(
            indexer.get_node(evicted).is_none(),
            "eviction must reclaim the arena exactly as remove_session does"
        );
        assert!(
            indexer.get_node_from_hash(chain[0]).is_none(),
            "the evicted session's hash index entry must go with its node"
        );
    }

    #[test]
    fn a_zero_session_cap_is_clamped_to_one_tracked_session() {
        let chain = hashes(vec![1]);
        let indexer = SessionPrefixIndexer::with_max_sessions(0);

        assert_eq!(
            indexer.max_sessions(),
            1,
            "a cap of zero would track nothing"
        );
        indexer.update_session_from_match("s1", chain[0]).unwrap();
        assert!(
            !lineage_of(&indexer, "s1").is_empty(),
            "the sole tracked session must survive its own insertion"
        );
    }

    #[test]
    fn lineage_depth_limit_reclaims_detached_ancestors() {
        let chain = hashes(vec![1, 2, 3, 4, 5]);
        let indexer = SessionPrefixIndexer::with_limits(DEFAULT_MAX_SESSIONS, 3);

        indexer
            .update_session_from_stored_blocks("s1", None, &chain)
            .unwrap();

        assert_eq!(lineage_of(&indexer, "s1"), vec![chain[2..].to_vec()]);
        assert_eq!(indexer.node_count(), 3);
        for &block_hash in &chain[..2] {
            assert_eq!(indexer.get_node_from_hash(block_hash), None);
        }
        for &block_hash in &chain[2..] {
            assert!(indexer.get_node_from_hash(block_hash).is_some());
        }
    }

    #[test]
    fn one_session_cutoff_preserves_another_sessions_shared_lineage() {
        let chain = hashes(vec![1, 2, 3, 4]);
        let indexer = SessionPrefixIndexer::with_limits(DEFAULT_MAX_SESSIONS, 3);

        indexer
            .update_session_from_stored_blocks("s1", None, &chain[..3])
            .unwrap();
        indexer
            .update_session_from_stored_blocks("s2", None, &chain[..3])
            .unwrap();
        indexer.update_session_from_match("s1", chain[3]).unwrap();

        assert_eq!(lineage_of(&indexer, "s1"), vec![chain[1..].to_vec()]);
        assert_eq!(lineage_of(&indexer, "s2"), vec![chain[..3].to_vec()]);
        assert!(
            indexer
                .get_session_block_lineage("s1", Some(chain[0]))
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            indexer
                .get_session_block_lineage("s1", Some(chain[1]))
                .unwrap(),
            vec![chain[1..].to_vec()]
        );

        assert!(indexer.remove_session("s2"));
        assert_eq!(indexer.get_node_from_hash(chain[0]), None);
        assert_eq!(lineage_of(&indexer, "s1"), vec![chain[1..].to_vec()]);

        let reverse = SessionPrefixIndexer::with_limits(DEFAULT_MAX_SESSIONS, 3);
        reverse
            .update_session_from_stored_blocks("s1", None, &chain[..3])
            .unwrap();
        reverse
            .update_session_from_stored_blocks("s2", None, &chain[..3])
            .unwrap();
        reverse.update_session_from_match("s1", chain[3]).unwrap();

        assert!(reverse.remove_session("s1"));
        assert_eq!(lineage_of(&reverse, "s2"), vec![chain[..3].to_vec()]);
        assert_eq!(reverse.get_node_from_hash(chain[3]), None);
        assert!(reverse.remove_session("s2"));
        assert_eq!(reverse.node_count(), 0);
    }

    #[test]
    fn depth_one_and_sibling_branches_release_exact_intervals() {
        let chain = hashes(vec![1, 2, 3, 4]);
        let depth_one = SessionPrefixIndexer::with_limits(DEFAULT_MAX_SESSIONS, 1);
        depth_one
            .update_session_from_stored_blocks("s1", None, &chain[..3])
            .unwrap();
        assert_eq!(lineage_of(&depth_one, "s1"), vec![vec![chain[2]]]);
        assert_eq!(depth_one.node_count(), 1);

        let branches = SessionPrefixIndexer::with_limits(DEFAULT_MAX_SESSIONS, 3);
        branches
            .update_session_from_stored_blocks("s1", None, &chain[..3])
            .unwrap();
        branches
            .update_session_from_stored_blocks("s1", Some(chain[1]), &chain[3..])
            .unwrap();
        assert_eq!(
            lineage_of(&branches, "s1"),
            vec![chain[..3].to_vec(), vec![chain[0], chain[1], chain[3]]]
        );
        assert!(branches.remove_session("s1"));
        assert_eq!(branches.node_count(), 0);
    }

    #[test]
    fn a_chain_that_would_close_a_cycle_is_rejected() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", None, &chain)
            .unwrap();

        let err = indexer
            .update_session_from_stored_blocks("s1", Some(chain[2]), &chain[..1])
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
            .update_session_from_stored_blocks("s1", None, &repeating)
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
