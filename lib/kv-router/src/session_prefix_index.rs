// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session-aware logical prefix index.
//!
//! The physical indexers (`indexer::*`) answer "which worker holds this block
//! right now". They are driven by engine cache events, so a block that the
//! engine evicts disappears from them immediately. That is correct for routing
//! but it destroys the only record of what a *session* has previously produced.
//!
//! This module keeps a separate, logical view: a forest of
//! [`ExternalSequenceBlockHash`] nodes plus, per session, the set of nodes that
//! are currently the deepest known point of that session's block chains. Nodes
//! are never dropped because the engine evicted or cleared the underlying
//! blocks; they are dropped only when the owning session is cleaned up through
//! [`SessionPrefixIndexer::remove_session`]. Eviction changes where a block
//! lives, not whether the session ever produced it.
//!
//! Structure follows the enhancement proposal:
//!
//! - `nodes`: a [`SlotMap`] arena of [`LogicalNode`], so a [`NodeId`] handed out
//!   to a caller cannot silently alias a different node after a removal.
//! - `hash_to_node`: reverse index from block hash to arena slot. Numeric key,
//!   so [`FxHashMap`] per the crate guidance.
//! - `sessions`: session id to frontier node set. The key is caller-supplied
//!   text, so this uses [`std::collections::HashMap`] rather than `FxHashMap`,
//!   again per the crate guidance.
//!
//! [`ExternalSequenceBlockHash`] is treated as opaque: this module compares and
//! hashes it and never interprets, derives, or recomputes its value.
//!
//! Deviation from the proposal's literal signatures: the session parameters are
//! taken as `&str` rather than an owned `SessionId`. The route-time caller only
//! holds a borrow, and an owned parameter would force a `String` allocation on
//! every routed request even when the session is already known. The owned
//! [`SessionId`] alias is still what the map stores, and a clone happens only on
//! first insert.

use std::collections::HashMap;

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};

use crate::protocols::ExternalSequenceBlockHash;

/// Logical session identity as owned by this index.
pub type SessionId = String;

new_key_type! {
    /// Generational handle to a [`LogicalNode`] in the arena.
    pub struct NodeId;
}

/// One logical block in the session forest.
///
/// Holds its own hash, its parent link, and the liveness counters that decide
/// when the node can be reclaimed: how many session frontier sets point at it,
/// and how many children it still has.
#[derive(Clone, Copy, Debug)]
pub struct LogicalNode {
    block_hash: ExternalSequenceBlockHash,
    parent: Option<NodeId>,
    frontier_refs: u32,
    child_count: u32,
}

impl LogicalNode {
    /// The block hash this node stands for.
    pub fn block_hash(&self) -> ExternalSequenceBlockHash {
        self.block_hash
    }

    /// The parent node, or `None` when this node is a root of the forest.
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// Number of sessions whose frontier set currently contains this node.
    pub fn frontier_refs(&self) -> u32 {
        self.frontier_refs
    }

    /// Number of direct children currently attached to this node.
    pub fn child_count(&self) -> u32 {
        self.child_count
    }
}

/// Errors surfaced by [`SessionPrefixIndexer`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SessionPrefixIndexError {
    /// A lineage query named an anchor hash the index has never seen. Returning
    /// an empty lineage here would be indistinguishable from "this session has
    /// no blocks past the anchor", so it is reported as an error instead.
    #[error("unknown anchor block hash {0:?}")]
    UnknownAnchor(ExternalSequenceBlockHash),

    /// A stored-blocks update tried to attach a block under a parent other than
    /// the one already recorded. Within one hash domain a unique block chain
    /// maps to a unique external sequence-hash chain, so this is a producer,
    /// configuration, or protocol error rather than a routing condition.
    #[error("block {block:?} is already parented elsewhere")]
    ConflictingParent {
        /// The block whose recorded parent disagrees with the update.
        block: ExternalSequenceBlockHash,
    },
}

/// Session-aware logical prefix index.
///
/// Cheap to share: all methods take `&self` and lock internally, so the router
/// can hold this behind an `Arc` next to its physical indexer.
#[derive(Debug, Default)]
pub struct SessionPrefixIndexer {
    state: RwLock<IndexState>,
}

#[derive(Debug, Default)]
struct IndexState {
    nodes: SlotMap<NodeId, LogicalNode>,
    hash_to_node: FxHashMap<ExternalSequenceBlockHash, NodeId>,
    sessions: HashMap<SessionId, FxHashSet<NodeId>>,
}

impl SessionPrefixIndexer {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resolve a block hash to its arena slot, if this index knows the block.
    pub fn get_node_from_hash(&self, block_hash: ExternalSequenceBlockHash) -> Option<NodeId> {
        self.state.read().hash_to_node.get(&block_hash).copied()
    }

    /// Read a node by handle. Returns `None` for a stale handle.
    pub fn get_node(&self, node_id: NodeId) -> Option<LogicalNode> {
        self.state.read().nodes.get(node_id).copied()
    }

    /// The deepest known nodes of every chain this session has touched.
    ///
    /// Empty for an unknown session. Order is unspecified.
    pub fn get_session_frontiers(&self, session_id: &str) -> Vec<NodeId> {
        self.state
            .read()
            .sessions
            .get(session_id)
            .map(|frontiers| frontiers.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Every block chain this session has touched, root first.
    ///
    /// One inner vector per frontier. With `anchor_hash` set, each chain is
    /// truncated to start at the anchor and chains that do not pass through it
    /// are omitted; an anchor the index has never seen is an error. An unknown
    /// session yields an empty result, which is not an error: a session that has
    /// not been routed yet legitimately has no lineage.
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

        let Some(frontiers) = state.sessions.get(session_id) else {
            return Ok(Vec::new());
        };

        let mut lineages = Vec::with_capacity(frontiers.len());
        for &frontier in frontiers {
            let path = state.path_to_root(frontier);
            let start = match anchor_node {
                Some(anchor) => match path.iter().position(|&node| node == anchor) {
                    Some(position) => position,
                    // This frontier's chain does not pass through the anchor.
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

    /// Record a route-time cache hit for `session_id` at `matched_hash`.
    ///
    /// Returns whether the session's frontier set moved. A hit on a block at or
    /// above a frontier the session already holds is not an advance, so a
    /// re-route of the same prefix leaves the index untouched. An unknown hash
    /// is admitted as a new root: the router matched the block, so it exists,
    /// even though a match alone carries no parent link. A later
    /// [`Self::update_session_from_stored_blocks`] carrying that parent grafts
    /// the chain onto this same node rather than creating a second root.
    pub fn update_session_from_match(
        &self,
        session_id: &str,
        matched_hash: ExternalSequenceBlockHash,
    ) -> Result<bool, SessionPrefixIndexError> {
        let mut state = self.state.write();
        let node = state.resolve_or_insert_root(matched_hash);
        Ok(state.advance_frontier(session_id, node))
    }

    /// Record a chain of blocks stored under `parent_hash` for `session_id`.
    ///
    /// Blocks are applied in order and existing nodes are reused, so replaying
    /// the same chain does not grow the arena. Returns whether the session's
    /// frontier set moved.
    ///
    /// Not currently driven by the KV event stream: the cache event schema
    /// carries no session identity, so there is no honest way to attribute a
    /// stored-block event to a session yet. It is the ingest half of the
    /// proposal's API and is exercised by this module's tests.
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
        // Validate the whole chain before touching the arena, so a rejected
        // update leaves no half-applied nodes behind.
        state.validate_chain(parent_hash, block_hashes)?;

        let mut parent = parent_hash.map(|hash| state.resolve_or_insert_root(hash));
        for &block_hash in block_hashes {
            let node = match state.hash_to_node.get(&block_hash).copied() {
                Some(existing) => {
                    // Known only as a root so far; graft it under the parent
                    // this event supplies instead of leaving it detached.
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
        Ok(state.advance_frontier(session_id, leaf))
    }

    /// Drop a session and reclaim the nodes no other session still needs.
    ///
    /// Returns whether the session was known. This is the only path that
    /// removes logical nodes; block eviction never does.
    pub fn remove_session(&self, session_id: &str) -> bool {
        let mut state = self.state.write();
        let Some(frontiers) = state.sessions.remove(session_id) else {
            return false;
        };
        for frontier in frontiers {
            state.release_frontier(frontier);
        }
        true
    }

    /// Number of logical nodes currently retained.
    pub fn node_count(&self) -> usize {
        self.state.read().nodes.len()
    }

    /// Number of sessions currently tracked.
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
            frontier_refs: 0,
            child_count: 0,
        });
        self.hash_to_node.insert(block_hash, node);
        if let Some(parent) = parent {
            self.nodes[parent].child_count += 1;
        }
        node
    }

    /// Read-only check that no block in the chain is already parented somewhere
    /// other than where this chain would put it.
    fn validate_chain(
        &self,
        parent_hash: Option<ExternalSequenceBlockHash>,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> Result<(), SessionPrefixIndexError> {
        let mut expected_parent = parent_hash;
        for &block_hash in block_hashes {
            if let Some(&existing) = self.hash_to_node.get(&block_hash)
                && let Some(recorded) = self.nodes[existing].parent
                && Some(self.nodes[recorded].block_hash) != expected_parent
            {
                return Err(SessionPrefixIndexError::ConflictingParent { block: block_hash });
            }
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

    fn path_to_root(&self, tail: NodeId) -> Vec<NodeId> {
        let mut path = Vec::new();
        let mut current = Some(tail);
        while let Some(node) = current {
            path.push(node);
            current = self.nodes[node].parent;
        }
        path.reverse();
        path
    }

    /// Is `candidate` at or above `node` in the forest?
    fn is_ancestor_or_self(&self, candidate: NodeId, node: NodeId) -> bool {
        let mut current = Some(node);
        while let Some(walk) = current {
            if walk == candidate {
                return true;
            }
            current = self.nodes[walk].parent;
        }
        false
    }

    /// Move `session_id`'s frontier set to include `node`, returning whether
    /// anything changed. Frontiers that `node` now subsumes are dropped so a
    /// session holds only the deepest point of each chain it has touched.
    fn advance_frontier(&mut self, session_id: &str, node: NodeId) -> bool {
        if let Some(frontiers) = self.sessions.get(session_id)
            && frontiers
                .iter()
                .any(|&frontier| self.is_ancestor_or_self(node, frontier))
        {
            return false;
        }

        let subsumed: Vec<NodeId> = self
            .sessions
            .get(session_id)
            .map(|frontiers| {
                frontiers
                    .iter()
                    .copied()
                    .filter(|&frontier| self.is_ancestor_or_self(frontier, node))
                    .collect()
            })
            .unwrap_or_default();

        for frontier in &subsumed {
            self.nodes[*frontier].frontier_refs -= 1;
        }
        self.nodes[node].frontier_refs += 1;

        let frontiers = self.sessions.entry(session_id.to_string()).or_default();
        for frontier in subsumed {
            frontiers.remove(&frontier);
        }
        frontiers.insert(node);
        true
    }

    /// Drop one frontier reference and reclaim the now-unreachable tail above
    /// it. A node survives while any session still points at it or any child
    /// still hangs off it, so shared prefixes outlive the first session to go.
    fn release_frontier(&mut self, frontier: NodeId) {
        self.nodes[frontier].frontier_refs -= 1;

        let mut current = Some(frontier);
        while let Some(node) = current {
            let entry = self.nodes[node];
            if entry.frontier_refs > 0 || entry.child_count > 0 {
                break;
            }
            self.nodes.remove(node);
            self.hash_to_node.remove(&entry.block_hash);
            if let Some(parent) = entry.parent {
                self.nodes[parent].child_count -= 1;
            }
            current = entry.parent;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_blocks;

    /// Block hashes as the engine publishes them, via the shared event builder
    /// so these fixtures cannot drift from the real event shape.
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
    fn deeper_match_replaces_the_shallower_frontier() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        indexer
            .update_session_from_stored_blocks("s1", None, &chain)
            .unwrap();
        assert_eq!(indexer.get_session_frontiers("s1").len(), 1);

        // Re-route lands on the middle of the chain the session already owns.
        assert!(
            !indexer.update_session_from_match("s1", chain[1]).unwrap(),
            "a match above the current frontier must not move it"
        );
        assert_eq!(lineage_of(&indexer, "s1"), vec![chain.clone()]);

        // A brand new session hitting the middle stops there.
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

        // Route-time hit on the child arrives first, with no parent context.
        indexer.update_session_from_match("s1", chain[1]).unwrap();
        let child = indexer.get_node_from_hash(chain[1]).unwrap();
        assert_eq!(indexer.get_node(child).unwrap().parent(), None);

        // The stored-block chain then supplies the missing parent link.
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
        // The physical indexers would have dropped these blocks on a Removed or
        // Cleared event. This index has no such path: only remove_session drops
        // nodes, so a session re-routed after eviction still reads back its
        // full chain.
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
}
