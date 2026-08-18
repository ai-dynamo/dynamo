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
//! blocks; they are dropped only when the owning session goes, either through
//! [`SessionPrefixIndexer::remove_session`] or through the capacity bound
//! below. Eviction changes where a block lives, not whether the session ever
//! produced it.
//!
//! Retention is bounded. The router calls [`SessionPrefixIndexer::remove_session`]
//! when a request marks its session final, but that signal is optional and many
//! clients never send it, so the index also caps how many sessions it tracks
//! ([`DEFAULT_MAX_SESSIONS`], overridable via
//! [`SessionPrefixIndexer::with_max_sessions`]). Passing the cap evicts the
//! least recently touched session by exactly the path `remove_session` takes.
//! Both halves are needed: the lifecycle signal reclaims promptly when it
//! arrives, and the cap is what makes growth bounded when it never does.
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

use std::collections::{BTreeMap, HashMap};

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};
use slotmap::{SlotMap, new_key_type};

use crate::protocols::ExternalSequenceBlockHash;

/// Logical session identity as owned by this index.
pub type SessionId = String;

/// Default ceiling on tracked sessions, past which the least recently touched
/// session is evicted.
///
/// Sized so the common case never reaches it — a router serving far fewer
/// concurrent sessions than this behaves exactly as if the index were
/// unbounded — while a router that never receives an end-of-session signal
/// still settles at a fixed footprint instead of growing for the life of the
/// process.
pub const DEFAULT_MAX_SESSIONS: usize = 16_384;

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

    /// A stored-blocks update tried to attach a block underneath a block that
    /// the first one already sits at or above. Applying it would close a parent
    /// cycle, and every walk of the forest follows parent links, so the cycle
    /// would turn later reads into non-terminating loops holding the write
    /// lock. Rejected before the arena is touched.
    #[error("block {block:?} would become its own ancestor")]
    CyclicParent {
        /// The block whose graft would close the cycle.
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

/// One session's retained state: the deepest nodes it has reached, and when it
/// was last touched, which is what the capacity bound evicts on.
///
/// `last_touch` is `None` only between the entry's creation and the
/// [`IndexState::touch_session`] call that immediately follows it. Making that
/// gap explicit matters: a numeric sentinel would alias whichever session
/// legitimately holds that sequence number and evict it instead.
#[derive(Debug, Default)]
struct SessionEntry {
    frontiers: FxHashSet<NodeId>,
    last_touch: Option<u64>,
}

#[derive(Debug)]
struct IndexState {
    nodes: SlotMap<NodeId, LogicalNode>,
    hash_to_node: FxHashMap<ExternalSequenceBlockHash, NodeId>,
    sessions: HashMap<SessionId, SessionEntry>,
    /// Touch sequence to session id, so the least recently touched session is
    /// the first entry. Kept in lockstep with [`SessionEntry::last_touch`]:
    /// every session in `sessions` has exactly one entry here.
    lru: BTreeMap<u64, SessionId>,
    /// Monotonic touch counter. `u64` at one touch per routed request does not
    /// wrap in any realistic process lifetime.
    next_touch: u64,
    max_sessions: usize,
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
        }
    }
}

impl SessionPrefixIndexer {
    /// Create an empty index holding at most [`DEFAULT_MAX_SESSIONS`] sessions.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty index holding at most `max_sessions` sessions.
    ///
    /// A cap of zero is raised to one: an index that could retain nothing would
    /// discard each session as it was recorded, which is not a useful state to
    /// let a caller configure by accident.
    pub fn with_max_sessions(max_sessions: usize) -> Self {
        Self {
            state: RwLock::new(IndexState {
                max_sessions: max_sessions.max(1),
                ..IndexState::default()
            }),
        }
    }

    /// The ceiling on tracked sessions.
    pub fn max_sessions(&self) -> usize {
        self.state.read().max_sessions
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
            .map(|entry| entry.frontiers.iter().copied().collect())
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

        let Some(entry) = state.sessions.get(session_id) else {
            return Ok(Vec::new());
        };

        let mut lineages = Vec::with_capacity(entry.frontiers.len());
        for &frontier in &entry.frontiers {
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
        state.drop_session(session_id)
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
    /// other than where this chain would put it, and that applying the chain
    /// cannot close a parent cycle.
    ///
    /// The cycle half matters because a block already known only as a root has
    /// `parent == None` and so passes the conflict check, after which the apply
    /// loop would graft it under whatever parent this call supplies — including
    /// one of its own descendants. `dominators` is every hash that will sit at
    /// or above the chain once it is applied: the supplied parent, everything
    /// already above that parent, and the chain's own earlier blocks. A block
    /// that appears in that set is being asked to become its own ancestor.
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

    /// Remove a session and reclaim what no other session needs, returning
    /// whether the session was known. Shared by the public
    /// [`SessionPrefixIndexer::remove_session`] and by capacity eviction, so
    /// both reclaim by exactly the same path.
    fn drop_session(&mut self, session_id: &str) -> bool {
        let Some(entry) = self.sessions.remove(session_id) else {
            return false;
        };
        if let Some(last_touch) = entry.last_touch {
            self.lru.remove(&last_touch);
        }
        for frontier in entry.frontiers {
            self.release_frontier(frontier);
        }
        true
    }

    /// Mark `session_id` as the most recently used session.
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

    /// Evict least recently touched sessions until the cap is respected.
    ///
    /// Called after a session is recorded rather than before, so the session
    /// that just arrived is the most recently touched one and is never the
    /// victim of its own insertion.
    fn enforce_session_cap(&mut self) {
        while self.sessions.len() > self.max_sessions {
            let Some((_, victim)) = self.lru.pop_first() else {
                // `lru` and `sessions` are maintained together, so this is
                // unreachable; breaking rather than looping keeps a bookkeeping
                // bug from becoming a hang.
                debug_assert!(false, "lru is empty while sessions is over capacity");
                break;
            };
            if let Some(entry) = self.sessions.remove(&victim) {
                for frontier in entry.frontiers {
                    self.release_frontier(frontier);
                }
            }
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

    /// Walk parent links from `tail`, root first.
    ///
    /// Bounded by the arena size. An acyclic forest cannot yield a longer path,
    /// so the bound never truncates a well-formed walk; it is a backstop that
    /// makes a corrupted arena degrade into a short answer rather than spin
    /// forever while holding the index lock. `validate_chain` is what actually
    /// keeps the forest acyclic.
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

    /// Is `candidate` at or above `node` in the forest?
    ///
    /// Bounded on the same reasoning as [`Self::path_to_root`].
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

    /// Move `session_id`'s frontier set to include `node`, returning whether
    /// anything changed. Frontiers that `node` now subsumes are dropped so a
    /// session holds only the deepest point of each chain it has touched.
    fn advance_frontier(&mut self, session_id: &str, node: NodeId) -> bool {
        let already_reached = self.sessions.get(session_id).is_some_and(|entry| {
            entry
                .frontiers
                .iter()
                .any(|&frontier| self.is_ancestor_or_self(node, frontier))
        });
        if already_reached {
            // The frontier does not move, but the session is demonstrably still
            // being routed, so it must not drift towards eviction while a
            // genuinely idle session outranks it.
            self.touch_session(session_id);
            return false;
        }

        let subsumed: Vec<NodeId> = self
            .sessions
            .get(session_id)
            .map(|entry| {
                entry
                    .frontiers
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

        let entry = self.sessions.entry(session_id.to_string()).or_default();
        for frontier in subsumed {
            entry.frontiers.remove(&frontier);
        }
        entry.frontiers.insert(node);

        self.touch_session(session_id);
        self.enforce_session_cap();
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

    #[test]
    fn passing_the_session_cap_evicts_the_least_recently_used_session() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::with_max_sessions(2);
        assert_eq!(indexer.max_sessions(), 2);

        indexer.update_session_from_match("s1", chain[0]).unwrap();
        indexer.update_session_from_match("s2", chain[1]).unwrap();
        // Touch s1 so s2 becomes the least recently used of the two.
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
    fn a_chain_that_would_close_a_cycle_is_rejected() {
        let chain = hashes(vec![1, 2, 3]);
        let indexer = SessionPrefixIndexer::new();

        // Build A -> B -> C.
        indexer
            .update_session_from_stored_blocks("s1", None, &chain)
            .unwrap();

        // Now claim A is stored under C. Accepting that would make A its own
        // ancestor and leave the parent walks looping forever.
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

        // The rejected update must leave the original forest intact.
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

        // A -> B -> A within a single event: the cycle closes on a node this
        // very chain created, so the check cannot rely on existing parents.
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
