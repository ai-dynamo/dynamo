// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CRTC-node-depth sharding: routing keyed on a shadow trie that mirrors the
//! top-K levels of the underlying CRTC's compressed radix tree structure.
//!
//! ## Why node depth instead of block depth
//!
//! [`BranchShardedIndexer`] keys on `FNV(first prefix_depth blocks)`, which
//! requires knowing in advance how long any shared prefix is.  On the `.jcs`
//! workload (15-block system prompt), block depth 2 collapses to one branch key
//! because every request shares `[b0, b1]`.  Fixing it requires setting
//! `prefix_depth=16`, which requires knowing the system-prompt length.
//!
//! This indexer instead keys on the first `routing_node_depth` *CRTC nodes*
//! traversed.  Because the CRTC path-compresses shared prefixes into single
//! edges, the 15-block system prompt is **one node**.  At node depth 2, routing
//! already sees all distinct continuations without knowing the prompt length.
//!
//! ## Shadow trie
//!
//! A `ShadowNode` trie mirrors the CRTC's top-`routing_node_depth` node
//! structure.  Each edge is a `Vec<LocalBlockHash>` (variable length, exactly
//! one CRTC-node edge).  Edges split when two sequences diverge mid-edge, just
//! as the CRTC splits its own nodes.
//!
//! - **Routing (`find_matches`)**: read-lock the trie, walk `routing_node_depth`
//!   edges, return the leaf shard.  O(routing_node_depth) hash lookups.
//! - **Insertion (`apply_event`)**: write-lock the trie, walk/create/split.
//!   Only occurs for root events (or all events when
//!   `inherit_parent_shard=false`).  Continuation events fast-path via
//!   `block_to_shard` without touching the trie.
//!
//! ## Hybrid depth-aware routing
//!
//! Rather than naively re-inserting incremental blocks from the trie root
//! (which creates phantom sibling branches instead of children), this indexer
//! uses `last_block_to_path` to track the full prefix for each shallow event:
//!
//! - **Parent reached a routing leaf** (`last_block_to_path` entry is `None`):
//!   inherit the shard.  Routing leaves are never split, so this is permanently
//!   correct and requires no trie access.
//! - **Parent is shallow** (`last_block_to_path` entry is `Some(prefix)`):
//!   reconstruct the full sequence (`prefix + incremental`) and re-walk the
//!   trie, inserting the continuation as a proper child of the existing node.
//! - **Parent not tracked** (root event, OOO, or evicted): fall back to
//!   inserting the incremental blocks from root.
//!
//! Once a conversation's prefix reaches `routing_node_depth`, all subsequent
//! events inherit without touching the trie or storing further path data,
//! bounding memory overhead to the shallow "crossing" events only.

use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

use async_trait::async_trait;
use dashmap::DashMap;
use rustc_hash::{FxBuildHasher, FxHashMap};
use tokio::sync::{RwLock, oneshot};

use super::{KvIndexerInterface, KvRouterError, ShardSizeSnapshot, SyncIndexer, ThreadPoolIndexer};
use crate::protocols::*;

// ---------------------------------------------------------------------------
// Per-shard read thread pool (same pattern as branch_sharded.rs)
// ---------------------------------------------------------------------------

type ReadRequest = (Vec<LocalBlockHash>, oneshot::Sender<OverlapScores>);

struct ShardReadPool {
    sender: flume::Sender<ReadRequest>,
    _threads: Vec<std::thread::JoinHandle<()>>,
}

impl ShardReadPool {
    fn new<T: SyncIndexer>(backend: Arc<T>, num_threads: usize) -> Self {
        let (tx, rx) = flume::unbounded::<ReadRequest>();
        let mut threads = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            let backend = Arc::clone(&backend);
            let rx = rx.clone();
            threads.push(std::thread::spawn(move || {
                while let Ok((seq, resp_tx)) = rx.recv() {
                    let result = backend.find_matches(&seq, false);
                    let _ = resp_tx.send(result);
                }
            }));
        }
        Self {
            sender: tx,
            _threads: threads,
        }
    }
}

// ---------------------------------------------------------------------------
// Shadow routing trie
// ---------------------------------------------------------------------------

/// One node in the shadow routing trie.
///
/// Non-root nodes hold a compressed edge (variable-length block sequence)
/// representing one CRTC-node edge.  `node_depth` counts the number of
/// complete edges from the root to this node.
struct ShadowNode {
    /// Compressed edge: blocks consumed to reach this node from its parent.
    /// Empty for the root.
    edge: Vec<LocalBlockHash>,
    /// Number of complete CRTC-node edges from root to this node.
    /// Root = 0, first-level children = 1, etc.
    node_depth: usize,
    /// Shard for sequences that land on this node without going deeper.
    /// `None` for interior nodes (they have children and no direct assignment).
    shard: Option<usize>,
    /// Fallback shard for sequences whose next block matches no child edge.
    default_shard: usize,
    /// Children keyed by the first `LocalBlockHash` of the child's edge.
    children: FxHashMap<LocalBlockHash, ShadowNode>,
}

impl ShadowNode {
    fn new_root() -> Self {
        Self {
            edge: Vec::new(),
            node_depth: 0,
            shard: None,
            default_shard: 0,
            children: FxHashMap::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Trie algorithms (free functions to avoid borrowing `self`)
// ---------------------------------------------------------------------------

/// Length of the longest common prefix between `a` and `b`.
fn common_prefix_len(a: &[LocalBlockHash], b: &[LocalBlockHash]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Assign the shard with the fewest routing leaves.
///
/// Called only from within a `routing_trie.write()` critical section, so
/// `Relaxed` ordering is sufficient — the write lock provides sequencing.
fn least_loaded(counts: &[AtomicUsize]) -> usize {
    let (idx, _) = counts
        .iter()
        .enumerate()
        .min_by_key(|&(_, a)| a.load(Ordering::Relaxed))
        .unwrap();
    counts[idx].fetch_add(1, Ordering::Relaxed);
    idx
}

/// Walk the shadow trie and return the shard for `seq`, or `None` on miss.
///
/// Returns `None` when the sequence doesn't match any routing entry (early exit
/// — no worker has ever stored under this prefix at the required depth).
fn route_sequence(
    root: &ShadowNode,
    seq: &[LocalBlockHash],
    routing_depth: usize,
) -> Option<usize> {
    let mut node = root;
    let mut pos = 0;

    loop {
        // Reached routing depth: this node is the routing leaf.
        if node.node_depth >= routing_depth {
            return node.shard;
        }

        if pos >= seq.len() {
            // Sequence exhausted before reaching routing depth.
            // Interior nodes return None (miss); premature leaves return their shard.
            return node.shard;
        }

        let first = seq[pos];
        let Some(child) = node.children.get(&first) else {
            // No child for this block.
            // Premature leaves have shard=Some, interior nodes have shard=None.
            return node.shard;
        };

        let common = common_prefix_len(&child.edge, &seq[pos..]);
        if common < child.edge.len() {
            // Diverges mid-edge: falls under this child's routing group.
            return child.shard;
        }
        // Full edge match — descend.
        pos += child.edge.len();
        node = child;
    }
}

/// Insert `seq` (starting at `seq_pos`) into the shadow trie rooted at `node`
/// and return the assigned shard plus whether the insertion reached a routing
/// leaf (`node_depth >= routing_depth`).
///
/// Creates leaves as needed and splits edges on divergence, up to
/// `routing_depth`.  Never splits nodes already at routing_depth (they are
/// permanent routing leaves).
///
/// The second return value (`reached_routing_depth`) is `true` when the
/// function terminated at a node whose `node_depth >= routing_depth`.
/// Because routing leaves are never split, a `true` result is a permanent
/// guarantee: future continuations from those blocks may safely inherit the
/// shard without consulting the trie.
fn insert_and_get_shard(
    node: &mut ShadowNode,
    seq: &[LocalBlockHash],
    seq_pos: usize,
    routing_depth: usize,
    counts: &[AtomicUsize],
    node_count: &AtomicUsize,
) -> (usize, bool) {
    // At routing depth — permanent routing leaf.
    if node.node_depth >= routing_depth {
        return (node.shard.unwrap_or(node.default_shard), true);
    }
    // Sequence exhausted before reaching routing depth.
    if seq_pos >= seq.len() {
        return (node.shard.unwrap_or(node.default_shard), false);
    }

    let first = seq[seq_pos];

    if !node.children.contains_key(&first) {
        // No matching child — create a new leaf.
        let s = least_loaded(counts);
        let leaf_depth = node.node_depth + 1;
        let leaf = ShadowNode {
            edge: seq[seq_pos..].to_vec(),
            node_depth: leaf_depth,
            shard: Some(s),
            default_shard: s,
            children: FxHashMap::default(),
        };
        node_count.fetch_add(1, Ordering::Relaxed);
        // Update default_shard for future unseen extensions at this node.
        if node.shard.is_none() {
            node.default_shard = s;
        }
        node.children.insert(first, leaf);
        return (s, leaf_depth >= routing_depth);
    }

    // There is a child for `first`.
    // Read all needed fields in a single immutable borrow before any mutation.
    let (common, child_edge_len, child_depth) = {
        let child = &node.children[&first];
        (
            common_prefix_len(&child.edge, &seq[seq_pos..]),
            child.edge.len(),
            child.node_depth,
        )
    };

    if common == child_edge_len {
        // Full edge match.
        if child_depth >= routing_depth {
            // Child is at routing depth — return its shard directly.
            let c = &node.children[&first];
            return (c.shard.unwrap_or(c.default_shard), true);
        }
        // Descend.
        let child = node.children.get_mut(&first).unwrap();
        return insert_and_get_shard(
            child,
            seq,
            seq_pos + child_edge_len,
            routing_depth,
            counts,
            node_count,
        );
    }

    // Partial edge match at `common` blocks.
    // Don't split nodes that are already at routing depth.
    if child_depth >= routing_depth {
        let c = &node.children[&first];
        return (c.shard.unwrap_or(c.default_shard), true);
    }

    // --- Split the child's edge at `common` ---
    //
    // Before:
    //   node → child { edge: [A..common, B..end], depth: D, shard: S, children: ... }
    //
    // After:
    //   node → child { edge: [A..common], depth: D, shard: None, children: {
    //       B → suffix { edge: [B..end], depth: D+1, shard: S, children: ... }
    //       new_block → new_leaf { edge: [new..], depth: D+1, shard: new_S }
    //   }}

    let mut old_child = node.children.remove(&first).unwrap();

    // Split old_child.edge: prefix stays, suffix becomes a new child.
    let suffix_edge = old_child.edge.split_off(common); // old_child.edge is now the prefix
    let suffix_first = suffix_edge[0];
    let old_depth = old_child.node_depth;

    let suffix_shard = old_child.shard;
    let suffix_default = old_child.default_shard;

    let suffix = ShadowNode {
        edge: suffix_edge,
        node_depth: old_depth + 1,
        shard: suffix_shard,
        default_shard: suffix_default,
        children: std::mem::take(&mut old_child.children),
    };
    node_count.fetch_add(1, Ordering::Relaxed); // suffix node

    let suffix_s = suffix_shard.unwrap_or(suffix_default);

    // old_child becomes the common-prefix interior node.
    old_child.shard = None;
    old_child.default_shard = suffix_s;
    old_child.children.insert(suffix_first, suffix);

    let new_seq_pos = seq_pos + common;
    let (new_s, reached) = if new_seq_pos < seq.len() {
        let s = least_loaded(counts);
        let new_block = seq[new_seq_pos];
        let new_depth = old_depth + 1;
        let new_leaf = ShadowNode {
            edge: seq[new_seq_pos..].to_vec(),
            node_depth: new_depth,
            shard: Some(s),
            default_shard: s,
            children: FxHashMap::default(),
        };
        node_count.fetch_add(1, Ordering::Relaxed); // new leaf
        old_child.children.insert(new_block, new_leaf);
        (s, new_depth >= routing_depth)
    } else {
        // New sequence ends exactly at the split point (interior node).
        (old_child.default_shard, false)
    };

    node.children.insert(first, old_child);
    (new_s, reached)
}

// ---------------------------------------------------------------------------
// NodeDepthRoutingState — shared between in-process and multi-process variants
// ---------------------------------------------------------------------------

/// Shared routing state for node-depth sharding.
///
/// Encapsulates the shadow trie, `block_to_shard` map, and
/// `last_block_to_path` map.  Both [`NodeDepthShardedIndexer`] (in-process)
/// and `MultiProcessNodeDepthShardedIndexer` (cross-process HTTP) use this
/// to drive shard selection; only the dispatch mechanism differs.
///
/// # Thread safety
///
/// All fields use interior mutability (`RwLock`, `DashMap`).
/// `NodeDepthRoutingState` is `Send + Sync` and may be shared via `Arc`.
pub struct NodeDepthRoutingState {
    routing_trie: RwLock<ShadowNode>,
    /// Per-shard leaf counts for load-balanced assignment.
    /// Mutated only under `routing_trie.write()`; `Relaxed` atomics suffice.
    shard_counts: Vec<AtomicUsize>,
    /// Total shadow-trie node count (root=1 + all created nodes).
    trie_node_count: AtomicUsize,
    /// Remove index: `ExternalSequenceBlockHash.0` → shard index.
    block_to_shard: DashMap<u64, usize, FxBuildHasher>,
    /// Hybrid depth-aware path tracking.
    ///
    /// - `None`  — last block's event reached a routing leaf; future
    ///   continuations inherit the shard without touching the trie.
    /// - `Some(path)` — last block's event is shallow; `path` is the full
    ///   prefix from the trie root, used to reconstruct the correct insertion
    ///   point for the next continuation.
    last_block_to_path: DashMap<u64, Option<Vec<LocalBlockHash>>, FxBuildHasher>,
    routing_node_depth: usize,
    num_shards: usize,
}

impl NodeDepthRoutingState {
    pub fn new(num_shards: usize, routing_node_depth: usize) -> Self {
        let shard_counts = (0..num_shards).map(|_| AtomicUsize::new(0)).collect();
        Self {
            routing_trie: RwLock::new(ShadowNode::new_root()),
            shard_counts,
            trie_node_count: AtomicUsize::new(1), // root node
            block_to_shard: DashMap::with_hasher(FxBuildHasher),
            last_block_to_path: DashMap::with_hasher(FxBuildHasher),
            routing_node_depth: routing_node_depth.max(1),
            num_shards,
        }
    }

    /// Number of CRTC-node levels used for routing.
    pub fn routing_node_depth(&self) -> usize {
        self.routing_node_depth
    }

    /// Number of shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Look up which shard owns `seq` without modifying any state.
    /// Returns `None` on a trie miss (sequence not yet seen).
    pub async fn route(&self, seq: &[LocalBlockHash]) -> Option<usize> {
        let trie = self.routing_trie.read().await;
        route_sequence(&trie, seq, self.routing_node_depth)
    }

    async fn assign_via_trie(&self, seq: &[LocalBlockHash]) -> (usize, bool) {
        let mut trie = self.routing_trie.write().await;
        insert_and_get_shard(
            &mut trie,
            seq,
            0,
            self.routing_node_depth,
            &self.shard_counts,
            &self.trie_node_count,
        )
    }

    /// Resolve which shard a `Stored` event belongs to, then update
    /// `block_to_shard` and `last_block_to_path`.
    ///
    /// This is the authoritative routing decision point.  Callers should
    /// dispatch the event to the returned shard index after this call.
    pub async fn resolve_stored(&self, store_data: &KvCacheStoreData) -> usize {
        // Hybrid depth-aware routing:
        //
        // A. Parent reached a routing leaf (`last_block_to_path` → None):
        //    inherit its shard; mark this event's last block as leaf-reached too.
        //
        // B. Parent is shallow (`last_block_to_path` → Some(prefix)):
        //    reconstruct full sequence (prefix + incremental) and re-walk the trie,
        //    inserting as a proper child of the existing node.
        //
        // C. Parent not tracked (OOO, evicted) or root event:
        //    insert the incremental blocks from the trie root (best effort).
        enum Resolution {
            Inherit(usize),
            TrieInsert(Vec<LocalBlockHash>),
        }

        let incremental: Vec<LocalBlockHash> =
            store_data.blocks.iter().map(|b| b.tokens_hash).collect();

        let resolution = if let Some(parent_hash) = &store_data.parent_hash {
            if let Some(entry) = self.last_block_to_path.get(&parent_hash.0) {
                match entry.value() {
                    None => {
                        // Case A: parent reached a routing leaf — inherit its shard.
                        let s = match self.block_to_shard.get(&parent_hash.0).map(|v| *v) {
                            Some(s) => s,
                            None => {
                                drop(entry);
                                self.assign_via_trie(&incremental).await.0
                            }
                        };
                        Resolution::Inherit(s)
                    }
                    Some(prefix) => {
                        // Case B: parent is shallow — reconstruct full sequence from root.
                        let mut full = prefix.clone();
                        drop(entry);
                        full.extend_from_slice(&incremental);
                        Resolution::TrieInsert(full)
                    }
                }
            } else {
                // Case C: parent not in last_block_to_path.
                //
                // This fires for intermediate-block parents: when a cold worker's
                // first request stores [sys_0..sys_14, uniq_0..uniq_N] as one batch,
                // only the last block (uniq_N) is tracked in last_block_to_path.
                // The next request on that worker has parent=block_hash(sys_14) —
                // an intermediate block of the previous batch.  It IS in block_to_shard
                // (every stored block is indexed there), so we can safely inherit.
                //
                // Only fall through to trie insert for a genuinely unknown parent
                // (true OOO or evicted from block_to_shard).
                if let Some(s) = self.block_to_shard.get(&parent_hash.0).map(|v| *v) {
                    Resolution::Inherit(s)
                } else {
                    Resolution::TrieInsert(incremental)
                }
            }
        } else {
            // Case C: root event — no parent.
            Resolution::TrieInsert(incremental)
        };

        let shard_idx = match resolution {
            Resolution::Inherit(s) => {
                // Mark this event's last block as routing-leaf-reached so future
                // continuations also inherit without touching the trie.
                if let Some(last) = store_data.blocks.last() {
                    self.last_block_to_path.insert(last.block_hash.0, None);
                }
                s
            }
            Resolution::TrieInsert(full_seq) => {
                let (s, reached) = self.assign_via_trie(&full_seq).await;
                if let Some(last) = store_data.blocks.last() {
                    let path_entry = if reached { None } else { Some(full_seq) };
                    self.last_block_to_path
                        .insert(last.block_hash.0, path_entry);
                }
                s
            }
        };

        for block in &store_data.blocks {
            self.block_to_shard.insert(block.block_hash.0, shard_idx);
        }
        shard_idx
    }

    /// Resolve routing for a `Removed` event.
    ///
    /// Removes each block from the tracking maps and returns:
    /// - `per_shard`: one `Vec<ExternalSequenceBlockHash>` per shard — blocks
    ///   whose shard is known; these go directly to the owning shard.
    /// - `broadcast`: blocks whose shard is unknown (evicted from the map);
    ///   callers should fan these out to all shards.
    pub fn resolve_removed(
        &self,
        block_hashes: &[ExternalSequenceBlockHash],
    ) -> (
        Vec<Vec<ExternalSequenceBlockHash>>,
        Vec<ExternalSequenceBlockHash>,
    ) {
        let mut per_shard: Vec<Vec<ExternalSequenceBlockHash>> = vec![Vec::new(); self.num_shards];
        let mut broadcast: Vec<ExternalSequenceBlockHash> = Vec::new();

        for &bh in block_hashes {
            self.last_block_to_path.remove(&bh.0);
            match self.block_to_shard.remove(&bh.0) {
                Some((_, s)) => per_shard[s].push(bh),
                None => broadcast.push(bh),
            }
        }
        (per_shard, broadcast)
    }

    /// Clear all routing state (called on `Cleared` events).
    pub fn clear_maps(&self) {
        self.block_to_shard.clear();
        self.last_block_to_path.clear();
    }

    /// Returns `(total_leaves, per_shard_leaves, total_node_count)`.
    ///
    /// Reads from atomic counters — no lock acquired, safe to call from sync
    /// contexts (e.g. `timing_report`).  Values are eventually consistent.
    pub fn trie_stats(&self) -> (usize, Vec<usize>, usize) {
        let per_shard: Vec<usize> = self
            .shard_counts
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect();
        let total_leaves = per_shard.iter().sum();
        let total_nodes = self.trie_node_count.load(Ordering::Relaxed);
        (total_leaves, per_shard, total_nodes)
    }
}

// ---------------------------------------------------------------------------
// NodeDepthShardedIndexer — in-process variant
// ---------------------------------------------------------------------------

/// Shard by CRTC-node depth rather than block depth.
///
/// Routes by the first `routing_node_depth` compressed-radix-tree nodes
/// traversed, using a shadow trie that mirrors the CRTC's top-K node
/// structure.  Unlike [`BranchShardedIndexer`], this does not require knowing
/// the shared prefix length in advance.
///
/// Routing state is held in a [`NodeDepthRoutingState`]; the multi-process
/// variant reuses that state with HTTP dispatch instead of in-process calls.
pub struct NodeDepthShardedIndexer<T: SyncIndexer> {
    shards: Vec<Arc<ThreadPoolIndexer<T>>>,
    /// Shared routing state: shadow trie, block→shard map, path-tracking map.
    routing: NodeDepthRoutingState,
    kv_block_size: u32,
    read_pools: Option<Vec<ShardReadPool>>,

    // Timing / observability
    timing_calls: AtomicU64,
    timing_sum_routing_ns: AtomicU64,
    timing_sum_shard_ns: AtomicU64,
    find_matches_miss_count: AtomicU64,
    remove_broadcast_count: AtomicU64,
}

impl<T: SyncIndexer> NodeDepthShardedIndexer<T> {
    pub fn new(
        shards: Vec<ThreadPoolIndexer<T>>,
        routing_node_depth: usize,
        kv_block_size: u32,
        num_read_threads_per_shard: usize,
    ) -> Self {
        assert!(
            !shards.is_empty(),
            "NodeDepthShardedIndexer requires at least one shard"
        );
        let num_shards = shards.len();
        let shards: Vec<Arc<ThreadPoolIndexer<T>>> = shards.into_iter().map(Arc::new).collect();

        let read_pools = if num_read_threads_per_shard > 0 {
            Some(
                shards
                    .iter()
                    .map(|s| ShardReadPool::new(s.backend_arc(), num_read_threads_per_shard))
                    .collect(),
            )
        } else {
            None
        };

        Self {
            shards,
            routing: NodeDepthRoutingState::new(num_shards, routing_node_depth),
            kv_block_size,
            read_pools,
            timing_calls: AtomicU64::new(0),
            timing_sum_routing_ns: AtomicU64::new(0),
            timing_sum_shard_ns: AtomicU64::new(0),
            find_matches_miss_count: AtomicU64::new(0),
            remove_broadcast_count: AtomicU64::new(0),
        }
    }
}

#[async_trait]
impl<T: SyncIndexer> KvIndexerInterface for NodeDepthShardedIndexer<T> {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let t_routing = std::time::Instant::now();
        let shard_idx = match self.routing.route(&sequence).await {
            Some(idx) => idx,
            None => {
                self.find_matches_miss_count.fetch_add(1, Ordering::Relaxed);
                return Ok(OverlapScores::new());
            }
        };
        let routing_ns = t_routing.elapsed().as_nanos() as u64;

        let t_shard = std::time::Instant::now();
        let result = if let Some(pools) = &self.read_pools {
            let (resp_tx, resp_rx) = oneshot::channel();
            pools[shard_idx]
                .sender
                .send((sequence, resp_tx))
                .map_err(|_| KvRouterError::IndexerOffline)?;
            resp_rx.await.map_err(|_| KvRouterError::IndexerOffline)
        } else {
            self.shards[shard_idx].find_matches(sequence).await
        };
        let shard_ns = t_shard.elapsed().as_nanos() as u64;

        self.timing_calls.fetch_add(1, Ordering::Relaxed);
        self.timing_sum_routing_ns
            .fetch_add(routing_ns, Ordering::Relaxed);
        self.timing_sum_shard_ns
            .fetch_add(shard_ns, Ordering::Relaxed);

        result
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
        is_eagle: Option<bool>,
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.kv_block_size,
            BlockHashOptions {
                lora_name,
                is_eagle,
                block_mm_infos: None,
            },
        );
        match self.routing.route(&sequence).await {
            Some(idx) => self.shards[idx].find_matches(sequence).await,
            None => Ok(OverlapScores::new()),
        }
    }

    async fn apply_event(&self, event: RouterEvent) {
        match &event.event.data {
            KvCacheEventData::Stored(store_data) => {
                let shard_idx = self.routing.resolve_stored(store_data).await;
                self.shards[shard_idx].apply_event(event).await;
            }

            KvCacheEventData::Removed(remove_data) => {
                let (per_shard, broadcast) =
                    self.routing.resolve_removed(&remove_data.block_hashes);

                for (s, blocks) in per_shard.into_iter().enumerate() {
                    if blocks.is_empty() {
                        continue;
                    }
                    let e = RouterEvent {
                        worker_id: event.worker_id,
                        storage_tier: event.storage_tier,
                        event: KvCacheEvent {
                            event_id: event.event.event_id,
                            dp_rank: event.event.dp_rank,
                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                block_hashes: blocks,
                            }),
                        },
                    };
                    self.shards[s].apply_event(e).await;
                }

                if !broadcast.is_empty() {
                    self.remove_broadcast_count
                        .fetch_add(broadcast.len() as u64, Ordering::Relaxed);
                    for shard in &self.shards {
                        let e = RouterEvent {
                            worker_id: event.worker_id,
                            storage_tier: event.storage_tier,
                            event: KvCacheEvent {
                                event_id: event.event.event_id,
                                dp_rank: event.event.dp_rank,
                                data: KvCacheEventData::Removed(KvCacheRemoveData {
                                    block_hashes: broadcast.clone(),
                                }),
                            },
                        };
                        shard.apply_event(e).await;
                    }
                }
            }

            KvCacheEventData::Cleared => {
                self.routing.clear_maps();
                for shard in &self.shards {
                    shard.apply_event(event.clone()).await;
                }
            }
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        for shard in &self.shards {
            shard.remove_worker(worker_id).await;
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        for shard in &self.shards {
            shard.remove_worker_dp_rank(worker_id, dp_rank).await;
        }
    }

    fn shutdown(&self) {
        for shard in &self.shards {
            shard.shutdown();
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let mut all = Vec::new();
        for shard in &self.shards {
            all.extend(shard.dump_events().await?);
        }
        Ok(all)
    }

    async fn process_routing_decision_for_request(
        &self,
        _tokens_with_hashes: &mut TokensWithHashes,
        _worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        Ok(())
    }

    async fn flush(&self) -> usize {
        let mut total = 0;
        for shard in &self.shards {
            total += <ThreadPoolIndexer<T> as KvIndexerInterface>::flush(shard).await;
        }
        total
    }

    fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(idx, shard)| {
                // ThreadPoolIndexer::shard_sizes() already populates node_count
                // via backend.node_count() (O(1)); no need for node_edge_lengths().
                shard.shard_sizes().into_iter().map(move |mut s| {
                    s.shard_idx = idx;
                    s
                })
            })
            .collect()
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        self.shards
            .iter()
            .flat_map(|shard| shard.node_edge_lengths())
            .collect()
    }

    fn timing_report(&self) -> String {
        let dispatched = self.timing_calls.load(Ordering::Relaxed);
        let misses = self.find_matches_miss_count.load(Ordering::Relaxed);
        let total = dispatched + misses;
        if total == 0 {
            return String::new();
        }
        let miss_pct = 100.0 * misses as f64 / total as f64;
        let avg_routing_ns = if dispatched > 0 {
            self.timing_sum_routing_ns.load(Ordering::Relaxed) / dispatched
        } else {
            0
        };
        let avg_shard_us = if dispatched > 0 {
            self.timing_sum_shard_ns.load(Ordering::Relaxed) / dispatched / 1000
        } else {
            0
        };
        let mode = if self.read_pools.is_some() {
            "dedicated per-shard OS thread pool"
        } else {
            "inline on caller thread"
        };
        let (total_leaves, leaf_dist, total_trie_nodes) = self.routing.trie_stats();
        let leaf_dist_str: Vec<String> = leaf_dist
            .iter()
            .enumerate()
            .map(|(i, &c)| format!("shard[{}]={}", i, c))
            .collect();
        let broadcasts = self.remove_broadcast_count.load(Ordering::Relaxed);
        format!(
            "NodeDepthShardedIndexer(node_depth={}, {}) — {}/{} dispatched ({:.1}% miss)\n  \
             avg routing = {}ns | avg shard = {}µs\n  \
             shadow trie: {} nodes total ({} routing leaves: {}) | remove broadcasts: {}",
            self.routing.routing_node_depth(),
            mode,
            dispatched,
            total,
            miss_pct,
            avg_routing_ns,
            avg_shard_us,
            total_trie_nodes,
            total_leaves,
            leaf_dist_str.join(", "),
            broadcasts,
        )
    }
}
