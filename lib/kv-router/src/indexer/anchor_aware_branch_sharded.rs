// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anchor-aware routing-trie sharding over `ThreadPoolIndexer<T>`.
//!
//! This is an opt-in alternative to [`super::BranchShardedIndexer`]. The
//! existing branch-sharded indexer keeps its flat branch-key routing semantics;
//! this type owns a bounded routing prefix tree that can answer shallow drained
//! reads and route depth-boundary suffixes through explicit backend anchors.

use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

#[cfg(feature = "bench")]
use std::time::Instant;

use async_trait::async_trait;
use dashmap::{DashMap, DashSet};
use rustc_hash::{FxBuildHasher, FxHashSet};

#[cfg(feature = "bench")]
use super::ShardedIndexerMetrics;
use super::{
    AnchorRef, AnchorTask, KvIndexerInterface, KvRouterError, ShardSizeSnapshot, SyncIndexer,
    ThreadPoolIndexer,
};
use crate::protocols::*;

type WorkerRoutingLookup = DashMap<ExternalSequenceBlockHash, BlockRoutingEntry, FxBuildHasher>;
type DumpedBlockSet = FxHashSet<(WorkerWithDpRank, ExternalSequenceBlockHash)>;

struct RoutingNode {
    key: Option<LocalBlockHash>,
    external_hash: Option<ExternalSequenceBlockHash>,
    depth: usize,
    shard: AtomicUsize,
    children: DashMap<LocalBlockHash, Arc<RoutingNode>, FxBuildHasher>,
    create_lock: Mutex<()>,
    live_workers: DashSet<WorkerWithDpRank, FxBuildHasher>,
}

impl RoutingNode {
    fn root() -> Self {
        Self {
            key: None,
            external_hash: None,
            depth: 0,
            shard: AtomicUsize::new(0),
            children: DashMap::with_hasher(FxBuildHasher),
            create_lock: Mutex::new(()),
            live_workers: DashSet::with_hasher(FxBuildHasher),
        }
    }

    fn new(
        key: LocalBlockHash,
        external_hash: ExternalSequenceBlockHash,
        depth: usize,
        shard: usize,
    ) -> Self {
        Self {
            key: Some(key),
            external_hash: Some(external_hash),
            depth,
            shard: AtomicUsize::new(shard),
            children: DashMap::with_hasher(FxBuildHasher),
            create_lock: Mutex::new(()),
            live_workers: DashSet::with_hasher(FxBuildHasher),
        }
    }

    fn shard(&self) -> usize {
        self.shard.load(Ordering::Relaxed)
    }
}

#[derive(Clone)]
struct BlockRoutingEntry {
    shard_idx: usize,
    routing_node: Arc<RoutingNode>,
    sequence_depth: usize,
    affects_router_node: bool,
}

struct StoreRouteDecision {
    shard_idx: usize,
    anchor: Option<AnchorRef>,
    anchor_block_offset: usize,
    rewrite_for_anchor: bool,
    skip_backend: bool,
}

/// Anchor-aware branch-sharded wrapper over N [`ThreadPoolIndexer<T>`] instances.
pub struct AnchorAwareBranchShardedIndexer<T: SyncIndexer> {
    shards: Vec<Arc<ThreadPoolIndexer<T>>>,
    num_shards: usize,
    max_routing_depth: usize,
    kv_block_size: u32,
    root: Arc<RoutingNode>,
    worker_block_index: DashMap<WorkerWithDpRank, WorkerRoutingLookup, FxBuildHasher>,
    installed_worker_anchors: DashSet<(usize, WorkerWithDpRank, u64), FxBuildHasher>,
    #[cfg(feature = "bench")]
    metrics: ShardedIndexerMetrics,
}

impl<T: SyncIndexer> AnchorAwareBranchShardedIndexer<T> {
    /// Create an anchor-aware branch-sharded indexer from pre-built shards.
    pub fn new(shards: Vec<ThreadPoolIndexer<T>>, prefix_depth: usize, kv_block_size: u32) -> Self {
        assert!(!shards.is_empty(), "Must provide at least one shard");
        let num_shards = shards.len();
        let shards = shards.into_iter().map(Arc::new).collect();

        Self {
            shards,
            num_shards,
            max_routing_depth: prefix_depth.max(1),
            kv_block_size,
            root: Arc::new(RoutingNode::root()),
            worker_block_index: DashMap::with_hasher(FxBuildHasher),
            installed_worker_anchors: DashSet::with_hasher(FxBuildHasher),
            #[cfg(feature = "bench")]
            metrics: ShardedIndexerMetrics::new(),
        }
    }

    /// Alias for [`AnchorAwareBranchShardedIndexer::new`].
    pub fn new_with_options(
        shards: Vec<ThreadPoolIndexer<T>>,
        prefix_depth: usize,
        kv_block_size: u32,
    ) -> Self {
        Self::new(shards, prefix_depth, kv_block_size)
    }

    fn static_divergent_shard(
        &self,
        parent_shard: usize,
        parent: &RoutingNode,
        block: &KvCacheStoredBlockData,
    ) -> usize {
        if self.num_shards == 1 {
            return 0;
        }

        // TODO: Static hashing still cannot split one very hot branch after it
        // lands on a shard. If real traces remain imbalanced, add adaptive or
        // deeper hot-branch splitting on top of this deterministic baseline.
        let parent_seq_hash = parent.external_hash.map(|hash| hash.0).unwrap_or(0);
        let hash = compute_next_seq_hash(parent_seq_hash, block.tokens_hash);
        let slot = (hash as usize) % (self.num_shards - 1);
        if slot >= parent_shard { slot + 1 } else { slot }
    }

    fn anchor_for_parent(&self, parent: &RoutingNode) -> Option<AnchorRef> {
        if parent.depth == 0 {
            return None;
        }
        let anchor_id = parent.external_hash?;
        Some(AnchorRef {
            anchor_id,
            anchor_local_hash: parent.key.unwrap_or(LocalBlockHash(anchor_id.0)),
            anchor_depth: parent.depth,
        })
    }

    fn get_or_create_child(
        &self,
        parent: &Arc<RoutingNode>,
        block: &KvCacheStoredBlockData,
    ) -> Arc<RoutingNode> {
        if let Some(child) = parent.children.get(&block.tokens_hash) {
            return child.clone();
        }

        let _guard = parent.create_lock.lock().unwrap();

        if let Some(child) = parent.children.get(&block.tokens_hash) {
            return child.clone();
        }

        let parent_shard = parent.shard();
        let shard = if parent.children.is_empty() {
            parent_shard
        } else {
            self.static_divergent_shard(parent_shard, parent, block)
        };

        let child = Arc::new(RoutingNode::new(
            block.tokens_hash,
            block.block_hash,
            parent.depth + 1,
            shard,
        ));
        parent.children.insert(block.tokens_hash, child.clone());
        child
    }

    fn worker_lookup(
        &self,
        worker: WorkerWithDpRank,
    ) -> dashmap::mapref::one::RefMut<'_, WorkerWithDpRank, WorkerRoutingLookup> {
        self.worker_block_index
            .entry(worker)
            .or_insert_with(|| DashMap::with_hasher(FxBuildHasher))
    }

    fn route_stored(
        &self,
        worker: WorkerWithDpRank,
        store_data: &KvCacheStoreData,
    ) -> StoreRouteDecision {
        let mut start_depth = 0usize;
        let mut node = self.root.clone();

        let parent_entry = store_data.parent_hash.and_then(|parent_hash| {
            self.worker_block_index
                .get(&worker)
                .and_then(|lookup| lookup.get(&parent_hash).map(|entry| entry.clone()))
        });
        if let Some(entry) = parent_entry {
            node = entry.routing_node.clone();
            start_depth = entry.sequence_depth;
        }

        let lookup = self.worker_lookup(worker);

        for (offset, block) in store_data.blocks.iter().enumerate() {
            let sequence_depth = start_depth + offset + 1;
            if node.depth < self.max_routing_depth {
                node = self.get_or_create_child(&node, block);
                node.live_workers.insert(worker);
                lookup.entry(block.block_hash).or_insert(BlockRoutingEntry {
                    shard_idx: node.shard(),
                    routing_node: node.clone(),
                    sequence_depth,
                    affects_router_node: true,
                });
            } else {
                lookup.entry(block.block_hash).or_insert(BlockRoutingEntry {
                    shard_idx: node.shard(),
                    routing_node: node.clone(),
                    sequence_depth,
                    affects_router_node: false,
                });
            }
        }
        drop(lookup);

        let shard_idx = node.shard();
        let router_owned_blocks = self
            .max_routing_depth
            .saturating_sub(start_depth)
            .min(store_data.blocks.len());
        let needs_boundary_anchor =
            start_depth <= self.max_routing_depth && router_owned_blocks < store_data.blocks.len();
        let anchor = needs_boundary_anchor
            .then(|| self.anchor_for_parent(&node))
            .flatten();
        let (anchor_block_offset, rewrite_for_anchor) = if anchor.is_some() {
            (router_owned_blocks, true)
        } else {
            (0, false)
        };
        let skip_backend =
            start_depth <= self.max_routing_depth && router_owned_blocks >= store_data.blocks.len();

        StoreRouteDecision {
            shard_idx,
            anchor,
            anchor_block_offset,
            rewrite_for_anchor,
            skip_backend,
        }
    }

    fn add_active_scores(
        scores: &mut OverlapScores,
        active: &FxHashSet<WorkerWithDpRank>,
        depth: usize,
    ) {
        let score = depth as u32;
        for &worker in active {
            let entry = scores.scores.entry(worker).or_insert(0);
            *entry = (*entry).max(score);
        }
    }

    fn collect_live_workers(node: &RoutingNode) -> FxHashSet<WorkerWithDpRank> {
        node.live_workers.iter().map(|worker| *worker).collect()
    }

    fn reconcile_active_workers(
        scores: &mut OverlapScores,
        active: &mut FxHashSet<WorkerWithDpRank>,
        node: &RoutingNode,
        drop_depth: usize,
    ) {
        if active
            .iter()
            .all(|worker| node.live_workers.contains(worker))
        {
            return;
        }
        let score = drop_depth as u32;
        active.retain(|worker| {
            if node.live_workers.contains(worker) {
                true
            } else {
                let entry = scores.scores.entry(*worker).or_insert(0);
                *entry = (*entry).max(score);
                false
            }
        });
    }

    fn dispatch_read(
        &self,
        node: Arc<RoutingNode>,
        sequence: Vec<LocalBlockHash>,
        mut scores: OverlapScores,
        active: FxHashSet<WorkerWithDpRank>,
    ) -> Result<OverlapScores, KvRouterError> {
        if active.is_empty() {
            return Ok(scores);
        }
        let shard_idx = node.shard();
        #[cfg(feature = "bench")]
        self.metrics
            .counters
            .find_match_dispatches
            .fetch_add(1, Ordering::Relaxed);
        let Some(anchor) = self.anchor_for_parent(&node) else {
            return Ok(scores);
        };
        let suffix = if anchor.anchor_depth <= sequence.len() {
            &sequence[anchor.anchor_depth..]
        } else {
            &[]
        };
        let mut shard_scores = self.shards[shard_idx]
            .backend()
            .find_matches_from_anchor(anchor, suffix)?;
        for (worker, shard_score) in shard_scores.scores.drain() {
            if !active.contains(&worker) {
                continue;
            }
            let entry = scores.scores.entry(worker).or_insert(0);
            *entry = (*entry).max(shard_score);
        }
        Ok(scores)
    }

    fn ensure_worker_anchor(&self, shard_idx: usize, worker: WorkerWithDpRank, anchor: AnchorRef) {
        let key = (shard_idx, worker, anchor.anchor_id.0);
        if self.installed_worker_anchors.insert(key) {
            let task = AnchorTask {
                anchor_id: anchor.anchor_id,
                anchor_local_hash: anchor.anchor_local_hash,
                anchor_depth: anchor.anchor_depth,
            };
            // Anchor installs are deduped per worker queue, not globally. Each
            // dependent worker carries its own Anchor-before-Stored FIFO edge,
            // while backend anchor application is idempotent by anchor_id.
            match self.shards[shard_idx].enqueue_anchor(worker, task) {
                Ok(()) => {
                    #[cfg(feature = "bench")]
                    self.metrics
                        .counters
                        .anchor_installs
                        .fetch_add(1, Ordering::Relaxed);
                }
                Err(error) => {
                    tracing::warn!(?error, shard_idx, ?worker, "Failed to enqueue anchor");
                }
            }
        } else {
            #[cfg(feature = "bench")]
            self.metrics
                .counters
                .anchor_reuses
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn rewritten_store_event(
        &self,
        mut event: RouterEvent,
        decision: &StoreRouteDecision,
    ) -> Option<RouterEvent> {
        let anchor = decision.anchor?;
        let KvCacheEventData::Stored(store_data) = &mut event.event.data else {
            return None;
        };
        if decision.anchor_block_offset > store_data.blocks.len() {
            return None;
        }
        let blocks = store_data.blocks.split_off(decision.anchor_block_offset);
        if blocks.is_empty() {
            return None;
        }
        store_data.parent_hash = Some(anchor.anchor_id);
        store_data.blocks = blocks;
        Some(event)
    }

    fn remove_worker_entries(&self, worker: WorkerWithDpRank) {
        let Some((_, lookup)) = self.worker_block_index.remove(&worker) else {
            return;
        };
        let mut seen_nodes = FxHashSet::default();
        for (_, entry) in lookup {
            if entry.affects_router_node {
                let ptr = Arc::as_ptr(&entry.routing_node) as usize;
                if seen_nodes.insert(ptr) {
                    entry.routing_node.live_workers.remove(&worker);
                }
            }
        }
    }

    fn dump_router_events(&self) -> (Vec<RouterEvent>, DumpedBlockSet) {
        // TODO: Static shard routing treats the first structural child under a
        // parent as special. This dump is replayable, but it does not yet make
        // sibling traversal canonical by original creation order, so byte-stable
        // dump -> replay -> dump comparisons are not guaranteed for siblings.
        let mut events = Vec::new();
        let mut dumped_blocks = FxHashSet::default();
        let mut event_id = 0u64;
        let mut queue = VecDeque::new();

        for child in self.root.children.iter() {
            queue.push_back((child.clone(), None, None::<FxHashSet<WorkerWithDpRank>>));
        }

        while let Some((node, parent_hash, parent_live_workers)) = queue.pop_front() {
            let node_workers = Self::collect_live_workers(&node);
            let live_workers = match parent_live_workers {
                Some(parent_workers) => node_workers
                    .intersection(&parent_workers)
                    .copied()
                    .collect::<FxHashSet<_>>(),
                None => node_workers,
            };
            if live_workers.is_empty() {
                continue;
            }

            let tokens_hash = node.key.expect("non-root routing node must have key");
            let block_hash = node
                .external_hash
                .expect("non-root routing node must have external hash");
            let block = KvCacheStoredBlockData {
                tokens_hash,
                block_hash,
                mm_extra_info: None,
            };

            for worker in &live_workers {
                events.push(RouterEvent::new(
                    worker.worker_id,
                    KvCacheEvent {
                        event_id,
                        data: KvCacheEventData::Stored(KvCacheStoreData {
                            parent_hash,
                            start_position: None,
                            blocks: vec![block.clone()],
                        }),
                        dp_rank: worker.dp_rank,
                    },
                ));
                dumped_blocks.insert((*worker, block_hash));
                event_id += 1;
            }

            for child in node.children.iter() {
                queue.push_back((child.clone(), Some(block_hash), Some(live_workers.clone())));
            }
        }

        (events, dumped_blocks)
    }

    fn append_reachable_shard_events(
        all_events: &mut Vec<RouterEvent>,
        dumped_blocks: &mut DumpedBlockSet,
        shard_events: Vec<RouterEvent>,
    ) {
        for event in shard_events {
            let KvCacheEventData::Stored(store_data) = &event.event.data else {
                all_events.push(event);
                continue;
            };
            let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
            if let Some(parent_hash) = store_data.parent_hash
                && !dumped_blocks.contains(&(worker, parent_hash))
            {
                continue;
            }
            for block in &store_data.blocks {
                dumped_blocks.insert((worker, block.block_hash));
            }
            all_events.push(event);
        }
    }

    async fn apply_stored(&self, event: RouterEvent) {
        let KvCacheEventData::Stored(store_data) = &event.event.data else {
            return;
        };
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let decision = self.route_stored(worker, store_data);

        if decision.skip_backend {
            return;
        }

        if decision.rewrite_for_anchor
            && let Some(anchor) = decision.anchor
        {
            self.ensure_worker_anchor(decision.shard_idx, worker, anchor);
            if let Some(rewritten) = self.rewritten_store_event(event, &decision) {
                self.shards[decision.shard_idx].apply_event(rewritten).await;
            }
            return;
        }

        self.shards[decision.shard_idx].apply_event(event).await;
    }

    async fn apply_removed(&self, event: RouterEvent) {
        let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
        let KvCacheEventData::Removed(remove_data) = &event.event.data else {
            return;
        };

        let mut shard_blocks = vec![Vec::new(); self.num_shards];
        let mut broadcast_blocks = Vec::new();

        if let Some(lookup) = self.worker_block_index.get(&worker) {
            for &block_hash in &remove_data.block_hashes {
                let Some(entry) = lookup.get(&block_hash).map(|entry| entry.clone()) else {
                    #[cfg(feature = "bench")]
                    self.metrics
                        .counters
                        .remove_broadcasts
                        .fetch_add(1, Ordering::Relaxed);
                    broadcast_blocks.push(block_hash);
                    continue;
                };

                if entry.affects_router_node {
                    entry.routing_node.live_workers.remove(&worker);
                    lookup.remove(&block_hash);
                } else {
                    shard_blocks[entry.shard_idx].push(block_hash);
                    lookup.remove(&block_hash);
                }
            }
        } else {
            for &block_hash in &remove_data.block_hashes {
                #[cfg(feature = "bench")]
                self.metrics
                    .counters
                    .remove_broadcasts
                    .fetch_add(1, Ordering::Relaxed);
                broadcast_blocks.push(block_hash);
            }
        }

        for (shard_idx, blocks) in shard_blocks.into_iter().enumerate() {
            if blocks.is_empty() {
                continue;
            }
            let shard_event = RouterEvent {
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
            self.shards[shard_idx].apply_event(shard_event).await;
        }

        if !broadcast_blocks.is_empty() {
            for shard in &self.shards {
                let broadcast_event = RouterEvent {
                    worker_id: event.worker_id,
                    storage_tier: event.storage_tier,
                    event: KvCacheEvent {
                        event_id: event.event.event_id,
                        dp_rank: event.event.dp_rank,
                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: broadcast_blocks.clone(),
                        }),
                    },
                };
                shard.apply_event(broadcast_event).await;
            }
        }
    }
}

#[async_trait]
impl<T: SyncIndexer> KvIndexerInterface for AnchorAwareBranchShardedIndexer<T> {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        #[cfg(feature = "bench")]
        let t_routing = Instant::now();

        let mut node = self.root.clone();
        let mut depth = 0usize;
        let mut router_scores = OverlapScores::new();
        let mut active = FxHashSet::default();

        for hash in &sequence {
            if depth == self.max_routing_depth {
                Self::add_active_scores(&mut router_scores, &active, depth);
                #[cfg(feature = "bench")]
                let routing_ns = t_routing.elapsed().as_nanos() as u64;
                #[cfg(feature = "bench")]
                let t_shard = Instant::now();
                let result = self.dispatch_read(node, sequence, router_scores, active);
                #[cfg(feature = "bench")]
                {
                    self.metrics.timing.calls.fetch_add(1, Ordering::Relaxed);
                    self.metrics
                        .timing
                        .routing_ns
                        .fetch_add(routing_ns, Ordering::Relaxed);
                    self.metrics
                        .timing
                        .shard_ns
                        .fetch_add(t_shard.elapsed().as_nanos() as u64, Ordering::Relaxed);
                }
                return result;
            }

            let child = node.children.get(hash).map(|child| child.clone());
            match child {
                Some(child) => {
                    node = child;
                    depth += 1;
                    if depth == 1 {
                        active = Self::collect_live_workers(&node);
                    } else {
                        Self::reconcile_active_workers(
                            &mut router_scores,
                            &mut active,
                            &node,
                            depth - 1,
                        );
                    }
                    if active.is_empty() {
                        #[cfg(feature = "bench")]
                        self.metrics
                            .counters
                            .find_match_early_returns
                            .fetch_add(1, Ordering::Relaxed);
                        return Ok(router_scores);
                    }
                }
                None => {
                    #[cfg(feature = "bench")]
                    self.metrics
                        .counters
                        .find_match_early_returns
                        .fetch_add(1, Ordering::Relaxed);
                    Self::add_active_scores(&mut router_scores, &active, depth);
                    return Ok(router_scores);
                }
            }
        }

        #[cfg(feature = "bench")]
        self.metrics
            .counters
            .find_match_early_returns
            .fetch_add(1, Ordering::Relaxed);
        Self::add_active_scores(&mut router_scores, &active, depth);
        Ok(router_scores)
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
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        match &event.event.data {
            KvCacheEventData::Stored(_) => self.apply_stored(event).await,
            KvCacheEventData::Removed(_) => self.apply_removed(event).await,
            KvCacheEventData::Cleared => {
                let worker_id = event.worker_id;
                let workers: Vec<_> = self
                    .worker_block_index
                    .iter()
                    .filter(|entry| entry.key().worker_id == worker_id)
                    .map(|entry| *entry.key())
                    .collect();
                for worker in workers {
                    self.remove_worker_entries(worker);
                }
                for shard in &self.shards {
                    shard.apply_event(event.clone()).await;
                }
            }
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        let workers: Vec<_> = self
            .worker_block_index
            .iter()
            .filter(|entry| entry.key().worker_id == worker_id)
            .map(|entry| *entry.key())
            .collect();
        for worker in workers {
            self.remove_worker_entries(worker);
        }
        for shard in &self.shards {
            shard.remove_worker(worker_id).await;
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.remove_worker_entries(WorkerWithDpRank::new(worker_id, dp_rank));
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
        let (mut all_events, mut dumped_blocks) = self.dump_router_events();
        for shard in &self.shards {
            Self::append_reachable_shard_events(
                &mut all_events,
                &mut dumped_blocks,
                shard.dump_events().await?,
            );
        }
        for (idx, event) in all_events.iter_mut().enumerate() {
            event.event.event_id = idx as u64;
        }
        Ok(all_events)
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

    async fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        let mut sizes = Vec::new();
        for (idx, shard) in self.shards.iter().enumerate() {
            sizes.extend(
                shard
                    .shard_sizes()
                    .await
                    .into_iter()
                    .map(move |mut snapshot| {
                        snapshot.shard_idx = idx;
                        snapshot
                    }),
            );
        }
        sizes
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        self.shards
            .iter()
            .flat_map(|shard| shard.node_edge_lengths())
            .collect()
    }

    fn timing_report(&self) -> String {
        #[cfg(not(feature = "bench"))]
        {
            String::new()
        }

        #[cfg(feature = "bench")]
        {
            let dispatched = self
                .metrics
                .counters
                .find_match_dispatches
                .load(Ordering::Relaxed);
            let shallow = self
                .metrics
                .counters
                .find_match_early_returns
                .load(Ordering::Relaxed);
            let total_calls = dispatched + shallow;
            if total_calls == 0 {
                return String::new();
            }
            let broadcasts = self
                .metrics
                .counters
                .remove_broadcasts
                .load(Ordering::Relaxed);
            let anchor_installs = self
                .metrics
                .counters
                .anchor_installs
                .load(Ordering::Relaxed);
            let anchor_reuses = self.metrics.counters.anchor_reuses.load(Ordering::Relaxed);

            let timing = {
                let calls = self.metrics.timing.calls.load(Ordering::Relaxed);
                let avg_routing_ns = if calls > 0 {
                    self.metrics.timing.routing_ns.load(Ordering::Relaxed) / calls
                } else {
                    0
                };
                let avg_shard_us = if calls > 0 {
                    self.metrics.timing.shard_ns.load(Ordering::Relaxed) / calls / 1000
                } else {
                    0
                };
                format!("\n  avg routing = {avg_routing_ns}ns\n  avg shard = {avg_shard_us}µs")
            };

            format!(
                "AnchorAwareBranchShardedIndexer find_matches ({total_calls} total: {dispatched} dispatched, \
             {shallow} shallow):{timing}\n  \
             remove broadcasts = {broadcasts}\n  \
             anchors = {anchor_installs} installs / {anchor_reuses} reuses"
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;
    use crate::indexer::concurrent_radix_tree_compressed::ConcurrentRadixTreeCompressed;
    use crate::test_utils::{remove_event, router_event, stored_blocks_with_sequence_hashes};
    use tokio::sync::Barrier as AsyncBarrier;

    fn make_indexer(
        num_shards: usize,
        depth: usize,
    ) -> AnchorAwareBranchShardedIndexer<ConcurrentRadixTreeCompressed> {
        let shards = (0..num_shards)
            .map(|_| ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 32))
            .collect();
        AnchorAwareBranchShardedIndexer::new(shards, depth, 32)
    }

    fn local_hashes(values: &[u64]) -> Vec<LocalBlockHash> {
        values.iter().copied().map(LocalBlockHash).collect()
    }

    fn stored_blocks(values: &[u64]) -> Vec<KvCacheStoredBlockData> {
        let locals = local_hashes(values);
        let seq_hashes = compute_seq_hash_for_block(&locals);
        stored_blocks_with_sequence_hashes(&locals, &seq_hashes)
    }

    fn store_event(worker_id: u64, values: &[u64]) -> RouterEvent {
        store_event_with_dp_rank(worker_id, 0, values)
    }

    fn store_event_with_dp_rank(worker_id: u64, dp_rank: u32, values: &[u64]) -> RouterEvent {
        router_event(
            worker_id,
            0,
            dp_rank,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: stored_blocks(values),
            }),
        )
    }

    fn store_event_with_parent(
        worker_id: u64,
        parent_values: &[u64],
        suffix_values: &[u64],
    ) -> RouterEvent {
        let parent_hashes = compute_seq_hash_for_block(&local_hashes(parent_values));
        let parent_hash = parent_hashes.last().copied().map(ExternalSequenceBlockHash);
        let mut full_values = parent_values.to_vec();
        full_values.extend_from_slice(suffix_values);
        let full_hashes = local_hashes(&full_values);
        let seq_hashes = compute_seq_hash_for_block(&full_hashes);
        let suffix_hashes = local_hashes(suffix_values);
        let suffix_seq_hashes = &seq_hashes[parent_values.len()..];

        router_event(
            worker_id,
            0,
            0,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash,
                start_position: None,
                blocks: stored_blocks_with_sequence_hashes(&suffix_hashes, suffix_seq_hashes),
            }),
        )
    }

    fn remove_hash_event(
        worker_id: u64,
        dp_rank: u32,
        full_sequence: &[u64],
        removed_idx: usize,
    ) -> RouterEvent {
        let locals = local_hashes(full_sequence);
        let seq_hashes = compute_seq_hash_for_block(&locals);
        remove_event(
            worker_id,
            0,
            dp_rank,
            vec![ExternalSequenceBlockHash(seq_hashes[removed_idx])],
        )
    }

    fn clear_event(worker_id: u64) -> RouterEvent {
        router_event(worker_id, 0, 0, KvCacheEventData::Cleared)
    }

    fn child(parent: &Arc<RoutingNode>, key: u64) -> Arc<RoutingNode> {
        parent
            .children
            .get(&LocalBlockHash(key))
            .expect("expected routing child")
            .clone()
    }

    fn worker(worker_id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(worker_id, 0)
    }

    fn score(scores: &OverlapScores, worker: WorkerWithDpRank) -> Option<u32> {
        scores.scores.get(&worker).copied()
    }

    #[tokio::test]
    async fn linear_chain_inherits_parent_and_caps_construction() {
        let index = make_indexer(2, 4);
        index.apply_event(store_event(0, &[1, 2, 3, 4, 5, 6])).await;

        let a = child(&index.root, 1);
        let b = child(&a, 2);
        let c = child(&b, 3);
        let d = child(&c, 4);

        assert_eq!(a.shard(), 0);
        assert_eq!(b.shard(), 0);
        assert_eq!(c.shard(), 0);
        assert_eq!(d.shard(), 0);
        assert!(d.children.get(&LocalBlockHash(5)).is_none());

        let lookup = index.worker_block_index.get(&worker(0)).unwrap();
        let seq_hashes = compute_seq_hash_for_block(&local_hashes(&[1, 2, 3, 4, 5, 6]));
        let suffix_entry = lookup
            .get(&ExternalSequenceBlockHash(seq_hashes[5]))
            .expect("suffix block should be reverse-indexed");
        assert_eq!(suffix_entry.shard_idx, 0);
        assert!(!suffix_entry.affects_router_node);
        assert!(Arc::ptr_eq(&suffix_entry.routing_node, &d));
    }

    #[tokio::test]
    async fn structural_divergence_and_zombie_history_are_sticky() {
        let index = make_indexer(2, 4);
        index.apply_event(store_event(0, &[1, 2, 3])).await;

        let a = child(&index.root, 1);
        let b = child(&a, 2);
        let c = child(&b, 3);
        assert_eq!(c.shard(), 0);

        index
            .apply_event(remove_hash_event(0, 0, &[1, 2, 3], 2))
            .await;
        assert!(c.live_workers.is_empty());
        assert!(b.children.get(&LocalBlockHash(3)).is_some());

        index.apply_event(store_event(1, &[1, 2, 5])).await;
        let e = child(&b, 5);
        assert_eq!(e.shard(), 1);

        index.apply_event(store_event(2, &[1, 2, 6])).await;
        let f = child(&b, 6);
        assert_eq!(f.shard(), 1);
        assert_eq!(b.children.len(), 3);
    }

    #[tokio::test]
    async fn static_divergent_assignment_excludes_parent_shard() {
        let index = make_indexer(4, 4);
        index.apply_event(store_event(0, &[1, 2, 3])).await;

        let b = child(&child(&index.root, 1), 2);
        let block = stored_blocks(&[1, 2, 5]).remove(2);
        let expected = index.static_divergent_shard(b.shard(), &b, &block);
        assert_ne!(expected, b.shard());

        index.apply_event(store_event(1, &[1, 2, 5])).await;
        let e = child(&b, 5);
        assert_eq!(e.shard(), expected);
    }

    #[test]
    fn concurrent_sibling_creation_under_hot_prefix_has_one_first_child() {
        let index = Arc::new(make_indexer(2, 4));
        let ab_blocks = stored_blocks(&[1, 2]);
        let a = index.get_or_create_child(&index.root, &ab_blocks[0]);
        let b = index.get_or_create_child(&a, &ab_blocks[1]);

        let sibling_count = 4;
        let barrier = Arc::new(Barrier::new(sibling_count));
        let mut handles = Vec::new();

        for key in 10..(10 + sibling_count as u64) {
            let index = index.clone();
            let barrier = barrier.clone();
            let parent = b.clone();
            handles.push(std::thread::spawn(move || {
                let block = stored_blocks(&[1, 2, key]).remove(2);
                barrier.wait();
                index.get_or_create_child(&parent, &block)
            }));
        }

        let children: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let inherited_children = children
            .iter()
            .filter(|node| node.shard() == b.shard())
            .count();

        assert_eq!(inherited_children, 1);
        assert_eq!(b.children.len(), sibling_count);
    }

    #[tokio::test]
    async fn drained_read_returns_router_scores_for_all_live_branch_workers() {
        let index = make_indexer(2, 4);
        index.apply_event(store_event(0, &[1, 2, 3, 4])).await;
        index.apply_event(store_event(1, &[1, 2, 5, 6])).await;

        let scores = index
            .find_matches(local_hashes(&[1, 2, 9, 10]))
            .await
            .unwrap();

        assert_eq!(score(&scores, worker(0)), Some(2));
        assert_eq!(score(&scores, worker(1)), Some(2));
        #[cfg(feature = "bench")]
        {
            assert_eq!(
                index
                    .metrics
                    .counters
                    .find_match_dispatches
                    .load(Ordering::Relaxed),
                0
            );
        }
    }

    #[tokio::test]
    async fn prefix_only_parent_continuations_stay_router_only() {
        let index = make_indexer(2, 3);
        index.apply_event(store_event(0, &[1])).await;
        index
            .apply_event(store_event_with_parent(0, &[1], &[2]))
            .await;
        index
            .apply_event(store_event_with_parent(0, &[1, 2], &[3]))
            .await;
        index.flush().await;

        let scores = index.find_matches(local_hashes(&[1, 2, 3])).await.unwrap();
        let backend_blocks: usize = index
            .shard_sizes()
            .await
            .iter()
            .map(|snapshot| snapshot.block_count)
            .sum();

        assert_eq!(score(&scores, worker(0)), Some(3));
        #[cfg(feature = "bench")]
        {
            assert_eq!(
                index
                    .metrics
                    .counters
                    .anchor_installs
                    .load(Ordering::Relaxed),
                0
            );
        }
        assert_eq!(backend_blocks, 0);
    }

    #[tokio::test]
    async fn depth_dispatch_uses_boundary_anchor_for_divergent_suffix() {
        let index = make_indexer(2, 3);
        index.apply_event(store_event(0, &[1, 2, 3, 4])).await;
        index.apply_event(store_event(1, &[1, 2, 5, 6])).await;
        index.flush().await;

        let scores = index
            .find_matches(local_hashes(&[1, 2, 5, 6]))
            .await
            .unwrap();

        assert_eq!(score(&scores, worker(1)), Some(4));
        #[cfg(feature = "bench")]
        {
            assert_eq!(
                index
                    .metrics
                    .counters
                    .anchor_installs
                    .load(Ordering::Relaxed),
                2
            );
            assert_eq!(
                index
                    .metrics
                    .counters
                    .find_match_dispatches
                    .load(Ordering::Relaxed),
                1
            );
        }

        index
            .apply_event(remove_hash_event(1, 0, &[1, 2, 5, 6], 2))
            .await;
        index.flush().await;

        let scores_after_remove = index
            .find_matches(local_hashes(&[1, 2, 5, 6]))
            .await
            .unwrap();
        assert_eq!(score(&scores_after_remove, worker(0)), Some(2));
        assert_eq!(score(&scores_after_remove, worker(1)), Some(2));
    }

    #[tokio::test]
    async fn anchor_installed_by_one_worker_is_visible_to_another_worker_queue() {
        let index = make_indexer(2, 3);
        index.apply_event(store_event(0, &[1, 2, 3, 4])).await;
        index.apply_event(store_event(1, &[1, 2, 5, 6])).await;
        index.flush().await;

        let b = child(&child(&index.root, 1), 2);
        let e = child(&b, 5);
        let anchor = index
            .anchor_for_parent(&e)
            .expect("expected boundary anchor");
        let shard_idx = e.shard();

        let full = local_hashes(&[1, 2, 5, 7]);
        let seq_hashes = compute_seq_hash_for_block(&full);
        let suffix_blocks = stored_blocks_with_sequence_hashes(&full[3..], &seq_hashes[3..]);
        let direct_worker_event = router_event(
            2,
            0,
            0,
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(anchor.anchor_id),
                start_position: None,
                blocks: suffix_blocks,
            }),
        );

        index.shards[shard_idx]
            .apply_event(direct_worker_event)
            .await;
        index.flush().await;

        let scores = index.shards[shard_idx]
            .backend()
            .find_matches_from_anchor(anchor, &local_hashes(&[7]))
            .unwrap();
        assert_eq!(score(&scores, worker(2)), Some(4));
    }

    #[tokio::test]
    async fn many_workers_inducing_same_anchor_divergence_all_carry_idempotent_anchor() {
        let index = Arc::new(make_indexer(2, 3));
        index.apply_event(store_event(0, &[1, 2, 3, 4])).await;

        let worker_count = 4usize;
        let barrier = Arc::new(AsyncBarrier::new(worker_count));
        let mut tasks = Vec::with_capacity(worker_count);

        for worker_id in 1..=worker_count as u64 {
            let index = index.clone();
            let barrier = barrier.clone();
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                index
                    .apply_event(store_event(worker_id, &[1, 2, 5, 6]))
                    .await;
            }));
        }

        for task in tasks {
            task.await.unwrap();
        }
        index.flush().await;

        let b = child(&child(&index.root, 1), 2);
        let e = child(&b, 5);
        assert_eq!(b.children.len(), 2);
        assert_eq!(e.shard(), 1);
        assert_eq!(e.live_workers.len(), worker_count);
        assert_eq!(index.installed_worker_anchors.len(), worker_count + 1);
        #[cfg(feature = "bench")]
        {
            assert_eq!(
                index
                    .metrics
                    .counters
                    .anchor_installs
                    .load(Ordering::Relaxed),
                worker_count as u64 + 1
            );
        }

        let scores = index
            .find_matches(local_hashes(&[1, 2, 5, 6]))
            .await
            .unwrap();
        for worker_id in 1..=worker_count as u64 {
            assert_eq!(score(&scores, worker(worker_id)), Some(4));
        }
    }

    #[tokio::test]
    async fn removing_anchor_parent_prefix_does_not_leak_suffix_scores() {
        let index = make_indexer(2, 3);
        index.apply_event(store_event(0, &[1, 2, 3, 4])).await;
        index.apply_event(store_event(1, &[1, 2, 5, 6])).await;
        index.flush().await;

        index
            .apply_event(remove_hash_event(1, 0, &[1, 2, 5, 6], 1))
            .await;
        index.flush().await;

        let full = index
            .find_matches(local_hashes(&[1, 2, 5, 6]))
            .await
            .unwrap();
        assert_eq!(score(&full, worker(0)), Some(2));
        assert_eq!(score(&full, worker(1)), Some(1));

        let drained = index.find_matches(local_hashes(&[1, 2, 9])).await.unwrap();
        assert_eq!(score(&drained, worker(0)), Some(2));
        assert_eq!(score(&drained, worker(1)), Some(1));
    }

    #[tokio::test]
    async fn read_stops_at_dead_router_node_before_walking_zombie_descendants() {
        let index = make_indexer(2, 4);
        index.apply_event(store_event(0, &[1, 2, 3, 4])).await;
        index
            .apply_event(remove_hash_event(0, 0, &[1, 2, 3, 4], 2))
            .await;
        index.flush().await;

        let scores = index
            .find_matches(local_hashes(&[1, 2, 3, 4, 5]))
            .await
            .unwrap();

        assert_eq!(score(&scores, worker(0)), Some(2));
        #[cfg(feature = "bench")]
        {
            assert_eq!(
                index
                    .metrics
                    .counters
                    .find_match_early_returns
                    .load(Ordering::Relaxed),
                1
            );
            assert_eq!(
                index
                    .metrics
                    .counters
                    .find_match_dispatches
                    .load(Ordering::Relaxed),
                0
            );
        }
    }

    #[tokio::test]
    async fn suffix_beyond_cap_remove_routes_without_clearing_prefix_liveness() {
        let index = make_indexer(2, 2);
        index.apply_event(store_event(0, &[1, 2, 3, 4, 5])).await;
        index.flush().await;

        let b = child(&child(&index.root, 1), 2);
        assert!(b.live_workers.contains(&worker(0)));

        index
            .apply_event(remove_hash_event(0, 0, &[1, 2, 3, 4, 5], 4))
            .await;
        index.flush().await;

        assert!(b.live_workers.contains(&worker(0)));
        #[cfg(feature = "bench")]
        {
            assert_eq!(
                index
                    .metrics
                    .counters
                    .remove_broadcasts
                    .load(Ordering::Relaxed),
                0
            );
        }

        let prefix = index.find_matches(local_hashes(&[1, 2])).await.unwrap();
        assert_eq!(score(&prefix, worker(0)), Some(2));

        let full = index
            .find_matches(local_hashes(&[1, 2, 3, 4, 5]))
            .await
            .unwrap();
        assert_eq!(score(&full, worker(0)), Some(4));
    }

    #[tokio::test]
    async fn parent_hash_continuation_past_depth_cap_matches_from_prefix_anchor() {
        let index = make_indexer(2, 2);
        index.apply_event(store_event(0, &[1, 2])).await;
        index
            .apply_event(store_event_with_parent(0, &[1, 2], &[3, 4, 5]))
            .await;
        index.flush().await;

        let scores = index
            .find_matches(local_hashes(&[1, 2, 3, 4, 5]))
            .await
            .unwrap();

        assert_eq!(score(&scores, worker(0)), Some(5));
    }

    #[tokio::test]
    async fn per_worker_reverse_index_removes_only_one_owner_of_shared_hash() {
        let index = make_indexer(2, 4);
        index.apply_event(store_event(0, &[7, 8, 9])).await;
        index.apply_event(store_event(1, &[7, 8, 9])).await;

        index
            .apply_event(remove_hash_event(0, 0, &[7, 8, 9], 2))
            .await;

        let full = index.find_matches(local_hashes(&[7, 8, 9])).await.unwrap();
        assert_eq!(score(&full, worker(0)), Some(2));
        assert_eq!(score(&full, worker(1)), Some(3));

        let prefix = index.find_matches(local_hashes(&[7, 8])).await.unwrap();
        assert_eq!(score(&prefix, worker(0)), Some(2));
        assert_eq!(score(&prefix, worker(1)), Some(2));
    }

    #[tokio::test]
    async fn duplicate_store_keeps_one_live_worker_entry() {
        let index = make_indexer(2, 4);
        index.apply_event(store_event(0, &[1, 2, 3])).await;
        index.apply_event(store_event(0, &[1, 2, 3])).await;

        let c = child(&child(&child(&index.root, 1), 2), 3);
        assert_eq!(c.live_workers.len(), 1);
    }

    #[tokio::test]
    async fn cleanup_updates_router_state() {
        let index = make_indexer(2, 4);
        index
            .apply_event(store_event_with_dp_rank(0, 0, &[1, 2, 3]))
            .await;
        index
            .apply_event(store_event_with_dp_rank(0, 1, &[1, 2, 4]))
            .await;

        index.remove_worker_dp_rank(0, 0).await;
        let after_dp_remove = index.find_matches(local_hashes(&[1, 2, 3])).await.unwrap();
        assert_eq!(score(&after_dp_remove, WorkerWithDpRank::new(0, 0)), None);

        index.apply_event(clear_event(0)).await;
        let after_clear = index.find_matches(local_hashes(&[1, 2])).await.unwrap();
        assert!(after_clear.scores.is_empty());
    }
}
