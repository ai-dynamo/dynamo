// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Branch-based prefix sharding with background hot-shard rebalancing.
//!
//! [`RebalancingBranchShardedIndexer`] extends the branch-sharding approach of
//! [`BranchShardedIndexer`] with a background task that periodically detects hot
//! shards and migrates the hottest branch to a cooler shard.
//!
//! ## Why a separate file?
//!
//! `BranchShardedIndexer` (in `branch_sharded.rs`) is the baseline with no
//! rebalancing overhead.  This file adds the rebalancing machinery so both can
//! be run side-by-side in benchmarks for direct comparison:
//!
//! ```text
//! mooncake_bench --compare branch-sharded-crtc,rebalancing-branch-sharded-crtc
//! ```
//!
//! ## What was implemented
//!
//! `RebalancingBranchShardedIndexer<T>` is identical to `BranchShardedIndexer<T>`
//! in every routing and write semantic.  It adds:
//!
//! 1. **Per-shard query counters** (`shard_query_counts: Vec<AtomicU64>`) and
//!    **per-branch query counters** (`branch_query_counts`) incremented on every
//!    dispatched `find_matches`.  These feed the rebalancer without adding any
//!    synchronisation to the hot path.
//!
//! 2. **Background rebalancer task** spawned via `start_rebalancer(arc, interval,
//!    threshold)`.  The task wakes every `interval`, checks whether
//!    `max_shard_hits / avg_shard_hits > threshold`, and if so migrates the
//!    hottest branch from the hottest shard to the coolest.
//!
//! 3. **Two-phase migration protocol** (see below) that eliminates the race
//!    window present in a naive "replay-then-switch" approach.
//!
//! ## Two-phase migration protocol
//!
//! ### Why two phases?
//!
//! A naive approach (dump → replay → switch routing) has a race: Stored events
//! arriving *between the dump and the routing switch* end up only on the old shard.
//! After the switch, `find_matches` goes to the new shard and misses those blocks.
//!
//! The fix is **dual-write + scatter-gather** during migration.  Events go to both
//! shards; reads query both and merge results.  But enabling dual-write naively
//! causes an ordering problem: the CRTC drops continuation blocks whose parent
//! was not yet stored (`KvCacheEventError::ParentBlockNotFound`).  If a live
//! dual-write continuation event arrives at the new shard before its root is
//! replayed, it is permanently lost.
//!
//! ### Ordering guarantee via FIFO channels
//!
//! `ThreadPoolIndexer` uses **flume** channels (strictly FIFO per worker thread).
//! Workers are sticky-assigned by `WorkerId`, so all events for worker W always
//! go through the same thread's channel.  This gives us:
//!
//! > If replay events for worker W are *enqueued* into new_shard's channel
//! > **before** dual-write is activated, flume guarantees they are *processed*
//! > before any subsequent dual-write event for W.
//!
//! The migration therefore has two phases:
//!
//! ```text
//! Phase 1 — Replaying
//!   New Stored events  →  old_shard only   (no dual-write yet)
//!   find_matches       →  old_shard only   (correct: new shard is empty)
//!
//!   Migration task:
//!     a) dump_events(old_shard)            [FIFO barrier]
//!     b) filter Stored events for branch
//!     c) enqueue replay events into new_shard's channels (via apply_event)
//!        — these are now in the FIFO queue ahead of any future dual-write events
//!
//! Phase 2 — DualWrite  [activated atomically after (c)]
//!   New Stored events  →  old_shard AND new_shard
//!   find_matches       →  scatter-gather both shards, merge OverlapScores
//!
//!   Migration task:
//!     d) flush(new_shard)                  [wait for replay + early dual-writes]
//!     e) sleep(dual_write_window)          [let gap-period traffic re-fill new_shard]
//!     f) update branch_to_shard → new_shard
//!     g) remove from dualwrite_branches    [DualWrite ends, single-shard resumes]
//! ```
//!
//! **Gap events** (arriving between the dump and DualWrite activation) are on
//! old_shard only.  During the DualWrite window, those sequences generate new
//! traffic that is dual-written, so new_shard warms up naturally.  After the
//! window, new_shard handles the branch alone.
//!
//! ### OverlapScores merge
//!
//! During DualWrite, `find_matches` queries both shards concurrently and merges
//! results: for each `WorkerWithDpRank`, keep the **higher** overlap score and
//! the larger tree size.  Frequency histograms are element-wise maxed.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;
use dashmap::{DashMap, DashSet};
use rustc_hash::FxBuildHasher;
use tokio::sync::oneshot;

use super::{KvIndexerInterface, KvRouterError, ShardSizeSnapshot, SyncIndexer, ThreadPoolIndexer};
use crate::protocols::*;

// ---------------------------------------------------------------------------
// Per-shard read thread pool (same as branch_sharded.rs)
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
// Migration state
// ---------------------------------------------------------------------------

/// Per-branch migration entry stored in `dualwrite_branches`.
///
/// A branch enters `dualwrite_branches` when Phase 1 (replay event enqueuing)
/// is complete and Phase 2 (dual-write + scatter-gather) begins.  It is removed
/// when migration finalises and `branch_to_shard` is switched to `new_shard`.
#[derive(Clone, Copy)]
struct DualWriteEntry {
    old_shard: usize,
    new_shard: usize,
}

// ---------------------------------------------------------------------------
// RebalancingBranchShardedIndexer
// ---------------------------------------------------------------------------

/// Branch-sharded indexer with background hot-shard rebalancing.
///
/// Identical to [`BranchShardedIndexer`] for routing and write semantics during
/// normal (non-migration) operation.  Adds per-shard/per-branch query counters
/// and a two-phase migration protocol described in the module doc.
pub struct RebalancingBranchShardedIndexer<T: SyncIndexer> {
    shards: Vec<Arc<ThreadPoolIndexer<T>>>,
    num_shards: usize,

    /// Number of leading blocks used to identify a branch.  Default: 2.
    prefix_depth: usize,

    /// Routing table: FNV-1a(first `prefix_depth` block hashes) → shard index.
    branch_to_shard: DashMap<u64, usize, FxBuildHasher>,

    /// Number of branches assigned to each shard.
    branch_counts: Mutex<Vec<usize>>,

    /// Remove index: `ExternalSequenceBlockHash.0` → `(shard_index, block_depth)`.
    ///
    /// `block_depth` is the 0-based position of the block in its sequence,
    /// used to determine whether a continuation event is past `prefix_depth`.
    block_to_shard: DashMap<u64, (usize, u32), FxBuildHasher>,

    kv_block_size: u32,

    read_pools: Option<Vec<ShardReadPool>>,

    // --- query-rate tracking ---

    /// Cumulative `find_matches` dispatched hits per shard.
    shard_query_counts: Vec<AtomicU64>,

    /// Per-branch cumulative `find_matches` hits: branch_key → count.
    branch_query_counts: DashMap<u64, AtomicU64, FxBuildHasher>,

    // --- migration state ---

    /// Branches currently in Phase 1 (Replaying): replay is being enqueued,
    /// new Stored events still go to old_shard only.
    replaying_branches: DashSet<u64, FxBuildHasher>,

    /// Branches currently in Phase 2 (DualWrite): new Stored events go to both
    /// shards; `find_matches` scatter-gathers and merges.
    dualwrite_branches: DashMap<u64, DualWriteEntry, FxBuildHasher>,

    // --- observability (same as BranchShardedIndexer) ---
    timing_calls: AtomicU64,
    timing_sum_routing_ns: AtomicU64,
    timing_sum_shard_ns: AtomicU64,
    find_matches_miss_count: AtomicU64,
    remove_broadcast_count: AtomicU64,

    /// AbortHandle for the background rebalancer task.  Aborted in `shutdown()`.
    rebalancer_handle: Mutex<Option<tokio::task::AbortHandle>>,
}

impl<T: SyncIndexer> RebalancingBranchShardedIndexer<T> {
    pub fn new(
        shards: Vec<ThreadPoolIndexer<T>>,
        prefix_depth: usize,
        kv_block_size: u32,
        num_read_threads_per_shard: usize,
    ) -> Self {
        Self::new_with_options(shards, prefix_depth, kv_block_size, num_read_threads_per_shard)
    }

    pub fn new_with_options(
        shards: Vec<ThreadPoolIndexer<T>>,
        prefix_depth: usize,
        kv_block_size: u32,
        num_read_threads_per_shard: usize,
    ) -> Self {
        assert!(!shards.is_empty(), "Must provide at least one shard");
        let num_shards = shards.len();

        let shards: Vec<Arc<ThreadPoolIndexer<T>>> =
            shards.into_iter().map(Arc::new).collect();

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

        let shard_query_counts = (0..num_shards).map(|_| AtomicU64::new(0)).collect();

        Self {
            shards,
            num_shards,
            prefix_depth: prefix_depth.max(1),
            branch_to_shard: DashMap::with_hasher(FxBuildHasher),
            branch_counts: Mutex::new(vec![0usize; num_shards]),
            block_to_shard: DashMap::with_hasher(FxBuildHasher),
            kv_block_size,
            read_pools,
            shard_query_counts,
            branch_query_counts: DashMap::with_hasher(FxBuildHasher),
            replaying_branches: DashSet::with_hasher(FxBuildHasher),
            dualwrite_branches: DashMap::with_hasher(FxBuildHasher),
            timing_calls: AtomicU64::new(0),
            timing_sum_routing_ns: AtomicU64::new(0),
            timing_sum_shard_ns: AtomicU64::new(0),
            find_matches_miss_count: AtomicU64::new(0),
            remove_broadcast_count: AtomicU64::new(0),
            rebalancer_handle: Mutex::new(None),
        }
    }

    // --- branch key computation ---

    fn branch_key_for_local_hashes(&self, hashes: &[LocalBlockHash]) -> u64 {
        let k = self.prefix_depth.min(hashes.len());
        let mut h: u64 = 14695981039346656037;
        for block in &hashes[..k] {
            for b in block.0.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        h
    }

    fn branch_key_for_stored_blocks(&self, blocks: &[KvCacheStoredBlockData]) -> u64 {
        let k = self.prefix_depth.min(blocks.len());
        let mut h: u64 = 14695981039346656037;
        for block in &blocks[..k] {
            for b in block.tokens_hash.0.to_le_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        h
    }

    // --- routing table ---

    fn lookup_shard(&self, branch_key: u64) -> Option<usize> {
        self.branch_to_shard.get(&branch_key).map(|v| *v)
    }

    fn assign_shard(&self, branch_key: u64) -> usize {
        if let Some(shard_idx) = self.branch_to_shard.get(&branch_key).map(|v| *v) {
            return shard_idx;
        }
        let mut counts = self.branch_counts.lock().unwrap();
        if let Some(shard_idx) = self.branch_to_shard.get(&branch_key).map(|v| *v) {
            return shard_idx;
        }
        let selected = counts
            .iter()
            .enumerate()
            .min_by_key(|&(_, c)| c)
            .unwrap()
            .0;
        counts[selected] += 1;
        drop(counts);
        self.branch_to_shard.insert(branch_key, selected);
        selected
    }

    // --- per-shard query dispatch (shared between normal and scatter-gather paths) ---

    /// Dispatch `find_matches` to a single shard, using the read pool if configured.
    async fn dispatch_find_matches(
        &self,
        shard_idx: usize,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        if let Some(pools) = &self.read_pools {
            let (resp_tx, resp_rx) = oneshot::channel();
            pools[shard_idx]
                .sender
                .send((sequence, resp_tx))
                .map_err(|_| KvRouterError::IndexerOffline)?;
            resp_rx.await.map_err(|_| KvRouterError::IndexerOffline)
        } else {
            self.shards[shard_idx].find_matches(sequence).await
        }
    }

    /// Merge two `OverlapScores` from different shards.
    ///
    /// For each `WorkerWithDpRank` present in either shard:
    /// - `scores`: take the higher overlap count (more prefix cached = better).
    /// - `tree_sizes`: take the larger size.
    /// - `frequencies`: element-wise max (both histograms track the same query).
    fn merge_overlap_scores(mut a: OverlapScores, b: OverlapScores) -> OverlapScores {
        for (worker, score) in b.scores {
            let entry = a.scores.entry(worker).or_insert(0);
            *entry = (*entry).max(score);
        }
        for (worker, size) in b.tree_sizes {
            let entry = a.tree_sizes.entry(worker).or_insert(0);
            *entry = (*entry).max(size);
        }
        // frequencies is positional; extend shorter vec with 0 then element-wise max.
        if b.frequencies.len() > a.frequencies.len() {
            a.frequencies.resize(b.frequencies.len(), 0);
        }
        for (i, &freq) in b.frequencies.iter().enumerate() {
            a.frequencies[i] = a.frequencies[i].max(freq);
        }
        a
    }

    // --- rebalancing ---

    /// Check shard query-rate balance and trigger migration if skewed.
    ///
    /// Called periodically by the task from [`start_rebalancer`][Self::start_rebalancer].
    /// Safe to call manually in tests.
    pub async fn rebalance_once(
        &self,
        imbalance_threshold: f64,
        dual_write_window: std::time::Duration,
    ) {
        if self.num_shards < 2 {
            return;
        }

        let counts: Vec<u64> = self
            .shard_query_counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .collect();
        let total: u64 = counts.iter().sum();
        if total == 0 {
            return;
        }
        let avg = total as f64 / self.num_shards as f64;
        let (hot_shard, &max_count) = counts
            .iter()
            .enumerate()
            .max_by_key(|&(_, c)| c)
            .unwrap();

        if max_count as f64 / avg < imbalance_threshold {
            return;
        }

        let cool_shard = counts
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != hot_shard)
            .min_by_key(|&(_, c)| c)
            .unwrap()
            .0;

        // Pick the highest-traffic branch on hot_shard that is not already migrating.
        let hot_branch = self
            .branch_query_counts
            .iter()
            .filter_map(|entry| {
                let key = *entry.key();
                let hits = entry.value().load(Ordering::Relaxed);
                let on_hot = self.branch_to_shard.get(&key).map(|v| *v) == Some(hot_shard);
                let idle = !self.replaying_branches.contains(&key)
                    && !self.dualwrite_branches.contains_key(&key);
                if on_hot && idle { Some((key, hits)) } else { None }
            })
            .max_by_key(|&(_, hits)| hits);

        if let Some((branch_key, branch_hits)) = hot_branch {
            tracing::info!(
                "rebalancer: migrating branch {branch_key:#018x} \
                 shard {hot_shard} → {cool_shard} \
                 (imbalance {:.2}×, branch hits = {branch_hits})",
                max_count as f64 / avg,
            );
            self.migrate_branch(branch_key, hot_shard, cool_shard, dual_write_window)
                .await;
        }
    }

    /// Two-phase migration of `branch_key` from `old_shard` to `new_shard`.
    ///
    /// See the module-level doc for a full description of the protocol.
    async fn migrate_branch(
        &self,
        branch_key: u64,
        old_shard: usize,
        new_shard: usize,
        dual_write_window: std::time::Duration,
    ) {
        // ── Phase 1: Replaying ──────────────────────────────────────────────
        // Mark as replaying so that concurrent apply_event calls do NOT
        // dual-write yet (which would race with the replay).
        if !self.replaying_branches.insert(branch_key) {
            return; // already in progress
        }

        // Dump all events from the old shard via FIFO barrier.
        // All in-flight writes drain before the snapshot is taken.
        let all_events = match self.shards[old_shard].dump_events().await {
            Ok(evts) => evts,
            Err(e) => {
                tracing::error!("rebalancer: dump_events failed on shard {old_shard}: {e:?}");
                self.replaying_branches.remove(&branch_key);
                return;
            }
        };

        // Filter to Stored events for this branch only.
        // Removed/Cleared are not replayed: old_shard handles them via
        // block_to_shard routing until those blocks evict naturally.
        let branch_events: Vec<RouterEvent> = all_events
            .into_iter()
            .filter(|e| {
                if let KvCacheEventData::Stored(d) = &e.event.data {
                    self.branch_key_for_stored_blocks(&d.blocks) == branch_key
                } else {
                    false
                }
            })
            .collect();

        tracing::debug!(
            "rebalancer: enqueuing {} Stored events for branch {branch_key:#018x} \
             into shard {new_shard} channels",
            branch_events.len()
        );

        // Enqueue replay events directly into new_shard's ThreadPoolIndexer.
        // These calls are non-blocking (send into flume channel and return).
        // Because flume is FIFO per worker-thread channel, these replay events
        // are guaranteed to be processed BEFORE any subsequent dual-write events
        // for the same worker, which we enable atomically in the next step.
        for event in branch_events {
            self.shards[new_shard].apply_event(event).await;
        }

        // ── Phase 1 → Phase 2 transition ────────────────────────────────────
        // Insert into dualwrite_branches BEFORE removing from replaying_branches.
        // After this insert, apply_event will dual-write to new_shard as well.
        // The brief overlap (entry in both maps) is harmless: replaying_branches
        // is checked first in apply_event, so dual-write won't fire until we
        // remove from replaying_branches below.
        self.dualwrite_branches.insert(
            branch_key,
            DualWriteEntry { old_shard, new_shard },
        );
        self.replaying_branches.remove(&branch_key);

        // ── Phase 2: DualWrite ───────────────────────────────────────────────
        // Flush new_shard: wait until all replayed events (and any early
        // dual-write events that snuck in) are fully processed by the CRTC.
        <ThreadPoolIndexer<T> as KvIndexerInterface>::flush(&*self.shards[new_shard]).await;

        tracing::debug!(
            "rebalancer: new shard {new_shard} flushed, \
             dual-write window = {:?}",
            dual_write_window
        );

        // Hold dual-write for the configured window.  During this time, new
        // events for the branch go to both shards and find_matches scatter-gathers.
        // Gap-period sequences (events between the dump and Phase 2 start) will
        // have generated new traffic that is dual-written, warming up new_shard.
        tokio::time::sleep(dual_write_window).await;

        // ── Finalise migration ───────────────────────────────────────────────
        // Switch the routing table.  All subsequent apply_event and find_matches
        // calls route to new_shard only.
        self.branch_to_shard.insert(branch_key, new_shard);
        {
            let mut counts = self.branch_counts.lock().unwrap();
            counts[old_shard] = counts[old_shard].saturating_sub(1);
            counts[new_shard] += 1;
        }

        // Transfer the branch's accumulated query load to new_shard counters
        // so the rebalancer doesn't immediately re-trigger.
        let branch_hits = self
            .branch_query_counts
            .get(&branch_key)
            .map(|v| v.load(Ordering::Relaxed))
            .unwrap_or(0);
        let old_count = self.shard_query_counts[old_shard].load(Ordering::Relaxed);
        self.shard_query_counts[old_shard]
            .fetch_sub(branch_hits.min(old_count), Ordering::Relaxed);
        self.shard_query_counts[new_shard].fetch_add(branch_hits, Ordering::Relaxed);
        self.branch_query_counts.remove(&branch_key);

        // Remove from dualwrite_branches last — this atomically ends Phase 2.
        self.dualwrite_branches.remove(&branch_key);

        tracing::info!(
            "rebalancer: migration of branch {branch_key:#018x} complete \
             ({old_shard} → {new_shard})"
        );
    }

    /// Spawn a background tokio task that periodically calls
    /// [`rebalance_once`][Self::rebalance_once].
    ///
    /// The task holds a `Weak<Self>` and stops automatically when the indexer
    /// is dropped.  `shutdown()` aborts it immediately.
    ///
    /// # Arguments
    ///
    /// * `interval` — how often to check for imbalance.
    /// * `imbalance_threshold` — `max / avg` ratio that triggers migration (e.g. `1.5`).
    /// * `dual_write_window` — how long to keep dual-write active after replay is
    ///   complete, giving gap-period traffic time to warm up the new shard.
    pub fn start_rebalancer(
        self: Arc<Self>,
        interval: std::time::Duration,
        imbalance_threshold: f64,
        dual_write_window: std::time::Duration,
    ) -> tokio::task::JoinHandle<()> {
        let weak = Arc::downgrade(&self);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                ticker.tick().await;
                match weak.upgrade() {
                    Some(idx) => idx.rebalance_once(imbalance_threshold, dual_write_window).await,
                    None => break,
                }
            }
        });
        *self.rebalancer_handle.lock().unwrap() = Some(handle.abort_handle());
        handle
    }
}

#[async_trait]
impl<T: SyncIndexer> KvIndexerInterface for RebalancingBranchShardedIndexer<T> {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        let t_routing = std::time::Instant::now();
        let branch_key = self.branch_key_for_local_hashes(&sequence);

        // Check if branch is in DualWrite phase → scatter-gather.
        if let Some(entry) = self.dualwrite_branches.get(&branch_key) {
            let DualWriteEntry { old_shard, new_shard } = *entry;
            drop(entry); // release DashMap guard before await
            let routing_ns = t_routing.elapsed().as_nanos() as u64;

            let (r_old, r_new) = tokio::join!(
                self.dispatch_find_matches(old_shard, sequence.clone()),
                self.dispatch_find_matches(new_shard, sequence),
            );
            let merged = Self::merge_overlap_scores(r_old?, r_new?);

            self.timing_calls.fetch_add(1, Ordering::Relaxed);
            self.timing_sum_routing_ns.fetch_add(routing_ns, Ordering::Relaxed);
            // shard timing not tracked for scatter-gather (two concurrent calls)
            return Ok(merged);
        }

        // Normal path: single-shard dispatch.
        let shard_idx = match self.lookup_shard(branch_key) {
            Some(idx) => idx,
            None => {
                self.find_matches_miss_count.fetch_add(1, Ordering::Relaxed);
                return Ok(OverlapScores::new());
            }
        };
        let routing_ns = t_routing.elapsed().as_nanos() as u64;

        self.shard_query_counts[shard_idx].fetch_add(1, Ordering::Relaxed);
        self.branch_query_counts
            .entry(branch_key)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);

        let t_shard = std::time::Instant::now();
        let result = self.dispatch_find_matches(shard_idx, sequence).await;
        let shard_ns = t_shard.elapsed().as_nanos() as u64;

        self.timing_calls.fetch_add(1, Ordering::Relaxed);
        self.timing_sum_routing_ns.fetch_add(routing_ns, Ordering::Relaxed);
        self.timing_sum_shard_ns.fetch_add(shard_ns, Ordering::Relaxed);

        result
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None, lora_name);
        let branch_key = self.branch_key_for_local_hashes(&sequence);

        if let Some(entry) = self.dualwrite_branches.get(&branch_key) {
            let DualWriteEntry { old_shard, new_shard } = *entry;
            drop(entry);
            let (r_old, r_new) = tokio::join!(
                self.shards[old_shard].find_matches(sequence.clone()),
                self.shards[new_shard].find_matches(sequence),
            );
            return Ok(Self::merge_overlap_scores(r_old?, r_new?));
        }

        match self.lookup_shard(branch_key) {
            Some(idx) => self.shards[idx].find_matches(sequence).await,
            None => Ok(OverlapScores::new()),
        }
    }

    async fn apply_event(&self, event: RouterEvent) {
        match &event.event.data {
            KvCacheEventData::Stored(store_data) => {
                // Depth-aware shard routing (mirrors BranchShardedIndexer):
                // inherit parent's shard only when past prefix_depth.
                let (shard_idx, first_block_depth) =
                    if let Some(parent_hash) = &store_data.parent_hash {
                        match self.block_to_shard.get(&parent_hash.0).map(|v| *v) {
                            Some((parent_shard, parent_depth)) => {
                                let new_depth = parent_depth + 1;
                                if new_depth >= self.prefix_depth as u32 {
                                    (parent_shard, new_depth)
                                } else {
                                    let key =
                                        self.branch_key_for_stored_blocks(&store_data.blocks);
                                    (self.assign_shard(key), new_depth)
                                }
                            }
                            None => {
                                let key = self.branch_key_for_stored_blocks(&store_data.blocks);
                                (self.assign_shard(key), 0)
                            }
                        }
                    } else {
                        let key = self.branch_key_for_stored_blocks(&store_data.blocks);
                        (self.assign_shard(key), 0)
                    };

                // Record block → (shard, depth) before dispatching.
                for (i, block) in store_data.blocks.iter().enumerate() {
                    self.block_to_shard
                        .insert(block.block_hash.0, (shard_idx, first_block_depth + i as u32));
                }

                // Check DualWrite: if active for this branch, also send to new_shard.
                // We check replaying_branches first: if Phase 1 is still in progress,
                // dual-write must NOT fire yet (replay events must arrive first in
                // new_shard's channels to preserve FIFO ordering).
                let branch_key = self.branch_key_for_stored_blocks(&store_data.blocks);
                let dual_write_target: Option<usize> =
                    if !self.replaying_branches.contains(&branch_key) {
                        self.dualwrite_branches
                            .get(&branch_key)
                            .map(|e| e.new_shard)
                    } else {
                        None
                    };

                if let Some(new_shard) = dual_write_target {
                    // Send to new_shard first so it processes events in the
                    // same logical order as old_shard where possible.
                    self.shards[new_shard].apply_event(event.clone()).await;
                }
                self.shards[shard_idx].apply_event(event).await;
            }

            KvCacheEventData::Removed(remove_data) => {
                let mut shard_blocks: Vec<Vec<ExternalSequenceBlockHash>> =
                    vec![Vec::new(); self.num_shards];
                let mut broadcast_blocks: Vec<ExternalSequenceBlockHash> = Vec::new();

                for &block_hash in &remove_data.block_hashes {
                    match self.block_to_shard.remove(&block_hash.0) {
                        Some((_, (shard_idx, _depth))) => shard_blocks[shard_idx].push(block_hash),
                        None => {
                            self.remove_broadcast_count.fetch_add(1, Ordering::Relaxed);
                            broadcast_blocks.push(block_hash);
                        }
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

            KvCacheEventData::Cleared => {
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
        if let Some(handle) = self.rebalancer_handle.lock().unwrap().take() {
            handle.abort();
        }
        for shard in &self.shards {
            shard.shutdown();
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let mut all_events = Vec::new();
        for shard in &self.shards {
            all_events.extend(shard.dump_events().await?);
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

    fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(idx, shard)| {
                let node_count = shard.node_edge_lengths().len();
                shard.shard_sizes().into_iter().map(move |mut s| {
                    s.shard_idx = idx;
                    s.node_count = node_count;
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
        let total_calls = dispatched + misses;
        let broadcasts = self.remove_broadcast_count.load(Ordering::Relaxed);
        if total_calls == 0 {
            return String::new();
        }
        let miss_pct = 100.0 * misses as f64 / total_calls as f64;
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
        let total_branches: usize = self.branch_counts.lock().unwrap().iter().sum();
        let branch_dist: Vec<String> = self
            .branch_counts
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, c)| format!("shard[{i}]={c}"))
            .collect();
        let query_dist: Vec<String> = self
            .shard_query_counts
            .iter()
            .enumerate()
            .map(|(i, c)| format!("shard[{i}]={}", c.load(Ordering::Relaxed)))
            .collect();
        let replaying = self.replaying_branches.len();
        let dualwriting = self.dualwrite_branches.len();
        format!(
            "RebalancingBranchShardedIndexer find_matches \
             ({total_calls} total: {dispatched} dispatched, \
             {misses} early-exit / {miss_pct:.1}% miss):\n  \
             avg routing       = {avg_routing_ns}ns  (routing table lookup)\n  \
             avg shard         = {avg_shard_us}µs  (CRTC traversal, {mode})\n  \
             branches known    = {total_branches}  ({})\n  \
             query hits/shard  = {}  (cumulative, drives rebalancing)\n  \
             migrating         = {replaying} replaying, {dualwriting} dual-writing\n  \
             remove broadcasts = {broadcasts}  (fallback for blocks absent from index)",
            branch_dist.join(", "),
            query_dist.join(", "),
        )
    }
}
