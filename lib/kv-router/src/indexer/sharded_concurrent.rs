// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-based sharding over `ThreadPoolIndexer<T>`.
//!
//! [`ShardedConcurrentIndexer`] partitions workers across N independent
//! [`ThreadPoolIndexer`] instances (shards).  Each shard holds a disjoint
//! subset of workers, so per-shard worker sets are 1/N the size of the full
//! set, reducing the cost of `find_matches` traversal steps (set clones,
//! retains, contains checks) and write contention on per-node locks.
//!
//! ## Sharding strategy
//!
//! - **Events**: routed to the worker's assigned shard (sticky).  New workers
//!   are assigned to the shard with the fewest workers (least-loaded).
//! - **find_matches**: scattered concurrently to all N shards via
//!   `tokio::spawn`; results are merged (union of disjoint worker sets).
//! - **remove_worker / remove_worker_dp_rank**: routed to the assigned shard.
//!
//! ## Expected performance characteristics
//!
//! Per-shard `find_matches` is cheaper because worker sets shrink by factor N.
//! Lock contention on shared nodes is also reduced (fewer writers per shard).
//! The scatter-gather adds some overhead, but at ≥4 shards and ≥5k workers the
//! net benefit should dominate.  See the sharding design doc for the benchmark
//! matrix.
//!
//! ## Read isolation (per-shard read pools) — experimental
//!
//! By default `find_matches` uses `tokio::task::spawn_blocking` to scatter read
//! requests to all shards concurrently.  The dominant overhead (78-90% of outer
//! wall-clock time) is not the spawn mechanism itself — it is the **scatter-gather
//! architecture**: every shard must be queried for every `find_matches` call, so
//! total query volume = N × offered_rate.  No dispatch mechanism eliminates this;
//! see [`PrefixShardedIndexer`] for a single-shard-per-query design.
//!
//! When `num_read_threads_per_shard > 0` is passed to [`ShardedConcurrentIndexer::new`],
//! each shard gets a **dedicated bounded OS thread pool** of that size.
//! `find_matches` sends all N shard requests over channels simultaneously and
//! awaits N `oneshot` replies.  In benchmarks this mode performed **worse** than
//! `spawn_blocking` at all tested concurrency levels: sleeping OS threads blocked
//! on `flume::recv()` incur ~200-400 µs futex wake-up latency per shard, while
//! tokio's `spawn_blocking` reuses warm threads with lower wake latency.  This
//! option is retained for experimentation but `0` (spawn_blocking) remains the
//! recommended default.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use async_trait::async_trait;
use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use tokio::sync::oneshot;

use super::{KvIndexerInterface, KvRouterError, ShardSizeSnapshot, SyncIndexer, ThreadPoolIndexer};
use crate::protocols::*;

// ---------------------------------------------------------------------------
// Per-shard read thread pool
// ---------------------------------------------------------------------------

/// `(sequence, response_sender)` — response carries the scores *and* the
/// wall-clock time the CRTC traversal took (nanoseconds), so the caller can
/// compute `max(shard_ns)` for the timing report.
type ReadRequest = (Vec<LocalBlockHash>, oneshot::Sender<(OverlapScores, u64)>);

/// A bounded pool of OS threads dedicated to `find_matches` requests for one
/// shard.  Mirrors the equivalent struct in `prefix_sharded.rs` and
/// `branch_sharded.rs`, but returns timing alongside the scores so that
/// `ShardedConcurrentIndexer` can track the critical-path shard cost.
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
                    let t = std::time::Instant::now();
                    let result = backend.find_matches(&seq, false);
                    let _ = resp_tx.send((result, t.elapsed().as_nanos() as u64));
                }
            }));
        }
        Self {
            sender: tx,
            _threads: threads,
        }
    }
}

/// Merge `src` scores into `dst` in-place.
///
/// Workers are shard-disjoint so `scores` and `tree_sizes` can be extended
/// without key conflicts.  `frequencies` (prefix-depth histogram) are
/// element-wise summed.
fn merge_scores(dst: &mut OverlapScores, src: OverlapScores) {
    dst.scores.extend(src.scores);
    dst.tree_sizes.extend(src.tree_sizes);
    let diff = (src.frequencies.len() as i64) - (dst.frequencies.len() as i64);
    if diff > 0 {
        dst.frequencies
            .extend(std::iter::repeat_n(0usize, diff as usize));
    }
    for i in 0..src.frequencies.len() {
        dst.frequencies[i] += src.frequencies[i];
    }
}

/// Worker-sharded wrapper over N [`ThreadPoolIndexer<T>`] instances.
///
/// Construct with [`ShardedConcurrentIndexer::new`]; for CRTC specifically,
/// use [`ShardedConcurrentIndexer::new_crtc`].
pub struct ShardedConcurrentIndexer<T: SyncIndexer> {
    shards: Vec<Arc<ThreadPoolIndexer<T>>>,
    worker_assignments: DashMap<WorkerId, usize, FxBuildHasher>,
    worker_counts: Arc<Mutex<Vec<usize>>>,
    kv_block_size: u32,
    /// Per-shard dedicated read thread pools.
    ///
    /// `None`  → reads scatter via `spawn_blocking` (original behavior).
    /// `Some`  → reads dispatch to isolated per-shard OS thread pools, sending
    ///           all N requests before awaiting any reply so shards run in
    ///           parallel without touching the async executor.
    read_pools: Option<Vec<ShardReadPool>>,
    /// When `true`, `find_matches` iterates shards sequentially on the caller's
    /// async thread instead of scattering via `spawn_blocking`.  Eliminates all
    /// task-scheduling overhead at the cost of shard parallelism: total traversal
    /// time = sum(shard times) rather than max(shard times).  For small N (2-4)
    /// this is a net win because the `spawn_blocking` scheduling roundtrip
    /// (~2-3ms under load) exceeds the parallelism gain (~half a shard time).
    inline_sequential: bool,
    /// Number of completed `find_matches` calls (for timing averages).
    timing_calls: AtomicU64,
    /// Sum of total `find_matches` wall-clock time (ns) across all calls.
    timing_sum_outer_ns: AtomicU64,
    /// Sum of `max(per-shard CRTC time)` (ns) across all calls.
    ///
    /// Because shards execute concurrently, the CRTC work on the critical path
    /// equals the slowest shard, not the sum.  `timing_sum_outer_ns -
    /// timing_sum_max_shard_ns` is the cumulative scatter/gather overhead.
    timing_sum_max_shard_ns: AtomicU64,
}

impl<T: SyncIndexer> ShardedConcurrentIndexer<T> {
    /// Create a sharded indexer from a pre-built list of [`ThreadPoolIndexer`] instances.
    ///
    /// # Arguments
    ///
    /// * `shards` - One `ThreadPoolIndexer` per shard.
    /// * `kv_block_size` - Block size for KV cache.
    /// * `num_read_threads_per_shard` - Number of OS threads in each shard's
    ///   dedicated read pool.  Pass `0` to use the original `spawn_blocking`
    ///   scatter mode.  Pass `> 0` to pre-spin dedicated threads per shard,
    ///   eliminating `spawn_blocking` overhead at the cost of some idle threads.
    /// * `inline_sequential` - When `true`, shards are queried one after another
    ///   on the caller's async thread with no task spawning.  Takes priority over
    ///   `num_read_threads_per_shard`.
    ///
    /// # Panics
    ///
    /// Panics if `shards` is empty.
    pub fn new(
        shards: Vec<ThreadPoolIndexer<T>>,
        kv_block_size: u32,
        num_read_threads_per_shard: usize,
        inline_sequential: bool,
    ) -> Self {
        assert!(!shards.is_empty(), "Must provide at least one shard");
        let num_shards = shards.len();
        let shards: Vec<Arc<ThreadPoolIndexer<T>>> =
            shards.into_iter().map(Arc::new).collect();

        let read_pools = if !inline_sequential && num_read_threads_per_shard > 0 {
            Some(
                shards
                    .iter()
                    .map(|shard| ShardReadPool::new(shard.backend_arc(), num_read_threads_per_shard))
                    .collect(),
            )
        } else {
            None
        };

        Self {
            shards,
            worker_assignments: DashMap::with_hasher(FxBuildHasher),
            worker_counts: Arc::new(Mutex::new(vec![0usize; num_shards])),
            kv_block_size,
            read_pools,
            inline_sequential,
            timing_calls: AtomicU64::new(0),
            timing_sum_outer_ns: AtomicU64::new(0),
            timing_sum_max_shard_ns: AtomicU64::new(0),
        }
    }

    /// Get or assign a shard for the given worker, using least-loaded balancing.
    fn assign_shard(&self, worker_id: WorkerId) -> usize {
        *self.worker_assignments.entry(worker_id).or_insert_with(|| {
            let mut counts = self.worker_counts.lock().unwrap();
            let selected = counts
                .iter()
                .enumerate()
                .min_by_key(|&(_, count)| count)
                .unwrap()
                .0;
            counts[selected] += 1;
            selected
        })
    }
}

#[async_trait]
impl<T: SyncIndexer> KvIndexerInterface for ShardedConcurrentIndexer<T> {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        // Record wall-clock start for the full call (scatter + shard work + gather).
        let outer_start = std::time::Instant::now();

        // Two scatter modes depending on whether per-shard read pools are active.
        //
        // ── Pool mode (num_read_threads_per_shard > 0) ──────────────────────
        //
        // All N shard requests are sent to pre-spun OS threads before any
        // response is awaited, so every shard starts its CRTC traversal in
        // parallel without touching the tokio executor.  The call then awaits
        // the N `oneshot` receivers in sequence — by the time the first await
        // completes, the slower shards are typically already done or very close,
        // so the sequential await adds near-zero extra latency.
        //
        // This eliminates the `spawn_blocking` overhead that dominated (77% of
        // outer time) when the blocking thread pool was saturated under load.
        //
        // ── spawn_blocking mode (num_read_threads_per_shard == 0) ──────────
        //
        // Original behavior.  Each shard's CRTC traversal is submitted as a
        // `spawn_blocking` task.  Under high caller concurrency the blocking
        // thread pool saturates and queuing overhead dominates; use pool mode
        // to fix this.
        let (scores, max_shard_ns) = if self.inline_sequential {
            // --- Sequential inline path ---
            //
            // Each shard's CRTC traversal runs directly on the caller's async
            // thread, one after another.  No task spawning, no channel dispatch,
            // no wakeup latency.  Total traversal time = sum(shard times) instead
            // of max(shard times), but at small N the spawn_blocking scheduling
            // roundtrip (~2-3ms under load) exceeds the parallelism gain, so the
            // net result is faster end-to-end latency.
            let mut scores = OverlapScores::new();
            let mut max_shard_ns: u64 = 0;
            for shard in &self.shards {
                let backend = shard.backend_arc();
                let shard_start = std::time::Instant::now();
                let shard_scores = backend.find_matches(&sequence, false);
                let shard_ns = shard_start.elapsed().as_nanos() as u64;
                if shard_ns > max_shard_ns {
                    max_shard_ns = shard_ns;
                }
                merge_scores(&mut scores, shard_scores);
            }
            (scores, max_shard_ns)
        } else if let Some(pools) = &self.read_pools {
            // --- Pool path ---
            // Send all requests before awaiting any, so shards start in parallel.
            let mut receivers = Vec::with_capacity(pools.len());
            for pool in pools {
                let (resp_tx, resp_rx) = oneshot::channel();
                pool.sender
                    .send((sequence.clone(), resp_tx))
                    .map_err(|_| KvRouterError::IndexerOffline)?;
                receivers.push(resp_rx);
            }

            let mut scores = OverlapScores::new();
            let mut max_shard_ns: u64 = 0;
            for resp_rx in receivers {
                let (shard_scores, shard_ns) =
                    resp_rx.await.map_err(|_| KvRouterError::IndexerOffline)?;
                if shard_ns > max_shard_ns {
                    max_shard_ns = shard_ns;
                }
                merge_scores(&mut scores, shard_scores);
            }
            (scores, max_shard_ns)
        } else {
            // --- spawn_blocking path (original) ---
            //
            // Use `spawn_blocking` rather than `tokio::spawn` because each shard's
            // `find_matches` is synchronous (it calls the backing CRTC directly and
            // returns immediately without yielding).  Spawning it as a plain task
            // would block a tokio *async* runtime thread for the full traversal
            // duration.  Under high caller concurrency (many trace workers or a
            // tight-loop stressor), this causes a task-spawn feedback loop:
            //   - each caller enqueues 2 blocking tasks on the async thread pool
            //   - the pool (bounded by CPU core count) saturates
            //   - tasks queue up, latency climbs, more calls become in-flight,
            //     which enqueues more tasks — a runaway cycle
            //
            // `spawn_blocking` uses a separate, lazily-grown blocking thread pool
            // that is not subject to the async pool's core-count bound, so blocking
            // shard queries never starve async work or compound with each other.
            let mut handles = Vec::with_capacity(self.shards.len());
            for shard in &self.shards {
                let backend: Arc<T> = shard.backend_arc();
                let seq = sequence.clone();
                handles.push(tokio::task::spawn_blocking(move || {
                    let shard_start = std::time::Instant::now();
                    let result = backend.find_matches(&seq, false);
                    (result, shard_start.elapsed().as_nanos() as u64)
                }));
            }

            let mut scores = OverlapScores::new();
            let mut max_shard_ns: u64 = 0;
            for handle in handles {
                let (shard_scores, shard_ns): (OverlapScores, u64) =
                    handle.await.map_err(|_| KvRouterError::IndexerOffline)?;
                if shard_ns > max_shard_ns {
                    max_shard_ns = shard_ns;
                }
                merge_scores(&mut scores, shard_scores);
            }
            (scores, max_shard_ns)
        };

        // Accumulate timing stats (Relaxed: these are counters, not guards).
        let outer_ns = outer_start.elapsed().as_nanos() as u64;
        self.timing_calls.fetch_add(1, Ordering::Relaxed);
        self.timing_sum_outer_ns.fetch_add(outer_ns, Ordering::Relaxed);
        self.timing_sum_max_shard_ns.fetch_add(max_shard_ns, Ordering::Relaxed);

        Ok(scores)
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None, lora_name);
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        let shard_idx = self.assign_shard(event.worker_id);
        self.shards[shard_idx].apply_event(event).await;
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        if let Some((_, shard_idx)) = self.worker_assignments.remove(&worker_id) {
            self.worker_counts.lock().unwrap()[shard_idx] -= 1;
            self.shards[shard_idx].remove_worker(worker_id).await;
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        if let Some(shard_idx) = self.worker_assignments.get(&worker_id) {
            self.shards[*shard_idx]
                .remove_worker_dp_rank(worker_id, dp_rank)
                .await;
        }
    }

    fn shutdown(&self) {
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
        // No-op: pruning not supported in ShardedConcurrentIndexer.
        Ok(())
    }

    async fn flush(&self) -> usize {
        let mut total = 0;
        for shard in &self.shards {
            // Use the trait method explicitly to avoid shadowing by the inherent
            // `ThreadPoolIndexer::flush() -> ()`.
            total += <ThreadPoolIndexer<T> as KvIndexerInterface>::flush(shard).await;
        }
        total
    }

    fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(idx, shard)| {
                shard.shard_sizes().into_iter().map(move |mut s| {
                    s.shard_idx = idx;
                    s
                })
            })
            .collect()
    }

    /// Break down average `find_matches` time into:
    /// - **max-shard** — the CRTC traversal on the critical-path shard (shards
    ///   run concurrently so only the slowest one contributes to wall-clock time)
    /// - **overhead** — everything else: scatter cost + result merge + rendezvous
    ///
    /// In `spawn_blocking` mode, overhead = blocking thread pool queue time +
    /// `handle.await` + merge.  Under heavy concurrency this dominates (up to
    /// 90%+ of outer time).
    ///
    /// In pool mode, overhead = flume channel send + `oneshot` await + merge,
    /// typically < 5% of outer time.
    fn timing_report(&self) -> String {
        let calls = self.timing_calls.load(Ordering::Relaxed);
        if calls == 0 {
            return String::new();
        }
        let sum_outer_us = self.timing_sum_outer_ns.load(Ordering::Relaxed) / 1000;
        let sum_shard_us = self.timing_sum_max_shard_ns.load(Ordering::Relaxed) / 1000;
        let avg_outer_us = sum_outer_us / calls;
        let avg_shard_us = sum_shard_us / calls;
        let avg_overhead_us = avg_outer_us.saturating_sub(avg_shard_us);
        let overhead_pct = if avg_outer_us > 0 {
            100.0 * avg_overhead_us as f64 / avg_outer_us as f64
        } else {
            0.0
        };
        let mode = if self.inline_sequential {
            "sequential inline"
        } else if self.read_pools.is_some() {
            "dedicated per-shard OS thread pool"
        } else {
            "spawn_blocking scatter"
        };
        format!(
            "ShardedConcurrentIndexer find_matches ({calls} calls, {mode}):\n  \
             avg outer      = {avg_outer_us}µs\n  \
             avg max-shard  = {avg_shard_us}µs  (pure CRTC work, critical path)\n  \
             avg overhead   = {avg_overhead_us}µs  ({overhead_pct:.1}% of outer)"
        )
    }
}
