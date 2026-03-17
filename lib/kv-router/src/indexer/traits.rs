// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_trait::async_trait;
use parking_lot::Mutex;
#[cfg(feature = "lock-stats")]
use std::sync::atomic::{AtomicU64, Ordering};

use super::{KvRouterError, WorkerTask};
use crate::protocols::*;

#[async_trait]
pub trait KvIndexerInterface {
    /// Find matches for a given sequence of `LocalBlockHash`es.
    ///
    /// ### Arguments
    ///
    /// * `sequence` - A vector of `LocalBlockHash` representing the sequence to match.
    ///
    /// ### Returns
    ///
    /// An `OverlapScores` representing the match scores.
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError>;

    /// Find matches for a given sequence of tokens.
    ///
    /// ### Arguments
    ///
    /// * `tokens` - A vector of `u32` tokens.
    /// * `lora_name` - Optional LoRA adapter name to include in block hash computation.
    ///
    /// ### Returns
    ///
    /// An `OverlapScores` representing the match scores.
    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
        lora_name: Option<&str>,
    ) -> Result<OverlapScores, KvRouterError>;

    /// Apply a `RouterEvent` to the KV store.
    ///
    /// ### Arguments
    ///
    /// * `event` - The `RouterEvent` to apply.
    async fn apply_event(&self, event: RouterEvent);

    /// Remove a worker's entries from the trie.
    ///
    /// ### Arguments
    ///
    /// * `worker` - The worker to remove from the trie.
    async fn remove_worker(&self, worker: WorkerId);

    /// Remove a single dp_rank for a worker from the trie.
    ///
    /// Default implementation falls back to removing the entire worker.
    /// Indexers that track dp_rank-level granularity should override this.
    async fn remove_worker_dp_rank(&self, worker: WorkerId, _dp_rank: DpRank) {
        self.remove_worker(worker).await;
    }

    /// Shutdown the KV Indexer.
    fn shutdown(&self);

    /// Dump the entire tree as RouterEvents.
    ///
    /// ### Returns
    ///
    /// A vector of RouterEvents representing the current state of the tree.
    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError>;

    /// Process a routing decision for a request with tokens.
    ///
    /// Uses TokensWithHashes for lazy hash computation - if hashes were already
    /// computed (e.g., by find_best_match), they will be reused.
    ///
    /// ### Arguments
    ///
    /// * `tokens_with_hashes` - Tokens with lazily computed hashes.
    /// * `worker` - The worker (with dp_rank) that was selected.
    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError>;

    /// Async task that returns when all pending events have been processed.
    /// For now, we assume that no requests or events are being sent in the meantime.
    /// Returns the amount of events still in the queue at the time of the flush.
    /// Used primarily for debugging.
    async fn flush(&self) -> usize;
}

// ============================================================================
// SyncIndexer trait
// ============================================================================

/// Trait for thread-safe data structures that support KV cache indexing operations.
///
/// All methods take `&self` and are synchronous. Implementations must be safe for
/// concurrent access (via internal locking, DashMap, etc).
///
/// This trait is used with [`ThreadPoolIndexer`](super::ThreadPoolIndexer), which wraps a `SyncIndexer` to
/// provide the async [`KvIndexerInterface`] with:
/// - Sticky event routing to N worker threads
/// - Inline reads on the caller's thread (no channel dispatch for find_matches)
pub trait SyncIndexer: Send + Sync + 'static {
    fn worker(&self, event_receiver: flume::Receiver<WorkerTask>) -> anyhow::Result<()>;

    /// Find matches for a sequence of block hashes.
    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores;

    /// Dump events directly from the shared structure, bypassing worker channels.
    /// Returns `Some(events)` for backends whose tree state is fully shared (e.g.
    /// ConcurrentRadixTree). Returns `None` for backends that keep per-thread
    /// state and must dump via the worker channel.
    fn dump_events(&self) -> Option<Vec<RouterEvent>> {
        None
    }

    fn total_tree_size(&self) -> usize {
        0
    }
}

/// Which operation type a node-touch counter belongs to.
pub enum NodeTouchOp {
    FindMatches,
    Store,
    Remove,
}

/// RAII guard that records the accumulated node-touch count when dropped.
/// Created via [`NodeTouchStats::guard`]. Increment `count` at each
/// node lock acquisition; the total is automatically pushed into the
/// correct per-operation `Vec` on any return path.
pub struct NodeTouchGuard<'a> {
    stats: &'a NodeTouchStats,
    op: NodeTouchOp,
    pub count: u64,
}

impl Drop for NodeTouchGuard<'_> {
    fn drop(&mut self) {
        let vec = match self.op {
            NodeTouchOp::FindMatches => &self.stats.find_matches_counts,
            NodeTouchOp::Store => &self.stats.store_counts,
            NodeTouchOp::Remove => &self.stats.remove_counts,
        };
        vec.lock().push(self.count);
    }
}

/// Per-operation node traversal counters for benchmarking.
///
/// Records how many tree nodes are lock-acquired during each
/// `find_matches`, `apply_stored`, and `apply_removed` call.
/// Enable the `node-stats` feature, then call `report()` or rely
/// on the automatic report printed when the tree is dropped.
pub struct NodeTouchStats {
    find_matches_counts: Mutex<Vec<u64>>,
    store_counts: Mutex<Vec<u64>>,
    remove_counts: Mutex<Vec<u64>>,
}

impl NodeTouchStats {
    pub fn new() -> Self {
        Self {
            find_matches_counts: Mutex::new(Vec::new()),
            store_counts: Mutex::new(Vec::new()),
            remove_counts: Mutex::new(Vec::new()),
        }
    }

    pub fn guard(&self, op: NodeTouchOp) -> NodeTouchGuard<'_> {
        NodeTouchGuard {
            stats: self,
            op,
            count: 0,
        }
    }

    fn percentile_stats(counts: &mut [u64]) -> (f64, u64, u64, u64) {
        if counts.is_empty() {
            return (0.0, 0, 0, 0);
        }
        counts.sort_unstable();
        let mean = counts.iter().sum::<u64>() as f64 / counts.len() as f64;
        let p50 = counts[counts.len() * 50 / 100];
        let p99 = counts[counts.len() * 99 / 100];
        let max = *counts.iter().max().unwrap();
        (mean, p50, p99, max)
    }

    pub fn report(&self, label: &str) {
        let mut fm = self.find_matches_counts.lock();
        let mut st = self.store_counts.lock();
        let mut rm = self.remove_counts.lock();

        let (fm_mean, fm_p50, fm_p99, fm_max) = Self::percentile_stats(&mut fm);
        let (st_mean, st_p50, st_p99, st_max) = Self::percentile_stats(&mut st);
        let (rm_mean, rm_p50, rm_p99, rm_max) = Self::percentile_stats(&mut rm);

        println!("Node touch stats ({label}):");
        println!(
            "  find_matches ({} calls): mean={:.1}  p50={}  p99={}  max={}",
            fm.len(),
            fm_mean,
            fm_p50,
            fm_p99,
            fm_max
        );
        println!(
            "  store ({} calls):        mean={:.1}  p50={}  p99={}  max={}",
            st.len(),
            st_mean,
            st_p50,
            st_p99,
            st_max
        );
        println!(
            "  remove ({} calls):       mean={:.1}  p50={}  p99={}  max={}",
            rm.len(),
            rm_mean,
            rm_p50,
            rm_p99,
            rm_max
        );
    }
}

// ============================================================================
// Lock contention statistics (feature-gated behind "lock-stats")
// ============================================================================

/// Per-instance lock contention counters for benchmarking.
///
/// Tracks how often `RwLock::read()` and `write()` calls find the lock already
/// held (detected via `try_read()`/`try_write()` failing) and cumulative time
/// spent waiting. Timing uses `std::time::Instant` only on the contended slow
/// path, so the clock read overhead (~25ns) is negligible relative to the
/// microsecond-scale waits being measured.
///
/// On the uncontended fast path, the only added cost is a single
/// `fetch_add(1, Relaxed)` (~1-2ns) to increment the total counter.
#[cfg(feature = "lock-stats")]
pub struct LockContentionStats {
    read_total: AtomicU64,
    read_contended: AtomicU64,
    read_wait_ns: AtomicU64,
    write_total: AtomicU64,
    write_contended: AtomicU64,
    write_wait_ns: AtomicU64,
}

#[cfg(feature = "lock-stats")]
impl LockContentionStats {
    pub fn new() -> Self {
        Self {
            read_total: AtomicU64::new(0),
            read_contended: AtomicU64::new(0),
            read_wait_ns: AtomicU64::new(0),
            write_total: AtomicU64::new(0),
            write_contended: AtomicU64::new(0),
            write_wait_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn tracked_read<'a, T>(
        &self,
        lock: &'a parking_lot::RwLock<T>,
    ) -> parking_lot::RwLockReadGuard<'a, T> {
        self.read_total.fetch_add(1, Ordering::Relaxed);
        match lock.try_read() {
            Some(guard) => guard,
            None => {
                self.read_contended.fetch_add(1, Ordering::Relaxed);
                let start = std::time::Instant::now();
                let guard = lock.read();
                self.read_wait_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                guard
            }
        }
    }

    #[inline]
    pub fn tracked_write<'a, T>(
        &self,
        lock: &'a parking_lot::RwLock<T>,
    ) -> parking_lot::RwLockWriteGuard<'a, T> {
        self.write_total.fetch_add(1, Ordering::Relaxed);
        match lock.try_write() {
            Some(guard) => guard,
            None => {
                self.write_contended.fetch_add(1, Ordering::Relaxed);
                let start = std::time::Instant::now();
                let guard = lock.write();
                self.write_wait_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                guard
            }
        }
    }

    /// Acquire an upgradable read lock with contention tracking.
    /// Counted as a read acquisition since it is compatible with plain readers.
    #[inline]
    pub fn tracked_upgradable_read<'a, T>(
        &self,
        lock: &'a parking_lot::RwLock<T>,
    ) -> parking_lot::RwLockUpgradableReadGuard<'a, T> {
        self.read_total.fetch_add(1, Ordering::Relaxed);
        match lock.try_upgradable_read() {
            Some(guard) => guard,
            None => {
                self.read_contended.fetch_add(1, Ordering::Relaxed);
                let start = std::time::Instant::now();
                let guard = lock.upgradable_read();
                self.read_wait_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                guard
            }
        }
    }

    /// Upgrade an upgradable read guard to a write guard with contention tracking.
    /// Counted as a write acquisition since it becomes exclusive.
    #[inline]
    pub fn tracked_upgrade<'a, T>(
        &self,
        guard: parking_lot::RwLockUpgradableReadGuard<'a, T>,
    ) -> parking_lot::RwLockWriteGuard<'a, T> {
        self.write_total.fetch_add(1, Ordering::Relaxed);
        match parking_lot::RwLockUpgradableReadGuard::try_upgrade(guard) {
            Ok(write_guard) => write_guard,
            Err(guard) => {
                self.write_contended.fetch_add(1, Ordering::Relaxed);
                let start = std::time::Instant::now();
                let write_guard = parking_lot::RwLockUpgradableReadGuard::upgrade(guard);
                self.write_wait_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                write_guard
            }
        }
    }

    pub fn report(&self, label: &str) {
        let rt = self.read_total.load(Ordering::Relaxed);
        let rc = self.read_contended.load(Ordering::Relaxed);
        let rw = self.read_wait_ns.load(Ordering::Relaxed);
        let wt = self.write_total.load(Ordering::Relaxed);
        let wc = self.write_contended.load(Ordering::Relaxed);
        let ww = self.write_wait_ns.load(Ordering::Relaxed);

        println!("Lock contention stats ({label}):");
        println!(
            "  Read:  {} total, {} contended ({:.2}%), {:.1}ms total wait",
            rt,
            rc,
            if rt > 0 {
                rc as f64 / rt as f64 * 100.0
            } else {
                0.0
            },
            rw as f64 / 1_000_000.0,
        );
        println!(
            "  Write: {} total, {} contended ({:.2}%), {:.1}ms total wait",
            wt,
            wc,
            if wt > 0 {
                wc as f64 / wt as f64 * 100.0
            } else {
                0.0
            },
            ww as f64 / 1_000_000.0,
        );
    }
}
