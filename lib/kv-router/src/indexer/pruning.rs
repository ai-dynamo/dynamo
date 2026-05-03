// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pruning and TTL utilities for KV Indexers
//!
//! This module provides utilities for managing TTL-based expiration of
//! approximate-mode blocks in the radix tree.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use dashmap::DashMap;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use tokio::sync::watch;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use crate::protocols::{ExternalSequenceBlockHash, WorkerId, WorkerWithDpRank};

const HEAP_REBUILD_THRESHOLD: usize = 50;
const WORKER_EXPIRY_HEAP_REBUILD_THRESHOLD: usize = 10;

/// Block entry to be inserted in the [`PruneManager::expirations`] heap.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BlockEntry {
    /// The key of the block entry.
    pub key: ExternalSequenceBlockHash,
    /// The worker (with dp_rank) that stored this block.
    pub worker: WorkerWithDpRank,
    /// The position of this block in the sequence (0-indexed).
    pub seq_position: usize,
}

impl PartialOrd for BlockEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BlockEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Break ties by sequence position, then by key, then by worker.
        self.seq_position
            .cmp(&other.seq_position)
            .then_with(|| self.key.cmp(&other.key))
            .then_with(|| self.worker.cmp(&other.worker))
    }
}

#[derive(Debug, Clone)]
pub struct PruneConfig {
    /// Time-to-live duration for blocks before they expire.
    pub ttl: Duration,
}

impl Default for PruneConfig {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(120), // 120 seconds
        }
    }
}

/// A data structure to manage a collection of timers, addressable by a key.
/// This is structured as a sort of "priority queue" of keys, where the priority is the expiration time.
/// It supports insertion as well as updating the expiration time of a key.
/// The [`PruneManager::expirations`] heap is lazily updated to reflect the true expiration times in [`PruneManager::timers`]
/// For now, we have a fixed expiration time for all keys.
#[derive(Debug)]
pub struct PruneManager<K: Clone + Hash + Eq + Ord> {
    /// The source of truth. Maps a key to its current expiration instant.
    timers: FxHashMap<K, Instant>,

    /// A max-heap of (Reverse<expiration_instant>, key) used to efficiently find the
    /// next expiring timer. Reverse<Instant> makes earlier times pop first.
    /// An entry in this heap is "stale" if the instant does not match the one in the `timers` map.
    expirations: BinaryHeap<(Reverse<Instant>, K)>,

    /// The expiration duration of the timers.
    ttl: Duration,
}

impl<K: Clone + Hash + Eq + Ord> PruneManager<K> {
    /// Creates a new, empty PruneManager.
    pub fn new(prune_config: PruneConfig) -> Self {
        let ttl = prune_config.ttl;
        PruneManager {
            timers: FxHashMap::default(),
            expirations: BinaryHeap::new(),
            ttl,
        }
    }

    /// Rebuilds the expirations heap from the timers map, removing all stale entries.
    fn rebuild_heap(&mut self) {
        self.expirations = self
            .timers
            .iter()
            .map(|(key, &expiry)| (Reverse(expiry), key.clone()))
            .collect();
    }

    /// Inserts a new timer or updates an existing one for the given key.
    ///
    /// # Arguments
    /// * `key` - The unique key for the timer.
    /// * `duration` - The duration from now when the timer should expire.
    pub fn insert(&mut self, keys: Vec<K>) {
        self.insert_at(keys, Instant::now());
    }

    /// Inserts timers using a caller-provided timestamp.
    pub fn insert_at(&mut self, keys: Vec<K>, now: Instant) {
        let expiry_time = now + self.ttl;

        self.timers.reserve(keys.len());
        self.expirations.reserve(keys.len());
        for key in keys {
            // Insert or update the authoritative time in the map.
            self.timers.insert(key.clone(), expiry_time);

            // Push the new expiration onto the heap. If the key was updated,
            // this leaves a "stale" entry on the heap for the old time,
            // which will be ignored when it's popped.
            self.expirations.push((Reverse(expiry_time), key));
        }

        // Check if we should rebuild the heap to remove stale entries
        if !self.timers.is_empty()
            && self.expirations.len() > self.timers.len() * HEAP_REBUILD_THRESHOLD
        {
            self.rebuild_heap();
        }
    }

    /// Removes a timer for the given key.
    pub fn remove(&mut self, key: &K) -> bool {
        self.timers.remove(key).is_some()
    }

    /// Polls for expired timers and returns a list of keys for all timers
    /// that have expired up to `now`.
    pub fn pop_expired(&mut self, now: Instant) -> Vec<K> {
        let mut expired_keys = Vec::new();

        while let Some((Reverse(expiry_time), _)) = self.expirations.peek() {
            // If the next timer in the heap is not yet expired, we can stop.
            if *expiry_time > now {
                break;
            }

            // The timer might be expired, so pop it from the heap.
            let (Reverse(expiry_time), key) = self.expirations.pop().unwrap();

            if self.timers.get(&key) == Some(&expiry_time) {
                // This is a valid, non-stale, expired timer.
                self.timers.remove(&key);
                expired_keys.push(key);
            }
        }

        expired_keys
    }

    /// Returns the next non-stale expiry time, if it exists.
    pub fn peek_next_valid_expiry(&mut self) -> Option<Instant> {
        while let Some((Reverse(expiry_time), key)) = self.expirations.peek() {
            if self.timers.get(key) == Some(expiry_time) {
                return Some(*expiry_time);
            }
            self.expirations.pop();
        }
        None
    }

    pub fn len(&self) -> usize {
        self.timers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timers.is_empty()
    }
}

#[derive(Debug)]
struct WorkerPruneState {
    timers: PruneManager<BlockEntry>,
}

impl WorkerPruneState {
    fn new(config: PruneConfig) -> Self {
        Self {
            timers: PruneManager::new(config),
        }
    }

    fn insert_block_entries(&mut self, entries: Vec<BlockEntry>, now: Instant) {
        self.timers.insert_at(entries, now);
    }

    fn remove_block_entry(&mut self, entry: &BlockEntry) {
        self.timers.remove(entry);
    }

    fn pop_expired(&mut self, now: Instant) -> Vec<BlockEntry> {
        self.timers.pop_expired(now)
    }

    fn peek_next_valid_expiry(&mut self) -> Option<Instant> {
        self.timers.peek_next_valid_expiry()
    }
}

#[derive(Clone)]
pub struct WorkerPruneManager {
    inner: Arc<WorkerPruneManagerInner>,
}

struct WorkerPruneManagerInner {
    config: PruneConfig,
    workers: DashMap<WorkerWithDpRank, Mutex<WorkerPruneState>, FxBuildHasher>,
    next_expiries: Mutex<BinaryHeap<(Reverse<Instant>, WorkerWithDpRank)>>,
    pending_removes: Mutex<VecDeque<BlockEntry>>,
    ready_tx: watch::Sender<u64>,
    schedule_tx: watch::Sender<u64>,
    cancel: CancellationToken,
}

impl Drop for WorkerPruneManagerInner {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

impl WorkerPruneManager {
    pub fn new(config: PruneConfig) -> Self {
        let (ready_tx, _) = watch::channel(0);
        let (schedule_tx, schedule_rx) = watch::channel(0);
        let inner = Arc::new(WorkerPruneManagerInner {
            config,
            workers: DashMap::with_hasher(FxBuildHasher),
            next_expiries: Mutex::new(BinaryHeap::new()),
            pending_removes: Mutex::new(VecDeque::new()),
            ready_tx,
            schedule_tx,
            cancel: CancellationToken::new(),
        });

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(Self::ttl_task(Arc::clone(&inner), schedule_rx));
        }

        Self { inner }
    }

    async fn ttl_task(inner: Arc<WorkerPruneManagerInner>, mut schedule_rx: watch::Receiver<u64>) {
        loop {
            let Some(next_expiry) = inner.next_global_expiry() else {
                tokio::select! {
                    _ = inner.cancel.cancelled() => break,
                    changed = schedule_rx.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                }
                continue;
            };

            tokio::select! {
                _ = inner.cancel.cancelled() => break,
                _ = tokio::time::sleep_until(next_expiry) => {
                    inner.queue_due(Instant::now());
                }
                changed = schedule_rx.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
            }
        }
    }

    pub fn insert_block_entries(&self, entries: Vec<BlockEntry>) {
        if entries.is_empty() {
            return;
        }

        let now = Instant::now();
        let mut by_worker: FxHashMap<WorkerWithDpRank, Vec<BlockEntry>> =
            FxHashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for entry in entries {
            by_worker.entry(entry.worker).or_default().push(entry);
        }

        let mut should_bump_schedule = false;
        for (worker, worker_entries) in by_worker {
            should_bump_schedule |=
                self.insert_worker_block_entries_at(worker, worker_entries, now);
        }

        if should_bump_schedule {
            self.inner.bump_schedule();
        }
    }

    pub fn insert_worker_block_entries(&self, worker: WorkerWithDpRank, entries: Vec<BlockEntry>) {
        if entries.is_empty() {
            return;
        }

        if self.insert_worker_block_entries_at(worker, entries, Instant::now()) {
            self.inner.bump_schedule();
        }
    }

    fn insert_worker_block_entries_at(
        &self,
        worker: WorkerWithDpRank,
        entries: Vec<BlockEntry>,
        now: Instant,
    ) -> bool {
        let (old_next, new_next) = {
            let state =
                self.inner.workers.entry(worker).or_insert_with(|| {
                    Mutex::new(WorkerPruneState::new(self.inner.config.clone()))
                });
            let mut state = state.lock().expect("worker prune state mutex poisoned");
            let old_next = state.peek_next_valid_expiry();
            state.insert_block_entries(entries, now);
            let new_next = state.peek_next_valid_expiry();
            (old_next, new_next)
        };

        match (old_next, new_next) {
            (None, Some(next_expiry)) => {
                self.inner.push_worker_expiry(worker, next_expiry);
                true
            }
            (Some(old_next), Some(new_next)) if new_next < old_next => {
                tracing::warn!(
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    ?old_next,
                    ?new_next,
                    "Approximate prune expiry moved earlier during insert; rescheduling"
                );
                debug_assert!(
                    new_next >= old_next,
                    "approximate prune expiry moved earlier during insert"
                );
                self.inner.push_worker_expiry(worker, new_next);
                true
            }
            _ => false,
        }
    }

    pub fn remove_block_entries(&self, entries: &[BlockEntry]) {
        if entries.is_empty() {
            return;
        }

        let removed_entries: FxHashSet<_> = entries.iter().copied().collect();
        let mut by_worker: FxHashMap<WorkerWithDpRank, Vec<BlockEntry>> =
            FxHashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for entry in entries {
            by_worker.entry(entry.worker).or_default().push(*entry);
        }

        for (worker, worker_entries) in by_worker {
            let next_expiry = {
                let Some(state) = self.inner.workers.get(&worker) else {
                    continue;
                };
                let mut state = state.lock().expect("worker prune state mutex poisoned");
                for entry in &worker_entries {
                    state.remove_block_entry(entry);
                }
                state.peek_next_valid_expiry()
            };
            if let Some(next_expiry) = next_expiry {
                self.inner.push_worker_expiry(worker, next_expiry);
            }
        }

        self.inner
            .pending_removes
            .lock()
            .expect("pending prune remove queue mutex poisoned")
            .retain(|entry| !removed_entries.contains(entry));
        self.inner.bump_schedule();
    }

    pub fn remove_worker(&self, worker_id: WorkerId) {
        let workers: Vec<_> = self
            .inner
            .workers
            .iter()
            .filter_map(|entry| {
                let worker = *entry.key();
                (worker.worker_id == worker_id).then_some(worker)
            })
            .collect();
        for worker in workers {
            self.inner.workers.remove(&worker);
        }
        self.inner
            .pending_removes
            .lock()
            .expect("pending prune remove queue mutex poisoned")
            .retain(|entry| entry.worker.worker_id != worker_id);
        self.inner.bump_schedule();
    }

    pub fn remove_worker_dp_rank(&self, worker: WorkerWithDpRank) {
        self.inner.workers.remove(&worker);
        self.inner
            .pending_removes
            .lock()
            .expect("pending prune remove queue mutex poisoned")
            .retain(|entry| entry.worker != worker);
        self.inner.bump_schedule();
    }

    pub fn drain_due_and_pending(&self, now: Instant) -> Vec<BlockEntry> {
        self.inner.queue_due(now);
        self.drain_pending_removes()
    }

    pub fn drain_pending_removes(&self) -> Vec<BlockEntry> {
        self.inner
            .pending_removes
            .lock()
            .expect("pending prune remove queue mutex poisoned")
            .drain(..)
            .collect()
    }

    pub fn subscribe_ready(&self) -> watch::Receiver<u64> {
        self.inner.ready_tx.subscribe()
    }

    pub fn shutdown(&self) {
        self.inner.cancel.cancel();
    }
}

impl WorkerPruneManagerInner {
    fn bump_sender(sender: &watch::Sender<u64>) {
        let next = sender.borrow().wrapping_add(1);
        let _ = sender.send(next);
    }

    fn bump_schedule(&self) {
        Self::bump_sender(&self.schedule_tx);
    }

    fn bump_ready(&self) {
        Self::bump_sender(&self.ready_tx);
    }

    fn push_worker_expiry(&self, worker: WorkerWithDpRank, expiry: Instant) {
        self.next_expiries
            .lock()
            .expect("worker expiry index mutex poisoned")
            .push((Reverse(expiry), worker));
        self.rebuild_worker_expiry_heap_if_needed();
    }

    fn rebuild_worker_expiry_heap_if_needed(&self) {
        let workers_len = self.workers.len();
        if workers_len == 0 {
            return;
        }

        let should_rebuild = {
            let expiries = self
                .next_expiries
                .lock()
                .expect("worker expiry index mutex poisoned");
            expiries.len() > workers_len * WORKER_EXPIRY_HEAP_REBUILD_THRESHOLD
        };
        if !should_rebuild {
            return;
        }

        let mut rebuilt = BinaryHeap::new();
        for entry in self.workers.iter() {
            let worker = *entry.key();
            let next_expiry = entry
                .value()
                .lock()
                .expect("worker prune state mutex poisoned")
                .peek_next_valid_expiry();
            if let Some(next_expiry) = next_expiry {
                rebuilt.push((Reverse(next_expiry), worker));
            }
        }

        *self
            .next_expiries
            .lock()
            .expect("worker expiry index mutex poisoned") = rebuilt;
    }

    fn next_global_expiry(&self) -> Option<Instant> {
        loop {
            let candidate = self
                .next_expiries
                .lock()
                .expect("worker expiry index mutex poisoned")
                .peek()
                .copied();
            let (Reverse(expiry), worker) = candidate?;

            let next_valid = self.workers.get(&worker).and_then(|state| {
                state
                    .lock()
                    .expect("worker prune state mutex poisoned")
                    .peek_next_valid_expiry()
            });

            if next_valid == Some(expiry) {
                return Some(expiry);
            }

            let mut expiries = self
                .next_expiries
                .lock()
                .expect("worker expiry index mutex poisoned");
            if expiries.peek().copied() != candidate {
                continue;
            }
            expiries.pop();
            if let Some(next_valid) = next_valid {
                expiries.push((Reverse(next_valid), worker));
            }
        }
    }

    fn queue_due(&self, now: Instant) {
        let mut expired = Vec::new();

        loop {
            let Some((Reverse(expiry), worker)) = ({
                let mut expiries = self
                    .next_expiries
                    .lock()
                    .expect("worker expiry index mutex poisoned");
                let Some((Reverse(expiry), _)) = expiries.peek().copied() else {
                    break;
                };
                if expiry > now {
                    break;
                }
                expiries.pop()
            }) else {
                break;
            };

            let (mut worker_expired, next_expiry) = {
                let Some(state) = self.workers.get(&worker) else {
                    continue;
                };
                let mut state = state.lock().expect("worker prune state mutex poisoned");
                let next_valid = state.peek_next_valid_expiry();
                if next_valid != Some(expiry) {
                    (Vec::new(), next_valid)
                } else {
                    let expired = state.pop_expired(now);
                    let next_expiry = state.peek_next_valid_expiry();
                    (expired, next_expiry)
                }
            };

            if let Some(next_expiry) = next_expiry {
                self.push_worker_expiry(worker, next_expiry);
            }
            expired.append(&mut worker_expired);
        }

        if expired.is_empty() {
            return;
        }

        self.pending_removes
            .lock()
            .expect("pending prune remove queue mutex poisoned")
            .extend(expired);
        self.bump_ready();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::WorkerWithDpRank;
    use tokio::time::{self, Duration, Instant};

    impl<T: Clone + Hash + Eq + Ord> PruneManager<T> {
        pub fn get_expiry(&self, key: &T) -> Option<&Instant> {
            self.timers.get(key)
        }
    }

    impl WorkerPruneManager {
        fn worker_expiry_heap_len(&self) -> usize {
            self.inner
                .next_expiries
                .lock()
                .expect("worker expiry index mutex poisoned")
                .len()
        }
    }

    fn test_block(worker: WorkerWithDpRank, key: u64, seq_position: usize) -> BlockEntry {
        BlockEntry {
            key: ExternalSequenceBlockHash(key),
            worker,
            seq_position,
        }
    }

    /// Validate basic insert / expiry behaviour of [`PruneManager`].
    #[tokio::test]
    async fn test_prune_manager_expiry() {
        const TTL: Duration = Duration::from_millis(50);
        let prune_config = PruneConfig { ttl: TTL };
        let mut pm: PruneManager<u32> = PruneManager::new(prune_config);

        pm.insert(vec![1, 2, 3]);
        assert!(pm.get_expiry(&1).is_some());
        assert!(pm.get_expiry(&2).is_some());
        assert!(pm.get_expiry(&3).is_some());

        // Wait until after the TTL
        time::sleep(TTL + Duration::from_millis(20)).await;
        let expired = pm.pop_expired(Instant::now());
        assert_eq!(expired.len(), 3);
        assert!(pm.get_expiry(&1).is_none());
        assert!(pm.get_expiry(&2).is_none());
        assert!(pm.get_expiry(&3).is_none());
    }

    /// Validate that reinserting an existing key extends its TTL and prevents premature expiry.
    #[tokio::test]
    async fn test_prune_manager_update_resets_ttl() {
        // Validate that reinserting an existing key extends its TTL and prevents premature expiry.
        const TTL: Duration = Duration::from_millis(50);
        let prune_config = PruneConfig { ttl: TTL };
        let mut pm: PruneManager<u32> = PruneManager::new(prune_config);

        // Initial insert and capture the original expiry.
        pm.insert(vec![42]);
        let first_expiry = *pm
            .get_expiry(&42)
            .expect("expiry missing after first insert");

        // Wait for half of the original TTL before reinserting.
        time::sleep(Duration::from_millis(25)).await;
        pm.insert(vec![42]);
        let second_expiry = *pm
            .get_expiry(&42)
            .expect("expiry missing after reinsertion");

        // The expiry after reinsertion must be strictly later than the first one.
        assert!(second_expiry > first_expiry);

        // Wait until *after* the first expiry would have fired, but *before* the new expiry.
        time::sleep(Duration::from_millis(30)).await; // 25ms already elapsed, +30ms = 55ms > first TTL
        let expired = pm.pop_expired(Instant::now());
        assert!(
            expired.is_empty(),
            "key expired prematurely despite TTL refresh"
        );

        // Now wait until after the second expiry should have occurred.
        time::sleep(Duration::from_millis(30)).await; // Ensure we pass the refreshed TTL
        let expired_after = pm.pop_expired(Instant::now());
        assert_eq!(expired_after, vec![42]);
    }

    /// Test that BlockEntry ordering prioritizes sequence position.
    #[test]
    fn test_block_entry_ordering() {
        let worker = WorkerWithDpRank::from_worker_id(0);

        let entry1 = BlockEntry {
            key: ExternalSequenceBlockHash(100),
            worker,
            seq_position: 0,
        };
        let entry2 = BlockEntry {
            key: ExternalSequenceBlockHash(50),
            worker,
            seq_position: 1,
        };

        // entry1 < entry2 because seq_position 0 < 1
        assert!(entry1 < entry2);
    }

    #[test]
    fn test_worker_expiry_heap_rebuilds_under_churn() {
        let manager = WorkerPruneManager::new(PruneConfig {
            ttl: Duration::from_secs(60),
        });
        let worker = WorkerWithDpRank::from_worker_id(7);

        for idx in 0..=WORKER_EXPIRY_HEAP_REBUILD_THRESHOLD {
            manager.insert_block_entries(vec![test_block(worker, 100 + idx as u64, idx)]);
        }

        assert_eq!(manager.worker_expiry_heap_len(), 1);
        manager.shutdown();
    }

    #[tokio::test(start_paused = true)]
    async fn test_worker_expiry_queue_drains_staggered_workers() {
        const TTL: Duration = Duration::from_secs(10);
        let manager = WorkerPruneManager::new(PruneConfig { ttl: TTL });
        let first_worker = WorkerWithDpRank::from_worker_id(1);
        let second_worker = WorkerWithDpRank::from_worker_id(2);

        let first = test_block(first_worker, 101, 0);
        let second = test_block(second_worker, 202, 0);

        manager.insert_block_entries(vec![first]);
        time::advance(Duration::from_secs(5)).await;
        manager.insert_block_entries(vec![second]);

        time::advance(Duration::from_secs(5)).await;
        assert_eq!(manager.drain_due_and_pending(Instant::now()), vec![first]);

        time::advance(Duration::from_secs(5)).await;
        assert_eq!(manager.drain_due_and_pending(Instant::now()), vec![second]);
        manager.shutdown();
    }

    #[tokio::test(start_paused = true)]
    async fn test_worker_expiry_queue_ignores_stale_global_entries() {
        const TTL: Duration = Duration::from_secs(10);
        let manager = WorkerPruneManager::new(PruneConfig { ttl: TTL });
        let worker = WorkerWithDpRank::from_worker_id(7);
        let block = test_block(worker, 707, 0);

        manager.insert_block_entries(vec![block]);
        time::advance(Duration::from_secs(5)).await;
        manager.insert_block_entries(vec![block]);

        time::advance(Duration::from_secs(5)).await;
        assert!(manager.drain_due_and_pending(Instant::now()).is_empty());

        time::advance(Duration::from_secs(5)).await;
        assert_eq!(manager.drain_due_and_pending(Instant::now()), vec![block]);
        manager.shutdown();
    }
}
