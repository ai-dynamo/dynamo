// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize},
    },
    thread::JoinHandle,
};

use async_trait::async_trait;
use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use tokio::sync::oneshot;

use super::{
    KvIndexerInterface, KvIndexerMetrics, KvRouterError, ShardSizeSnapshot, SyncIndexer, WorkerTask,
};
use crate::indexer::pruning::{BlockEntry, PruneConfig, WorkerPruneManager};
use crate::protocols::*;
use dynamo_tokens::SequenceHash;

/// Generic wrapper that provides [`KvIndexerInterface`] for any [`SyncIndexer`] backend.
///
/// Spawns N OS threads for processing write events (sticky-routed by WorkerId).
/// Read operations (find_matches) are executed inline on the caller's thread,
/// avoiding channel overhead and allowing reads to scale with callers.
///
/// # Architecture
///
/// ```text
///                                       +------------------------------------+
///                                       |     N Worker Threads (OS threads)  |
///                                       |                                    |
///  worker_event_channels[0] ----------> |   Thread 0: blocking recv loop     |
///  worker_event_channels[1] ----------> |   Thread 1: blocking recv loop     |
///  worker_event_channels[N] ----------> |   Thread N: blocking recv loop     |
///                                       |                                    |
///  find_matches() ---(inline)---------> |   Arc<T: SyncIndexer>              |
///                                       |   (shared, thread-safe)            |
///                                       +------------------------------------+
/// ```
pub struct ThreadPoolIndexer<T: SyncIndexer> {
    /// Shared backend - thread-safe via internal locking.
    backend: Arc<T>,

    /// Maps WorkerId to worker thread index for sticky routing.
    worker_assignments: Arc<DashMap<WorkerId, usize, FxBuildHasher>>,
    /// Counter for round-robin assignment of new WorkerIds.
    worker_assignment_count: Arc<AtomicUsize>,

    /// Channels to send tasks to worker threads (one per thread).
    /// Sending `WorkerTask::Terminate` signals the thread to shut down.
    worker_event_channels: Vec<flume::Sender<WorkerTask>>,

    /// Number of worker threads.
    num_workers: usize,
    /// Block size for KV cache.
    kv_block_size: u32,

    /// Handles to worker threads for joining on shutdown.
    thread_handles: Mutex<Vec<JoinHandle<()>>>,

    /// Approximate-mode TTL pruning manager. None for normal event-driven mode.
    prune_manager: Option<WorkerPruneManager>,

    /// Cancellation token for the threaded prune pump.
    prune_pump_cancel: Option<tokio_util::sync::CancellationToken>,

    /// Synthetic event IDs for approximate store/remove events.
    synthetic_event_id: Arc<AtomicU64>,
}

impl<T: SyncIndexer> ThreadPoolIndexer<T> {
    /// Create a new `ThreadPoolIndexer` wrapping the given backend.
    ///
    /// Spawns `num_workers` OS threads, each running a blocking recv loop
    /// that processes events by calling `backend.apply_event()`.
    ///
    /// # Arguments
    ///
    /// * `backend` - The thread-safe data structure to wrap
    /// * `num_workers` - Number of worker threads for event processing
    /// * `kv_block_size` - Block size for KV cache
    ///
    /// # Panics
    ///
    /// Panics if `num_workers` is 0.
    pub fn new(backend: T, num_workers: usize, kv_block_size: u32) -> Self {
        Self::new_with_metrics(backend, num_workers, kv_block_size, None)
    }

    /// Create a new `ThreadPoolIndexer` with optional metrics.
    ///
    /// Same as [`new`](Self::new) but allows passing `KvIndexerMetrics` so that
    /// each worker thread records `kv_cache_events_applied` counters, matching
    /// the observability of the single-threaded `KvIndexer` path.
    ///
    /// # Arguments
    ///
    /// * `backend` - The thread-safe data structure to wrap
    /// * `num_workers` - Number of worker threads for event processing
    /// * `kv_block_size` - Block size for KV cache
    /// * `metrics` - Optional metrics to record event application counts
    ///
    /// # Panics
    ///
    /// Panics if `num_workers` is 0.
    pub fn new_with_metrics(
        backend: T,
        num_workers: usize,
        kv_block_size: u32,
        metrics: Option<Arc<KvIndexerMetrics>>,
    ) -> Self {
        Self::new_with_metrics_and_pruning(backend, num_workers, kv_block_size, metrics, None)
    }

    pub fn new_with_pruning(
        backend: T,
        num_workers: usize,
        kv_block_size: u32,
        prune_config: PruneConfig,
    ) -> Self {
        Self::new_with_metrics_and_pruning(
            backend,
            num_workers,
            kv_block_size,
            None,
            Some(prune_config),
        )
    }

    pub fn new_with_metrics_and_pruning(
        backend: T,
        num_workers: usize,
        kv_block_size: u32,
        metrics: Option<Arc<KvIndexerMetrics>>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        assert!(num_workers > 0, "Number of workers must be greater than 0");
        super::warn_on_unit_block_size("thread_pool", kv_block_size);

        let backend = Arc::new(backend);
        let mut worker_event_senders = Vec::new();
        let mut thread_handles = Vec::new();
        let worker_assignments = Arc::new(DashMap::with_hasher(FxBuildHasher));
        let worker_assignment_count = Arc::new(AtomicUsize::new(0));
        let synthetic_event_id = Arc::new(AtomicU64::new(0));
        for _ in 0..num_workers {
            let (event_sender, event_receiver) = flume::unbounded::<WorkerTask>();
            worker_event_senders.push(event_sender);

            let backend = Arc::clone(&backend);
            let metrics = metrics.clone();

            let handle = std::thread::spawn(move || {
                backend.worker(event_receiver, metrics).unwrap();
            });
            thread_handles.push(handle);
        }

        let prune_manager = prune_config.map(WorkerPruneManager::new);
        let prune_pump_cancel = prune_manager.as_ref().map(|prune_manager| {
            let cancel = tokio_util::sync::CancellationToken::new();
            Self::spawn_prune_pump(
                prune_manager.clone(),
                worker_event_senders.clone(),
                Arc::clone(&worker_assignments),
                Arc::clone(&worker_assignment_count),
                num_workers,
                Arc::clone(&synthetic_event_id),
                cancel.clone(),
            );
            cancel
        });

        Self {
            backend,
            worker_assignments,
            worker_assignment_count,
            worker_event_channels: worker_event_senders,
            num_workers,
            kv_block_size,
            thread_handles: Mutex::new(thread_handles),
            prune_manager,
            prune_pump_cancel,
            synthetic_event_id,
        }
    }

    /// Get a reference to the underlying backend.
    pub fn backend(&self) -> &T {
        &self.backend
    }

    /// Get a cloned `Arc` to the underlying backend.
    ///
    /// Useful when a caller needs to hand off an owned `Arc<T>` to a blocking
    /// task (e.g. `tokio::task::spawn_blocking`) without cloning the backend
    /// itself.
    pub fn backend_arc(&self) -> Arc<T> {
        Arc::clone(&self.backend)
    }

    /// Wait until all previously queued worker tasks have completed.
    ///
    /// Used primarily for testing and benchmarking to ensure writes are visible
    /// before checking results.
    pub async fn flush(&self) {
        let mut receivers = Vec::new();
        for channel in &self.worker_event_channels {
            let (resp_tx, resp_rx) = oneshot::channel();
            if channel.send(WorkerTask::Flush(resp_tx)).is_ok() {
                receivers.push(resp_rx);
            }
        }
        for receiver in receivers {
            let _ = receiver.await;
        }
    }

    fn get_or_assign_thread_idx(
        worker_assignments: &DashMap<WorkerId, usize, FxBuildHasher>,
        worker_assignment_count: &AtomicUsize,
        worker_id: WorkerId,
        num_workers: usize,
    ) -> usize {
        *worker_assignments.entry(worker_id).or_insert_with(|| {
            let idx = worker_assignment_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            idx % num_workers
        })
    }

    fn next_synthetic_event_id(synthetic_event_id: &AtomicU64) -> u64 {
        synthetic_event_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    fn block_entries_for_hashes(
        worker: WorkerWithDpRank,
        sequence_hashes: &[SequenceHash],
    ) -> Vec<BlockEntry> {
        sequence_hashes
            .iter()
            .enumerate()
            .map(|(idx, h)| BlockEntry {
                key: ExternalSequenceBlockHash(*h),
                worker,
                seq_position: idx,
            })
            .collect()
    }

    fn stored_event_for_hashes(
        worker: WorkerWithDpRank,
        local_hashes: &[LocalBlockHash],
        sequence_hashes: &[SequenceHash],
        event_id: u64,
    ) -> RouterEvent {
        let blocks = local_hashes
            .iter()
            .zip(sequence_hashes.iter())
            .map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                tokens_hash: *local_hash,
                block_hash: ExternalSequenceBlockHash(*sequence_hash),
                mm_extra_info: None,
            })
            .collect();

        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks,
                }),
                dp_rank: worker.dp_rank,
            },
        )
    }

    fn enqueue_prune_removes(
        worker_event_channels: &[flume::Sender<WorkerTask>],
        worker_assignments: &DashMap<WorkerId, usize, FxBuildHasher>,
        worker_assignment_count: &AtomicUsize,
        num_workers: usize,
        synthetic_event_id: &AtomicU64,
        entries: Vec<BlockEntry>,
    ) {
        let mut by_worker: BTreeMap<WorkerWithDpRank, BTreeSet<ExternalSequenceBlockHash>> =
            BTreeMap::new();
        for entry in entries {
            by_worker.entry(entry.worker).or_default().insert(entry.key);
        }

        for (worker, hashes) in by_worker {
            let event_id = Self::next_synthetic_event_id(synthetic_event_id);
            let event = RouterEvent::new(
                worker.worker_id,
                KvCacheEvent {
                    event_id,
                    data: KvCacheEventData::Removed(KvCacheRemoveData {
                        block_hashes: hashes.into_iter().collect(),
                    }),
                    dp_rank: worker.dp_rank,
                },
            );
            let thread_idx = Self::get_or_assign_thread_idx(
                worker_assignments,
                worker_assignment_count,
                worker.worker_id,
                num_workers,
            );
            if let Err(error) = worker_event_channels[thread_idx].send(WorkerTask::Event(event)) {
                tracing::warn!(
                    thread_idx,
                    ?error,
                    "Failed to enqueue approximate TTL remove event"
                );
            }
        }
    }

    fn spawn_prune_pump(
        prune_manager: WorkerPruneManager,
        worker_event_channels: Vec<flume::Sender<WorkerTask>>,
        worker_assignments: Arc<DashMap<WorkerId, usize, FxBuildHasher>>,
        worker_assignment_count: Arc<AtomicUsize>,
        num_workers: usize,
        synthetic_event_id: Arc<AtomicU64>,
        cancel: tokio_util::sync::CancellationToken,
    ) {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let mut ready_rx = prune_manager.subscribe_ready();
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => break,
                        changed = ready_rx.changed() => {
                            if changed.is_err() {
                                break;
                            }
                            loop {
                                let entries = prune_manager.drain_pending_removes();
                                if entries.is_empty() {
                                    break;
                                }
                                Self::enqueue_prune_removes(
                                    &worker_event_channels,
                                    &worker_assignments,
                                    &worker_assignment_count,
                                    num_workers,
                                    &synthetic_event_id,
                                    entries,
                                );
                            }
                        }
                    }
                }
            });
        }
    }

    fn maybe_enqueue_cleanup(&self, thread_idx: usize) {
        if !self.backend.try_schedule_cleanup() {
            return;
        }

        if let Err(e) =
            self.worker_event_channels[thread_idx].send(WorkerTask::CleanupStaleChildren)
        {
            self.backend.cancel_scheduled_cleanup();
            tracing::error!(
                "Failed to send cleanup task to worker thread {}: {:?}",
                thread_idx,
                e
            );
        }
    }
}

impl<T: SyncIndexer> Drop for ThreadPoolIndexer<T> {
    fn drop(&mut self) {
        if let Some(cancel) = &self.prune_pump_cancel {
            cancel.cancel();
        }
        if let Some(prune_manager) = &self.prune_manager {
            prune_manager.shutdown();
        }

        // Send Terminate to all worker threads so they exit their recv loops
        // and drop their Arc<T> clones. Then join the threads to ensure the
        // clones are actually dropped before the compiler drops `self.backend`.
        // Without this, worker threads may still be alive when `backend` drops,
        // keeping the Arc refcount > 0 and preventing T::drop() from running.
        for channel in self.worker_event_channels.iter() {
            let _ = channel.send(WorkerTask::Terminate);
        }
        let handles = std::mem::take(
            &mut *self
                .thread_handles
                .lock()
                .expect("thread_handles mutex poisoned"),
        );
        for handle in handles {
            let _ = handle.join();
        }
    }
}

#[async_trait]
impl<T: SyncIndexer> KvIndexerInterface for ThreadPoolIndexer<T> {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        // Execute inline on caller's thread - no channel dispatch
        Ok(self.backend.find_matches(&sequence, false))
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
                ..Default::default()
            },
        );
        Ok(self.backend.find_matches(&sequence, false))
    }

    async fn apply_event(&self, event: RouterEvent) {
        let worker_id = event.worker_id;

        // Get or assign worker thread index using sticky round-robin
        let thread_idx = Self::get_or_assign_thread_idx(
            &self.worker_assignments,
            &self.worker_assignment_count,
            worker_id,
            self.num_workers,
        );

        // Send event to the assigned worker thread
        if let Err(e) = self.worker_event_channels[thread_idx].send(WorkerTask::Event(event)) {
            tracing::error!(
                "Failed to send event to worker thread {}: {:?}",
                thread_idx,
                e
            );
            return;
        }

        self.maybe_enqueue_cleanup(thread_idx);
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        if let Some(prune_manager) = &self.prune_manager {
            prune_manager.remove_worker(worker_id);
        }

        // Route to the worker's assigned thread (if any), otherwise broadcast
        // to all threads since dp_ranks may be spread across threads.
        let thread_idx = self.worker_assignments.get(&worker_id).map(|v| *v);
        match thread_idx {
            Some(idx) => {
                if let Err(e) =
                    self.worker_event_channels[idx].send(WorkerTask::RemoveWorker(worker_id))
                {
                    tracing::error!(
                        "Failed to send RemoveWorker to worker thread {}: {:?}",
                        idx,
                        e
                    );
                    return;
                }

                self.maybe_enqueue_cleanup(idx);
            }
            None => {
                // Worker was never assigned a thread - broadcast to all
                for channel in &self.worker_event_channels {
                    let _ = channel.send(WorkerTask::RemoveWorker(worker_id));
                }
                self.maybe_enqueue_cleanup(0);
            }
        }
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        if let Some(prune_manager) = &self.prune_manager {
            prune_manager.remove_worker_dp_rank(WorkerWithDpRank::new(worker_id, dp_rank));
        }

        // Broadcast to all threads — the dp_rank may be on any thread.
        // Don't remove from worker_assignments since other dp_ranks may still exist.
        for channel in &self.worker_event_channels {
            let _ = channel.send(WorkerTask::RemoveWorkerDpRank(worker_id, dp_rank));
        }
        self.maybe_enqueue_cleanup(0);
    }

    fn shutdown(&self) {
        if let Some(cancel) = &self.prune_pump_cancel {
            cancel.cancel();
        }
        if let Some(prune_manager) = &self.prune_manager {
            prune_manager.shutdown();
        }

        // Send shutdown signal to all worker threads
        for channel in self.worker_event_channels.iter() {
            let _ = channel.send(WorkerTask::Terminate);
        }

        // Take ownership of thread handles and join them
        let handles = std::mem::take(
            &mut *self
                .thread_handles
                .lock()
                .expect("thread_handles mutex poisoned"),
        );
        for handle in handles {
            if let Err(e) = handle.join() {
                tracing::error!("Worker thread panicked during shutdown: {:?}", e);
            }
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        // Send DumpEvents to every worker as a FIFO barrier: each worker must
        // finish processing all previously queued Events before it handles
        // DumpEvents, so by the time all workers respond we know the shared
        // tree (if any) reflects every event that was enqueued before this call.
        let mut receivers = Vec::new();

        for channel in &self.worker_event_channels {
            let (resp_tx, resp_rx) = oneshot::channel::<anyhow::Result<Vec<RouterEvent>>>();
            let dump_req = WorkerTask::DumpEvents(resp_tx);

            channel
                .send(dump_req)
                .map_err(|_| KvRouterError::IndexerOffline)?;
            receivers.push(resp_rx);
        }

        let mut all_events = Vec::new();
        let mut event_id_counter = 0u64;

        for resp_rx in receivers {
            let mut events = resp_rx
                .await
                .map_err(|_| KvRouterError::IndexerDroppedRequest)?
                .map_err(|_| KvRouterError::IndexerOffline)?;
            for event in &mut events {
                event.event.event_id = event_id_counter;
                event_id_counter += 1;
            }
            all_events.extend(events);
        }

        // Shared-state backends keep their tree in concurrent structures
        // readable from any thread. Now that the barrier above guarantees
        // all queued writes have landed, dump directly.
        if let Some(events) = self.backend.dump_events() {
            return Ok(events);
        }

        // Per-thread-state backends returned their events through the DumpEvents
        // responses collected above.
        Ok(all_events)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let Some(prune_manager) = &self.prune_manager else {
            // Approximate routing decisions are only recorded when explicitly enabled.
            return Ok(());
        };

        tokens_with_hashes.get_or_compute_seq_hashes();
        let local_hashes = tokens_with_hashes
            .block_hashes()
            .expect("block hashes missing after computing sequence hashes");
        let sequence_hashes = tokens_with_hashes
            .seq_hashes()
            .expect("sequence hashes missing after computing sequence hashes");
        let event_id = Self::next_synthetic_event_id(&self.synthetic_event_id);
        let event = Self::stored_event_for_hashes(worker, local_hashes, sequence_hashes, event_id);
        let prune_entries = Self::block_entries_for_hashes(worker, sequence_hashes);
        let thread_idx = Self::get_or_assign_thread_idx(
            &self.worker_assignments,
            &self.worker_assignment_count,
            worker.worker_id,
            self.num_workers,
        );

        let (resp_tx, resp_rx) = oneshot::channel();
        self.worker_event_channels[thread_idx]
            .send(WorkerTask::EventWithAck {
                event,
                resp: resp_tx,
            })
            .map_err(|_| KvRouterError::IndexerOffline)?;

        let applied = resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)?;
        if applied {
            prune_manager.insert_worker_block_entries(worker, prune_entries);
        }

        Ok(())
    }

    async fn flush(&self) -> usize {
        let curr_size: usize = self.worker_event_channels.iter().map(|ch| ch.len()).sum();
        self.flush().await;

        if let Some(prune_manager) = &self.prune_manager {
            let entries = prune_manager.drain_due_and_pending(tokio::time::Instant::now());
            Self::enqueue_prune_removes(
                &self.worker_event_channels,
                &self.worker_assignments,
                &self.worker_assignment_count,
                self.num_workers,
                &self.synthetic_event_id,
                entries,
            );
            self.flush().await;

            let entries = prune_manager.drain_pending_removes();
            let has_entries = !entries.is_empty();
            Self::enqueue_prune_removes(
                &self.worker_event_channels,
                &self.worker_assignments,
                &self.worker_assignment_count,
                self.num_workers,
                &self.synthetic_event_id,
                entries,
            );
            if has_entries {
                self.flush().await;
            }
        }
        curr_size
    }

    fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        vec![ShardSizeSnapshot {
            shard_idx: 0,
            worker_count: self.backend.worker_count(),
            block_count: self.backend.block_count(),
            node_count: self.backend.node_count(),
        }]
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        self.backend.node_edge_lengths()
    }
}
