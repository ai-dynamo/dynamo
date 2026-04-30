// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    sync::{Arc, Mutex, atomic::AtomicUsize},
    thread::JoinHandle,
};

use async_trait::async_trait;
use dashmap::DashMap;
use rustc_hash::FxBuildHasher;
use tokio::sync::oneshot;

use super::{
    KvIndexerInterface, KvIndexerMetrics, KvRouterError, ShardSizeSnapshot, SyncIndexer, WorkerTask,
};
use crate::protocols::*;

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
    worker_assignments: DashMap<WorkerId, usize, FxBuildHasher>,
    /// Counter for round-robin assignment of new WorkerIds.
    worker_assignment_count: AtomicUsize,

    /// Channels to send tasks to worker threads (one per thread).
    /// Sending `WorkerTask::Terminate` signals the thread to shut down.
    worker_event_channels: Vec<flume::Sender<WorkerTask>>,

    /// Number of worker threads.
    num_workers: usize,
    /// Block size for KV cache.
    kv_block_size: u32,

    /// Handles to worker threads for joining on shutdown.
    thread_handles: Mutex<Vec<JoinHandle<()>>>,
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
        assert!(num_workers > 0, "Number of workers must be greater than 0");
        super::warn_on_unit_block_size("thread_pool", kv_block_size);

        let backend = Arc::new(backend);
        let mut worker_event_senders = Vec::new();
        let mut thread_handles = Vec::new();
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

        Self {
            backend,
            worker_assignments: DashMap::with_hasher(FxBuildHasher),
            worker_assignment_count: AtomicUsize::new(0),
            worker_event_channels: worker_event_senders,
            num_workers,
            kv_block_size,
            thread_handles: Mutex::new(thread_handles),
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
        let thread_idx = *self.worker_assignments.entry(worker_id).or_insert_with(|| {
            let idx = self
                .worker_assignment_count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            idx % self.num_workers
        });

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
        // Broadcast to all threads — the dp_rank may be on any thread.
        // Don't remove from worker_assignments since other dp_ranks may still exist.
        for channel in &self.worker_event_channels {
            let _ = channel.send(WorkerTask::RemoveWorkerDpRank(worker_id, dp_rank));
        }
        self.maybe_enqueue_cleanup(0);
    }

    fn shutdown(&self) {
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
        _tokens_with_hashes: &mut TokensWithHashes,
        _worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // No-op: pruning not supported in ThreadPoolIndexer
        Ok(())
    }

    async fn flush(&self) -> usize {
        let curr_size: usize = self.worker_event_channels.iter().map(|ch| ch.len()).sum();
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
