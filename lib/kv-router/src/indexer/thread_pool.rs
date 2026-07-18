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

#[cfg(feature = "bench")]
use super::{
    EventCompletionBuffer, EventCompletionWriter, ObservationError, ObservationSeal,
    ObservedEnqueueReceipt, ThreadPoolObservationPlan, ThreadPoolObservationSnapshot,
};
use super::{
    KvIndexerInterface, KvIndexerMetrics, KvRouterError, ShardSizeSnapshot, SyncIndexer,
    WorkerLookupStats, WorkerTask, panic_payload_message,
};
use crate::indexer::pruning::{BlockEntry, PruneConfig, WorkerPruneManager};
use crate::protocols::*;
use dynamo_tokens::SequenceHash;
#[cfg(feature = "bench")]
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

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

    /// Maps WorkerId to worker thread index for indexer-lifetime sticky routing.
    ///
    /// This is not a live-worker registry: entries intentionally remain for the
    /// indexer's lifetime. Producers read this assignment before enqueueing, so
    /// removing an entry without a generation-aware handoff could let later work
    /// overtake tasks already queued on the old thread.
    worker_assignments: Arc<DashMap<WorkerId, usize, FxBuildHasher>>,
    /// Monotonic round-robin reservation sequence for new WorkerIds.
    ///
    /// This is not an active-worker count: cold removals also reserve a slot so
    /// the removal sweep and any subsequent restore share one FIFO.
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

    #[cfg(feature = "bench")]
    observation_active: AtomicBool,
}

#[cfg(feature = "bench")]
struct ObservationLease<'a> {
    active: &'a AtomicBool,
}

#[cfg(feature = "bench")]
impl Drop for ObservationLease<'_> {
    fn drop(&mut self) {
        self.active.store(false, AtomicOrdering::Release);
    }
}

#[cfg(feature = "bench")]
pub struct ThreadPoolObservationSession<'a, T: SyncIndexer> {
    indexer: &'a ThreadPoolIndexer<T>,
    epoch: std::time::Instant,
    seal_tasks: Vec<Option<WorkerTask>>,
    seal_receivers: Vec<oneshot::Receiver<Option<ObservationSeal>>>,
    harvest_tasks: Vec<Option<WorkerTask>>,
    harvest_receivers: Vec<oneshot::Receiver<EventCompletionBuffer>>,
    queue_depth_at_stop: Vec<usize>,
    lease: ObservationLease<'a>,
}

#[cfg(feature = "bench")]
pub struct ThreadPoolObservationDrain<'a, T: SyncIndexer> {
    indexer: &'a ThreadPoolIndexer<T>,
    seal_tasks: Vec<Option<WorkerTask>>,
    seal_receivers: Vec<oneshot::Receiver<Option<ObservationSeal>>>,
    harvest_tasks: Vec<Option<WorkerTask>>,
    harvest_receivers: Vec<oneshot::Receiver<EventCompletionBuffer>>,
    queue_depth_at_stop: Vec<usize>,
    lease: ObservationLease<'a>,
}

#[cfg(feature = "bench")]
pub struct ThreadPoolSealedObservation<'a, T: SyncIndexer> {
    indexer: &'a ThreadPoolIndexer<T>,
    seals: Vec<ObservationSeal>,
    harvest_tasks: Vec<Option<WorkerTask>>,
    harvest_receivers: Vec<oneshot::Receiver<EventCompletionBuffer>>,
    queue_depth_at_stop: Vec<usize>,
    lease: ObservationLease<'a>,
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
        mut backend: T,
        num_workers: usize,
        kv_block_size: u32,
        metrics: Option<Arc<KvIndexerMetrics>>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        assert!(num_workers > 0, "Number of workers must be greater than 0");
        assert!(
            prune_config.is_none() || backend.supports_routing_decision_pruning(),
            "backend does not support routing-decision pruning"
        );
        super::warn_on_unit_block_size("thread_pool", kv_block_size);
        backend.configure_metrics(metrics.as_deref());

        let backend = Arc::new(backend);
        let mut worker_event_senders = Vec::new();
        let mut thread_handles = Vec::new();
        let worker_assignments = Arc::new(DashMap::with_hasher(FxBuildHasher));
        let worker_assignment_count = Arc::new(AtomicUsize::new(0));
        let synthetic_event_id = Arc::new(AtomicU64::new(0));
        for worker_idx in 0..num_workers {
            let (event_sender, event_receiver) = flume::unbounded::<WorkerTask>();
            worker_event_senders.push(event_sender);

            let backend = Arc::clone(&backend);
            let metrics = metrics.clone();

            let handle = std::thread::spawn(move || {
                // This is observability, not recovery: if the worker panics, log
                // through tracing and then preserve the panic for join().
                let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if let Err(error) = backend.worker(event_receiver, metrics) {
                        tracing::error!(
                            worker_thread_index = worker_idx,
                            ?error,
                            "Thread pool worker exited with an error; worker thread is now dead"
                        );
                    }
                }));

                if let Err(panic_payload) = panic_result {
                    let panic_msg = panic_payload_message(&*panic_payload);
                    tracing::error!(
                        target: "dynamo_kv_router::thread_pool_worker_panic",
                        worker_thread_index = worker_idx,
                        panic_message = %panic_msg,
                        "Thread pool worker panicked; worker thread is now dead"
                    );
                    std::panic::resume_unwind(panic_payload);
                }
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
            #[cfg(feature = "bench")]
            observation_active: AtomicBool::new(false),
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

    #[cfg(feature = "bench")]
    pub async fn begin_observation(
        &self,
        plan: ThreadPoolObservationPlan,
    ) -> Result<ThreadPoolObservationSession<'_, T>, ObservationError> {
        if plan.expected_events_by_worker.is_empty() {
            return Err(ObservationError::EmptyPlan);
        }
        if self
            .observation_active
            .compare_exchange(false, true, AtomicOrdering::AcqRel, AtomicOrdering::Acquire)
            .is_err()
        {
            return Err(ObservationError::AlreadyActive);
        }
        let lease = ObservationLease {
            active: &self.observation_active,
        };

        let mut seen = BTreeSet::new();
        let mut expected_events_per_queue = vec![0usize; self.num_workers];
        for &(worker_id, expected_events) in &plan.expected_events_by_worker {
            if !seen.insert(worker_id) {
                return Err(ObservationError::DuplicateWorker(worker_id));
            }
            let worker_idx = Self::get_or_assign_thread_idx(
                &self.worker_assignments,
                &self.worker_assignment_count,
                worker_id,
                self.num_workers,
            );
            expected_events_per_queue[worker_idx] = expected_events_per_queue[worker_idx]
                .checked_add(expected_events)
                .ok_or(ObservationError::CapacityOverflow(worker_idx))?;
        }

        let mut install_receivers = Vec::with_capacity(self.num_workers);
        let mut seal_tasks = Vec::with_capacity(self.num_workers);
        let mut seal_receivers = Vec::with_capacity(self.num_workers);
        let mut harvest_tasks = Vec::with_capacity(self.num_workers);
        let mut harvest_receivers = Vec::with_capacity(self.num_workers);

        for (worker_idx, &capacity) in expected_events_per_queue.iter().enumerate() {
            let (install_tx, install_rx) = oneshot::channel();
            self.worker_event_channels[worker_idx]
                .send(WorkerTask::InstallObservation {
                    writer: EventCompletionWriter::new(plan.epoch, capacity),
                    resp: install_tx,
                })
                .map_err(|_| ObservationError::WorkerOffline(worker_idx))?;
            install_receivers.push(install_rx);

            let (seal_tx, seal_rx) = oneshot::channel();
            seal_tasks.push(Some(WorkerTask::SealObservation(seal_tx)));
            seal_receivers.push(seal_rx);

            let (harvest_tx, harvest_rx) = oneshot::channel();
            harvest_tasks.push(Some(WorkerTask::HarvestObservation(harvest_tx)));
            harvest_receivers.push(harvest_rx);
        }

        for (worker_idx, receiver) in install_receivers.into_iter().enumerate() {
            let accepted = receiver
                .await
                .map_err(|_| ObservationError::InstallCanceled(worker_idx))?;
            if !accepted {
                return Err(ObservationError::InstallRejected(worker_idx));
            }
        }

        Ok(ThreadPoolObservationSession {
            indexer: self,
            epoch: plan.epoch,
            seal_tasks,
            seal_receivers,
            harvest_tasks,
            harvest_receivers,
            queue_depth_at_stop: vec![0; self.num_workers],
            lease,
        })
    }

    #[cfg(feature = "bench")]
    #[doc(hidden)]
    pub fn enqueue_active_observation_owned(
        &self,
        event: RouterEvent,
        correlation_id: u32,
        epoch: std::time::Instant,
    ) -> Result<ObservedEnqueueReceipt, ObservationError> {
        if !self.observation_active.load(AtomicOrdering::Acquire) {
            return Err(ObservationError::ProducerClosed);
        }
        let worker_id = event.worker_id;
        let worker_idx = self
            .worker_assignments
            .get(&worker_id)
            .map(|entry| *entry)
            .ok_or(ObservationError::UnknownWorker(worker_id))?;

        self.worker_event_channels[worker_idx]
            .send(WorkerTask::ObservedEvent {
                event,
                correlation_id,
            })
            .map_err(|_| ObservationError::WorkerOffline(worker_idx))?;
        let accepted_ns = epoch.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.maybe_enqueue_cleanup(worker_idx);

        Ok(ObservedEnqueueReceipt {
            accepted_ns,
            event_worker: worker_idx,
        })
    }

    #[cfg(feature = "bench")]
    fn capture_queue_depths(&self, depths: &mut [usize]) {
        for (depth, channel) in depths.iter_mut().zip(&self.worker_event_channels) {
            *depth = channel.len();
        }
    }

    pub(crate) async fn worker_lookup_stats(&self) -> WorkerLookupStats {
        let mut receivers = Vec::new();
        for channel in &self.worker_event_channels {
            let (resp_tx, resp_rx) = oneshot::channel();
            if channel.send(WorkerTask::Stats(resp_tx)).is_ok() {
                receivers.push(resp_rx);
            }
        }

        let mut worker_blocks = BTreeMap::new();
        for receiver in receivers {
            if let Ok(stats) = receiver.await {
                for (worker, block_count) in stats.worker_blocks {
                    *worker_blocks.entry(worker).or_insert(0usize) += block_count;
                }
            }
        }

        WorkerLookupStats::from_worker_block_counts(worker_blocks)
    }

    pub async fn get_workers(&self) -> Vec<WorkerId> {
        self.worker_lookup_stats()
            .await
            .worker_blocks
            .into_iter()
            .map(|(worker, _)| worker.worker_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Enqueue a structural anchor on the same worker queue used by normal
    /// events for this worker. Branch-sharded routing uses this to preserve
    /// Anchor-before-Stored ordering for the dependent suffix on that queue.
    pub(crate) fn enqueue_anchor(
        &self,
        worker: WorkerWithDpRank,
        anchor: super::AnchorTask,
    ) -> Result<(), KvRouterError> {
        let thread_idx = Self::get_or_assign_thread_idx(
            &self.worker_assignments,
            &self.worker_assignment_count,
            worker.worker_id,
            self.num_workers,
        );
        self.worker_event_channels[thread_idx]
            .send(WorkerTask::Anchor { worker, anchor })
            .map_err(|_| KvRouterError::IndexerOffline)?;
        self.maybe_enqueue_cleanup(thread_idx);
        Ok(())
    }

    /// Enqueue an event and report whether the worker queue accepted it.
    pub fn enqueue_event(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        let worker_id = event.worker_id;
        let thread_idx = Self::get_or_assign_thread_idx(
            &self.worker_assignments,
            &self.worker_assignment_count,
            worker_id,
            self.num_workers,
        );
        self.worker_event_channels[thread_idx]
            .send(WorkerTask::Event(event))
            .map_err(|_| KvRouterError::IndexerOffline)?;
        self.maybe_enqueue_cleanup(thread_idx);
        Ok(())
    }

    /// Wait until every worker queue has completed tasks accepted before this call.
    ///
    /// This is a FIFO progress barrier, not an event-result acknowledgement. Ordinary
    /// `WorkerTask::Event` application errors are logged by the worker and are not returned here.
    pub async fn flush_and_wait(&self) -> Result<(), KvRouterError> {
        let mut receivers = Vec::with_capacity(self.worker_event_channels.len());
        for channel in &self.worker_event_channels {
            let (resp_tx, resp_rx) = oneshot::channel();
            channel
                .send(WorkerTask::Flush(resp_tx))
                .map_err(|_| KvRouterError::IndexerOffline)?;
            receivers.push(resp_rx);
        }
        for receiver in receivers {
            receiver
                .await
                .map_err(|_| KvRouterError::IndexerDroppedRequest)?;
        }
        Ok(())
    }

    /// Wait until the FIFO lane for `worker_id` has completed tasks accepted before this call.
    ///
    /// NOTE: Worker-to-queue assignment is stable for the indexer's lifetime. Every DP rank and
    /// worker-wide `Cleared` mutation therefore shares this lane, so this barrier must never use a
    /// non-FIFO or recomputed assignment. It proves queue progress only: ordinary
    /// `WorkerTask::Event` application errors are logged by the worker and are not returned here.
    pub async fn flush_worker_and_wait(&self, worker_id: WorkerId) -> Result<(), KvRouterError> {
        let thread_idx = Self::get_or_assign_thread_idx(
            &self.worker_assignments,
            &self.worker_assignment_count,
            worker_id,
            self.num_workers,
        );
        let (resp_tx, resp_rx) = oneshot::channel();
        self.worker_event_channels[thread_idx]
            .send(WorkerTask::Flush(resp_tx))
            .map_err(|_| KvRouterError::IndexerOffline)?;
        resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)
    }

    /// Wait until all previously queued worker tasks have completed.
    ///
    /// Used primarily for testing and benchmarking to stop at a stable queue boundary before
    /// checking results. Individual event-application errors are observable only through logs and
    /// metrics, not this barrier.
    pub async fn flush(&self) {
        if let Err(error) = self.flush_and_wait().await {
            tracing::error!(%error, "Failed to flush thread-pool indexer");
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

    fn enqueue_worker_removal(&self, worker_id: WorkerId, task: WorkerTask) {
        // All mutations for one WorkerId use this indexer-lifetime assignment,
        // so the removal and any subsequent restore share one FIFO.
        let thread_idx = Self::get_or_assign_thread_idx(
            &self.worker_assignments,
            &self.worker_assignment_count,
            worker_id,
            self.num_workers,
        );

        if let Err(error) = self.worker_event_channels[thread_idx].send(task) {
            tracing::error!(
                worker_id,
                worker_thread_index = thread_idx,
                ?error,
                "Failed to send worker removal task"
            );
            return;
        }

        self.maybe_enqueue_cleanup(thread_idx);
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

    async fn record_routing_decision_hashes(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: &[LocalBlockHash],
        sequence_hashes: &[SequenceHash],
    ) -> Result<(), KvRouterError> {
        if local_hashes.len() != sequence_hashes.len() {
            tracing::warn!(
                local_len = local_hashes.len(),
                sequence_len = sequence_hashes.len(),
                "Mismatched routing-decision hash lengths"
            );
            return Err(KvRouterError::IndexerDroppedRequest);
        }

        let Some(prune_manager) = &self.prune_manager else {
            // Approximate routing decisions are only recorded when explicitly enabled.
            return Ok(());
        };

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

    pub async fn process_routing_decision_with_hashes(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        self.record_routing_decision_hashes(worker, &local_hashes, &sequence_hashes)
            .await
    }

    pub async fn process_routing_decision_hash_slices(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: &[LocalBlockHash],
        sequence_hashes: &[SequenceHash],
    ) -> Result<(), KvRouterError> {
        self.record_routing_decision_hashes(worker, local_hashes, sequence_hashes)
            .await
    }
}

#[cfg(feature = "bench")]
impl<'a, T: SyncIndexer> ThreadPoolObservationSession<'a, T> {
    pub fn enqueue_observed_owned(
        &mut self,
        event: RouterEvent,
        correlation_id: u32,
    ) -> Result<ObservedEnqueueReceipt, ObservationError> {
        self.indexer
            .enqueue_active_observation_owned(event, correlation_id, self.epoch)
    }

    pub fn close_observed_producers(mut self) -> ThreadPoolObservationDrain<'a, T> {
        self.indexer
            .capture_queue_depths(&mut self.queue_depth_at_stop);

        ThreadPoolObservationDrain {
            indexer: self.indexer,
            seal_tasks: self.seal_tasks,
            seal_receivers: self.seal_receivers,
            harvest_tasks: self.harvest_tasks,
            harvest_receivers: self.harvest_receivers,
            queue_depth_at_stop: self.queue_depth_at_stop,
            lease: self.lease,
        }
    }
}

#[cfg(feature = "bench")]
impl<'a, T: SyncIndexer> ThreadPoolObservationDrain<'a, T> {
    pub async fn seal(mut self) -> Result<ThreadPoolSealedObservation<'a, T>, ObservationError> {
        for worker_idx in 0..self.indexer.num_workers {
            let Some(task) = self.seal_tasks[worker_idx].take() else {
                return Err(ObservationError::SealTaskConsumed(worker_idx));
            };
            self.indexer.worker_event_channels[worker_idx]
                .send(task)
                .map_err(|_| ObservationError::WorkerOffline(worker_idx))?;
        }

        let mut seals = Vec::with_capacity(self.indexer.num_workers);
        for (worker_idx, receiver) in self.seal_receivers.into_iter().enumerate() {
            let seal = receiver
                .await
                .map_err(|_| ObservationError::SealCanceled(worker_idx))?
                .ok_or(ObservationError::SealRejected(worker_idx))?;
            seals.push(seal);
        }

        Ok(ThreadPoolSealedObservation {
            indexer: self.indexer,
            seals,
            harvest_tasks: self.harvest_tasks,
            harvest_receivers: self.harvest_receivers,
            queue_depth_at_stop: self.queue_depth_at_stop,
            lease: self.lease,
        })
    }
}

#[cfg(feature = "bench")]
impl<T: SyncIndexer> ThreadPoolSealedObservation<'_, T> {
    pub fn latest_seal_ns(&self) -> u64 {
        self.seals
            .iter()
            .map(|seal| seal.sealed_ns)
            .max()
            .unwrap_or_default()
    }

    pub async fn harvest(mut self) -> Result<ThreadPoolObservationSnapshot, ObservationError> {
        for worker_idx in 0..self.indexer.num_workers {
            let Some(task) = self.harvest_tasks[worker_idx].take() else {
                return Err(ObservationError::HarvestTaskConsumed(worker_idx));
            };
            self.indexer.worker_event_channels[worker_idx]
                .send(task)
                .map_err(|_| ObservationError::WorkerOffline(worker_idx))?;
        }

        let mut buffers = Vec::with_capacity(self.indexer.num_workers);
        for (worker_idx, receiver) in self.harvest_receivers.into_iter().enumerate() {
            buffers.push(
                receiver
                    .await
                    .map_err(|_| ObservationError::HarvestCanceled(worker_idx))?,
            );
        }

        let snapshot = ThreadPoolObservationSnapshot {
            seals: self.seals,
            buffers,
            queue_depth_at_stop: self.queue_depth_at_stop,
        };
        drop(self.lease);
        Ok(snapshot)
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
        cache_namespace: Option<&str>,
        is_eagle: Option<bool>,
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(
            tokens,
            self.kv_block_size,
            BlockHashOptions {
                lora_name,
                cache_namespace,
                is_eagle,
                ..Default::default()
            },
        );
        Ok(self.backend.find_matches(&sequence, false))
    }

    async fn apply_event(&self, event: RouterEvent) {
        if let Err(error) = self.enqueue_event(event) {
            tracing::error!(%error, "Failed to enqueue event to thread-pool indexer");
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        if let Some(prune_manager) = &self.prune_manager {
            prune_manager.remove_worker(worker_id);
        }

        self.enqueue_worker_removal(
            worker_id,
            WorkerTask::RemoveWorker {
                worker_id,
                sweep_tree: true,
            },
        );
    }

    async fn remove_worker_dp_rank(&self, worker_id: WorkerId, dp_rank: DpRank) {
        if let Some(prune_manager) = &self.prune_manager {
            prune_manager.remove_worker_dp_rank(WorkerWithDpRank::new(worker_id, dp_rank));
        }

        self.enqueue_worker_removal(
            worker_id,
            WorkerTask::RemoveWorkerDpRank {
                worker_id,
                dp_rank,
                sweep_tree: true,
            },
        );
    }

    async fn reset_worker_dp_rank_and_wait(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> Result<(), KvRouterError> {
        if let Some(prune_manager) = &self.prune_manager {
            prune_manager.remove_worker_dp_rank(WorkerWithDpRank::new(worker_id, dp_rank));
        }

        let thread_idx = Self::get_or_assign_thread_idx(
            &self.worker_assignments,
            &self.worker_assignment_count,
            worker_id,
            self.num_workers,
        );
        self.worker_event_channels[thread_idx]
            .send(WorkerTask::RemoveWorkerDpRank {
                worker_id,
                dp_rank,
                sweep_tree: true,
            })
            .map_err(|_| KvRouterError::IndexerOffline)?;

        self.flush_worker_and_wait(worker_id).await
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
        if !self.backend.supports_event_dump() {
            return Err(KvRouterError::Unsupported(
                "backend cannot reconstruct router events".to_string(),
            ));
        }

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
        tokens_with_hashes.get_or_compute_seq_hashes();
        let local_hashes = tokens_with_hashes
            .block_hashes()
            .expect("block hashes missing after computing sequence hashes");
        let sequence_hashes = tokens_with_hashes
            .seq_hashes()
            .expect("sequence hashes missing after computing sequence hashes");
        self.record_routing_decision_hashes(worker, local_hashes, sequence_hashes)
            .await
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

    fn timing_report(&self) -> String {
        self.backend.timing_report()
    }

    async fn shard_sizes(&self) -> Vec<ShardSizeSnapshot> {
        let stats = self.worker_lookup_stats().await;
        vec![ShardSizeSnapshot {
            shard_idx: 0,
            worker_count: stats.worker_count(),
            block_count: stats.block_count(),
            node_count: self.backend.node_count(),
        }]
    }

    fn node_edge_lengths(&self) -> Vec<usize> {
        self.backend.node_edge_lengths()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ConcurrentRadixTreeCompressed,
        test_utils::{assert_score, make_store_event},
    };
    use std::time::Duration;

    struct BlockingBackend {
        blocked_worker: WorkerId,
        entered: flume::Sender<()>,
        release: flume::Receiver<()>,
    }

    impl SyncIndexer for BlockingBackend {
        fn worker(
            &self,
            event_receiver: flume::Receiver<WorkerTask>,
            _metrics: Option<Arc<KvIndexerMetrics>>,
        ) -> anyhow::Result<()> {
            for task in event_receiver {
                match task {
                    WorkerTask::Event(event) if event.worker_id == self.blocked_worker => {
                        self.entered.send(())?;
                        self.release.recv()?;
                    }
                    WorkerTask::Flush(response) => {
                        let _ = response.send(());
                    }
                    WorkerTask::Terminate => break,
                    _ => {}
                }
            }
            Ok(())
        }

        fn find_matches(&self, _sequence: &[LocalBlockHash], _early_exit: bool) -> OverlapScores {
            OverlapScores::default()
        }
    }

    fn blocking_indexer(
        blocked_worker: WorkerId,
    ) -> (
        ThreadPoolIndexer<BlockingBackend>,
        flume::Receiver<()>,
        flume::Sender<()>,
    ) {
        let (entered_tx, entered_rx) = flume::bounded(1);
        let (release_tx, release_rx) = flume::bounded(1);
        (
            ThreadPoolIndexer::new(
                BlockingBackend {
                    blocked_worker,
                    entered: entered_tx,
                    release: release_rx,
                },
                2,
                16,
            ),
            entered_rx,
            release_tx,
        )
    }

    fn assigned_thread(
        indexer: &ThreadPoolIndexer<ConcurrentRadixTreeCompressed>,
        worker_id: WorkerId,
    ) -> Option<usize> {
        indexer
            .worker_assignments
            .get(&worker_id)
            .map(|entry| *entry)
    }

    enum ColdRemoval {
        Worker,
        DpRank,
    }

    async fn assert_cold_remove_reserves_sticky_queue(removal: ColdRemoval) {
        let indexer = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 16);
        indexer.apply_event(make_store_event(1, &[1])).await;
        assert_eq!(assigned_thread(&indexer, 1), Some(0));

        match removal {
            ColdRemoval::Worker => indexer.remove_worker(2).await,
            ColdRemoval::DpRank => indexer.remove_worker_dp_rank(2, 0).await,
        }

        assert_eq!(assigned_thread(&indexer, 2), Some(1));
        indexer.apply_event(make_store_event(2, &[2])).await;
        indexer.flush().await;
        assert_eq!(assigned_thread(&indexer, 2), Some(1));
        assert_score(&indexer, &[2], WorkerWithDpRank::new(2, 0), 1).await;
    }

    #[tokio::test]
    async fn cold_worker_remove_reserves_sticky_queue() {
        assert_cold_remove_reserves_sticky_queue(ColdRemoval::Worker).await;
    }

    #[tokio::test]
    async fn cold_rank_remove_reserves_sticky_queue() {
        assert_cold_remove_reserves_sticky_queue(ColdRemoval::DpRank).await;
    }

    #[tokio::test]
    async fn worker_flush_waits_for_its_fifo_lane() {
        let (indexer, entered, release) = blocking_indexer(1);
        indexer.enqueue_event(make_store_event(1, &[1])).unwrap();
        entered.recv_async().await.unwrap();

        let mut flush = Box::pin(indexer.flush_worker_and_wait(1));
        let completed_early = tokio::time::timeout(Duration::from_millis(25), &mut flush).await;
        release.send(()).unwrap();
        if completed_early.is_err() {
            flush.await.unwrap();
        }
        assert!(
            completed_early.is_err(),
            "flush passed its blocked FIFO event"
        );
    }

    #[tokio::test]
    async fn worker_flush_ignores_an_unrelated_blocked_lane() {
        let (indexer, entered, release) = blocking_indexer(2);
        indexer.enqueue_event(make_store_event(1, &[1])).unwrap();
        indexer.enqueue_event(make_store_event(2, &[2])).unwrap();
        entered.recv_async().await.unwrap();

        let result =
            tokio::time::timeout(Duration::from_secs(1), indexer.flush_worker_and_wait(1)).await;
        release.send(()).unwrap();
        result
            .expect("worker flush waited on an unrelated lane")
            .unwrap();
    }

    #[cfg(feature = "bench")]
    #[tokio::test]
    async fn observed_events_use_fixed_worker_buffers_and_fifo_seals() {
        let indexer = ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 2, 16);
        let epoch = std::time::Instant::now();
        let mut observation = indexer
            .begin_observation(ThreadPoolObservationPlan {
                epoch,
                expected_events_by_worker: vec![(1, 1), (2, 1)],
            })
            .await
            .unwrap();

        let first = observation
            .enqueue_observed_owned(make_store_event(1, &[11]), 10)
            .unwrap();
        let second = observation
            .enqueue_observed_owned(make_store_event(2, &[22]), 20)
            .unwrap();
        assert_ne!(first.event_worker, second.event_worker);

        let sealed = observation.close_observed_producers().seal().await.unwrap();
        assert!(sealed.latest_seal_ns() >= first.accepted_ns);
        let snapshot = sealed.harvest().await.unwrap();
        assert_eq!(snapshot.buffers.len(), 2);
        assert!(snapshot.buffers.iter().all(|buffer| !buffer.overflowed()));
        let mut ids = snapshot
            .buffers
            .iter()
            .flat_map(|buffer| buffer.records().iter())
            .map(|record| record.correlation_id)
            .collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, [10, 20]);
    }
}
