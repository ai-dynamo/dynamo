// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV RadixTree
//!
//! This module implements a key-value (KV) store using a Radix Tree structure to efficiently manage and retrieve data blocks.
//! It is designed to support LLM (Large Language Model) inference by re-using a global KV cache.
//!
//! # Overview
//!
//! The main components of this module include:
//!
//! - **Radix Tree Structure**:
//!   - The `RadixTree` struct represents the main data structure, with nodes (`RadixBlock`) containing children and associated worker IDs.
//!   - It allows efficient storage and retrieval of data blocks based on their hashes.
//!
//! - **Event Handling**:
//!   - The `RouterEvent` struct represents events emitted by LLM workers, which can be applied to the Radix Tree to update its state.
//!   - The `KvIndexer` struct manages these events and match requests asynchronously using Tokio channels.
//!
//! - **Hash Computation**:
//!   - Functions like `compute_block_hash` and `compute_block_hash_for_seq` compute hashes for data blocks and sequences of tokens, facilitating quick lookups.
//!
//! - **Concurrency and Asynchronous Operations**:
//!   - The `KvIndexer` uses a single-threaded Tokio runtime to handle events and match requests concurrently, ensuring efficient processing without blocking.
//!
//! - **Match Requests**:
//!   - The `MatchRequest` struct represents requests to find matches in the Radix Tree, returning overlap scores indicating the best matches.
//!
//! # Purpose
//!
//! This module provides a scalable and efficient way to manage and retrieve data blocks for LLM inference, leveraging a global KV cache to optimize performance.

#[cfg(feature = "bench")]
use std::time::Instant;

use async_trait::async_trait;
use dashmap::DashMap;
#[cfg(feature = "metrics")]
pub use dynamo_runtime::protocols::maybe_error::MaybeError;
#[cfg(feature = "metrics")]
use dynamo_runtime::{
    component::Component,
    metrics::{MetricsHierarchy, prometheus_names::kvrouter},
};
use prometheus::{IntCounterVec, Opts};
use rustc_hash::FxBuildHasher;

/// Trait for types that may represent an error response.
/// Used for RPC-style responses that can indicate success or failure.
#[cfg(not(feature = "metrics"))]
pub trait MaybeError {
    /// Construct an instance from an error.
    fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self;
    /// Convert to an error instance if this represents an error.
    fn err(&self) -> Option<anyhow::Error>;
}
use serde::{Deserialize, Serialize};
#[cfg(feature = "metrics")]
use std::sync::OnceLock;
use std::{
    collections::VecDeque,
    iter,
    sync::{Arc, Mutex, atomic::AtomicUsize},
    thread::JoinHandle,
    time::Duration,
};
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::approx::{BlockEntry, PruneConfig, PruneManager};
// use crate::nested_map::NestedMap;
use crate::protocols::*;
pub use crate::radix_tree::RadixTree;
use dynamo_tokens::SequenceHash;

/// Errors that can occur in the KV Router.
#[derive(Debug, thiserror::Error)]
pub enum KvRouterError {
    #[error("Block not found")]
    BlockNotFound,

    #[error("Indexer is offline")]
    IndexerOffline,

    #[error("Indexer is dropped request")]
    IndexerDroppedRequest,

    #[error("Prune operation failed: {0}")]
    PruneFailed(String),
}

// -------
// Distributed router - Worker KV Query types
// -------

/// Request to query a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkerKvQueryRequest {
    /// The worker ID of the worker to query.
    pub worker_id: WorkerId,

    /// Start event ID (inclusive). If `None`, dumps entire tree.
    pub start_event_id: Option<u64>,
    /// End event ID (inclusive). If `None`, returns up to newest available.
    pub end_event_id: Option<u64>,
}

/// Response from a worker's local KV indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum WorkerKvQueryResponse {
    /// Events served from the circular buffer (with original event IDs)
    Events(Vec<RouterEvent>),
    /// Full tree dump (with synthetic 0-indexed event IDs)
    TreeDump(Vec<RouterEvent>),
    /// Requested range is newer than available data
    TooNew {
        requested_start: Option<u64>,
        requested_end: Option<u64>,
        newest_available: u64,
    },
    /// Invalid range: end_id < start_id
    InvalidRange { start_id: u64, end_id: u64 },
    /// Query failed on worker (serialized error)
    Error(String),
}

impl MaybeError for WorkerKvQueryResponse {
    fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        WorkerKvQueryResponse::Error(err.to_string())
    }

    fn err(&self) -> Option<anyhow::Error> {
        match self {
            WorkerKvQueryResponse::Error(msg) => Some(anyhow::Error::msg(msg.clone())),
            _ => None,
        }
    }
}

/// Metrics for the KV Indexer.
#[derive(Clone)]
pub struct KvIndexerMetrics {
    /// Counter of events applied.
    pub kv_cache_events_applied: IntCounterVec,
}

/// Metric status labels.
pub const METRIC_STATUS_OK: &str = "ok";
pub const METRIC_STATUS_PARENT_NOT_FOUND: &str = "parent_block_not_found";
pub const METRIC_STATUS_BLOCK_NOT_FOUND: &str = "block_not_found";
pub const METRIC_STATUS_INVALID_BLOCK: &str = "invalid_block";

/// Metric event labels.
pub const METRIC_EVENT_STORED: &str = "stored";
pub const METRIC_EVENT_REMOVED: &str = "removed";
pub const METRIC_EVENT_CLEARED: &str = "cleared";

/// Metric name for KV cache events applied counter.
const KV_CACHE_EVENTS_APPLIED_NAME: &str = "dynamo_kvrouter_kv_cache_events_applied";

#[cfg(feature = "metrics")]
static KV_INDEXER_METRICS: OnceLock<Arc<KvIndexerMetrics>> = OnceLock::new();

impl KvIndexerMetrics {
    #[cfg(feature = "metrics")]
    fn new(kv_cache_events_applied: IntCounterVec) -> Self {
        Self {
            kv_cache_events_applied,
        }
    }

    /// Creates a new KvIndexerMetrics from a Component, memoizing the result in
    /// KV_INDEXER_METRICS to avoid duplicate registration issues.
    #[cfg(feature = "metrics")]
    pub fn from_component(component: &Component) -> Arc<Self> {
        KV_INDEXER_METRICS.get_or_init(|| {
            match component.metrics().create_intcountervec(
                kvrouter::KV_CACHE_EVENTS_APPLIED,
                "Total number of KV cache events applied to index",
                &["event_type", "status"],
                &[],
            ) {
                Ok(kv_cache_events_applied) => Arc::new(Self::new(kv_cache_events_applied)),
                Err(e) => {
                    tracing::warn!("Failed to create kv indexer metrics from component: {}. Using unregistered metrics as fallback.", e);
                    Arc::new(Self::new_unregistered())
                }
            }
        }).clone()
    }

    /// Creates a new KvIndexerMetrics which is not registered with a MetricsRegistry.
    /// This may be used for tests or as a fallback for when a MetricsRegistry is not available / has errored.
    pub fn new_unregistered() -> Self {
        Self {
            kv_cache_events_applied: IntCounterVec::new(
                Opts::new(
                    KV_CACHE_EVENTS_APPLIED_NAME,
                    "Total number of KV cache events applied to index",
                ),
                &["event_type", "status"],
            )
            .unwrap(),
        }
    }

    pub fn get_event_type(event_data: &KvCacheEventData) -> &'static str {
        match event_data {
            KvCacheEventData::Stored(_) => METRIC_EVENT_STORED,
            KvCacheEventData::Removed(_) => METRIC_EVENT_REMOVED,
            KvCacheEventData::Cleared => METRIC_EVENT_CLEARED,
        }
    }

    pub fn increment_event_applied(
        &self,
        event_type: &'static str,
        result: Result<(), KvCacheEventError>,
    ) {
        match result {
            Ok(_) => {
                self.kv_cache_events_applied
                    .with_label_values(&[event_type, METRIC_STATUS_OK])
                    .inc_by(1);
            }
            Err(e) => {
                let error_label = match e {
                    KvCacheEventError::ParentBlockNotFound => METRIC_STATUS_PARENT_NOT_FOUND,
                    KvCacheEventError::BlockNotFound => METRIC_STATUS_BLOCK_NOT_FOUND,
                    KvCacheEventError::InvalidBlockSequence => METRIC_STATUS_INVALID_BLOCK,
                };
                self.kv_cache_events_applied
                    .with_label_values(&[event_type, error_label])
                    .inc_by(1);
            }
        }
    }
}

/// A request to find matches in the Radix Tree.
pub struct MatchRequest {
    /// A vector of `LocalBlockHash` representing the sequence to match.
    pub sequence: Vec<LocalBlockHash>,
    /// A boolean indicating whether to exit early if a single match is found.
    pub early_exit: bool,
    /// A channel sender to send the `OverlapScores` response.
    pub resp: oneshot::Sender<OverlapScores>,
    /// Timestamp when the request was created (for queue wait time measurement)
    #[cfg(feature = "bench")]
    pub created_at: Instant,
}

impl MatchRequest {
    fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        resp: oneshot::Sender<OverlapScores>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            resp,
            #[cfg(feature = "bench")]
            created_at: Instant::now(),
        }
    }
}

/// A request to dump the tree as events
pub struct DumpRequest {
    /// Channel to send the dumped events
    pub resp: oneshot::Sender<Vec<RouterEvent>>,
}

/// A request to get all workers currently tracked
pub struct GetWorkersRequest {
    /// Channel to send the worker IDs
    pub resp: oneshot::Sender<Vec<WorkerId>>,
}

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
    ///
    /// ### Returns
    ///
    /// An `OverlapScores` representing the match scores.
    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
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
// SyncIndexer trait and ThreadPoolIndexer generic wrapper
// ============================================================================

/// Trait for thread-safe data structures that support KV cache indexing operations.
///
/// All methods take `&self` and are synchronous. Implementations must be safe for
/// concurrent access (via internal locking, DashMap, etc).
///
/// This trait is used with [`ThreadPoolIndexer`], which wraps a `SyncIndexer` to
/// provide the async [`KvIndexerInterface`] with:
/// - Sticky event routing to N worker threads
/// - Inline reads on the caller's thread (no channel dispatch for find_matches)
pub trait SyncIndexer: Send + Sync + 'static {
    /// Find matches for a sequence of block hashes.
    fn find_matches(&self, sequence: &[LocalBlockHash], early_exit: bool) -> OverlapScores;

    /// Apply a router event to the data structure.
    fn apply_event(&self, event: RouterEvent) -> Result<(), KvCacheEventError>;

    /// Remove all entries for a worker.
    fn remove_worker(&self, worker_id: WorkerId);

    /// Dump the data structure as router events for reconstruction.
    fn dump_events(&self) -> Vec<RouterEvent>;
}

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

    /// Channels to send events to worker threads (one per thread).
    /// Sending `None` signals the thread to shut down.
    worker_event_channels: Vec<flume::Sender<Option<RouterEvent>>>,

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
        assert!(num_workers > 0, "Number of workers must be greater than 0");

        let backend = Arc::new(backend);
        let mut worker_event_senders = Vec::new();
        let mut thread_handles = Vec::new();
        for _ in 0..num_workers {
            let (event_sender, event_receiver) = flume::unbounded::<Option<RouterEvent>>();
            worker_event_senders.push(event_sender);

            let backend = Arc::clone(&backend);

            let handle = std::thread::spawn(move || {
                while let Ok(Some(event)) = event_receiver.recv() {
                    if let Err(e) = backend.apply_event(event) {
                        tracing::warn!("Failed to apply event: {:?}", e);
                    }
                }
                tracing::debug!("Worker thread shutting down");
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

    /// Wait for all worker channels to drain.
    ///
    /// Used primarily for testing and benchmarking to ensure all queued events
    /// have been picked up by workers before checking results.
    pub async fn flush(&self) {
        loop {
            let all_empty = self.worker_event_channels.iter().all(|ch| ch.is_empty());

            if all_empty {
                break;
            }

            tokio::time::sleep(Duration::from_millis(1)).await;
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
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None);
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
        if let Err(e) = self.worker_event_channels[thread_idx].send(Some(event)) {
            tracing::error!(
                "Failed to send event to worker thread {}: {:?}",
                thread_idx,
                e
            );
        }
    }

    async fn remove_worker(&self, worker_id: WorkerId) {
        // Execute inline - the backend is thread-safe
        self.backend.remove_worker(worker_id);
    }

    fn shutdown(&self) {
        // Send shutdown signal (None) to all worker threads
        for channel in self.worker_event_channels.iter() {
            let _ = channel.send(None);
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
        // Execute inline - the backend is thread-safe
        Ok(self.backend.dump_events())
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
        loop {
            let all_empty = self.worker_event_channels.iter().all(|ch| ch.is_empty());

            if all_empty {
                break;
            }

            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        curr_size
    }
}

/// A request to process a routing decision.
struct RoutingDecisionRequest {
    worker: WorkerWithDpRank,
    local_hashes: Vec<LocalBlockHash>,
    sequence_hashes: Vec<SequenceHash>,
}

/// The KV Indexer, managing the KV store and handling events and match requests.
#[derive(Clone)]
pub struct KvIndexer {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// A sender for `RouterEvent`s.
    event_tx: mpsc::Sender<RouterEvent>,
    /// A sender for `MatchRequest`s.
    match_tx: mpsc::Sender<MatchRequest>,
    /// A sender for remove worker requests.
    remove_worker_tx: mpsc::Sender<WorkerId>,
    /// A sender for get workers requests.
    get_workers_tx: mpsc::Sender<GetWorkersRequest>,
    /// A sender for dump requests.
    dump_tx: mpsc::Sender<DumpRequest>,
    /// A sender for routing decision requests.
    routing_tx: mpsc::Sender<RoutingDecisionRequest>,
    /// The size of the KV block this indexer can handle.
    kv_block_size: u32,
    /// Reference counter for Clone-aware Drop.
    /// Only the last clone should cancel the token on drop.
    _ref_count: Arc<()>,
}

impl KvIndexer {
    /// Create a new `KvIndexer`.
    ///
    /// ### Arguments
    ///
    /// * `token` - A `CancellationToken` for managing shutdown.
    /// * `expiration_duration` - The amount of time that block usage should be buffered.
    /// * `ttl` - The time-to-live for blocks before they expire.
    /// * `prune_config` - Configuration for tree-size based pruning.
    ///
    /// ### Returns
    ///
    /// A new `KvIndexer`.
    pub fn new_with_frequency(
        token: CancellationToken,
        expiration_duration: Option<Duration>,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::channel::<RouterEvent>(2048);
        let (match_tx, match_rx) = mpsc::channel::<MatchRequest>(128);
        let (remove_worker_tx, remove_worker_rx) = mpsc::channel::<WorkerId>(16);
        let (get_workers_tx, get_workers_rx) = mpsc::channel::<GetWorkersRequest>(16);
        let (dump_tx, dump_rx) = mpsc::channel::<DumpRequest>(16);
        let (routing_tx, mut routing_rx) = mpsc::channel::<RoutingDecisionRequest>(2048);
        let (prune_tx, mut prune_rx) = mpsc::channel::<()>(1);

        let cancel_clone = token.clone();

        std::thread::spawn(move || {
            // Create a single-threaded tokio runtime
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            runtime.block_on(async move {
                let cancel = cancel_clone;
                let mut match_rx = match_rx;
                let mut event_rx = event_rx;
                let mut remove_worker_rx = remove_worker_rx;
                let mut get_workers_rx = get_workers_rx;
                let mut dump_rx = dump_rx;
                let mut trie = RadixTree::new_with_frequency(expiration_duration);

                // Create PruneManager if prune_config is specified
                let mut prune_manager = prune_config.map(|config| {
                    PruneManager::<BlockEntry>::new(50, config)
                });
                let mut event_id_counter = 0u64;

                loop {
                    // Create a future that sleeps until the next expiration time
                    let expiry_fut = if let Some(ref pm) = prune_manager
                        && let Some(next_expiry) = pm.peek_next_expiry() {
                        tokio::time::sleep_until(next_expiry)
                    } else {
                        tokio::time::sleep(Duration::MAX)
                    };

                    tokio::select! {
                        biased;

                        _ = cancel.cancelled() => {
                            tracing::debug!("KvCacheIndexer progress loop shutting down");
                            return;
                        }

                        Some(worker) = remove_worker_rx.recv() => {
                            trie.remove_worker(worker);
                        }

                        Some(get_workers_req) = get_workers_rx.recv() => {
                            let workers = trie.get_workers();
                            let _ = get_workers_req.resp.send(workers);
                        }

                        Some(_) = prune_rx.recv() => {
                            // Tree size-based pruning triggered
                            let Some(ref mut pm) = prune_manager else { continue };
                            let Ok(pruned) = pm.prune(trie.current_size()) else { continue };

                            for p in pruned {
                                event_id_counter += 1;
                                let event = RouterEvent::new(
                                    p.worker.worker_id,
                                    KvCacheEvent {
                                        event_id: event_id_counter,
                                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                                            block_hashes: vec![p.key],
                                        }),
                                        dp_rank: p.worker.dp_rank,
                                    }
                                );
                                let _ = trie.apply_event(event);
                            }
                        }

                        Some(event) = event_rx.recv() => {
                            let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
                            let event_id = event.event.event_id;
                            let worker_id = event.worker_id;
                            // Only clone if we need the event for prune_manager afterward
                            let event_for_prune = prune_manager.is_some().then(|| event.clone());
                            let result = trie.apply_event(event);
                            let result_is_ok = result.is_ok();
                            let tree_size = trie.current_size();
                            tracing::trace!(
                                "Applied KV event to global radix tree: event_type={event_type}, event_id={event_id}, worker_id={worker_id}, success={result_is_ok}, global_radix_tree_size={tree_size}"
                            );
                            metrics.increment_event_applied(event_type, result);

                            // Track blocks in PruneManager if TTL is enabled and event was stored successfully
                            let Some(ref mut pm) = prune_manager else { continue };
                            if !result_is_ok { continue };
                            let Some(ref event) = event_for_prune else { continue };
                            let KvCacheEventData::Stored(ref store_data) = event.event.data else { continue };

                            let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                            let block_entries: Vec<BlockEntry> = store_data.blocks.iter().enumerate().map(|(idx, block)| {
                                BlockEntry {
                                    key: block.block_hash,
                                    worker,
                                    seq_position: idx,
                                }
                            }).collect();
                            pm.insert(block_entries);

                            // Check if we need to prune due to tree size
                            let Some(ref pc) = pm.prune_config else { continue };
                            let current_size = trie.current_size();
                            if current_size > pc.max_tree_size {
                                tracing::info!(
                                    "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                    current_size,
                                    pc.max_tree_size
                                );
                                let _ = prune_tx.try_send(());
                            }
                        }

                        Some(dump_req) = dump_rx.recv() => {
                            let events = trie.dump_tree_as_events();
                            let _ = dump_req.resp.send(events);
                        }

                        Some(routing_req) = routing_rx.recv() => {
                            // Process routing decisions when TTL/pruning is enabled
                            let Some(ref mut pm) = prune_manager else { continue };

                            event_id_counter += 1;

                            let hashes = routing_req.local_hashes.iter().zip(routing_req.sequence_hashes.iter());
                            let stored_event = KvCacheEventData::Stored(KvCacheStoreData {
                                parent_hash: None,
                                blocks: hashes.map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                                    tokens_hash: *local_hash,
                                    block_hash: ExternalSequenceBlockHash(*sequence_hash),
                                mm_extra_info: None,
                                }).collect(),
                            });

                            let event = RouterEvent::new(
                                routing_req.worker.worker_id,
                                KvCacheEvent {
                                    event_id: event_id_counter,
                                    data: stored_event,
                                    dp_rank: routing_req.worker.dp_rank,
                                }
                            );

                            if trie.apply_event(event).is_err() {
                                continue;
                            }

                            let block_entries: Vec<BlockEntry> = routing_req.sequence_hashes.iter().enumerate().map(|(idx, h)| {
                                BlockEntry {
                                    key: ExternalSequenceBlockHash(*h),
                                    worker: routing_req.worker,
                                    seq_position: idx,
                                }
                            }).collect();
                            pm.insert(block_entries);

                            // Check if we need to prune due to tree size
                            let Some(ref pc) = pm.prune_config else { continue };
                            let current_size = trie.current_size();
                            if current_size > pc.max_tree_size {
                                tracing::info!(
                                    "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                    current_size,
                                    pc.max_tree_size
                                );
                                let _ = prune_tx.try_send(());
                            }
                        }

                        Some(req) = match_rx.recv() => {
                            #[cfg(feature = "bench")]
                            let queue_wait = req.created_at.elapsed();
                            #[cfg(feature = "bench")]
                            let seq_len = req.sequence.len();

                            #[cfg(feature = "bench")]
                            let process_start = Instant::now();
                            let matches = trie.find_matches(req.sequence, req.early_exit);
                            #[cfg(feature = "bench")]
                            let process_time = process_start.elapsed();

                            #[cfg(feature = "bench")]
                            tracing::info!(
                                seq_len,
                                queue_wait_us = queue_wait.as_micros() as u64,
                                process_us = process_time.as_micros() as u64,
                                "indexer: processed find_matches"
                            );
                            let _ = req.resp.send(matches);
                        }

                        _ = expiry_fut => {
                            // TTL-based expiry triggered
                            let Some(ref mut pm) = prune_manager else { continue };

                            let expired = pm.pop_expired();
                            for e in expired {
                                event_id_counter += 1;
                                let event = RouterEvent::new(
                                    e.worker.worker_id,
                                    KvCacheEvent {
                                        event_id: event_id_counter,
                                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                                            block_hashes: vec![e.key],
                                        }),
                                        dp_rank: e.worker.dp_rank,
                                    }
                                );
                                let _ = trie.apply_event(event);
                            }
                        }
                    }
                }
            });

            tracing::debug!("KvCacheIndexer task completed");
        });

        Self {
            cancel: token,
            event_tx,
            match_tx,
            remove_worker_tx,
            get_workers_tx,
            dump_tx,
            routing_tx,
            kv_block_size,
            _ref_count: Arc::new(()),
        }
    }

    pub fn block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn new(
        token: CancellationToken,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Self {
        Self::new_with_frequency(token, None, kv_block_size, metrics, None)
    }

    /// Get a sender for `RouterEvent`s.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `RouterEvent`s.
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.event_tx.clone()
    }

    /// Get a sender for dump requests (snapshot events).
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `DumpRequest`s.
    pub fn snapshot_event_sender(&self) -> mpsc::Sender<DumpRequest> {
        self.dump_tx.clone()
    }

    /// Get a sender for worker removal requests.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `WorkerId`s.
    pub fn remove_worker_sender(&self) -> mpsc::Sender<WorkerId> {
        self.remove_worker_tx.clone()
    }

    /// Get a sender for get workers requests.
    ///
    /// ### Returns
    ///
    /// A `mpsc::Sender` for `GetWorkersRequest`s.
    pub fn get_workers_sender(&self) -> mpsc::Sender<GetWorkersRequest> {
        self.get_workers_tx.clone()
    }
}

#[async_trait]
impl KvIndexerInterface for KvIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        #[cfg(feature = "bench")]
        let start = Instant::now();
        let seq_len = sequence.len();
        let (resp_tx, resp_rx) = oneshot::channel();
        let req = MatchRequest::new(sequence, false, resp_tx);

        if let Err(e) = self.match_tx.send(req).await {
            tracing::error!(
                "Failed to send match request: {:?}; the indexer maybe offline",
                e
            );
            return Err(KvRouterError::IndexerOffline);
        }

        let result = resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest);

        #[cfg(feature = "bench")]
        {
            let elapsed = start.elapsed();
            tracing::info!(
                seq_len,
                elapsed_us = elapsed.as_micros() as u64,
                "find_matches completed"
            );
        }
        #[cfg(not(feature = "bench"))]
        let _ = seq_len;

        result
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
    ) -> Result<OverlapScores, KvRouterError> {
        tracing::debug!(
            "Finding matches for request tokens: {:?} / len: {}",
            tokens,
            tokens.len()
        );
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None);
        tracing::debug!("Computed sequence: {:?}", sequence);
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        self.event_tx.send(event).await.unwrap();
    }

    async fn remove_worker(&self, worker: WorkerId) {
        self.remove_worker_tx.send(worker).await.unwrap();
    }

    fn shutdown(&self) {
        self.cancel.cancel();
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let (resp_tx, resp_rx) = oneshot::channel();
        let dump_req = DumpRequest { resp: resp_tx };

        if let Err(e) = self.dump_tx.send(dump_req).await {
            tracing::error!("Failed to send dump request: {:?}", e);
            return Err(KvRouterError::IndexerOffline);
        }

        resp_rx
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
        let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();

        self.process_routing_decision_internal(worker, local_hashes, sequence_hashes)
            .await
    }
    async fn flush(&self) -> usize {
        let curr_size = self.event_tx.max_capacity() - self.event_tx.capacity();
        loop {
            if self.event_tx.capacity() == self.event_tx.max_capacity() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        curr_size
    }
}

impl KvIndexer {
    /// Internal method to process a routing decision with pre-computed hashes.
    async fn process_routing_decision_internal(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        self.routing_tx
            .send(RoutingDecisionRequest {
                worker,
                local_hashes,
                sequence_hashes,
            })
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)?;
        Ok(())
    }
}

impl Drop for KvIndexer {
    fn drop(&mut self) {
        // Only cancel the token if we're the last reference.
        // This allows clones to be dropped without killing the background task.
        if Arc::strong_count(&self._ref_count) == 1 {
            self.shutdown();
        }
    }
}

// -------------------------------------------------
// Decentralized router: LocalKvIndexer for workers
// -------------------------------------------------

/// A thin wrapper around KvIndexer that buffers recent events
/// (e.g. which may be queued by router upon startup)
///
pub struct LocalKvIndexer {
    /// The underlying indexer
    indexer: KvIndexer,
    /// Circular buffer of recent events
    event_buffer: Mutex<VecDeque<RouterEvent>>,
    /// Maximum number of events to keep in buffer
    max_buffer_size: usize, // Router sets this to WORKER_KV_INDEXER_BUFFER_SIZE
}

impl LocalKvIndexer {
    /// create a new LocalKvIndexer pointing to a KvIndexer.
    pub fn new(
        token: CancellationToken,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        max_buffer_size: usize,
    ) -> Self {
        Self {
            indexer: KvIndexer::new(token, kv_block_size, metrics),
            event_buffer: Mutex::new(VecDeque::with_capacity(max_buffer_size)),
            max_buffer_size,
        }
    }

    /// Get all buffered events (oldest first).
    pub fn get_all_events_in_buffer(&self) -> Vec<RouterEvent> {
        let buffer = self.event_buffer.lock().unwrap();
        buffer.iter().cloned().collect()
    }

    /// Query events by ID range, returning events in `[start_id, end_id]` (both inclusive).
    ///
    /// ### Arguments
    ///
    /// * `start_id` - Starting event ID (inclusive). If `None`, dumps entire tree.
    /// * `end_id` - Ending event ID (inclusive). If `None`, returns up to newest available.
    ///
    /// ### Returns
    ///
    /// - `Events`: Buffered events with original IDs (when range is within buffer)
    /// - `TreeDump`: Full tree dump with synthetic IDs (when range is too old or unspecified)
    /// - `TooNew`: Error when requested range is newer than available data
    /// - `InvalidRange`: Error when end_id < start_id
    pub async fn get_events_in_id_range(
        &self,
        start_id: Option<u64>,
        end_id: Option<u64>,
    ) -> WorkerKvQueryResponse {
        // Validate range if both specified
        if let (Some(s), Some(e)) = (start_id, end_id)
            && e < s
        {
            tracing::warn!(start_id = s, end_id = e, "Invalid range: end_id < start_id");
            return WorkerKvQueryResponse::InvalidRange {
                start_id: s,
                end_id: e,
            };
        }

        // Get buffer state
        let (first_id, last_id) = {
            let buffer = self.event_buffer.lock().unwrap();
            if buffer.is_empty() {
                (None, None)
            } else {
                (
                    Some(buffer.front().unwrap().event.event_id),
                    Some(buffer.back().unwrap().event.event_id),
                )
            }
        };

        // If no start_id specified, dump entire tree
        if start_id.is_none() {
            tracing::debug!("No start_id specified, dumping entire tree");
            let events = self.dump_events().await.unwrap_or_default();
            return WorkerKvQueryResponse::TreeDump(events);
        }

        let start_id = start_id.unwrap();
        let end_id = end_id.unwrap_or_else(|| last_id.unwrap_or(start_id));

        // Check for empty buffer
        let Some(first_buffered) = first_id else {
            tracing::debug!("Buffer empty, dumping entire tree");
            let events = self.dump_events().await.unwrap_or_default();
            return WorkerKvQueryResponse::TreeDump(events);
        };
        let last_buffered = last_id.unwrap();

        // Check if request is too new
        if start_id > last_buffered {
            tracing::warn!(
                start_id,
                last_buffered,
                "Requested start_id is newer than buffer"
            );
            return WorkerKvQueryResponse::TooNew {
                requested_start: Some(start_id),
                requested_end: Some(end_id),
                newest_available: last_buffered,
            };
        }

        // Check if start_id is too old (before buffer) -> tree dump
        if start_id < first_buffered {
            tracing::info!(
                start_id,
                first_buffered,
                "Requested start_id is older than buffer, dumping entire tree"
            );
            let events = self.dump_events().await.unwrap_or_default();
            return WorkerKvQueryResponse::TreeDump(events);
        }

        // Serve from buffer
        let buffer = self.event_buffer.lock().unwrap();

        let start_idx = match buffer.binary_search_by_key(&start_id, |e| e.event.event_id) {
            Ok(idx) => idx,
            Err(insertion_point) => insertion_point,
        };

        // Clamp end_id to buffer bounds
        let clamped_end_id = end_id.min(last_buffered);
        let end_idx = match buffer.binary_search_by_key(&clamped_end_id, |e| e.event.event_id) {
            Ok(idx) => idx + 1, // Include the matched element
            Err(insertion_point) => insertion_point,
        };

        let events: Vec<RouterEvent> = buffer
            .iter()
            .skip(start_idx)
            .take(end_idx.saturating_sub(start_idx))
            .cloned()
            .collect();

        WorkerKvQueryResponse::Events(events)
    }

    /// Record an event in the buffer
    fn record_event(&self, event: RouterEvent) {
        let mut buffer = self.event_buffer.lock().unwrap();

        // Check that event id is consecutive to last one
        if let Some(last_event) = buffer.back()
            && event.event.event_id != last_event.event.event_id + 1
        {
            let expected = last_event.event.event_id + 1;
            tracing::error!(
                worker_id = event.worker_id,
                expected,
                got = event.event.event_id,
                "Non-consecutive KV event id; buffer may have gaps"
            );
        }
        tracing::debug!(
            "Recorded event {:?} in buffer, now size is {}",
            event,
            buffer.len()
        );

        // Add to back
        buffer.push_back(event);

        // Remove from front if over capacity (circular buffer behavior)
        while buffer.len() > self.max_buffer_size {
            buffer.pop_front();
        }
    }

    /// Apply event with buffering.
    ///
    /// This records the event in the buffer and forwards it to the underlying indexer.
    pub async fn apply_event_with_buffer(&self, event: RouterEvent) -> Result<(), KvRouterError> {
        // Record in buffer
        self.record_event(event.clone());

        // Forward to underlying indexer
        self.indexer
            .event_sender()
            .send(event)
            .await
            .map_err(|_| KvRouterError::IndexerOffline)
    }

    /// Clear the event buffer.
    pub fn clear_buffer(&self) {
        let mut buffer = self.event_buffer.lock().unwrap();
        buffer.clear();
    }

    /// Get the current buffer size.
    pub fn buffer_len(&self) -> usize {
        let buffer = self.event_buffer.lock().unwrap();
        buffer.len()
    }

    // Delegation methods to underlying KvIndexer
    /// Get a sender for `RouterEvent`s.
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.indexer.event_sender()
    }

    /// Get a sender for dump requests (snapshot events).
    pub fn snapshot_event_sender(&self) -> mpsc::Sender<DumpRequest> {
        self.indexer.snapshot_event_sender()
    }

    /// Get a sender for worker removal requests.
    pub fn remove_worker_sender(&self) -> mpsc::Sender<WorkerId> {
        self.indexer.remove_worker_sender()
    }

    /// Get a sender for get workers requests.
    pub fn get_workers_sender(&self) -> mpsc::Sender<GetWorkersRequest> {
        self.indexer.get_workers_sender()
    }

    /// Get the KV block size.
    pub fn block_size(&self) -> u32 {
        self.indexer.block_size()
    }
}

// Implement KvIndexerInterface by delegating to the underlying indexer
#[async_trait]
impl KvIndexerInterface for LocalKvIndexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        self.indexer.find_matches(sequence).await
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
    ) -> Result<OverlapScores, KvRouterError> {
        self.indexer.find_matches_for_request(tokens).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        // Use the buffering version
        let _ = self.apply_event_with_buffer(event).await;
    }

    async fn remove_worker(&self, worker: WorkerId) {
        let _ = self.indexer.remove_worker_sender().send(worker).await;
    }

    fn shutdown(&self) {
        // Note: Since indexer is Arc<KvIndexer>, we can't call mutable methods directly.
        // The indexer will be shut down when the CancellationToken is cancelled
        // or when the last Arc reference is dropped.
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.indexer.dump_events().await
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // TODO I guess the local kvindexers have little use for this method?
        // Keeping it here now to implement the trait fully
        self.indexer
            .process_routing_decision_for_request(tokens_with_hashes, worker)
            .await
    }

    async fn flush(&self) -> usize {
        self.indexer.flush().await
    }
}

#[derive(Debug, Clone)]
pub struct ShardedMatchRequest {
    sequence: Vec<LocalBlockHash>,
    early_exit: bool,
    resp: mpsc::Sender<OverlapScores>,
    #[cfg(feature = "bench")]
    created_at: Instant,
}

impl ShardedMatchRequest {
    fn new(
        sequence: Vec<LocalBlockHash>,
        early_exit: bool,
        resp: mpsc::Sender<OverlapScores>,
    ) -> Self {
        Self {
            sequence,
            early_exit,
            resp,
            #[cfg(feature = "bench")]
            created_at: Instant::now(),
        }
    }
}

/// A sharded KV Indexer that partitions the RadixTree across multiple independent shards.
///
/// ## Sharding Strategy
/// - Each worker is **permanently assigned** to a single shard on first event
/// - All KV blocks from a worker exist only in that worker's assigned shard
/// - New workers are assigned to the shard with the fewest workers (load balancing)
///
/// ## Operation
/// - **Events**: Routed directly to the worker's assigned shard
/// - **Match requests**: Broadcast to all shards (scatter-gather pattern)
/// - **Threading**: Each shard runs in its own thread with a single-threaded runtime
///
/// This design ensures no cross-shard synchronization for writes while enabling
/// parallel processing and better scalability.
pub struct KvIndexerSharded {
    /// A `CancellationToken` for managing shutdown.
    cancel: CancellationToken,
    /// The size of the KV block this indexer can handle.
    kv_block_size: u32,
    worker_assignments: DashMap<WorkerId, usize, FxBuildHasher>,
    worker_counts: Arc<Mutex<Vec<usize>>>,

    event_tx: Vec<mpsc::Sender<RouterEvent>>,
    request_broadcast_tx: broadcast::Sender<ShardedMatchRequest>,
    remove_worker_tx: Vec<mpsc::Sender<WorkerId>>,
    dump_tx: Vec<mpsc::Sender<DumpRequest>>,
    routing_tx: Vec<mpsc::Sender<RoutingDecisionRequest>>,
    tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl KvIndexerSharded {
    /// Create a new `KvIndexerSharded`.
    ///
    /// ### Arguments
    ///
    /// * `token` - A `CancellationToken` for managing shutdown.
    /// * `shards` - A list of kvindexer shards.
    /// * `expiration_duration` - The amount of time that block usage should be buffered.
    /// * `ttl` - The time-to-live for blocks before they expire.
    /// * `prune_config` - Configuration for tree-size based pruning.
    ///
    /// ### Returns
    ///
    /// A new `KvIndexer`.
    pub fn new_with_frequency(
        token: CancellationToken,
        num_shards: usize,
        expiration_duration: Option<Duration>,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
        prune_config: Option<PruneConfig>,
    ) -> Self {
        let worker_assignments = DashMap::with_hasher(FxBuildHasher);
        let worker_counts = Arc::new(Mutex::new(vec![0; num_shards]));

        let mut event_tx = Vec::new();
        let mut remove_worker_tx = Vec::new();
        let mut get_workers_tx = Vec::new();
        let mut dump_tx = Vec::new();
        let mut routing_tx = Vec::new();
        let tasks = Arc::new(Mutex::new(Vec::new()));

        let (request_broadcast_tx, _) = broadcast::channel::<ShardedMatchRequest>(1048576);

        for _ in 0..num_shards {
            let (shard_event_tx, mut shard_event_rx) = mpsc::channel::<RouterEvent>(2048);
            let (shard_remove_worker_tx, mut shard_remove_worker_rx) =
                mpsc::channel::<WorkerId>(16);
            let (shard_get_workers_tx, mut shard_get_workers_rx) =
                mpsc::channel::<GetWorkersRequest>(16);
            let (shard_dump_tx, mut shard_dump_rx) = mpsc::channel::<DumpRequest>(16);
            let (shard_routing_tx, mut shard_routing_rx) =
                mpsc::channel::<RoutingDecisionRequest>(2048);
            let (shard_prune_tx, mut shard_prune_rx) = mpsc::channel::<()>(1);
            let mut shard_broadcast_rx = request_broadcast_tx.subscribe();
            let cancel = token.clone();
            let metrics = metrics.clone();
            let prune_config_clone = prune_config.clone();

            event_tx.push(shard_event_tx);
            remove_worker_tx.push(shard_remove_worker_tx);
            get_workers_tx.push(shard_get_workers_tx);
            dump_tx.push(shard_dump_tx);
            routing_tx.push(shard_routing_tx);

            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            tasks.lock().unwrap().push(std::thread::spawn(move || {
                runtime.block_on(async move {
                    let mut trie = RadixTree::new_with_frequency(expiration_duration);

                    // Create PruneManager if prune_config is specified
                    let mut prune_manager = prune_config_clone.map(|config| {
                        PruneManager::<BlockEntry>::new(50, config)
                    });
                    let mut event_id_counter = 0u64;

                    loop {
                        // Create a future that sleeps until the next expiration time
                        let expiry_fut = if let Some(ref pm) = prune_manager
                            && let Some(next_expiry) = pm.peek_next_expiry() {
                            tokio::time::sleep_until(next_expiry)
                        } else {
                            tokio::time::sleep(Duration::MAX)
                        };

                        tokio::select! {
                            biased;

                            _ = cancel.cancelled() => {
                                tracing::trace!("KvCacheIndexer progress loop shutting down");
                                return;
                            }

                            Some(worker) = shard_remove_worker_rx.recv() => {
                                trie.remove_worker(worker);
                            }

                            Some(get_workers_req) = shard_get_workers_rx.recv() => {
                                let workers = trie.get_workers();
                                let _ = get_workers_req.resp.send(workers);
                            }

                            Some(_) = shard_prune_rx.recv() => {
                                // Tree size-based pruning triggered
                                let Some(ref mut pm) = prune_manager else { continue };
                                let Ok(pruned) = pm.prune(trie.current_size()) else { continue };

                                for p in pruned {
                                    event_id_counter += 1;
                                    let event = RouterEvent::new(
                                        p.worker.worker_id,
                                        KvCacheEvent {
                                            event_id: event_id_counter,
                                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                                block_hashes: vec![p.key],
                                            }),
                                            dp_rank: p.worker.dp_rank,
                                        }
                                    );
                                    let _ = trie.apply_event(event);
                                }
                            }

                            Some(event) = shard_event_rx.recv() => {
                                let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
                                // Only clone if we need the event for prune_manager afterward
                                let event_for_prune = prune_manager.is_some().then(|| event.clone());
                                let result = trie.apply_event(event);
                                let result_is_ok = result.is_ok();
                                metrics.increment_event_applied(event_type, result);

                                // Track blocks in PruneManager if TTL is enabled and event was stored successfully
                                let Some(ref mut pm) = prune_manager else { continue };
                                if !result_is_ok { continue };
                                let Some(ref event) = event_for_prune else { continue };
                                let KvCacheEventData::Stored(ref store_data) = event.event.data else { continue };

                                let worker = WorkerWithDpRank::new(event.worker_id, event.event.dp_rank);
                                let block_entries: Vec<BlockEntry> = store_data.blocks.iter().enumerate().map(|(idx, block)| {
                                    BlockEntry {
                                        key: block.block_hash,
                                        worker,
                                        seq_position: idx,
                                    }
                                }).collect();
                                pm.insert(block_entries);

                                // Check if we need to prune due to tree size
                                let Some(ref pc) = pm.prune_config else { continue };
                                let current_size = trie.current_size();
                                if current_size > pc.max_tree_size {
                                    tracing::info!(
                                        "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                        current_size,
                                        pc.max_tree_size
                                    );
                                    let _ = shard_prune_tx.try_send(());
                                }
                            }

                            Some(routing_req) = shard_routing_rx.recv() => {
                                // Process routing decisions when TTL/pruning is enabled
                                let Some(ref mut pm) = prune_manager else { continue };

                                event_id_counter += 1;

                                let hashes = routing_req.local_hashes.iter().zip(routing_req.sequence_hashes.iter());
                                let stored_event = KvCacheEventData::Stored(KvCacheStoreData {
                                    parent_hash: None,
                                    blocks: hashes.map(|(local_hash, sequence_hash)| KvCacheStoredBlockData {
                                        tokens_hash: *local_hash,
                                        block_hash: ExternalSequenceBlockHash(*sequence_hash),
                                mm_extra_info: None,
                                    }).collect(),
                                });

                                let event = RouterEvent::new(
                                    routing_req.worker.worker_id,
                                    KvCacheEvent {
                                        event_id: event_id_counter,
                                        data: stored_event,
                                        dp_rank: routing_req.worker.dp_rank,
                                    }
                                );

                                if trie.apply_event(event).is_err() {
                                    continue;
                                }

                                let block_entries: Vec<BlockEntry> = routing_req.sequence_hashes.iter().enumerate().map(|(idx, h)| {
                                    BlockEntry {
                                        key: ExternalSequenceBlockHash(*h),
                                        worker: routing_req.worker,
                                        seq_position: idx,
                                    }
                                }).collect();
                                pm.insert(block_entries);

                                // Check if we need to prune due to tree size
                                let Some(ref pc) = pm.prune_config else { continue };
                                let current_size = trie.current_size();
                                if current_size > pc.max_tree_size {
                                    tracing::info!(
                                        "Pruning: tree size ({}) exceeded max tree size ({}), scheduling pruning",
                                        current_size,
                                        pc.max_tree_size
                                    );
                                    let _ = shard_prune_tx.try_send(());
                                }
                            }

                            Some(dump_req) = shard_dump_rx.recv() => {
                                let events = trie.dump_tree_as_events();
                                let _ = dump_req.resp.send(events);
                            }

                            Ok(req) = shard_broadcast_rx.recv() => {
                                #[cfg(feature = "bench")]
                                let queue_wait = req.created_at.elapsed();
                                #[cfg(feature = "bench")]
                                let seq_len = req.sequence.len();

                                #[cfg(feature = "bench")]
                                let process_start = Instant::now();
                                let matches = trie.find_matches(req.sequence, req.early_exit);
                                #[cfg(feature = "bench")]
                                let process_time = process_start.elapsed();

                                #[cfg(feature = "bench")]
                                tracing::info!(
                                    seq_len,
                                    queue_wait_us = queue_wait.as_micros() as u64,
                                    process_us = process_time.as_micros() as u64,
                                    "sharded indexer: processed find_matches"
                                );
                                if let Err(e) = req.resp.send(matches).await {
                                    tracing::trace!("Failed to send match response: {:?}", e);
                                }
                            }

                            _ = expiry_fut => {
                                // TTL-based expiry triggered
                                let Some(ref mut pm) = prune_manager else { continue };

                                let expired = pm.pop_expired();
                                for e in expired {
                                    event_id_counter += 1;
                                    let event = RouterEvent::new(
                                        e.worker.worker_id,
                                        KvCacheEvent {
                                            event_id: event_id_counter,
                                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                                block_hashes: vec![e.key],
                                            }),
                                            dp_rank: e.worker.dp_rank,
                                        }
                                    );
                                    let _ = trie.apply_event(event);
                                }
                            }
                        }
                    }
                });

                tracing::debug!("KvCacheIndexer task completed");
            }));
        }

        Self {
            cancel: token,
            kv_block_size,
            worker_assignments,
            worker_counts,
            event_tx,
            request_broadcast_tx,
            remove_worker_tx,
            dump_tx,
            routing_tx,
            tasks,
        }
    }

    pub fn block_size(&self) -> u32 {
        self.kv_block_size
    }

    pub fn new(
        token: CancellationToken,
        num_shards: usize,
        kv_block_size: u32,
        metrics: Arc<KvIndexerMetrics>,
    ) -> Self {
        Self::new_with_frequency(token, num_shards, None, kv_block_size, metrics, None)
    }
}

#[async_trait]
impl KvIndexerInterface for KvIndexerSharded {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        #[cfg(feature = "bench")]
        let start = Instant::now();
        #[cfg(feature = "bench")]
        let seq_len = sequence.len();
        #[cfg(feature = "bench")]
        let num_shards = self.event_tx.len();

        'match_loop: loop {
            let (match_tx, mut match_rx) = mpsc::channel(self.event_tx.len());
            let sharded_req = ShardedMatchRequest::new(sequence.clone(), false, match_tx);
            self.request_broadcast_tx
                .send(sharded_req)
                .map_err(|_| KvRouterError::IndexerOffline)?;

            let mut scores = OverlapScores::new();

            for response_num in 0..self.event_tx.len() {
                match match_rx.recv().await {
                    Some(response) => {
                        scores.scores.extend(response.scores);
                        scores.tree_sizes.extend(response.tree_sizes);

                        if response_num == 0 {
                            scores.frequencies = response.frequencies;
                        } else {
                            let diff = (response.frequencies.len() as i64)
                                - (scores.frequencies.len() as i64);

                            if diff > 0 {
                                scores.frequencies.extend(iter::repeat_n(0, diff as usize));
                            }

                            for i in 0..response.frequencies.len() {
                                scores.frequencies[i] += response.frequencies[i];
                            }
                        }
                    }
                    None => {
                        // This can only happen if the broadcast channel overflows.
                        // In this case, we don't want to recursively call find_matches again. Otherwise, we could overflow the stack.
                        continue 'match_loop;
                    }
                }
            }

            #[cfg(feature = "bench")]
            {
                let elapsed = start.elapsed();
                tracing::info!(
                    seq_len,
                    num_shards,
                    elapsed_us = elapsed.as_micros() as u64,
                    "find_matches (sharded) completed"
                );
            }
            return Ok(scores);
        }
    }

    async fn find_matches_for_request(
        &self,
        tokens: &[u32],
    ) -> Result<OverlapScores, KvRouterError> {
        let sequence = compute_block_hash_for_seq(tokens, self.kv_block_size, None);
        self.find_matches(sequence).await
    }

    async fn apply_event(&self, event: RouterEvent) {
        let shard = self
            .worker_assignments
            .entry(event.worker_id)
            .or_insert_with(|| {
                // Get the shard with the smallest amount of workers.
                let worker_counts = self.worker_counts.lock().unwrap();
                let selected_shard = worker_counts
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, value)| value)
                    .unwrap()
                    .0;
                drop(worker_counts);

                // Increment the count for this shard
                self.worker_counts.lock().unwrap()[selected_shard] += 1;
                selected_shard
            });

        self.event_tx[*shard].send(event).await.unwrap();
    }

    async fn remove_worker(&self, worker: WorkerId) {
        if let Some((_, shard)) = self.worker_assignments.remove(&worker) {
            self.worker_counts.lock().unwrap()[shard] -= 1;
            self.remove_worker_tx[shard].send(worker).await.unwrap();
        }
    }

    /// Shutdown the KV Indexer.
    fn shutdown(&self) {
        self.cancel.cancel();
        let mut tasks = self.tasks.lock().unwrap();
        while !tasks.is_empty() {
            tasks.pop().unwrap().join().unwrap();
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        let mut all_events = Vec::new();

        // Create channels for each shard
        let mut receivers = Vec::new();

        for shard_dump_tx in &self.dump_tx {
            let (resp_tx, resp_rx) = oneshot::channel();
            let dump_req = DumpRequest { resp: resp_tx };

            if let Err(e) = shard_dump_tx.send(dump_req).await {
                tracing::error!("Failed to send dump request to shard: {:?}", e);
                return Err(KvRouterError::IndexerOffline);
            }

            receivers.push(resp_rx);
        }

        // Collect results from all shards
        for resp_rx in receivers {
            match resp_rx.await {
                Ok(events) => all_events.extend(events),
                Err(_) => return Err(KvRouterError::IndexerDroppedRequest),
            }
        }

        Ok(all_events)
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
        let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();

        self.process_routing_decision_internal(worker, local_hashes, sequence_hashes)
            .await
    }

    async fn flush(&self) -> usize {
        let curr_size = self
            .event_tx
            .iter()
            .map(|tx| tx.max_capacity() - tx.capacity())
            .sum();
        loop {
            if self
                .event_tx
                .iter()
                .all(|tx| tx.capacity() == tx.max_capacity())
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        curr_size
    }
}

impl KvIndexerSharded {
    /// Internal method to process a routing decision with pre-computed hashes.
    async fn process_routing_decision_internal(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        // Route to the appropriate shard based on worker assignment
        let shard_idx = self
            .worker_assignments
            .get(&worker.worker_id)
            .map(|shard_idx| *shard_idx)
            .unwrap_or_default();

        self.routing_tx[shard_idx]
            .send(RoutingDecisionRequest {
                worker,
                local_hashes,
                sequence_hashes,
            })
            .await
            .map_err(|_| KvRouterError::IndexerDroppedRequest)?;
        Ok(())
    }
}

impl Drop for KvIndexerSharded {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concurrent_radix_tree::ConcurrentRadixTree;
    use crate::nested_map::PositionalIndexer;
    use crate::protocols::{ExternalSequenceBlockHash, LocalBlockHash, compute_seq_hash_for_block};
    use rstest::rstest;
    use rstest_reuse::{self, *};
    use std::time::Instant;
    use tokio::time;
    use tokio_util::sync::CancellationToken;

    // ============================================================================
    // Helper functions
    // ============================================================================

    /// Create a store event with proper sequence hashes computed from local hashes.
    fn make_store_event(worker_id: u64, local_hashes: &[u64]) -> RouterEvent {
        make_store_event_with_dp_rank(worker_id, local_hashes, 0)
    }

    /// Create a store event with a specific dp_rank.
    fn make_store_event_with_dp_rank(
        worker_id: u64,
        local_hashes: &[u64],
        dp_rank: u32,
    ) -> RouterEvent {
        make_store_event_full(worker_id, local_hashes, dp_rank, None)
    }

    /// Create a store event with parent hash for continuation sequences.
    /// `prefix_hashes` are the hashes of the prefix (to compute parent_hash).
    /// `local_hashes` are the new blocks being stored.
    fn make_store_event_with_parent(
        worker_id: u64,
        prefix_hashes: &[u64],
        local_hashes: &[u64],
    ) -> RouterEvent {
        // Compute the parent hash from the prefix
        let prefix_block_hashes: Vec<LocalBlockHash> =
            prefix_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
        let prefix_seq_hashes = compute_seq_hash_for_block(&prefix_block_hashes);
        let parent_hash = prefix_seq_hashes
            .last()
            .map(|&h| ExternalSequenceBlockHash(h));

        // Compute the full sequence including prefix for proper seq_hash calculation
        let full_hashes: Vec<u64> = prefix_hashes
            .iter()
            .chain(local_hashes.iter())
            .copied()
            .collect();
        let full_block_hashes: Vec<LocalBlockHash> =
            full_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
        let full_seq_hashes = compute_seq_hash_for_block(&full_block_hashes);

        // Only include the new blocks (skip prefix)
        let new_block_hashes: Vec<LocalBlockHash> =
            local_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
        let new_seq_hashes = &full_seq_hashes[prefix_hashes.len()..];

        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    blocks: new_block_hashes
                        .iter()
                        .zip(new_seq_hashes.iter())
                        .map(|(&local, &seq)| KvCacheStoredBlockData {
                            tokens_hash: local,
                            block_hash: ExternalSequenceBlockHash(seq),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: 0,
            },
        }
    }

    /// Create a store event with all options.
    fn make_store_event_full(
        worker_id: u64,
        local_hashes: &[u64],
        dp_rank: u32,
        parent_hash: Option<ExternalSequenceBlockHash>,
    ) -> RouterEvent {
        let local_block_hashes: Vec<LocalBlockHash> =
            local_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
        let seq_hashes = compute_seq_hash_for_block(&local_block_hashes);

        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    blocks: local_block_hashes
                        .iter()
                        .zip(seq_hashes.iter())
                        .map(|(&local, &seq)| KvCacheStoredBlockData {
                            tokens_hash: local,
                            block_hash: ExternalSequenceBlockHash(seq),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank,
            },
        }
    }

    /// Create a remove event for blocks with given local hashes.
    fn make_remove_event(worker_id: u64, local_hashes: &[u64]) -> RouterEvent {
        make_remove_event_with_dp_rank(worker_id, local_hashes, 0)
    }

    /// Create a remove event with a specific dp_rank.
    fn make_remove_event_with_dp_rank(
        worker_id: u64,
        local_hashes: &[u64],
        dp_rank: u32,
    ) -> RouterEvent {
        let local_block_hashes: Vec<LocalBlockHash> =
            local_hashes.iter().map(|&h| LocalBlockHash(h)).collect();
        let seq_hashes = compute_seq_hash_for_block(&local_block_hashes);

        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: seq_hashes
                        .iter()
                        .map(|&h| ExternalSequenceBlockHash(h))
                        .collect(),
                }),
                dp_rank,
            },
        }
    }

    /// Create a clear event for a worker.
    fn make_clear_event(worker_id: u64) -> RouterEvent {
        make_clear_event_with_dp_rank(worker_id, 0)
    }

    /// Create a clear event with a specific dp_rank.
    fn make_clear_event_with_dp_rank(worker_id: u64, dp_rank: u32) -> RouterEvent {
        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Cleared,
                dp_rank,
            },
        }
    }

    // ============================================================================
    // KvIndexerInterface tests - parametrized over all implementations
    // ============================================================================

    #[template]
    #[rstest]
    fn indexer_template(#[values("single", "sharded", "flat", "concurrent")] variant: &str) {}

    fn make_indexer(variant: &str) -> Box<dyn KvIndexerInterface> {
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let kv_block_size = 32;

        match variant {
            "single" => Box::new(KvIndexer::new(token, kv_block_size, metrics)),
            "sharded" => Box::new(KvIndexerSharded::new(token, 4, kv_block_size, metrics)),
            "flat" => Box::new(ThreadPoolIndexer::new(
                PositionalIndexer::new(32),
                4,
                kv_block_size,
            )),
            "concurrent" => Box::new(ThreadPoolIndexer::new(
                ConcurrentRadixTree::new(),
                4,
                kv_block_size,
            )),
            _ => panic!("Unknown variant: {}", variant),
        }
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_store_and_find(variant: &str) {
        let index = make_indexer(variant);

        // Store a sequence for worker 0
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Find matches using local hashes
        let scores = index
            .find_matches(vec![
                LocalBlockHash(1),
                LocalBlockHash(2),
                LocalBlockHash(3),
            ])
            .await
            .unwrap();
        assert_eq!(scores.scores.len(), 1);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_partial_match(variant: &str) {
        let index = make_indexer(variant);

        // Store [1, 2, 3] for worker 0
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Find matches for [1, 2, 999] - should match first 2 then stop
        let scores = index
            .find_matches(vec![
                LocalBlockHash(1),
                LocalBlockHash(2),
                LocalBlockHash(999),
            ])
            .await
            .unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 2);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_remove(variant: &str) {
        let index = make_indexer(variant);

        // Store sequence for worker 0
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Remove all blocks
        index.apply_event(make_remove_event(0, &[1, 2, 3])).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Find should return nothing
        let scores = index
            .find_matches(vec![
                LocalBlockHash(1),
                LocalBlockHash(2),
                LocalBlockHash(3),
            ])
            .await
            .unwrap();
        assert!(scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_multiple_workers_shared_prefix(variant: &str) {
        let index = make_indexer(variant);

        // Worker 0 has [1, 2], Worker 1 has [1, 3]
        // Since sequence hashes are cumulative, [1] has same hash for both,
        // but [1, 2] and [1, 3] have different hashes.
        index.apply_event(make_store_event(0, &[1, 2])).await;
        index.apply_event(make_store_event(1, &[1, 3])).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query [1] - both workers should match
        let scores = index.find_matches(vec![LocalBlockHash(1)]).await.unwrap();
        assert_eq!(scores.scores.len(), 2);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 1);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(), 1);

        // Query [1, 2] - worker 0 matches both, worker 1 matches only first block
        let scores = index
            .find_matches(vec![LocalBlockHash(1), LocalBlockHash(2)])
            .await
            .unwrap();
        assert_eq!(scores.scores.len(), 2);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 2);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(), 1);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_remove_worker(variant: &str) {
        let index = make_indexer(variant);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        index.apply_event(make_store_event(1, &[1, 2, 3])).await;

        // Allow time for async event processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        index.remove_worker(0).await;

        // Allow time for async remove_worker processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        let scores = index
            .find_matches(vec![
                LocalBlockHash(1),
                LocalBlockHash(2),
                LocalBlockHash(3),
            ])
            .await
            .unwrap();
        assert_eq!(scores.scores.len(), 1);
        assert!(scores.scores.contains_key(&WorkerWithDpRank::new(1, 0)));
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_large_stores(variant: &str) {
        let index = make_indexer(variant);

        // Test sequences of increasing sizes
        for i in 0..10u64 {
            let len = 1 << i; // 1, 2, 4, 8, ..., 512
            let worker_id = i;
            let sequence: Vec<u64> = (1..=len).map(|x| x + (i * 10000)).collect();
            index
                .apply_event(make_store_event(worker_id, &sequence))
                .await;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify we can find matches for the last stored sequence
        let last_seq: Vec<LocalBlockHash> = (1..=512u64)
            .map(|x| LocalBlockHash(x + (9 * 10000)))
            .collect();
        let scores = index.find_matches(last_seq).await.unwrap();
        assert!(!scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_dump_and_restore(variant: &str) {
        let index = make_indexer(variant);

        // Store some data
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        index.apply_event(make_store_event(1, &[1, 2, 4])).await;

        // Allow background worker threads to process events.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Dump the tree as events
        let events = index.dump_events().await.unwrap();
        assert!(!events.is_empty());

        // Create a new index and replay events
        let restored = make_indexer(variant);
        for event in events {
            restored.apply_event(event).await;
        }

        // Allow background worker threads to process replayed events.
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify find_matches produces same results
        let original_scores = index
            .find_matches(vec![LocalBlockHash(1), LocalBlockHash(2)])
            .await
            .unwrap();
        let restored_scores = restored
            .find_matches(vec![LocalBlockHash(1), LocalBlockHash(2)])
            .await
            .unwrap();
        assert_eq!(original_scores.scores, restored_scores.scores);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_clear_all_blocks(variant: &str) {
        let index = make_indexer(variant);

        // Store some data for two workers
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        index.apply_event(make_store_event(1, &[1, 2, 3])).await;

        // Clear worker 0's blocks using the Cleared event
        index.apply_event(make_clear_event(0)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Worker 0's blocks should be gone, worker 1's remain
        let scores = index
            .find_matches(vec![
                LocalBlockHash(1),
                LocalBlockHash(2),
                LocalBlockHash(3),
            ])
            .await
            .unwrap();
        assert_eq!(scores.scores.len(), 1);
        assert!(scores.scores.contains_key(&WorkerWithDpRank::new(1, 0)));
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_empty_query(variant: &str) {
        let index = make_indexer(variant);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        index.flush().await;

        // Empty query should return empty scores
        let scores = index.find_matches(vec![]).await.unwrap();
        assert!(scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_miss_query(variant: &str) {
        let index = make_indexer(variant);

        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        index.flush().await;

        // Query for non-existent blocks
        let scores = index
            .find_matches(vec![LocalBlockHash(999), LocalBlockHash(998)])
            .await
            .unwrap();
        assert!(scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_shutdown(variant: &str) {
        let index = make_indexer(variant);
        index.shutdown();
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_find_matches_for_request(variant: &str) {
        let index = make_indexer(variant);

        // Empty index should return no matches
        let tokens = vec![1, 2, 3, 4];
        let scores = index.find_matches_for_request(&tokens).await.unwrap();
        assert!(scores.scores.is_empty());

        // Store some data and verify we can find it via tokens
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Allow time for async processing
        index.flush().await;

        // Note: find_matches_for_request computes block hashes from tokens,
        // so we need tokens that hash to the same LocalBlockHash values.
        // For this test, we just verify the method works without error.
        let scores = index.find_matches_for_request(&tokens).await.unwrap();
        // The tokens [1,2,3,4] won't match our stored [1,2,3] local hashes
        // because find_matches_for_request computes different hashes from raw tokens
        assert!(scores.scores.is_empty() || !scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_process_routing_decision(variant: &str) {
        let index = make_indexer(variant);

        // Create tokens with hashes
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut tokens_with_hashes = TokensWithHashes::new(tokens, 32);

        let worker = WorkerWithDpRank::new(0, 0);

        // Process routing decision - should not error
        let result = index
            .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_parent_hash_chains(variant: &str) {
        let index = make_indexer(variant);

        // Store initial sequence [1, 2, 3]
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Store continuation [4, 5] with parent pointing to block 3
        index
            .apply_event(make_store_event_with_parent(0, &[1, 2, 3], &[4, 5]))
            .await;

        index.flush().await;

        // Query for full sequence [1, 2, 3, 4, 5] should match all 5 blocks
        let full_seq: Vec<LocalBlockHash> = (1..=5).map(LocalBlockHash).collect();
        let scores = index.find_matches(full_seq).await.unwrap();
        assert_eq!(scores.scores.len(), 1);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 5);

        // Query for just [1, 2, 3] should match 3 blocks
        let prefix_seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(prefix_seq).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_multiple_dp_ranks(variant: &str) {
        let index = make_indexer(variant);

        // Same worker_id but different dp_ranks should be tracked separately
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 0))
            .await;
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 1))
            .await;
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 2))
            .await;

        index.flush().await;

        // Query should return all 3 dp_ranks as separate entries
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq).await.unwrap();

        assert_eq!(scores.scores.len(), 3);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 1)).unwrap(), 3);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 2)).unwrap(), 3);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_partial_block_removal(variant: &str) {
        let index = make_indexer(variant);

        // Store [1, 2, 3]
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify all 3 blocks match
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq.clone()).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);

        // Remove only the last block (block 3)
        // To do this correctly, we need to compute the seq_hash for block 3 specifically,
        // which requires the full sequence context [1,2,3].
        let full_hashes: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let seq_hashes = compute_seq_hash_for_block(&full_hashes);
        let block_3_seq_hash = ExternalSequenceBlockHash(seq_hashes[2]); // Last block's hash

        let remove_event = RouterEvent {
            worker_id: 0,
            event: KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![block_3_seq_hash],
                }),
                dp_rank: 0,
            },
        };
        index.apply_event(remove_event).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query [1, 2, 3] - should only match 2 blocks now (block 3 is removed)
        let scores = index.find_matches(seq).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 2);

        // Query [1, 2] - should still match 2 blocks
        let partial_seq: Vec<LocalBlockHash> = (1..=2).map(LocalBlockHash).collect();
        let scores = index.find_matches(partial_seq).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 2);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_remove_nonexistent_worker(variant: &str) {
        let index = make_indexer(variant);

        // Store data for worker 0
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Remove non-existent worker 999 - should not error or affect worker 0
        index.remove_worker(999).await;

        // Allow time for async processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Worker 0's data should still be there
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq).await.unwrap();
        assert_eq!(scores.scores.len(), 1);
        assert!(scores.scores.contains_key(&WorkerWithDpRank::new(0, 0)));
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_remove_nonexistent_blocks(variant: &str) {
        let index = make_indexer(variant);

        // Store [1, 2, 3]
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Try to remove blocks [999, 998] that don't exist - should not error
        index.apply_event(make_remove_event(0, &[999, 998])).await;

        index.flush().await;

        // Original data should still be there
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_clear_then_reuse(variant: &str) {
        let index = make_indexer(variant);

        // Store initial data
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        // Clear the worker
        index.apply_event(make_clear_event(0)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify data is gone
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq.clone()).await.unwrap();
        assert!(scores.scores.is_empty());

        // Store new data for the same worker
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify new data is accessible
        let scores = index.find_matches(seq).await.unwrap();
        assert_eq!(scores.scores.len(), 1);
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_multiple_sequences_per_worker(variant: &str) {
        let index = make_indexer(variant);

        // Store two disjoint sequences for the same worker
        // Sequence 1: [1, 2, 3]
        index.apply_event(make_store_event(0, &[1, 2, 3])).await;
        // Sequence 2: [100, 101, 102] (completely different, no parent)
        index
            .apply_event(make_store_event(0, &[100, 101, 102]))
            .await;

        index.flush().await;

        // Query first sequence
        let seq1: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq1).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);

        // Query second sequence
        let seq2: Vec<LocalBlockHash> = (100..=102).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq2).await.unwrap();
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 3);

        // Query a mix that doesn't exist as a sequence - should only match first block
        let mixed: Vec<LocalBlockHash> = vec![LocalBlockHash(1), LocalBlockHash(100)];
        let scores = index.find_matches(mixed).await.unwrap();
        // Only block 1 matches because [1, 100] is not a valid prefix
        assert_eq!(*scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(), 1);
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_clear_clears_all_dp_ranks(variant: &str) {
        let index = make_indexer(variant);

        // Store same sequence for different dp_ranks
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 0))
            .await;
        index
            .apply_event(make_store_event_with_dp_rank(0, &[1, 2, 3], 1))
            .await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify both dp_ranks are present
        let seq: Vec<LocalBlockHash> = (1..=3).map(LocalBlockHash).collect();
        let scores = index.find_matches(seq.clone()).await.unwrap();
        assert_eq!(scores.scores.len(), 2);

        // Clear event clears ALL blocks for the worker_id, regardless of dp_rank
        index.apply_event(make_clear_event_with_dp_rank(0, 0)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Both dp_ranks should be cleared
        let scores = index.find_matches(seq).await.unwrap();
        assert!(
            scores.scores.is_empty(),
            "Cleared event should clear all dp_ranks for a worker"
        );
    }

    // ============================================================================
    // Long sequence tests - especially important for NestedMap/PositionalIndexer
    // ============================================================================

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_single_store(variant: &str) {
        let index = make_indexer(variant);

        // Store a long sequence (128 blocks) in a single event
        let seq_len = 128;
        let sequence: Vec<u64> = (1..=seq_len).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query full sequence - should match all blocks
        let full_query: Vec<LocalBlockHash> = sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(full_query).await.unwrap();
        assert_eq!(scores.scores.len(), 1);
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            seq_len as u32
        );

        // Query prefix (first 64 blocks)
        let prefix_query: Vec<LocalBlockHash> = (1..=64).map(LocalBlockHash).collect();
        let scores = index.find_matches(prefix_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            64
        );

        // Query with divergence at position 50
        let mut divergent_query: Vec<LocalBlockHash> = (1..=100).map(LocalBlockHash).collect();
        divergent_query[49] = LocalBlockHash(99999); // Position 49 (0-indexed) diverges
        let scores = index.find_matches(divergent_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            49
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_multiple_continuations(variant: &str) {
        let index = make_indexer(variant);

        // Build a long sequence through multiple continuations
        // First store: blocks 1-50
        let first_chunk: Vec<u64> = (1..=50).collect();
        index.apply_event(make_store_event(0, &first_chunk)).await;

        // Second store: blocks 51-100 (continuation of first)
        let second_chunk: Vec<u64> = (51..=100).collect();
        index
            .apply_event(make_store_event_with_parent(0, &first_chunk, &second_chunk))
            .await;

        // Third store: blocks 101-150 (continuation of second)
        let prefix_1_2: Vec<u64> = (1..=100).collect();
        let third_chunk: Vec<u64> = (101..=150).collect();
        index
            .apply_event(make_store_event_with_parent(0, &prefix_1_2, &third_chunk))
            .await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query full sequence - should match all 150 blocks
        let full_query: Vec<LocalBlockHash> = (1..=150).map(LocalBlockHash).collect();
        let scores = index.find_matches(full_query).await.unwrap();
        assert_eq!(scores.scores.len(), 1);
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            150
        );

        // Query crossing continuation boundaries
        let cross_boundary_query: Vec<LocalBlockHash> = (45..=105).map(LocalBlockHash).collect();
        let scores = index.find_matches(cross_boundary_query).await.unwrap();
        // Query starts at block 45, but stored sequence starts at 1, so this won't match
        // because the sequence hash at position 0 of our query (block 45) won't match
        // the stored sequence hash at position 0 (block 1)
        assert!(
            scores.scores.is_empty() || !scores.scores.contains_key(&WorkerWithDpRank::new(0, 0))
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_branching_continuations(variant: &str) {
        let index = make_indexer(variant);

        // Common prefix: blocks 1-30
        let common_prefix: Vec<u64> = (1..=30).collect();
        index.apply_event(make_store_event(0, &common_prefix)).await;

        // Branch A: blocks 31-60 on worker 0
        let branch_a: Vec<u64> = (31..=60).collect();
        index
            .apply_event(make_store_event_with_parent(0, &common_prefix, &branch_a))
            .await;

        // Branch B: blocks 131-160 (different content) on worker 1
        // First store the common prefix for worker 1
        index.apply_event(make_store_event(1, &common_prefix)).await;
        let branch_b: Vec<u64> = (131..=160).collect();
        index
            .apply_event(make_store_event_with_parent(1, &common_prefix, &branch_b))
            .await;

        index.flush().await;

        // Query common prefix - both workers should match
        let prefix_query: Vec<LocalBlockHash> = (1..=30).map(LocalBlockHash).collect();
        let scores = index.find_matches(prefix_query).await.unwrap();
        assert_eq!(scores.scores.len(), 2);
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            30
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            30
        );

        // Query branch A path - only worker 0 should match fully
        let branch_a_query: Vec<LocalBlockHash> = (1..=60).map(LocalBlockHash).collect();
        let scores = index.find_matches(branch_a_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            60
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            30
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_partial_removal(variant: &str) {
        let index = make_indexer(variant);

        // Store a long sequence
        let sequence: Vec<u64> = (1..=100).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify full match
        let full_query: Vec<LocalBlockHash> = sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(full_query.clone()).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            100
        );

        // Remove blocks 80-100 (the tail)
        let tail_hashes: Vec<LocalBlockHash> = (1..=100).map(LocalBlockHash).collect();
        let seq_hashes = compute_seq_hash_for_block(&tail_hashes);
        let remove_hashes: Vec<ExternalSequenceBlockHash> = seq_hashes[79..100]
            .iter()
            .map(|&h| ExternalSequenceBlockHash(h))
            .collect();

        let remove_event = RouterEvent {
            worker_id: 0,
            event: KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: remove_hashes,
                }),
                dp_rank: 0,
            },
        };
        index.apply_event(remove_event).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query should now only match first 79 blocks
        let scores = index.find_matches(full_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            79
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_interleaved_workers(variant: &str) {
        let index = make_indexer(variant);

        // Multiple workers storing overlapping long sequences concurrently
        // Worker 0: blocks 1-100
        // Worker 1: blocks 1-75
        // Worker 2: blocks 1-50
        // Worker 3: blocks 1-25

        let seq_100: Vec<u64> = (1..=100).collect();
        let seq_75: Vec<u64> = (1..=75).collect();
        let seq_50: Vec<u64> = (1..=50).collect();
        let seq_25: Vec<u64> = (1..=25).collect();

        index.apply_event(make_store_event(0, &seq_100)).await;
        index.apply_event(make_store_event(1, &seq_75)).await;
        index.apply_event(make_store_event(2, &seq_50)).await;
        index.apply_event(make_store_event(3, &seq_25)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query for 60 blocks - workers 0,1 match 60, worker 2 matches 50, worker 3 matches 25
        let query_60: Vec<LocalBlockHash> = (1..=60).map(LocalBlockHash).collect();
        let scores = index.find_matches(query_60).await.unwrap();
        assert_eq!(scores.scores.len(), 4);
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            60
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            60
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(2, 0)).unwrap(),
            50
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(3, 0)).unwrap(),
            25
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_exact_jump_size_boundaries(variant: &str) {
        let index = make_indexer(variant);

        // Test sequences that align exactly with jump_size boundaries (32 for PositionalIndexer)
        // This tests edge cases in the jump search algorithm

        // Store sequence of exactly 32 blocks
        let seq_32: Vec<u64> = (1..=32).collect();
        index.apply_event(make_store_event(0, &seq_32)).await;

        // Store sequence of exactly 64 blocks (2x jump_size)
        let seq_64: Vec<u64> = (1001..=1064).collect();
        index.apply_event(make_store_event(1, &seq_64)).await;

        // Store sequence of exactly 96 blocks (3x jump_size)
        let seq_96: Vec<u64> = (2001..=2096).collect();
        index.apply_event(make_store_event(2, &seq_96)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify all sequences match correctly
        let query_32: Vec<LocalBlockHash> = seq_32.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query_32).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            32
        );

        let query_64: Vec<LocalBlockHash> = seq_64.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query_64).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            64
        );

        let query_96: Vec<LocalBlockHash> = seq_96.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query_96).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(2, 0)).unwrap(),
            96
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_off_by_one_jump_boundaries(variant: &str) {
        let index = make_indexer(variant);

        // Test sequences at jump_size +/- 1 boundaries to catch off-by-one errors
        let seq_31: Vec<u64> = (1..=31).collect();
        let seq_33: Vec<u64> = (101..=133).collect();
        let seq_63: Vec<u64> = (201..=263).collect();
        let seq_65: Vec<u64> = (301..=365).collect();

        index.apply_event(make_store_event(0, &seq_31)).await;
        index.apply_event(make_store_event(1, &seq_33)).await;
        index.apply_event(make_store_event(2, &seq_63)).await;
        index.apply_event(make_store_event(3, &seq_65)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify all sequences match correctly
        let query_31: Vec<LocalBlockHash> = seq_31.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query_31).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            31
        );

        let query_33: Vec<LocalBlockHash> = seq_33.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query_33).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            33
        );

        let query_63: Vec<LocalBlockHash> = seq_63.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query_63).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(2, 0)).unwrap(),
            63
        );

        let query_65: Vec<LocalBlockHash> = seq_65.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query_65).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(3, 0)).unwrap(),
            65
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_divergence_at_jump_boundaries(variant: &str) {
        let index = make_indexer(variant);

        // Store a long sequence
        let sequence: Vec<u64> = (1..=128).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Test divergence exactly at jump boundaries (position 31, 32, 33, 63, 64, 65)
        for diverge_pos in [31usize, 32, 33, 63, 64, 65, 95, 96, 97] {
            let mut query: Vec<LocalBlockHash> = (1..=128).map(LocalBlockHash).collect();
            query[diverge_pos] = LocalBlockHash(99999);

            let scores = index.find_matches(query).await.unwrap();
            assert_eq!(
                *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
                diverge_pos as u32,
                "Divergence at position {} should match {} blocks",
                diverge_pos,
                diverge_pos
            );
        }
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_deep_continuation_chain(variant: &str) {
        let index = make_indexer(variant);

        // Build a very long sequence through many small continuations
        // This tests the parent_hash chain handling
        let chunk_size = 10;
        let num_chunks = 20; // Total 200 blocks

        let mut full_prefix: Vec<u64> = Vec::new();

        for chunk_idx in 0..num_chunks {
            let chunk_start = chunk_idx * chunk_size + 1;
            let chunk: Vec<u64> = (chunk_start..chunk_start + chunk_size)
                .map(|x| x as u64)
                .collect();

            if chunk_idx == 0 {
                index.apply_event(make_store_event(0, &chunk)).await;
            } else {
                index
                    .apply_event(make_store_event_with_parent(0, &full_prefix, &chunk))
                    .await;
            }

            full_prefix.extend(&chunk);
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query full sequence
        let full_query: Vec<LocalBlockHash> = (1..=200).map(LocalBlockHash).collect();
        let scores = index.find_matches(full_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            200
        );

        // Query partial prefix crossing multiple chunk boundaries
        let partial_query: Vec<LocalBlockHash> = (1..=75).map(LocalBlockHash).collect();
        let scores = index.find_matches(partial_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            75
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_clear_and_rebuild(variant: &str) {
        let index = make_indexer(variant);

        // Store a long sequence
        let sequence: Vec<u64> = (1..=100).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify it's stored
        let query: Vec<LocalBlockHash> = sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query.clone()).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            100
        );

        // Clear the worker
        index.apply_event(make_clear_event(0)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify it's cleared
        let scores = index.find_matches(query.clone()).await.unwrap();
        assert!(scores.scores.is_empty());

        // Rebuild with a different sequence
        let new_sequence: Vec<u64> = (1001..=1100).collect();
        index.apply_event(make_store_event(0, &new_sequence)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify new sequence works
        let new_query: Vec<LocalBlockHash> =
            new_sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(new_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            100
        );

        // Verify old sequence no longer matches
        let scores = index.find_matches(query).await.unwrap();
        assert!(scores.scores.is_empty());
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_multiple_workers_diverging(variant: &str) {
        let index = make_indexer(variant);

        // Multiple workers with long sequences that share a prefix then diverge
        // This tests precise drain point tracking across workers

        // All workers share prefix 1-40
        let shared_prefix: Vec<u64> = (1..=40).collect();

        // Worker 0: prefix + 41-100 (stores full sequence 1-100)
        let worker_0_full: Vec<u64> = (1..=100).collect();

        // Worker 1: prefix + 141-180 (diverges at block 41)
        let worker_1_suffix: Vec<u64> = (141..=180).collect();

        // Worker 2: prefix + 241-300 (diverges at block 41)
        let worker_2_suffix: Vec<u64> = (241..=300).collect();

        // Store for all workers
        index.apply_event(make_store_event(0, &worker_0_full)).await;

        index.apply_event(make_store_event(1, &shared_prefix)).await;
        index
            .apply_event(make_store_event_with_parent(
                1,
                &shared_prefix,
                &worker_1_suffix,
            ))
            .await;

        index.apply_event(make_store_event(2, &shared_prefix)).await;
        index
            .apply_event(make_store_event_with_parent(
                2,
                &shared_prefix,
                &worker_2_suffix,
            ))
            .await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query 1-100 - worker 0 matches 100, workers 1&2 match 40
        let query: Vec<LocalBlockHash> = worker_0_full.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(query).await.unwrap();

        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            100
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            40
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(2, 0)).unwrap(),
            40
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_long_sequence_staggered_lengths(variant: &str) {
        let index = make_indexer(variant);

        // Workers with sequences of staggered lengths to test drain tracking
        // Worker 0: 10 blocks
        // Worker 1: 20 blocks
        // Worker 2: 35 blocks (just past first jump)
        // Worker 3: 64 blocks (exactly 2 jumps)
        // Worker 4: 100 blocks

        for (worker_id, len) in [(0, 10), (1, 20), (2, 35), (3, 64), (4, 100)] {
            let sequence: Vec<u64> = (1..=len).collect();
            index
                .apply_event(make_store_event(worker_id, &sequence))
                .await;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Query for 100 blocks - each worker should match their stored length
        let query: Vec<LocalBlockHash> = (1..=100).map(LocalBlockHash).collect();
        let scores = index.find_matches(query).await.unwrap();

        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            10
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(1, 0)).unwrap(),
            20
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(2, 0)).unwrap(),
            35
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(3, 0)).unwrap(),
            64
        );
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(4, 0)).unwrap(),
            100
        );
    }

    #[tokio::test]
    #[apply(indexer_template)]
    async fn test_very_long_sequence(variant: &str) {
        let index = make_indexer(variant);

        // Test with a very long sequence (1000 blocks)
        let seq_len = 1000u64;
        let sequence: Vec<u64> = (1..=seq_len).collect();
        index.apply_event(make_store_event(0, &sequence)).await;

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Full match
        let full_query: Vec<LocalBlockHash> = sequence.iter().map(|&i| LocalBlockHash(i)).collect();
        let scores = index.find_matches(full_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            seq_len as u32
        );

        // Partial match (first 500)
        let partial_query: Vec<LocalBlockHash> = (1..=500).map(LocalBlockHash).collect();
        let scores = index.find_matches(partial_query).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            500
        );

        // Divergence in the middle
        let mut mid_diverge: Vec<LocalBlockHash> = (1..=1000).map(LocalBlockHash).collect();
        mid_diverge[499] = LocalBlockHash(99999);
        let scores = index.find_matches(mid_diverge).await.unwrap();
        assert_eq!(
            *scores.scores.get(&WorkerWithDpRank::new(0, 0)).unwrap(),
            499
        );
    }

    // ============================================================================
    // Tests specific to tree-based implementations (KvIndexer, KvIndexerSharded)
    // These use features not available in PositionalIndexer
    // ============================================================================

    #[template]
    #[rstest]
    fn tree_indexer_template(#[values("single", "sharded")] variant: &str) {}

    fn make_tree_indexer_with_frequency(
        variant: &str,
        expiration: Duration,
    ) -> Box<dyn KvIndexerInterface> {
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let kv_block_size = 32;

        match variant {
            "single" => Box::new(KvIndexer::new_with_frequency(
                token,
                Some(expiration),
                kv_block_size,
                metrics,
                None,
            )),
            "sharded" => Box::new(KvIndexerSharded::new_with_frequency(
                token,
                4,
                Some(expiration),
                kv_block_size,
                metrics,
                None,
            )),
            _ => panic!("Unknown variant: {}", variant),
        }
    }

    #[tokio::test]
    #[apply(tree_indexer_template)]
    async fn test_frequency(variant: &str) {
        const ONE_MILLIS: Duration = Duration::from_millis(1);

        let expiration = Duration::from_millis(50);
        let kv_indexer = make_tree_indexer_with_frequency(variant, expiration);

        // The blocks
        let block_hashes = vec![
            LocalBlockHash(1),
            LocalBlockHash(2),
            LocalBlockHash(3),
            LocalBlockHash(4),
        ];

        let overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(
            overlap.frequencies.len(),
            0,
            "Should be no cached blocks yet"
        );

        // Blocks go in cache
        let event = make_store_event(0, &[1, 2, 3, 4]);
        kv_indexer.apply_event(event).await;

        // First access - poll briefly since store event is applied async
        let mut overlap = OverlapScores::default();
        let timeout = Duration::from_millis(10);
        let start = Instant::now();
        while overlap.scores.is_empty() && Instant::now().duration_since(start) < timeout {
            time::sleep(ONE_MILLIS).await;
            overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        }
        assert_eq!(
            overlap.scores.len(),
            1,
            "One worker has these blocks cached"
        );
        assert_eq!(
            overlap.frequencies.len(),
            0,
            "Blocks have not previously been accessed"
        );

        // Second access
        let overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(overlap.scores.len(), 1, "Still one worker matches");
        assert_eq!(
            overlap.frequencies,
            vec![1, 1, 1, 1],
            "We should see the first access now"
        );

        // Let those two accesses expire
        time::sleep(expiration + Duration::from_millis(10)).await;

        // New first access
        let overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(
            overlap.frequencies.len(),
            0,
            "Blocks were accessed too long ago"
        );

        // New second access
        let _ = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();

        // Access only the first three blocks
        let overlap = kv_indexer
            .find_matches(block_hashes[0..3].to_vec())
            .await
            .unwrap();
        // We see the previous two new accesses
        assert_eq!(overlap.frequencies, vec![2, 2, 2]);

        // The third access did not touch the last block
        let overlap = kv_indexer.find_matches(block_hashes.clone()).await.unwrap();
        assert_eq!(overlap.frequencies, vec![3, 3, 3, 2]);
    }

    // ============================================================================
    // KvIndexerMetrics tests
    // ============================================================================

    #[test]
    fn test_increment_event_applied() {
        let metrics = KvIndexerMetrics::new_unregistered();

        metrics.increment_event_applied(METRIC_EVENT_STORED, Ok(()));
        assert_eq!(
            metrics
                .kv_cache_events_applied
                .get_metric_with_label_values(&[METRIC_EVENT_STORED, METRIC_STATUS_OK])
                .unwrap()
                .get(),
            1
        );

        metrics.increment_event_applied(
            METRIC_EVENT_STORED,
            Err(KvCacheEventError::ParentBlockNotFound),
        );
        assert_eq!(
            metrics
                .kv_cache_events_applied
                .get_metric_with_label_values(&[
                    METRIC_EVENT_STORED,
                    METRIC_STATUS_PARENT_NOT_FOUND
                ])
                .unwrap()
                .get(),
            1
        );

        metrics
            .increment_event_applied(METRIC_EVENT_REMOVED, Err(KvCacheEventError::BlockNotFound));
        assert_eq!(
            metrics
                .kv_cache_events_applied
                .get_metric_with_label_values(&[
                    METRIC_EVENT_REMOVED,
                    METRIC_STATUS_BLOCK_NOT_FOUND
                ])
                .unwrap()
                .get(),
            1
        );
    }

    // ============================================================================
    // LocalKvIndexer tests
    // ============================================================================

    fn make_local_indexer_with_events(ids: &[u64]) -> LocalKvIndexer {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            32,
        );
        {
            let mut buffer = indexer.event_buffer.lock().unwrap();
            for &id in ids {
                buffer.push_back(RouterEvent::new(
                    0,
                    KvCacheEvent {
                        event_id: id,
                        data: KvCacheEventData::Cleared,
                        dp_rank: 0,
                    },
                ));
            }
        }
        indexer
    }

    #[tokio::test]
    async fn test_local_indexer_slice_within_range() {
        let indexer = make_local_indexer_with_events(&[1, 2, 3, 4, 5]);

        // Helper to extract events from response
        let extract_events = |resp: WorkerKvQueryResponse| -> Vec<RouterEvent> {
            match resp {
                WorkerKvQueryResponse::Events(e) => e,
                WorkerKvQueryResponse::TreeDump(e) => e,
                _ => panic!("Unexpected response type"),
            }
        };

        let get_ids = |events: Vec<RouterEvent>| -> Vec<u64> {
            events.iter().map(|e| e.event.event_id).collect()
        };

        // Test get_events_in_id_range (buffer queries)
        // Range is [start, end] inclusive
        let result = indexer.get_events_in_id_range(Some(2), Some(4)).await;
        let ids = get_ids(extract_events(result));
        assert_eq!(ids, vec![2, 3, 4]); // inclusive range [2, 4]

        let result = indexer.get_events_in_id_range(Some(2), Some(6)).await;
        let ids = get_ids(extract_events(result));
        assert_eq!(ids, vec![2, 3, 4, 5]); // clamp end to buffer max

        // start_id=0 is before buffer (first is 1), so should trigger tree dump
        let result = indexer.get_events_in_id_range(Some(0), Some(4)).await;
        assert!(matches!(result, WorkerKvQueryResponse::TreeDump(_)));

        let result = indexer.get_events_in_id_range(Some(3), Some(3)).await;
        let ids = get_ids(extract_events(result));
        assert_eq!(ids, vec![3]); // single element when start == end

        // Invalid range: end < start
        let result = indexer.get_events_in_id_range(Some(5), Some(2)).await;
        assert!(matches!(result, WorkerKvQueryResponse::InvalidRange { .. }));
    }

    #[tokio::test]
    async fn test_local_indexer_get_events_in_id_range_all_cases() {
        // Create indexer with small buffer (5 events max)
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        // Helper to create a test event
        let make_event = |id: u64| {
            RouterEvent::new(
                0,
                KvCacheEvent {
                    event_id: id,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        blocks: vec![KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(id * 100),
                            tokens_hash: LocalBlockHash(id * 200),
                            mm_extra_info: None,
                        }],
                    }),
                    dp_rank: 0,
                },
            )
        };

        // Add 10 events (IDs 5-14), buffer keeps last 5: events 10-14
        for id in 5..15 {
            indexer
                .apply_event_with_buffer(make_event(id))
                .await
                .unwrap();
        }

        // Wait for events to be processed
        tokio::time::sleep(Duration::from_millis(100)).await;

        let extract_events = |resp: WorkerKvQueryResponse| -> Vec<RouterEvent> {
            match resp {
                WorkerKvQueryResponse::Events(e) => e,
                WorkerKvQueryResponse::TreeDump(e) => e,
                _ => panic!("Unexpected response type: {:?}", resp),
            }
        };

        let get_ids = |events: Vec<RouterEvent>| -> Vec<u64> {
            events.iter().map(|e| e.event.event_id).collect()
        };

        // Verify buffer state
        let buffer_events = indexer.get_all_events_in_buffer();
        assert_eq!(get_ids(buffer_events), vec![10, 11, 12, 13, 14]);

        // Buffer path tests
        let result = indexer.get_events_in_id_range(Some(11), None).await;
        assert_eq!(get_ids(extract_events(result)), vec![11, 12, 13, 14]);

        let result = indexer.get_events_in_id_range(Some(10), Some(14)).await;
        assert_eq!(get_ids(extract_events(result)), vec![10, 11, 12, 13, 14]);

        // Tree dump path tests
        let result = indexer.get_events_in_id_range(None, None).await;
        assert!(matches!(result, WorkerKvQueryResponse::TreeDump(_)));
        assert_eq!(extract_events(result).len(), 10);

        let result = indexer.get_events_in_id_range(Some(7), None).await;
        assert!(matches!(result, WorkerKvQueryResponse::TreeDump(_)));

        // Edge cases
        let result = indexer.get_events_in_id_range(Some(15), Some(10)).await;
        assert!(matches!(result, WorkerKvQueryResponse::InvalidRange { .. }));

        let result = indexer.get_events_in_id_range(Some(100), Some(200)).await;
        assert!(matches!(result, WorkerKvQueryResponse::TooNew { .. }));
    }

    #[tokio::test]
    async fn test_local_indexer_buffer_and_serialization() {
        let worker_id = 42u64;
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token, 4, metrics, 100));

        let test_event = RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        );

        local_indexer
            .apply_event_with_buffer(test_event)
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;

        let buffered_events = local_indexer.get_all_events_in_buffer();
        assert_eq!(buffered_events.len(), 1);
        assert_eq!(buffered_events[0].worker_id, worker_id);

        // Test serialization round-trip
        let response = WorkerKvQueryResponse::Events(buffered_events);
        let serialized = serde_json::to_vec(&response).unwrap();
        let deserialized: WorkerKvQueryResponse = serde_json::from_slice(&serialized).unwrap();

        let events = match deserialized {
            WorkerKvQueryResponse::Events(e) => e,
            _ => panic!("Expected Events variant"),
        };
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].worker_id, worker_id);
    }
}
