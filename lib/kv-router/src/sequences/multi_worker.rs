// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-worker extension of [`ActiveSequences`] with per-worker `parking_lot::RwLock` for
//! fine-grained concurrent access, with pluggable event publishing and metric observation via
//! traits.
//!
//! The two traits [`SequencePublisher`] and [`SequenceSubscriber`] abstract the runtime-specific
//! transport (e.g., NATS EventPublisher, Prometheus gauges) so that all business logic lives in
//! this crate while the runtime glue stays in `lib/llm`.

use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};
use tokio::sync::watch;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::prefill_tracker::PrefillTimeLoad;
#[cfg(any(test, feature = "bench"))]
use super::prompt_membership_trie::lookup_live_hashes;
use super::prompt_registry::{PromptRegistry, WorkerLoadSnapshot};
use super::request_maps::RequestIndex;
use super::single::{ActiveSequences, PromptMembershipDelta, RequestId};
use super::topology::{WorkerDpRange, WorkerTable, WorkerTopologyChange, WorkerTopologyError};
use super::{PotentialLoadMaps, PrefillTokenDeltas, WorkerLoadProjection};
use crate::protocols::{
    ActiveLoad, ActiveSequenceEvent, ActiveSequenceEventData, PrefillLoadHint, WorkerId,
    WorkerWithDpRank,
};

// How often we force expire stale requests across all workers. See the comment
// in ActiveSequencesMultiWorker::force_expire_requests_across_all_workers for
// more details.
const FORCE_EXPIRE_REQUESTS_ACROSS_ALL_WORKERS_INTERVAL: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Abstraction over event publishing and metrics observation.
///
/// Implementations provide the runtime-specific transport (e.g., NATS EventPublisher,
/// Prometheus gauges) while the business logic in [`ActiveSequencesMultiWorker`] stays
/// runtime-agnostic.
pub trait SequencePublisher: Send + Sync {
    /// Publish a replica-sync event to peer routers.
    fn publish_event(
        &self,
        event: &ActiveSequenceEvent,
    ) -> impl Future<Output = anyhow::Result<()>> + Send;

    /// Fire-and-forget publish of an [`ActiveLoad`] metric payload.
    fn publish_load(&self, load: ActiveLoad);

    /// Fire-and-forget publish of a batch of [`ActiveLoad`] metric payloads.
    fn publish_load_batch(&self, loads: Vec<ActiveLoad>) {
        for load in loads {
            self.publish_load(load);
        }
    }

    /// Record per-worker load in Prometheus gauges.
    fn observe_load(
        &self,
        worker: &WorkerWithDpRank,
        worker_type: &str,
        blocks: usize,
        tokens: usize,
    );

    /// Observe that a worker/dp_rank is currently registered in the router.
    fn observe_worker_registered(&self, _worker: &WorkerWithDpRank, _worker_type: &str) {}

    /// Observe that a worker/dp_rank was removed from the router.
    fn observe_worker_removed(&self, _worker: &WorkerWithDpRank, _worker_type: &str) {}
}

/// Abstraction over event subscription for replica sync.
pub trait SequenceSubscriber: Send {
    /// Receive the next replica-sync event, or `None` if the stream is closed.
    fn next_event(
        &mut self,
    ) -> impl Future<Output = Option<anyhow::Result<ActiveSequenceEvent>>> + Send;

    /// Poll for an event that is already ready without waiting.
    fn poll_next_event(
        &mut self,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<anyhow::Result<ActiveSequenceEvent>>> {
        Poll::Pending
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Controls how replica-sync events handle workers missing from local topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaWorkerPolicy {
    /// Preserve the legacy router behavior by creating the exact missing worker rank.
    LazyRegister,
    /// Drop replica events until the worker rank has been registered locally.
    RequireRegistered,
}

/// Errors that can occur during sequence management operations.
#[derive(Debug, thiserror::Error)]
pub enum SequenceError {
    #[error("Worker {worker:?} not found")]
    WorkerNotFound { worker: WorkerWithDpRank },

    #[error("Request {request_id} already exists (assigned to worker {worker:?})")]
    DuplicateRequest {
        request_id: String,
        worker: WorkerWithDpRank,
    },

    #[error("Request {request_id} not found")]
    RequestNotFound { request_id: String },

    #[error("Failed to publish replica-sync event: {0}")]
    ReplicaSyncPublishFailed(String),
}

/// Bundled parameters for adding a request to the sequence tracker.
pub struct SequenceRequest {
    pub request_id: RequestId,
    pub token_sequence: Option<Vec<SequenceHash>>,
    pub track_prefill_tokens: bool,
    pub expected_output_tokens: Option<u32>,
    pub prefill_load_hint: Option<PrefillLoadHint>,
    pub worker: WorkerWithDpRank,
    pub lora_name: Option<String>,
}

/// Multi-worker extension of [`ActiveSequences`] with per-worker `parking_lot::RwLock` for
/// fine-grained concurrent access.
///
/// The outer `RwLock<WorkerTable>` is held only during sync blocks (never across `.await`),
/// while each worker slot has its own `RwLock<ActiveSequences>` for per-worker fine-grained
/// locking with cache-friendly Vec layout.
///
/// Generic over `P: SequencePublisher` to decouple from runtime-specific event transport
/// and metrics infrastructure.
pub struct ActiveSequencesMultiWorker<P: SequencePublisher> {
    pub(super) workers: RwLock<WorkerTable>,
    pub(super) request_index: RequestIndex,
    pub(super) prompt_registry: PromptRegistry,
    block_size: usize,
    pub(super) router_id: u64,
    pub(super) publisher: Arc<P>,
    remote_state_updates: watch::Sender<()>,
    #[cfg(test)]
    remote_state_update_count: AtomicUsize,
    replica_sync: bool,
    pub(super) replica_worker_policy: ReplicaWorkerPolicy,
    worker_type: &'static str,
}

impl<P: SequencePublisher + 'static> ActiveSequencesMultiWorker<P> {
    /// Create a new multi-worker sequence tracker.
    ///
    /// `dp_sizes` maps worker IDs to their data-parallel size (number of dp_ranks).
    pub fn new(
        publisher: P,
        block_size: usize,
        dp_range: HashMap<u64, (u32, u32)>,
        replica_sync: bool,
        router_id: u64,
        worker_type: &'static str,
    ) -> Self {
        Self::new_with_replica_worker_policy(
            publisher,
            block_size,
            dp_range,
            replica_sync,
            router_id,
            worker_type,
            ReplicaWorkerPolicy::LazyRegister,
        )
    }

    /// Create a tracker with an explicit worker-admission policy for replica events.
    pub fn new_with_replica_worker_policy(
        publisher: P,
        block_size: usize,
        dp_range: HashMap<u64, (u32, u32)>,
        replica_sync: bool,
        router_id: u64,
        worker_type: &'static str,
        replica_worker_policy: ReplicaWorkerPolicy,
    ) -> Self {
        assert!(block_size > 0, "block_size must be greater than 0");
        let (remote_state_updates, _) = watch::channel(());
        let workers = WorkerTable::new(block_size, &dp_range);
        let initial_workers: Vec<_> = workers.workers().collect();
        let prompt_registry = PromptRegistry::new(initial_workers.iter().copied());
        let publisher = Arc::new(publisher);
        for worker in &initial_workers {
            publisher.observe_worker_registered(worker, worker_type);
        }

        Self {
            workers: RwLock::new(workers),
            request_index: RequestIndex::default(),
            prompt_registry,
            block_size,
            router_id,
            publisher,
            remote_state_updates,
            #[cfg(test)]
            remote_state_update_count: AtomicUsize::new(0),
            replica_sync,
            replica_worker_policy,
            worker_type,
        }
    }

    #[cfg(any(test, feature = "bench"))]
    pub fn assert_completely_drained(&self, decay_now: Instant) {
        let active_blocks = self.active_blocks();
        assert!(
            active_blocks.values().all(|&count| count == 0),
            "expected all workers to have zero active blocks, got {active_blocks:?}",
        );

        let active_tokens = self.active_tokens(decay_now);
        assert!(
            active_tokens.values().all(|&count| count == 0),
            "expected all workers to have zero active tokens, got {active_tokens:?}",
        );

        let active_requests = self.active_request_counts();
        assert!(
            active_requests.values().all(|&count| count == 0),
            "expected all workers to have zero active requests, got {active_requests:?}",
        );

        assert!(
            self.request_index.is_empty(),
            "expected no active request-to-worker mappings, found {}",
            self.request_index.worker_len(),
        );
        assert!(
            self.get_active_lora_counts().is_empty(),
            "expected no active LoRA counts, found {:?}",
            self.get_active_lora_counts(),
        );
        assert!(
            self.prompt_registry.is_block_index_empty(),
            "expected reverse block index to be empty after drain",
        );

        let trie_lookup_live_hashes: Vec<_> = {
            let table = self.workers.read();
            table
                .slots
                .iter()
                .filter_map(|slot| {
                    let live_hashes = lookup_live_hashes(&slot.trie_lookup);
                    (!live_hashes.is_empty()).then_some((slot.worker, live_hashes))
                })
                .collect()
        };
        assert!(
            trie_lookup_live_hashes.is_empty(),
            "expected all worker trie lookups to reference only dead nodes after drain, found {:?}",
            trie_lookup_live_hashes,
        );
    }

    fn publish_worker_load_snapshot(
        &self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
        decay_now: Instant,
    ) {
        let active_load = self.observe_worker_load_snapshot(worker, load, decay_now);
        self.publisher.publish_load(active_load);
    }

    pub(super) fn observe_worker_load_snapshot(
        &self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
        decay_now: Instant,
    ) -> ActiveLoad {
        let active_blocks = load.active_blocks;
        let active_tokens = load.active_tokens(decay_now);

        self.publisher
            .observe_load(&worker, self.worker_type, active_blocks, active_tokens);

        ActiveLoad {
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            active_decode_blocks: Some(active_blocks as u64),
            active_prefill_tokens: Some(active_tokens as u64),
            kv_used_blocks: None,
        }
    }

    fn spawn_publish_event(&self, event: ActiveSequenceEvent) {
        if !self.replica_sync {
            return;
        }

        // TODO: Publish explicit prompt-load decay timestamps with these events so peer routers
        // can mirror the same oldest-prefill anchor instead of approximating from receive time.
        let publisher = Arc::clone(&self.publisher);
        tokio::spawn(async move {
            if let Err(e) = publisher.publish_event(&event).await {
                tracing::error!(
                    request_id = %event.request_id,
                    worker = ?event.worker,
                    "failed to publish active sequence event: {e}"
                );
            }
        });
    }

    /// Subscribe to remote lifecycle updates that were applied through replica sync.
    ///
    /// The queue uses this to react immediately when a peer router frees prompt
    /// capacity locally.
    pub fn subscribe_remote_state_changes(&self) -> watch::Receiver<()> {
        self.remote_state_updates.subscribe()
    }

    pub(super) fn notify_remote_state_update(&self) {
        #[cfg(test)]
        self.remote_state_update_count
            .fetch_add(1, Ordering::Relaxed);
        let _ = self.remote_state_updates.send(());
    }

    #[cfg(test)]
    fn remote_state_update_count(&self) -> usize {
        self.remote_state_update_count.load(Ordering::Relaxed)
    }

    /// Register one worker and reject duplicate worker IDs.
    pub fn register_worker(&self, range: WorkerDpRange) -> Result<(), WorkerTopologyError> {
        let change = {
            let mut table = self.workers.write();
            table.register_worker(self.block_size, range)?
        };

        for worker in &change.added {
            tracing::debug!("Registering worker {:?}", worker);
        }
        self.apply_worker_topology_change(change);
        Ok(())
    }

    /// Add or update one worker without changing unrelated workers.
    ///
    /// This supports external routing calls that provide different worker
    /// subsets over time. Updating a worker replaces only that worker's DP
    /// ranks and preserves unrelated lazily-created slots.
    pub fn upsert_worker(&self, range: WorkerDpRange) -> Result<(), WorkerTopologyError> {
        let change = {
            let mut table = self.workers.write();
            table.upsert_worker(self.block_size, range)?
        };

        for removed in &change.removed {
            tracing::debug!("Removing external worker rank {:?}", removed.worker);
        }
        for worker in &change.added {
            tracing::debug!("Registering external worker rank {:?}", worker);
        }
        self.apply_worker_topology_change(change);
        Ok(())
    }

    /// Unregister one worker and all of its DP ranks.
    pub fn unregister_worker(&self, worker_id: WorkerId) -> Result<(), WorkerTopologyError> {
        let change = {
            let mut table = self.workers.write();
            table.unregister_worker(self.block_size, worker_id)?
        };

        for removed in &change.removed {
            tracing::warn!("Removing worker {:?}", removed.worker);
        }
        self.apply_worker_topology_change(change);
        Ok(())
    }

    /// Replace the complete authoritative worker topology.
    ///
    /// Workers absent from `ranges`, including lazily-created slots, are removed.
    pub fn reconcile_workers(
        &self,
        ranges: impl IntoIterator<Item = WorkerDpRange>,
    ) -> Result<(), WorkerTopologyError> {
        let change = {
            let mut table = self.workers.write();
            table.reconcile(self.block_size, ranges.into_iter().collect())?
        };

        for removed in &change.removed {
            tracing::warn!("Removing worker {:?}", removed.worker);
        }
        for worker in &change.added {
            tracing::warn!("Adding worker {:?}", worker);
        }

        self.apply_worker_topology_change(change);
        Ok(())
    }

    /// Return the authoritative worker ranges, sorted by worker ID.
    pub fn worker_ranges(&self) -> Vec<WorkerDpRange> {
        self.workers.read().worker_ranges()
    }

    /// Return whether the authoritative topology contains any workers.
    pub fn has_registered_workers(&self) -> bool {
        self.workers.read().has_registered_workers()
    }

    fn apply_worker_topology_change(&self, change: WorkerTopologyChange) {
        for removed in &change.removed {
            self.request_index.remove_worker_requests(removed.worker);
            self.publisher
                .observe_worker_removed(&removed.worker, self.worker_type);
        }
        for worker in &change.added {
            self.publisher
                .observe_worker_registered(worker, self.worker_type);
        }
        self.prompt_registry.apply_topology_change(change);
    }

    pub fn add_request(
        &self,
        req: SequenceRequest,
        decay_now: Instant,
    ) -> Result<(), SequenceError> {
        self.add_request_impl(req, decay_now, true)
    }

    /// Add a request only when its worker is already registered.
    ///
    /// External control planes use this stricter variant so a request racing
    /// worker removal cannot lazily recreate the removed worker.
    pub fn add_request_if_registered(
        &self,
        req: SequenceRequest,
        decay_now: Instant,
    ) -> Result<(), SequenceError> {
        self.add_request_impl(req, decay_now, false)
    }

    fn add_request_impl(
        &self,
        req: SequenceRequest,
        decay_now: Instant,
        lazily_register_worker: bool,
    ) -> Result<(), SequenceError> {
        let event = self.replica_sync.then(|| ActiveSequenceEvent {
            request_id: req.request_id.clone(),
            worker: req.worker,
            data: ActiveSequenceEventData::AddRequest {
                token_sequence: req.token_sequence.clone(),
                track_prefill_tokens: req.track_prefill_tokens,
                expected_output_tokens: req.expected_output_tokens,
                prefill_load_hint: req.prefill_load_hint,
            },
            router_id: self.router_id,
            lora_name: req.lora_name.clone(),
        });
        self.add_request_local(req, decay_now, lazily_register_worker)?;
        if let Some(event) = event {
            self.spawn_publish_event(event);
        }
        Ok(())
    }

    /// Free all blocks associated with a request.
    ///
    /// Note: This operation is idempotent. Calling it multiple times for the same request
    /// will log a warning but not return an error (double free is allowed).
    ///
    /// This also performs the underlying prefill-complete cleanup via
    /// [`ActiveSequences::free`], so callers do not need to call
    /// [`Self::mark_prefill_completed`] before freeing a completed request.
    pub fn free(&self, request_id: &RequestId, decay_now: Instant) -> Result<(), SequenceError> {
        match self.mutate_request_worker_prompt_state(
            request_id,
            decay_now,
            ActiveSequenceEventData::Free,
            |seqs, rid, decay_now| seqs.free(rid, decay_now),
            true,
        ) {
            Ok(()) => Ok(()),
            Err(SequenceError::RequestNotFound { .. }) => {
                tracing::debug!("Request {request_id} not found, already freed (idempotent)");
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    /// Mark prefill as completed for a request.
    ///
    /// Note: Calling this multiple times for the same request is allowed and will be a no-op
    /// after the first call (idempotent).
    pub fn mark_prefill_completed(
        &self,
        request_id: &RequestId,
        decay_now: Instant,
    ) -> Result<(), SequenceError> {
        self.mutate_request_worker_load_state(
            request_id,
            decay_now,
            ActiveSequenceEventData::MarkPrefillCompleted,
            |seqs, rid, decay_now| {
                seqs.mark_prefill_completed(rid, decay_now);
            },
        )
    }

    /// Add an output block with optional fractional decay weight.
    ///
    /// This is used during generation to track output blocks as they are created.
    /// The decay_fraction represents how "temporary" the block is based on generation progress.
    // NOTE: Output blocks remain local and are intentionally not replicated because their
    // frequency would consume disproportionate replica-sync network bandwidth.
    pub fn add_output_block(
        &self,
        request_id: &RequestId,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        let worker = self.request_index.worker_for(request_id).ok_or_else(|| {
            SequenceError::RequestNotFound {
                request_id: request_id.clone(),
            }
        })?;

        let load = {
            let table = self.workers.read();
            let Some(&idx) = table.index.get(&worker) else {
                drop(table);
                return Err(self.stale_request_not_found(request_id, worker, "add_output_block"));
            };
            let mut seq = table.slots[idx].sequences.write();
            let Some(_new_block_hash) = seq.add_output_block(request_id, decay_fraction) else {
                return Err(SequenceError::RequestNotFound {
                    request_id: request_id.clone(),
                });
            };
            let load = seq.worker_load_snapshot();
            self.prompt_registry.replace_worker_load_state(worker, load);
            load
        };

        self.publish_worker_load_snapshot(worker, load, Instant::now());

        Ok(())
    }

    /// Get the number of workers.
    #[cfg(test)]
    pub(crate) fn num_workers(&self) -> usize {
        self.workers.read().slots.len()
    }

    /// Get the worker type for this router ("prefill" or "decode").
    pub fn worker_type(&self) -> &'static str {
        self.worker_type
    }

    /// Query all workers for the potential blocks and tokens.
    pub fn potential_blocks_and_tokens<const INCLUDE_ACTIVE_REQUESTS: bool>(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        prefill_token_deltas: &PrefillTokenDeltas,
    ) -> PotentialLoadMaps {
        self.potential_blocks_and_tokens_at::<INCLUDE_ACTIVE_REQUESTS>(
            token_sequence,
            prefill_token_deltas,
            Instant::now(),
        )
    }

    pub fn potential_blocks_and_tokens_at<const INCLUDE_ACTIVE_REQUESTS: bool>(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        prefill_token_deltas: &PrefillTokenDeltas,
        decay_now: Instant,
    ) -> PotentialLoadMaps {
        #[cfg(feature = "bench")]
        let start = tokio::time::Instant::now();

        #[cfg(feature = "bench")]
        let num_workers = self.workers.read().slots.len();

        let result = self
            .prompt_registry
            .potential_blocks_and_tokens::<INCLUDE_ACTIVE_REQUESTS>(
                token_sequence,
                prefill_token_deltas,
                decay_now,
            );

        #[cfg(feature = "bench")]
        {
            let total_elapsed = start.elapsed();
            tracing::info!(
                num_workers,
                total_us = total_elapsed.as_micros() as u64,
                "potential_blocks_and_tokens completed"
            );
        }

        result
    }

    pub fn project_worker_loads(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        decay_now: Instant,
    ) -> FxHashMap<WorkerWithDpRank, WorkerLoadProjection> {
        #[cfg(feature = "bench")]
        let start = tokio::time::Instant::now();

        #[cfg(feature = "bench")]
        let num_workers = self.workers.read().slots.len();

        let result = self
            .prompt_registry
            .project_worker_loads(token_sequence, decay_now);

        #[cfg(feature = "bench")]
        {
            let total_elapsed = start.elapsed();
            tracing::info!(
                num_workers,
                total_us = total_elapsed.as_micros() as u64,
                "project_worker_loads completed"
            );
        }

        result
    }

    /// Query all workers for their current number of active blocks.
    pub fn active_blocks(&self) -> HashMap<WorkerWithDpRank, usize> {
        self.prompt_registry.active_blocks()
    }

    /// Query all workers for their current number of active tokens.
    pub fn active_tokens(&self, decay_now: Instant) -> HashMap<WorkerWithDpRank, usize> {
        self.prompt_registry.active_tokens(decay_now)
    }

    /// Return modeled remaining prefill time by worker from the derived read model.
    ///
    /// Values are non-negative milliseconds. For workers whose active prefills
    /// all have modeled durations, the formula is:
    /// `total_modeled_prefill_ms.saturating_sub(elapsed_since_oldest_anchor_ms)`.
    ///
    /// Elapsed time from the oldest active prefill can reduce later modeled
    /// backlog, then the final worker backlog is clipped at zero. This accounts
    /// for engines that batch or overlap multiple prefills.
    ///
    /// `Err(MissingExpectedDuration)` is expected for workers with any active
    /// prefill that lacks an AIC prediction, including the default no-AIC path or
    /// failed predictions. Replica-synced remote values are receive-time anchored
    /// advisory reads, not producer-time truth.
    #[allow(dead_code)]
    pub(crate) fn modeled_remaining_prefill_time_loads_at(
        &self,
        now: Instant,
    ) -> Vec<PrefillTimeLoad> {
        self.prompt_registry
            .modeled_remaining_prefill_times_ms(now)
            .into_iter()
            .map(
                |(worker, modeled_remaining_prefill_time_ms)| PrefillTimeLoad {
                    worker,
                    modeled_remaining_prefill_time_ms,
                },
            )
            .collect()
    }

    /// Return true if any worker satisfies the provided predicate on active token count.
    pub fn any_worker_matches_active_tokens(
        &self,
        decay_now: Instant,
        predicate: impl FnMut(WorkerWithDpRank, usize) -> bool,
    ) -> bool {
        self.prompt_registry
            .any_worker_matches_active_tokens(decay_now, predicate)
    }

    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        self.request_index.active_lora_counts()
    }

    pub fn active_request_counts(&self) -> HashMap<WorkerWithDpRank, usize> {
        self.prompt_registry.active_request_counts()
    }

    /// Force expire stale requests across all workers (one-shot).
    ///
    /// This is necessary because worker expiration otherwise only runs as a side-effect
    /// of `add_request`. If a worker has many expired active sequences and no new
    /// requests are added, expiration never runs. This method forces it on all workers.
    ///
    /// To run this periodically, use start_periodic_force_expiry_across_all_workers.
    pub fn force_expire_requests_across_all_workers(&self) {
        let now = Instant::now();
        let table = self.workers.read();
        let mut removed_request_count = 0;
        for slot in &table.slots {
            let mut seq = slot.sequences.write();
            let outcome = seq.force_expiry();
            if !outcome.expired_request_ids.is_empty() {
                let load = seq.worker_load_snapshot();
                self.prompt_registry.apply_membership_delta_and_load(
                    slot.worker,
                    &slot.trie_lookup,
                    outcome.membership_delta,
                    load,
                );
                removed_request_count += outcome.expired_request_ids.len();
                self.request_index
                    .remove_requests(outcome.expired_request_ids.iter());
                self.publish_worker_load_snapshot(slot.worker, load, now);
            }
        }
        drop(table);
        let duration = now.elapsed();
        tracing::debug!(
            duration = duration.as_secs_f64(),
            removed_request_count,
            "Force expired stale requests across all workers"
        );
    }

    /// Spawn a background task that calls `force_expire_requests_across_all_workers`
    /// at the given interval until `cancel_token` is cancelled.
    ///
    /// **Concurrency note:** This type is always used as `Arc<ActiveSequencesMultiWorker>`. All
    /// mutation is via interior mutability (`RwLock<WorkerTable>`, `DashMap`), so the periodic
    /// task only needs `&self` and does not block other callers.
    pub fn start_periodic_force_expiry_across_all_workers(
        self: &Arc<Self>,
        cancel_token: CancellationToken,
    ) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            let mut expiry_interval =
                tokio::time::interval(FORCE_EXPIRE_REQUESTS_ACROSS_ALL_WORKERS_INTERVAL);
            expiry_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = expiry_interval.tick() => {
                        this.force_expire_requests_across_all_workers();
                    }
                    _ = cancel_token.cancelled() => {
                        break;
                    }
                }
            }
        });
    }

    pub(super) fn ensure_worker_registered(&self, worker: WorkerWithDpRank) {
        {
            let table = self.workers.read();
            if table.index.contains_key(&worker) {
                return;
            }
        }

        self.ensure_worker_registered_after_miss(worker);
    }

    fn ensure_worker_registered_after_miss(&self, worker: WorkerWithDpRank) {
        // Called only after any read guard has been dropped.
        let mut table = self.workers.write();
        if table.index.contains_key(&worker) {
            return;
        }

        tracing::debug!(?worker, "Lazily registering worker in slot tracker");
        let change = table.ensure_worker(self.block_size, worker);
        drop(table);

        self.apply_worker_topology_change(change);
    }

    fn add_request_local(
        &self,
        req: SequenceRequest,
        decay_now: Instant,
        lazily_register_worker: bool,
    ) -> Result<(), SequenceError> {
        let SequenceRequest {
            request_id,
            token_sequence,
            track_prefill_tokens,
            expected_output_tokens,
            prefill_load_hint,
            worker,
            lora_name,
        } = req;

        let mut attempted_lazy_registration = false;

        let (expired_request_ids, load) = loop {
            let table = self.workers.read();
            let Some(&idx) = table.index.get(&worker) else {
                drop(table);
                if !lazily_register_worker || attempted_lazy_registration {
                    return Err(SequenceError::WorkerNotFound { worker });
                }
                attempted_lazy_registration = true;
                self.ensure_worker_registered_after_miss(worker);
                continue;
            };
            if let Err(existing_worker) =
                self.request_index
                    .try_insert_request(request_id.clone(), worker, lora_name)
            {
                return Err(SequenceError::DuplicateRequest {
                    request_id,
                    worker: existing_worker,
                });
            }
            let slot = &table.slots[idx];
            let mut seq = slot.sequences.write();
            let outcome = seq.add_request_with_prefill_tracking(
                request_id,
                token_sequence,
                expected_output_tokens,
                track_prefill_tokens,
                prefill_load_hint,
                decay_now,
            );
            let load = seq.worker_load_snapshot();
            self.prompt_registry.apply_membership_delta_and_load(
                worker,
                &slot.trie_lookup,
                outcome.membership_delta,
                load,
            );
            break (outcome.expired_request_ids, load);
        };

        self.request_index
            .remove_requests(expired_request_ids.iter());

        self.publish_worker_load_snapshot(worker, load, decay_now);

        Ok(())
    }

    fn stale_request_not_found(
        &self,
        request_id: &RequestId,
        worker: WorkerWithDpRank,
        operation: &'static str,
    ) -> SequenceError {
        if self.request_index.worker_for(request_id) == Some(worker) {
            self.request_index.remove_request(request_id);
            tracing::warn!(
                %request_id,
                ?worker,
                operation,
                "request index referenced a missing worker slot; removed stale mapping"
            );
        } else {
            tracing::warn!(
                %request_id,
                ?worker,
                operation,
                "request worker slot disappeared before the mutation ran"
            );
        }

        SequenceError::RequestNotFound {
            request_id: request_id.clone(),
        }
    }

    fn mutate_request_worker_prompt_state_local(
        &self,
        worker: WorkerWithDpRank,
        request_id: &RequestId,
        decay_now: Instant,
        mutate_fn: impl FnOnce(&mut ActiveSequences, &RequestId, Instant) -> PromptMembershipDelta,
        remove_mapping: bool,
    ) -> Result<(), SequenceError> {
        let load = {
            let table = self.workers.read();
            let Some(&idx) = table.index.get(&worker) else {
                drop(table);
                return Err(self.stale_request_not_found(request_id, worker, "free_or_mutate"));
            };
            let slot = &table.slots[idx];
            let mut seq = slot.sequences.write();
            let delta = mutate_fn(&mut seq, request_id, decay_now);
            let load = seq.worker_load_snapshot();
            self.prompt_registry.apply_membership_delta_and_load(
                worker,
                &slot.trie_lookup,
                delta,
                load,
            );
            load
        };

        if remove_mapping {
            self.request_index.remove_request(request_id);
        }

        self.publish_worker_load_snapshot(worker, load, decay_now);

        Ok(())
    }

    fn mutate_request_worker_load_state_local(
        &self,
        worker: WorkerWithDpRank,
        request_id: &RequestId,
        decay_now: Instant,
        mutate_fn: impl FnOnce(&mut ActiveSequences, &RequestId, Instant),
    ) -> Result<(), SequenceError> {
        let load = {
            let table = self.workers.read();
            let Some(&idx) = table.index.get(&worker) else {
                drop(table);
                return Err(self.stale_request_not_found(request_id, worker, "load_only_mutate"));
            };
            let mut seq = table.slots[idx].sequences.write();
            mutate_fn(&mut seq, request_id, decay_now);
            let load = seq.worker_load_snapshot();
            self.prompt_registry.replace_worker_load_state(worker, load);
            load
        };

        self.publish_worker_load_snapshot(worker, load, decay_now);

        Ok(())
    }

    fn mutate_request_worker_prompt_state(
        &self,
        request_id: &RequestId,
        decay_now: Instant,
        event_data: ActiveSequenceEventData,
        mutate_fn: impl FnOnce(&mut ActiveSequences, &RequestId, Instant) -> PromptMembershipDelta,
        remove_mapping: bool,
    ) -> Result<(), SequenceError> {
        let worker = self.request_index.worker_for(request_id).ok_or_else(|| {
            SequenceError::RequestNotFound {
                request_id: request_id.clone(),
            }
        })?;

        let lora_name = self.request_index.lora_for(request_id);
        self.mutate_request_worker_prompt_state_local(
            worker,
            request_id,
            decay_now,
            mutate_fn,
            remove_mapping,
        )?;
        self.spawn_publish_event(ActiveSequenceEvent {
            request_id: request_id.clone(),
            worker,
            data: event_data,
            router_id: self.router_id,
            lora_name,
        });
        Ok(())
    }

    fn mutate_request_worker_load_state(
        &self,
        request_id: &RequestId,
        decay_now: Instant,
        event_data: ActiveSequenceEventData,
        mutate_fn: impl FnOnce(&mut ActiveSequences, &RequestId, Instant),
    ) -> Result<(), SequenceError> {
        let worker = self.request_index.worker_for(request_id).ok_or_else(|| {
            SequenceError::RequestNotFound {
                request_id: request_id.clone(),
            }
        })?;

        let lora_name = self.request_index.lora_for(request_id);
        self.mutate_request_worker_load_state_local(worker, request_id, decay_now, mutate_fn)?;
        self.spawn_publish_event(ActiveSequenceEvent {
            request_id: request_id.clone(),
            worker,
            data: event_data,
            router_id: self.router_id,
            lora_name,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, VecDeque};
    use std::future::{self, Future};
    use std::sync::Mutex;
    use std::time::Duration;

    use rustc_hash::FxHashMap;
    use tokio::sync::mpsc;

    use super::*;
    use crate::protocols::{
        ActiveSequenceEvent, ActiveSequenceEventData, BlockHashOptions, PrefillLoadHint,
        compute_block_hash_for_seq, compute_seq_hash_for_block,
    };
    use crate::sequences::prefill_tracker::PrefillTimeLoadError;
    use crate::test_utils::NoopSequencePublisher;

    fn make_sequences() -> ActiveSequencesMultiWorker<NoopSequencePublisher> {
        ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::from([(1_u64, (0_u32, 1_u32))]),
            false,
            0,
            "test",
        )
    }

    fn make_multi_sequences() -> ActiveSequencesMultiWorker<NoopSequencePublisher> {
        ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::from([(1_u64, (0_u32, 1_u32)), (2_u64, (0_u32, 1_u32))]),
            false,
            0,
            "test",
        )
    }

    fn make_multi_sequences_with_block_size(
        block_size: usize,
    ) -> ActiveSequencesMultiWorker<NoopSequencePublisher> {
        ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size,
            HashMap::from([(1_u64, (0_u32, 1_u32)), (2_u64, (0_u32, 1_u32))]),
            false,
            0,
            "test",
        )
    }

    fn naive_potential_loads(
        sequences: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
        token_sequence: Option<&[SequenceHash]>,
        prefill_token_deltas: &PrefillTokenDeltas,
        decay_now: Instant,
    ) -> (
        FxHashMap<WorkerWithDpRank, usize>,
        FxHashMap<WorkerWithDpRank, usize>,
    ) {
        let table = sequences.workers.read();
        let mut potential_blocks = FxHashMap::default();
        let mut potential_tokens = FxHashMap::default();
        for slot in &table.slots {
            let seq = slot.sequences.read();
            let overlap_depth = token_sequence.map_or(0, |query| {
                let active_hashes = seq.active_prompt_hashes();
                query
                    .iter()
                    .position(|hash| !active_hashes.contains(hash))
                    .unwrap_or(query.len())
            });
            let new_blocks =
                token_sequence.map_or(0, |query| query.len().saturating_sub(overlap_depth));
            let added_tokens = prefill_token_deltas.tokens_for(slot.worker);
            potential_blocks.insert(slot.worker, seq.active_blocks() + new_blocks);
            potential_tokens.insert(slot.worker, seq.active_tokens(decay_now) + added_tokens);
        }
        (potential_blocks, potential_tokens)
    }

    fn seq_hashes_for_tokens(tokens: &[u32], lora_name: Option<&str>) -> Vec<SequenceHash> {
        seq_hashes_for_tokens_with_block_size(tokens, 4, lora_name)
    }

    fn seq_hashes_for_tokens_with_block_size(
        tokens: &[u32],
        block_size: u32,
        lora_name: Option<&str>,
    ) -> Vec<SequenceHash> {
        let block_hashes = compute_block_hash_for_seq(
            tokens,
            block_size,
            BlockHashOptions {
                lora_name,
                ..Default::default()
            },
        );
        compute_seq_hash_for_block(&block_hashes)
    }

    fn tracking_hint(tokens: usize) -> Option<PrefillLoadHint> {
        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: None,
        })
    }

    fn modeled_hint(tokens: usize, duration_secs: u64) -> Option<PrefillLoadHint> {
        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: tokens,
            expected_prefill_duration: Some(Duration::from_secs(duration_secs)),
        })
    }

    fn modeled_time_loads_by_worker(
        sequences: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
        now: Instant,
    ) -> HashMap<WorkerWithDpRank, Result<u64, PrefillTimeLoadError>> {
        sequences
            .modeled_remaining_prefill_time_loads_at(now)
            .into_iter()
            .map(|load| (load.worker, load.modeled_remaining_prefill_time_ms))
            .collect()
    }

    fn active_request_count(
        sequences: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
        worker: WorkerWithDpRank,
    ) -> usize {
        sequences
            .active_request_counts()
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    struct VecSubscriber {
        events: VecDeque<anyhow::Result<ActiveSequenceEvent>>,
    }

    impl SequenceSubscriber for VecSubscriber {
        fn next_event(
            &mut self,
        ) -> impl Future<Output = Option<anyhow::Result<ActiveSequenceEvent>>> + Send {
            future::ready(self.events.pop_front())
        }

        fn poll_next_event(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<anyhow::Result<ActiveSequenceEvent>>> {
            Poll::Ready(self.events.pop_front())
        }
    }

    struct ChannelSubscriber {
        rx: mpsc::UnboundedReceiver<anyhow::Result<ActiveSequenceEvent>>,
    }

    impl SequenceSubscriber for ChannelSubscriber {
        async fn next_event(&mut self) -> Option<anyhow::Result<ActiveSequenceEvent>> {
            self.rx.recv().await
        }

        fn poll_next_event(
            &mut self,
            cx: &mut Context<'_>,
        ) -> Poll<Option<anyhow::Result<ActiveSequenceEvent>>> {
            self.rx.poll_recv(cx)
        }
    }

    struct CancelOnPollSubscriber {
        event: Option<anyhow::Result<ActiveSequenceEvent>>,
        cancel_token: CancellationToken,
    }

    impl SequenceSubscriber for CancelOnPollSubscriber {
        fn next_event(
            &mut self,
        ) -> impl Future<Output = Option<anyhow::Result<ActiveSequenceEvent>>> + Send {
            future::ready(self.event.take())
        }

        fn poll_next_event(
            &mut self,
            _cx: &mut Context<'_>,
        ) -> Poll<Option<anyhow::Result<ActiveSequenceEvent>>> {
            self.cancel_token.cancel();
            Poll::Pending
        }
    }

    #[derive(Default)]
    struct RecordingPublisherState {
        single_loads: Mutex<Vec<ActiveLoad>>,
        load_batches: Mutex<Vec<Vec<ActiveLoad>>>,
        observations: Mutex<Vec<(WorkerWithDpRank, usize, usize)>>,
        registered: Mutex<Vec<WorkerWithDpRank>>,
        removed: Mutex<Vec<WorkerWithDpRank>>,
    }

    impl RecordingPublisherState {
        fn load_batches(&self) -> Vec<Vec<ActiveLoad>> {
            self.load_batches.lock().unwrap().clone()
        }

        fn clear(&self) {
            self.single_loads.lock().unwrap().clear();
            self.load_batches.lock().unwrap().clear();
            self.observations.lock().unwrap().clear();
            self.registered.lock().unwrap().clear();
            self.removed.lock().unwrap().clear();
        }
    }

    struct RecordingPublisher {
        state: Arc<RecordingPublisherState>,
    }

    impl SequencePublisher for RecordingPublisher {
        fn publish_event(
            &self,
            _event: &ActiveSequenceEvent,
        ) -> impl Future<Output = anyhow::Result<()>> + Send {
            future::ready(Ok(()))
        }

        fn publish_load(&self, load: ActiveLoad) {
            self.state.single_loads.lock().unwrap().push(load);
        }

        fn publish_load_batch(&self, loads: Vec<ActiveLoad>) {
            self.state.load_batches.lock().unwrap().push(loads);
        }

        fn observe_load(
            &self,
            worker: &WorkerWithDpRank,
            _worker_type: &str,
            blocks: usize,
            tokens: usize,
        ) {
            self.state
                .observations
                .lock()
                .unwrap()
                .push((*worker, blocks, tokens));
        }

        fn observe_worker_registered(&self, worker: &WorkerWithDpRank, _worker_type: &str) {
            self.state.registered.lock().unwrap().push(*worker);
        }

        fn observe_worker_removed(&self, worker: &WorkerWithDpRank, _worker_type: &str) {
            self.state.removed.lock().unwrap().push(*worker);
        }
    }

    fn make_recording_sequences(
        workers: HashMap<u64, (u32, u32)>,
    ) -> (
        ActiveSequencesMultiWorker<RecordingPublisher>,
        Arc<RecordingPublisherState>,
    ) {
        let state = Arc::new(RecordingPublisherState::default());
        let sequences = ActiveSequencesMultiWorker::new(
            RecordingPublisher {
                state: Arc::clone(&state),
            },
            4,
            workers,
            true,
            0,
            "test",
        );
        (sequences, state)
    }

    #[test]
    fn worker_topology_observes_registered_and_removed_workers() {
        let (sequences, state) = make_recording_sequences(HashMap::from([(1, (0, 2))]));
        assert_eq!(
            *state.registered.lock().unwrap(),
            vec![WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(1, 1)]
        );

        state.clear();
        sequences
            .register_worker(WorkerDpRange::new(2, 0, 1))
            .unwrap();
        assert_eq!(
            *state.registered.lock().unwrap(),
            vec![WorkerWithDpRank::new(2, 0)]
        );

        state.clear();
        sequences.unregister_worker(1).unwrap();
        assert_eq!(
            *state.removed.lock().unwrap(),
            vec![WorkerWithDpRank::new(1, 0), WorkerWithDpRank::new(1, 1)]
        );
    }

    fn replica_add(
        request_id: impl Into<String>,
        worker: WorkerWithDpRank,
        token_sequence: Vec<SequenceHash>,
    ) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker,
            data: ActiveSequenceEventData::AddRequest {
                token_sequence: Some(token_sequence),
                track_prefill_tokens: true,
                expected_output_tokens: None,
                prefill_load_hint: tracking_hint(12),
            },
            router_id: 99,
            lora_name: None,
        }
    }

    fn replica_free(
        request_id: impl Into<String>,
        payload_worker: WorkerWithDpRank,
    ) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: payload_worker,
            data: ActiveSequenceEventData::Free,
            router_id: 99,
            lora_name: None,
        }
    }

    fn replica_mark(
        request_id: impl Into<String>,
        payload_worker: WorkerWithDpRank,
    ) -> ActiveSequenceEvent {
        ActiveSequenceEvent {
            request_id: request_id.into(),
            worker: payload_worker,
            data: ActiveSequenceEventData::MarkPrefillCompleted,
            router_id: 99,
            lora_name: None,
        }
    }

    #[tokio::test]
    async fn add_request_can_skip_prefill_token_tracking() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);
        let decay_now = Instant::now();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-1".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();

        assert_eq!(
            sequences.active_tokens(decay_now).get(&worker).copied(),
            Some(0)
        );
        assert_eq!(active_request_count(&sequences, worker), 1);
    }

    #[test]
    fn modeled_remaining_prefill_time_loads_follow_multi_worker_projection() {
        let sequences = make_multi_sequences();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let start = Instant::now();
        let a_oldest = "a-oldest".to_string();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: a_oldest.clone(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: modeled_hint(100, 10),
                    worker: worker_a,
                    lora_name: None,
                },
                start,
            )
            .unwrap();
        sequences
            .add_request(
                SequenceRequest {
                    request_id: "a-later".to_string(),
                    token_sequence: Some(vec![4, 5, 6]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: modeled_hint(60, 4),
                    worker: worker_a,
                    lora_name: None,
                },
                start + Duration::from_secs(1),
            )
            .unwrap();
        sequences
            .add_request(
                SequenceRequest {
                    request_id: "b-unmodeled".to_string(),
                    token_sequence: Some(vec![7, 8, 9]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: tracking_hint(12),
                    worker: worker_b,
                    lora_name: None,
                },
                start,
            )
            .unwrap();

        let query = start + Duration::from_secs(3);
        let loads = modeled_time_loads_by_worker(&sequences, query);
        assert_eq!(loads.get(&worker_a).copied(), Some(Ok(11_000)));
        assert_eq!(
            loads.get(&worker_b).copied(),
            Some(Err(PrefillTimeLoadError::MissingExpectedDuration))
        );

        sequences.mark_prefill_completed(&a_oldest, query).unwrap();
        assert_eq!(active_request_count(&sequences, worker_a), 2);

        let loads = modeled_time_loads_by_worker(&sequences, query + Duration::from_secs(2));
        assert_eq!(loads.get(&worker_a).copied(), Some(Ok(2_000)));
        assert_eq!(
            loads.get(&worker_b).copied(),
            Some(Err(PrefillTimeLoadError::MissingExpectedDuration))
        );
    }

    #[test]
    fn free_nonanchor_modeled_prefill_applies_anchor_credit() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);
        let start = Instant::now();
        let later = "later".to_string();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "oldest".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: modeled_hint(100, 10),
                    worker,
                    lora_name: None,
                },
                start,
            )
            .unwrap();
        sequences
            .add_request(
                SequenceRequest {
                    request_id: later.clone(),
                    token_sequence: Some(vec![4, 5, 6]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: modeled_hint(40, 4),
                    worker,
                    lora_name: None,
                },
                start,
            )
            .unwrap();

        let completion_time = start + Duration::from_secs(5);
        assert_eq!(
            sequences
                .active_tokens(completion_time)
                .get(&worker)
                .copied(),
            Some(90)
        );
        assert_eq!(
            modeled_time_loads_by_worker(&sequences, completion_time)
                .get(&worker)
                .copied(),
            Some(Ok(9_000))
        );

        sequences.free(&later, completion_time).unwrap();
        assert_eq!(active_request_count(&sequences, worker), 1);

        assert_eq!(
            sequences
                .active_tokens(completion_time)
                .get(&worker)
                .copied(),
            Some(90)
        );
        assert_eq!(
            modeled_time_loads_by_worker(&sequences, completion_time)
                .get(&worker)
                .copied(),
            Some(Ok(9_000))
        );
    }

    #[test]
    fn block_membership_index_matches_naive_loads_with_output_blocks_and_prefill_updates() {
        let sequences = make_multi_sequences();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let decay_now = Instant::now();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-a".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: tracking_hint(12),
                    worker: worker_a,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();
        sequences
            .add_output_block(&"req-a".to_string(), Some(0.5))
            .unwrap();
        sequences
            .mark_prefill_completed(&"req-a".to_string(), decay_now)
            .unwrap();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-b".to_string(),
                    token_sequence: Some(vec![1, 2, 4]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: tracking_hint(12),
                    worker: worker_b,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();

        let prompt = vec![1, 2, 3, 5];
        let mut deltas = FxHashMap::default();
        deltas.insert(worker_a, 8);
        deltas.insert(worker_b, 12);
        let prefill_token_deltas = PrefillTokenDeltas::new(16, deltas);
        let expected =
            naive_potential_loads(&sequences, Some(&prompt), &prefill_token_deltas, decay_now);

        let actual = sequences.potential_blocks_and_tokens_at::<false>(
            Some(&prompt),
            &prefill_token_deltas,
            decay_now,
        );
        let projections = sequences.project_worker_loads(Some(&prompt), decay_now);

        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);
        assert_eq!(
            projections.get(&worker_a).copied(),
            Some(WorkerLoadProjection {
                active_prefill_tokens: 0,
                active_decode_blocks: 2,
                additional_active_blocks: 1,
            })
        );
        assert_eq!(
            projections.get(&worker_b).copied(),
            Some(WorkerLoadProjection {
                active_prefill_tokens: 12,
                active_decode_blocks: 3,
                additional_active_blocks: 2,
            })
        );
    }

    #[test]
    fn potential_blocks_use_prefix_membership_not_set_membership() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);
        let decay_now = Instant::now();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-a".to_string(),
                    token_sequence: Some(vec![1, 2, 4, 5]),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();

        let (potential_blocks, _, _) = sequences.potential_blocks_and_tokens_at::<false>(
            Some(&[1, 2, 3, 5]),
            &PrefillTokenDeltas::none(),
            decay_now,
        );

        assert_eq!(potential_blocks.get(&worker).copied(), Some(6));
    }

    #[test]
    fn lora_specific_sequence_hashes_do_not_cross_match() {
        let sequences = make_multi_sequences();
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let decay_now = Instant::now();
        let tokens = [1_u32, 2, 3, 4, 5, 6, 7, 8];
        let base_prompt = seq_hashes_for_tokens(&tokens, None);
        let lora_prompt = seq_hashes_for_tokens(&tokens, Some("adapter-a"));

        assert_ne!(base_prompt, lora_prompt);

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "base".to_string(),
                    token_sequence: Some(base_prompt.clone()),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker: worker_a,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();
        sequences
            .add_request(
                SequenceRequest {
                    request_id: "lora".to_string(),
                    token_sequence: Some(lora_prompt),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker: worker_b,
                    lora_name: Some("adapter-a".to_string()),
                },
                decay_now,
            )
            .unwrap();

        let expected = naive_potential_loads(
            &sequences,
            Some(&base_prompt),
            &PrefillTokenDeltas::none(),
            decay_now,
        );
        let actual = sequences.potential_blocks_and_tokens_at::<false>(
            Some(&base_prompt),
            &PrefillTokenDeltas::none(),
            decay_now,
        );

        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);

        let active_blocks = sequences.active_blocks();
        assert_eq!(
            actual.0.get(&worker_b).copied(),
            Some(active_blocks[&worker_b] + base_prompt.len()),
        );
    }

    #[test]
    fn unit_block_size_repeated_tokens_preserve_membership_and_trim() {
        let sequences = make_multi_sequences_with_block_size(1);
        let worker_a = WorkerWithDpRank::new(1, 0);
        let worker_b = WorkerWithDpRank::new(2, 0);
        let decay_now = Instant::now();
        let prompt_a = seq_hashes_for_tokens_with_block_size(&[7_u32, 7, 7], 1, None);
        let prompt_b = seq_hashes_for_tokens_with_block_size(&[7_u32, 7, 8], 1, None);

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-a".to_string(),
                    token_sequence: Some(prompt_a.clone()),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker: worker_a,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();
        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-b".to_string(),
                    token_sequence: Some(prompt_b.clone()),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker: worker_b,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();

        let expected = naive_potential_loads(
            &sequences,
            Some(&prompt_b),
            &PrefillTokenDeltas::none(),
            decay_now,
        );
        let actual = sequences.potential_blocks_and_tokens_at::<false>(
            Some(&prompt_b),
            &PrefillTokenDeltas::none(),
            decay_now,
        );
        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);
        assert_eq!(actual.0.get(&worker_a).copied(), Some(4));
        assert_eq!(actual.0.get(&worker_b).copied(), Some(3));

        sequences.free(&"req-b".to_string(), decay_now).unwrap();

        let expected_after_free = naive_potential_loads(
            &sequences,
            Some(&prompt_b),
            &PrefillTokenDeltas::none(),
            decay_now,
        );
        let actual_after_free = sequences.potential_blocks_and_tokens_at::<false>(
            Some(&prompt_b),
            &PrefillTokenDeltas::none(),
            decay_now,
        );
        assert_eq!(actual_after_free.0, expected_after_free.0);
        assert_eq!(actual_after_free.1, expected_after_free.1);
        assert_eq!(actual_after_free.0.get(&worker_a).copied(), Some(4));
        assert_eq!(actual_after_free.0.get(&worker_b).copied(), Some(3));

        sequences.free(&"req-a".to_string(), decay_now).unwrap();
        sequences.assert_completely_drained(decay_now);
    }

    #[tokio::test(start_paused = true)]
    async fn force_expiry_clears_block_membership_index() {
        let sequences = make_multi_sequences();
        let worker = WorkerWithDpRank::new(1, 0);

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-1".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: tracking_hint(12),
                    worker,
                    lora_name: None,
                },
                Instant::now(),
            )
            .unwrap();

        tokio::time::advance(Duration::from_secs(331)).await;
        sequences.force_expire_requests_across_all_workers();

        assert!(sequences.request_index.is_empty());
        assert!(sequences.prompt_registry.is_block_index_empty());
        assert_eq!(sequences.active_blocks().get(&worker).copied(), Some(0));
        assert_eq!(active_request_count(&sequences, worker), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn expiry_then_immediate_readd_preserves_block_membership() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-1".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: tracking_hint(12),
                    worker,
                    lora_name: None,
                },
                Instant::now(),
            )
            .unwrap();

        tokio::time::advance(Duration::from_secs(331)).await;

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-2".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: tracking_hint(12),
                    worker,
                    lora_name: None,
                },
                Instant::now(),
            )
            .unwrap();

        assert!(!sequences.prompt_registry.is_block_index_empty());
        assert_eq!(sequences.active_blocks().get(&worker).copied(), Some(3));

        let expected = naive_potential_loads(
            &sequences,
            Some(&[1, 2, 3]),
            &PrefillTokenDeltas::none(),
            Instant::now(),
        );
        let actual = sequences.potential_blocks_and_tokens_at::<false>(
            Some(&[1, 2, 3]),
            &PrefillTokenDeltas::none(),
            Instant::now(),
        );
        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);
    }

    #[tokio::test(start_paused = true)]
    async fn replica_sync_coalesces_latest_load_wake_and_cleanup() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (sequences, publisher) =
            make_recording_sequences(HashMap::from([(1_u64, (0_u32, 1_u32))]));
        let subscriber = VecSubscriber {
            events: VecDeque::from(vec![
                Ok(replica_add("req-1", worker, vec![1, 2, 3])),
                Ok(replica_add("req-2", worker, vec![4, 5, 6])),
                Ok(replica_mark("req-2", worker)),
                Ok(replica_free("req-1", worker)),
            ]),
        };

        sequences
            .run_replica_sync(subscriber, CancellationToken::new())
            .await
            .unwrap();

        let batches = publisher.load_batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[0][0].worker_id, worker.worker_id);
        assert_eq!(batches[0][0].dp_rank, worker.dp_rank);
        assert_eq!(batches[0][0].active_decode_blocks, Some(3));
        assert_eq!(batches[0][0].active_prefill_tokens, Some(0));
        assert_eq!(sequences.remote_state_update_count(), 1);
        assert_eq!(sequences.prompt_registry.cleanup_attempts(), 1);
        assert_eq!(
            sequences.request_index.worker_for(&"req-2".to_string()),
            Some(worker)
        );
        assert_eq!(
            sequences.request_index.worker_for(&"req-1".to_string()),
            None
        );
        assert_eq!(
            sequences.active_request_counts().get(&worker).copied(),
            Some(1)
        );
    }

    #[tokio::test]
    async fn replica_sync_mark_only_batch_does_not_publish_load() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (sequences, publisher) =
            make_recording_sequences(HashMap::from([(1_u64, (0_u32, 1_u32))]));

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![Ok(replica_add("req-1", worker, vec![1, 2, 3]))]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        publisher.clear();
        let wake_count_before = sequences.remote_state_update_count();
        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![
                        Ok(replica_mark("req-1", worker)),
                        Ok(replica_mark("req-1", worker)),
                    ]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert!(publisher.load_batches().is_empty());
        assert_eq!(sequences.remote_state_update_count(), wake_count_before + 1);
    }

    #[tokio::test(start_paused = true)]
    async fn replica_sync_free_uses_canonical_worker_for_collapsed_load() {
        let worker = WorkerWithDpRank::new(1, 0);
        let wrong_payload_worker = WorkerWithDpRank::new(2, 0);
        let (sequences, publisher) = make_recording_sequences(HashMap::from([
            (1_u64, (0_u32, 1_u32)),
            (2_u64, (0_u32, 1_u32)),
        ]));
        let subscriber = VecSubscriber {
            events: VecDeque::from(vec![
                Ok(replica_add("req-1", worker, vec![1, 2, 3])),
                Ok(replica_free("req-1", wrong_payload_worker)),
            ]),
        };

        sequences
            .run_replica_sync(subscriber, CancellationToken::new())
            .await
            .unwrap();

        let batches = publisher.load_batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[0][0].worker_id, worker.worker_id);
        assert_eq!(batches[0][0].dp_rank, worker.dp_rank);
        assert_eq!(batches[0][0].active_decode_blocks, Some(0));
        assert_eq!(sequences.active_blocks().get(&worker).copied(), Some(0));
        assert_eq!(
            sequences
                .active_blocks()
                .get(&wrong_payload_worker)
                .copied(),
            Some(0)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn replica_sync_batch_cap_splits_deferred_side_effects() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (sequences, publisher) =
            make_recording_sequences(HashMap::from([(1_u64, (0_u32, 1_u32))]));
        let events = (0..300)
            .map(|i| Ok(replica_add(format!("req-{i}"), worker, vec![i])))
            .collect();

        sequences
            .run_replica_sync(VecSubscriber { events }, CancellationToken::new())
            .await
            .unwrap();

        let batches = publisher.load_batches();
        assert_eq!(batches.len(), 2);
        assert!(batches.iter().all(|batch| batch.len() == 1));
        assert_eq!(sequences.prompt_registry.cleanup_attempts(), 2);
        assert_eq!(sequences.remote_state_update_count(), 0);
    }

    #[tokio::test]
    async fn replica_sync_flushes_sparse_event_without_waiting_for_another() {
        let worker = WorkerWithDpRank::new(1, 0);
        let state = Arc::new(RecordingPublisherState::default());
        let sequences = Arc::new(ActiveSequencesMultiWorker::new(
            RecordingPublisher {
                state: Arc::clone(&state),
            },
            4,
            HashMap::from([(1_u64, (0_u32, 1_u32))]),
            true,
            0,
            "test",
        ));
        let (tx, rx) = mpsc::unbounded_channel();
        let cancel_token = CancellationToken::new();
        sequences.start_replica_sync(ChannelSubscriber { rx }, cancel_token.clone());

        tx.send(Ok(replica_add("req-1", worker, vec![1, 2, 3])))
            .unwrap();

        tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                if state.load_batches.lock().unwrap().len() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        cancel_token.cancel();
    }

    #[tokio::test]
    async fn replica_sync_receive_error_flushes_applied_prefix() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (sequences, publisher) =
            make_recording_sequences(HashMap::from([(1_u64, (0_u32, 1_u32))]));
        let subscriber = VecSubscriber {
            events: VecDeque::from(vec![
                Ok(replica_add("req-1", worker, vec![1, 2, 3])),
                Err(anyhow::anyhow!("synthetic receive error")),
            ]),
        };

        sequences
            .run_replica_sync(subscriber, CancellationToken::new())
            .await
            .unwrap();

        let batches = publisher.load_batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(
            sequences.request_index.worker_for(&"req-1".to_string()),
            Some(worker)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn replica_sync_cancellation_flushes_applied_prefix() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (sequences, publisher) =
            make_recording_sequences(HashMap::from([(1_u64, (0_u32, 1_u32))]));
        let cancel_token = CancellationToken::new();
        let subscriber = CancelOnPollSubscriber {
            event: Some(Ok(replica_add("req-1", worker, vec![1, 2, 3]))),
            cancel_token: cancel_token.clone(),
        };

        sequences
            .run_replica_sync(subscriber, cancel_token)
            .await
            .unwrap();

        let batches = publisher.load_batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(
            sequences.request_index.worker_for(&"req-1".to_string()),
            Some(worker)
        );
    }

    #[tokio::test]
    async fn replica_sync_add_and_free_keep_block_membership_consistent() {
        let sequences = ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::from([(1_u64, (0_u32, 1_u32))]),
            true,
            0,
            "test",
        );
        let worker = WorkerWithDpRank::new(1, 0);
        let subscriber = VecSubscriber {
            events: VecDeque::from(vec![
                Ok(ActiveSequenceEvent {
                    request_id: "req-1".to_string(),
                    worker,
                    data: ActiveSequenceEventData::AddRequest {
                        token_sequence: Some(vec![1, 2, 3]),
                        track_prefill_tokens: true,
                        expected_output_tokens: None,
                        prefill_load_hint: tracking_hint(12),
                    },
                    router_id: 99,
                    lora_name: None,
                }),
                Ok(ActiveSequenceEvent {
                    request_id: "req-1".to_string(),
                    worker,
                    data: ActiveSequenceEventData::Free,
                    router_id: 99,
                    lora_name: None,
                }),
            ]),
        };

        sequences
            .run_replica_sync(subscriber, CancellationToken::new())
            .await
            .unwrap();

        assert!(sequences.request_index.is_empty());
        assert!(sequences.prompt_registry.is_block_index_empty());
        assert_eq!(sequences.active_blocks().get(&worker).copied(), Some(0));
        assert_eq!(active_request_count(&sequences, worker), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn replica_sync_modeled_remaining_prefill_time_uses_receive_time_and_clears() {
        let sequences = ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::from([(1_u64, (0_u32, 1_u32))]),
            true,
            0,
            "test",
        );
        let worker = WorkerWithDpRank::new(1, 0);

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![Ok(ActiveSequenceEvent {
                        request_id: "req-1".to_string(),
                        worker,
                        data: ActiveSequenceEventData::AddRequest {
                            token_sequence: Some(vec![1, 2, 3]),
                            track_prefill_tokens: true,
                            expected_output_tokens: None,
                            prefill_load_hint: modeled_hint(12, 10),
                        },
                        router_id: 99,
                        lora_name: None,
                    })]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        let loads = modeled_time_loads_by_worker(&sequences, Instant::now());
        assert_eq!(loads.get(&worker).copied(), Some(Ok(10_000)));

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![Ok(ActiveSequenceEvent {
                        request_id: "req-1".to_string(),
                        worker,
                        data: ActiveSequenceEventData::MarkPrefillCompleted,
                        router_id: 99,
                        lora_name: None,
                    })]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        let loads = modeled_time_loads_by_worker(&sequences, Instant::now());
        assert_eq!(loads.get(&worker).copied(), Some(Ok(0)));

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![Ok(ActiveSequenceEvent {
                        request_id: "req-2".to_string(),
                        worker,
                        data: ActiveSequenceEventData::AddRequest {
                            token_sequence: Some(vec![4, 5, 6]),
                            track_prefill_tokens: true,
                            expected_output_tokens: None,
                            prefill_load_hint: modeled_hint(12, 6),
                        },
                        router_id: 99,
                        lora_name: None,
                    })]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        let loads = modeled_time_loads_by_worker(&sequences, Instant::now());
        assert_eq!(loads.get(&worker).copied(), Some(Ok(6_000)));

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![Ok(ActiveSequenceEvent {
                        request_id: "req-2".to_string(),
                        worker,
                        data: ActiveSequenceEventData::Free,
                        router_id: 99,
                        lora_name: None,
                    })]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        let loads = modeled_time_loads_by_worker(&sequences, Instant::now());
        assert_eq!(loads.get(&worker).copied(), Some(Ok(0)));
    }

    #[tokio::test(start_paused = true)]
    async fn replica_sync_nonanchor_free_applies_receive_time_anchor_credit() {
        let sequences = ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::from([(1_u64, (0_u32, 1_u32))]),
            true,
            0,
            "test",
        );
        let worker = WorkerWithDpRank::new(1, 0);
        let later = "later".to_string();

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![
                        Ok(ActiveSequenceEvent {
                            request_id: "oldest".to_string(),
                            worker,
                            data: ActiveSequenceEventData::AddRequest {
                                token_sequence: Some(vec![1, 2, 3]),
                                track_prefill_tokens: true,
                                expected_output_tokens: None,
                                prefill_load_hint: modeled_hint(100, 10),
                            },
                            router_id: 99,
                            lora_name: None,
                        }),
                        Ok(ActiveSequenceEvent {
                            request_id: later.clone(),
                            worker,
                            data: ActiveSequenceEventData::AddRequest {
                                token_sequence: Some(vec![4, 5, 6]),
                                track_prefill_tokens: true,
                                expected_output_tokens: None,
                                prefill_load_hint: modeled_hint(40, 4),
                            },
                            router_id: 99,
                            lora_name: None,
                        }),
                    ]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        tokio::time::advance(Duration::from_secs(5)).await;
        let completion_time = Instant::now();
        assert_eq!(
            modeled_time_loads_by_worker(&sequences, completion_time)
                .get(&worker)
                .copied(),
            Some(Ok(9_000))
        );

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![Ok(ActiveSequenceEvent {
                        request_id: later,
                        worker,
                        data: ActiveSequenceEventData::Free,
                        router_id: 99,
                        lora_name: None,
                    })]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(
            modeled_time_loads_by_worker(&sequences, completion_time)
                .get(&worker)
                .copied(),
            Some(Ok(9_000))
        );
    }

    #[tokio::test]
    async fn replica_sync_add_lazily_registers_missing_worker() {
        let sequences = ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            4,
            HashMap::new(),
            true,
            0,
            "test",
        );
        let worker = WorkerWithDpRank::new(1, 0);
        let subscriber = VecSubscriber {
            events: VecDeque::from(vec![Ok(ActiveSequenceEvent {
                request_id: "req-1".to_string(),
                worker,
                data: ActiveSequenceEventData::AddRequest {
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: tracking_hint(12),
                },
                router_id: 99,
                lora_name: None,
            })]),
        };

        sequences
            .run_replica_sync(subscriber, CancellationToken::new())
            .await
            .unwrap();

        assert_eq!(sequences.num_workers(), 1);
        assert_eq!(
            sequences.request_index.worker_for(&"req-1".to_string()),
            Some(worker)
        );
        assert!(!sequences.prompt_registry.is_block_index_empty());
        assert_eq!(sequences.active_blocks().get(&worker).copied(), Some(3));
    }

    #[tokio::test]
    async fn replica_sync_require_registered_drops_missing_worker() {
        let sequences = ActiveSequencesMultiWorker::new_with_replica_worker_policy(
            NoopSequencePublisher,
            4,
            HashMap::new(),
            true,
            0,
            "test",
            ReplicaWorkerPolicy::RequireRegistered,
        );
        let worker = WorkerWithDpRank::new(1, 0);

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![Ok(replica_add("req-1", worker, vec![1, 2, 3]))]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(sequences.num_workers(), 0);
        assert_eq!(
            sequences.request_index.worker_for(&"req-1".to_string()),
            None
        );
        assert!(sequences.prompt_registry.is_block_index_empty());
    }

    #[tokio::test]
    async fn replica_sync_require_registered_drops_event_after_worker_removal() {
        let sequences = ActiveSequencesMultiWorker::new_with_replica_worker_policy(
            NoopSequencePublisher,
            4,
            HashMap::from([(1_u64, (0_u32, 1_u32))]),
            true,
            0,
            "test",
            ReplicaWorkerPolicy::RequireRegistered,
        );
        let worker = WorkerWithDpRank::new(1, 0);
        sequences.reconcile_workers([]).unwrap();

        sequences
            .run_replica_sync(
                VecSubscriber {
                    events: VecDeque::from(vec![Ok(replica_add("req-1", worker, vec![1, 2, 3]))]),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();

        assert_eq!(sequences.num_workers(), 0);
        assert_eq!(
            sequences.request_index.worker_for(&"req-1".to_string()),
            None
        );
    }

    #[test]
    fn worker_removal_then_readd_starts_with_empty_registry_state() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);
        let decay_now = Instant::now();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-1".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();

        assert_eq!(active_request_count(&sequences, worker), 1);
        sequences.reconcile_workers([]).unwrap();
        assert!(sequences.prompt_registry.is_block_index_empty());
        assert!(sequences.active_blocks().is_empty());
        assert!(!sequences.active_request_counts().contains_key(&worker));
        assert!(sequences.request_index.is_empty());

        sequences
            .reconcile_workers([WorkerDpRange::new(1, 0, 1)])
            .unwrap();
        assert_eq!(sequences.active_blocks().get(&worker).copied(), Some(0));
        assert_eq!(active_request_count(&sequences, worker), 0);
        assert!(sequences.prompt_registry.is_block_index_empty());
    }

    #[test]
    fn strict_add_after_worker_removal_cannot_recreate_worker() {
        let sequences = Arc::new(make_sequences());
        let worker = WorkerWithDpRank::new(1, 0);
        let ready = Arc::new(std::sync::Barrier::new(2));
        let proceed = Arc::new(std::sync::Barrier::new(2));
        let add_sequences = Arc::clone(&sequences);
        let add_ready = Arc::clone(&ready);
        let add_proceed = Arc::clone(&proceed);

        let add = std::thread::spawn(move || {
            add_ready.wait();
            add_proceed.wait();
            add_sequences.add_request_if_registered(
                SequenceRequest {
                    request_id: "req-1".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                },
                Instant::now(),
            )
        });

        ready.wait();
        sequences.reconcile_workers([]).unwrap();
        proceed.wait();
        let result = add.join().unwrap();

        assert!(
            matches!(result, Err(SequenceError::WorkerNotFound { worker: missing }) if missing == worker)
        );
        assert_eq!(sequences.num_workers(), 0);
        assert!(sequences.request_index.is_empty());
        assert!(sequences.prompt_registry.is_block_index_empty());
    }

    #[test]
    fn free_is_idempotent_after_request_is_removed() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);
        let request_id = "req-1".to_string();
        let decay_now = Instant::now();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: request_id.clone(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                },
                decay_now,
            )
            .unwrap();

        sequences.free(&request_id, decay_now).unwrap();
        sequences.free(&request_id, decay_now).unwrap();

        sequences.assert_completely_drained(decay_now);
    }

    #[test]
    fn free_cleans_stale_request_mapping_when_worker_slot_is_missing() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);
        let request_id = "stale-request".to_string();

        sequences.request_index.set_request(
            request_id.clone(),
            worker,
            Some("adapter".to_string()),
        );
        {
            let mut table = sequences.workers.write();
            *table = WorkerTable::new(sequences.block_size, &HashMap::new());
        }

        sequences.free(&request_id, Instant::now()).unwrap();

        assert!(sequences.request_index.is_empty());
    }

    #[test]
    fn explicit_decay_time_drives_multi_worker_load_queries_consistently() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);
        let start = Instant::now();

        sequences
            .add_request(
                SequenceRequest {
                    request_id: "req-1".to_string(),
                    token_sequence: Some(vec![1, 2, 3]),
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: Some(PrefillLoadHint {
                        initial_effective_prefill_tokens: 100,
                        expected_prefill_duration: Some(Duration::from_secs(10)),
                    }),
                    worker,
                    lora_name: None,
                },
                start,
            )
            .unwrap();

        let decay_now = start + Duration::from_secs(5);
        let active_tokens = sequences.active_tokens(decay_now);
        assert_eq!(active_tokens.get(&worker).copied(), Some(50));

        let (_, potential_tokens, _) = sequences.potential_blocks_and_tokens_at::<false>(
            None,
            &PrefillTokenDeltas::none(),
            decay_now,
        );
        assert_eq!(potential_tokens.get(&worker).copied(), Some(50));

        assert!(
            sequences.any_worker_matches_active_tokens(decay_now, |candidate, tokens| {
                candidate == worker && tokens == 50
            })
        );
    }
}
