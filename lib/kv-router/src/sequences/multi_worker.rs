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
use tokio::sync::watch;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::prompt_registry::{PromptRegistry, WorkerLoadSnapshot};
use super::request_maps::RequestIndex;
use super::single::{ActiveSequences, PromptMembershipDelta, RequestId};
use super::topology::WorkerTable;
use crate::protocols::{
    ActiveLoad, ActiveSequenceEvent, ActiveSequenceEventData, OverlapScores, PrefillLoadHint,
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

    /// Record per-worker load in Prometheus gauges.
    fn observe_load(
        &self,
        worker: &WorkerWithDpRank,
        worker_type: &str,
        blocks: usize,
        tokens: usize,
    );
}

/// Abstraction over event subscription for replica sync.
pub trait SequenceSubscriber: Send {
    /// Receive the next replica-sync event, or `None` if the stream is closed.
    fn next_event(
        &mut self,
    ) -> impl Future<Output = Option<anyhow::Result<ActiveSequenceEvent>>> + Send;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
    workers: RwLock<WorkerTable>,
    request_index: RequestIndex,
    prompt_registry: PromptRegistry,
    block_size: usize,
    router_id: u64,
    publisher: Arc<P>,
    remote_state_updates: watch::Sender<()>,
    replica_sync: bool,
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
        assert!(block_size > 1, "block_size must be greater than 1");
        let (remote_state_updates, _) = watch::channel(());
        let workers = WorkerTable::new(block_size, &dp_range);
        let prompt_registry = PromptRegistry::new(workers.workers());

        Self {
            workers: RwLock::new(workers),
            request_index: RequestIndex::default(),
            prompt_registry,
            block_size,
            router_id,
            publisher: Arc::new(publisher),
            remote_state_updates,
            replica_sync,
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
    }

    fn publish_worker_load_snapshot(
        &self,
        worker: WorkerWithDpRank,
        load: WorkerLoadSnapshot,
        decay_now: Instant,
    ) {
        let active_blocks = load.active_blocks;
        let active_tokens = load.active_tokens(decay_now);

        self.publisher
            .observe_load(&worker, self.worker_type, active_blocks, active_tokens);

        let active_load = ActiveLoad {
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            active_decode_blocks: Some(active_blocks as u64),
            active_prefill_tokens: Some(active_tokens as u64),
            kv_used_blocks: None,
        };

        self.publisher.publish_load(active_load);
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

    /// Spawn a background task that subscribes to replica-sync events from peer routers
    /// and applies them to the local state.
    pub fn start_replica_sync<S: SequenceSubscriber + 'static>(
        self: &Arc<Self>,
        subscriber: S,
        cancel_token: CancellationToken,
    ) {
        let this = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(e) = this.run_replica_sync(subscriber, cancel_token).await {
                tracing::error!("Error in active sequences events subscription: {}", e);
            }
        });
    }

    async fn run_replica_sync<S: SequenceSubscriber>(
        &self,
        mut subscriber: S,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        loop {
            tokio::select! {
                result = subscriber.next_event() => {
                    let Some(result) = result else {
                        break;
                    };

                    let Ok(event) = result else {
                        tracing::error!(
                            "Error receiving active sequence event: {}",
                            result.unwrap_err()
                        );
                        continue;
                    };

                    if event.router_id == self.router_id {
                        continue;
                    }

                    // TODO: ActiveSequenceEvent does not carry prompt-load decay timestamps yet.
                    // Peer routers still approximate decay anchoring with local receive time.
                    let decay_now = Instant::now();
                    let mut remote_capacity_changed = false;
                    match &event.data {
                        ActiveSequenceEventData::AddRequest {
                            token_sequence,
                            track_prefill_tokens,
                            expected_output_tokens,
                            prefill_load_hint,
                        } => {
                            self.ensure_worker_registered(event.worker);
                            let table = self.workers.read();
                            if let Some(&idx) = table.index.get(&event.worker) {
                                self.request_index.set_request(
                                    event.request_id.clone(),
                                    event.worker,
                                    event.lora_name.clone(),
                                );
                                let (expired_request_ids, load) = {
                                    let slot = &table.slots[idx];
                                    let mut seq = slot.sequences.write();
                                    let outcome = seq.add_request_with_prefill_tracking(
                                        event.request_id.clone(),
                                        token_sequence.clone(),
                                        *expected_output_tokens,
                                        *track_prefill_tokens,
                                        *prefill_load_hint,
                                        decay_now,
                                    );
                                    let load = seq.worker_load_snapshot();
                                    self.prompt_registry.apply_membership_delta_and_load(
                                        event.worker,
                                        &slot.trie_lookup,
                                        outcome.membership_delta,
                                        load,
                                    );
                                    (outcome.expired_request_ids, load)
                                };
                                drop(table);
                                self.request_index.remove_requests(expired_request_ids.iter());
                                self.publish_worker_load_snapshot(event.worker, load, decay_now);
                                continue;
                            } else {
                                tracing::warn!(
                                    "Worker {:?} not found, cannot process AddRequest",
                                    event.worker
                                );
                            }
                        }
                        ActiveSequenceEventData::Free => {
                            if let Some(worker) = self.request_index.remove_request(&event.request_id) {
                                let table = self.workers.read();
                                if let Some(&idx) = table.index.get(&worker) {
                                    let load = {
                                        let slot = &table.slots[idx];
                                        let mut seq = slot.sequences.write();
                                        let delta = seq.free(&event.request_id, decay_now);
                                        let load = seq.worker_load_snapshot();
                                        self.prompt_registry.apply_membership_delta_and_load(
                                            worker,
                                            &slot.trie_lookup,
                                            delta,
                                            load,
                                        );
                                        load
                                    };
                                    drop(table);
                                    self.publish_worker_load_snapshot(worker, load, decay_now);
                                    remote_capacity_changed = true;
                                }
                            }
                        }
                        ActiveSequenceEventData::MarkPrefillCompleted => {
                            let worker = self.request_index.worker_for(&event.request_id);
                            if let Some(worker) = worker {
                                let table = self.workers.read();
                                if let Some(&idx) = table.index.get(&worker) {
                                    {
                                        let mut seq = table.slots[idx].sequences.write();
                                        seq.mark_prefill_completed(&event.request_id, decay_now);
                                        let load = seq.worker_load_snapshot();
                                        self.prompt_registry.replace_worker_load_state(worker, load);
                                    }
                                    drop(table);
                                    remote_capacity_changed = true;
                                }
                            }
                        }
                    }

                    if remote_capacity_changed {
                        let _ = self.remote_state_updates.send(());
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Subscription task cancelled");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Register externally-provided workers (e.g. from EPP) in the slot tracker,
    /// adding any that are missing.
    ///
    /// Unlike [`update_workers`], this does not remove workers absent from the
    /// input — it only adds new ones.  This is intentional: the EPP may send
    /// different subsets of workers on different requests, and one routing call
    /// must not evict workers registered by another.
    ///
    /// Worker removal in External mode will be handled separately via GAIE
    /// lifecycle events (not yet implemented). TODO (atchernych) once we upgrade to GAIE latest.
    pub fn register_external_workers(&self, dp_range: &HashMap<u64, (u32, u32)>) {
        let change = {
            let mut table = self.workers.write();
            table.register_external(self.block_size, dp_range)
        };

        for worker in &change.added {
            tracing::debug!("Lazily registering external worker {:?}", worker);
        }
        self.prompt_registry.apply_topology_change(change);
    }

    /// Update the set of workers, adding and removing as needed.
    ///
    /// `new_dp_range` maps worker IDs to their data-parallel range (start, size).
    pub fn update_workers(&self, new_dp_range: &HashMap<u64, (u32, u32)>) {
        let change = {
            let mut table = self.workers.write();
            table.reconcile(self.block_size, new_dp_range)
        };

        for removed in &change.removed {
            tracing::warn!("Removing worker {:?}", removed.worker);
            self.request_index.remove_worker_requests(removed.worker);
        }
        for worker in &change.added {
            tracing::warn!("Adding worker {:?}", worker);
        }

        self.prompt_registry.apply_topology_change(change);
    }

    pub fn add_request(
        &self,
        req: SequenceRequest,
        decay_now: Instant,
    ) -> Result<(), SequenceError> {
        let event = ActiveSequenceEvent {
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
        };
        self.add_request_local(req, decay_now)?;
        self.spawn_publish_event(event);
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
    // TODO: output blocks are not replicated via replica_sync — add an
    // ActiveSequenceEventData variant if cross-instance accuracy matters.
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

    /// Query all workers for the number of new blocks that would be added by a token sequence.
    pub fn new_blocks(&self, token_sequence: &[SequenceHash]) -> HashMap<WorkerWithDpRank, usize> {
        let table = self.workers.read();
        let mut results = HashMap::with_capacity(table.slots.len());
        for slot in &table.slots {
            results.insert(
                slot.worker,
                slot.sequences.read().new_blocks(token_sequence),
            );
        }
        results
    }

    /// Query all workers for the total number of blocks (new + active) that would be used.
    pub fn potential_blocks(
        &self,
        token_sequence: &[SequenceHash],
    ) -> HashMap<WorkerWithDpRank, usize> {
        let table = self.workers.read();
        let mut results = HashMap::with_capacity(table.slots.len());
        for slot in &table.slots {
            results.insert(
                slot.worker,
                slot.sequences.read().potential_blocks(token_sequence),
            );
        }
        results
    }

    /// Query all workers for the potential blocks and tokens.
    pub fn potential_blocks_and_tokens(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlaps: OverlapScores,
        decay_now: Instant,
    ) -> (
        FxHashMap<WorkerWithDpRank, usize>,
        FxHashMap<WorkerWithDpRank, usize>,
    ) {
        self.potential_blocks_and_tokens_with_prefill_tracking(
            token_sequence,
            isl,
            overlaps,
            true,
            decay_now,
        )
    }

    pub fn potential_blocks_and_tokens_with_prefill_tracking(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlaps: OverlapScores,
        track_prefill_tokens: bool,
        decay_now: Instant,
    ) -> (
        FxHashMap<WorkerWithDpRank, usize>,
        FxHashMap<WorkerWithDpRank, usize>,
    ) {
        self.prompt_registry
            .potential_blocks_and_tokens_with_prefill_tracking(
                token_sequence,
                isl,
                &overlaps,
                track_prefill_tokens,
                self.block_size,
                decay_now,
            )
    }

    /// Query all workers for their current number of active blocks.
    pub fn active_blocks(&self) -> HashMap<WorkerWithDpRank, usize> {
        self.prompt_registry.active_blocks()
    }

    /// Query all workers for their current number of active tokens.
    pub fn active_tokens(&self, decay_now: Instant) -> HashMap<WorkerWithDpRank, usize> {
        self.prompt_registry.active_tokens(decay_now)
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

    fn ensure_worker_registered(&self, worker: WorkerWithDpRank) {
        if self.workers.read().index.contains_key(&worker) {
            return;
        }

        let mut table = self.workers.write();
        if table.index.contains_key(&worker) {
            return;
        }

        tracing::debug!(?worker, "Lazily registering worker in slot tracker");
        let change = table.ensure_worker(self.block_size, worker);
        drop(table);

        self.prompt_registry.apply_topology_change(change);
    }

    fn add_request_local(
        &self,
        req: SequenceRequest,
        decay_now: Instant,
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

        self.ensure_worker_registered(worker);

        let (expired_request_ids, load) = {
            let table = self.workers.read();
            let &idx = table
                .index
                .get(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
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
            (outcome.expired_request_ids, load)
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
    use std::time::Duration;

    use rustc_hash::FxHashMap;

    use super::super::prefill_tracker::added_prefill_tokens;
    use super::*;
    use crate::protocols::{
        ActiveSequenceEvent, ActiveSequenceEventData, BlockHashOptions, OverlapScores,
        PrefillLoadHint, compute_block_hash_for_seq, compute_seq_hash_for_block,
    };
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

    fn naive_potential_loads(
        sequences: &ActiveSequencesMultiWorker<NoopSequencePublisher>,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlaps: &OverlapScores,
        track_prefill_tokens: bool,
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
            let overlap = *overlaps.scores.get(&slot.worker).unwrap_or(&0);
            let added_tokens = if track_prefill_tokens {
                added_prefill_tokens(sequences.block_size, isl, overlap)
            } else {
                0
            };
            potential_blocks.insert(slot.worker, seq.active_blocks() + new_blocks);
            potential_tokens.insert(slot.worker, seq.active_tokens(decay_now) + added_tokens);
        }
        (potential_blocks, potential_tokens)
    }

    fn seq_hashes_for_tokens(tokens: &[u32], lora_name: Option<&str>) -> Vec<SequenceHash> {
        let block_hashes = compute_block_hash_for_seq(
            tokens,
            4,
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

    struct VecSubscriber {
        events: VecDeque<anyhow::Result<ActiveSequenceEvent>>,
    }

    impl SequenceSubscriber for VecSubscriber {
        fn next_event(
            &mut self,
        ) -> impl Future<Output = Option<anyhow::Result<ActiveSequenceEvent>>> + Send {
            future::ready(self.events.pop_front())
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
        let mut expected_overlaps = OverlapScores::new();
        expected_overlaps.scores.insert(worker_a, 2);
        expected_overlaps.scores.insert(worker_b, 1);
        let expected = naive_potential_loads(
            &sequences,
            Some(&prompt),
            16,
            &expected_overlaps,
            true,
            decay_now,
        );

        let mut actual_overlaps = OverlapScores::new();
        actual_overlaps.scores.insert(worker_a, 2);
        actual_overlaps.scores.insert(worker_b, 1);
        let actual = sequences.potential_blocks_and_tokens_with_prefill_tracking(
            Some(&prompt),
            16,
            actual_overlaps,
            true,
            decay_now,
        );

        assert_eq!(actual.0, expected.0);
        assert_eq!(actual.1, expected.1);
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
            8,
            &OverlapScores::default(),
            false,
            decay_now,
        );
        let actual = sequences.potential_blocks_and_tokens_with_prefill_tracking(
            Some(&base_prompt),
            8,
            OverlapScores::default(),
            false,
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
            12,
            &OverlapScores::default(),
            false,
            Instant::now(),
        );
        let actual = sequences.potential_blocks_and_tokens_with_prefill_tracking(
            Some(&[1, 2, 3]),
            12,
            OverlapScores::default(),
            false,
            Instant::now(),
        );
        assert_eq!(actual, expected);
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

        sequences.update_workers(&HashMap::new());
        assert!(sequences.prompt_registry.is_block_index_empty());
        assert!(sequences.active_blocks().is_empty());
        assert!(sequences.request_index.is_empty());

        sequences.update_workers(&HashMap::from([(1_u64, (0_u32, 1_u32))]));
        assert_eq!(sequences.active_blocks().get(&worker).copied(), Some(0));
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

        assert!(sequences.request_index.is_empty());
        assert!(sequences.prompt_registry.is_block_index_empty());
        assert_eq!(sequences.active_blocks().get(&worker).copied(), Some(0));
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

        let (_, potential_tokens) = sequences.potential_blocks_and_tokens_with_prefill_tracking(
            None,
            0,
            OverlapScores::default(),
            false,
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
