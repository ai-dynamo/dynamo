// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-worker extension of [`ActiveSequences`] with per-worker `parking_lot::RwLock` for
//! fine-grained concurrent access, with pluggable event publishing and metric observation via
//! traits.
//!
//! The two traits [`SequencePublisher`] and [`SequenceSubscriber`] abstract the runtime-specific
//! transport (e.g., NATS EventPublisher, Prometheus gauges) so that all business logic lives in
//! this crate while the runtime glue stays in `lib/llm`.

use dashmap::DashMap;
use dynamo_tokens::SequenceHash;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::Arc;
use tokio::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;

use super::single::{ActiveSequences, RequestId};
use crate::protocols::{
    ActiveLoad, ActiveSequenceEvent, ActiveSequenceEventData, OverlapScores, WorkerWithDpRank,
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
    pub isl: usize,
    pub overlap: u32,
    pub track_prefill_tokens: bool,
    pub expected_output_tokens: Option<u32>,
    pub worker: WorkerWithDpRank,
    pub lora_name: Option<String>,
}

// ---------------------------------------------------------------------------
// WorkerTable
// ---------------------------------------------------------------------------

struct WorkerTable {
    slots: Vec<(WorkerWithDpRank, RwLock<ActiveSequences>)>,
    index: HashMap<WorkerWithDpRank, usize>,
}

impl WorkerTable {
    fn new(block_size: usize, dp_range: &HashMap<u64, (u32, u32)>) -> Self {
        let mut slots = Vec::new();
        let mut index = HashMap::new();
        for (&worker_id, &(dp_start, dp_size)) in dp_range {
            for dp_rank in dp_start..dp_start + dp_size {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                let idx = slots.len();
                slots.push((worker, RwLock::new(ActiveSequences::new(block_size))));
                index.insert(worker, idx);
            }
        }
        Self { slots, index }
    }
}

// ---------------------------------------------------------------------------
// ActiveSequencesMultiWorker
// ---------------------------------------------------------------------------

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
    request_to_worker: DashMap<RequestId, WorkerWithDpRank>,
    request_to_lora: DashMap<RequestId, String>,
    block_size: usize,
    router_id: u64,
    publisher: Arc<P>,
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

        Self {
            workers: RwLock::new(WorkerTable::new(block_size, &dp_range)),
            request_to_worker: DashMap::new(),
            request_to_lora: DashMap::new(),
            block_size,
            router_id,
            publisher: Arc::new(publisher),
            replica_sync,
            worker_type,
        }
    }

    fn spawn_publish_event(&self, event: ActiveSequenceEvent) {
        if !self.replica_sync {
            return;
        }

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

                    match &event.data {
                        ActiveSequenceEventData::AddRequest {
                            token_sequence,
                            isl,
                            overlap,
                            track_prefill_tokens,
                            expected_output_tokens,
                        } => {
                            self.request_to_worker
                                .insert(event.request_id.clone(), event.worker);

                            if let Some(ref lora_name) = event.lora_name {
                                self.request_to_lora
                                    .insert(event.request_id.clone(), lora_name.clone());
                            }

                            let table = self.workers.read();
                            if let Some(&idx) = table.index.get(&event.worker) {
                                table.slots[idx].1.write().add_request_with_prefill_tracking(
                                    event.request_id.clone(),
                                    token_sequence.clone(),
                                    *isl,
                                    *overlap,
                                    *expected_output_tokens,
                                    *track_prefill_tokens,
                                );
                            } else {
                                tracing::warn!(
                                    "Worker {:?} not found, cannot process AddRequest",
                                    event.worker
                                );
                            }
                        }
                        ActiveSequenceEventData::Free => {
                            if let Some((_, worker)) =
                                self.request_to_worker.remove(&event.request_id)
                            {
                                let table = self.workers.read();
                                if let Some(&idx) = table.index.get(&worker) {
                                    table.slots[idx].1.write().free(&event.request_id);
                                }
                            }
                            self.request_to_lora.remove(&event.request_id);
                        }
                        ActiveSequenceEventData::MarkPrefillCompleted => {
                            let worker =
                                self.request_to_worker.get(&event.request_id).map(|r| *r);
                            if let Some(worker) = worker {
                                let table = self.workers.read();
                                if let Some(&idx) = table.index.get(&worker) {
                                    table.slots[idx]
                                        .1
                                        .write()
                                        .mark_prefill_completed(&event.request_id);
                                }
                            }
                        }
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
        let mut table = self.workers.write();
        for (&worker_id, &(dp_start, dp_size)) in dp_range {
            for dp_rank in dp_start..(dp_start + dp_size) {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                if !table.index.contains_key(&worker) {
                    tracing::debug!("Lazily registering external worker {:?}", worker);
                    let idx = table.slots.len();
                    table
                        .slots
                        .push((worker, RwLock::new(ActiveSequences::new(self.block_size))));
                    table.index.insert(worker, idx);
                }
            }
        }
    }

    /// Update the set of workers, adding and removing as needed.
    ///
    /// `new_dp_range` maps worker IDs to their data-parallel range (start, size).
    pub fn update_workers(&self, new_dp_range: &HashMap<u64, (u32, u32)>) {
        let mut table = self.workers.write();

        let mut target_workers: HashSet<WorkerWithDpRank> = HashSet::new();
        for (&worker_id, &(dp_start, dp_size)) in new_dp_range {
            for dp_rank in dp_start..(dp_start + dp_size) {
                target_workers.insert(WorkerWithDpRank::new(worker_id, dp_rank));
            }
        }

        // Clean up request mappings for workers being removed.
        for (worker, _) in &table.slots {
            if target_workers.contains(worker) {
                continue;
            }
            tracing::warn!("Removing worker {:?}", worker);

            let requests_to_remove: Vec<RequestId> = self
                .request_to_worker
                .iter()
                .filter(|entry| entry.value() == worker)
                .map(|entry| entry.key().clone())
                .collect();

            self.request_to_worker
                .retain(|_request_id, mapped_worker| mapped_worker != worker);

            for request_id in requests_to_remove {
                self.request_to_lora.remove(&request_id);
            }
        }

        // Drain old slots, preserving ActiveSequences for retained workers.
        let mut old: HashMap<WorkerWithDpRank, ActiveSequences> = table
            .slots
            .drain(..)
            .map(|(w, lock)| (w, lock.into_inner()))
            .collect();
        table.index.clear();

        // Rebuild with target workers, reusing state where possible.
        for worker in target_workers {
            if !old.contains_key(&worker) {
                tracing::warn!("Adding worker {:?}", worker);
            }
            let idx = table.slots.len();
            let seq = old
                .remove(&worker)
                .unwrap_or_else(|| ActiveSequences::new(self.block_size));
            table.slots.push((worker, RwLock::new(seq)));
            table.index.insert(worker, idx);
        }
    }

    fn add_request_local(&self, req: SequenceRequest) -> Result<(), SequenceError> {
        let SequenceRequest {
            request_id,
            token_sequence,
            isl,
            overlap,
            track_prefill_tokens,
            expected_output_tokens,
            worker,
            lora_name,
        } = req;

        if !self.workers.read().index.contains_key(&worker) {
            return Err(SequenceError::WorkerNotFound { worker });
        }

        if let Some(existing_worker) = self.request_to_worker.get(&request_id) {
            return Err(SequenceError::DuplicateRequest {
                request_id,
                worker: *existing_worker,
            });
        }

        self.request_to_worker.insert(request_id.clone(), worker);

        if let Some(lora) = lora_name {
            self.request_to_lora.insert(request_id.clone(), lora);
        }

        let removed_requests = {
            let table = self.workers.read();
            let &idx = table
                .index
                .get(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            let mut seq = table.slots[idx].1.write();
            seq.add_request_with_prefill_tracking(
                request_id,
                token_sequence,
                isl,
                overlap,
                expected_output_tokens,
                track_prefill_tokens,
            )
        };

        for expired_id in &removed_requests {
            self.request_to_worker.remove(expired_id);
            self.request_to_lora.remove(expired_id);
        }

        self.publish_active_load_for_worker(worker);

        Ok(())
    }

    pub fn add_request(&self, req: SequenceRequest) -> Result<(), SequenceError> {
        self.spawn_publish_event(ActiveSequenceEvent {
            request_id: req.request_id.clone(),
            worker: req.worker,
            data: ActiveSequenceEventData::AddRequest {
                token_sequence: req.token_sequence.clone(),
                isl: req.isl,
                overlap: req.overlap,
                track_prefill_tokens: req.track_prefill_tokens,
                expected_output_tokens: req.expected_output_tokens,
            },
            router_id: self.router_id,
            lora_name: req.lora_name.clone(),
        });
        self.add_request_local(req)
    }

    /// Send a mutation to the worker assigned to a request, optionally publishing
    /// a replica-sync event and cleaning up request mappings afterward.
    fn mutate_request_worker_local(
        &self,
        request_id: &RequestId,
        mutate_fn: impl FnOnce(&mut ActiveSequences, &RequestId),
        remove_mapping: bool,
    ) -> Result<(), SequenceError> {
        let worker = self
            .request_to_worker
            .get(request_id)
            .map(|entry| *entry)
            .ok_or_else(|| SequenceError::RequestNotFound {
                request_id: request_id.clone(),
            })?;

        {
            let table = self.workers.read();
            let &idx = table
                .index
                .get(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            let mut seq = table.slots[idx].1.write();
            mutate_fn(&mut seq, request_id);
        }

        if remove_mapping {
            self.request_to_worker.remove(request_id);
            self.request_to_lora.remove(request_id);
        }

        self.publish_active_load_for_worker(worker);

        Ok(())
    }

    fn mutate_request_worker(
        &self,
        request_id: &RequestId,
        event_data: ActiveSequenceEventData,
        mutate_fn: impl FnOnce(&mut ActiveSequences, &RequestId),
        remove_mapping: bool,
    ) -> Result<(), SequenceError> {
        let worker = self
            .request_to_worker
            .get(request_id)
            .map(|entry| *entry)
            .ok_or_else(|| SequenceError::RequestNotFound {
                request_id: request_id.clone(),
            })?;

        let lora_name = self
            .request_to_lora
            .get(request_id)
            .map(|entry| entry.value().clone());
        self.spawn_publish_event(ActiveSequenceEvent {
            request_id: request_id.clone(),
            worker,
            data: event_data,
            router_id: self.router_id,
            lora_name,
        });

        self.mutate_request_worker_local(request_id, mutate_fn, remove_mapping)
    }

    /// Free all blocks associated with a request.
    ///
    /// Note: This operation is idempotent. Calling it multiple times for the same request
    /// will log a warning but not return an error (double free is allowed).
    ///
    /// This also performs the underlying prefill-complete cleanup via
    /// [`ActiveSequences::free`], so callers do not need to call
    /// [`Self::mark_prefill_completed`] before freeing a completed request.
    pub fn free(&self, request_id: &RequestId) -> Result<(), SequenceError> {
        if !self.request_to_worker.contains_key(request_id) {
            tracing::debug!("Request {request_id} not found, already freed (idempotent)");
            return Ok(());
        }

        self.mutate_request_worker(
            request_id,
            ActiveSequenceEventData::Free,
            |seqs, rid| {
                seqs.free(rid);
            },
            true,
        )
    }

    /// Mark prefill as completed for a request.
    ///
    /// Note: Calling this multiple times for the same request is allowed and will be a no-op
    /// after the first call (idempotent).
    pub fn mark_prefill_completed(&self, request_id: &RequestId) -> Result<(), SequenceError> {
        self.mutate_request_worker(
            request_id,
            ActiveSequenceEventData::MarkPrefillCompleted,
            |seqs, rid| {
                seqs.mark_prefill_completed(rid);
            },
            false,
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
        let worker = self
            .request_to_worker
            .get(request_id)
            .map(|entry| *entry)
            .ok_or_else(|| SequenceError::RequestNotFound {
                request_id: request_id.clone(),
            })?;

        let success = {
            let table = self.workers.read();
            let &idx = table
                .index
                .get(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            let mut seq = table.slots[idx].1.write();
            seq.add_output_block(request_id, decay_fraction)
        };

        if !success {
            return Err(SequenceError::RequestNotFound {
                request_id: request_id.clone(),
            });
        }

        self.publish_active_load_for_worker(worker);

        Ok(())
    }

    /// Read active blocks/tokens from a worker and publish ActiveLoad metrics.
    fn publish_active_load_for_worker(&self, worker: WorkerWithDpRank) {
        let (active_blocks, active_tokens) = {
            let table = self.workers.read();
            let Some(&idx) = table.index.get(&worker) else {
                tracing::warn!("Worker {worker:?} not found when publishing ActiveLoad");
                return;
            };
            let seq = table.slots[idx].1.read();
            (seq.active_blocks(), seq.active_tokens())
        };

        self.publisher
            .observe_load(&worker, self.worker_type, active_blocks, active_tokens);

        let active_load = ActiveLoad {
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            active_decode_blocks: Some(active_blocks as u64),
            active_prefill_tokens: Some(active_tokens as u64),
        };

        self.publisher.publish_load(active_load);
    }

    /// Get the number of workers.
    pub fn num_workers(&self) -> usize {
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
        for (worker, lock) in &table.slots {
            results.insert(*worker, lock.read().new_blocks(token_sequence));
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
        for (worker, lock) in &table.slots {
            results.insert(*worker, lock.read().potential_blocks(token_sequence));
        }
        results
    }

    /// Query all workers for the potential blocks and tokens.
    pub fn potential_blocks_and_tokens(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlaps: OverlapScores,
    ) -> (
        HashMap<WorkerWithDpRank, usize>,
        HashMap<WorkerWithDpRank, usize>,
    ) {
        self.potential_blocks_and_tokens_with_prefill_tracking(token_sequence, isl, overlaps, true)
    }

    pub fn potential_blocks_and_tokens_with_prefill_tracking(
        &self,
        token_sequence: Option<&[SequenceHash]>,
        isl: usize,
        overlaps: OverlapScores,
        track_prefill_tokens: bool,
    ) -> (
        HashMap<WorkerWithDpRank, usize>,
        HashMap<WorkerWithDpRank, usize>,
    ) {
        #[cfg(feature = "bench")]
        let start = tokio::time::Instant::now();

        let table = self.workers.read();

        #[cfg(feature = "bench")]
        let num_workers = table.slots.len();

        let mut potential_blocks = HashMap::with_capacity(table.slots.len());
        let mut potential_tokens = HashMap::with_capacity(table.slots.len());

        for (worker, lock) in &table.slots {
            let overlap = *overlaps.scores.get(worker).unwrap_or(&0);

            let (blocks, tokens) = lock
                .read()
                .potential_blocks_and_tokens_with_prefill_tracking(
                    token_sequence,
                    isl,
                    overlap,
                    track_prefill_tokens,
                );
            potential_blocks.insert(*worker, blocks);
            potential_tokens.insert(*worker, tokens);
        }

        #[cfg(feature = "bench")]
        {
            let total_elapsed = start.elapsed();
            tracing::info!(
                num_workers,
                total_us = total_elapsed.as_micros() as u64,
                "potential_blocks_and_tokens completed"
            );
        }

        (potential_blocks, potential_tokens)
    }

    /// Query all workers for their current number of active blocks.
    pub fn active_blocks(&self) -> HashMap<WorkerWithDpRank, usize> {
        let table = self.workers.read();
        let mut results = HashMap::with_capacity(table.slots.len());
        for (worker, lock) in &table.slots {
            results.insert(*worker, lock.read().active_blocks());
        }
        results
    }

    /// Query all workers for their current number of active tokens.
    pub fn active_tokens(&self) -> HashMap<WorkerWithDpRank, usize> {
        let table = self.workers.read();
        let mut results = HashMap::with_capacity(table.slots.len());
        for (worker, lock) in &table.slots {
            results.insert(*worker, lock.read().active_tokens());
        }
        results
    }

    /// Return true if any worker satisfies the provided predicate on active token count.
    pub fn any_worker_matches_active_tokens(
        &self,
        mut predicate: impl FnMut(WorkerWithDpRank, usize) -> bool,
    ) -> bool {
        let table = self.workers.read();
        for (worker, lock) in &table.slots {
            if predicate(*worker, lock.read().active_tokens()) {
                return true;
            }
        }
        false
    }

    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for entry in self.request_to_lora.iter() {
            let lora_name = entry.value().clone();
            *counts.entry(lora_name).or_insert(0) += 1;
        }
        counts
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
        for (worker, lock) in &table.slots {
            let removed_requests = lock.write().force_expiry();
            if !removed_requests.is_empty() {
                for expired_id in &removed_requests {
                    self.request_to_worker.remove(expired_id);
                    self.request_to_lora.remove(expired_id);
                    removed_request_count += 1;
                }
                self.publish_active_load_for_worker(*worker);
            }
        }
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
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
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

    #[tokio::test]
    async fn add_request_can_skip_prefill_token_tracking() {
        let sequences = make_sequences();
        let worker = WorkerWithDpRank::new(1, 0);

        sequences
            .add_request(SequenceRequest {
                request_id: "req-1".to_string(),
                token_sequence: Some(vec![1, 2, 3]),
                isl: 12,
                overlap: 0,
                track_prefill_tokens: false,
                expected_output_tokens: None,
                worker,
                lora_name: None,
            })
            .unwrap();

        assert_eq!(sequences.active_tokens().get(&worker).copied(), Some(0));
    }
}
