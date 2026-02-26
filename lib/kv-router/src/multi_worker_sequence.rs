// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multi-worker extension of [`ActiveSequences`] with per-worker `RwLock` for
//! fine-grained concurrent access, plus pluggable event publishing and metric
//! observation via traits.
//!
//! The two traits [`SequencePublisher`] and [`SequenceSubscriber`] abstract the
//! runtime-specific transport (e.g., NATS EventPublisher, Prometheus gauges) so
//! that all business logic lives in this crate while the runtime glue stays in
//! `lib/llm`.

use dashmap::DashMap;
use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::sync::{Arc, RwLock};
use tokio_util::sync::CancellationToken;

use crate::protocols::{
    ActiveLoad, ActiveSequenceEvent, ActiveSequenceEventData, OverlapScores, WorkerWithDpRank,
};
use crate::sequence::{ActiveSequences, RequestId};

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

    #[error("Failed to publish event: {0}")]
    PublishFailed(#[from] anyhow::Error),
}

/// Bundled parameters for adding a request to the sequence tracker.
pub struct SequenceRequest {
    pub request_id: RequestId,
    pub token_sequence: Option<Vec<SequenceHash>>,
    pub isl: usize,
    pub overlap: u32,
    pub expected_output_tokens: Option<u32>,
    pub worker: WorkerWithDpRank,
    pub lora_name: Option<String>,
}

// ---------------------------------------------------------------------------
// WorkerTable
// ---------------------------------------------------------------------------

/// Dense table of workers indexed by position, with a side map for
/// `WorkerWithDpRank → index` lookup.  Each slot has its own `RwLock` so
/// readers (e.g. `potential_blocks_and_tokens` fan-out) never contend with
/// writers targeting a different worker.
struct WorkerTable {
    slots: Vec<(WorkerWithDpRank, RwLock<ActiveSequences>)>,
    index: HashMap<WorkerWithDpRank, usize>,
}

impl WorkerTable {
    fn new(block_size: usize, dp_sizes: &HashMap<u64, u32>) -> Self {
        let mut slots = Vec::new();
        let mut index = HashMap::new();
        for (&worker_id, &dp_size) in dp_sizes {
            for dp_rank in 0..dp_size {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                index.insert(worker, slots.len());
                slots.push((worker, RwLock::new(ActiveSequences::new(block_size))));
            }
        }
        Self { slots, index }
    }
}

// ---------------------------------------------------------------------------
// ActiveSequencesMultiWorker
// ---------------------------------------------------------------------------

/// Multi-worker extension of [`ActiveSequences`] with per-worker `RwLock` for
/// fine-grained concurrent access.
///
/// The outer `RwLock<WorkerTable>` is held briefly for every operation (read
/// lock) and exclusively only during the rare `update_workers` call.  The inner
/// per-slot `RwLock<ActiveSequences>` provides independent read/write access
/// per worker so that a mutation on worker A never blocks readers of worker B.
///
/// Generic over `P: SequencePublisher` to decouple from runtime-specific event
/// transport and metrics infrastructure.
pub struct ActiveSequencesMultiWorker<P: SequencePublisher> {
    workers: RwLock<WorkerTable>,
    request_to_worker: Arc<DashMap<RequestId, WorkerWithDpRank>>,
    request_to_lora: Arc<DashMap<RequestId, String>>,
    block_size: usize,
    router_id: u64,
    publisher: P,
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
        dp_sizes: HashMap<u64, u32>,
        replica_sync: bool,
        router_id: u64,
        worker_type: &'static str,
    ) -> Self {
        assert!(block_size > 1, "block_size must be greater than 1");

        Self {
            workers: RwLock::new(WorkerTable::new(block_size, &dp_sizes)),
            request_to_worker: Arc::new(DashMap::new()),
            request_to_lora: Arc::new(DashMap::new()),
            block_size,
            router_id,
            publisher,
            replica_sync,
            worker_type,
        }
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
                            expected_output_tokens,
                        } => {
                            self.request_to_worker
                                .insert(event.request_id.clone(), event.worker);

                            if let Some(ref lora_name) = event.lora_name {
                                self.request_to_lora
                                    .insert(event.request_id.clone(), lora_name.clone());
                            }

                            let table = self.workers.read().unwrap();
                            if let Some(&idx) = table.index.get(&event.worker) {
                                let mut seq = table.slots[idx].1.write().unwrap();
                                seq.add_request(
                                    event.request_id.clone(),
                                    token_sequence.clone(),
                                    *isl,
                                    *overlap,
                                    *expected_output_tokens,
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
                                let table = self.workers.read().unwrap();
                                if let Some(&idx) = table.index.get(&worker) {
                                    let mut seq = table.slots[idx].1.write().unwrap();
                                    seq.free(&event.request_id);
                                }
                            }
                            self.request_to_lora.remove(&event.request_id);
                        }
                        ActiveSequenceEventData::MarkPrefillCompleted => {
                            if let Some(worker) =
                                self.request_to_worker.get(&event.request_id)
                            {
                                let table = self.workers.read().unwrap();
                                if let Some(&idx) = table.index.get(&*worker) {
                                    let mut seq = table.slots[idx].1.write().unwrap();
                                    seq.mark_prefill_completed(&event.request_id);
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

    /// Update the set of workers, adding and removing as needed.
    ///
    /// `new_dp_sizes` maps worker IDs to their data-parallel size.
    pub fn update_workers(&self, new_dp_sizes: &HashMap<u64, u32>) {
        let mut table = self.workers.write().unwrap();

        let current_workers: HashSet<WorkerWithDpRank> = table.index.keys().copied().collect();

        let mut target_workers: HashSet<WorkerWithDpRank> = HashSet::new();
        for (&worker_id, &dp_size) in new_dp_sizes {
            for dp_rank in 0..dp_size {
                target_workers.insert(WorkerWithDpRank::new(worker_id, dp_rank));
            }
        }

        let workers_to_remove: HashSet<WorkerWithDpRank> = current_workers
            .difference(&target_workers)
            .copied()
            .collect();

        // Clean up request mappings for removed workers
        for worker in &workers_to_remove {
            tracing::warn!("Removing worker {:?}", worker);

            self.request_to_worker
                .retain(|_request_id, mapped_worker| mapped_worker != worker);

            let requests_to_remove: Vec<RequestId> = self
                .request_to_worker
                .iter()
                .filter(|entry| entry.value() == worker)
                .map(|entry| entry.key().clone())
                .collect();

            for request_id in requests_to_remove {
                self.request_to_lora.remove(&request_id);
            }
        }

        // Drain old slots, preserving ActiveSequences for retained workers
        let old_slots = std::mem::take(&mut table.slots);
        let mut preserved: HashMap<WorkerWithDpRank, ActiveSequences> = old_slots
            .into_iter()
            .filter(|(w, _)| !workers_to_remove.contains(w))
            .map(|(w, lock)| (w, lock.into_inner().unwrap()))
            .collect();

        // Rebuild table with new worker set
        let mut new_slots = Vec::new();
        let mut new_index = HashMap::new();
        for &worker in &target_workers {
            let seq = preserved.remove(&worker).unwrap_or_else(|| {
                tracing::warn!("Adding worker {:?}", worker);
                ActiveSequences::new(self.block_size)
            });
            new_index.insert(worker, new_slots.len());
            new_slots.push((worker, RwLock::new(seq)));
        }

        table.slots = new_slots;
        table.index = new_index;
    }

    pub async fn add_request(&self, req: SequenceRequest) -> Result<(), SequenceError> {
        let SequenceRequest {
            request_id,
            token_sequence,
            isl,
            overlap,
            expected_output_tokens,
            worker,
            lora_name,
        } = req;

        {
            let table = self.workers.read().unwrap();
            if !table.index.contains_key(&worker) {
                return Err(SequenceError::WorkerNotFound { worker });
            }
        }

        if let Some(existing_worker) = self.request_to_worker.get(&request_id) {
            return Err(SequenceError::DuplicateRequest {
                request_id,
                worker: *existing_worker,
            });
        }

        if self.replica_sync {
            let event = ActiveSequenceEvent {
                request_id: request_id.clone(),
                worker,
                data: ActiveSequenceEventData::AddRequest {
                    token_sequence: token_sequence.clone(),
                    isl,
                    overlap,
                    expected_output_tokens,
                },
                router_id: self.router_id,
                lora_name: lora_name.clone(),
            };
            self.publisher.publish_event(&event).await?;
        }

        self.request_to_worker.insert(request_id.clone(), worker);

        if let Some(lora) = lora_name {
            self.request_to_lora.insert(request_id.clone(), lora);
        }

        let removed_requests = {
            let table = self.workers.read().unwrap();
            let &idx = table
                .index
                .get(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            let mut seq = table.slots[idx].1.write().unwrap();
            seq.add_request(
                request_id,
                token_sequence,
                isl,
                overlap,
                expected_output_tokens,
            )
        };

        for expired_id in &removed_requests {
            self.request_to_worker.remove(expired_id);
            self.request_to_lora.remove(expired_id);
        }

        self.publish_active_load_for_worker(worker);

        Ok(())
    }

    /// Send a mutation to the worker assigned to a request, optionally publishing
    /// a replica-sync event and cleaning up request mappings afterward.
    async fn mutate_request_worker(
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

        if self.replica_sync {
            let lora_name = self
                .request_to_lora
                .get(request_id)
                .map(|entry| entry.value().clone());

            let event = ActiveSequenceEvent {
                request_id: request_id.clone(),
                worker,
                data: event_data,
                router_id: self.router_id,
                lora_name,
            };
            self.publisher.publish_event(&event).await?;
        }

        {
            let table = self.workers.read().unwrap();
            let &idx = table
                .index
                .get(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            let mut seq = table.slots[idx].1.write().unwrap();
            mutate_fn(&mut seq, request_id);
        }

        if remove_mapping {
            self.request_to_worker.remove(request_id);
            self.request_to_lora.remove(request_id);
        }

        self.publish_active_load_for_worker(worker);

        Ok(())
    }

    /// Free all blocks associated with a request.
    ///
    /// Note: This operation is idempotent. Calling it multiple times for the same request
    /// will log a warning but not return an error (double free is allowed).
    pub async fn free(&self, request_id: &RequestId) -> Result<(), SequenceError> {
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
        .await
    }

    /// Mark prefill as completed for a request.
    ///
    /// Note: Calling this multiple times for the same request is allowed and will be a no-op
    /// after the first call (idempotent).
    pub async fn mark_prefill_completed(
        &self,
        request_id: &RequestId,
    ) -> Result<(), SequenceError> {
        self.mutate_request_worker(
            request_id,
            ActiveSequenceEventData::MarkPrefillCompleted,
            |seqs, rid| {
                seqs.mark_prefill_completed(rid);
            },
            false,
        )
        .await
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
            let table = self.workers.read().unwrap();
            let &idx = table
                .index
                .get(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            let mut seq = table.slots[idx].1.write().unwrap();
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
            let table = self.workers.read().unwrap();
            let Some(&idx) = table.index.get(&worker) else {
                tracing::warn!("Worker {worker:?} not found when publishing ActiveLoad");
                return;
            };
            let seq = table.slots[idx].1.read().unwrap();
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
        self.workers.read().unwrap().slots.len()
    }

    /// Get the worker type for this router ("prefill" or "decode").
    pub fn worker_type(&self) -> &'static str {
        self.worker_type
    }

    /// Query all workers for the number of new blocks that would be added by a token sequence.
    pub fn new_blocks(&self, token_sequence: &[SequenceHash]) -> HashMap<WorkerWithDpRank, usize> {
        let table = self.workers.read().unwrap();
        let mut results = HashMap::with_capacity(table.slots.len());
        for (worker, lock) in &table.slots {
            let seq = lock.read().unwrap();
            results.insert(*worker, seq.new_blocks(token_sequence));
        }
        results
    }

    /// Query all workers for the total number of blocks (new + active) that would be used.
    pub fn potential_blocks(
        &self,
        token_sequence: &[SequenceHash],
    ) -> HashMap<WorkerWithDpRank, usize> {
        let table = self.workers.read().unwrap();
        let mut results = HashMap::with_capacity(table.slots.len());
        for (worker, lock) in &table.slots {
            let seq = lock.read().unwrap();
            results.insert(*worker, seq.potential_blocks(token_sequence));
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
        #[cfg(feature = "bench")]
        let start = tokio::time::Instant::now();

        let table = self.workers.read().unwrap();

        #[cfg(feature = "bench")]
        let num_workers = table.slots.len();

        let mut potential_blocks = HashMap::with_capacity(table.slots.len());
        let mut potential_tokens = HashMap::with_capacity(table.slots.len());

        for (worker, lock) in &table.slots {
            let overlap = *overlaps.scores.get(worker).unwrap_or(&0);
            let seq = lock.read().unwrap();
            let (blocks, tokens) = seq.potential_blocks_and_tokens(token_sequence, isl, overlap);
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
        let table = self.workers.read().unwrap();
        let mut results = HashMap::with_capacity(table.slots.len());
        for (worker, lock) in &table.slots {
            let seq = lock.read().unwrap();
            results.insert(*worker, seq.active_blocks());
        }
        results
    }

    /// Query all workers for their current number of active tokens.
    pub fn active_tokens(&self) -> HashMap<WorkerWithDpRank, usize> {
        let table = self.workers.read().unwrap();
        let mut results = HashMap::with_capacity(table.slots.len());
        for (worker, lock) in &table.slots {
            let seq = lock.read().unwrap();
            results.insert(*worker, seq.active_tokens());
        }
        results
    }

    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for entry in self.request_to_lora.iter() {
            let lora_name = entry.value().clone();
            *counts.entry(lora_name).or_insert(0) += 1;
        }
        counts
    }
}
