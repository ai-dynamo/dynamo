// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV Cache Sequence Management for LLM Inference
//!
//! This module provides the multi-worker extension [`ActiveSequencesMultiWorker`] that wraps
//! per-worker [`ActiveSequences`] (from `dynamo_kv_router`) in a shared `DashMap` for
//! lock-free concurrent access, with event-plane replica sync and Prometheus metrics.

pub use dynamo_kv_router::sequence::{ActiveSequences, RequestId};

use anyhow::Result;
use dashmap::DashMap;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::{EventPublisher, EventSubscriber};
use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::metrics::WORKER_LOAD_METRICS;
use super::protocols::{
    ActiveLoad, ActiveSequenceEvent, ActiveSequenceEventData, WorkerWithDpRank,
};
use crate::kv_router::protocols::OverlapScores;
use crate::kv_router::{ACTIVE_SEQUENCES_SUBJECT, KV_METRICS_SUBJECT};
use crate::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_runtime::CancellationToken;

/// Errors that can occur during sequence management operations
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

/// Multi-worker extension of ActiveSequences using shared DashMap for lock-free concurrent access
pub struct ActiveSequencesMultiWorker {
    workers: Arc<DashMap<WorkerWithDpRank, ActiveSequences>>,
    request_to_worker: Arc<DashMap<RequestId, WorkerWithDpRank>>,
    request_to_lora: Arc<DashMap<RequestId, String>>,
    block_size: usize,
    router_id: u64,
    event_publisher: EventPublisher,
    metrics_publisher: Arc<EventPublisher>,
    replica_sync: bool,
    worker_type: &'static str,
}

impl ActiveSequencesMultiWorker {
    pub async fn new(
        component: Component,
        block_size: usize,
        workers_with_configs: HashMap<u64, ModelRuntimeConfig>,
        replica_sync: bool,
        router_id: u64,
        worker_type: &'static str,
    ) -> Result<Self> {
        assert!(block_size > 1, "block_size must be greater than 1");

        let workers = Arc::new(DashMap::new());
        let request_to_worker = Arc::new(DashMap::new());
        let request_to_lora = Arc::new(DashMap::new());

        for (worker_id, config) in workers_with_configs {
            let dp_size = config.data_parallel_size;

            for dp_rank in 0..dp_size {
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                workers.insert(worker, ActiveSequences::new(block_size));
            }
        }

        let event_publisher =
            EventPublisher::for_component(&component, ACTIVE_SEQUENCES_SUBJECT).await?;
        let metrics_publisher = Arc::new(
            EventPublisher::for_namespace(component.namespace(), KV_METRICS_SUBJECT).await?,
        );

        let multi_worker = Self {
            workers: workers.clone(),
            request_to_worker: request_to_worker.clone(),
            request_to_lora: request_to_lora.clone(),
            block_size,
            event_publisher,
            metrics_publisher,
            router_id,
            replica_sync,
            worker_type,
        };

        if replica_sync {
            let workers_clone = workers.clone();
            let request_to_worker_clone = request_to_worker.clone();
            let request_to_lora_clone = request_to_lora.clone();
            let component_clone = component.clone();
            let router_id_clone = router_id;
            let cancel_token = component.drt().runtime().child_token();

            tokio::spawn(async move {
                if let Err(e) = Self::subscribe_to_events(
                    workers_clone,
                    request_to_worker_clone,
                    request_to_lora_clone,
                    component_clone,
                    router_id_clone,
                    cancel_token,
                )
                .await
                {
                    tracing::error!("Error in active sequences events subscription: {}", e);
                }
            });
        }

        Ok(multi_worker)
    }

    /// Background task to subscribe to active sequence events and update all workers
    async fn subscribe_to_events(
        workers: Arc<DashMap<WorkerWithDpRank, ActiveSequences>>,
        request_to_worker: Arc<DashMap<RequestId, WorkerWithDpRank>>,
        request_to_lora: Arc<DashMap<RequestId, String>>,
        component: Component,
        router_id: u64,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        let mut subscriber = EventSubscriber::for_component(&component, ACTIVE_SEQUENCES_SUBJECT)
            .await?
            .typed::<ActiveSequenceEvent>();

        loop {
            tokio::select! {
                result = subscriber.next() => {
                    let Some(result) = result else {
                        break;
                    };

                    let Ok((_envelope, event)) = result else {
                        tracing::error!(
                            "Error receiving active sequence event: {}",
                            result.unwrap_err()
                        );
                        continue;
                    };

                    if event.router_id == router_id {
                        continue;
                    }

                    match &event.data {
                        ActiveSequenceEventData::AddRequest {
                            token_sequence,
                            isl,
                            overlap,
                            expected_output_tokens,
                        } => {
                            request_to_worker.insert(event.request_id.clone(), event.worker);

                            if let Some(ref lora_name) = event.lora_name {
                                request_to_lora.insert(event.request_id.clone(), lora_name.clone());
                            }

                            if let Some(mut entry) = workers.get_mut(&event.worker) {
                                entry.add_request(
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
                            if let Some((_, worker)) = request_to_worker.remove(&event.request_id)
                                && let Some(mut entry) = workers.get_mut(&worker)
                            {
                                entry.free(&event.request_id);
                            }
                            request_to_lora.remove(&event.request_id);
                        }
                        ActiveSequenceEventData::MarkPrefillCompleted => {
                            if let Some(worker) = request_to_worker.get(&event.request_id)
                                && let Some(mut entry) = workers.get_mut(&*worker)
                            {
                                entry.mark_prefill_completed(&event.request_id);
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

    /// Update the set of workers, adding and removing as needed
    pub fn update_workers(&self, new_workers_with_configs: HashMap<u64, ModelRuntimeConfig>) {
        let current_workers: HashSet<WorkerWithDpRank> =
            self.workers.iter().map(|entry| *entry.key()).collect();

        let mut new_workers: HashSet<WorkerWithDpRank> = HashSet::new();
        for (worker_id, config) in &new_workers_with_configs {
            let dp_size = config.data_parallel_size;

            for dp_rank in 0..dp_size {
                new_workers.insert(WorkerWithDpRank::new(*worker_id, dp_rank));
            }
        }

        let workers_to_remove: Vec<WorkerWithDpRank> =
            current_workers.difference(&new_workers).copied().collect();
        let workers_to_add: Vec<WorkerWithDpRank> =
            new_workers.difference(&current_workers).copied().collect();

        for worker in &workers_to_remove {
            tracing::warn!("Removing worker {:?}", worker);

            self.workers.remove(worker);

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

        for worker in &workers_to_add {
            tracing::warn!("Adding worker {:?}", worker);
            self.workers
                .insert(*worker, ActiveSequences::new(self.block_size));
        }
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

        if !self.workers.contains_key(&worker) {
            return Err(SequenceError::WorkerNotFound { worker });
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
            self.event_publisher.publish(&event).await?;
        }

        self.request_to_worker.insert(request_id.clone(), worker);

        if let Some(lora) = lora_name {
            self.request_to_lora.insert(request_id.clone(), lora);
        }

        let removed_requests = {
            let mut entry = self
                .workers
                .get_mut(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            entry.add_request(
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
            self.event_publisher.publish(&event).await?;
        }

        {
            let mut entry = self
                .workers
                .get_mut(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            mutate_fn(&mut entry, request_id);
        }

        if remove_mapping {
            self.request_to_worker.remove(request_id);
            self.request_to_lora.remove(request_id);
        }

        self.publish_active_load_for_worker(worker);

        Ok(())
    }

    /// Free all blocks associated with a request
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

    /// Mark prefill as completed for a request
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

    /// Add an output block with optional fractional decay weight
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
            let mut entry = self
                .workers
                .get_mut(&worker)
                .ok_or(SequenceError::WorkerNotFound { worker })?;
            entry.add_output_block(request_id, decay_fraction)
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
    /// The NATS publish is spawned as a background task to avoid blocking the caller.
    fn publish_active_load_for_worker(&self, worker: WorkerWithDpRank) {
        let (active_blocks, active_tokens) = {
            let Some(entry) = self.workers.get(&worker) else {
                tracing::warn!("Worker {worker:?} not found when publishing ActiveLoad");
                return;
            };
            (entry.active_blocks(), entry.active_tokens())
        };

        WORKER_LOAD_METRICS.observe(
            worker.worker_id,
            worker.dp_rank,
            self.worker_type,
            active_blocks,
            active_tokens,
        );

        let active_load = ActiveLoad {
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            active_decode_blocks: Some(active_blocks as u64),
            active_prefill_tokens: Some(active_tokens as u64),
        };

        let publisher = self.metrics_publisher.clone();
        tokio::spawn(async move {
            if let Err(e) = publisher.publish(&active_load).await {
                tracing::trace!(
                    "Failed to publish ActiveLoad to NATS for worker {worker:?}: {e:?}"
                );
            }
        });
    }

    /// Get the number of workers
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Get the worker type for this router ("prefill" or "decode").
    /// Used for Prometheus metric labeling.
    pub fn worker_type(&self) -> &'static str {
        self.worker_type
    }

    /// Query all workers for the number of new blocks that would be added by a token sequence
    pub fn new_blocks(
        &self,
        token_sequence: Vec<SequenceHash>,
    ) -> HashMap<WorkerWithDpRank, usize> {
        let mut results = HashMap::with_capacity(self.workers.len());
        for entry in self.workers.iter() {
            results.insert(*entry.key(), entry.value().new_blocks(&token_sequence));
        }
        results
    }

    /// Query all workers for the total number of blocks (new + active) that would be used by a token sequence
    pub fn potential_blocks(
        &self,
        token_sequence: Vec<SequenceHash>,
    ) -> HashMap<WorkerWithDpRank, usize> {
        let mut results = HashMap::with_capacity(self.workers.len());
        for entry in self.workers.iter() {
            results.insert(
                *entry.key(),
                entry.value().potential_blocks(&token_sequence),
            );
        }
        results
    }

    /// Query all workers for the potential blocks and tokens
    pub fn potential_blocks_and_tokens(
        &self,
        token_sequence: Option<Vec<SequenceHash>>,
        isl: usize,
        overlaps: OverlapScores,
    ) -> (
        HashMap<WorkerWithDpRank, usize>,
        HashMap<WorkerWithDpRank, usize>,
    ) {
        #[cfg(feature = "bench")]
        let start = tokio::time::Instant::now();
        #[cfg(feature = "bench")]
        let num_workers = self.workers.len();

        let mut potential_blocks = HashMap::with_capacity(self.workers.len());
        let mut potential_tokens = HashMap::with_capacity(self.workers.len());

        for entry in self.workers.iter() {
            let worker = *entry.key();
            let overlap = *overlaps.scores.get(&worker).unwrap_or(&0);

            let (blocks, tokens) =
                entry
                    .value()
                    .potential_blocks_and_tokens(token_sequence.as_deref(), isl, overlap);
            potential_blocks.insert(worker, blocks);
            potential_tokens.insert(worker, tokens);
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

    /// Query all workers for their current number of active blocks
    pub fn active_blocks(&self) -> HashMap<WorkerWithDpRank, usize> {
        let mut results = HashMap::with_capacity(self.workers.len());
        for entry in self.workers.iter() {
            results.insert(*entry.key(), entry.value().active_blocks());
        }
        results
    }

    /// Query all workers for their current number of active tokens
    pub fn active_tokens(&self) -> HashMap<WorkerWithDpRank, usize> {
        let mut results = HashMap::with_capacity(self.workers.len());
        for entry in self.workers.iter() {
            results.insert(*entry.key(), entry.value().active_tokens());
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

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::{DistributedRuntime, Runtime};
    use std::sync::Arc;

    #[test]
    fn test_active_sequences_shared_blocks() {
        let block_size = 4;
        let mut seq_manager = ActiveSequences::new(block_size);

        seq_manager.add_request("request_1".to_string(), Some(vec![1, 2, 3]), 12, 0, None);
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.add_request("request_2".to_string(), Some(vec![4]), 4, 0, None);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 16);

        seq_manager.add_request("request_3".to_string(), Some(vec![1, 2, 3, 4]), 16, 4, None);
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 16);

        seq_manager.free(&"request_2".to_string());
        assert_eq!(seq_manager.active_blocks(), 4);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.free(&"request_3".to_string());
        assert_eq!(seq_manager.active_blocks(), 3);
        assert_eq!(seq_manager.active_tokens(), 12);

        seq_manager.free(&"request_1".to_string());
        assert_eq!(seq_manager.active_blocks(), 0);
        assert_eq!(seq_manager.active_tokens(), 0);
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_cross_instance_sync() -> Result<()> {
        // Initialize logging once
        dynamo_runtime::logging::init();

        let block_size = 4; // arbitrary block size

        // Create runtime and distributed runtime
        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Create namespace and shared component for both seq_managers
        let namespace = distributed.namespace("test_cross_instance_sync")?;
        let component = namespace.component("sequences")?;

        // Create multi-worker sequence managers with:
        // - Worker 0 with dp_size=2 (dp_ranks 0 and 1)
        // - Worker 1 with dp_size=1 (dp_rank 0)
        // This gives us 3 effective workers total to test dp_rank effect
        // Both seq_managers use the same component to ensure event synchronization works
        let mut workers_with_configs = HashMap::new();

        // Create runtime config for worker 0 with dp_size=2
        let mut config_worker_0 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        config_worker_0.data_parallel_size = 2;
        workers_with_configs.insert(0, config_worker_0);

        // Create runtime config for worker 1 with dp_size=1 (default)
        let config_worker_1 = crate::local_model::runtime_config::ModelRuntimeConfig::new();
        workers_with_configs.insert(1, config_worker_1);

        let seq_manager_1 = Arc::new(
            ActiveSequencesMultiWorker::new(
                component.clone(),
                block_size,
                workers_with_configs.clone(),
                true,
                1,
                crate::discovery::WORKER_TYPE_DECODE,
            )
            .await?,
        );
        let seq_manager_2 = Arc::new(
            ActiveSequencesMultiWorker::new(
                component,
                block_size,
                workers_with_configs,
                true,
                2,
                crate::discovery::WORKER_TYPE_DECODE,
            )
            .await?,
        );

        // Give some time for the subscription loops to start
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // PHASE 1: Add requests using both seq_manager_1 and seq_manager_2

        // Add request_0 to worker 0, dp_rank 0: sequence [0, 1, 2]
        seq_manager_1
            .add_request(SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: Some(vec![0, 1, 2]),
                isl: 12,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::new(0, 0),
                lora_name: None,
            })
            .await?;

        // Add request_1 to worker 0, dp_rank 1: sequence [3, 4]
        seq_manager_1
            .add_request(SequenceRequest {
                request_id: "request_1".to_string(),
                token_sequence: Some(vec![3, 4]),
                isl: 8,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::new(0, 1),
                lora_name: None,
            })
            .await?;

        // Add request_2 to worker 1, dp_rank 0: sequence [0, 1, 2, 3] using seq_manager_2
        seq_manager_2
            .add_request(SequenceRequest {
                request_id: "request_2".to_string(),
                token_sequence: Some(vec![0, 1, 2, 3]),
                isl: 16,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::new(1, 0),
                lora_name: None,
            })
            .await?;

        // Give some time for synchronization
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Query seq_manager_1 to verify it sees all requests including request_2 from seq_manager_2
        let blocks_phase1 = seq_manager_1.active_blocks();
        let tokens_phase1 = seq_manager_1.active_tokens();

        // Verify that seq_manager_1 sees all requests including request_2 from seq_manager_2
        // We now have:
        // - Worker 0, dp_rank 0: request_0
        // - Worker 0, dp_rank 1: request_1
        // - Worker 1, dp_rank 0: request_2
        let worker_0_dp0 = WorkerWithDpRank::new(0, 0);
        let worker_0_dp1 = WorkerWithDpRank::new(0, 1);
        let worker_1_dp0 = WorkerWithDpRank::new(1, 0);

        assert_eq!(
            blocks_phase1[&worker_0_dp0], 3,
            "Worker 0 dp_rank 0 should have 3 active blocks (from request_0)"
        );
        assert_eq!(
            blocks_phase1[&worker_0_dp1], 2,
            "Worker 0 dp_rank 1 should have 2 active blocks (from request_1)"
        );
        assert_eq!(
            blocks_phase1[&worker_1_dp0], 4,
            "Worker 1 dp_rank 0 should have 4 active blocks (from request_2 added by seq_manager_2)"
        );
        assert_eq!(
            tokens_phase1[&worker_0_dp0], 12,
            "Worker 0 dp_rank 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_0_dp1], 8,
            "Worker 0 dp_rank 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_1_dp0], 16,
            "Worker 1 dp_rank 0 should have 16 active tokens (from request_2 added by seq_manager_2)"
        );

        // PHASE 2: Free requests using opposite sequence managers, verify on seq_manager_2

        // Free request_2 (which was added by seq_manager_2) using seq_manager_1
        seq_manager_1.free(&"request_2".to_string()).await?;

        // Free request_0 and request_1 (which were added by seq_manager_1) using seq_manager_2
        seq_manager_2.free(&"request_0".to_string()).await?;
        seq_manager_2.free(&"request_1".to_string()).await?;

        // Give some time for synchronization
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Query seq_manager_2 to verify everything is empty
        let blocks_phase2 = seq_manager_2.active_blocks();
        let tokens_phase2 = seq_manager_2.active_tokens();

        // Verify phase 2 results - everything should be empty for all 3 workers
        let all_workers = vec![
            WorkerWithDpRank::new(0, 0),
            WorkerWithDpRank::new(0, 1),
            WorkerWithDpRank::new(1, 0),
        ];

        for worker in all_workers {
            assert_eq!(
                blocks_phase2[&worker], 0,
                "Worker (id={}, dp_rank={}) should have 0 active blocks after all requests freed",
                worker.worker_id, worker.dp_rank
            );
            assert_eq!(
                tokens_phase2[&worker], 0,
                "Worker (id={}, dp_rank={}) should have 0 active tokens after all requests freed",
                worker.worker_id, worker.dp_rank
            );
        }

        Ok(())
    }

    #[tokio::test]
    #[ignore]
    async fn test_multi_worker_no_token_sequence_sync() -> Result<()> {
        // Initialize logging once
        dynamo_runtime::logging::init();

        let block_size = 4; // arbitrary block size

        // Create runtime and distributed runtime
        let runtime = Runtime::from_current()?;
        let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;

        // Create namespace and shared component for both seq_managers
        let namespace = distributed.namespace("test_no_token_seq_sync")?;
        let component = namespace.component("sequences")?;

        // Create multi-worker sequence managers with ALL workers [0, 1, 2]
        // Both use the same component to ensure event synchronization works
        let mut workers_with_configs = HashMap::new();
        workers_with_configs.insert(
            0,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );
        workers_with_configs.insert(
            1,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );
        workers_with_configs.insert(
            2,
            crate::local_model::runtime_config::ModelRuntimeConfig::new(),
        );

        let seq_manager_1 = Arc::new(
            ActiveSequencesMultiWorker::new(
                component.clone(),
                block_size,
                workers_with_configs.clone(),
                true,
                1,
                crate::discovery::WORKER_TYPE_DECODE,
            )
            .await?,
        );
        let seq_manager_2 = Arc::new(
            ActiveSequencesMultiWorker::new(
                component,
                block_size,
                workers_with_configs,
                true,
                2,
                crate::discovery::WORKER_TYPE_DECODE,
            )
            .await?,
        );

        // Give some time for the subscription loops to start
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;

        // PHASE 1: Add requests (without token sequences) using both seq_managers

        // Add request_0 to worker 0 with no token sequence
        seq_manager_1
            .add_request(SequenceRequest {
                request_id: "request_0".to_string(),
                token_sequence: None,
                isl: 12,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::from_worker_id(0),
                lora_name: None,
            })
            .await?;

        // Add request_1 to worker 1 with no token sequence
        seq_manager_1
            .add_request(SequenceRequest {
                request_id: "request_1".to_string(),
                token_sequence: None,
                isl: 8,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::from_worker_id(1),
                lora_name: None,
            })
            .await?;

        // Add request_2 to worker 2 with no token sequence using seq_manager_2
        seq_manager_2
            .add_request(SequenceRequest {
                request_id: "request_2".to_string(),
                token_sequence: None,
                isl: 16,
                overlap: 0,
                expected_output_tokens: None,
                worker: WorkerWithDpRank::from_worker_id(2),
                lora_name: None,
            })
            .await?;

        // Give some time for synchronization
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Query seq_manager_1 to verify it sees all requests including request_2 from seq_manager_2
        let tokens_phase1 = seq_manager_1.active_tokens();

        // Verify that seq_manager_1 sees all requests including request_2 from thread 2
        let worker_0 = WorkerWithDpRank::from_worker_id(0);
        let worker_1 = WorkerWithDpRank::from_worker_id(1);
        let worker_2 = WorkerWithDpRank::from_worker_id(2);

        assert_eq!(
            tokens_phase1[&worker_0], 12,
            "Worker 0 should have 12 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_1], 8,
            "Worker 1 should have 8 active tokens"
        );
        assert_eq!(
            tokens_phase1[&worker_2], 16,
            "Worker 2 should have 16 active tokens (from request_2 added by seq_manager_2)"
        );

        // PHASE 2: Free requests using opposite sequence managers, verify on seq_manager_2

        // Mark prefill completed and free request_2 (which was added by seq_manager_2) using seq_manager_1
        seq_manager_1
            .mark_prefill_completed(&"request_2".to_string())
            .await?;
        seq_manager_1.free(&"request_2".to_string()).await?;

        // Mark prefill completed and free requests 0 and 1 (which were added by seq_manager_1) using seq_manager_2
        seq_manager_2
            .mark_prefill_completed(&"request_0".to_string())
            .await?;
        seq_manager_2
            .mark_prefill_completed(&"request_1".to_string())
            .await?;
        seq_manager_2.free(&"request_0".to_string()).await?;
        seq_manager_2.free(&"request_1".to_string()).await?;

        // Give some time for synchronization
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        // Query seq_manager_2 to verify everything is empty
        let tokens_phase2 = seq_manager_2.active_tokens();

        // Verify phase 2 results - everything should be empty
        for worker_id in 0..=2 {
            let worker = WorkerWithDpRank::from_worker_id(worker_id);
            assert_eq!(
                tokens_phase2[&worker], 0,
                "Worker {} should have 0 active tokens after all requests freed",
                worker_id
            );
        }

        Ok(())
    }
}
