// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use dynamo_kv_router::scheduling::{
    KvSchedulerError, PotentialLoad, SchedulingRequest, SchedulingResponse,
};
pub use dynamo_kv_router::selector::DefaultWorkerSelector;

use super::KvRouterConfig;
use super::RouterConfigOverride;
use super::WorkerSelector;
use super::protocols::{OverlapScores, WorkerId};
use super::queue::SchedulerQueue;
use super::sequence::{
    ActiveSequencesMulti, SequenceError, SequenceRequest, create_multi_worker_sequences,
};
use crate::discovery::RuntimeConfigWatch;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use anyhow::Result;
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
#[cfg(feature = "bench")]
use std::time::Instant;

use dynamo_tokens::SequenceHash;

pub struct KvScheduler {
    request_tx: tokio::sync::mpsc::Sender<SchedulingRequest>,
    slots: Arc<ActiveSequencesMulti>,
    queue: Arc<SchedulerQueue>,
}

impl KvScheduler {
    pub async fn start(
        component: Component,
        block_size: u32,
        workers_with_configs: RuntimeConfigWatch,
        selector: Option<Box<WorkerSelector>>,
        kv_router_config: &KvRouterConfig,
        worker_type: &'static str,
    ) -> Result<Self, KvSchedulerError> {
        let selector = selector.unwrap_or(Box::new(DefaultWorkerSelector::default()));

        // Get initial workers from watch receiver.
        // Caller must ensure at least one worker is present (via wait_for).
        let initial_workers: HashMap<WorkerId, ModelRuntimeConfig> =
            workers_with_configs.borrow().clone();

        let router_id = component.drt().discovery().instance_id();
        let slots = create_multi_worker_sequences(
            component.clone(),
            block_size as usize,
            initial_workers,
            kv_router_config.router_replica_sync,
            router_id,
            worker_type,
        )
        .await
        .map_err(|e| KvSchedulerError::InitFailed(e.to_string()))?;

        // Spawn background task to sync slots when the watch value changes.
        let slots_monitor = slots.clone();
        let mut monitor_rx = workers_with_configs.clone();
        let monitor_cancel_token = component.drt().child_token();
        tokio::spawn(async move {
            tracing::trace!("KvScheduler workers monitoring task started");
            let mut last_workers: HashMap<WorkerId, ModelRuntimeConfig> = HashMap::new();

            loop {
                tokio::select! {
                    _ = monitor_cancel_token.cancelled() => {
                        tracing::trace!("KvScheduler workers monitoring task shutting down");
                        break;
                    }
                    result = monitor_rx.changed() => {
                        if result.is_err() {
                            tracing::warn!("KvScheduler: config watch sender dropped, shutting down");
                            break;
                        }
                    }
                }

                let current_workers = monitor_rx.borrow_and_update().clone();

                if current_workers != last_workers {
                    let dp_sizes: HashMap<u64, u32> = current_workers
                        .iter()
                        .map(|(&id, c)| (id, c.data_parallel_size))
                        .collect();
                    slots_monitor.update_workers(&dp_sizes);
                    last_workers = current_workers;
                }
            }
        });

        let (request_tx, request_rx) = tokio::sync::mpsc::channel::<SchedulingRequest>(1024);
        let scheduler_cancel_token = component.drt().primary_token();

        let queue = Arc::new(SchedulerQueue::new(
            slots.clone(),
            workers_with_configs.clone(),
            kv_router_config.router_queue_threshold,
            block_size,
            selector,
        ));
        let queue_clone = queue.clone();

        // Background task: receive requests and periodically recheck pending
        tokio::spawn(async move {
            let mut request_rx = request_rx;
            let mut recheck_interval = tokio::time::interval(Duration::from_secs(60));
            tracing::trace!("scheduler background task started");

            loop {
                tokio::select! {
                    _ = scheduler_cancel_token.cancelled() => {
                        tracing::trace!("scheduler background task shutting down");
                        break;
                    }
                    request = request_rx.recv() => {
                        let Some(request) = request else {
                            tracing::warn!("scheduler shutdown");
                            break;
                        };
                        tracing::trace!("received request to be scheduled");
                        queue_clone.enqueue(request).await;
                    }
                    _ = recheck_interval.tick() => {
                        queue_clone.update().await;
                    }
                }
            }

            tracing::trace!("background endpoint subscriber shutting down");
        });

        Ok(KvScheduler {
            request_tx,
            slots,
            queue,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn schedule(
        &self,
        maybe_request_id: Option<String>,
        isl_tokens: usize,
        token_seq: Option<Vec<SequenceHash>>,
        overlaps: OverlapScores,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        #[cfg(feature = "bench")]
        let start = Instant::now();

        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        let request = SchedulingRequest {
            maybe_request_id,
            token_seq,
            isl_tokens,
            overlaps,
            decode_blocks: HashMap::new(),
            prefill_tokens: HashMap::new(),
            router_config_override: router_config_override.cloned(),
            update_states,
            lora_name,
            priority_jump,
            allowed_worker_ids,
            resp_tx: Some(resp_tx),
        };

        self.request_tx
            .send(request)
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?;

        #[cfg(feature = "bench")]
        let send_elapsed = start.elapsed();

        let response = resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)??;

        #[cfg(feature = "bench")]
        let total_elapsed = start.elapsed();
        #[cfg(feature = "bench")]
        tracing::info!(
            isl_tokens,
            send_us = send_elapsed.as_micros() as u64,
            total_us = total_elapsed.as_micros() as u64,
            "scheduler.schedule completed"
        );

        Ok(response)
    }

    pub async fn add_request(&self, req: SequenceRequest) -> Result<(), SequenceError> {
        self.slots.add_request(req).await
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.slots
            .mark_prefill_completed(&request_id.to_string())
            .await?;
        self.queue.update().await;
        Ok(())
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.slots.free(&request_id.to_string()).await?;
        self.queue.update().await;
        Ok(())
    }

    /// Get the worker type for this scheduler ("prefill" or "decode").
    /// Used for Prometheus metric labeling.
    pub fn worker_type(&self) -> &'static str {
        self.slots.worker_type()
    }

    pub fn add_output_block(
        &self,
        request_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        self.slots
            .add_output_block(&request_id.to_string(), decay_fraction)
    }

    pub fn get_potential_loads(
        &self,
        token_seq: Option<Vec<SequenceHash>>,
        isl_tokens: usize,
        overlaps: OverlapScores,
    ) -> Vec<PotentialLoad> {
        let (decode_blocks, prefill_tokens) =
            self.slots
                .potential_blocks_and_tokens(token_seq.as_deref(), isl_tokens, overlaps);

        // Get all unique WorkerWithDpRank from both hashmaps
        let mut workers: HashSet<dynamo_kv_router::protocols::WorkerWithDpRank> = HashSet::new();
        workers.extend(decode_blocks.keys().copied());
        workers.extend(prefill_tokens.keys().copied());

        // Create PotentialLoad for each worker
        let mut loads = Vec::new();
        for worker in workers {
            loads.push(PotentialLoad {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                potential_prefill_tokens: prefill_tokens
                    .get(&worker)
                    .copied()
                    .unwrap_or(isl_tokens),
                potential_decode_blocks: decode_blocks.get(&worker).copied().unwrap_or(0),
            });
        }

        loads
    }

    /// Get active request counts grouped by LORA name
    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        self.slots.get_active_lora_counts()
    }
}
