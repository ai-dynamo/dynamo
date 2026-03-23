// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use dynamo_kv_router::scheduling::policy::RouterSchedulingPolicy;
pub use dynamo_kv_router::scheduling::{
    KvSchedulerError, LocalScheduler, PotentialLoad, SchedulingRequest, SchedulingResponse,
};
pub use dynamo_kv_router::selector::DefaultWorkerSelector;
use dynamo_kv_router::selector::WorkerSelector as WorkerSelectorTrait;

use super::metrics::ROUTER_QUEUE_METRICS;
use super::sequence::{
    RuntimeSequencePublisher, SequenceError, SequenceRequest, create_multi_worker_sequences,
};
use crate::discovery::RuntimeConfigWatch;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use anyhow::Result;
use dynamo_kv_router::{
    config::{KvRouterConfig, RouterConfigOverride},
    protocols::{OverlapScores, WorkerId},
};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

pub struct KvScheduler<Sel = DefaultWorkerSelector>
where
    Sel: WorkerSelectorTrait<ModelRuntimeConfig>,
{
    inner: Arc<
        LocalScheduler<RuntimeSequencePublisher, ModelRuntimeConfig, RouterSchedulingPolicy, Sel>,
    >,
}

impl<Sel> KvScheduler<Sel>
where
    Sel: WorkerSelectorTrait<ModelRuntimeConfig> + Send + Sync + 'static,
{
    pub async fn start(
        component: Component,
        block_size: u32,
        workers_with_configs: RuntimeConfigWatch,
        selector: Sel,
        kv_router_config: &KvRouterConfig,
        worker_type: &'static str,
    ) -> Result<Self, KvSchedulerError> {
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

        let watch_worker_configs = !kv_router_config.skip_initial_worker_wait;
        if !watch_worker_configs {
            tracing::info!("skipping discovery-based worker monitoring");
        }

        let policy =
            RouterSchedulingPolicy::new(kv_router_config.router_queue_policy, block_size as usize);
        tracing::info!(
            "Router queue policy: {}",
            kv_router_config.router_queue_policy
        );

        let inner = Arc::new(LocalScheduler::new(
            slots,
            workers_with_configs.clone(),
            kv_router_config.router_queue_threshold,
            block_size,
            selector,
            policy,
            component.drt().child_token(),
            worker_type,
            watch_worker_configs,
        ));

        let metrics_scheduler = Arc::clone(&inner);
        let metrics_cancel_token = component.drt().child_token();
        tokio::spawn(async move {
            let mut recheck_interval = tokio::time::interval(Duration::from_secs(60));
            ROUTER_QUEUE_METRICS.set_pending(worker_type, metrics_scheduler.pending_count());

            loop {
                tokio::select! {
                    _ = metrics_cancel_token.cancelled() => break,
                    _ = recheck_interval.tick() => {
                        ROUTER_QUEUE_METRICS
                            .set_pending(worker_type, metrics_scheduler.pending_count());
                    }
                }
            }
        });

        Ok(Self { inner })
    }

    #[expect(clippy::too_many_arguments)]
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
        expected_output_tokens: Option<u32>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        let response = self
            .inner
            .schedule(
                maybe_request_id,
                isl_tokens,
                token_seq,
                overlaps,
                router_config_override,
                update_states,
                lora_name,
                priority_jump,
                expected_output_tokens,
                allowed_worker_ids,
            )
            .await;
        ROUTER_QUEUE_METRICS.set_pending(self.worker_type(), self.pending_count());
        response
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        self.inner.register_workers(worker_ids);
    }

    pub async fn add_request(&self, req: SequenceRequest) -> Result<(), SequenceError> {
        self.inner.add_request(req).await
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.inner.mark_prefill_completed(request_id).await?;
        ROUTER_QUEUE_METRICS.set_pending(self.worker_type(), self.pending_count());
        Ok(())
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.inner.free(request_id).await?;
        ROUTER_QUEUE_METRICS.set_pending(self.worker_type(), self.pending_count());
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    pub fn worker_type(&self) -> &'static str {
        self.inner.worker_type()
    }

    pub fn add_output_block(
        &self,
        request_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        self.inner.add_output_block(request_id, decay_fraction)
    }

    pub fn get_potential_loads(
        &self,
        token_seq: Option<Vec<SequenceHash>>,
        isl_tokens: usize,
        overlaps: OverlapScores,
    ) -> Vec<PotentialLoad> {
        self.inner
            .get_potential_loads(token_seq, isl_tokens, overlaps)
    }

    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        self.inner.get_active_lora_counts()
    }
}
