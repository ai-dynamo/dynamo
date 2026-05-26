// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::{LocalBlockHash, RouterBackpressureReason, SharedCacheHits};
pub use dynamo_kv_router::scheduling::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap,
};
pub use dynamo_kv_router::scheduling::policy::RouterSchedulingPolicy;
pub use dynamo_kv_router::scheduling::{
    KvSchedulerError, LocalScheduler, OverloadedWorkerProvider, PotentialLoad, SchedulingRequest,
    SchedulingResponse, TierOverlapBlocks,
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
    PrefillLoadEstimator,
    config::{KvRouterConfig, RouterConfigOverride},
    protocols::{RoutingConstraints, WorkerId, WorkerWithDpRank},
};
use dynamo_runtime::component::Component;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

pub struct KvScheduler<Sel = DefaultWorkerSelector, RF = NoopOverlapScoresRefresh>
where
    Sel: WorkerSelectorTrait<ModelRuntimeConfig>,
    RF: OverlapScoresRefresh,
{
    inner: Arc<
        LocalScheduler<
            RuntimeSequencePublisher,
            ModelRuntimeConfig,
            RouterSchedulingPolicy,
            Sel,
            RF,
        >,
    >,
}

impl<Sel, RF> KvScheduler<Sel, RF>
where
    Sel: WorkerSelectorTrait<ModelRuntimeConfig> + Send + Sync + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
{
    /// Start the scheduler, optionally wiring an [`OverlapScoresRefresh`] into the queue so
    /// long-waiting requests can be re-scored at dequeue time.
    #[expect(clippy::too_many_arguments)]
    pub async fn start(
        component: Component,
        block_size: u32,
        workers_with_configs: RuntimeConfigWatch,
        selector: Sel,
        kv_router_config: &KvRouterConfig,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
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

        let policy = RouterSchedulingPolicy::new(kv_router_config.router_queue_policy);
        tracing::info!(
            "Router queue policy: {}",
            kv_router_config.router_queue_policy
        );

        let inner = Arc::new(LocalScheduler::new_with_overlap_refresh(
            slots,
            workers_with_configs.clone(),
            kv_router_config.router_queue_threshold,
            kv_router_config
                .router_queue_by_incoming_missing_isl
                .clone(),
            block_size,
            selector,
            policy,
            prefill_load_estimator,
            overlap_scores_refresh,
            overloaded_worker_provider,
            kv_router_config.router_queue_recheck_interval(),
            kv_router_config.router_track_prefill_tokens,
            component.drt().child_token(),
            worker_type,
            watch_worker_configs,
        ));

        let metrics_scheduler = Arc::clone(&inner);
        let metrics_cancel_token = component.drt().child_token();
        let mut queue_updates = inner.subscribe_queue_updates();
        tokio::spawn(async move {
            let mut recheck_interval = tokio::time::interval(Duration::from_secs(60));
            ROUTER_QUEUE_METRICS.set_pending(worker_type, metrics_scheduler.pending_count());
            ROUTER_QUEUE_METRICS
                .set_pending_isl_tokens(worker_type, metrics_scheduler.pending_isl_tokens());

            loop {
                tokio::select! {
                    _ = metrics_cancel_token.cancelled() => break,
                    result = queue_updates.changed() => {
                        if result.is_err() {
                            break;
                        }
                        ROUTER_QUEUE_METRICS
                            .set_pending(worker_type, metrics_scheduler.pending_count());
                        ROUTER_QUEUE_METRICS.set_pending_isl_tokens(
                            worker_type,
                            metrics_scheduler.pending_isl_tokens(),
                        );
                    }
                    _ = recheck_interval.tick() => {
                        ROUTER_QUEUE_METRICS.set_pending(worker_type, metrics_scheduler.pending_count());
                        ROUTER_QUEUE_METRICS.set_pending_isl_tokens(
                            worker_type,
                            metrics_scheduler.pending_isl_tokens(),
                        );
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
        tier_overlap_blocks: TierOverlapBlocks,
        effective_overlap_blocks: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, f64>,
        effective_cached_tokens: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, usize>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        shared_cache_hits: Option<SharedCacheHits>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        self.schedule_with_block_hashes(
            maybe_request_id,
            isl_tokens,
            token_seq,
            None,
            tier_overlap_blocks,
            effective_overlap_blocks,
            effective_cached_tokens,
            router_config_override,
            update_states,
            lora_name,
            priority_jump,
            expected_output_tokens,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            shared_cache_hits,
        )
        .await
    }

    /// Like [`schedule`](Self::schedule) but forwards the block hashes used to compute the
    /// initial overlap scores. Required to enable dequeue-time overlap refresh; ignored if
    /// the scheduler was not constructed with an [`OverlapScoresRefresh`].
    #[expect(clippy::too_many_arguments)]
    pub async fn schedule_with_block_hashes(
        &self,
        maybe_request_id: Option<String>,
        isl_tokens: usize,
        token_seq: Option<Vec<SequenceHash>>,
        block_hashes: Option<Vec<LocalBlockHash>>,
        tier_overlap_blocks: TierOverlapBlocks,
        effective_overlap_blocks: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, f64>,
        effective_cached_tokens: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, usize>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        shared_cache_hits: Option<SharedCacheHits>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        let response = self
            .inner
            .schedule_with_block_hashes(
                maybe_request_id,
                isl_tokens,
                token_seq,
                block_hashes,
                tier_overlap_blocks,
                effective_overlap_blocks,
                effective_cached_tokens,
                router_config_override,
                update_states,
                lora_name,
                priority_jump,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
                shared_cache_hits,
            )
            .await;
        if let Err(KvSchedulerError::Backpressure { reason, .. }) = &response {
            ROUTER_QUEUE_METRICS
                .inc_backpressure(self.worker_type(), router_backpressure_reason_label(reason));
        }
        ROUTER_QUEUE_METRICS.set_pending(self.worker_type(), self.pending_count());
        ROUTER_QUEUE_METRICS.set_pending_isl_tokens(self.worker_type(), self.pending_isl_tokens());
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
        ROUTER_QUEUE_METRICS.set_pending_isl_tokens(self.worker_type(), self.pending_isl_tokens());
        Ok(())
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.inner.free(request_id).await?;
        ROUTER_QUEUE_METRICS.set_pending(self.worker_type(), self.pending_count());
        ROUTER_QUEUE_METRICS.set_pending_isl_tokens(self.worker_type(), self.pending_isl_tokens());
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    pub fn pending_isl_tokens(&self) -> usize {
        self.inner.pending_isl_tokens()
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
        effective_cached_tokens: HashMap<dynamo_kv_router::protocols::WorkerWithDpRank, usize>,
        track_prefill_tokens: bool,
    ) -> Vec<PotentialLoad> {
        self.inner.get_potential_loads(
            token_seq,
            isl_tokens,
            effective_cached_tokens,
            track_prefill_tokens,
        )
    }

    pub fn get_active_lora_counts(&self) -> HashMap<String, usize> {
        self.inner.get_active_lora_counts()
    }

    pub fn supports_overlap_refresh(&self) -> bool {
        self.inner.supports_overlap_refresh()
    }
}

fn router_backpressure_reason_label(reason: &RouterBackpressureReason) -> &'static str {
    match reason {
        RouterBackpressureReason::MaxQueuedIslTokensExceeded => "max_queued_isl_tokens_exceeded",
    }
}
