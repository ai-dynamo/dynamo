// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kv_router::protocols::{LocalBlockHash, SharedCacheHits};
use dynamo_kv_router::scheduling::PolicyClassAdmissionPolicies;
pub use dynamo_kv_router::scheduling::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap,
};
pub use dynamo_kv_router::scheduling::{
    KvSchedulerError, LocalScheduler, OverloadedWorkerProvider, PotentialLoad, ScheduleRequest,
    SchedulingRequest, SchedulingResponse, TierOverlapBlocks,
};
pub use dynamo_kv_router::selector::DefaultWorkerSelector;
use dynamo_kv_router::selector::WorkerSelector as WorkerSelectorTrait;

use super::metrics::{ROUTER_QUEUE_METRICS, RouterQueueMetricHandles};
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
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_tokens::SequenceHash;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

pub struct KvScheduler<Sel = DefaultWorkerSelector, RF = NoopOverlapScoresRefresh>
where
    Sel: WorkerSelectorTrait<ModelRuntimeConfig>,
    RF: OverlapScoresRefresh,
{
    inner: Arc<LocalScheduler<RuntimeSequencePublisher, ModelRuntimeConfig, Sel, RF>>,
    queue_metrics: Vec<RouterQueueMetricHandles>,
    queue_metric_indices: HashMap<String, usize>,
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
        endpoint: Endpoint,
        block_size: u32,
        workers_with_configs: RuntimeConfigWatch,
        selector: Sel,
        kv_router_config: &KvRouterConfig,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        model_name: Option<&str>,
        worker_type: &'static str,
        cancellation_token: CancellationToken,
        admission_policies: PolicyClassAdmissionPolicies,
    ) -> Result<Self, KvSchedulerError> {
        let initial_workers: HashMap<WorkerId, ModelRuntimeConfig> =
            workers_with_configs.borrow().clone();

        let router_id = endpoint.drt().discovery().instance_id();
        let slots = create_multi_worker_sequences(
            endpoint,
            block_size as usize,
            initial_workers,
            kv_router_config.router_replica_sync,
            router_id,
            worker_type,
            cancellation_token.child_token(),
        )
        .await
        .map_err(|e| KvSchedulerError::InitFailed(e.to_string()))?;

        // `skip_initial_worker_wait` only controls boot-time blocking. The scheduler must
        // keep monitoring runtime config changes so late-joining workers enter routing.
        let watch_worker_configs = true;

        let profile = kv_router_config
            .policy_profile(model_name)
            .map_err(|error| KvSchedulerError::InitFailed(error.to_string()))?;
        let queue_recheck_interval = kv_router_config.router_queue_recheck_interval();
        let metric_model = model_name.unwrap_or("unknown");
        let queue_metrics = profile
            .classes()
            .iter()
            .map(|class| ROUTER_QUEUE_METRICS.handles(metric_model, worker_type, &class.name))
            .collect::<Vec<_>>();
        let queue_metric_indices = profile
            .classes()
            .iter()
            .enumerate()
            .map(|(index, class)| (class.name.clone(), index))
            .collect();

        let inner = Arc::new(LocalScheduler::new_with_policy_profile(
            slots,
            workers_with_configs.clone(),
            profile,
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overloaded_worker_provider,
            queue_recheck_interval,
            kv_router_config.router_track_prefill_tokens,
            cancellation_token.child_token(),
            worker_type,
            watch_worker_configs,
            admission_policies,
        )?);

        let metrics_scheduler = Arc::clone(&inner);
        let background_metrics = queue_metrics.clone();
        let metrics_cancel_token = cancellation_token.child_token();
        let mut queue_updates = inner.subscribe_queue_updates();
        tokio::spawn(async move {
            let mut recheck_interval = tokio::time::interval(Duration::from_secs(60));
            update_queue_metrics(&background_metrics, |class_index| {
                metrics_scheduler.class_queue_stats(class_index)
            });

            loop {
                tokio::select! {
                    _ = metrics_cancel_token.cancelled() => break,
                    result = queue_updates.changed() => {
                        if result.is_err() {
                            break;
                        }
                        update_queue_metrics(&background_metrics, |class_index| {
                            metrics_scheduler.class_queue_stats(class_index)
                        });
                    }
                    _ = recheck_interval.tick() => {
                        update_queue_metrics(&background_metrics, |class_index| {
                            metrics_scheduler.class_queue_stats(class_index)
                        });
                    }
                }
            }
        });

        Ok(Self {
            inner,
            queue_metrics,
            queue_metric_indices,
        })
    }

    pub async fn schedule_request(
        &self,
        request: ScheduleRequest,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        let response = self.inner.schedule_request(request).await;
        self.observe_schedule_result(&response);
        response
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
        strict_priority: u32,
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
            strict_priority,
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
        strict_priority: u32,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        shared_cache_hits: Option<SharedCacheHits>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        self.schedule_with_policy_class_and_block_hashes(
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
            strict_priority,
            None,
            expected_output_tokens,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            shared_cache_hits,
        )
        .await
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn schedule_with_policy_class_and_block_hashes(
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
        strict_priority: u32,
        policy_class: Option<String>,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        shared_cache_hits: Option<SharedCacheHits>,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        let response = self
            .inner
            .schedule_with_policy_class_and_block_hashes(
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
                strict_priority,
                policy_class,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
                shared_cache_hits,
            )
            .await;
        self.observe_schedule_result(&response);
        response
    }

    fn observe_schedule_result(&self, response: &Result<SchedulingResponse, KvSchedulerError>) {
        if let Err(KvSchedulerError::QueueRejected(rejection)) = response
            && let Some(metrics) = self
                .queue_metric_indices
                .get(&rejection.policy_class)
                .and_then(|index| self.queue_metrics.get(*index))
        {
            match rejection.limit_kind {
                dynamo_kv_router::scheduling::QueueLimitKind::Requests => {
                    metrics.request_limit_rejections.inc();
                }
                dynamo_kv_router::scheduling::QueueLimitKind::RawIslTokens => {
                    metrics.raw_isl_limit_rejections.inc();
                }
                dynamo_kv_router::scheduling::QueueLimitKind::CachedTokens => {
                    metrics.cached_token_limit_rejections.inc();
                }
            }
        }
        self.update_queue_metrics();
    }

    /// Select a worker from current scheduler state without queue admission or booking.
    pub async fn select_without_admission(
        &self,
        request: ScheduleRequest,
    ) -> Result<SchedulingResponse, KvSchedulerError> {
        self.inner.select_without_admission(request).await
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        self.inner.register_workers(worker_ids);
    }

    pub async fn add_request(&self, req: SequenceRequest) -> Result<(), SequenceError> {
        self.inner.add_request(req).await
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.inner.mark_prefill_completed(request_id).await?;
        self.update_queue_metrics();
        Ok(())
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.inner.free(request_id).await?;
        self.update_queue_metrics();
        Ok(())
    }

    pub fn pending_count(&self) -> usize {
        self.inner.pending_count()
    }

    pub fn pending_isl_tokens(&self) -> usize {
        self.inner.pending_isl_tokens()
    }

    pub fn worker_is_prefill_busy(
        &self,
        worker: WorkerWithDpRank,
        decay_now: tokio::time::Instant,
        threshold: f64,
    ) -> Option<bool> {
        self.inner
            .worker_is_prefill_busy(worker, decay_now, threshold)
    }

    pub fn projected_decode_load_exceeds(
        &self,
        worker: WorkerWithDpRank,
        projected_blocks: usize,
        threshold: f64,
    ) -> Option<bool> {
        self.inner
            .projected_decode_load_exceeds(worker, projected_blocks, threshold)
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

    fn update_queue_metrics(&self) {
        update_queue_metrics(&self.queue_metrics, |class_index| {
            self.inner.class_queue_stats(class_index)
        });
    }
}

fn update_queue_metrics(
    handles: &[RouterQueueMetricHandles],
    mut stats_for: impl FnMut(usize) -> Option<dynamo_kv_router::queue::ClassQueueStats>,
) {
    for (class_index, handles) in handles.iter().enumerate() {
        let Some(stats) = stats_for(class_index) else {
            debug_assert!(
                false,
                "missing queue counters for policy class {class_index}"
            );
            continue;
        };
        handles.pending_requests.set(stats.pending_count as i64);
        handles
            .pending_isl_tokens
            .set(stats.pending_isl_tokens as i64);
        handles
            .pending_cached_tokens
            .set(stats.pending_cached_tokens as i64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_runtime::component::Component;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio::sync::watch;

    async fn make_test_component(name: &str) -> Component {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace = drt.namespace(format!("test-ns-{name}")).unwrap();
        namespace
            .component(format!("test-component-{name}"))
            .unwrap()
    }

    #[test]
    fn queue_metrics_are_updated_by_class_index() {
        let handles = ["latency", "bulk"]
            .map(|class| ROUTER_QUEUE_METRICS.handles("index-test", "decode", class));
        let stats = [
            dynamo_kv_router::queue::ClassQueueStats {
                pending_count: 2,
                pending_isl_tokens: 128,
                pending_cached_tokens: 64,
            },
            dynamo_kv_router::queue::ClassQueueStats {
                pending_count: 3,
                pending_isl_tokens: 384,
                pending_cached_tokens: 192,
            },
        ];

        update_queue_metrics(&handles, |class_index| stats.get(class_index).copied());

        for (handles, stats) in handles.iter().zip(stats) {
            assert_eq!(handles.pending_requests.get(), stats.pending_count as i64);
            assert_eq!(
                handles.pending_isl_tokens.get(),
                stats.pending_isl_tokens as i64
            );
            assert_eq!(
                handles.pending_cached_tokens.get(),
                stats.pending_cached_tokens as i64
            );
        }
    }

    #[tokio::test]
    async fn skip_initial_worker_wait_still_monitors_worker_config_updates() {
        let component = make_test_component("skip-initial-worker-watch").await;
        let mut workers = HashMap::new();
        workers.insert(0, ModelRuntimeConfig::default());
        let (cfg_tx, cfg_rx) = watch::channel(workers);
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            ..Default::default()
        };
        let cancellation_token = CancellationToken::new();

        let scheduler = KvScheduler::start(
            component.endpoint("generate"),
            64,
            cfg_rx,
            DefaultWorkerSelector::new(Some(config.clone()), "decode"),
            &config,
            None,
            None::<Arc<NoopOverlapScoresRefresh>>,
            None,
            Some("test-model"),
            "decode",
            cancellation_token.clone(),
            Default::default(),
        )
        .await
        .unwrap();

        assert_eq!(
            scheduler
                .get_potential_loads(None, 64, HashMap::new(), true)
                .len(),
            1
        );

        let mut updated_workers = HashMap::new();
        updated_workers.insert(0, ModelRuntimeConfig::default());
        updated_workers.insert(1, ModelRuntimeConfig::default());
        cfg_tx.send(updated_workers).unwrap();

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if scheduler
                    .get_potential_loads(None, 64, HashMap::new(), true)
                    .iter()
                    .any(|load| load.worker_id == 1)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        cancellation_token.cancel();
    }
}
