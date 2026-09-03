// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded selection backend for the frontend `KvRouter`.
//!
//! When `KvRouterConfig::router_embedded_selection` is set, the router runs its
//! scheduling and active-sequence accounting on one partition of an in-process
//! `SelectionService` (the same core the standalone selection service and the
//! EPP use) instead of the runtime-bound `KvScheduler`. The router keeps its own
//! KV indexer and request transport: it computes overlap against that indexer
//! and hands the partition scheduler a fully formed `ScheduleRequest`, exactly
//! as it does with `KvScheduler`. The partition's worker catalog is fed from the
//! runtime-config watch, so discovery remains the source of truth for
//! membership while the scheduler and its replica sync become runtime-free.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::indexer::TieredMatchProvider;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::scheduling::queue::{SchedulerBookingCleanup, SchedulerBookingDescriptor};
use dynamo_kv_router::scheduling::selector::WorkerSelectionPolicy;
use dynamo_kv_router::scheduling::{
    AdmittedSchedulingResponse, AdvisorySchedulingResponse, AttemptId, KvSchedulerError,
    OverloadedWorkerProvider, PotentialLoad, ScheduleRequest, WorkerAvailabilityProvider,
};
use dynamo_kv_router::sequences::{SequenceError, SequenceRequest};
use dynamo_kv_router::services::selection::{
    OverlapRefreshSource, SelectionHostHooks, SelectionPartition, SelectionService,
    SelectionServiceBuilder, WorkerRequest, WorkerSelectionPolicyRegistry,
};
use dynamo_kv_router::{DEFAULT_ROUTING_GROUP, PrefillLoadEstimator, WorkerSelectionPolicyFactory};
use dynamo_tokens::SequenceHash;
use tokio_util::sync::CancellationToken;

use crate::discovery::RuntimeConfigWatch;
use crate::local_model::runtime_config::ModelRuntimeConfig;

/// Inputs the embedded backend needs from the router at construction.
pub(crate) struct EmbeddedSelectionArgs {
    pub kv_router_config: KvRouterConfig,
    pub worker_role: Option<WorkerType>,
    pub metric_worker_type: &'static str,
    pub model_name: Option<String>,
    pub block_size: u32,
    pub is_eagle: bool,
    pub prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    pub overloaded_worker_provider: OverloadedWorkerProvider,
    pub available_worker_provider: WorkerAvailabilityProvider,
    /// The router's own indexer, used for dequeue-time overlap refresh so
    /// queued requests are re-scored against the index that scored them.
    pub overlap_refresh: Option<Arc<dyn TieredMatchProvider>>,
    /// Scheduler-owned load snapshots for the worker monitor's overload
    /// detection, the same feed the runtime scheduler publishes.
    pub scheduler_load: crate::kv_router::routing_load::SchedulerLoadSender,
    /// Endpoint whose event plane carries replica sync when
    /// `router_replica_sync` is set.
    pub endpoint: dynamo_runtime::component::Endpoint,
    /// This router replica's id (ignored on receipt of its own events).
    pub router_id: u64,
}

/// One `SelectionService` partition driven directly by the router.
pub(crate) struct EmbeddedSelection {
    /// Keeps the service (listeners, sweep, replica sync) alive for as long as
    /// the router holds the partition.
    _service: Arc<SelectionService>,
    partition: SelectionPartition,
    worker_type: &'static str,
}

static INSTALLED_POLICY_REGISTRY: std::sync::OnceLock<WorkerSelectionPolicyRegistry> =
    std::sync::OnceLock::new();

/// Install the process-wide worker-selection policy registry (linked custom
/// policies) that embedded selection partitions resolve `KvRouterConfig`
/// policy instances against. Returns `false` if one is already installed.
pub fn install_worker_selection_policy_registry(registry: WorkerSelectionPolicyRegistry) -> bool {
    INSTALLED_POLICY_REGISTRY.set(registry).is_ok()
}

/// The installed registry, or the built-in default.
pub fn worker_selection_policy_registry() -> WorkerSelectionPolicyRegistry {
    INSTALLED_POLICY_REGISTRY.get().cloned().unwrap_or_default()
}

/// Bridges the partition's scheduler load snapshots to the router's
/// `SchedulerLoadSender`, which feeds `KvWorkerMonitor`.
struct SenderLoadSink(crate::kv_router::routing_load::SchedulerLoadSender);

impl dynamo_kv_router::services::selection::SchedulerLoadSink for SenderLoadSink {
    fn publish(&self, snapshot: dynamo_kv_router::sequences::SchedulerLoadSnapshot) {
        self.0.publish(snapshot);
    }

    fn publish_batch(&self, snapshots: Vec<dynamo_kv_router::sequences::SchedulerLoadSnapshot>) {
        self.0.publish_batch(snapshots);
    }
}

impl EmbeddedSelection {
    pub(crate) async fn start(
        args: EmbeddedSelectionArgs,
        workers_with_configs: RuntimeConfigWatch,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        let worker_type = args.worker_role.unwrap_or(WorkerType::Aggregated);
        let key = RoutingPartitionId::new(
            args.model_name
                .clone()
                .unwrap_or_else(|| "default".to_string()),
            DEFAULT_ROUTING_GROUP,
        );

        // Same selector the runtime path builds: the registry-resolved policy
        // when router configuration names one, else Dynamo's default scorer
        // and picker under the router's metric worker label.
        let label = args.metric_worker_type;
        let policy_factory: WorkerSelectionPolicyFactory = match worker_selection_policy_registry()
            .resolve_for_worker_type(&args.kv_router_config, worker_type)?
        {
            Some(factory) => factory,
            None => Arc::new(move |config: &KvRouterConfig, _worker_type, _partition| {
                WorkerSelectionPolicy::default(config.clone(), label)
            }),
        };

        // Replica sync rides the runtime event plane, as the runtime scheduler's
        // did; the partition publishes and consumes through host channels.
        let replica_sync: Option<dynamo_kv_router::services::selection::HostReplicaSyncFactory> =
            if args.kv_router_config.router_replica_sync {
                let channels = crate::kv_router::sequence::host_replica_channels(
                    &args.endpoint,
                    args.router_id,
                    cancellation_token.child_token(),
                )
                .await
                .context("start replica sync for the embedded selection partition")?;
                let slot = std::sync::Mutex::new(Some(channels));
                Some(Arc::new(move |_partition| {
                    slot.lock().ok().and_then(|mut s| s.take())
                }))
            } else {
                None
            };

        let service = SelectionServiceBuilder::new(
            args.kv_router_config.clone(),
            worker_type,
            worker_selection_policy_registry(),
        )
        .worker_selection_policy_factory(policy_factory)
        .indexer_threads(1)
        .external_kv_events()
        .host_hooks(SelectionHostHooks {
            prefill_load_estimator: args.prefill_load_estimator,
            overloaded_worker_provider: Some(args.overloaded_worker_provider),
            available_worker_provider: Some(args.available_worker_provider),
            shared_cache: None,
            lora_worker_filter: None,
            overlap_refresh: match args.overlap_refresh {
                Some(provider) => OverlapRefreshSource::External(provider),
                None => OverlapRefreshSource::Disabled,
            },
            scheduler_load_sink: Some(Arc::new(SenderLoadSink(args.scheduler_load))),
            replica_sync,
        })
        .build()
        .await
        .context("failed to start embedded selection service")?;
        let service = Arc::new(service);
        let partition = service
            .ensure_partition(key.clone(), args.block_size, args.is_eagle)
            .context("failed to create embedded selection partition")?;

        let feeder = CatalogFeeder {
            service: Arc::clone(&service),
            key,
            block_size: args.block_size,
            is_eagle: args.is_eagle,
            known: HashMap::new(),
        };
        feeder
            .run(workers_with_configs, cancellation_token.child_token())
            .await;

        tracing::info!(
            worker_type = %worker_type,
            "KvRouter scheduling on embedded selection partition"
        );
        Ok(Self {
            _service: service,
            partition,
            worker_type: args.metric_worker_type,
        })
    }

    /// Membership is catalog-driven (runtime-config watch); explicit worker
    /// registration is a no-op here.
    pub(crate) fn register_workers(&self, _worker_ids: &HashSet<WorkerId>) {}

    pub(crate) async fn schedule_request_admitted(
        &self,
        request: ScheduleRequest,
    ) -> Result<AdmittedSchedulingResponse, KvSchedulerError> {
        self.partition
            .scheduler()
            .schedule_request_admitted(request)
            .await
    }

    pub(crate) async fn select_without_admission(
        &self,
        request: ScheduleRequest,
    ) -> Result<AdvisorySchedulingResponse, KvSchedulerError> {
        self.partition
            .scheduler()
            .select_without_admission(request)
            .await
    }

    pub(crate) async fn add_request_admitted(
        &self,
        req: SequenceRequest,
    ) -> Result<AttemptId, SequenceError> {
        self.partition.scheduler().add_request_admitted(req).await
    }

    pub(crate) async fn mark_prefill_completed(
        &self,
        request_id: &str,
    ) -> Result<(), SequenceError> {
        self.partition
            .scheduler()
            .mark_prefill_completed(request_id)
            .await
    }

    pub(crate) async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.partition.scheduler().free(request_id).await
    }

    pub(crate) async fn free_if_worker(
        &self,
        request_id: &str,
        worker: WorkerWithDpRank,
    ) -> Result<(), SequenceError> {
        self.partition
            .scheduler()
            .free_if_worker(request_id, worker)
            .await
    }

    pub(crate) fn booking_cleanup(&self) -> SchedulerBookingCleanup {
        self.partition.scheduler().booking_cleanup()
    }

    pub(crate) async fn mark_prefill_completed_if_booking(
        &self,
        booking: &SchedulerBookingDescriptor,
    ) -> Result<(), KvSchedulerError> {
        self.partition
            .scheduler()
            .mark_prefill_completed_if_booking(booking)
            .await
    }

    pub(crate) fn pending_count(&self) -> usize {
        self.partition.scheduler().pending_count()
    }

    pub(crate) fn pending_isl_tokens(&self) -> usize {
        self.partition.scheduler().pending_isl_tokens()
    }

    pub(crate) fn worker_type(&self) -> &'static str {
        self.worker_type
    }

    pub(crate) fn add_output_block(
        &self,
        request_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        self.partition
            .scheduler()
            .add_output_block(request_id, decay_fraction)
    }

    pub(crate) async fn enqueue_output_block_if_booking(
        &self,
        booking: &SchedulerBookingDescriptor,
        decay_fraction: Option<f64>,
    ) -> Result<(), KvSchedulerError> {
        self.partition
            .scheduler()
            .enqueue_output_block_if_booking(booking, decay_fraction)
            .await
    }

    pub(crate) fn get_potential_loads(
        &self,
        token_seq: Option<Vec<SequenceHash>>,
        isl_tokens: usize,
        effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
        track_prefill_tokens: bool,
    ) -> Vec<PotentialLoad> {
        self.partition.scheduler().get_potential_loads(
            token_seq,
            isl_tokens,
            effective_cached_tokens,
            track_prefill_tokens,
        )
    }

    pub(crate) fn supports_overlap_refresh(&self) -> bool {
        self.partition.scheduler().supports_overlap_refresh()
    }
}

/// Mirrors the runtime-config watch into the partition's worker catalog.
struct CatalogFeeder {
    service: Arc<SelectionService>,
    key: RoutingPartitionId,
    block_size: u32,
    is_eagle: bool,
    known: HashMap<WorkerId, ModelRuntimeConfig>,
}

impl CatalogFeeder {
    /// Apply the current snapshot, then keep applying changes until cancelled.
    async fn run(mut self, mut watch: RuntimeConfigWatch, cancel: CancellationToken) {
        let snapshot = watch.borrow_and_update().clone();
        self.apply(snapshot).await;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    changed = watch.changed() => {
                        if changed.is_err() {
                            break;
                        }
                    }
                }
                let snapshot = watch.borrow_and_update().clone();
                self.apply(snapshot).await;
            }
        });
    }

    async fn apply(&mut self, snapshot: HashMap<WorkerId, ModelRuntimeConfig>) {
        let removed: Vec<WorkerId> = self
            .known
            .keys()
            .copied()
            .filter(|worker_id| !snapshot.contains_key(worker_id))
            .collect();
        for worker_id in removed {
            self.known.remove(&worker_id);
            if let Err(error) = self.service.delete_worker(worker_id).await {
                tracing::debug!(worker_id, %error, "embedded selection: worker delete skipped");
            }
        }
        for (worker_id, config) in snapshot {
            if self.known.get(&worker_id) == Some(&config) {
                continue;
            }
            let request = worker_request_from_runtime_config(
                worker_id,
                &config,
                &self.key,
                self.block_size,
                self.is_eagle,
            );
            match self.service.upsert_worker(request).await {
                Ok(record) => {
                    if !record.not_schedulable_reasons.is_empty() {
                        tracing::warn!(
                            worker_id,
                            reasons = ?record.not_schedulable_reasons,
                            "embedded selection: worker is not schedulable"
                        );
                    }
                    self.known.insert(worker_id, config);
                }
                Err(error) => {
                    tracing::warn!(worker_id, %error, "embedded selection: worker upsert failed");
                }
            }
        }
    }
}

/// Build the catalog record for a discovered worker. The endpoint is a
/// placeholder: dispatch stays on the router's request transport, keyed by
/// worker id.
pub(crate) fn worker_request_from_runtime_config(
    worker_id: WorkerId,
    config: &ModelRuntimeConfig,
    key: &RoutingPartitionId,
    block_size: u32,
    is_eagle: bool,
) -> WorkerRequest {
    let dp_start = config.data_parallel_start_rank();
    let dp_size = config.data_parallel_size();
    let mut router_hint_worker_type = None;
    let mut router_hint_source_control_endpoints = HashMap::new();
    for dp_rank in dp_start..dp_start.saturating_add(dp_size) {
        if let Some(metadata) = config.router_hint_metadata_for_dp_rank(dp_rank) {
            router_hint_worker_type.get_or_insert_with(|| metadata.worker_type.to_string());
            if let Some(endpoint) = metadata.source_control_endpoint {
                router_hint_source_control_endpoints.insert(dp_rank, endpoint.to_string());
            }
        }
    }
    WorkerRequest {
        worker_id,
        model_name: key.model_name.clone(),
        routing_group: key.routing_group.clone(),
        endpoint: Some(format!("dyn://{worker_id}")),
        block_size: Some(block_size),
        data_parallel_start_rank: Some(dp_start),
        data_parallel_size: Some(dp_size),
        max_num_batched_tokens: config.max_num_batched_tokens,
        total_kv_blocks: config.total_kv_blocks,
        stable_routing_id: config.stable_routing_id.clone(),
        is_eagle: Some(is_eagle),
        taints: config.taints().clone(),
        topology_domains: config.topology_domains.clone(),
        kv_transfer_domain: config.kv_transfer_domain.clone(),
        kv_transfer_enforcement: config.kv_transfer_enforcement,
        kv_transfer_preferred_weight: config.kv_transfer_preferred_weight,
        router_hint_worker_type,
        router_hint_source_control_endpoints,
        ..WorkerRequest::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_request_mirrors_runtime_config() {
        let mut config = ModelRuntimeConfig {
            data_parallel_start_rank: 2,
            data_parallel_size: 2,
            max_num_batched_tokens: Some(8192),
            total_kv_blocks: Some(4096),
            stable_routing_id: Some("worker-0".to_string()),
            ..ModelRuntimeConfig::default()
        };
        config.taints.insert("gpu=h100".to_string());
        config.runtime_data.insert(
            dynamo_kv_router::router_hint::ROUTER_HINT_RUNTIME_CAPABILITY_KEY.to_string(),
            serde_json::Value::Bool(true),
        );
        config.runtime_data.insert(
            dynamo_kv_router::router_hint::ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY.to_string(),
            serde_json::Value::String("decode".to_string()),
        );
        config.runtime_data.insert(
            dynamo_kv_router::router_hint::ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY
                .to_string(),
            serde_json::json!({"2": "tcp://w:9002", "3": "tcp://w:9003"}),
        );
        let key = RoutingPartitionId::new("model", DEFAULT_ROUTING_GROUP);
        let request = worker_request_from_runtime_config(7, &config, &key, 16, false);
        assert_eq!(request.worker_id, 7);
        assert_eq!(request.model_name, "model");
        assert_eq!(request.endpoint.as_deref(), Some("dyn://7"));
        assert_eq!(request.block_size, Some(16));
        assert_eq!(request.data_parallel_start_rank, Some(2));
        assert_eq!(request.data_parallel_size, Some(2));
        assert_eq!(request.max_num_batched_tokens, Some(8192));
        assert_eq!(request.total_kv_blocks, Some(4096));
        assert_eq!(request.stable_routing_id.as_deref(), Some("worker-0"));
        assert!(request.taints.contains("gpu=h100"));
        assert_eq!(request.router_hint_worker_type.as_deref(), Some("decode"));
        assert_eq!(
            request.router_hint_source_control_endpoints,
            HashMap::from([
                (2, "tcp://w:9002".to_string()),
                (3, "tcp://w:9003".to_string())
            ])
        );
    }
}
