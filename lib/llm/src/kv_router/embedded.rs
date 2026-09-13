// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded selection backend for the frontend `KvRouter`.
//!
//! The runtime-config watch feeds one selection partition's worker catalog.
//! Runtime ingress feeds the partition's index, and the frontend prepares overlap
//! and request constraints before scheduling directly on that partition. The
//! frontend retains transport, stream leases, and request-expiry ownership.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result};
use dynamo_kv_router::WorkerType;
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank};
use dynamo_kv_router::scheduling::queue::DEFAULT_MAX_BATCHED_TOKENS;
use dynamo_kv_router::scheduling::{
    NonMaxOverlapSelectionObserver, OverloadedWorkerProvider, QueueLimitKind, QueueRejection,
    WorkerAvailabilityProvider,
};
use dynamo_kv_router::services::selection::{
    CatalogObserver, CatalogReconciler, DEFAULT_MODEL_NAME, HostCache, HostEligibility, HostLoad,
    HostReplication, HostTelemetry, KvEventIngress, KvIndexSource, SelectionHost,
    SelectionOperation, SelectionOutcome, SelectionPartition, SelectionRun, SelectionScheduler,
    SelectionService, SelectionServiceBuilder, WorkerCatalogRecord, WorkerCatalogSource,
    WorkerRequest, WorkerSelectionPolicyRegistry,
};
use dynamo_kv_router::{DEFAULT_ROUTING_GROUP, PrefillLoadEstimator, WorkerSelectionPolicyFactory};
use tokio_util::sync::CancellationToken;

use crate::discovery::RuntimeConfigWatch;
use crate::kv_router::metrics::{
    ROUTER_QUEUE_METRICS, RouterQueueMetricHandles, RouterRequestMetrics, WORKER_LOAD_METRICS,
};
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
    pub shared_cache: Option<Arc<dyn dynamo_kv_router::SharedKvCache>>,
    pub lora_worker_filter: Option<Arc<dyn dynamo_kv_router::scheduling::LoraWorkerFilter>>,
    /// Builds and feeds the router's index from the runtime; the partition
    /// takes its index from it.
    pub ingress: Arc<dyn KvEventIngress>,
    /// Scheduler-owned load snapshots for the worker monitor's overload
    /// detection.
    pub scheduler_load: crate::kv_router::routing_load::SchedulerLoadSender,
    /// Endpoint whose event plane carries replica sync when
    /// `router_replica_sync` is set.
    pub endpoint: dynamo_runtime::component::Endpoint,
    /// This router replica's id (ignored on receipt of its own events).
    pub router_id: u64,
    /// Builds the partition's worker-selection policy (see
    /// `SelectionPolicySource::resolve`).
    pub policy_factory: WorkerSelectionPolicyFactory,
}

fn record_queue_rejection(
    per_class: &[RouterQueueMetricHandles],
    indices: &HashMap<String, usize>,
    rejection: &QueueRejection,
) {
    let Some(handles) = indices
        .get(&rejection.policy_class)
        .and_then(|index| per_class.get(*index))
    else {
        return;
    };
    match rejection.limit_kind {
        QueueLimitKind::Requests => handles.request_limit_rejections.inc(),
        QueueLimitKind::RawIslTokens => handles.raw_isl_limit_rejections.inc(),
        QueueLimitKind::CachedTokens => handles.cached_token_limit_rejections.inc(),
    }
}

fn update_queue_metrics(per_class: &[RouterQueueMetricHandles], scheduler: &SelectionScheduler) {
    for (class_index, handles) in per_class.iter().enumerate() {
        let Some(stats) = scheduler.class_queue_stats(class_index) else {
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

/// Refresh queue gauges on worker-config changes and at least once a minute.
fn spawn_queue_metrics_updater(
    partition: SelectionPartition,
    handles: Vec<RouterQueueMetricHandles>,
    cancellation_token: CancellationToken,
) {
    let mut queue_updates = partition.scheduler().subscribe_queue_updates();
    tokio::spawn(async move {
        let period = Duration::from_secs(60);
        let mut recheck = tokio::time::interval_at(tokio::time::Instant::now() + period, period);
        loop {
            update_queue_metrics(&handles, partition.scheduler());
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                changed = queue_updates.changed() => {
                    if changed.is_err() {
                        break;
                    }
                }
                _ = recheck.tick() => {}
            }
        }
    });
}

fn non_max_overlap_observer(worker_type: &'static str) -> NonMaxOverlapSelectionObserver {
    Arc::new(move |request_id, selection| {
        let overlap_blocks_lost = selection.overlap_blocks_lost();
        if let Some(metrics) = RouterRequestMetrics::get() {
            metrics.observe_non_max_overlap_selection(worker_type, overlap_blocks_lost);
        }
        tracing::debug!(
            request_id,
            worker_type,
            selected_worker_id = selection.selected_worker.worker_id,
            selected_dp_rank = selection.selected_worker.dp_rank,
            selected_overlap_blocks = selection.selected_overlap_blocks,
            highest_overlap_worker_id = selection.highest_overlap_worker.worker_id,
            highest_overlap_dp_rank = selection.highest_overlap_worker.dp_rank,
            highest_overlap_blocks = selection.highest_overlap_blocks,
            overlap_blocks_lost,
            "Router selected a worker with lower KV cache overlap"
        );
    })
}

/// One `SelectionService` partition driven directly by the router.
pub(crate) struct EmbeddedSelection {
    /// Keeps the service (listeners, sweep, replica sync) alive for as long as
    /// the router holds the partition.
    service: Arc<SelectionService>,
    partition: SelectionPartition,
    affinity: OnceLock<crate::session_affinity::AffinityCoordinator>,
    worker_type: &'static str,
    /// Queue gauges and rejection counters per policy class, index-aligned with
    /// the scheduler's `class_queue_stats`.
    queue_metrics: Vec<RouterQueueMetricHandles>,
    queue_metric_indices: HashMap<String, usize>,
}

static INSTALLED_POLICY_REGISTRY: OnceLock<WorkerSelectionPolicyRegistry> = OnceLock::new();

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
/// `SchedulerLoadSender`, which feeds `KvWorkerMonitor`, and its per-worker
/// load to the frontend gauges.
struct SenderLoadSink {
    sender: crate::kv_router::routing_load::SchedulerLoadSender,
    worker_type: &'static str,
}

impl dynamo_kv_router::services::selection::SchedulerLoadSink for SenderLoadSink {
    fn publish(&self, snapshot: dynamo_kv_router::sequences::SchedulerLoadSnapshot) {
        self.sender.publish(snapshot);
    }

    fn publish_batch(&self, snapshots: Vec<dynamo_kv_router::sequences::SchedulerLoadSnapshot>) {
        self.sender.publish_batch(snapshots);
    }

    fn observe_local_load(&self, worker: &WorkerWithDpRank, blocks: usize, tokens: usize) {
        WORKER_LOAD_METRICS.observe(
            worker.worker_id,
            worker.dp_rank,
            self.worker_type,
            blocks,
            tokens,
        );
    }
}

impl EmbeddedSelection {
    /// Returns the partition's inbound replica ingress, not yet running. The
    /// caller starts it only after the lease manager it passed as
    /// `request_leases` has its scheduler set, so no lifecycle event reaches
    /// the manager before it can release the booking.
    pub(crate) async fn start(
        args: EmbeddedSelectionArgs,
        workers_with_configs: RuntimeConfigWatch,
        request_leases: Option<Arc<dyn dynamo_kv_router::sequences::ReplicaRequestLeaseObserver>>,
        cancellation_token: CancellationToken,
    ) -> Result<(Self, crate::kv_router::sequence::ReplicaIngress)> {
        let worker_type = args.worker_role.unwrap_or(WorkerType::Aggregated);
        let key = RoutingPartitionId::new(
            args.model_name
                .clone()
                .unwrap_or_else(|| DEFAULT_MODEL_NAME.to_string()),
            DEFAULT_ROUTING_GROUP,
        );

        // Replica sync rides the runtime event plane. Worker-origin completion
        // marks are consumed even when router-to-router replica sync is
        // disabled; only publishing is gated.
        let (channels, replica_ingress) = crate::kv_router::sequence::host_replica_channels(
            &args.endpoint,
            args.router_id,
            args.kv_router_config.router_replica_sync,
            cancellation_token.child_token(),
        )
        .await
        .context("start replica sync for the embedded selection partition")?;
        let slot = std::sync::Mutex::new(Some(channels));
        let replica_sync: Option<dynamo_kv_router::services::selection::HostReplicaSyncFactory> =
            Some(Arc::new(move |_partition| {
                slot.lock().ok().and_then(|mut s| s.take())
            }));

        // The registry is only consulted when no factory is set; `policy_factory`
        // always is, so the builder never reads it.
        let service = SelectionServiceBuilder::new(
            args.kv_router_config.clone(),
            worker_type,
            WorkerSelectionPolicyRegistry::default(),
        )
        .worker_selection_policy_factory(args.policy_factory)
        .indexer_threads(1)
        .host(SelectionHost {
            load: HostLoad {
                prefill_estimator: args.prefill_load_estimator,
                overloaded_workers: Some(args.overloaded_worker_provider),
                available_workers: Some(args.available_worker_provider),
            },
            cache: HostCache {
                shared: args.shared_cache,
                index: KvIndexSource::Owned(args.ingress),
            },
            eligibility: HostEligibility {
                lora_worker_filter: args.lora_worker_filter,
            },
            telemetry: HostTelemetry {
                scheduler_load: Some(Arc::new(SenderLoadSink {
                    sender: args.scheduler_load,
                    worker_type: args.metric_worker_type,
                })),
            },
            replication: HostReplication {
                channels: replica_sync,
                request_leases,
            },
        })
        .build()
        .await
        .context("failed to start embedded selection service")?;
        let service = Arc::new(service);
        let partition = service
            .core()
            .ensure_partition(key.clone(), args.block_size, args.is_eagle)
            .context("failed to create embedded selection partition")?;

        // Same profile the partition scheduler resolved, so class indices align.
        let profile = args
            .kv_router_config
            .policy_profile(Some(&key.model_name))
            .context("failed to resolve the embedded selection policy profile")?;
        let queue_metrics: Vec<_> = profile
            .classes()
            .iter()
            .map(|class| {
                ROUTER_QUEUE_METRICS.handles(&key.model_name, args.metric_worker_type, &class.name)
            })
            .collect();
        let queue_metric_indices = profile
            .classes()
            .iter()
            .enumerate()
            .map(|(index, class)| (class.name.clone(), index))
            .collect();
        spawn_queue_metrics_updater(
            partition.clone(),
            queue_metrics.clone(),
            cancellation_token.child_token(),
        );
        if worker_type == WorkerType::Prefill
            && !partition
                .scheduler()
                .set_non_max_overlap_selection_observer(non_max_overlap_observer(
                    args.metric_worker_type,
                ))
        {
            anyhow::bail!("non-max-overlap observer is already installed");
        }

        let mut source = RuntimeDiscoverySource {
            watch: workers_with_configs,
            key,
            block_size: args.block_size,
            is_eagle: args.is_eagle,
            primed: false,
        };
        let mut reconciler = CatalogReconciler::new(Arc::clone(service.core())).with_observer(
            Arc::new(RegisteredGauge::new(
                super::metrics::RouterWorkerStatusMetrics::from_component(
                    args.endpoint.component(),
                ),
                args.metric_worker_type,
            )),
        );
        // The current membership is in the catalog before the router serves.
        if let Some(snapshot) = source.next_snapshot().await
            && let Err(error) = reconciler.apply(snapshot).await
        {
            tracing::warn!(%error, "embedded selection: initial membership reconcile failed");
        }
        tokio::spawn(reconciler.run(source, cancellation_token.child_token()));

        tracing::info!(
            worker_type = %worker_type,
            "KvRouter scheduling on embedded selection partition"
        );
        Ok((
            Self {
                service,
                partition,
                affinity: OnceLock::new(),
                worker_type: args.metric_worker_type,
                queue_metrics,
                queue_metric_indices,
            },
            replica_ingress,
        ))
    }

    fn observe_queue(&self, rejection: Option<&QueueRejection>) {
        if let Some(rejection) = rejection {
            record_queue_rejection(&self.queue_metrics, &self.queue_metric_indices, rejection);
        }
        update_queue_metrics(&self.queue_metrics, self.partition.scheduler());
    }

    pub(crate) fn affinity_coordinator(
        &self,
        ttl: Duration,
        mode: crate::session_affinity::SessionAffinityMode,
    ) -> Result<crate::session_affinity::AffinityCoordinator> {
        let table = self.partition.session_affinity(
            dynamo_kv_router::services::selection::affinity::SessionAffinityConfig::new(ttl)
                .with_mode(mode),
        )?;
        Ok(self
            .affinity
            .get_or_init(|| crate::session_affinity::AffinityCoordinator::wrap(table))
            .clone())
    }

    pub(crate) fn partition_key(&self) -> &RoutingPartitionId {
        self.partition.key()
    }

    /// Run one selection through the shared core.
    pub(crate) async fn run_selection(&self, operation: SelectionOperation<'_>) -> SelectionRun {
        // Keep the selection state out of the frontend's nested request future.
        let run = Box::pin(self.service.core().run_selection(operation)).await;
        self.observe_queue(match &run.result {
            Ok(SelectionOutcome::QueueRejected { rejection }) => Some(rejection),
            _ => None,
        });
        run
    }

    pub(crate) fn worker_type(&self) -> &'static str {
        self.worker_type
    }

    /// The partition's scheduler, for bookings and load queries.
    pub(crate) fn scheduler(&self) -> &SelectionScheduler {
        self.partition.scheduler()
    }
}

/// The runtime-config watch as worker membership for the partition.
struct RuntimeDiscoverySource {
    watch: RuntimeConfigWatch,
    key: RoutingPartitionId,
    block_size: u32,
    is_eagle: bool,
    primed: bool,
}

#[async_trait::async_trait]
impl WorkerCatalogSource for RuntimeDiscoverySource {
    async fn next_snapshot(&mut self) -> Option<Vec<WorkerRequest>> {
        if self.primed {
            self.watch.changed().await.ok()?;
        }
        self.primed = true;
        let snapshot = self
            .watch
            .borrow_and_update()
            .iter()
            .map(|(worker_id, config)| {
                worker_request_from_runtime_config(
                    *worker_id,
                    config,
                    &self.key,
                    self.block_size,
                    self.is_eagle,
                )
            })
            .collect();
        Some(snapshot)
    }
}

/// Keeps the per-worker `router_worker_registered` gauge in step with the catalog.
struct RegisteredGauge {
    metrics: Arc<super::metrics::RouterWorkerStatusMetrics>,
    worker_label: &'static str,
    /// Last published rank range per worker, so a shrinking or shifted
    /// `data_parallel_size` clears the ranks that left instead of stranding
    /// them at 1.
    ranks: std::sync::Mutex<HashMap<WorkerId, std::ops::Range<u32>>>,
}

impl RegisteredGauge {
    fn new(
        metrics: Arc<super::metrics::RouterWorkerStatusMetrics>,
        worker_label: &'static str,
    ) -> Self {
        Self {
            metrics,
            worker_label,
            ranks: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl CatalogObserver for RegisteredGauge {
    fn upserted(&self, record: &WorkerCatalogRecord) {
        let current = record.dp_ranks();
        let previous = self
            .ranks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(record.worker_id, current.clone());
        for dp_rank in previous
            .into_iter()
            .flatten()
            .filter(|dp_rank| !current.contains(dp_rank))
        {
            self.metrics
                .remove_worker(record.worker_id, dp_rank, self.worker_label);
        }
        for dp_rank in current {
            self.metrics
                .set_registered(record.worker_id, dp_rank, self.worker_label);
        }
    }

    fn removed(&self, record: &WorkerCatalogRecord) {
        let previous = self
            .ranks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&record.worker_id);
        // The record's own ranks plus whatever an earlier upsert published.
        for dp_rank in record.dp_ranks().chain(previous.into_iter().flatten()) {
            self.metrics
                .remove_worker(record.worker_id, dp_rank, self.worker_label);
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
        if let Some(metadata) = config.kv_hint_transfer_metadata_for_dp_rank(dp_rank) {
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
        // Default unreported capacity so the catalog's queueing gate does not
        // mark the worker Incomplete; the scheduler applies the same fallback.
        max_num_batched_tokens: Some(
            config
                .max_num_batched_tokens
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS),
        ),
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
        kv_event_source_mode: config.kv_event_source_mode.clone(),
        ..WorkerRequest::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::services::selection::SchedulerLoadSink;

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
            dynamo_kv_router::kv_hints::KV_HINT_TRANSFER_CAPABILITY_KEY.to_string(),
            serde_json::Value::Bool(true),
        );
        config.runtime_data.insert(
            dynamo_kv_router::kv_hints::KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY.to_string(),
            serde_json::Value::String("decode".to_string()),
        );
        config.runtime_data.insert(
            dynamo_kv_router::kv_hints::KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY
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

    #[test]
    fn worker_request_defaults_unreported_capacity() {
        let key = RoutingPartitionId::new("model", DEFAULT_ROUTING_GROUP);
        let request =
            worker_request_from_runtime_config(7, &ModelRuntimeConfig::default(), &key, 16, false);
        assert_eq!(
            request.max_num_batched_tokens,
            Some(DEFAULT_MAX_BATCHED_TOKENS)
        );
    }

    /// A data-parallel shrink clears the gauges of the ranks that left; a
    /// removal clears every rank the worker ever published.
    #[test]
    fn registered_gauge_clears_ranks_that_leave_the_worker() {
        let metrics = Arc::new(super::super::metrics::RouterWorkerStatusMetrics::unregistered());
        let gauge = RegisteredGauge::new(Arc::clone(&metrics), "decode");
        let record = |dp_size: u32| {
            WorkerCatalogRecord::new(WorkerRequest {
                worker_id: 7,
                data_parallel_start_rank: Some(0),
                data_parallel_size: Some(dp_size),
                ..WorkerRequest::default()
            })
        };
        // `get_metric_with_label_values` creates the child it looks up;
        // `collect` reports only the children that exist.
        let registered = |dp_rank: u32| {
            use prometheus::core::Collector;
            let dp_rank = dp_rank.to_string();
            metrics.registered.collect().into_iter().find_map(|family| {
                family
                    .get_metric()
                    .iter()
                    .find(|metric| {
                        metric
                            .get_label()
                            .iter()
                            .any(|label| label.name() == "dp_rank" && label.value() == dp_rank)
                    })
                    .map(|metric| metric.get_gauge().value() as i64)
            })
        };

        gauge.upserted(&record(4));
        assert!((0..4).all(|dp_rank| registered(dp_rank) == Some(1)));

        gauge.upserted(&record(2));
        assert_eq!(registered(0), Some(1));
        assert_eq!(registered(1), Some(1));
        assert_eq!(registered(2), None, "rank 2 left the worker");
        assert_eq!(registered(3), None, "rank 3 left the worker");

        gauge.removed(&record(2));
        assert!((0..4).all(|dp_rank| registered(dp_rank).is_none()));
    }

    #[test]
    fn local_load_observation_sets_worker_gauges() {
        let sink = SenderLoadSink {
            sender: crate::kv_router::routing_load::SchedulerLoadSender::disabled(
                CancellationToken::new(),
            ),
            worker_type: "decode",
        };
        sink.observe_local_load(&WorkerWithDpRank::new(3, 1), 5, 7);
        let labels = ["3", "1", "decode"];
        assert_eq!(
            WORKER_LOAD_METRICS
                .active_decode_blocks
                .with_label_values(&labels)
                .get(),
            5
        );
        assert_eq!(
            WORKER_LOAD_METRICS
                .active_prefill_tokens
                .with_label_values(&labels)
                .get(),
            7
        );
        let _ = WORKER_LOAD_METRICS
            .active_decode_blocks
            .remove_label_values(&labels);
        let _ = WORKER_LOAD_METRICS
            .active_prefill_tokens
            .remove_label_values(&labels);
    }

    #[test]
    fn queue_rejections_count_against_their_policy_class() {
        let per_class = vec![
            ROUTER_QUEUE_METRICS.handles("m-reject", "decode", "interactive"),
            ROUTER_QUEUE_METRICS.handles("m-reject", "decode", "batch"),
        ];
        let indices = HashMap::from([("interactive".to_string(), 0), ("batch".to_string(), 1)]);
        record_queue_rejection(
            &per_class,
            &indices,
            &QueueRejection {
                policy_class: "batch".to_string(),
                limit_kind: QueueLimitKind::RawIslTokens,
                current: 9,
                limit: 8,
            },
        );
        assert_eq!(per_class[1].raw_isl_limit_rejections.get(), 1);
        assert_eq!(per_class[0].raw_isl_limit_rejections.get(), 0);
        assert_eq!(per_class[1].request_limit_rejections.get(), 0);
    }
}
