// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The selection service under test, built as the frontend builds its embedded
//! selection (`lib/llm/src/kv_router/embedded.rs`), with a host-fed index in place
//! of the event plane. Frontend glue (Prometheus gauges, the request lease
//! manager, replica publishing) is reduced to no-ops; the routing code is not.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use dynamo_kv_indexer::host::{
    ConcurrentRadixTreeCompressed, IndexRegistry, Indexer, KvIndexerMetrics, LowerTierIndexers,
    ThreadPoolIndexer,
};
use dynamo_kv_router::config::KvRouterConfig;
use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::services::selection::{SelectionService, SelectionServiceBuilder};
use dynamo_router_protocol::identity::{DEFAULT_ROUTING_GROUP, RoutingPartitionId};
use dynamo_router_protocol::worker::WorkerId;
use dynamo_router_protocol::worker_type::WorkerType;
use dynamo_scheduler::host::{
    CatalogReconciler, DEFAULT_MODEL_NAME, HostCache, HostEligibility, HostHints, HostLoad,
    HostReplicaChannels, HostReplicaSyncFactory, HostReplication, HostTelemetry, KvEventIngress,
    KvIndexSource, RejectThresholdsProvider, SchedulerLoadSink, SelectionHost, SelectionPartition,
    WorkerRequest,
};
use dynamo_slot_tracker::host::{
    ActiveSequenceEvent, BusyThresholds, ReplicaRequestLeaseObserver, ReplicaWorkerPolicy,
    SchedulerBookingDescriptor, SchedulerLoadSnapshot,
};
use tokio::sync::mpsc;

use super::corpus::WorkloadParams;

/// `lib/llm/src/kv_router/sequence.rs` sizes both replica channels this way.
const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;
/// The bench's router identity on the replica channels; distinct from every worker ID.
const BENCH_ROUTER_ID: u64 = u64::MAX;

/// `KvRouterConfig`'s defaults with replica sync on, as the AgentX frontend runs, and
/// `json`'s fields over them: deserialization fills the defaults and rejects unknown
/// fields. The policy config is applied and the result validated, as the frontend does
/// at startup, so the bench reads the settings the service runs.
pub fn router_config(json: Option<&str>) -> Result<KvRouterConfig> {
    let mut fields: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(json.unwrap_or("{}")).context("--router-config-json")?;
    fields
        .entry("router_replica_sync")
        .or_insert(serde_json::Value::Bool(true));
    let mut config: KvRouterConfig =
        serde_json::from_value(fields.into()).context("--router-config-json")?;
    config.apply_policy_config().map_err(anyhow::Error::msg)?;
    config.validate().map_err(anyhow::Error::msg)?;
    Ok(config)
}

/// Router options the replay does not model. Accepting one would measure a router that
/// differs from the one the config describes.
pub fn check_modeled(router: &KvRouterConfig) -> Result<()> {
    ensure!(
        router.use_kv_events,
        "the bench feeds KV events; use_kv_events must stay on"
    );
    ensure!(
        router.router_event_threads > 1,
        "router_event_threads must be at least 2: the frontend's concurrent indexer"
    );
    ensure!(
        !router.router_track_output_blocks,
        "router_track_output_blocks is not modeled: the replay reports no output blocks \
         to the scheduler"
    );
    ensure!(
        router.router_predicted_ttl_secs.is_none(),
        "router_predicted_ttl_secs is not modeled: the replay builds no predict-on-route index"
    );
    ensure!(
        !router.enable_session_prefix_index,
        "enable_session_prefix_index is not modeled: the replay builds no session prefix index"
    );
    Ok(())
}

/// The policies Dynamo ships, as the frontend links them; the router config picks one.
fn plugin_registry() -> Result<RouterPluginRegistry> {
    let mut registry = dynamo_router_policies::default_registry();
    dynamo_router_policies::register(&mut registry).context("register the builtin policies")?;
    Ok(registry)
}

/// The workload's workers, as the frontend registers them from their runtime configs.
fn workers(params: &WorkloadParams) -> Vec<WorkerRequest> {
    (1..=params.workers as u64)
        .map(|worker_id| WorkerRequest {
            worker_id,
            model_name: DEFAULT_MODEL_NAME.to_string(),
            routing_group: DEFAULT_ROUTING_GROUP.to_string(),
            endpoint: Some(format!("dyn://{worker_id}")),
            block_size: Some(params.block_size),
            data_parallel_start_rank: Some(0),
            data_parallel_size: Some(params.dp_size),
            max_num_batched_tokens: Some(params.max_num_batched_tokens as u64),
            total_kv_blocks: Some(params.num_gpu_blocks as u64),
            is_eagle: Some(false),
            ..WorkerRequest::default()
        })
        .collect()
}

pub struct BenchHost {
    pub service: Arc<SelectionService>,
    pub partition: SelectionPartition,
    /// The index the partition reads, fed directly as the frontend's ingress does.
    pub indexer: Indexer,
    /// Worker-origin replica events, as the event plane would deliver them.
    pub inbound: mpsc::Sender<ActiveSequenceEvent>,
    /// The index's event counters, by event type and outcome.
    pub index_metrics: Arc<KvIndexerMetrics>,
    pub track_active_blocks: bool,
}

impl BenchHost {
    /// Build the service for `router` and register the workload's workers. Call inside
    /// the runtime under test, so the scheduler actor and replica tasks run on it.
    pub async fn start(
        router: &KvRouterConfig,
        busy: BusyThresholds,
        params: &WorkloadParams,
    ) -> Result<Self> {
        check_modeled(router)?;
        let workers = workers(params);
        let threads = router.router_event_threads as usize;

        // As `lib/llm/src/kv_router/indexer/mod.rs` builds the concurrent index.
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let indexer = Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new_with_metrics(
                ConcurrentRadixTreeCompressed::new(),
                threads,
                params.block_size,
                Some(Arc::clone(&metrics)),
            )),
            lower_tier: LowerTierIndexers::new_with_metrics(
                threads,
                params.block_size,
                Some(Arc::clone(&metrics)),
            ),
            approx: None,
            primary_records_routing_decisions: false,
            session_updates: None,
        };

        let all_workers: Arc<HashSet<WorkerId>> =
            Arc::new(workers.iter().map(|worker| worker.worker_id).collect());
        // The frontend's client reports these on every selection; here none is overloaded.
        let none_overloaded: Arc<HashSet<WorkerId>> = Arc::default();
        let load = HostLoad {
            prefill_estimator: None,
            overloaded_workers: Some(Arc::new(move || Some(Arc::clone(&none_overloaded)))),
            available_workers: Some(Arc::new(move |_| Some(Arc::clone(&all_workers)))),
            reject_thresholds: busy
                .is_configured()
                .then(|| Arc::new(move || busy) as RejectThresholdsProvider),
        };

        let (inbound_tx, inbound_rx) = mpsc::channel(REPLICA_EVENT_CHANNEL_CAPACITY);
        let outbound = router.router_replica_sync.then(|| {
            let (tx, mut rx) = mpsc::channel::<ActiveSequenceEvent>(REPLICA_EVENT_CHANNEL_CAPACITY);
            // The frontend serializes these for peer routers; the bench has none.
            tokio::spawn(async move {
                let mut batch = Vec::new();
                while rx.recv_many(&mut batch, 256).await > 0 {
                    batch.clear();
                }
            });
            tx
        });
        let slot = Mutex::new(Some(HostReplicaChannels {
            outbound,
            inbound_tx: inbound_tx.clone(),
            inbound_rx,
            process_id: BENCH_ROUTER_ID,
            ingress_observer: None,
        }));
        let channels: HostReplicaSyncFactory =
            Arc::new(move |_| slot.lock().ok().and_then(|mut channels| channels.take()));

        // No factory is set, so the builder resolves the worker-selection and hint
        // policies the config names from the registry, as the frontend's catalog does.
        let service = SelectionServiceBuilder::new(
            router.clone(),
            WorkerType::Aggregated,
            plugin_registry()?,
        )
        .host_manages_request_lifecycle()
        .indexer_threads(1)
        .host(SelectionHost {
            load,
            cache: HostCache {
                shared: None,
                index: KvIndexSource::Owned(Arc::new(HostIngress(indexer.clone()))),
            },
            eligibility: HostEligibility::default(),
            telemetry: HostTelemetry {
                scheduler_load: Some(Arc::new(NoopLoadSink)),
            },
            replication: HostReplication {
                channels: Some(channels),
                request_leases: Some(Arc::new(NoopLeases)),
                replica_worker_policy: ReplicaWorkerPolicy::LazyRegister,
            },
            hints: HostHints::default(),
        })
        .build()
        .await
        .context("build selection service")?;
        let service = Arc::new(service);
        let partition = service
            .core()
            .ensure_partition(
                RoutingPartitionId::new(DEFAULT_MODEL_NAME, DEFAULT_ROUTING_GROUP),
                params.block_size,
                false,
            )
            .context("create selection partition")?;
        CatalogReconciler::new(Arc::clone(service.core()))
            .apply(&workers)
            .await
            .context("register workers")?;

        Ok(Self {
            service,
            partition,
            indexer,
            inbound: inbound_tx,
            index_metrics: metrics,
            track_active_blocks: router.router_track_active_blocks,
        })
    }

    /// Events the index has applied, keyed `type/outcome`.
    pub fn index_events_applied(&self) -> std::collections::BTreeMap<String, u64> {
        use prometheus::core::Collector;
        let mut applied = std::collections::BTreeMap::new();
        for family in self.index_metrics.kv_cache_events_applied.collect() {
            for metric in family.get_metric() {
                let labels: Vec<&str> = metric.get_label().iter().map(|l| l.value()).collect();
                *applied.entry(labels.join("/")).or_default() +=
                    metric.get_counter().value() as u64;
            }
        }
        applied
    }

    pub async fn shutdown(&self) {
        self.service.shutdown().await;
    }
}

/// The frontend's `RuntimeIngress`: hands the partition the index the host feeds.
struct HostIngress(Indexer);

impl KvEventIngress for HostIngress {
    fn open(&self, _: &IndexRegistry, _: &RoutingPartitionId, _: u32) -> Indexer {
        self.0.clone()
    }
}

/// Stands in for the frontend's Prometheus gauges and load publisher. The core
/// still builds every snapshot it would hand them.
struct NoopLoadSink;

impl SchedulerLoadSink for NoopLoadSink {
    fn publish(&self, _: SchedulerLoadSnapshot) {}
}

/// Stands in for the frontend's request lease manager. Having one tells the core
/// the host owns request expiry, as in the frontend.
struct NoopLeases;

impl ReplicaRequestLeaseObserver for NoopLeases {
    fn admitted(&self, _: SchedulerBookingDescriptor) {}
    fn progressed(&self, _: &SchedulerBookingDescriptor) {}
    fn completed(&self, _: &SchedulerBookingDescriptor) {}
}
