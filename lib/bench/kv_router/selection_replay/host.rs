// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The selection service under test, built as the frontend builds its embedded
//! selection (`lib/llm/src/kv_router/embedded.rs`), with a host-fed index in place
//! of the event plane. Frontend glue (Prometheus gauges, the request lease
//! manager, replica publishing) is reduced to no-ops; the routing code is not.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use dynamo_kv_router::identity::RoutingPartitionId;
use dynamo_kv_router::indexer::{KvIndexerMetrics, LowerTierIndexers};
use dynamo_kv_router::plugins::RouterPluginRegistry;
use dynamo_kv_router::protocols::{ActiveSequenceEvent, WorkerId};
use dynamo_kv_router::scheduling::policy_config::{BusyThresholds, RejectThresholdsProvider};
use dynamo_kv_router::scheduling::queue::SchedulerBookingDescriptor;
use dynamo_kv_router::sequences::{
    ReplicaRequestLeaseObserver, ReplicaWorkerPolicy, SchedulerLoadSnapshot,
};
use dynamo_kv_router::services::indexer::backend::Indexer;
use dynamo_kv_router::services::indexer::registry::WorkerRegistry;
use dynamo_kv_router::services::selection::{
    CatalogReconciler, DEFAULT_MODEL_NAME, HostCache, HostEligibility, HostLoad,
    HostReplicaChannels, HostReplicaSyncFactory, HostReplication, HostTelemetry, KvEventIngress,
    KvIndexSource, SchedulerLoadSink, SelectionHost, SelectionPartition, SelectionService,
    SelectionServiceBuilder, WorkerRequest,
};
use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, DEFAULT_ROUTING_GROUP, KvRouterConfig, ThreadPoolIndexer,
    WorkerType,
};
use dynamo_router_policies::default_factory;
use tokio::sync::mpsc;

/// `lib/llm/src/kv_router/sequence.rs` sizes both replica channels this way.
const REPLICA_EVENT_CHANNEL_CAPACITY: usize = 100_000;
/// The bench's router identity on the replica channels; distinct from every worker ID.
const BENCH_ROUTER_ID: u64 = u64::MAX;

#[derive(Clone, Debug)]
pub struct HostConfig {
    pub router: KvRouterConfig,
    pub block_size: u32,
    pub busy: BusyThresholds,
}

/// One worker as the frontend registers it from its runtime config.
pub fn worker_request(
    worker_id: WorkerId,
    block_size: u32,
    dp_size: u32,
    max_num_batched_tokens: u64,
    total_kv_blocks: u64,
) -> WorkerRequest {
    WorkerRequest {
        worker_id,
        model_name: DEFAULT_MODEL_NAME.to_string(),
        routing_group: DEFAULT_ROUTING_GROUP.to_string(),
        endpoint: Some(format!("dyn://{worker_id}")),
        block_size: Some(block_size),
        data_parallel_start_rank: Some(0),
        data_parallel_size: Some(dp_size),
        max_num_batched_tokens: Some(max_num_batched_tokens),
        total_kv_blocks: Some(total_kv_blocks),
        is_eagle: Some(false),
        ..WorkerRequest::default()
    }
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
    /// Build the service and register `workers`. Call inside the runtime under test,
    /// so the scheduler actor and replica tasks run on it.
    pub async fn start(config: &HostConfig, workers: &[WorkerRequest]) -> Result<Self> {
        let router = &config.router;
        ensure!(
            router.use_kv_events,
            "the bench feeds KV events; use_kv_events must stay on"
        );
        ensure!(
            !router.router_track_output_blocks,
            "output-block tracking is not modeled: the corpus keeps no per-block output timing"
        );
        let threads = router.router_event_threads as usize;
        ensure!(
            threads > 1,
            "router_event_threads must be at least 2: the frontend's concurrent indexer"
        );

        // As `lib/llm/src/kv_router/indexer/mod.rs` builds the concurrent index.
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let indexer = Indexer::Concurrent {
            primary: Arc::new(ThreadPoolIndexer::new_with_metrics(
                ConcurrentRadixTreeCompressed::new(),
                threads,
                config.block_size,
                Some(Arc::clone(&metrics)),
            )),
            lower_tier: LowerTierIndexers::new_with_metrics(
                threads,
                config.block_size,
                Some(Arc::clone(&metrics)),
            ),
            approx: None,
            primary_records_routing_decisions: false,
            session_updates: None,
        };

        let all_workers: Arc<HashSet<WorkerId>> =
            Arc::new(workers.iter().map(|worker| worker.worker_id).collect());
        let busy = config.busy;
        let load = HostLoad {
            prefill_estimator: None,
            overloaded_workers: None,
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

        let service = SelectionServiceBuilder::new(
            router.clone(),
            WorkerType::Aggregated,
            RouterPluginRegistry::default(),
        )
        .worker_selection_policy_factory(default_factory())
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
        })
        .build()
        .await
        .context("build selection service")?;
        let service = Arc::new(service);
        let partition = service
            .core()
            .ensure_partition(
                RoutingPartitionId::new(DEFAULT_MODEL_NAME, DEFAULT_ROUTING_GROUP),
                config.block_size,
                false,
            )
            .context("create selection partition")?;
        CatalogReconciler::new(Arc::clone(service.core()))
            .apply(workers)
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
    fn open(&self, _: &WorkerRegistry, _: &RoutingPartitionId, _: u32) -> Indexer {
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
