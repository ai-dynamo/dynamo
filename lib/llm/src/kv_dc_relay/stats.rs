// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dynamo_kv_router::identity::{IdentitySource as DynamoIdentitySource, PoolId};
use dynamo_kv_router::indexer::cuckoo::{
    ConsumerInstanceId, DcCkfDelta, LaneLease, ProducerIdentity,
};
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, EventChannelQuery};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use futures::Stream;
use prost::Message;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::{ReceiverStream, TcpListenerStream};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

use super::actor::{DcCkfSubscription, KvDcRelayHandle};
use super::host::{SharedEndpointStatus, SlotLifecycle};
use super::identity::{
    CanonicalModelRegistration, DcPoolCatalog, DcRelayIdentity, ModelTarget, WorkerRole,
};
use super::pool_registry::PoolRegistry;
use crate::frontend_load::{
    FRONTEND_LOAD_TOPIC, FRONTEND_LOAD_WINDOW_MS, FrontendLoadFrame, FrontendModelLoad,
};
use crate::worker_type::WorkerType;

type EndpointStatuses =
    Arc<tokio::sync::RwLock<HashMap<dynamo_runtime::protocols::EndpointId, SharedEndpointStatus>>>;

pub mod proto {
    tonic::include_proto!("dynamo.kvdc.relay.v1");
}

use proto::kv_dc_relay_server::{KvDcRelay, KvDcRelayServer};

const SNAPSHOT_INTERVAL: Duration = Duration::from_secs(1);
const SOURCE_FRESHNESS: Duration = Duration::from_secs(3);
const MAX_CKF_MESSAGE_BYTES: usize = 64 * 1024 * 1024;
const CKF_STREAM_CAPACITY: usize = 64;

pub(super) struct RelayStatsRuntime {
    tasks: Vec<JoinHandle<()>>,
}

impl RelayStatsRuntime {
    pub(super) async fn start(
        component: Component,
        identity: DcRelayIdentity,
        statuses: EndpointStatuses,
        pools: Arc<PoolRegistry>,
        listen_address: SocketAddr,
        cancel: CancellationToken,
    ) -> anyhow::Result<Self> {
        validate_listen_address(listen_address)?;
        let listener = tokio::net::TcpListener::bind(listen_address).await?;
        let metadata = relay_metadata(identity);
        let (usage_tx, usage_rx) = watch::channel(proto::KvUsageSnapshot {
            metadata: Some(metadata),
            pools: Vec::new(),
        });
        let (load_tx, load_rx) = watch::channel(proto::LoadSnapshot {
            metadata: Some(metadata),
            window_ms: FRONTEND_LOAD_WINDOW_MS,
            pools: Vec::new(),
            models: Vec::new(),
        });
        let (frontend_tx, frontend_rx) = mpsc::channel(CKF_STREAM_CAPACITY);

        let frontend_component = component.clone();
        let frontend_cancel = cancel.child_token();
        let frontend_task = tokio::spawn(async move {
            run_frontend_subscriber(frontend_component, frontend_tx, frontend_cancel).await;
        });

        let aggregate_component = component.clone();
        let aggregate_statuses = statuses.clone();
        let aggregate_cancel = cancel.child_token();
        let aggregate_pools = pools.clone();
        let aggregate_task = tokio::spawn(async move {
            run_aggregate_publisher(
                aggregate_component,
                identity,
                aggregate_statuses,
                aggregate_pools,
                frontend_rx,
                (usage_tx, load_tx),
                aggregate_cancel,
            )
            .await;
        });

        let service = RelayStatsService {
            identity,
            pools,
            statuses,
            usage: usage_rx,
            load: load_rx,
        };
        let server_cancel = cancel.child_token();
        let server_task = tokio::spawn(async move {
            let server = KvDcRelayServer::new(service)
                .max_encoding_message_size(MAX_CKF_MESSAGE_BYTES)
                .max_decoding_message_size(MAX_CKF_MESSAGE_BYTES);
            let result = tonic::transport::Server::builder()
                .add_service(server)
                .serve_with_incoming_shutdown(
                    TcpListenerStream::new(listener),
                    server_cancel.cancelled_owned(),
                )
                .await;
            if let Err(error) = result {
                tracing::error!(%error, "KV DC Relay gRPC server stopped");
            }
        });

        Ok(Self {
            tasks: vec![frontend_task, aggregate_task, server_task],
        })
    }

    pub(super) async fn shutdown(self) {
        for task in self.tasks {
            if let Err(error) = task.await
                && !error.is_cancelled()
            {
                tracing::warn!(%error, "KV DC Relay stats task failed during shutdown");
            }
        }
    }
}

fn validate_listen_address(address: SocketAddr) -> anyhow::Result<()> {
    anyhow::ensure!(
        address.ip().is_loopback(),
        "KV DC Relay gRPC must bind loopback unless mTLS termination is configured"
    );
    Ok(())
}

#[derive(Clone)]
struct RelayStatsService {
    identity: DcRelayIdentity,
    pools: Arc<PoolRegistry>,
    statuses: EndpointStatuses,
    usage: watch::Receiver<proto::KvUsageSnapshot>,
    load: watch::Receiver<proto::LoadSnapshot>,
}

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl KvDcRelay for RelayStatsService {
    type WatchKvCuckooFilterStream = ResponseStream<proto::KvCuckooFilterUpdate>;
    type WatchKvUsageStream = ResponseStream<proto::KvUsageSnapshot>;
    type WatchLoadStream = ResponseStream<proto::LoadSnapshot>;

    async fn watch_kv_cuckoo_filter(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::WatchKvCuckooFilterStream>, Status> {
        let (sender, receiver) = mpsc::channel(CKF_STREAM_CAPACITY);
        let pools = self.pools.clone();
        let statuses = self.statuses.clone();
        let identity = self.identity;
        tokio::spawn(async move {
            if let Err(error) = run_ckf_stream(identity, pools, statuses, sender.clone()).await {
                let _ = sender.send(Err(error)).await;
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }

    async fn watch_kv_usage(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::WatchKvUsageStream>, Status> {
        Ok(Response::new(watch_stream(self.usage.clone())))
    }

    async fn watch_load(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::WatchLoadStream>, Status> {
        Ok(Response::new(watch_stream(self.load.clone())))
    }
}

fn watch_stream<T>(mut receiver: watch::Receiver<T>) -> ResponseStream<T>
where
    T: Clone + Send + Sync + 'static,
{
    Box::pin(async_stream::stream! {
        loop {
            let value = receiver.borrow_and_update().clone();
            yield Ok(value);
            if receiver.changed().await.is_err() {
                break;
            }
        }
    })
}

struct FrontendEvent {
    publisher_id: u64,
    sequence: u64,
    published_at: u64,
    received_at: Instant,
    frame: FrontendLoadFrame,
}

async fn run_frontend_subscriber(
    component: Component,
    sender: mpsc::Sender<FrontendEvent>,
    cancel: CancellationToken,
) {
    let namespace = component.namespace().clone();
    loop {
        let mut subscriber = tokio::select! {
            _ = cancel.cancelled() => return,
            result = EventSubscriber::for_namespace(&namespace, FRONTEND_LOAD_TOPIC) => match result {
                Ok(subscriber) => subscriber.typed::<FrontendLoadFrame>(),
                Err(error) => {
                    tracing::warn!(%error, "KV DC Relay frontend-load subscription failed");
                    tokio::select! {
                        _ = cancel.cancelled() => return,
                        _ = tokio::time::sleep(SNAPSHOT_INTERVAL) => continue,
                    }
                }
            },
        };

        loop {
            let event = tokio::select! {
                _ = cancel.cancelled() => return,
                event = subscriber.next() => event,
            };
            let Some(event) = event else {
                break;
            };
            match event {
                Ok((envelope, frame)) if frame.window_ms == FRONTEND_LOAD_WINDOW_MS => {
                    let event = FrontendEvent {
                        publisher_id: envelope.publisher_id,
                        sequence: envelope.sequence,
                        published_at: envelope.published_at,
                        received_at: Instant::now(),
                        frame,
                    };
                    if sender.send(event).await.is_err() {
                        return;
                    }
                }
                Ok((envelope, frame)) => tracing::warn!(
                    publisher_id = envelope.publisher_id,
                    window_ms = frame.window_ms,
                    "ignoring frontend load frame with unsupported window"
                ),
                Err(error) => {
                    tracing::warn!(%error, "KV DC Relay frontend-load stream failed; reconnecting");
                    break;
                }
            }
        }
    }
}

async fn run_aggregate_publisher(
    component: Component,
    identity: DcRelayIdentity,
    statuses: EndpointStatuses,
    pools: Arc<PoolRegistry>,
    mut frontend_rx: mpsc::Receiver<FrontendEvent>,
    snapshots: (
        watch::Sender<proto::KvUsageSnapshot>,
        watch::Sender<proto::LoadSnapshot>,
    ),
    cancel: CancellationToken,
) {
    let (usage_tx, load_tx) = snapshots;
    let mut frontends = HashMap::<u64, FrontendEvent>::new();
    let mut interval = tokio::time::interval(SNAPSHOT_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            event = frontend_rx.recv() => match event {
                Some(event) => record_frontend_event(&mut frontends, event),
                None => return,
            },
            _ = interval.tick() => {
                let (expected_publishers, discovery_complete) = expected_frontend_publishers(&component).await;
                retain_discovered_frontends(
                    &mut frontends,
                    &expected_publishers,
                    discovery_complete,
                );
                let catalog = pools.catalog();
                let worker_pools = collect_worker_pools(
                    &statuses,
                    &catalog,
                    pools.load_snapshots(),
                ).await;
                let now = Instant::now();
                let usage = build_usage_snapshot(identity, &worker_pools);
                let load = build_load_snapshot(
                    identity,
                    &catalog,
                    &worker_pools,
                    &frontends,
                    expected_publishers,
                    discovery_complete,
                    now,
                );
                usage_tx.send_replace(usage);
                load_tx.send_replace(load);
            }
        }
    }
}

fn record_frontend_event(frontends: &mut HashMap<u64, FrontendEvent>, event: FrontendEvent) {
    let frontend_id = event.frame.frontend_instance_id;
    if frontends.values().any(|current| {
        current.publisher_id == event.publisher_id && event.sequence <= current.sequence
    }) {
        return;
    }
    frontends.retain(|current_frontend, current| {
        *current_frontend == frontend_id || current.publisher_id != event.publisher_id
    });
    frontends.insert(frontend_id, event);
}

async fn expected_frontend_publishers(component: &Component) -> (HashSet<u64>, bool) {
    let query = DiscoveryQuery::EventChannels(EventChannelQuery::namespace_topic(
        component.namespace().name(),
        FRONTEND_LOAD_TOPIC,
    ));
    match component.drt().discovery().list(query).await {
        Ok(instances) => (
            instances
                .into_iter()
                .filter_map(|instance| match instance {
                    DiscoveryInstance::EventChannel { instance_id, .. } => Some(instance_id),
                    _ => None,
                })
                .collect(),
            true,
        ),
        Err(error) => {
            tracing::warn!(%error, "KV DC Relay could not reconcile frontend-load publishers");
            (HashSet::new(), false)
        }
    }
}

fn retain_discovered_frontends(
    frontends: &mut HashMap<u64, FrontendEvent>,
    expected_publishers: &HashSet<u64>,
    discovery_complete: bool,
) {
    if !discovery_complete {
        return;
    }
    frontends.retain(|_, event| expected_publishers.contains(&event.publisher_id));
}

struct WorkerPool {
    pool_id: PoolId,
    registrations: Vec<CanonicalModelRegistration>,
    role: WorkerType,
    block_size_tokens: u32,
    expected_ranks: u64,
    observed_ranks: u64,
    live_workers: Option<u64>,
    capacity_blocks: Option<u64>,
    used_blocks: Option<u64>,
    active_decode_blocks: Option<u64>,
    active_prefill_tokens: Option<u64>,
    max_concurrency: Option<u64>,
    complete: bool,
    observed_at_unix_ms: u64,
}

async fn collect_worker_pools(
    statuses: &EndpointStatuses,
    catalog: &DcPoolCatalog,
    load_snapshots: Vec<super::load::PoolLoadSnapshot>,
) -> Vec<WorkerPool> {
    let status_map = statuses.read().await.clone();
    let load_snapshots = load_snapshots
        .into_iter()
        .map(|snapshot| (snapshot.producer.pool_id(), snapshot))
        .collect::<HashMap<_, _>>();
    let mut result = Vec::with_capacity(catalog.pools().len());
    for descriptor in catalog.pools() {
        let Some(status) = status_map.get(descriptor.serving_endpoint()).cloned() else {
            continue;
        };
        let status = status.read().await;
        let Some(membership) = status.membership.clone() else {
            continue;
        };
        let role = endpoint_role(descriptor.pool_roles());
        let load_snapshot = load_snapshots
            .get(&descriptor.pool_id())
            .filter(|snapshot| snapshot.producer == descriptor.producer());
        let lifecycle_active = status.lifecycle == SlotLifecycle::Active;
        drop(status);

        let expected_ranks = load_snapshot.as_ref().map_or(0, |snapshot| {
            u64::try_from(snapshot.kv_expected_ranks).unwrap_or(u64::MAX)
        });
        let observed_ranks = load_snapshot.as_ref().map_or(0, |snapshot| {
            u64::try_from(snapshot.kv_observed_ranks).unwrap_or(u64::MAX)
        });
        let complete = lifecycle_active
            && load_snapshot
                .as_ref()
                .is_some_and(|snapshot| !snapshot.has_degraded_coverage());
        let (capacity_blocks, used_blocks) = if complete {
            load_snapshot
                .as_ref()
                .map(|snapshot| (snapshot.total_kv_blocks, snapshot.kv_used_blocks))
                .unwrap_or_default()
        } else {
            (None, None)
        };
        let max_concurrency = checked_max_concurrency(&membership.runtime_configs);
        result.push(WorkerPool {
            pool_id: descriptor.pool_id(),
            registrations: descriptor.registrations().to_vec(),
            role,
            block_size_tokens: descriptor.query_semantics().kv_block_size(),
            expected_ranks,
            observed_ranks,
            live_workers: complete
                .then(|| u64::try_from(membership.runtime_configs.len()).unwrap_or(u64::MAX)),
            capacity_blocks,
            used_blocks,
            active_decode_blocks: None,
            active_prefill_tokens: None,
            max_concurrency: complete.then_some(max_concurrency).flatten(),
            complete,
            observed_at_unix_ms: 0,
        });
    }
    result.sort_unstable_by_key(|pool| pool.pool_id);
    result
}

fn checked_max_concurrency(
    configs: &HashMap<u64, crate::local_model::runtime_config::ModelRuntimeConfig>,
) -> Option<u64> {
    configs.values().try_fold(0_u64, |total, config| {
        let per_rank = config.max_num_seqs?;
        total.checked_add(per_rank.checked_mul(u64::from(config.data_parallel_size))?)
    })
}

fn endpoint_role(roles: &[WorkerRole]) -> WorkerType {
    match roles {
        [WorkerRole::Prefill] => WorkerType::Prefill,
        [WorkerRole::Decode] => WorkerType::Decode,
        [WorkerRole::Encode] => WorkerType::Encode,
        _ => WorkerType::Aggregated,
    }
}

fn build_usage_snapshot(identity: DcRelayIdentity, pools: &[WorkerPool]) -> proto::KvUsageSnapshot {
    proto::KvUsageSnapshot {
        metadata: Some(relay_metadata(identity)),
        pools: pools
            .iter()
            .map(|pool| proto::PoolKvUsage {
                pool: Some(pool_identity(pool.pool_id)),
                models: pool.registrations.iter().map(model_registration).collect(),
                role: worker_role(pool.role),
                block_size_tokens: pool.block_size_tokens,
                expected_ranks: pool.expected_ranks,
                observed_ranks: pool.observed_ranks,
                capacity_blocks: pool.capacity_blocks,
                used_blocks: pool.used_blocks,
                status: data_status(pool.complete, pool.expected_ranks > 0),
                source_observed_at_unix_ms: pool.observed_at_unix_ms,
            })
            .collect(),
    }
}

fn build_load_snapshot(
    identity: DcRelayIdentity,
    catalog: &DcPoolCatalog,
    worker_pools: &[WorkerPool],
    frontends: &HashMap<u64, FrontendEvent>,
    mut expected_publishers: HashSet<u64>,
    discovery_complete: bool,
    now: Instant,
) -> proto::LoadSnapshot {
    for event in frontends.values() {
        expected_publishers.insert(event.publisher_id);
    }

    let pools = worker_pools
        .iter()
        .map(|pool| proto::PoolLoad {
            pool: Some(pool_identity(pool.pool_id)),
            role: worker_role(pool.role),
            live_workers: pool.live_workers,
            active_prefill_tokens: pool.active_prefill_tokens,
            active_decode_blocks: pool.active_decode_blocks,
            max_concurrency: pool.max_concurrency,
            scheduler_status: data_status(pool.complete, pool.expected_ranks > 0),
            scheduler_observed_at_unix_ms: pool.observed_at_unix_ms,
        })
        .collect();

    let mut registrations = BTreeMap::<String, ModelAggregate>::new();
    for descriptor in catalog.pools() {
        for registration in descriptor.registrations() {
            let aggregate = registrations
                .entry(registration.model().as_str().to_string())
                .or_insert_with(|| ModelAggregate::new(model_registration(registration)));
            aggregate.serving_pools.insert(descriptor.pool_id());
        }
    }
    for event in frontends.values() {
        for model in &event.frame.models {
            registrations
                .entry(model.model.clone())
                .or_insert_with(|| ModelAggregate::from_frontend(model));
        }
    }

    for aggregate in registrations.values_mut() {
        for event in frontends.values() {
            if !expected_publishers.contains(&event.publisher_id) {
                continue;
            }
            let Some(model) = event
                .frame
                .models
                .iter()
                .find(|model| model.model == aggregate.registration.model)
            else {
                continue;
            };
            aggregate.observe(event, model, now);
        }
    }

    let expected_frontends = u32::try_from(expected_publishers.len()).unwrap_or(u32::MAX);
    let models = registrations
        .into_values()
        .map(|aggregate| aggregate.finish(expected_frontends, discovery_complete))
        .collect();

    proto::LoadSnapshot {
        metadata: Some(relay_metadata(identity)),
        window_ms: FRONTEND_LOAD_WINDOW_MS,
        pools,
        models,
    }
}

struct ModelAggregate {
    registration: proto::ModelRegistration,
    serving_pools: HashSet<PoolId>,
    ready_frontends: u64,
    observed_frontends: u32,
    pending_first_output_requests: u64,
    pending_first_output_input_tokens: Option<u64>,
    live_input_tokens: Option<u64>,
    input_processing_requests: u64,
    output_generation_requests: u64,
    requests_started: u64,
    requests_completed: u64,
    requests_failed: u64,
    requests_cancelled: u64,
    input_tokens: Option<u64>,
    output_tokens: u64,
    source_observed_at_unix_ms: Option<u64>,
    overflowed: bool,
}

impl ModelAggregate {
    fn new(registration: proto::ModelRegistration) -> Self {
        Self {
            registration,
            serving_pools: HashSet::new(),
            ready_frontends: 0,
            observed_frontends: 0,
            pending_first_output_requests: 0,
            pending_first_output_input_tokens: Some(0),
            live_input_tokens: Some(0),
            input_processing_requests: 0,
            output_generation_requests: 0,
            requests_started: 0,
            requests_completed: 0,
            requests_failed: 0,
            requests_cancelled: 0,
            input_tokens: Some(0),
            output_tokens: 0,
            source_observed_at_unix_ms: None,
            overflowed: false,
        }
    }

    fn from_frontend(model: &FrontendModelLoad) -> Self {
        Self::new(proto::ModelRegistration {
            model: model.model.clone(),
            base_model: model.model.clone(),
            adapter: None,
            aliases: model.aliases.clone(),
        })
    }

    fn observe(&mut self, event: &FrontendEvent, model: &FrontendModelLoad, now: Instant) {
        if now.saturating_duration_since(event.received_at) > SOURCE_FRESHNESS {
            return;
        }
        self.observed_frontends = self.observed_frontends.saturating_add(1);
        self.ready_frontends = self
            .ready_frontends
            .saturating_add(u64::from(event.frame.serving_ready));
        self.add(
            model.pending_first_output_requests,
            Counter::PendingRequests,
        );
        add_optional_counter(
            &mut self.pending_first_output_input_tokens,
            model.pending_first_output_input_tokens,
            &mut self.overflowed,
        );
        add_optional_counter(
            &mut self.live_input_tokens,
            model.live_input_tokens,
            &mut self.overflowed,
        );
        self.add(model.input_processing_requests, Counter::InputRequests);
        self.add(model.output_generation_requests, Counter::OutputRequests);
        self.add(model.requests_started, Counter::Started);
        self.add(model.requests_completed, Counter::Completed);
        self.add(model.requests_failed, Counter::Failed);
        self.add(model.requests_cancelled, Counter::Cancelled);
        self.add(model.output_tokens, Counter::OutputTokens);
        self.input_tokens = match (self.input_tokens, model.input_tokens) {
            (Some(total), Some(value)) => total.checked_add(value),
            _ => None,
        };
        self.source_observed_at_unix_ms = Some(
            self.source_observed_at_unix_ms
                .map_or(event.published_at, |current| {
                    current.min(event.published_at)
                }),
        );
    }

    fn add(&mut self, value: u64, counter: Counter) {
        let target = match counter {
            Counter::PendingRequests => &mut self.pending_first_output_requests,
            Counter::InputRequests => &mut self.input_processing_requests,
            Counter::OutputRequests => &mut self.output_generation_requests,
            Counter::Started => &mut self.requests_started,
            Counter::Completed => &mut self.requests_completed,
            Counter::Failed => &mut self.requests_failed,
            Counter::Cancelled => &mut self.requests_cancelled,
            Counter::OutputTokens => &mut self.output_tokens,
        };
        if let Some(total) = target.checked_add(value) {
            *target = total;
        } else {
            self.overflowed = true;
        }
    }

    fn finish(mut self, expected_frontends: u32, discovery_complete: bool) -> proto::ModelLoad {
        let complete = discovery_complete
            && expected_frontends > 0
            && self.observed_frontends == expected_frontends
            && self
                .source_observed_at_unix_ms
                .is_some_and(|timestamp| timestamp > 0)
            && !self.overflowed;
        let mut serving_pools = self.serving_pools.drain().collect::<Vec<_>>();
        serving_pools.sort_unstable();
        proto::ModelLoad {
            model: Some(self.registration),
            ready_frontends: complete.then_some(self.ready_frontends),
            pending_first_output_requests: complete.then_some(self.pending_first_output_requests),
            pending_first_output_input_tokens: complete
                .then_some(self.pending_first_output_input_tokens)
                .flatten(),
            live_input_tokens: complete.then_some(self.live_input_tokens).flatten(),
            input_processing_requests: complete.then_some(self.input_processing_requests),
            output_generation_requests: complete.then_some(self.output_generation_requests),
            serving_pools: serving_pools.into_iter().map(pool_identity).collect(),
            requests_started: self.requests_started,
            requests_completed: self.requests_completed,
            requests_failed: self.requests_failed,
            requests_cancelled: self.requests_cancelled,
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            status: data_status(complete, self.observed_frontends > 0),
            expected_frontends,
            observed_frontends: self.observed_frontends,
            source_observed_at_unix_ms: self.source_observed_at_unix_ms.unwrap_or_default(),
        }
    }
}

enum Counter {
    PendingRequests,
    InputRequests,
    OutputRequests,
    Started,
    Completed,
    Failed,
    Cancelled,
    OutputTokens,
}

fn add_optional_counter(total: &mut Option<u64>, value: Option<u64>, overflowed: &mut bool) {
    *total = match (*total, value) {
        (Some(total), Some(value)) => match total.checked_add(value) {
            Some(total) => Some(total),
            None => {
                *overflowed = true;
                None
            }
        },
        _ => None,
    };
}

enum PoolEvent {
    Delta(PoolId, DcCkfDelta),
    Closed(PoolId, Status),
}

struct CkfPoolStream {
    identity: ProducerIdentity,
    endpoint: dynamo_runtime::protocols::EndpointId,
    task: JoinHandle<()>,
    capacity_omissions: u64,
    sequence: u64,
    fenced: bool,
}

impl Drop for CkfPoolStream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn run_ckf_stream(
    relay_identity: DcRelayIdentity,
    pools: Arc<PoolRegistry>,
    statuses: EndpointStatuses,
    output: mpsc::Sender<Result<proto::KvCuckooFilterUpdate, Status>>,
) -> Result<(), Status> {
    let mut catalog_rx = pools.watch_catalog();
    let (event_tx, mut event_rx) = mpsc::channel(CKF_STREAM_CAPACITY);
    let mut active = HashMap::<PoolId, CkfPoolStream>::new();

    loop {
        let catalog = catalog_rx.borrow_and_update().clone();
        reconcile_ckf_catalog(
            relay_identity,
            &pools,
            &statuses,
            &output,
            &event_tx,
            &mut active,
            &catalog,
        )
        .await?;
        if !catalog_rx
            .has_changed()
            .map_err(|_| Status::unavailable("Relay catalog closed"))?
        {
            send_ckf(
                &output,
                relay_identity,
                proto::kv_cuckoo_filter_update::Update::Heartbeat(proto::CuckooStreamHeartbeat {
                    catalog_revision: catalog.revision(),
                    initial_sync_complete: true,
                }),
            )
            .await?;
            break;
        }
        catalog_rx
            .changed()
            .await
            .map_err(|_| Status::unavailable("Relay catalog closed"))?;
    }

    let mut heartbeat = tokio::time::interval(SNAPSHOT_INTERVAL);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            changed = catalog_rx.changed() => {
                changed.map_err(|_| Status::unavailable("Relay catalog closed"))?;
                let catalog = catalog_rx.borrow_and_update().clone();
                reconcile_ckf_catalog(
                    relay_identity,
                    &pools,
                    &statuses,
                    &output,
                    &event_tx,
                    &mut active,
                    &catalog,
                ).await?;
            }
            event = event_rx.recv() => match event {
                Some(PoolEvent::Delta(pool_id, delta)) => {
                    let Some(stream) = active.get_mut(&pool_id) else { continue; };
                    if delta.identity() != stream.identity {
                        continue;
                    }
                    let Some(handle) = pools.active_handle(pool_id) else { continue; };
                    let (stats, sequence, members) = handle.state_stats().await
                        .map_err(|error| Status::unavailable(error.to_string()))?;
                    let omissions = stats.aggregation().capacity_failures();
                    let expected_ranks = expected_ranks(&statuses, &stream.endpoint).await;
                    let materialized_ranks = u64::try_from(members.len()).unwrap_or(u64::MAX);
                    let complete = omissions == 0
                        && expected_ranks > 0
                        && materialized_ranks == expected_ranks;
                    if omissions > stream.capacity_omissions || (!complete && !stream.fenced) {
                        stream.capacity_omissions = omissions;
                        stream.fenced = true;
                        send_ckf(
                            &output,
                            relay_identity,
                            proto::kv_cuckoo_filter_update::Update::Stats(ckf_stats(
                                stream.identity,
                                sequence,
                                &members,
                                stats.aggregation().unique_block_count(),
                                omissions,
                                complete,
                            )),
                        ).await?;
                    }
                    if delta.sequence() <= stream.sequence {
                        continue;
                    }
                    if delta.base_sequence() != stream.sequence {
                        return Err(Status::data_loss("Relay CKF sequence gap"));
                    }
                    stream.sequence = delta.sequence();
                    if complete && !stream.fenced {
                        send_ckf(
                            &output,
                            relay_identity,
                            proto::kv_cuckoo_filter_update::Update::Delta(
                                ckf_delta(delta).map_err(Status::resource_exhausted)?,
                            ),
                        ).await?;
                    }
                }
                Some(PoolEvent::Closed(pool_id, error)) if active.contains_key(&pool_id) => return Err(error),
                Some(PoolEvent::Closed(_, _)) => {}
                None => return Err(Status::unavailable("Relay CKF publication task closed")),
            },
            _ = heartbeat.tick() => {
                refresh_ckf_pools(
                    relay_identity,
                    &pools,
                    &statuses,
                    &output,
                    &event_tx,
                    &mut active,
                ).await?;
                let revision = catalog_rx.borrow().revision();
                send_ckf(
                    &output,
                    relay_identity,
                    proto::kv_cuckoo_filter_update::Update::Heartbeat(proto::CuckooStreamHeartbeat {
                        catalog_revision: revision,
                        initial_sync_complete: true,
                    }),
                ).await?;
            }
        }
    }
}

async fn reconcile_ckf_catalog(
    relay_identity: DcRelayIdentity,
    pools: &Arc<PoolRegistry>,
    statuses: &EndpointStatuses,
    output: &mpsc::Sender<Result<proto::KvCuckooFilterUpdate, Status>>,
    event_tx: &mpsc::Sender<PoolEvent>,
    active: &mut HashMap<PoolId, CkfPoolStream>,
    catalog: &DcPoolCatalog,
) -> Result<(), Status> {
    let desired = catalog
        .pools()
        .iter()
        .map(|descriptor| (descriptor.pool_id(), descriptor.producer()))
        .collect::<HashMap<_, _>>();
    let retired = active
        .iter()
        .filter_map(|(pool_id, stream)| {
            (desired.get(pool_id) != Some(&stream.identity)).then_some((*pool_id, stream.identity))
        })
        .collect::<Vec<_>>();
    for (pool_id, identity) in retired {
        if let Some(stream) = active.remove(&pool_id) {
            stream.task.abort();
        }
        send_ckf(
            output,
            relay_identity,
            proto::kv_cuckoo_filter_update::Update::Retired(proto::CuckooPoolRetired {
                pool: Some(pool_identity(pool_id)),
                producer_incarnation: identity.producer_incarnation(),
                layout_generation: identity.layout_generation(),
            }),
        )
        .await?;
    }

    for descriptor in catalog.pools() {
        if active.contains_key(&descriptor.pool_id()) {
            continue;
        }
        let handle = pools
            .active_handle(descriptor.pool_id())
            .ok_or_else(|| Status::unavailable("Relay pool disappeared during CKF sync"))?;
        let subscription = subscribe_ckf(&handle).await?;
        let DcCkfSubscription {
            snapshot,
            deltas,
            stats,
            members,
        } = subscription;
        let capacity_omissions = stats.aggregation().capacity_failures();
        let snapshot_sequence = snapshot.sequence();
        let expected_ranks = expected_ranks(statuses, descriptor.serving_endpoint()).await;
        let complete = capacity_omissions == 0
            && expected_ranks > 0
            && u64::try_from(members.len()).ok() == Some(expected_ranks);
        let snapshot = ckf_snapshot(
            snapshot,
            &members,
            stats.aggregation().unique_block_count(),
            capacity_omissions,
            complete,
        )
        .map_err(Status::resource_exhausted)?;
        let identity = descriptor.producer();
        send_ckf(
            output,
            relay_identity,
            proto::kv_cuckoo_filter_update::Update::Snapshot(snapshot),
        )
        .await?;
        let task = spawn_delta_forwarder(descriptor.pool_id(), deltas, event_tx.clone());
        active.insert(
            descriptor.pool_id(),
            CkfPoolStream {
                identity,
                endpoint: descriptor.serving_endpoint().clone(),
                task,
                capacity_omissions,
                sequence: snapshot_sequence,
                fenced: !complete,
            },
        );
    }
    Ok(())
}

async fn refresh_ckf_pools(
    relay_identity: DcRelayIdentity,
    pools: &Arc<PoolRegistry>,
    statuses: &EndpointStatuses,
    output: &mpsc::Sender<Result<proto::KvCuckooFilterUpdate, Status>>,
    event_tx: &mpsc::Sender<PoolEvent>,
    active: &mut HashMap<PoolId, CkfPoolStream>,
) -> Result<(), Status> {
    let pool_ids = active.keys().copied().collect::<Vec<_>>();
    for pool_id in pool_ids {
        let Some(stream) = active.get(&pool_id) else {
            continue;
        };
        let endpoint = stream.endpoint.clone();
        let was_fenced = stream.fenced;
        let prior_omissions = stream.capacity_omissions;
        let expected = expected_ranks(statuses, &endpoint).await;
        let Some(handle) = pools.active_handle(pool_id) else {
            continue;
        };
        let (stats, sequence, members) = handle
            .state_stats()
            .await
            .map_err(|error| Status::unavailable(error.to_string()))?;
        let omissions = stats.aggregation().capacity_failures();
        let complete =
            omissions == 0 && expected > 0 && u64::try_from(members.len()).ok() == Some(expected);
        if !complete {
            if omissions > prior_omissions || !was_fenced {
                send_ckf(
                    output,
                    relay_identity,
                    proto::kv_cuckoo_filter_update::Update::Stats(ckf_stats(
                        handle.identity(),
                        sequence,
                        &members,
                        stats.aggregation().unique_block_count(),
                        omissions,
                        false,
                    )),
                )
                .await?;
            }
            if let Some(stream) = active.get_mut(&pool_id) {
                stream.capacity_omissions = omissions;
                stream.fenced = true;
            }
            continue;
        }
        if !was_fenced {
            continue;
        }

        let DcCkfSubscription {
            snapshot,
            deltas,
            stats,
            members,
        } = subscribe_ckf(&handle).await?;
        let omissions = stats.aggregation().capacity_failures();
        let complete =
            omissions == 0 && expected > 0 && u64::try_from(members.len()).ok() == Some(expected);
        if !complete {
            continue;
        }
        let identity = snapshot.identity();
        let sequence = snapshot.sequence();
        let snapshot = ckf_snapshot(
            snapshot,
            &members,
            stats.aggregation().unique_block_count(),
            omissions,
            true,
        )
        .map_err(Status::resource_exhausted)?;
        send_ckf(
            output,
            relay_identity,
            proto::kv_cuckoo_filter_update::Update::Snapshot(snapshot),
        )
        .await?;
        let task = spawn_delta_forwarder(pool_id, deltas, event_tx.clone());
        let Some(stream) = active.get_mut(&pool_id) else {
            task.abort();
            continue;
        };
        stream.task.abort();
        stream.identity = identity;
        stream.task = task;
        stream.capacity_omissions = omissions;
        stream.sequence = sequence;
        stream.fenced = false;
    }
    Ok(())
}

async fn expected_ranks(
    statuses: &EndpointStatuses,
    endpoint: &dynamo_runtime::protocols::EndpointId,
) -> u64 {
    let Some(status) = statuses.read().await.get(endpoint).cloned() else {
        return 0;
    };
    let status = status.read().await;
    status
        .membership
        .as_ref()
        .and_then(|membership| {
            membership
                .runtime_configs
                .values()
                .try_fold(0_u64, |total, config| {
                    total.checked_add(u64::from(config.data_parallel_size))
                })
        })
        .unwrap_or_default()
}

async fn subscribe_ckf(handle: &KvDcRelayHandle) -> Result<DcCkfSubscription, Status> {
    let identity = handle.identity();
    let lease = LaneLease::new(
        ConsumerInstanceId::new(identity.producer_incarnation()),
        0,
        identity.layout_generation(),
    );
    handle
        .subscribe(lease)
        .await
        .map_err(|error| Status::unavailable(error.to_string()))
}

fn spawn_delta_forwarder(
    pool_id: PoolId,
    mut deltas: tokio::sync::broadcast::Receiver<DcCkfDelta>,
    sender: mpsc::Sender<PoolEvent>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match deltas.recv().await {
                Ok(delta) => {
                    if sender.send(PoolEvent::Delta(pool_id, delta)).await.is_err() {
                        return;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    let _ = sender
                        .send(PoolEvent::Closed(
                            pool_id,
                            Status::resource_exhausted("CKF consumer lagged; reconnect required"),
                        ))
                        .await;
                    return;
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                    let _ = sender
                        .send(PoolEvent::Closed(
                            pool_id,
                            Status::unavailable("Relay CKF pool closed"),
                        ))
                        .await;
                    return;
                }
            }
        }
    })
}

async fn send_ckf(
    output: &mpsc::Sender<Result<proto::KvCuckooFilterUpdate, Status>>,
    identity: DcRelayIdentity,
    update: proto::kv_cuckoo_filter_update::Update,
) -> Result<(), Status> {
    let message = proto::KvCuckooFilterUpdate {
        metadata: Some(relay_metadata(identity)),
        update: Some(update),
    };
    if message.encoded_len() > MAX_CKF_MESSAGE_BYTES {
        return Err(Status::resource_exhausted(
            "encoded CKF update exceeds the v1 64 MiB limit",
        ));
    }
    output
        .send(Ok(message))
        .await
        .map_err(|_| Status::cancelled("CKF consumer disconnected"))
}

fn ckf_snapshot(
    snapshot: dynamo_kv_router::indexer::cuckoo::DcCkfSnapshot,
    members: &[(dynamo_kv_router::protocols::WorkerWithDpRank, usize)],
    unique_blocks: usize,
    capacity_omissions: u64,
    complete: bool,
) -> Result<proto::CuckooPoolSnapshot, &'static str> {
    let encoded_len = snapshot
        .buckets()
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or("CKF snapshot size overflow")?;
    if encoded_len > MAX_CKF_MESSAGE_BYTES {
        return Err("CKF snapshot exceeds the v1 64 MiB limit");
    }
    let mut packed_buckets = Vec::with_capacity(encoded_len);
    for bucket in snapshot.buckets() {
        packed_buckets.extend_from_slice(&bucket.to_le_bytes());
    }
    Ok(proto::CuckooPoolSnapshot {
        producer: Some(ckf_producer(snapshot.identity())),
        sequence: snapshot.sequence(),
        packed_buckets,
        status: data_status(complete, !members.is_empty()),
        stats: Some(ckf_stats(
            snapshot.identity(),
            snapshot.sequence(),
            members,
            unique_blocks,
            capacity_omissions,
            complete,
        )),
    })
}

fn ckf_delta(delta: DcCkfDelta) -> Result<proto::CuckooPoolDelta, &'static str> {
    let buckets = delta
        .images()
        .iter()
        .map(|image| {
            Ok(proto::CuckooBucketImage {
                bucket_index: u64::try_from(image.bucket())
                    .map_err(|_| "CKF bucket index exceeds u64")?,
                packed_bucket: image.value(),
            })
        })
        .collect::<Result<Vec<_>, &'static str>>()?;
    Ok(proto::CuckooPoolDelta {
        producer: Some(ckf_producer(delta.identity())),
        base_sequence: delta.base_sequence(),
        sequence: delta.sequence(),
        buckets,
    })
}

fn ckf_stats(
    identity: ProducerIdentity,
    sequence: u64,
    members: &[(dynamo_kv_router::protocols::WorkerWithDpRank, usize)],
    unique_blocks: usize,
    capacity_omissions: u64,
    complete: bool,
) -> proto::CuckooPoolStats {
    proto::CuckooPoolStats {
        producer: Some(ckf_producer(identity)),
        publication_sequence: sequence,
        materialized_ranks: u64::try_from(members.len()).unwrap_or(u64::MAX),
        unique_blocks: u64::try_from(unique_blocks).unwrap_or(u64::MAX),
        status: data_status(complete, !members.is_empty()),
        capacity_omissions,
        source_observed_at_unix_ms: unix_ms(),
    }
}

fn ckf_producer(identity: ProducerIdentity) -> proto::CuckooProducerIdentity {
    let format = identity.format();
    proto::CuckooProducerIdentity {
        pool: Some(pool_identity(identity.pool_id())),
        producer_incarnation: identity.producer_incarnation(),
        layout_generation: identity.layout_generation(),
        format: Some(proto::CuckooFormat {
            format_version: u32::from(format.format_version()),
            seed: format.seed(),
            bucket_count: u64::try_from(format.bucket_count()).unwrap_or(u64::MAX),
            fingerprint_bits: u32::from(format.fingerprint_bits()),
            slots_per_bucket: u32::from(format.slots_per_bucket()),
        }),
    }
}

fn relay_metadata(identity: DcRelayIdentity) -> proto::RelayMessageMetadata {
    proto::RelayMessageMetadata {
        drt_instance_id: identity.drt_instance_id(),
        relay_incarnation: identity.relay_incarnation(),
        observed_at_unix_ms: unix_ms(),
    }
}

fn pool_identity(pool_id: PoolId) -> proto::PoolIdentity {
    let domain = pool_id.indexer_domain();
    proto::PoolIdentity {
        cache_semantics_digest: domain.cache_semantics().digest().to_vec(),
        cache_semantics_source: identity_source(domain.cache_semantics().source()),
        routing_scope_digest: domain.routing_scope().digest().to_vec(),
        routing_scope_source: identity_source(domain.routing_scope().source()),
        dc_id: pool_id.dc_id().get(),
    }
}

fn identity_source(source: DynamoIdentitySource) -> i32 {
    match source {
        DynamoIdentitySource::DefaultDerived => proto::IdentitySource::DefaultDerived as i32,
        DynamoIdentitySource::Explicit => proto::IdentitySource::Explicit as i32,
    }
}

fn model_registration(registration: &CanonicalModelRegistration) -> proto::ModelRegistration {
    let (base_model, adapter) = match registration.target() {
        ModelTarget::Base { base_model } => (base_model.as_str().to_string(), None),
        ModelTarget::Lora {
            base_model,
            adapter,
        } => (
            base_model.as_str().to_string(),
            Some(adapter.as_str().to_string()),
        ),
    };
    proto::ModelRegistration {
        model: registration.model().as_str().to_string(),
        base_model,
        adapter,
        aliases: registration
            .aliases()
            .iter()
            .map(|alias| alias.as_str().to_string())
            .collect(),
    }
}

fn worker_role(role: WorkerType) -> i32 {
    match role {
        WorkerType::Aggregated => proto::WorkerRole::Aggregated as i32,
        WorkerType::Prefill => proto::WorkerRole::Prefill as i32,
        WorkerType::Decode => proto::WorkerRole::Decode as i32,
        WorkerType::Encode => proto::WorkerRole::Encode as i32,
    }
}

fn data_status(complete: bool, partial: bool) -> i32 {
    if complete {
        proto::DataStatus::Complete as i32
    } else if partial {
        proto::DataStatus::Degraded as i32
    } else {
        proto::DataStatus::Unavailable as i32
    }
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| u64::try_from(duration.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, RoutingScopeId,
    };

    use super::*;

    fn frontend_event(
        publisher_id: u64,
        sequence: u64,
        frontend_instance_id: u64,
        model: FrontendModelLoad,
    ) -> FrontendEvent {
        FrontendEvent {
            publisher_id,
            sequence,
            published_at: sequence,
            received_at: Instant::now(),
            frame: FrontendLoadFrame {
                frontend_instance_id,
                serving_ready: true,
                window_ms: FRONTEND_LOAD_WINDOW_MS,
                models: vec![model],
            },
        }
    }

    fn model_load(model: &str, pending: u64) -> FrontendModelLoad {
        FrontendModelLoad {
            model: model.to_string(),
            pending_first_output_requests: pending,
            pending_first_output_input_tokens: Some(17),
            live_input_tokens: Some(31),
            input_tokens: Some(11),
            output_tokens: 7,
            ..Default::default()
        }
    }

    fn registration(model: &str) -> proto::ModelRegistration {
        proto::ModelRegistration {
            model: model.to_string(),
            base_model: model.to_string(),
            adapter: None,
            aliases: Vec::new(),
        }
    }

    fn pool_id() -> PoolId {
        PoolId::new(
            IndexerDomainId::new(
                CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                RoutingScopeId::new([2; 16], IdentitySource::DefaultDerived),
            ),
            DcId::new(3),
        )
    }

    #[test]
    fn grpc_listener_accepts_only_loopback_addresses() {
        assert!(validate_listen_address("127.0.0.1:50051".parse().unwrap()).is_ok());
        assert!(validate_listen_address("[::1]:50051".parse().unwrap()).is_ok());
        assert!(validate_listen_address("0.0.0.0:50051".parse().unwrap()).is_err());
        assert!(validate_listen_address("192.0.2.10:50051".parse().unwrap()).is_err());
    }

    #[test]
    fn frontend_frames_replace_by_publisher_and_ignore_stale_sequences() {
        let mut frontends = HashMap::new();
        record_frontend_event(&mut frontends, frontend_event(1, 2, 10, model_load("a", 2)));
        record_frontend_event(&mut frontends, frontend_event(1, 1, 10, model_load("a", 9)));
        assert_eq!(
            frontends[&10].frame.models[0].pending_first_output_requests,
            2
        );

        record_frontend_event(&mut frontends, frontend_event(1, 3, 20, model_load("a", 3)));
        assert!(!frontends.contains_key(&10));
        assert_eq!(frontends[&20].publisher_id, 1);

        record_frontend_event(&mut frontends, frontend_event(2, 1, 20, model_load("a", 4)));
        retain_discovered_frontends(&mut frontends, &HashSet::from([2]), true);
        assert_eq!(frontends.len(), 1);
        assert_eq!(frontends[&20].publisher_id, 2);
    }

    #[test]
    fn failed_discovery_does_not_discard_last_known_frontend_frame() {
        let mut frontends = HashMap::from([(10, frontend_event(1, 1, 10, model_load("a", 1)))]);
        retain_discovered_frontends(&mut frontends, &HashSet::new(), false);
        assert_eq!(frontends.len(), 1);

        retain_discovered_frontends(&mut frontends, &HashSet::new(), true);
        assert!(frontends.is_empty());
    }

    #[test]
    fn model_load_is_complete_only_with_every_expected_fresh_frontend() {
        let mut aggregate = ModelAggregate::new(registration("a"));
        let event = frontend_event(1, 1, 10, model_load("a", 2));
        aggregate.observe(&event, &event.frame.models[0], Instant::now());
        let complete = aggregate.finish(1, true);
        assert_eq!(complete.status, proto::DataStatus::Complete as i32);
        assert_eq!(complete.ready_frontends, Some(1));
        assert_eq!(complete.pending_first_output_requests, Some(2));
        assert_eq!(complete.pending_first_output_input_tokens, Some(17));
        assert_eq!(complete.live_input_tokens, Some(31));
        assert_eq!(complete.input_tokens, Some(11));
        assert_eq!(complete.output_tokens, 7);
        assert_eq!(complete.source_observed_at_unix_ms, 1);

        let mut aggregate = ModelAggregate::new(registration("a"));
        aggregate.observe(&event, &event.frame.models[0], Instant::now());
        let incomplete = aggregate.finish(2, true);
        assert_eq!(incomplete.status, proto::DataStatus::Degraded as i32);
        assert_eq!(incomplete.ready_frontends, None);
        assert_eq!(incomplete.observed_frontends, 1);
    }

    #[test]
    fn unknown_exact_input_gauges_remain_optional_in_complete_load() {
        let mut aggregate = ModelAggregate::new(registration("a"));
        let mut load = model_load("a", 2);
        load.pending_first_output_input_tokens = None;
        load.live_input_tokens = None;
        let event = frontend_event(1, 1, 10, load);
        aggregate.observe(&event, &event.frame.models[0], Instant::now());

        let model = aggregate.finish(1, true);

        assert_eq!(model.status, proto::DataStatus::Complete as i32);
        assert_eq!(model.pending_first_output_input_tokens, None);
        assert_eq!(model.live_input_tokens, None);
    }

    #[test]
    fn model_load_uses_oldest_contributing_source_timestamp() {
        let mut aggregate = ModelAggregate::new(registration("a"));
        let older = frontend_event(1, 10, 10, model_load("a", 1));
        let newer = frontend_event(2, 20, 20, model_load("a", 2));
        aggregate.observe(&older, &older.frame.models[0], Instant::now());
        aggregate.observe(&newer, &newer.frame.models[0], Instant::now());

        let model = aggregate.finish(2, true);

        assert_eq!(model.status, proto::DataStatus::Complete as i32);
        assert_eq!(model.source_observed_at_unix_ms, 10);
    }

    #[test]
    fn stale_frontend_frame_makes_relay_only_model_unavailable() {
        let mut aggregate = ModelAggregate::new(registration("a"));
        let mut event = frontend_event(1, 1, 10, model_load("a", 2));
        event.received_at = Instant::now() - SOURCE_FRESHNESS - Duration::from_millis(1);
        aggregate.observe(&event, &event.frame.models[0], Instant::now());
        let model = aggregate.finish(1, true);
        assert_eq!(model.status, proto::DataStatus::Unavailable as i32);
        assert_eq!(model.observed_frontends, 0);
    }

    #[test]
    fn pool_identity_preserves_structural_digest_sources() {
        let identity = pool_identity(pool_id());
        assert_eq!(identity.cache_semantics_digest, vec![1; 16]);
        assert_eq!(
            identity.cache_semantics_source,
            proto::IdentitySource::Explicit as i32
        );
        assert_eq!(identity.routing_scope_digest, vec![2; 16]);
        assert_eq!(
            identity.routing_scope_source,
            proto::IdentitySource::DefaultDerived as i32
        );
        assert_eq!(identity.dc_id, 3);
    }
}
