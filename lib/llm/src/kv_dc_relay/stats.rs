// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, HashMap, HashSet};
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dynamo_kv_router::identity::{IdentitySource as DynamoIdentitySource, PoolId};
use dynamo_kv_router::indexer::cuckoo::ProducerIdentity;
use dynamo_runtime::component::Component;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryQuery, EventChannelQuery};
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::transports::event_plane::EventSubscriber;
use futures::{Stream, StreamExt};
use prost::Message;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinHandle, JoinSet};
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

use super::host::{HostTerminalState, SharedEndpointStatus, SlotLifecycle, record_host_failure};
use super::identity::{
    CanonicalModelRegistration, DcPoolCatalog, DcRelayIdentity, ModelTarget, WorkerRole,
};
use super::load::PoolLoadSnapshot;
use super::pool_registry::PoolRegistry;
use super::publication::{
    PoolPublicationStream, PublicationError, PublicationErrorKind, PublicationFrame,
    PublicationFrameKind, RelayPublicationSource,
};
use crate::frontend_load::{FRONTEND_LOAD_TOPIC, FrontendLoadFrame, FrontendModelLoad};
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
const FRONTEND_EVENT_CAPACITY: usize = 64;
// One native delta can approach the wire limit. Backpressure here is preferable
// to retaining another per-client backlog of large delta buffers.
const CKF_POOL_EVENT_CAPACITY: usize = 1;
const CKF_OUTPUT_CAPACITY: usize = 1;

pub(super) struct RelayStatsRuntime {
    cancel: CancellationToken,
    supervisor: JoinHandle<()>,
}

impl RelayStatsRuntime {
    pub(super) async fn start(
        component: Component,
        statuses: EndpointStatuses,
        pools: Arc<PoolRegistry>,
        publication_source: Arc<dyn RelayPublicationSource>,
        listen_address: SocketAddr,
        fatal_cancel: CancellationToken,
        terminal: Arc<HostTerminalState>,
    ) -> anyhow::Result<Self> {
        let identity = publication_source.relay_identity();
        validate_listen_address(listen_address)?;
        let listener = tokio::net::TcpListener::bind(listen_address).await?;
        let cancel = fatal_cancel.child_token();
        let metadata = relay_metadata(identity);
        let (usage_tx, usage_rx) = watch::channel(proto::KvUsageSnapshot {
            metadata: Some(metadata),
            pools: Vec::new(),
        });
        let (load_tx, load_rx) = watch::channel(proto::LoadSnapshot {
            metadata: Some(metadata),
            pools: Vec::new(),
            models: Vec::new(),
        });
        let (frontend_tx, frontend_rx) = mpsc::channel(FRONTEND_EVENT_CAPACITY);

        let frontend_component = component.clone();
        let frontend_cancel = cancel.child_token();
        let mut tasks = JoinSet::new();
        tasks.spawn(async move {
            (
                "frontend subscriber",
                run_frontend_subscriber(frontend_component, frontend_tx, frontend_cancel).await,
            )
        });

        let aggregate_component = component.clone();
        let aggregate_statuses = statuses.clone();
        let aggregate_cancel = cancel.child_token();
        let aggregate_pools = pools.clone();
        tasks.spawn(async move {
            (
                "aggregate publisher",
                run_aggregate_publisher(
                    aggregate_component,
                    identity,
                    aggregate_statuses,
                    aggregate_pools,
                    frontend_rx,
                    (usage_tx, load_tx),
                    aggregate_cancel,
                )
                .await,
            )
        });

        let service = RelayStatsService {
            pools,
            publication_source,
            statuses,
            usage: usage_rx,
            load: load_rx,
            cancel: cancel.child_token(),
        };
        let server_cancel = cancel.child_token();
        tasks.spawn(async move {
            (
                "gRPC server",
                tonic::transport::Server::builder()
                    .add_service(
                        KvDcRelayServer::new(service)
                            .max_encoding_message_size(MAX_CKF_MESSAGE_BYTES),
                    )
                    .serve_with_incoming_shutdown(
                        TcpListenerStream::new(listener),
                        server_cancel.cancelled_owned(),
                    )
                    .await
                    .map_err(anyhow::Error::from),
            )
        });

        let runtime_cancel = cancel.clone();
        let supervisor = tokio::spawn(supervise_stats_tasks(tasks, cancel, fatal_cancel, terminal));

        Ok(Self {
            cancel: runtime_cancel,
            supervisor,
        })
    }

    pub(super) async fn shutdown(self) {
        self.cancel.cancel();
        if let Err(error) = self.supervisor.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV DC Relay stats supervisor failed during shutdown");
        }
    }
}

async fn supervise_stats_tasks(
    mut tasks: JoinSet<(&'static str, anyhow::Result<()>)>,
    cancel: CancellationToken,
    fatal_cancel: CancellationToken,
    terminal: Arc<HostTerminalState>,
) {
    let failure = tokio::select! {
        _ = cancel.cancelled() => None,
        result = tasks.join_next() => {
            if cancel.is_cancelled() {
                None
            } else {
                Some(match result {
                    Some(Ok((task, Ok(())))) => {
                        format!("KV DC Relay stats {task} stopped unexpectedly")
                    }
                    Some(Ok((task, Err(error)))) => {
                        format!("KV DC Relay stats {task} failed: {error}")
                    }
                    Some(Err(error)) => format!("KV DC Relay stats task failed: {error}"),
                    None => "KV DC Relay stats supervisor lost all tasks".to_string(),
                })
            }
        }
    };
    if let Some(reason) = failure {
        record_host_failure(&fatal_cancel, &terminal, reason);
    }
    cancel.cancel();
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV DC Relay stats task failed during shutdown");
        }
    }
}

fn validate_listen_address(address: SocketAddr) -> anyhow::Result<()> {
    anyhow::ensure!(
        address.ip().is_loopback(),
        "KV DC Relay gRPC must bind to a loopback address"
    );
    Ok(())
}

#[derive(Clone)]
struct RelayStatsService {
    pools: Arc<PoolRegistry>,
    publication_source: Arc<dyn RelayPublicationSource>,
    statuses: EndpointStatuses,
    usage: watch::Receiver<proto::KvUsageSnapshot>,
    load: watch::Receiver<proto::LoadSnapshot>,
    cancel: CancellationToken,
}

type ResponseStream<T> = Pin<Box<dyn Stream<Item = Result<T, Status>> + Send + 'static>>;

struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

#[tonic::async_trait]
impl KvDcRelay for RelayStatsService {
    type WatchKvCuckooFilterStream = ResponseStream<proto::KvCuckooFilterUpdate>;
    type WatchKvUsageStream = ResponseStream<proto::KvUsageSnapshot>;
    type WatchLoadStream = ResponseStream<proto::LoadSnapshot>;

    async fn watch_kv_cuckoo_filter(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::WatchKvCuckooFilterStream>, Status> {
        let (sender, receiver) = mpsc::channel(CKF_OUTPUT_CAPACITY);
        let pools = self.pools.clone();
        let publication_source = self.publication_source.clone();
        let statuses = self.statuses.clone();
        let cancel = self.cancel.child_token();
        let task_cancel = cancel.clone();
        tokio::spawn(async move {
            tokio::select! {
                _ = task_cancel.cancelled() => {}
                result = run_ckf_stream(
                    pools,
                    publication_source,
                    statuses,
                    sender.clone(),
                ) => {
                    if let Err(error) = result {
                        tokio::select! {
                            _ = task_cancel.cancelled() => {}
                            _ = sender.send(Err(error)) => {}
                        }
                    }
                }
            }
        });
        Ok(Response::new(ckf_response_stream(receiver, cancel)))
    }

    async fn watch_kv_usage(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::WatchKvUsageStream>, Status> {
        Ok(Response::new(watch_stream(
            self.usage.clone(),
            self.cancel.child_token(),
        )))
    }

    async fn watch_load(
        &self,
        _request: Request<()>,
    ) -> Result<Response<Self::WatchLoadStream>, Status> {
        Ok(Response::new(watch_stream(
            self.load.clone(),
            self.cancel.child_token(),
        )))
    }
}

fn ckf_response_stream<T>(
    mut receiver: mpsc::Receiver<Result<T, Status>>,
    cancel: CancellationToken,
) -> ResponseStream<T>
where
    T: Send + 'static,
{
    let cancel = CancelOnDrop(cancel);
    Box::pin(async_stream::stream! {
        loop {
            let item = tokio::select! {
                biased;
                _ = cancel.0.cancelled() => break,
                item = receiver.recv() => item,
            };
            let Some(item) = item else {
                break;
            };
            yield item;
        }
    })
}

fn watch_stream<T>(mut receiver: watch::Receiver<T>, cancel: CancellationToken) -> ResponseStream<T>
where
    T: Clone + Send + Sync + 'static,
{
    let cancel = CancelOnDrop(cancel);
    Box::pin(async_stream::stream! {
        loop {
            if cancel.0.is_cancelled() {
                break;
            }
            let value = receiver.borrow_and_update().clone();
            yield Ok(value);
            tokio::select! {
                biased;
                _ = cancel.0.cancelled() => break,
                changed = receiver.changed() => if changed.is_err() { break },
            }
        }
    })
}

struct FrontendEvent {
    publisher_id: u64,
    published_at: u64,
    received_at: Instant,
    frame: FrontendLoadFrame,
}

#[derive(Default)]
struct RelayLoadState {
    frontends: HashMap<u64, FrontendSourceState>,
    totals: HashMap<String, TrafficCounters>,
}

#[derive(Default)]
struct FrontendSourceState {
    sequence: u64,
    counters: HashMap<String, TrafficCounters>,
    latest: Option<FrontendEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TrafficCounters {
    requests_started_total: Option<u64>,
    requests_completed_total: Option<u64>,
    requests_failed_total: Option<u64>,
    requests_cancelled_total: Option<u64>,
    input_tokens_total: Option<u64>,
    output_tokens_total: Option<u64>,
}

impl Default for TrafficCounters {
    fn default() -> Self {
        Self {
            requests_started_total: Some(0),
            requests_completed_total: Some(0),
            requests_failed_total: Some(0),
            requests_cancelled_total: Some(0),
            input_tokens_total: Some(0),
            output_tokens_total: Some(0),
        }
    }
}

impl From<&FrontendModelLoad> for TrafficCounters {
    fn from(model: &FrontendModelLoad) -> Self {
        Self {
            requests_started_total: model.requests_started_total,
            requests_completed_total: model.requests_completed_total,
            requests_failed_total: model.requests_failed_total,
            requests_cancelled_total: model.requests_cancelled_total,
            input_tokens_total: model.input_tokens_total,
            output_tokens_total: model.output_tokens_total,
        }
    }
}

async fn run_frontend_subscriber(
    component: Component,
    sender: mpsc::Sender<FrontendEvent>,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    let namespace = component.namespace().clone();
    loop {
        let mut subscriber = tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            result = EventSubscriber::for_namespace(&namespace, FRONTEND_LOAD_TOPIC) => match result {
                Ok(subscriber) => subscriber.typed::<FrontendLoadFrame>(),
                Err(error) => {
                    tracing::warn!(%error, "KV DC Relay frontend-load subscription failed");
                    tokio::select! {
                        _ = cancel.cancelled() => return Ok(()),
                        _ = tokio::time::sleep(SNAPSHOT_INTERVAL) => continue,
                    }
                }
            },
        };

        loop {
            let event = tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                event = subscriber.next() => event,
            };
            let Some(event) = event else {
                break;
            };
            match event {
                Ok((envelope, frame)) => {
                    let event = FrontendEvent {
                        publisher_id: envelope.publisher_id,
                        published_at: envelope.published_at,
                        received_at: Instant::now(),
                        frame,
                    };
                    let sent = tokio::select! {
                        biased;
                        _ = cancel.cancelled() => return Ok(()),
                        result = sender.send(event) => result,
                    };
                    if sent.is_err() {
                        if cancel.is_cancelled() {
                            return Ok(());
                        }
                        anyhow::bail!("frontend aggregate channel closed");
                    }
                }
                Err(error) => {
                    tracing::warn!(%error, "KV DC Relay frontend-load stream failed; reconnecting");
                    break;
                }
            }
        }
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            _ = tokio::time::sleep(SNAPSHOT_INTERVAL) => {}
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
) -> anyhow::Result<()> {
    let (usage_tx, load_tx) = snapshots;
    let mut state = RelayLoadState::default();
    let mut interval = tokio::time::interval(SNAPSHOT_INTERVAL);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = cancel.cancelled() => return Ok(()),
            event = frontend_rx.recv() => match event {
                Some(event) => record_frontend_event(&mut state, event),
                None if cancel.is_cancelled() => return Ok(()),
                None => anyhow::bail!("frontend subscriber channel closed"),
            },
            _ = interval.tick() => {
                let (expected_publishers, discovery_complete) = tokio::select! {
                    _ = cancel.cancelled() => return Ok(()),
                    expected = expected_frontend_publishers(&component) => expected,
                };
                retain_discovered_frontends(
                    &mut state,
                    &expected_publishers,
                    discovery_complete,
                );
                let catalog = pools.catalog();
                let load_snapshots = pools.load_snapshots();
                let worker_pools =
                    collect_worker_pools(&statuses, &catalog, &load_snapshots).await;
                let now = Instant::now();
                let usage = build_usage_snapshot(identity, &worker_pools);
                let load = build_load_snapshot(
                    identity,
                    &catalog,
                    &worker_pools,
                    &state,
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

fn record_frontend_event(state: &mut RelayLoadState, event: FrontendEvent) {
    let frontend_id = event.frame.frontend_instance_id;
    let sequence = event.frame.sequence;
    if sequence == 0 {
        tracing::warn!(
            frontend_id,
            "ignoring frontend load frame with zero sequence"
        );
        return;
    }
    if state
        .frontends
        .get(&frontend_id)
        .is_some_and(|current| sequence <= current.sequence)
    {
        return;
    }
    if state.frontends.iter().any(|(current_frontend, current)| {
        *current_frontend != frontend_id
            && current
                .latest
                .as_ref()
                .is_some_and(|latest| latest.publisher_id == event.publisher_id)
    }) {
        tracing::warn!(
            frontend_id,
            publisher_id = event.publisher_id,
            "ignoring frontend load publisher that changed frontend identity"
        );
        return;
    }

    let source = state.frontends.entry(frontend_id).or_default();
    for model in &event.frame.models {
        let current = TrafficCounters::from(model);
        let previous = source.counters.get(&model.model);
        state
            .totals
            .entry(model.model.clone())
            .or_default()
            .accumulate(previous, &current);
        source.counters.insert(model.model.clone(), current);
    }
    source.sequence = sequence;
    source.latest = Some(event);
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
    state: &mut RelayLoadState,
    expected_publishers: &HashSet<u64>,
    discovery_complete: bool,
) {
    if !discovery_complete {
        return;
    }
    for frontend in state.frontends.values_mut() {
        if frontend
            .latest
            .as_ref()
            .is_some_and(|event| !expected_publishers.contains(&event.publisher_id))
        {
            frontend.latest = None;
        }
    }
}

impl TrafficCounters {
    fn accumulate(&mut self, previous: Option<&Self>, current: &Self) {
        accumulate_counter(
            &mut self.requests_started_total,
            previous.map(|value| value.requests_started_total),
            current.requests_started_total,
        );
        accumulate_counter(
            &mut self.requests_completed_total,
            previous.map(|value| value.requests_completed_total),
            current.requests_completed_total,
        );
        accumulate_counter(
            &mut self.requests_failed_total,
            previous.map(|value| value.requests_failed_total),
            current.requests_failed_total,
        );
        accumulate_counter(
            &mut self.requests_cancelled_total,
            previous.map(|value| value.requests_cancelled_total),
            current.requests_cancelled_total,
        );
        accumulate_counter(
            &mut self.input_tokens_total,
            previous.map(|value| value.input_tokens_total),
            current.input_tokens_total,
        );
        accumulate_counter(
            &mut self.output_tokens_total,
            previous.map(|value| value.output_tokens_total),
            current.output_tokens_total,
        );
    }

    fn required_are_complete(&self) -> bool {
        self.requests_started_total.is_some()
            && self.requests_completed_total.is_some()
            && self.requests_failed_total.is_some()
            && self.requests_cancelled_total.is_some()
            && self.output_tokens_total.is_some()
    }
}

fn accumulate_counter(
    total: &mut Option<u64>,
    previous: Option<Option<u64>>,
    current: Option<u64>,
) {
    *total = match (*total, previous, current) {
        (Some(total), None, Some(current)) => total.checked_add(current),
        (Some(total), Some(Some(previous)), Some(current)) => current
            .checked_sub(previous)
            .and_then(|delta| total.checked_add(delta)),
        _ => None,
    };
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
    kv_complete: bool,
    scheduler_complete: bool,
    kv_observed_at_unix_ms: u64,
    scheduler_observed_at_unix_ms: u64,
}

async fn collect_worker_pools(
    statuses: &EndpointStatuses,
    catalog: &DcPoolCatalog,
    load_snapshots: &[PoolLoadSnapshot],
) -> Vec<WorkerPool> {
    let status_map = statuses.read().await.clone();
    let load_snapshots = load_snapshots
        .iter()
        .map(|snapshot| (snapshot.producer.pool_id(), *snapshot))
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
            .copied()
            .filter(|snapshot| snapshot.producer == descriptor.producer());
        let lifecycle_active = status.lifecycle == SlotLifecycle::Active;
        drop(status);

        let expected_ranks = load_snapshot.map_or(0, |snapshot| {
            u64::try_from(snapshot.kv_expected_ranks).unwrap_or(u64::MAX)
        });
        let observed_ranks = load_snapshot.map_or(0, |snapshot| {
            u64::try_from(snapshot.kv_observed_ranks).unwrap_or(u64::MAX)
        });
        let kv_complete =
            lifecycle_active && load_snapshot.is_some_and(PoolLoadSnapshot::is_kv_complete);
        let scheduler_complete =
            lifecycle_active && load_snapshot.is_some_and(PoolLoadSnapshot::is_scheduler_complete);
        let (capacity_blocks, used_blocks) = if kv_complete {
            load_snapshot
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
            live_workers: lifecycle_active
                .then(|| u64::try_from(membership.runtime_configs.len()).unwrap_or(u64::MAX)),
            capacity_blocks,
            used_blocks,
            active_decode_blocks: load_snapshot
                .filter(|_| scheduler_complete)
                .and_then(|snapshot| snapshot.active_decode_blocks),
            active_prefill_tokens: load_snapshot
                .filter(|_| scheduler_complete)
                .and_then(|snapshot| snapshot.active_prefill_tokens),
            max_concurrency: lifecycle_active.then_some(max_concurrency).flatten(),
            kv_complete,
            scheduler_complete,
            kv_observed_at_unix_ms: load_snapshot
                .map_or(0, |snapshot| snapshot.kv_source_observed_at_unix_ms),
            scheduler_observed_at_unix_ms: load_snapshot
                .map_or(0, |snapshot| snapshot.scheduler_source_observed_at_unix_ms),
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
                status: data_status(pool.kv_complete, pool.expected_ranks > 0),
                source_observed_at_unix_ms: pool.kv_observed_at_unix_ms,
            })
            .collect(),
    }
}

fn build_load_snapshot(
    identity: DcRelayIdentity,
    catalog: &DcPoolCatalog,
    worker_pools: &[WorkerPool],
    state: &RelayLoadState,
    mut expected_publishers: HashSet<u64>,
    discovery_complete: bool,
    now: Instant,
) -> proto::LoadSnapshot {
    let frontends = state
        .frontends
        .values()
        .filter_map(|frontend| frontend.latest.as_ref())
        .collect::<Vec<_>>();
    for event in &frontends {
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
            scheduler_status: data_status(pool.scheduler_complete, pool.expected_ranks > 0),
            scheduler_observed_at_unix_ms: pool.scheduler_observed_at_unix_ms,
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
    for event in frontends {
        for model in &event.frame.models {
            registrations
                .entry(model.model.clone())
                .or_insert_with(|| ModelAggregate::from_frontend(model))
                .observe(event, model, now);
        }
    }

    let expected_frontends = u32::try_from(expected_publishers.len()).unwrap_or(u32::MAX);
    let models = registrations
        .into_iter()
        .map(|(model, aggregate)| {
            aggregate.finish(
                expected_frontends,
                discovery_complete,
                state.totals.get(&model),
            )
        })
        .collect();

    proto::LoadSnapshot {
        metadata: Some(relay_metadata(identity)),
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
        add_counter(
            &mut self.pending_first_output_requests,
            model.pending_first_output_requests,
            &mut self.overflowed,
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
        add_counter(
            &mut self.input_processing_requests,
            model.input_processing_requests,
            &mut self.overflowed,
        );
        add_counter(
            &mut self.output_generation_requests,
            model.output_generation_requests,
            &mut self.overflowed,
        );
        self.source_observed_at_unix_ms = Some(
            self.source_observed_at_unix_ms
                .map_or(event.published_at, |current| {
                    current.min(event.published_at)
                }),
        );
    }

    fn finish(
        mut self,
        expected_frontends: u32,
        discovery_complete: bool,
        traffic: Option<&TrafficCounters>,
    ) -> proto::ModelLoad {
        let complete = discovery_complete
            && expected_frontends > 0
            && self.observed_frontends == expected_frontends
            && self
                .source_observed_at_unix_ms
                .is_some_and(|timestamp| timestamp > 0)
            && !self.overflowed
            && traffic.is_some_and(TrafficCounters::required_are_complete);
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
            requests_started_total: traffic.and_then(|value| value.requests_started_total),
            requests_completed_total: traffic.and_then(|value| value.requests_completed_total),
            requests_failed_total: traffic.and_then(|value| value.requests_failed_total),
            requests_cancelled_total: traffic.and_then(|value| value.requests_cancelled_total),
            input_tokens_total: traffic.and_then(|value| value.input_tokens_total),
            output_tokens_total: traffic.and_then(|value| value.output_tokens_total),
            status: data_status(complete, self.observed_frontends > 0),
            expected_frontends,
            observed_frontends: self.observed_frontends,
            source_observed_at_unix_ms: self.source_observed_at_unix_ms.unwrap_or_default(),
        }
    }
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

fn add_counter(total: &mut u64, value: u64, overflowed: &mut bool) {
    if let Some(value) = total.checked_add(value) {
        *total = value;
    } else {
        *overflowed = true;
    }
}

enum PoolEvent {
    Frame(PoolId, Arc<PublicationFrame>),
    Closed(PoolId, Status),
}

struct CkfPoolStream {
    identity: ProducerIdentity,
    endpoint: dynamo_runtime::protocols::EndpointId,
    task: JoinHandle<()>,
    capacity_omissions: u64,
    fenced: bool,
}

impl Drop for CkfPoolStream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl CkfPoolStream {
    async fn stop(mut self) {
        self.task.abort();
        let _ = (&mut self.task).await;
    }
}

async fn run_ckf_stream(
    pools: Arc<PoolRegistry>,
    publication_source: Arc<dyn RelayPublicationSource>,
    statuses: EndpointStatuses,
    output: mpsc::Sender<Result<proto::KvCuckooFilterUpdate, Status>>,
) -> Result<(), Status> {
    let relay_identity = publication_source.relay_identity();
    let mut catalog_rx = publication_source.watch_catalog();
    let (event_tx, mut event_rx) = mpsc::channel(CKF_POOL_EVENT_CAPACITY);
    let mut active = HashMap::<PoolId, CkfPoolStream>::new();

    loop {
        let catalog = catalog_rx.borrow_and_update().clone();
        reconcile_ckf_catalog(
            pools.as_ref(),
            publication_source.as_ref(),
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
                    pools.as_ref(),
                    publication_source.as_ref(),
                    &statuses,
                    &output,
                    &event_tx,
                    &mut active,
                    &catalog,
                ).await?;
            }
            event = event_rx.recv() => match event {
                Some(PoolEvent::Frame(pool_id, frame)) => {
                    let Some(stream) = active.get(&pool_id) else { continue; };
                    if frame.identity() != stream.identity {
                        continue;
                    }
                    if frame.kind() == PublicationFrameKind::SnapshotChunk {
                        send_ckf(
                            &output,
                            relay_identity,
                            proto::kv_cuckoo_filter_update::Update::Frame(ckf_frame(&frame)),
                        ).await?;
                        continue;
                    }
                    let identity = stream.identity;
                    let endpoint = stream.endpoint.clone();
                    let was_fenced = stream.fenced;
                    let prior_omissions = stream.capacity_omissions;
                    let Some(status) = ckf_pool_status(
                        &pools,
                        &statuses,
                        pool_id,
                        identity,
                        &endpoint,
                    ).await? else {
                        continue;
                    };
                    let complete = status.status == proto::DataStatus::Complete as i32;
                    if status.capacity_omissions > prior_omissions
                        || (!complete && !was_fenced)
                    {
                        send_ckf(
                            &output,
                            relay_identity,
                            proto::kv_cuckoo_filter_update::Update::Stats(status.clone()),
                        ).await?;
                    }
                    if let Some(stream) = active.get_mut(&pool_id) {
                        stream.capacity_omissions = status.capacity_omissions;
                        stream.fenced |= !complete;
                    }
                    if complete && !was_fenced {
                        send_ckf(
                            &output,
                            relay_identity,
                            proto::kv_cuckoo_filter_update::Update::Frame(ckf_frame(&frame)),
                        ).await?;
                    }
                }
                Some(PoolEvent::Closed(pool_id, error)) if active.contains_key(&pool_id) => return Err(error),
                Some(PoolEvent::Closed(_, _)) => {}
                None => return Err(Status::unavailable("Relay CKF publication task closed")),
            },
            _ = heartbeat.tick() => {
                refresh_ckf_pools(
                    pools.as_ref(),
                    publication_source.as_ref(),
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
                    }),
                ).await?;
            }
        }
    }
}

async fn reconcile_ckf_catalog(
    pools: &PoolRegistry,
    publication_source: &dyn RelayPublicationSource,
    statuses: &EndpointStatuses,
    output: &mpsc::Sender<Result<proto::KvCuckooFilterUpdate, Status>>,
    event_tx: &mpsc::Sender<PoolEvent>,
    active: &mut HashMap<PoolId, CkfPoolStream>,
    catalog: &DcPoolCatalog,
) -> Result<(), Status> {
    let relay_identity = publication_source.relay_identity();
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
            stream.stop().await;
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
        let identity = descriptor.producer();
        let Some(status) = ckf_pool_status(
            pools,
            statuses,
            descriptor.pool_id(),
            identity,
            descriptor.serving_endpoint(),
        )
        .await?
        else {
            return Err(Status::unavailable(
                "Relay pool disappeared during CKF sync",
            ));
        };
        let capacity_omissions = status.capacity_omissions;
        let complete = status.status == proto::DataStatus::Complete as i32;
        send_ckf(
            output,
            relay_identity,
            proto::kv_cuckoo_filter_update::Update::Stats(status),
        )
        .await?;
        let stream = publication_source
            .subscribe_pool(identity)
            .await
            .map_err(publication_status)?;
        let task = spawn_frame_forwarder(descriptor.pool_id(), stream, event_tx.clone());
        active.insert(
            descriptor.pool_id(),
            CkfPoolStream {
                identity,
                endpoint: descriptor.serving_endpoint().clone(),
                task,
                capacity_omissions,
                fenced: !complete,
            },
        );
    }
    Ok(())
}

async fn refresh_ckf_pools(
    pools: &PoolRegistry,
    publication_source: &dyn RelayPublicationSource,
    statuses: &EndpointStatuses,
    output: &mpsc::Sender<Result<proto::KvCuckooFilterUpdate, Status>>,
    event_tx: &mpsc::Sender<PoolEvent>,
    active: &mut HashMap<PoolId, CkfPoolStream>,
) -> Result<(), Status> {
    let relay_identity = publication_source.relay_identity();
    let pool_ids = active.keys().copied().collect::<Vec<_>>();
    for pool_id in pool_ids {
        let Some(stream) = active.get(&pool_id) else {
            continue;
        };
        let endpoint = stream.endpoint.clone();
        let identity = stream.identity;
        let was_fenced = stream.fenced;
        let prior_omissions = stream.capacity_omissions;
        let Some(status) = ckf_pool_status(pools, statuses, pool_id, identity, &endpoint).await?
        else {
            continue;
        };
        let capacity_omissions = status.capacity_omissions;
        let complete = status.status == proto::DataStatus::Complete as i32;
        if !complete {
            if status.capacity_omissions > prior_omissions || !was_fenced {
                send_ckf(
                    output,
                    relay_identity,
                    proto::kv_cuckoo_filter_update::Update::Stats(status),
                )
                .await?;
            }
            if let Some(stream) = active.get_mut(&pool_id) {
                stream.capacity_omissions = capacity_omissions;
                stream.fenced = true;
            }
            continue;
        }
        if !was_fenced {
            continue;
        }

        let old_stream = active
            .remove(&pool_id)
            .expect("refresh pool was present before resubscription");
        old_stream.stop().await;
        let stream = publication_source
            .subscribe_pool(identity)
            .await
            .map_err(publication_status)?;
        send_ckf(
            output,
            relay_identity,
            proto::kv_cuckoo_filter_update::Update::Stats(status),
        )
        .await?;
        let task = spawn_frame_forwarder(pool_id, stream, event_tx.clone());
        active.insert(
            pool_id,
            CkfPoolStream {
                identity,
                endpoint,
                task,
                capacity_omissions,
                fenced: false,
            },
        );
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

async fn ckf_pool_status(
    pools: &PoolRegistry,
    statuses: &EndpointStatuses,
    pool_id: PoolId,
    identity: ProducerIdentity,
    endpoint: &dynamo_runtime::protocols::EndpointId,
) -> Result<Option<proto::CuckooPoolStats>, Status> {
    let Some(handle) = pools.active_handle(pool_id) else {
        return Ok(None);
    };
    if handle.identity() != identity {
        return Ok(None);
    }
    let (stats, sequence, members) = handle
        .state_stats()
        .await
        .map_err(|error| Status::unavailable(error.to_string()))?;
    let capacity_omissions = stats.aggregation().capacity_failures();
    let expected_ranks = expected_ranks(statuses, endpoint).await;
    let materialized_ranks = u64::try_from(members.len()).unwrap_or(u64::MAX);
    let complete =
        capacity_omissions == 0 && expected_ranks > 0 && materialized_ranks == expected_ranks;
    Ok(Some(ckf_stats(
        identity,
        sequence,
        &members,
        stats.aggregation().unique_block_count(),
        capacity_omissions,
        complete,
    )))
}

fn spawn_frame_forwarder(
    pool_id: PoolId,
    mut stream: PoolPublicationStream,
    sender: mpsc::Sender<PoolEvent>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(result) = stream.next().await {
            match result {
                Ok(frame) => {
                    if sender.send(PoolEvent::Frame(pool_id, frame)).await.is_err() {
                        return;
                    }
                }
                Err(error) => {
                    let _ = sender
                        .send(PoolEvent::Closed(pool_id, publication_status(error)))
                        .await;
                    return;
                }
            }
        }
        let _ = sender
            .send(PoolEvent::Closed(
                pool_id,
                Status::unavailable("Relay CKF publication stream closed"),
            ))
            .await;
    })
}

fn publication_status(error: PublicationError) -> Status {
    match error.kind() {
        PublicationErrorKind::NotFound
        | PublicationErrorKind::ProducerMismatch
        | PublicationErrorKind::Unavailable => Status::unavailable(error.to_string()),
        PublicationErrorKind::ResourceExhausted => Status::resource_exhausted(error.to_string()),
        PublicationErrorKind::InvalidPublication => Status::data_loss(error.to_string()),
        PublicationErrorKind::Internal => Status::internal(error.to_string()),
    }
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

fn ckf_frame(frame: &PublicationFrame) -> proto::CuckooPoolFrame {
    proto::CuckooPoolFrame {
        producer: Some(ckf_producer(frame.identity())),
        base_sequence: frame.base_sequence(),
        sequence: frame.sequence(),
        kind: match frame.kind() {
            PublicationFrameKind::SnapshotChunk => proto::CuckooFrameKind::SnapshotChunk as i32,
            PublicationFrameKind::Delta => proto::CuckooFrameKind::Delta as i32,
        },
        cbi1_payload: frame.payload().to_vec(),
    }
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
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState};
    use futures::StreamExt;
    use tokio::sync::Semaphore;

    use super::*;
    use crate::kv_dc_relay::discovery::DcMembershipView;
    use crate::kv_dc_relay::pool_registry::PoolActorConfig;
    use crate::kv_dc_relay::publication::{
        DEFAULT_ACTIVE_POOL_STREAMS, DEFAULT_SNAPSHOT_ENCODING_CONCURRENCY,
        DEFAULT_SNAPSHOT_PROGRESS_TIMEOUT, RegistryPublicationSource,
    };
    use crate::kv_dc_relay::topology::TopologyPublisher;

    fn frontend_event(
        publisher_id: u64,
        sequence: u64,
        frontend_instance_id: u64,
        model: FrontendModelLoad,
    ) -> FrontendEvent {
        FrontendEvent {
            publisher_id,
            published_at: sequence,
            received_at: Instant::now(),
            frame: FrontendLoadFrame {
                frontend_instance_id,
                sequence,
                serving_ready: true,
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
            requests_started_total: Some(5),
            requests_completed_total: Some(3),
            requests_failed_total: Some(1),
            requests_cancelled_total: Some(1),
            input_tokens_total: Some(11),
            output_tokens_total: Some(7),
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

    fn empty_pool_registry(identity: DcRelayIdentity) -> Arc<PoolRegistry> {
        Arc::new(PoolRegistry::new(
            identity,
            PoolActorConfig {
                expected_unique_blocks: 32,
                publication_threshold: 1,
                publication_delay: Duration::from_millis(1),
            },
        ))
    }

    fn empty_publication_source(
        identity: DcRelayIdentity,
        pools: Arc<PoolRegistry>,
        lifecycle: CancellationToken,
    ) -> Arc<dyn RelayPublicationSource> {
        let topology = Arc::new(TopologyPublisher::new(
            DcMembershipView::default(),
            &pools.catalog(),
        ));
        Arc::new(RegistryPublicationSource::new(
            pools,
            topology,
            identity,
            lifecycle,
            Arc::new(Semaphore::new(DEFAULT_SNAPSHOT_ENCODING_CONCURRENCY)),
            DEFAULT_ACTIVE_POOL_STREAMS,
            DEFAULT_SNAPSHOT_PROGRESS_TIMEOUT,
        ))
    }

    #[test]
    fn grpc_listener_accepts_only_loopback_addresses() {
        assert!(validate_listen_address("127.0.0.1:50051".parse().unwrap()).is_ok());
        assert!(validate_listen_address("[::1]:50051".parse().unwrap()).is_ok());
        assert!(validate_listen_address("0.0.0.0:50051".parse().unwrap()).is_err());
        assert!(validate_listen_address("192.0.2.10:50051".parse().unwrap()).is_err());
    }

    #[tokio::test]
    async fn watch_stream_coalesces_to_the_latest_cumulative_snapshot() {
        let (sender, receiver) = watch::channel(0_u64);
        let mut stream = watch_stream(receiver, CancellationToken::new());
        assert_eq!(stream.next().await.unwrap().unwrap(), 0);

        sender.send_replace(1);
        sender.send_replace(3);

        assert_eq!(stream.next().await.unwrap().unwrap(), 3);
    }

    #[tokio::test]
    async fn watch_stream_drops_its_snapshot_on_cancellation() {
        let (_sender, receiver) = watch::channel(7_u64);
        let cancel = CancellationToken::new();
        let mut stream = watch_stream(receiver, cancel.clone());

        cancel.cancel();

        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn ckf_response_stream_drops_queued_data_on_cancellation() {
        let cancel = CancellationToken::new();
        let (sender, receiver) = mpsc::channel(1);
        sender.send(Ok(7_u64)).await.unwrap();
        let mut stream = ckf_response_stream(receiver, cancel.clone());

        cancel.cancel();

        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn stats_task_failure_cancels_the_relay_and_its_siblings() {
        let cancel = CancellationToken::new();
        let fatal_cancel = CancellationToken::new();
        let terminal = Arc::new(HostTerminalState::default());
        let mut tasks = JoinSet::new();
        tasks.spawn(async {
            (
                "failed task",
                Err(anyhow::anyhow!("intentional test failure")),
            )
        });
        let sibling_cancel = cancel.clone();
        tasks.spawn(async move {
            sibling_cancel.cancelled().await;
            ("sibling task", Ok(()))
        });

        tokio::time::timeout(
            Duration::from_secs(1),
            supervise_stats_tasks(tasks, cancel.clone(), fatal_cancel.clone(), terminal),
        )
        .await
        .expect("stats supervisor should stop after a child failure");

        assert!(cancel.is_cancelled());
        assert!(fatal_cancel.is_cancelled());
    }

    #[tokio::test]
    async fn cancelling_stats_closes_an_open_ckf_stream() {
        let identity = DcRelayIdentity::new(1, 2);
        let cancel = CancellationToken::new();
        let (_usage_tx, usage) = watch::channel(proto::KvUsageSnapshot::default());
        let (_load_tx, load) = watch::channel(proto::LoadSnapshot::default());
        let pools = empty_pool_registry(identity);
        let service = RelayStatsService {
            pools: pools.clone(),
            publication_source: empty_publication_source(identity, pools, cancel.child_token()),
            statuses: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            usage,
            load,
            cancel: cancel.clone(),
        };
        let mut stream = service
            .watch_kv_cuckoo_filter(Request::new(()))
            .await
            .unwrap()
            .into_inner();
        let first = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("CKF stream should publish its initial heartbeat")
            .unwrap()
            .unwrap();
        assert!(matches!(
            first.update,
            Some(proto::kv_cuckoo_filter_update::Update::Heartbeat(_))
        ));

        cancel.cancel();
        assert!(
            tokio::time::timeout(Duration::from_secs(1), stream.next())
                .await
                .expect("CKF stream should close after cancellation")
                .is_none()
        );
    }

    #[test]
    fn frontend_frames_replace_by_frontend_and_ignore_stale_sequences() {
        let mut state = RelayLoadState::default();
        record_frontend_event(&mut state, frontend_event(1, 2, 10, model_load("a", 2)));
        record_frontend_event(&mut state, frontend_event(1, 1, 10, model_load("a", 9)));
        assert_eq!(
            state.frontends[&10].latest.as_ref().unwrap().frame.models[0]
                .pending_first_output_requests,
            2
        );

        record_frontend_event(&mut state, frontend_event(1, 3, 20, model_load("a", 3)));
        assert!(!state.frontends.contains_key(&20));

        record_frontend_event(&mut state, frontend_event(2, 3, 10, model_load("a", 4)));
        retain_discovered_frontends(&mut state, &HashSet::from([2]), true);
        assert_eq!(state.frontends.len(), 1);
        assert_eq!(
            state.frontends[&10].latest.as_ref().unwrap().publisher_id,
            2
        );
    }

    #[test]
    fn failed_discovery_does_not_discard_last_known_frontend_frame() {
        let mut state = RelayLoadState::default();
        record_frontend_event(&mut state, frontend_event(1, 1, 10, model_load("a", 1)));
        retain_discovered_frontends(&mut state, &HashSet::new(), false);
        assert!(state.frontends[&10].latest.is_some());

        retain_discovered_frontends(&mut state, &HashSet::new(), true);
        assert!(state.frontends[&10].latest.is_none());
    }

    #[test]
    fn cumulative_frontend_frames_recover_skipped_publications() {
        let mut state = RelayLoadState::default();
        let mut first = model_load("a", 1);
        first.requests_started_total = Some(2);
        first.output_tokens_total = Some(10);
        record_frontend_event(&mut state, frontend_event(1, 1, 10, first));

        let mut latest = model_load("a", 1);
        latest.requests_started_total = Some(5);
        latest.output_tokens_total = Some(30);
        record_frontend_event(&mut state, frontend_event(1, 3, 10, latest));

        assert_eq!(state.totals["a"].requests_started_total, Some(5));
        assert_eq!(state.totals["a"].output_tokens_total, Some(30));
    }

    #[test]
    fn cumulative_frontend_totals_survive_departure_and_reconnect() {
        let mut state = RelayLoadState::default();
        record_frontend_event(&mut state, frontend_event(1, 1, 10, model_load("a", 1)));
        retain_discovered_frontends(&mut state, &HashSet::new(), true);
        assert_eq!(state.totals["a"].requests_started_total, Some(5));

        let mut resumed = model_load("a", 1);
        resumed.requests_started_total = Some(8);
        record_frontend_event(&mut state, frontend_event(2, 2, 10, resumed));
        assert_eq!(state.totals["a"].requests_started_total, Some(8));
    }

    #[test]
    fn regressed_frontend_counter_fails_closed() {
        let mut state = RelayLoadState::default();
        record_frontend_event(&mut state, frontend_event(1, 1, 10, model_load("a", 1)));
        let mut regressed = model_load("a", 1);
        regressed.requests_started_total = Some(4);
        record_frontend_event(&mut state, frontend_event(1, 2, 10, regressed));

        assert_eq!(state.totals["a"].requests_started_total, None);
    }

    #[test]
    fn model_load_is_complete_only_with_every_expected_fresh_frontend() {
        let mut aggregate = ModelAggregate::new(registration("a"));
        let event = frontend_event(1, 1, 10, model_load("a", 2));
        aggregate.observe(&event, &event.frame.models[0], Instant::now());
        let traffic = TrafficCounters::from(&event.frame.models[0]);
        let complete = aggregate.finish(1, true, Some(&traffic));
        assert_eq!(complete.status, proto::DataStatus::Complete as i32);
        assert_eq!(complete.ready_frontends, Some(1));
        assert_eq!(complete.pending_first_output_requests, Some(2));
        assert_eq!(complete.pending_first_output_input_tokens, Some(17));
        assert_eq!(complete.live_input_tokens, Some(31));
        assert_eq!(complete.input_tokens_total, Some(11));
        assert_eq!(complete.output_tokens_total, Some(7));
        assert_eq!(complete.source_observed_at_unix_ms, 1);

        let mut aggregate = ModelAggregate::new(registration("a"));
        aggregate.observe(&event, &event.frame.models[0], Instant::now());
        let incomplete = aggregate.finish(2, true, Some(&traffic));
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
        let traffic = TrafficCounters::from(&event.frame.models[0]);

        let model = aggregate.finish(1, true, Some(&traffic));

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

        let traffic = TrafficCounters::default();
        let model = aggregate.finish(2, true, Some(&traffic));

        assert_eq!(model.status, proto::DataStatus::Complete as i32);
        assert_eq!(model.source_observed_at_unix_ms, 10);
    }

    #[test]
    fn stale_frontend_frame_makes_relay_only_model_unavailable() {
        let mut aggregate = ModelAggregate::new(registration("a"));
        let mut event = frontend_event(1, 1, 10, model_load("a", 2));
        event.received_at = Instant::now() - SOURCE_FRESHNESS - Duration::from_millis(1);
        aggregate.observe(&event, &event.frame.models[0], Instant::now());
        let traffic = TrafficCounters::from(&event.frame.models[0]);
        let model = aggregate.finish(1, true, Some(&traffic));
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

    #[test]
    fn ckf_grpc_frame_preserves_canonical_publication_metadata() {
        let format = DcCkfState::new(CkfConfig::new(32)).unwrap().format();
        let producer = ProducerIdentity::new(pool_id(), 5, 7, format);
        let frame =
            PublicationFrame::test_frame(producer, 11, 12, PublicationFrameKind::SnapshotChunk);

        let grpc = ckf_frame(&frame);

        assert_eq!(grpc.producer, Some(ckf_producer(producer)));
        assert_eq!(grpc.base_sequence, 11);
        assert_eq!(grpc.sequence, 12);
        assert_eq!(grpc.kind, proto::CuckooFrameKind::SnapshotChunk as i32);
        assert!(grpc.cbi1_payload.is_empty());
    }
}
