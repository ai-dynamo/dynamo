// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use async_stream::{stream, try_stream};
use dynamo_kv_router::identity::PoolId;
use futures::Stream;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast, mpsc};
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

use super::super::identity::{DcPoolCatalog, DcRelayIdentity};
use super::super::protocol as proto;
use super::super::publication_codec::{
    PublicationFrame, PublicationFrameKind, encode_heartbeat, encode_snapshot,
};
use super::super::publication_hub::{PublicationHubError, PublicationHubSubscription};
use super::super::topology::{TopologyReadinessState, TopologySnapshot};
use super::identity::{
    descriptor_to_wire, endpoint_to_wire, pool_id_from_wire, pool_id_to_wire, producer_to_wire,
    relay_identity_to_wire, unix_timestamp, worker_role_to_wire,
};
use super::load::LoadUpdateHub;
use super::source::WanPublicationSource;

type CatalogStream =
    Pin<Box<dyn Stream<Item = Result<proto::KvPoolCatalogUpdate, Status>> + Send + 'static>>;
type ReadinessStream =
    Pin<Box<dyn Stream<Item = Result<proto::ServingReadinessUpdate, Status>> + Send + 'static>>;
type PoolStream = Pin<Box<dyn Stream<Item = Result<proto::FilterUpdate, Status>> + Send + 'static>>;
type LoadStream =
    Pin<Box<dyn Stream<Item = Result<proto::KvPoolLoadUpdate, Status>> + Send + 'static>>;

#[derive(Clone)]
struct SubscriberLimit {
    permits: Arc<Semaphore>,
    maximum: usize,
}

impl SubscriberLimit {
    fn new(maximum: usize) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(maximum)),
            maximum,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum StreamKind {
    Catalog,
    Pool,
    Readiness,
    Load,
}

#[derive(Clone)]
pub(crate) struct SubscriberLimits {
    catalog: SubscriberLimit,
    pool: SubscriberLimit,
    readiness: SubscriberLimit,
    load: SubscriberLimit,
}

impl SubscriberLimits {
    pub(crate) fn new(catalog: usize, pool: usize, readiness: usize, load: usize) -> Self {
        Self {
            catalog: SubscriberLimit::new(catalog),
            pool: SubscriberLimit::new(pool),
            readiness: SubscriberLimit::new(readiness),
            load: SubscriberLimit::new(load),
        }
    }

    #[allow(clippy::result_large_err)]
    fn acquire(&self, stream: StreamKind) -> Result<OwnedSemaphorePermit, Status> {
        let limit = match stream {
            StreamKind::Catalog => &self.catalog,
            StreamKind::Pool => &self.pool,
            StreamKind::Readiness => &self.readiness,
            StreamKind::Load => &self.load,
        };
        limit.permits.clone().try_acquire_owned().map_err(|_| {
            let resource = match stream {
                StreamKind::Catalog => "catalog stream",
                StreamKind::Pool => "total pool stream",
                StreamKind::Readiness => "readiness stream",
                StreamKind::Load => "load stream",
            };
            Status::resource_exhausted(format!("Relay {resource} limit {} reached", limit.maximum))
        })
    }
}

#[derive(Clone)]
pub(crate) struct KvEventRelayService {
    source: WanPublicationSource,
    cancel: CancellationToken,
    pool_heartbeat_interval: Duration,
    readiness_heartbeat_interval: Duration,
    snapshot_progress_timeout: Duration,
    load_updates: LoadUpdateHub,
    limits: SubscriberLimits,
    snapshot_encoding_permits: Arc<Semaphore>,
}

pub(crate) struct KvEventRelayServiceConfig {
    pub(crate) pool_heartbeat_interval: Duration,
    pub(crate) readiness_heartbeat_interval: Duration,
    pub(crate) snapshot_progress_timeout: Duration,
    pub(crate) load_updates: LoadUpdateHub,
    pub(crate) limits: SubscriberLimits,
    pub(crate) snapshot_encoding_permits: Arc<Semaphore>,
}

impl KvEventRelayService {
    pub(crate) fn new(
        source: WanPublicationSource,
        cancel: CancellationToken,
        config: KvEventRelayServiceConfig,
    ) -> Self {
        Self {
            source,
            cancel,
            pool_heartbeat_interval: config.pool_heartbeat_interval,
            readiness_heartbeat_interval: config.readiness_heartbeat_interval,
            snapshot_progress_timeout: config.snapshot_progress_timeout,
            load_updates: config.load_updates,
            limits: config.limits,
            snapshot_encoding_permits: config.snapshot_encoding_permits,
        }
    }

    #[allow(clippy::result_large_err)]
    fn acquire_stream_permit(&self, stream: StreamKind) -> Result<OwnedSemaphorePermit, Status> {
        self.limits.acquire(stream)
    }
}

#[tonic::async_trait]
impl proto::KvEventRelay for KvEventRelayService {
    type WatchKvPoolCatalogStream = CatalogStream;
    type SubscribeKvPoolStream = PoolStream;
    type SubscribeServingReadinessStream = ReadinessStream;
    type SubscribeKvPoolLoadStream = LoadStream;

    async fn get_relay_info(
        &self,
        request: Request<proto::RelayInfoRequest>,
    ) -> Result<Response<proto::RelayInfo>, Status> {
        require_contract(request.into_inner().contract_marker)?;
        Ok(Response::new(proto::RelayInfo {
            protocol_version: proto::RELAY_PROTOCOL_VERSION,
            relay: Some(relay_identity_to_wire(self.source.relay_identity())),
            contract_marker: proto::RELAY_CONTRACT_MARKER,
        }))
    }

    async fn watch_kv_pool_catalog(
        &self,
        request: Request<proto::WatchKvPoolCatalogRequest>,
    ) -> Result<Response<Self::WatchKvPoolCatalogStream>, Status> {
        let request = request.into_inner();
        require_contract(request.contract_marker)?;
        let subscriber_id = validate_subscriber_id(request.subscriber_id)?;
        let permit = self.acquire_stream_permit(StreamKind::Catalog)?;
        tracing::debug!(%subscriber_id, "KV Relay pool catalog subscriber connected");
        let mut catalogs = self.source.watch_catalog();
        let cancel = self.cancel.clone();
        let relay = self.source.relay_identity();
        let initial = catalogs.borrow().clone();
        let stream = try_stream! {
            let _permit = permit;
            yield catalog_to_wire(initial, relay);
            loop {
                let changed = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    changed = catalogs.changed() => changed,
                };
                if changed.is_err() {
                    break;
                }
                let update = {
                    let current = catalogs.borrow_and_update();
                    current.clone()
                };
                yield catalog_to_wire(update, relay);
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn subscribe_kv_pool(
        &self,
        request: Request<proto::SubscribeKvPoolRequest>,
    ) -> Result<Response<Self::SubscribeKvPoolStream>, Status> {
        let request = request.into_inner();
        require_contract(request.contract_marker)?;
        let subscriber_id = validate_subscriber_id(request.subscriber_id)?;
        let expected_producer = request.expected_producer.ok_or_else(|| {
            Status::invalid_argument("SubscribeKvPool requires expected_producer")
        })?;
        proto::validate_producer_identity(&expected_producer)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let wire_pool_id = expected_producer.pool_id.as_ref().ok_or_else(|| {
            Status::invalid_argument("SubscribeKvPool expected_producer requires pool_id")
        })?;
        let pool_id = pool_id_from_wire(wire_pool_id)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let permit = self.acquire_stream_permit(StreamKind::Pool)?;
        // Admit snapshot work before cloning the hub's generation snapshot. Waiters must not
        // retain old full CKF images while deltas advance the mutable hub state.
        let snapshot_encoding_permit = tokio::select! {
            biased;
            _ = self.cancel.cancelled() => {
                return Err(Status::unavailable("Relay snapshot encoder is shutting down"));
            }
            permit = self.snapshot_encoding_permits.clone().acquire_owned() => permit
                .map_err(|_| Status::unavailable("Relay snapshot encoder is shutting down"))?,
        };
        let mut subscription = match self
            .source
            .subscribe_pool(pool_id, move |actual| {
                producer_to_wire(actual) == expected_producer
            })
            .await
        {
            Ok(subscription) => subscription,
            Err(error) => return Err(publication_status(error)),
        };
        let snapshot = subscription.take_snapshot().map_err(publication_status)?;
        let identity = snapshot.identity();
        let initial_sequence = snapshot.sequence();
        let bootstrap = match encode_initial_snapshot(
            snapshot,
            snapshot_encoding_permit,
            self.snapshot_progress_timeout,
        ) {
            Ok(frames) => frames,
            Err(error) => {
                let reason = format!("failed to encode subscriber snapshot: {error}");
                self.source.fence_publication(identity, &reason);
                return Err(Status::failed_precondition(reason));
            }
        };
        subscription.ensure_active().map_err(publication_status)?;
        tracing::debug!(%subscriber_id, %pool_id, "KV Relay pool subscriber connected");
        Ok(Response::new(pool_update_stream(
            subscription,
            bootstrap,
            PoolStreamContext {
                relay: self.source.relay_identity(),
                cancel: self.cancel.clone(),
                heartbeat_interval: self.pool_heartbeat_interval,
                subscriber_id,
                pool_id,
                source: self.source.clone(),
                initial_identity: identity,
                initial_sequence,
                permit,
            },
        )))
    }

    async fn subscribe_serving_readiness(
        &self,
        request: Request<proto::SubscribeServingReadinessRequest>,
    ) -> Result<Response<Self::SubscribeServingReadinessStream>, Status> {
        let request = request.into_inner();
        require_contract(request.contract_marker)?;
        let subscriber_id = validate_subscriber_id(request.subscriber_id)?;
        let permit = self.acquire_stream_permit(StreamKind::Readiness)?;
        tracing::debug!(%subscriber_id, "KV Relay serving-readiness subscriber connected");
        let mut snapshots = self.source.watch_readiness();
        let cancel = self.cancel.clone();
        let relay = self.source.relay_identity();
        let heartbeat_interval = self.readiness_heartbeat_interval;
        let initial = snapshots.borrow().clone();
        let stream = try_stream! {
            let _permit = permit;
            yield readiness_to_wire(&initial, relay);
            let first_heartbeat = tokio::time::Instant::now() + heartbeat_interval;
            let mut heartbeat = tokio::time::interval_at(first_heartbeat, heartbeat_interval);
            heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                let snapshot = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    changed = snapshots.changed() => {
                        if changed.is_err() {
                            break;
                        }
                        snapshots.borrow_and_update().clone()
                    }
                    _ = heartbeat.tick() => snapshots.borrow().clone(),
                };
                yield readiness_to_wire(&snapshot, relay);
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }

    async fn subscribe_kv_pool_load(
        &self,
        request: Request<proto::SubscribeKvPoolLoadRequest>,
    ) -> Result<Response<Self::SubscribeKvPoolLoadStream>, Status> {
        let request = request.into_inner();
        require_contract(request.contract_marker)?;
        let subscriber_id = validate_subscriber_id(request.subscriber_id)?;
        let permit = self.acquire_stream_permit(StreamKind::Load)?;
        tracing::debug!(%subscriber_id, "KV Relay pool load subscriber connected");
        let mut updates = self.load_updates.subscribe();
        let initial = self.load_updates.current();
        let cancel = self.cancel.clone();
        let stream = try_stream! {
            let _permit = permit;
            let mut current_sequence = initial.window_sequence;
            yield initial;
            loop {
                let update = tokio::select! {
                    biased;
                    _ = cancel.cancelled() => break,
                    update = updates.recv() => update,
                };
                match update {
                    Ok(update) if update.window_sequence > current_sequence => {
                        current_sequence = update.window_sequence;
                        yield update;
                    }
                    Ok(_) => {}
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        tracing::warn!(%subscriber_id, skipped, "KV Relay load subscriber lagged; forcing resubscribe");
                        Err(Status::resource_exhausted(format!(
                            "load subscriber lagged by {skipped} complete windows; resubscribe"
                        )))?;
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        Err(Status::unavailable("pool load publication stopped"))?;
                    }
                }
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }
}

struct PoolStreamContext {
    relay: DcRelayIdentity,
    cancel: CancellationToken,
    heartbeat_interval: Duration,
    subscriber_id: String,
    pool_id: PoolId,
    source: WanPublicationSource,
    initial_identity: dynamo_kv_router::indexer::cuckoo::ProducerIdentity,
    initial_sequence: u64,
    permit: OwnedSemaphorePermit,
}

fn pool_update_stream(
    mut subscription: PublicationHubSubscription,
    mut bootstrap: SnapshotBootstrap,
    context: PoolStreamContext,
) -> PoolStream {
    let PoolStreamContext {
        relay,
        cancel,
        heartbeat_interval,
        subscriber_id,
        pool_id,
        source,
        initial_identity,
        initial_sequence,
        permit,
    } = context;
    Box::pin(stream! {
        let _permit = permit;
        let mut current_sequence = initial_sequence;
        loop {
            let frame = tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                frame = bootstrap.recv() => frame,
            };
            let frame = match frame {
                Ok(Some(frame)) => frame,
                Ok(None) => break,
                Err(SnapshotBootstrapError::EncoderTaskFailed(error)) => {
                    if !snapshot_encoder_failure_requires_fence(&error) {
                        tracing::warn!(
                            %subscriber_id,
                            %pool_id,
                            error = %error,
                            "KV Relay snapshot encoder task was cancelled; closing subscriber stream"
                        );
                        yield Err(Status::unavailable(
                            "pool snapshot encoder stopped; resubscribe",
                        ));
                        return;
                    }
                    let reason = format!("pool snapshot encoder task failed: {error}");
                    if source.fence_publication(initial_identity, &reason) {
                        tracing::error!(
                            %subscriber_id,
                            %pool_id,
                            error = %error,
                            "KV Relay snapshot encoder panicked; fenced producer generation"
                        );
                    } else {
                        tracing::warn!(
                            %subscriber_id,
                            %pool_id,
                            error = %error,
                            "KV Relay snapshot encoder panicked after producer generation was no longer active"
                        );
                    }
                    yield Err(Status::internal(
                        "pool snapshot encoder failed; refresh the catalog before resubscribing",
                    ));
                    return;
                }
                Err(SnapshotBootstrapError::ProgressTimeout(timeout)) => {
                    tracing::warn!(
                        %subscriber_id,
                        %pool_id,
                        timeout_ms = timeout.as_millis(),
                        "KV Relay snapshot subscriber made no progress; forcing resubscribe"
                    );
                    yield Err(Status::resource_exhausted(format!(
                        "pool snapshot subscriber made no progress for {} ms; resubscribe",
                        timeout.as_millis(),
                    )));
                    return;
                }
            };
            if let Err(error) = subscription.ensure_active() {
                yield Err(publication_status(error));
                return;
            }
            current_sequence = frame.sequence;
            yield Ok(filter_update(frame, relay));
        }

        if let Err(error) = subscription.ensure_active() {
            yield Err(publication_status(error));
            return;
        }

        let first_heartbeat = tokio::time::Instant::now() + heartbeat_interval;
        let mut heartbeat = tokio::time::interval_at(first_heartbeat, heartbeat_interval);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                update = subscription.recv() => match update {
                    Ok(frame) => {
                        if frame.identity != initial_identity {
                            yield Err(Status::failed_precondition("pool producer identity drifted"));
                            break;
                        }
                        current_sequence = frame.sequence;
                        yield Ok(filter_update((*frame).clone(), relay));
                        heartbeat.reset();
                    }
                    Err(PublicationHubError::SubscriberLagged(_)) => {
                        tracing::warn!(%subscriber_id, %pool_id, "KV Relay pool subscriber exceeded its bounded queue; forcing resubscribe");
                        yield Err(Status::resource_exhausted(
                            "pool subscriber exceeded its bounded queue; resubscribe for a fresh snapshot",
                        ));
                        break;
                    }
                    Err(error) => {
                        yield Err(publication_status(error));
                        break;
                    }
                },
                _ = heartbeat.tick() => {
                    yield Ok(filter_update(encode_heartbeat(initial_identity, current_sequence), relay));
                }
            }
        }
    })
}

fn encode_initial_snapshot(
    snapshot: super::super::publication_hub::HubSnapshot,
    permit: OwnedSemaphorePermit,
    progress_timeout: Duration,
) -> Result<SnapshotBootstrap, String> {
    let frames = encode_snapshot(snapshot).map_err(|error| error.to_string())?;
    Ok(spawn_snapshot_encoder(frames, permit, progress_timeout))
}

const SNAPSHOT_BOOTSTRAP_QUEUE_CAPACITY: usize = 1;
const SNAPSHOT_SEND_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Debug, thiserror::Error)]
enum SnapshotBootstrapError {
    #[error("snapshot encoding task failed: {0}")]
    EncoderTaskFailed(tokio::task::JoinError),
    #[error("snapshot subscriber made no progress for {0:?}")]
    ProgressTimeout(Duration),
}

fn snapshot_encoder_failure_requires_fence(error: &JoinError) -> bool {
    error.is_panic()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SnapshotEncoderExit {
    Complete,
    Cancelled,
    ReceiverClosed,
    ProgressTimeout,
}

struct SnapshotBootstrap {
    receiver: mpsc::Receiver<PublicationFrame>,
    task: Option<JoinHandle<SnapshotEncoderExit>>,
    cancel: CancellationToken,
    progress_timeout: Duration,
}

impl SnapshotBootstrap {
    async fn recv(&mut self) -> Result<Option<PublicationFrame>, SnapshotBootstrapError> {
        if let Some(frame) = self.receiver.recv().await {
            return Ok(Some(frame));
        }
        let Some(task) = self.task.take() else {
            return Ok(None);
        };
        match task
            .await
            .map_err(SnapshotBootstrapError::EncoderTaskFailed)?
        {
            SnapshotEncoderExit::ProgressTimeout => Err(SnapshotBootstrapError::ProgressTimeout(
                self.progress_timeout,
            )),
            SnapshotEncoderExit::Complete
            | SnapshotEncoderExit::Cancelled
            | SnapshotEncoderExit::ReceiverClosed => Ok(None),
        }
    }
}

impl Drop for SnapshotBootstrap {
    fn drop(&mut self) {
        self.cancel.cancel();
        self.receiver.close();
    }
}

fn spawn_snapshot_encoder<I>(
    frames: I,
    permit: OwnedSemaphorePermit,
    progress_timeout: Duration,
) -> SnapshotBootstrap
where
    I: Iterator<Item = PublicationFrame> + Send + 'static,
{
    let (sender, receiver) = mpsc::channel(SNAPSHOT_BOOTSTRAP_QUEUE_CAPACITY);
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        for mut frame in frames {
            let stalled_since = Instant::now();
            loop {
                if task_cancel.is_cancelled() {
                    return SnapshotEncoderExit::Cancelled;
                }
                match sender.try_send(frame) {
                    Ok(()) => break,
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        return SnapshotEncoderExit::ReceiverClosed;
                    }
                    Err(mpsc::error::TrySendError::Full(returned)) => {
                        frame = returned;
                        let elapsed = stalled_since.elapsed();
                        if elapsed >= progress_timeout {
                            return SnapshotEncoderExit::ProgressTimeout;
                        }
                        sleep(
                            progress_timeout
                                .saturating_sub(elapsed)
                                .min(SNAPSHOT_SEND_POLL_INTERVAL),
                        );
                    }
                }
            }
        }
        SnapshotEncoderExit::Complete
    });
    SnapshotBootstrap {
        receiver,
        task: Some(task),
        cancel,
        progress_timeout,
    }
}

fn catalog_to_wire(catalog: DcPoolCatalog, relay: DcRelayIdentity) -> proto::KvPoolCatalogUpdate {
    proto::KvPoolCatalogUpdate {
        protocol_version: proto::RELAY_PROTOCOL_VERSION,
        relay: Some(relay_identity_to_wire(relay)),
        revision: catalog.revision(),
        snapshot: Some(proto::KvPoolCatalogSnapshot {
            pools: catalog.pools().iter().map(descriptor_to_wire).collect(),
        }),
        contract_marker: proto::RELAY_CONTRACT_MARKER,
    }
}

fn readiness_to_wire(
    snapshot: &TopologySnapshot,
    relay: DcRelayIdentity,
) -> proto::ServingReadinessUpdate {
    proto::ServingReadinessUpdate {
        protocol_version: proto::RELAY_PROTOCOL_VERSION,
        relay: Some(relay_identity_to_wire(relay)),
        revision: snapshot.revision,
        entries: snapshot
            .entries
            .iter()
            .map(|entry| proto::TopologyEntry {
                namespace: entry.namespace.clone(),
                canonical_model_id: entry.model.as_str().to_string(),
                state: readiness_state_to_wire(entry.state) as i32,
                present_roles: entry
                    .present_roles
                    .iter()
                    .copied()
                    .map(worker_role_to_wire)
                    .map(|role| role as i32)
                    .collect(),
                missing_roles: entry
                    .missing_roles
                    .iter()
                    .copied()
                    .map(worker_role_to_wire)
                    .map(|role| role as i32)
                    .collect(),
                members: entry
                    .members
                    .iter()
                    .map(|member| proto::TopologyMember {
                        endpoint: Some(endpoint_to_wire(&member.endpoint)),
                        roles: member
                            .roles
                            .iter()
                            .copied()
                            .map(worker_role_to_wire)
                            .map(|role| role as i32)
                            .collect(),
                        pool_id: member.pool_id.map(pool_id_to_wire),
                    })
                    .collect(),
                duplicate_role_endpoints: entry
                    .duplicate_role_endpoints
                    .iter()
                    .copied()
                    .map(worker_role_to_wire)
                    .map(|role| role as i32)
                    .collect(),
                legacy_fallback_active: entry.legacy_fallback_active,
                adapters: entry
                    .adapters
                    .iter()
                    .map(|adapter| proto::AdapterReadiness {
                        canonical_model_id: adapter.model.as_str().to_string(),
                        state: readiness_state_to_wire(adapter.state) as i32,
                        missing_roles: adapter
                            .missing_roles
                            .iter()
                            .copied()
                            .map(worker_role_to_wire)
                            .map(|role| role as i32)
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
        contract_marker: proto::RELAY_CONTRACT_MARKER,
    }
}

fn readiness_state_to_wire(state: TopologyReadinessState) -> proto::ServingReadinessState {
    match state {
        TopologyReadinessState::Unknown => proto::ServingReadinessState::Unknown,
        TopologyReadinessState::Unavailable => proto::ServingReadinessState::Unavailable,
        TopologyReadinessState::Ready => proto::ServingReadinessState::Ready,
    }
}

fn filter_update(frame: PublicationFrame, relay: DcRelayIdentity) -> proto::FilterUpdate {
    proto::FilterUpdate {
        protocol_version: proto::RELAY_PROTOCOL_VERSION,
        relay: Some(relay_identity_to_wire(relay)),
        producer: Some(producer_to_wire(frame.identity)),
        base_sequence: frame.base_sequence,
        sequence: frame.sequence,
        send_ts_us: unix_timestamp::<1_000_000>(),
        kind: match frame.kind {
            PublicationFrameKind::SnapshotChunk => proto::FilterUpdateKind::SnapshotChunk,
            PublicationFrameKind::Delta => proto::FilterUpdateKind::Delta,
            PublicationFrameKind::Heartbeat => proto::FilterUpdateKind::Heartbeat,
        } as i32,
        payload: frame.payload,
        contract_marker: proto::RELAY_CONTRACT_MARKER,
    }
}

fn publication_status(error: PublicationHubError) -> Status {
    let message = error.to_string();
    match error {
        PublicationHubError::UnknownPool(_) => Status::not_found(message),
        PublicationHubError::Unavailable(_) => Status::unavailable(message),
        PublicationHubError::SubscriberLimit { .. }
        | PublicationHubError::InitializedHubLimit { .. }
        | PublicationHubError::SubscriberLagged(_) => Status::resource_exhausted(message),
        _ => Status::failed_precondition(message),
    }
}

#[allow(clippy::result_large_err)]
fn require_contract(marker: u32) -> Result<(), Status> {
    if marker != proto::RELAY_CONTRACT_MARKER {
        return Err(Status::failed_precondition(
            "unsupported Relay v1 wire contract",
        ));
    }
    Ok(())
}

#[allow(clippy::result_large_err)]
fn validate_subscriber_id(raw: String) -> Result<String, Status> {
    const MAX_BYTES: usize = 128;
    if raw.is_empty() {
        return Err(Status::invalid_argument("subscriber_id must not be empty"));
    }
    if raw.len() > MAX_BYTES {
        return Err(Status::invalid_argument(format!(
            "subscriber_id exceeds {MAX_BYTES} bytes"
        )));
    }
    if raw.chars().any(char::is_control) {
        return Err(Status::invalid_argument(
            "subscriber_id must not contain control characters",
        ));
    }
    Ok(raw)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, RoutingScopeId,
    };
    use dynamo_kv_router::indexer::cuckoo::{CkfConfig, DcCkfState, ProducerIdentity};

    use super::*;

    struct CountingFrames {
        frame: PublicationFrame,
        remaining: usize,
        produced: Arc<AtomicUsize>,
    }

    impl Iterator for CountingFrames {
        type Item = PublicationFrame;

        fn next(&mut self) -> Option<Self::Item> {
            if self.remaining == 0 {
                return None;
            }
            self.remaining -= 1;
            self.produced.fetch_add(1, Ordering::Relaxed);
            Some(self.frame.clone())
        }
    }

    struct PanicAfterFirstFrame(Option<PublicationFrame>);

    impl Iterator for PanicAfterFirstFrame {
        type Item = PublicationFrame;

        fn next(&mut self) -> Option<Self::Item> {
            self.0
                .take()
                .map(Some)
                .unwrap_or_else(|| panic!("injected snapshot iterator failure"))
        }
    }

    fn test_identity() -> ProducerIdentity {
        let format = DcCkfState::new(CkfConfig::new(32))
            .expect("fixture state")
            .format();
        ProducerIdentity::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            7,
            11,
            format,
        )
    }

    #[test]
    fn subscriber_ids_are_validated_without_truncation() {
        assert_eq!(
            validate_subscriber_id("consumer-a".to_string()).unwrap(),
            "consumer-a"
        );
        assert!(validate_subscriber_id(String::new()).is_err());
        assert!(validate_subscriber_id("x".repeat(129)).is_err());
        assert!(validate_subscriber_id("consumer\nspoof".to_string()).is_err());
    }

    #[test]
    fn total_pool_stream_limit_is_configurable_and_releases_permits() {
        const MAX_POOL_STREAMS: usize = 65;
        let limits = SubscriberLimits::new(1, MAX_POOL_STREAMS, 1, 1);
        let mut permits = (0..MAX_POOL_STREAMS)
            .map(|_| limits.acquire(StreamKind::Pool).unwrap())
            .collect::<Vec<_>>();

        let error = limits.acquire(StreamKind::Pool).err().unwrap();
        assert_eq!(error.code(), tonic::Code::ResourceExhausted);
        assert_eq!(error.message(), "Relay total pool stream limit 65 reached");

        permits.pop();
        assert!(limits.acquire(StreamKind::Pool).is_ok());
    }

    #[test]
    fn publication_status_distinguishes_absent_lagged_and_retired() {
        let pool_id = PoolId::new(
            dynamo_kv_router::identity::IndexerDomainId::new(
                dynamo_kv_router::identity::CacheSemanticsId::new(
                    [1; 16],
                    dynamo_kv_router::identity::IdentitySource::Explicit,
                ),
                dynamo_kv_router::identity::RoutingScopeId::new(
                    [2; 16],
                    dynamo_kv_router::identity::IdentitySource::Explicit,
                ),
            ),
            dynamo_kv_router::identity::DcId::new(3),
        );
        assert_eq!(
            publication_status(PublicationHubError::UnknownPool(pool_id)).code(),
            tonic::Code::NotFound
        );
        assert_eq!(
            publication_status(PublicationHubError::ProducerMismatch(pool_id)).code(),
            tonic::Code::FailedPrecondition
        );
        assert_eq!(
            publication_status(PublicationHubError::SubscriberLagged(pool_id)).code(),
            tonic::Code::ResourceExhausted
        );
        let per_pool =
            publication_status(PublicationHubError::SubscriberLimit { pool_id, limit: 7 });
        assert_eq!(per_pool.code(), tonic::Code::ResourceExhausted);
        assert!(per_pool.message().contains("subscriber limit 7"));
        let initialized_hub =
            publication_status(PublicationHubError::InitializedHubLimit { limit: 3 });
        assert_eq!(initialized_hub.code(), tonic::Code::ResourceExhausted);
        assert!(
            initialized_hub
                .message()
                .contains("publication hub limit 3")
        );
        assert_eq!(
            publication_status(PublicationHubError::Unavailable("retired".to_string())).code(),
            tonic::Code::Unavailable
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_bootstrap_stays_bounded_until_the_stream_reads() {
        let permits = Arc::new(Semaphore::new(1));
        let permit = permits.clone().acquire_owned().await.unwrap();
        let frame = encode_heartbeat(test_identity(), 1);
        let produced = Arc::new(AtomicUsize::new(0));
        let frames = CountingFrames {
            frame,
            remaining: 3,
            produced: produced.clone(),
        };
        let mut bootstrap = spawn_snapshot_encoder(frames, permit, Duration::from_secs(5));

        tokio::time::timeout(Duration::from_secs(1), async {
            while bootstrap.receiver.len() != SNAPSHOT_BOOTSTRAP_QUEUE_CAPACITY
                || produced.load(Ordering::Relaxed) < SNAPSHOT_BOOTSTRAP_QUEUE_CAPACITY + 1
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("producer filled the bounded bootstrap queue");
        assert_eq!(bootstrap.receiver.max_capacity(), 1);
        assert!(!bootstrap.task.as_ref().unwrap().is_finished());
        assert_eq!(permits.available_permits(), 0);
        assert_eq!(
            produced.load(Ordering::Relaxed),
            SNAPSHOT_BOOTSTRAP_QUEUE_CAPACITY + 1,
            "the lazy producer may hold only the queued frame and one pending frame"
        );

        for _ in 0..3 {
            assert!(bootstrap.recv().await.unwrap().is_some());
        }
        assert!(bootstrap.recv().await.unwrap().is_none());
        assert_eq!(permits.available_permits(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_snapshot_bootstrap_releases_a_blocked_encoder() {
        let permits = Arc::new(Semaphore::new(1));
        let permit = permits.clone().acquire_owned().await.unwrap();
        let frame = encode_heartbeat(test_identity(), 1);
        let bootstrap = spawn_snapshot_encoder(
            std::iter::repeat_n(frame, 3),
            permit,
            Duration::from_secs(5),
        );

        tokio::time::timeout(Duration::from_secs(1), async {
            while bootstrap.receiver.len() != SNAPSHOT_BOOTSTRAP_QUEUE_CAPACITY {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("producer filled the bounded bootstrap queue");
        drop(bootstrap);

        tokio::time::timeout(Duration::from_secs(1), permits.acquire())
            .await
            .expect("cancelled encoder released its permit")
            .expect("encoder semaphore remained open")
            .forget();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stalled_snapshot_bootstrap_releases_admission_capacity() {
        let permits = Arc::new(Semaphore::new(1));
        let permit = permits.clone().acquire_owned().await.unwrap();
        let frame = encode_heartbeat(test_identity(), 1);
        let timeout = Duration::from_millis(25);
        let mut bootstrap = spawn_snapshot_encoder(std::iter::repeat_n(frame, 3), permit, timeout);

        tokio::time::timeout(Duration::from_secs(1), async {
            while !bootstrap.task.as_ref().unwrap().is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("stalled snapshot encoder did not enforce its progress timeout");
        assert_eq!(permits.available_permits(), 1);
        assert!(bootstrap.recv().await.unwrap().is_some());
        assert!(matches!(
            bootstrap.recv().await.unwrap_err(),
            SnapshotBootstrapError::ProgressTimeout(actual) if actual == timeout
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_iterator_panic_is_reported_after_queued_frames() {
        let permits = Arc::new(Semaphore::new(1));
        let permit = permits.clone().acquire_owned().await.unwrap();
        let frame = encode_heartbeat(test_identity(), 1);
        let mut bootstrap = spawn_snapshot_encoder(
            PanicAfterFirstFrame(Some(frame)),
            permit,
            Duration::from_secs(5),
        );

        assert!(bootstrap.recv().await.unwrap().is_some());
        match bootstrap.recv().await.unwrap_err() {
            SnapshotBootstrapError::EncoderTaskFailed(error) => {
                assert!(error.is_panic());
                assert!(snapshot_encoder_failure_requires_fence(&error));
            }
            error => panic!("unexpected snapshot bootstrap error: {error}"),
        }
        assert_eq!(permits.available_permits(), 1);
    }

    #[tokio::test]
    async fn cancelled_snapshot_encoder_does_not_require_fencing() {
        let task = tokio::spawn(std::future::pending::<()>());
        task.abort();
        let error = task.await.unwrap_err();

        assert!(error.is_cancelled());
        assert!(!snapshot_encoder_failure_requires_fence(&error));
    }
}
