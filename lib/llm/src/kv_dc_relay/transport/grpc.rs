// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_stream::{stream, try_stream};
use dynamo_kv_router::identity::PoolId;
use futures::{Stream, StreamExt};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

use super::super::identity::{DcPoolCatalog, DcRelayIdentity};
use super::super::protocol as proto;
use super::super::publication::{
    PoolPublicationStream, PublicationError, PublicationErrorKind, PublicationFrame,
    PublicationFrameKind,
};
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
    load_updates: LoadUpdateHub,
    limits: SubscriberLimits,
}

pub(crate) struct KvEventRelayServiceConfig {
    pub(crate) pool_heartbeat_interval: Duration,
    pub(crate) readiness_heartbeat_interval: Duration,
    pub(crate) load_updates: LoadUpdateHub,
    pub(crate) limits: SubscriberLimits,
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
            load_updates: config.load_updates,
            limits: config.limits,
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
        let subscription = self
            .source
            .subscribe_pool(pool_id, move |actual| {
                producer_to_wire(actual) == expected_producer
            })
            .await?;
        tracing::debug!(%subscriber_id, %pool_id, "KV Relay pool subscriber connected");
        Ok(Response::new(pool_update_stream(
            subscription,
            PoolStreamContext {
                relay: self.source.relay_identity(),
                cancel: self.cancel.clone(),
                heartbeat_interval: self.pool_heartbeat_interval,
                subscriber_id,
                pool_id,
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
    permit: OwnedSemaphorePermit,
}

fn pool_update_stream(
    mut subscription: PoolPublicationStream,
    context: PoolStreamContext,
) -> PoolStream {
    let PoolStreamContext {
        relay,
        cancel,
        heartbeat_interval,
        subscriber_id,
        pool_id,
        permit,
    } = context;
    Box::pin(stream! {
        let _permit = permit;
        let mut current_identity = None;
        let mut current_sequence = 0;
        let mut snapshot_chunks_remaining = None;
        let mut heartbeat = tokio::time::interval_at(
            tokio::time::Instant::now() + heartbeat_interval, heartbeat_interval,
        );
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                update = subscription.next() => match update {
                    Some(Ok(frame)) => {
                        let identity = frame.identity();
                        current_identity = Some(identity);
                        current_sequence = frame.sequence();
                        if frame.kind() == PublicationFrameKind::SnapshotChunk {
                            let remaining = snapshot_chunks_remaining.get_or_insert_with(|| {
                                identity.format().bucket_count().div_ceil(
                                    proto::wire::images::SNAPSHOT_CHUNK_BUCKETS,
                                )
                            });
                            *remaining = remaining.saturating_sub(1);
                        }
                        yield Ok(filter_update(&frame, relay));
                        heartbeat.reset();
                    }
                    Some(Err(error)) => {
                        tracing::debug!(%subscriber_id, %pool_id, %error, "KV Relay pool publication ended");
                        yield Err(publication_status(error));
                        break;
                    }
                    None => break,
                },
                _ = heartbeat.tick(), if snapshot_chunks_remaining == Some(0) => {
                    if let Some(identity) = current_identity {
                        yield Ok(heartbeat_update(identity, current_sequence, relay));
                    }
                }
            }
        }
    })
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

fn filter_update(frame: &PublicationFrame, relay: DcRelayIdentity) -> proto::FilterUpdate {
    proto::FilterUpdate {
        protocol_version: proto::RELAY_PROTOCOL_VERSION,
        relay: Some(relay_identity_to_wire(relay)),
        producer: Some(producer_to_wire(frame.identity())),
        base_sequence: frame.base_sequence(),
        sequence: frame.sequence(),
        send_ts_us: unix_timestamp::<1_000_000>(),
        kind: match frame.kind() {
            PublicationFrameKind::SnapshotChunk => proto::FilterUpdateKind::SnapshotChunk,
            PublicationFrameKind::Delta => proto::FilterUpdateKind::Delta,
        } as i32,
        payload: frame.payload().clone(),
        contract_marker: proto::RELAY_CONTRACT_MARKER,
    }
}

fn heartbeat_update(
    identity: dynamo_kv_router::indexer::cuckoo::ProducerIdentity,
    sequence: u64,
    relay: DcRelayIdentity,
) -> proto::FilterUpdate {
    proto::FilterUpdate {
        protocol_version: proto::RELAY_PROTOCOL_VERSION,
        relay: Some(relay_identity_to_wire(relay)),
        producer: Some(producer_to_wire(identity)),
        base_sequence: sequence,
        sequence,
        send_ts_us: unix_timestamp::<1_000_000>(),
        kind: proto::FilterUpdateKind::Heartbeat as i32,
        payload: Default::default(),
        contract_marker: proto::RELAY_CONTRACT_MARKER,
    }
}

pub(super) fn publication_status(error: PublicationError) -> Status {
    let message = error.to_string();
    match error.kind() {
        PublicationErrorKind::NotFound => Status::not_found(message),
        PublicationErrorKind::ProducerMismatch | PublicationErrorKind::InvalidPublication => {
            Status::failed_precondition(message)
        }
        PublicationErrorKind::Unavailable => Status::unavailable(message),
        PublicationErrorKind::ResourceExhausted => Status::resource_exhausted(message),
        PublicationErrorKind::Internal => Status::internal(message),
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
    use super::*;

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
    fn canonical_publication_errors_map_to_grpc_status() {
        assert_eq!(
            publication_status(PublicationError::unavailable("retired")).code(),
            tonic::Code::Unavailable
        );
        assert_eq!(
            publication_status(PublicationError::resource_exhausted("lagged")).code(),
            tonic::Code::ResourceExhausted
        );
        assert_eq!(
            publication_status(PublicationError::invalid_publication("gap")).code(),
            tonic::Code::FailedPrecondition
        );
        assert_eq!(
            publication_status(PublicationError::internal("encoder")).code(),
            tonic::Code::Internal
        );
    }
}
