// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-aware KV state-source selection and router recovery.
//!
//! This controller lives at one router/indexer boundary. It owns source-mode
//! selection and publishes one aggregate CacheOwner projection; router-core
//! remains unaware of discovery and attachment lifecycles.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use anyhow::{Context, Result};
use dynamo_kv_router::{
    identity::CacheOwnerId,
    indexer::{
        KvStateAgentIdentity, KvStateAgentStatus, KvStateProtocolVersion, WorkerKvQueryResponse,
    },
    protocols::{
        KvCacheEvent, KvCacheEventData, ResetScope, ResidencyDomain, ResidencyProjection,
        RouterEvent, StorageTier, WorkerWithDpRank,
    },
};
use dynamo_runtime::{
    component::Component,
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryQuery, EventScope,
        EventSourceInstanceId, EventSourceQuery,
    },
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::EventSubscriber,
};
use futures::StreamExt;
use tokio::sync::{Mutex, mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;

use crate::discovery::kv_state_agent::{
    KV_STATE_ATTACHMENT_TOPIC_V2, KV_STATE_EVENT_TOPIC_V2, KV_STATE_SOURCE_TOPIC_V2,
    KvStateAttachmentAdvertisement, KvStateSourceAdvertisement, attachment_record_id,
};
use crate::{
    discovery::{KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus},
    kv_router::Indexer,
};

use super::{
    RuntimeWorkerQueryTransport, recovery_lane::RecoveryLane, worker_query_state::RankState,
};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
const RECOVERY_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy)]
struct OwnerRuntime {
    publisher_id: Option<u64>,
    recovered_cursor: u64,
    attachment_generation: Option<u64>,
    last_worker: Option<WorkerWithDpRank>,
    ready: bool,
    recovery_required: bool,
}

struct OwnerRecoveryTail {
    publisher_id: u64,
    attachment_generation: Option<u64>,
    schedule_generation: u64,
    rank: RankState,
}

#[derive(Clone)]
struct OwnerRecoveryPlan {
    owner: CacheOwnerId,
    source: KvStateSourceAdvertisement,
    attachment: Option<KvStateAttachmentAdvertisement>,
    expected: KvStateAgentIdentity,
    recognized_worker: Option<WorkerWithDpRank>,
    previous_publisher_id: Option<u64>,
    previous_cursor: u64,
    recovery_required: bool,
    observation_revision: u64,
    schedule_generation: u64,
}

enum OwnerRecoveryResult {
    Initial {
        plan: OwnerRecoveryPlan,
        status: Result<KvStateAgentStatus>,
        recovery: Option<Result<WorkerKvQueryResponse>>,
    },
    PostStatus {
        plan: OwnerRecoveryPlan,
        recovered_cursor: u64,
        status: Result<KvStateAgentStatus>,
    },
}

/// Router-owned state-agent work plus the filtered legacy source view.
pub(crate) struct KvStateRouterHandle {
    pub(crate) legacy_membership: KvSourceMembershipWatch,
    completion: oneshot::Receiver<()>,
}

impl KvStateRouterHandle {
    pub(crate) fn into_parts(self) -> (KvSourceMembershipWatch, oneshot::Receiver<()>) {
        (self.legacy_membership, self.completion)
    }
}

pub(crate) async fn start_state_agent_router(
    component: Component,
    indexer: Indexer,
    membership: KvSourceMembershipWatch,
    expected_block_size: u32,
    cancellation_token: CancellationToken,
) -> Result<KvStateRouterHandle> {
    let endpoint = membership
        .borrow()
        .resolved_kv_state_endpoint()
        .cloned()
        .context("KV state endpoint is ambiguous while starting state-agent reconciliation")?;
    let discovery = component.drt().discovery();
    let source_cancel = cancellation_token.child_token();
    let sources = discovery
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint.clone(),
                KV_STATE_SOURCE_TOPIC_V2,
            )),
            Some(source_cancel),
        )
        .await
        .context("failed to watch V2 KV state sources")?;

    let (recognized_tx, recognized_rx) =
        watch::channel(source_mode_suppressed_workers(&membership.borrow()));
    let legacy_membership = filtered_legacy_membership(
        membership.clone(),
        recognized_rx,
        cancellation_token.child_token(),
    );
    let (completion_tx, completion_rx) = oneshot::channel();
    let observation_revision = Arc::new(AtomicU64::new(0));
    let projection_commit = Arc::new(Mutex::new(()));
    let (observation_tx, observation_rx) = watch::channel(0u64);
    start_observation_fence(
        component.clone(),
        indexer.clone(),
        membership.clone(),
        endpoint.clone(),
        recognized_tx.clone(),
        observation_revision.clone(),
        projection_commit.clone(),
        observation_tx,
        cancellation_token.child_token(),
    )
    .await?;
    component.drt().runtime().secondary().spawn(async move {
        run_state_agent_router(
            indexer,
            membership,
            component,
            endpoint,
            expected_block_size,
            sources,
            recognized_tx,
            observation_revision,
            projection_commit,
            observation_rx,
            cancellation_token,
        )
        .await;
        let _ = completion_tx.send(());
    });
    Ok(KvStateRouterHandle {
        legacy_membership,
        completion: completion_rx,
    })
}

#[allow(clippy::too_many_arguments)]
async fn start_observation_fence(
    component: Component,
    indexer: Indexer,
    mut membership: KvSourceMembershipWatch,
    endpoint: EndpointId,
    recognized_tx: watch::Sender<HashSet<WorkerWithDpRank>>,
    revision: Arc<AtomicU64>,
    projection_commit: Arc<Mutex<()>>,
    revision_tx: watch::Sender<u64>,
    cancel: CancellationToken,
) -> Result<()> {
    let discovery = component.drt().discovery();
    let mut sources = discovery
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint.clone(),
                KV_STATE_SOURCE_TOPIC_V2,
            )),
            Some(cancel.child_token()),
        )
        .await?;
    let mut attachments = discovery
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint,
                KV_STATE_ATTACHMENT_TOPIC_V2,
            )),
            Some(cancel.child_token()),
        )
        .await?;
    component.drt().runtime().secondary().spawn(async move {
        // NOTE: This independent watch must invalidate projection while the
        // reconciliation loop is blocked on acknowledged clears or applying a
        // snapshot. Temporary loss of KV-aware routing is intentional;
        // ordinary serving continues without a stale projection.
        // TODO(#13044): share/index these model-wide watches, and invalidate
        // only the affected owner, once that preserves the same independent fence.
        loop {
            let observed = tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                changed = membership.changed() => {
                    if changed.is_err() { break; }
                    membership.borrow_and_update();
                    recognized_tx.send_modify(|recognized| {
                        recognized.extend(source_mode_suppressed_workers(&membership.borrow()));
                    });
                    true
                }
                event = sources.next() => event.is_some(),
                event = attachments.next() => event.is_some(),
            };
            if !observed {
                tracing::error!("KV state observation fence lost a discovery watch");
                let _commit = projection_commit.lock().await;
                let next = advance_observation_revision(&revision);
                indexer.set_residency_projection(ResidencyProjection::default());
                revision_tx.send_replace(next);
                break;
            }
            let _commit = projection_commit.lock().await;
            indexer.set_residency_projection(ResidencyProjection::default());
            let next = advance_observation_revision(&revision);
            revision_tx.send_replace(next);
        }
    });
    Ok(())
}

fn advance_observation_revision(revision: &AtomicU64) -> u64 {
    revision
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(1)
        })
        .unwrap_or_else(|_| {
            tracing::error!("KV state observation revision exhausted");
            u64::MAX
        })
        .saturating_add(1)
}

fn filtered_legacy_membership(
    membership: KvSourceMembershipWatch,
    mut recognized: watch::Receiver<HashSet<WorkerWithDpRank>>,
    cancel: CancellationToken,
) -> KvSourceMembershipWatch {
    let mut source = membership.clone();
    let initial = suppress_recognized(source.borrow().clone(), &recognized.borrow());
    let (tx, rx) = watch::channel(initial);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                changed = source.changed() => if changed.is_err() { break; },
                changed = recognized.changed() => if changed.is_err() { break; },
            }
            let view = suppress_recognized(
                source.borrow_and_update().clone(),
                &recognized.borrow_and_update(),
            );
            tx.send_replace(view);
        }
    });
    membership.with_receiver(rx)
}

fn suppress_recognized(
    mut view: KvSourceMembershipView,
    recognized: &HashSet<WorkerWithDpRank>,
) -> KvSourceMembershipView {
    for worker in recognized {
        if let Some(status) = view.sources.get_mut(worker) {
            *status = KvSourceStatus::Suppressed;
        }
    }
    view
}

fn source_mode_suppressed_workers(view: &KvSourceMembershipView) -> HashSet<WorkerWithDpRank> {
    view.sources
        .keys()
        .copied()
        .filter(|worker| {
            !matches!(
                view.kv_event_source_mode(worker.worker_id),
                None | Some("framework_v1")
            )
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
async fn run_state_agent_router(
    indexer: Indexer,
    mut membership: KvSourceMembershipWatch,
    component: Component,
    endpoint: EndpointId,
    expected_block_size: u32,
    mut source_stream: dynamo_runtime::discovery::DiscoveryStream,
    recognized_tx: watch::Sender<HashSet<WorkerWithDpRank>>,
    observation_revision: Arc<AtomicU64>,
    projection_commit: Arc<Mutex<()>>,
    mut observation_rx: watch::Receiver<u64>,
    cancel: CancellationToken,
) {
    let mut attachment_stream: Option<dynamo_runtime::discovery::DiscoveryStream> = None;
    let mut event_stream: Option<
        dynamo_runtime::transports::event_plane::TypedEventSubscriber<Vec<RouterEvent>>,
    > = None;
    let mut transport: Option<Arc<RuntimeWorkerQueryTransport>> = None;
    let mut sources = HashMap::new();
    let mut attachments = HashMap::new();
    let mut runtime: HashMap<CacheOwnerId, OwnerRuntime> = HashMap::new();
    let recovery_lane = Arc::new(RecoveryLane::new());
    let (recovery_tx, mut recovery_rx) = mpsc::unbounded_channel();
    let mut recovery_tails: HashMap<CacheOwnerId, OwnerRecoveryTail> = HashMap::new();
    let mut recovering_publishers: HashMap<u64, CacheOwnerId> = HashMap::new();
    let mut next_recovery_schedule = 0u64;
    let mut committed_revision = None;

    loop {
        enum Input {
            Membership,
            Source(Option<Result<DiscoveryEvent>>),
            Attachment(Option<Result<DiscoveryEvent>>),
            Events(
                Option<
                    Result<(
                        dynamo_runtime::transports::event_plane::EventEnvelope,
                        Vec<RouterEvent>,
                    )>,
                >,
            ),
            Recovery(Option<OwnerRecoveryResult>),
            Observation,
            Cancelled,
        }
        let input = tokio::select! {
            biased;
            _ = cancel.cancelled() => Input::Cancelled,
            changed = membership.changed() => {
                if changed.is_err() { Input::Cancelled } else { Input::Membership }
            },
            event = source_stream.next() => Input::Source(event),
            event = async {
                match attachment_stream.as_mut() {
                    Some(stream) => stream.next().await,
                    None => std::future::pending().await,
                }
            } => Input::Attachment(event),
            event = async {
                match event_stream.as_mut() {
                    Some(stream) => stream.next().await,
                    None => std::future::pending().await,
                }
            } => Input::Events(event),
            result = recovery_rx.recv() => Input::Recovery(result),
            changed = observation_rx.changed() => {
                if changed.is_err() { Input::Cancelled } else { Input::Observation }
            },
        };
        let mut reconcile = false;
        match input {
            Input::Cancelled => break,
            Input::Membership => {
                membership.borrow_and_update();
                let snapshot = membership.borrow();
                let mut recognized = source_mode_suppressed_workers(&snapshot);
                recognized.extend(
                    recognized_tx
                        .borrow()
                        .iter()
                        .copied()
                        .filter(|worker| snapshot.sources.contains_key(worker)),
                );
                recognized_tx.send_replace(recognized);
                reconcile = true;
            }
            Input::Observation => {
                observation_rx.borrow_and_update();
                reconcile = true;
            }
            Input::Source(Some(event)) => match update_advertisements(
                event,
                &endpoint,
                KV_STATE_SOURCE_TOPIC_V2,
                &mut sources,
            ) {
                Ok(changed) => reconcile = changed,
                Err(error) => tracing::warn!(%error, "Ignoring invalid V2 KV state source"),
            },
            Input::Attachment(Some(event)) => match update_advertisements(
                event,
                &endpoint,
                KV_STATE_ATTACHMENT_TOPIC_V2,
                &mut attachments,
            ) {
                Ok(changed) => reconcile = changed,
                Err(error) => tracing::warn!(%error, "Ignoring invalid V2 KV state attachment"),
            },
            Input::Source(None) | Input::Attachment(None) => {
                tracing::error!(%endpoint, "KV state discovery watch ended; keeping recognized ranks fail-closed");
                indexer.set_residency_projection(ResidencyProjection::default());
                // Keep the recognized sender alive so the derived legacy view
                // continues forwarding membership for unrecognized ranks.
                cancel.cancelled().await;
                break;
            }
            Input::Events(Some(Ok((envelope, events)))) => {
                if let Some(owner) = recovering_publishers.get(&envelope.publisher_id).copied()
                    && let Some(tail) = recovery_tails.get_mut(&owner)
                    && tail.publisher_id == envelope.publisher_id
                {
                    for event in events {
                        tail.rank.buffer_recovery_tail(event);
                    }
                    continue;
                }
                if apply_live_events(&indexer, envelope.publisher_id, events, &mut runtime).await
                    && let Some(revision) = committed_revision
                    && publish_projection_if_current(
                        &indexer,
                        &runtime,
                        &observation_revision,
                        &projection_commit,
                        revision,
                    )
                    .await
                    .is_err()
                {
                    committed_revision = None;
                }
            }
            Input::Events(Some(Err(error))) => {
                tracing::warn!(%error, %endpoint, "Failed to decode V2 KV state event batch");
            }
            Input::Events(None) => {
                tracing::error!(%endpoint, "V2 KV state event stream ended");
                indexer.set_residency_projection(ResidencyProjection::default());
                // V2 stays fail-closed, but unrelated legacy ranks must keep
                // receiving membership updates until the router shuts down.
                cancel.cancelled().await;
                break;
            }
            Input::Recovery(Some(result)) => {
                let owner = result.owner();
                let publisher_id = result.publisher_id();
                recovery_lane.finish((owner, result.schedule_generation()));
                if let Err(error) = finish_owner_recovery(
                    result,
                    &indexer,
                    transport.as_ref(),
                    &membership,
                    &sources,
                    &attachments,
                    &mut runtime,
                    &recovery_lane,
                    &recovery_tx,
                    &mut recovery_tails,
                    &mut recovering_publishers,
                    &observation_revision,
                    &projection_commit,
                )
                .await
                {
                    fail_owner_recovery(
                        owner,
                        publisher_id,
                        &mut runtime,
                        &mut recovery_tails,
                        &mut recovering_publishers,
                    );
                    tracing::warn!(%owner, %error, "KV state-source recovery completion failed closed");
                }
            }
            Input::Recovery(None) => break,
        }
        if reconcile {
            if !sources.is_empty() && transport.is_none() {
                match start_state_agent_consumers(&component, &endpoint, cancel.child_token()).await
                {
                    Ok((attachment_watch, events, state_transport)) => {
                        attachment_stream = Some(attachment_watch);
                        event_stream = Some(events);
                        transport = Some(Arc::new(state_transport));
                    }
                    Err(error) => {
                        tracing::warn!(%error, %endpoint, "Failed to activate V2 KV state consumers");
                        indexer.set_residency_projection(ResidencyProjection::default());
                        recognized_tx
                            .send_replace(membership.borrow().sources.keys().copied().collect());
                        continue;
                    }
                }
            }
            let Some(transport) = transport.as_ref() else {
                continue;
            };
            let membership_snapshot = membership.borrow().clone();
            let revision = observation_revision.load(Ordering::Acquire);
            let Some(schedule_generation) = next_recovery_schedule.checked_add(1) else {
                tracing::error!("KV state recovery schedule generation exhausted");
                break;
            };
            next_recovery_schedule = schedule_generation;
            recovery_lane.cancel_all().await;
            if let Err(error) = schedule_state_sources(
                &indexer,
                transport.clone(),
                membership_snapshot,
                &endpoint,
                expected_block_size,
                &sources,
                &attachments,
                &mut runtime,
                &recovery_lane,
                &recovery_tx,
                &mut recovery_tails,
                &mut recovering_publishers,
                &recognized_tx,
                &observation_revision,
                &projection_commit,
                revision,
                schedule_generation,
            )
            .await
            {
                tracing::warn!(%error, %endpoint, "KV state-source reconciliation failed closed");
                committed_revision = None;
            } else {
                committed_revision = Some(revision);
            }
        }
    }
    recovery_lane.cancel_all().await;
    indexer.set_residency_projection(ResidencyProjection::default());
}

async fn start_state_agent_consumers(
    component: &Component,
    endpoint: &EndpointId,
    cancel: CancellationToken,
) -> Result<(
    dynamo_runtime::discovery::DiscoveryStream,
    dynamo_runtime::transports::event_plane::TypedEventSubscriber<Vec<RouterEvent>>,
    RuntimeWorkerQueryTransport,
)> {
    let attachments = component
        .drt()
        .discovery()
        .list_and_watch(
            DiscoveryQuery::EventSources(EventSourceQuery::endpoint_topic(
                endpoint.clone(),
                KV_STATE_ATTACHMENT_TOPIC_V2,
            )),
            Some(cancel),
        )
        .await
        .context("failed to watch V2 KV state attachments")?;
    let events = EventSubscriber::for_endpoint_id_with_transport(
        component.drt(),
        endpoint,
        KV_STATE_EVENT_TOPIC_V2,
        component.drt().default_event_transport_kind(),
    )
    .await
    .context("failed to subscribe to V2 KV state events")?
    .typed::<Vec<RouterEvent>>();
    let transport = RuntimeWorkerQueryTransport::new(component).await?;
    Ok((attachments, events, transport))
}

trait Advertisement: serde::de::DeserializeOwned + Clone + PartialEq {}

impl Advertisement for KvStateSourceAdvertisement {}

impl Advertisement for KvStateAttachmentAdvertisement {}

fn update_advertisements<T: Advertisement>(
    event: Result<DiscoveryEvent>,
    endpoint: &EndpointId,
    expected_topic: &str,
    values: &mut HashMap<u64, T>,
) -> Result<bool> {
    let event = event?;
    let expected_scope = EventScope::Endpoint {
        endpoint: endpoint.clone(),
    };
    match event {
        DiscoveryEvent::Added(DiscoveryInstance::EventSource {
            scope,
            topic,
            publisher_id,
            metadata,
        }) => {
            if scope != expected_scope || topic != expected_topic {
                return Ok(false);
            }
            let value: T = serde_json::from_value(metadata)?;
            if values
                .get(&publisher_id)
                .is_some_and(|previous| previous != &value)
            {
                anyhow::bail!("discovery identity changed its immutable advertisement");
            }
            values.insert(publisher_id, value);
            Ok(true)
        }
        DiscoveryEvent::Removed(DiscoveryInstanceId::EventSource(EventSourceInstanceId {
            scope,
            topic,
            publisher_id,
        })) => {
            if scope == expected_scope && topic == expected_topic {
                Ok(values.remove(&publisher_id).is_some())
            } else {
                Ok(false)
            }
        }
        DiscoveryEvent::Added(_)
        | DiscoveryEvent::Removed(_)
        | DiscoveryEvent::ModelTaintsUpdated(_) => Ok(false),
    }
}

#[allow(clippy::too_many_arguments)]
async fn schedule_state_sources(
    indexer: &Indexer,
    transport: Arc<RuntimeWorkerQueryTransport>,
    membership: KvSourceMembershipView,
    endpoint: &EndpointId,
    expected_block_size: u32,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    recovery_lane: &Arc<RecoveryLane<(CacheOwnerId, u64)>>,
    recovery_tx: &mpsc::UnboundedSender<OwnerRecoveryResult>,
    recovery_tails: &mut HashMap<CacheOwnerId, OwnerRecoveryTail>,
    recovering_publishers: &mut HashMap<u64, CacheOwnerId>,
    recognized_tx: &watch::Sender<HashSet<WorkerWithDpRank>>,
    observation_revision: &AtomicU64,
    projection_commit: &Mutex<()>,
    expected_revision: u64,
    schedule_generation: u64,
) -> Result<()> {
    ensure_observation_revision(observation_revision, expected_revision)?;
    let live_workers: HashSet<_> = membership.sources.keys().copied().collect();
    let mut source_by_owner: HashMap<CacheOwnerId, Vec<&KvStateSourceAdvertisement>> =
        HashMap::new();
    for (discovery_id, source) in sources {
        let expected_ingress_protocol =
            crate::discovery::kv_state_agent::KvStateIngressProtocol::VllmResidencyV1;
        let expected_domain = source.cache_owner_id.pool().indexer_domain();
        if source.publisher_id != *discovery_id
            || source.protocol_version != KvStateProtocolVersion::V2
            || source.event_topic != KV_STATE_EVENT_TOPIC_V2
            || source.kv_state_endpoint != *endpoint
            || source.kv_block_size != expected_block_size
            || source.ingress_protocol != expected_ingress_protocol
            || expected_domain != source.indexer_domain_id
        {
            tracing::warn!(
                discovery_id = *discovery_id,
                advertised_publisher_id = source.publisher_id,
                advertised_protocol = ?source.protocol_version,
                expected_protocol = ?KvStateProtocolVersion::V2,
                advertised_event_topic = %source.event_topic,
                expected_event_topic = KV_STATE_EVENT_TOPIC_V2,
                advertised_endpoint = ?source.kv_state_endpoint,
                expected_endpoint = ?endpoint,
                advertised_block_size = source.kv_block_size,
                expected_block_size,
                advertised_ingress_protocol = ?source.ingress_protocol,
                expected_ingress_protocol = ?expected_ingress_protocol,
                advertised_domain = %source.indexer_domain_id,
                expected_domain = %expected_domain,
                owner = %source.cache_owner_id,
                "Ignoring incompatible V2 KV state source advertisement"
            );
            continue;
        }
        source_by_owner
            .entry(source.cache_owner_id)
            .or_default()
            .push(source);
    }
    let mut attachment_by_owner: HashMap<CacheOwnerId, Vec<&KvStateAttachmentAdvertisement>> =
        HashMap::new();
    for (discovery_id, attachment) in attachments {
        if attachment_record_id(attachment.publisher_id, attachment.attachment_generation)
            != *discovery_id
            || attachment.protocol_version != KvStateProtocolVersion::V2
        {
            continue;
        }
        attachment_by_owner
            .entry(attachment.cache_owner_id)
            .or_default()
            .push(attachment);
    }

    // Publish source-mode selection before applying any replacement reset or
    // snapshot. The legacy event client consults this watch at admission, so a
    // late legacy event cannot repopulate Worker ownership during activation.
    let mut provisional_recognized = source_mode_suppressed_workers(&membership);
    for (owner, matching_sources) in &source_by_owner {
        let [source] = matching_sources.as_slice() else {
            if matching_sources.len() > 1
                && let Some(worker) = runtime
                    .get(owner)
                    .and_then(|state| state.last_worker)
                    .filter(|worker| live_workers.contains(worker))
            {
                provisional_recognized.insert(worker);
            }
            continue;
        };
        let matching_attachments = attachment_by_owner
            .get(owner)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if let [attachment] = matching_attachments
            && attachment.publisher_id == source.publisher_id
            && attachment.protocol_version == source.protocol_version
            && attachment.worker.dp_rank == source.global_dp_rank
            && live_workers.contains(&attachment.worker)
        {
            provisional_recognized.insert(attachment.worker);
        } else if let Some(worker) = runtime
            .get(owner)
            .and_then(|state| state.last_worker)
            .filter(|worker| live_workers.contains(worker))
        {
            provisional_recognized.insert(worker);
        }
    }
    // Attachment presence is itself an explicit state-agent mode claim. This
    // closes the source/attachment discovery ordering window without treating
    // the attachment as sufficient for routing readiness.
    for matching_attachments in attachment_by_owner.values() {
        for attachment in matching_attachments {
            if live_workers.contains(&attachment.worker) {
                provisional_recognized.insert(attachment.worker);
            }
        }
    }
    recognized_tx.send_replace(provisional_recognized.clone());

    let known_owners: HashSet<_> = runtime
        .keys()
        .copied()
        .chain(source_by_owner.keys().copied())
        .chain(attachment_by_owner.keys().copied())
        .collect();
    recovery_tails.retain(|owner, tail| {
        let keep = source_by_owner.get(owner).is_some_and(|sources| {
            sources
                .iter()
                .any(|source| source.publisher_id == tail.publisher_id)
        });
        if !keep {
            recovering_publishers.remove(&tail.publisher_id);
        }
        keep
    });
    for owner in &known_owners {
        if let Some(state) = runtime.get_mut(owner) {
            state.ready = false;
        }
    }
    // Reconciliation performs asynchronous clears and recovery. Withdraw the
    // affected projection before any fallible work so an early error cannot
    // leave a stale CacheOwner mapping routable.
    publish_projection(indexer, runtime);
    for owner in known_owners {
        let matching_sources = source_by_owner
            .get(&owner)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if matching_sources.len() > 1 {
            if let Some(tail) = recovery_tails.remove(&owner) {
                recovering_publishers.remove(&tail.publisher_id);
            }
            if let Some(state) = runtime.get_mut(&owner) {
                // No overlapping source incarnation is authoritative. Fence
                // live traffic until one exact source wins and recovers.
                state.publisher_id = None;
                state.attachment_generation = None;
                state.ready = false;
                state.recovery_required = true;
            }
            tracing::warn!(%owner, "Ambiguous KV state-source incarnations; retaining exact ownership unprojected");
            continue;
        }
        let Some(source) = matching_sources.first().copied() else {
            if let Some(tail) = recovery_tails.remove(&owner) {
                recovering_publishers.remove(&tail.publisher_id);
            }
            if let Some(previous) = runtime.remove(&owner) {
                if let Some(worker) = previous.last_worker {
                    clear_worker(indexer, worker, owner).await?;
                }
                clear_cache_owner(indexer, owner, 0).await?;
            }
            if let Some(candidates) = attachment_by_owner.get(&owner) {
                for attachment in candidates {
                    if live_workers.contains(&attachment.worker) {
                        clear_worker(indexer, attachment.worker, owner).await?;
                    }
                }
            }
            continue;
        };
        let matching = attachment_by_owner
            .get(&owner)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let attachment = match matching {
            [attachment]
                if attachment.publisher_id == source.publisher_id
                    && attachment.protocol_version == source.protocol_version
                    && attachment.worker.dp_rank == source.global_dp_rank
                    && attachment.recovery_control_target == source.recovery_control_target
                    && attachment.ingress_protocol == source.ingress_protocol =>
            {
                Some(*attachment)
            }
            _ => None,
        };

        if attachment.is_none() {
            for candidate in matching {
                if live_workers.contains(&candidate.worker) {
                    clear_worker(indexer, candidate.worker, owner).await?;
                }
            }
        }

        let previous = runtime.get(&owner).copied();
        let expected_generation = attachment.map(|value| value.attachment_generation);
        let last_worker = attachment
            .map(|value| value.worker)
            .or_else(|| previous.and_then(|value| value.last_worker));
        if let Some(previous) = previous
            && (previous.publisher_id != Some(source.publisher_id)
                || previous.attachment_generation != expected_generation)
            && let Some(worker) = previous.last_worker
        {
            clear_worker(indexer, worker, owner).await?;
        }
        if let Some(worker) = last_worker
            && !live_workers.contains(&worker)
            && previous.is_some_and(|value| value.last_worker == Some(worker))
        {
            clear_worker(indexer, worker, owner).await?;
        }
        let recognized_worker = last_worker.filter(|worker| live_workers.contains(worker));
        if previous.is_none()
            && let Some(worker) = recognized_worker
        {
            clear_worker(indexer, worker, owner).await?;
        }
        let previous_cursor = previous
            .filter(|value| value.publisher_id == Some(source.publisher_id))
            .map_or(0, |value| value.recovered_cursor);
        let recovery_required = previous.is_none_or(|value| {
            value.publisher_id != Some(source.publisher_id)
                || value.attachment_generation != expected_generation
                || value.recovery_required
        });

        let expected = KvStateAgentIdentity {
            cache_owner_id: owner,
            publisher_id: source.publisher_id,
            protocol_version: source.protocol_version,
        };
        runtime.insert(
            owner,
            OwnerRuntime {
                publisher_id: Some(source.publisher_id),
                recovered_cursor: previous_cursor,
                attachment_generation: expected_generation,
                last_worker: recognized_worker,
                ready: false,
                recovery_required,
            },
        );
        let tail = recovery_tails
            .entry(owner)
            .or_insert_with(|| OwnerRecoveryTail {
                publisher_id: source.publisher_id,
                attachment_generation: expected_generation,
                schedule_generation,
                rank: RankState::default(),
            });
        if tail.publisher_id != source.publisher_id
            || tail.attachment_generation != expected_generation
        {
            recovering_publishers.remove(&tail.publisher_id);
            *tail = OwnerRecoveryTail {
                publisher_id: source.publisher_id,
                attachment_generation: expected_generation,
                schedule_generation,
                rank: RankState::default(),
            };
        } else {
            tail.schedule_generation = schedule_generation;
        }
        recovering_publishers.insert(source.publisher_id, owner);
        launch_initial_recovery(
            recovery_lane,
            recovery_tx,
            transport.clone(),
            OwnerRecoveryPlan {
                owner,
                source: source.clone(),
                attachment: attachment.cloned(),
                expected,
                recognized_worker,
                previous_publisher_id: previous.and_then(|value| value.publisher_id),
                previous_cursor,
                recovery_required,
                observation_revision: expected_revision,
                schedule_generation,
            },
        );
    }

    let mut recognized = provisional_recognized;
    recognized.extend(runtime.values().filter_map(|value| value.last_worker));
    recognized_tx.send_replace(recognized);
    publish_projection_if_current(
        indexer,
        runtime,
        observation_revision,
        projection_commit,
        expected_revision,
    )
    .await?;
    Ok(())
}

async fn publish_projection_if_current(
    indexer: &Indexer,
    runtime: &HashMap<CacheOwnerId, OwnerRuntime>,
    observation_revision: &AtomicU64,
    projection_commit: &Mutex<()>,
    expected_revision: u64,
) -> Result<()> {
    let _commit = projection_commit.lock().await;
    ensure_observation_revision(observation_revision, expected_revision)?;
    publish_projection(indexer, runtime);
    Ok(())
}

impl OwnerRecoveryResult {
    fn owner(&self) -> CacheOwnerId {
        match self {
            Self::Initial { plan, .. } | Self::PostStatus { plan, .. } => plan.owner,
        }
    }

    fn publisher_id(&self) -> u64 {
        match self {
            Self::Initial { plan, .. } | Self::PostStatus { plan, .. } => plan.source.publisher_id,
        }
    }

    fn schedule_generation(&self) -> u64 {
        match self {
            Self::Initial { plan, .. } | Self::PostStatus { plan, .. } => plan.schedule_generation,
        }
    }
}

fn launch_initial_recovery(
    lane: &Arc<RecoveryLane<(CacheOwnerId, u64)>>,
    tx: &mpsc::UnboundedSender<OwnerRecoveryResult>,
    transport: Arc<RuntimeWorkerQueryTransport>,
    plan: OwnerRecoveryPlan,
) {
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task_tx = tx.clone();
    let semaphore = lane.semaphore();
    let owner = plan.owner;
    let schedule_generation = plan.schedule_generation;
    let handle = tokio::spawn(async move {
        let status_query = transport.query_status(
            plan.attachment
                .as_ref()
                .map_or(0, |value| value.worker.worker_id),
            plan.source.global_dp_rank,
            plan.source.recovery_control_target.clone(),
            plan.expected.clone(),
            plan.attachment
                .as_ref()
                .map(|value| value.attachment_generation),
            CONTROL_TIMEOUT,
        );
        let status = tokio::select! {
            biased;
            _ = task_cancel.cancelled() => return,
            result = status_query => result,
        };
        let needs_recovery = status.as_ref().is_ok_and(|status| {
            plan.recovery_required
                || plan.previous_cursor
                    < plan
                        .attachment
                        .as_ref()
                        .map_or(status.outbound_cursor, |value| {
                            value.ready_at_outbound_cursor
                        })
        });
        let recovery = if needs_recovery {
            let permit = tokio::select! {
                biased;
                _ = task_cancel.cancelled() => return,
                permit = semaphore.acquire_owned() => match permit {
                    Ok(permit) => permit,
                    Err(_) => return,
                },
            };
            let query = transport.query_state_agent_recovery(
                plan.attachment
                    .as_ref()
                    .map_or(0, |value| value.worker.worker_id),
                plan.source.global_dp_rank,
                plan.source.recovery_control_target.clone(),
                plan.expected.clone(),
                plan.attachment
                    .as_ref()
                    .map(|value| value.attachment_generation),
                RECOVERY_TIMEOUT,
            );
            let result = tokio::select! {
                biased;
                _ = task_cancel.cancelled() => return,
                result = query => result,
            };
            drop(permit);
            Some(result)
        } else {
            None
        };
        let _ = task_tx.send(OwnerRecoveryResult::Initial {
            plan,
            status,
            recovery,
        });
    });
    lane.insert((owner, schedule_generation), cancel, handle);
}

fn launch_post_status(
    lane: &Arc<RecoveryLane<(CacheOwnerId, u64)>>,
    tx: &mpsc::UnboundedSender<OwnerRecoveryResult>,
    transport: Arc<RuntimeWorkerQueryTransport>,
    plan: OwnerRecoveryPlan,
    recovered_cursor: u64,
) {
    let cancel = CancellationToken::new();
    let task_cancel = cancel.clone();
    let task_tx = tx.clone();
    let owner = plan.owner;
    let schedule_generation = plan.schedule_generation;
    let handle = tokio::spawn(async move {
        let query = transport.query_status(
            plan.attachment
                .as_ref()
                .map_or(0, |value| value.worker.worker_id),
            plan.source.global_dp_rank,
            plan.source.recovery_control_target.clone(),
            plan.expected.clone(),
            plan.attachment
                .as_ref()
                .map(|value| value.attachment_generation),
            CONTROL_TIMEOUT,
        );
        let status = tokio::select! {
            biased;
            _ = task_cancel.cancelled() => return,
            result = query => result,
        };
        let _ = task_tx.send(OwnerRecoveryResult::PostStatus {
            plan,
            recovered_cursor,
            status,
        });
    });
    lane.insert((owner, schedule_generation), cancel, handle);
}

#[allow(clippy::too_many_arguments)]
async fn finish_owner_recovery(
    result: OwnerRecoveryResult,
    indexer: &Indexer,
    transport: Option<&Arc<RuntimeWorkerQueryTransport>>,
    membership: &KvSourceMembershipWatch,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    lane: &Arc<RecoveryLane<(CacheOwnerId, u64)>>,
    tx: &mpsc::UnboundedSender<OwnerRecoveryResult>,
    tails: &mut HashMap<CacheOwnerId, OwnerRecoveryTail>,
    recovering_publishers: &mut HashMap<u64, CacheOwnerId>,
    observation_revision: &AtomicU64,
    projection_commit: &Mutex<()>,
) -> Result<()> {
    let plan = match &result {
        OwnerRecoveryResult::Initial { plan, .. }
        | OwnerRecoveryResult::PostStatus { plan, .. } => plan,
    };
    if !owner_recovery_is_current(plan, membership, sources, attachments, observation_revision) {
        return Ok(());
    }
    if !tails
        .get(&plan.owner)
        .is_some_and(|tail| tail.schedule_generation == plan.schedule_generation)
    {
        return Ok(());
    }
    let expected_revision = plan.observation_revision;

    match result {
        OwnerRecoveryResult::Initial {
            plan,
            status,
            recovery,
        } => {
            let status = match status {
                Ok(status) => status,
                Err(error) => {
                    fail_owner_recovery(
                        plan.owner,
                        plan.source.publisher_id,
                        runtime,
                        tails,
                        recovering_publishers,
                    );
                    tracing::warn!(owner = %plan.owner, %error, "KV state-agent status handshake failed");
                    return Ok(());
                }
            };
            let recovered_cursor = if let Some(recovery) = recovery {
                let response = match recovery {
                    Ok(response) => response,
                    Err(error) => {
                        fail_owner_recovery(
                            plan.owner,
                            plan.source.publisher_id,
                            runtime,
                            tails,
                            recovering_publishers,
                        );
                        tracing::warn!(owner = %plan.owner, %error, "KV state-agent recovery query failed");
                        return Ok(());
                    }
                };
                if plan.recovery_required
                    && plan.previous_publisher_id != Some(plan.source.publisher_id)
                {
                    clear_cache_owner(indexer, plan.owner, 0).await?;
                }
                if let Some(worker) = plan.recognized_worker {
                    clear_worker(indexer, worker, plan.owner).await?;
                }
                let recovered_cursor = apply_recovery_response(
                    indexer,
                    plan.owner,
                    plan.recognized_worker,
                    &plan.expected,
                    plan.attachment
                        .as_ref()
                        .map(|value| value.attachment_generation),
                    response,
                )
                .await?;
                apply_recovery_tail(indexer, &plan, recovered_cursor, runtime, tails).await?;
                let Some(transport) = transport else {
                    anyhow::bail!("state-agent recovery transport disappeared");
                };
                launch_post_status(lane, tx, (*transport).clone(), plan, recovered_cursor);
                return Ok(());
            } else {
                apply_recovery_tail(indexer, &plan, plan.previous_cursor, runtime, tails).await?
            };
            finish_owner_readiness(
                &plan,
                status,
                recovered_cursor,
                membership,
                runtime,
                tails,
                recovering_publishers,
            );
        }
        OwnerRecoveryResult::PostStatus {
            plan,
            recovered_cursor,
            status,
        } => {
            let status = match status {
                Ok(status) => status,
                Err(error) => {
                    fail_owner_recovery(
                        plan.owner,
                        plan.source.publisher_id,
                        runtime,
                        tails,
                        recovering_publishers,
                    );
                    tracing::warn!(owner = %plan.owner, %error, "KV state-agent post-recovery status check failed");
                    return Ok(());
                }
            };
            let recovered_cursor =
                apply_recovery_tail(indexer, &plan, recovered_cursor, runtime, tails).await?;
            finish_owner_readiness(
                &plan,
                status,
                recovered_cursor,
                membership,
                runtime,
                tails,
                recovering_publishers,
            );
        }
    }
    publish_projection_if_current(
        indexer,
        runtime,
        observation_revision,
        projection_commit,
        expected_revision,
    )
    .await
}

fn owner_recovery_is_current(
    plan: &OwnerRecoveryPlan,
    membership: &KvSourceMembershipWatch,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    observation_revision: &AtomicU64,
) -> bool {
    if observation_revision.load(Ordering::Acquire) != plan.observation_revision
        || sources.get(&plan.source.publisher_id) != Some(&plan.source)
        || plan
            .recognized_worker
            .is_some_and(|worker| !membership.borrow().sources.contains_key(&worker))
    {
        return false;
    }
    match &plan.attachment {
        Some(attachment) => {
            attachments.get(&attachment_record_id(
                attachment.publisher_id,
                attachment.attachment_generation,
            )) == Some(attachment)
        }
        None => true,
    }
}

async fn apply_recovery_tail(
    indexer: &Indexer,
    plan: &OwnerRecoveryPlan,
    recovered_cursor: u64,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    tails: &mut HashMap<CacheOwnerId, OwnerRecoveryTail>,
) -> Result<u64> {
    let events = tails
        .get_mut(&plan.owner)
        .filter(|tail| {
            tail.publisher_id == plan.source.publisher_id
                && tail.attachment_generation
                    == plan
                        .attachment
                        .as_ref()
                        .map(|value| value.attachment_generation)
        })
        .map_or_else(Vec::new, |tail| {
            tail.rank.drain_advisory_tail_after(recovered_cursor)
        });
    let state = runtime.entry(plan.owner).or_insert(OwnerRuntime {
        publisher_id: Some(plan.source.publisher_id),
        recovered_cursor,
        attachment_generation: plan
            .attachment
            .as_ref()
            .map(|value| value.attachment_generation),
        last_worker: plan.recognized_worker,
        ready: false,
        recovery_required: false,
    });
    state.recovered_cursor = recovered_cursor;
    state.recovery_required = false;
    apply_events_for_owner(indexer, plan.owner, plan.source.publisher_id, events, state).await;
    if state.recovery_required {
        anyhow::bail!("buffered state-agent recovery tail was rejected");
    }
    Ok(state.recovered_cursor)
}

fn finish_owner_readiness(
    plan: &OwnerRecoveryPlan,
    status: KvStateAgentStatus,
    recovered_cursor: u64,
    membership: &KvSourceMembershipWatch,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    tails: &mut HashMap<CacheOwnerId, OwnerRecoveryTail>,
    recovering_publishers: &mut HashMap<u64, CacheOwnerId>,
) {
    let membership = membership.borrow();
    let live_workers = &membership.sources;
    let ready = plan.attachment.as_ref().is_some_and(|attachment| {
        attachment.cache_readable
            && live_workers.contains_key(&attachment.worker)
            && status.cache_owner_ready
            && status.identity.publisher_id == plan.source.publisher_id
            && status.attachment.as_ref().is_some_and(|status_attachment| {
                status_attachment.ready
                    && status_attachment.cache_readable
                    && status_attachment.generation == attachment.attachment_generation
                    && status_attachment.worker == attachment.worker
                    && status_attachment.ready_at_outbound_cursor
                        == attachment.ready_at_outbound_cursor
            })
            && recovered_cursor >= attachment.ready_at_outbound_cursor
    });
    runtime.insert(
        plan.owner,
        OwnerRuntime {
            publisher_id: Some(plan.source.publisher_id),
            recovered_cursor,
            attachment_generation: plan
                .attachment
                .as_ref()
                .map(|value| value.attachment_generation),
            last_worker: plan.recognized_worker,
            ready,
            recovery_required: false,
        },
    );
    tails.remove(&plan.owner);
    recovering_publishers.remove(&plan.source.publisher_id);
}

fn fail_owner_recovery(
    owner: CacheOwnerId,
    publisher_id: u64,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    tails: &mut HashMap<CacheOwnerId, OwnerRecoveryTail>,
    recovering_publishers: &mut HashMap<u64, CacheOwnerId>,
) {
    if let Some(state) = runtime.get_mut(&owner) {
        state.ready = false;
        state.recovery_required = true;
    }
    tails.remove(&owner);
    recovering_publishers.remove(&publisher_id);
}

fn ensure_observation_revision(revision: &AtomicU64, expected: u64) -> Result<()> {
    if revision.load(Ordering::Acquire) != expected {
        anyhow::bail!("KV state reconciliation was superseded by a newer observation");
    }
    Ok(())
}

async fn apply_recovery_response(
    indexer: &Indexer,
    owner: CacheOwnerId,
    worker: Option<WorkerWithDpRank>,
    expected: &KvStateAgentIdentity,
    expected_generation: Option<u64>,
    response: WorkerKvQueryResponse,
) -> Result<u64> {
    let WorkerKvQueryResponse::StateAgentRecovery { response, receipt } = response else {
        anyhow::bail!("state-agent recovery returned an unexpected response");
    };
    if receipt.identity != *expected || receipt.attachment_generation != expected_generation {
        anyhow::bail!("state-agent recovery receipt does not match the selected incarnation");
    }
    match *response {
        WorkerKvQueryResponse::TreeDump {
            events,
            last_event_id,
            reset_scope,
        } => {
            validate_recovery_events(&events, owner, worker)?;
            match reset_scope {
                ResetScope::All => {
                    if let Some(worker) = worker {
                        clear_worker(indexer, worker, owner).await?;
                    }
                    clear_cache_owner(indexer, owner, last_event_id).await?;
                }
                ResetScope::Domain(ResidencyDomain::CacheOwner) => {
                    clear_cache_owner(indexer, owner, last_event_id).await?;
                }
                ResetScope::Domain(ResidencyDomain::Worker) => {
                    if let Some(worker) = worker {
                        clear_worker(indexer, worker, owner).await?;
                    }
                }
            }
            for event in events {
                indexer.try_apply_event(event).await?;
            }
            Ok(receipt.recovered_through_cursor.max(last_event_id))
        }
        WorkerKvQueryResponse::Events {
            events,
            last_event_id,
        } => {
            validate_recovery_events(&events, owner, worker)?;
            for event in events {
                indexer.try_apply_event(event).await?;
            }
            Ok(receipt.recovered_through_cursor.max(last_event_id))
        }
        WorkerKvQueryResponse::TreeDumpFailed { message, .. } => {
            anyhow::bail!("state-agent snapshot failed: {message}")
        }
        WorkerKvQueryResponse::Error(message) => {
            anyhow::bail!("state-agent recovery failed: {message}")
        }
        _ => anyhow::bail!("state-agent recovery returned no applicable state"),
    }
}

fn validate_recovery_events(
    events: &[RouterEvent],
    owner: CacheOwnerId,
    worker: Option<WorkerWithDpRank>,
) -> Result<()> {
    for event in events {
        let valid = event
            .resolved_residency_domain()
            .is_ok_and(|domain| match domain {
                ResidencyDomain::CacheOwner => event.state_source == Some(owner),
                ResidencyDomain::Worker => worker.is_some_and(|worker| {
                    worker.worker_id == event.worker_id && worker.dp_rank == event.event.dp_rank
                }),
            });
        if !valid {
            anyhow::bail!(
                "state-agent recovery contains ownership outside the selected source or attachment"
            );
        }
    }
    Ok(())
}

async fn apply_live_events(
    indexer: &Indexer,
    publisher_id: u64,
    events: Vec<RouterEvent>,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
) -> bool {
    let Some((owner, state)) = runtime
        .iter_mut()
        .find(|(_, state)| state.publisher_id == Some(publisher_id))
    else {
        return false;
    };
    let was_ready = state.ready;
    apply_events_for_owner(indexer, *owner, publisher_id, events, state).await;
    was_ready != state.ready
}

async fn apply_events_for_owner(
    indexer: &Indexer,
    owner: CacheOwnerId,
    publisher_id: u64,
    events: Vec<RouterEvent>,
    state: &mut OwnerRuntime,
) {
    for event in events {
        if state.recovery_required {
            break;
        }
        let event_id = event.event.event_id;
        if event_id <= state.recovered_cursor {
            continue;
        }
        if let Some(expected) = state.recovered_cursor.checked_add(1)
            && event_id > expected
        {
            tracing::warn!(
                publisher_id,
                expected,
                actual = event_id,
                "KV state-agent event gap; continuing advisory ingestion"
            );
        }
        let domain = event.resolved_residency_domain();
        if matches!(domain.as_ref(), Ok(ResidencyDomain::Worker))
            && state.attachment_generation.is_none()
        {
            tracing::warn!(
                publisher_id,
                event_id,
                "Ignoring stale Worker event while state source is detached"
            );
            state.recovered_cursor = event_id;
            continue;
        }
        let valid = domain.is_ok_and(|domain| match domain {
            ResidencyDomain::CacheOwner => event.state_source == Some(owner),
            ResidencyDomain::Worker => state.last_worker.is_some_and(|worker| {
                worker.worker_id == event.worker_id && worker.dp_rank == event.event.dp_rank
            }),
        });
        if !valid {
            tracing::warn!(
                publisher_id,
                event_id,
                "Ignoring incorrectly attributed KV state event"
            );
            state.ready = false;
            state.recovery_required = true;
            break;
        }
        let is_clear = matches!(event.event.data, KvCacheEventData::Cleared);
        if let Err(error) = indexer.try_apply_event(event).await {
            tracing::warn!(publisher_id, event_id, %error, "Failed to apply advisory KV state event");
            if is_clear {
                state.ready = false;
                state.recovery_required = true;
                break;
            }
        }
        state.recovered_cursor = event_id;
    }
}

fn publish_projection(indexer: &Indexer, runtime: &HashMap<CacheOwnerId, OwnerRuntime>) {
    let projection = ResidencyProjection::new(runtime.iter().filter_map(|(owner, state)| {
        state
            .ready
            .then_some(state.last_worker)
            .flatten()
            .map(|worker| (*owner, worker))
    }))
    .unwrap_or_default();
    indexer.set_residency_projection(projection);
}

async fn clear_worker(
    indexer: &Indexer,
    worker: WorkerWithDpRank,
    owner: CacheOwnerId,
) -> Result<()> {
    indexer
        .try_apply_event(
            RouterEvent::with_residency_domain(
                worker.worker_id,
                KvCacheEvent {
                    event_id: 0,
                    data: KvCacheEventData::Cleared,
                    dp_rank: worker.dp_rank,
                },
                StorageTier::Device,
                ResidencyDomain::Worker,
            )
            .with_state_source(owner),
        )
        .await?;
    Ok(())
}

async fn clear_cache_owner(indexer: &Indexer, owner: CacheOwnerId, event_id: u64) -> Result<()> {
    indexer
        .try_apply_event(RouterEvent::with_cache_owner(
            0,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
            StorageTier::Device,
            owner,
        ))
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
        StableDpSlotId,
    };
    use dynamo_kv_router::indexer::{
        KvIndexer, KvIndexerInterface, KvIndexerMetrics, KvStateRecoveryReceipt, LowerTierIndexers,
    };
    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData,
        LocalBlockHash, WorkerWithDpRank,
    };
    use dynamo_runtime::protocols::EndpointId;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::discovery::{KvEventSource, KvSourceMembershipView, KvStateEndpointResolution};

    fn owner(slot: u8) -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([slot; 16], IdentitySource::Explicit),
        )
    }

    fn test_indexer() -> (KvIndexer, Indexer) {
        let primary = KvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
        );
        (
            primary.clone(),
            Indexer::KvIndexer {
                primary,
                lower_tier: LowerTierIndexers::new(1, 4),
                approx: None,
                primary_records_routing_decisions: false,
            },
        )
    }

    #[test]
    fn recognized_state_source_suppresses_legacy_without_reporting_missing() {
        let worker = WorkerWithDpRank::new(17, 3);
        let endpoint = EndpointId::from("ns.backend.generate");
        let source = KvEventSource {
            kv_state_endpoint: endpoint.clone(),
            worker,
            publisher_id: 41,
            recovery_target: None,
        };
        let view = KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint),
            sources: HashMap::from([(worker, KvSourceStatus::ActiveLiveOnly(source))]),
            kv_event_publishing_enabled: HashMap::from([(worker.worker_id, Some(true))]),
            kv_event_source_mode: HashMap::new(),
            recovery_expected: HashMap::from([(worker, false)]),
        };
        let filtered = suppress_recognized(view, &HashSet::from([worker]));
        assert_eq!(filtered.status(&worker), Some(&KvSourceStatus::Suppressed));
    }

    #[test]
    fn explicit_source_mode_suppresses_legacy_before_state_source_is_ready() {
        let worker = WorkerWithDpRank::new(7, 3);
        let endpoint = EndpointId::from("ns.backend.generate");
        let mut view = KvSourceMembershipView {
            serving_endpoint: endpoint.clone(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint),
            sources: HashMap::from([(worker, KvSourceStatus::Missing)]),
            kv_event_publishing_enabled: HashMap::new(),
            kv_event_source_mode: HashMap::from([(
                worker.worker_id,
                Some("state_agent_v2".to_string()),
            )]),
            recovery_expected: HashMap::new(),
        };
        let recognized = source_mode_suppressed_workers(&view);
        assert_eq!(recognized, HashSet::from([worker]));
        view.kv_event_source_mode
            .insert(worker.worker_id, Some("future_mode".to_string()));
        assert_eq!(source_mode_suppressed_workers(&view), recognized);
    }

    #[tokio::test]
    async fn recovery_rejects_ownership_from_another_state_source() {
        let selected = owner(4);
        let other = owner(5);
        let identity = KvStateAgentIdentity {
            cache_owner_id: selected,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
        };
        let response = WorkerKvQueryResponse::StateAgentRecovery {
            response: Box::new(WorkerKvQueryResponse::Events {
                events: vec![RouterEvent::with_cache_owner(
                    0,
                    KvCacheEvent {
                        event_id: 1,
                        data: KvCacheEventData::Removed(KvCacheRemoveData {
                            block_hashes: vec![ExternalSequenceBlockHash(7)],
                        }),
                        dp_rank: 0,
                    },
                    StorageTier::HostPinned,
                    other,
                )],
                last_event_id: 1,
            }),
            receipt: KvStateRecoveryReceipt {
                identity: identity.clone(),
                attachment_generation: None,
                recovered_through_cursor: 1,
            },
        };

        let error =
            apply_recovery_response(&Indexer::None, selected, None, &identity, None, response)
                .await
                .unwrap_err();
        assert!(error.to_string().contains("ownership outside"));
    }

    #[tokio::test]
    async fn detached_source_ignores_late_worker_events_until_reattached() {
        let owner = owner(4);
        let worker = WorkerWithDpRank::new(17, 3);
        let (primary, indexer) = test_indexer();
        let mut runtime = HashMap::from([(
            owner,
            OwnerRuntime {
                publisher_id: Some(41),
                recovered_cursor: 0,
                attachment_generation: None,
                last_worker: Some(worker),
                ready: false,
                recovery_required: false,
            },
        )]);
        let event = |event_id| {
            RouterEvent::with_residency_domain(
                worker.worker_id,
                KvCacheEvent {
                    event_id,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        start_position: None,
                        blocks: vec![KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(event_id),
                            tokens_hash: LocalBlockHash(event_id),
                            mm_extra_info: None,
                        }],
                    }),
                    dp_rank: worker.dp_rank,
                },
                StorageTier::Device,
                ResidencyDomain::Worker,
            )
            .with_state_source(owner)
        };

        apply_live_events(&indexer, 41, vec![event(1)], &mut runtime).await;
        assert!(primary.dump_events().await.unwrap().is_empty());

        runtime.get_mut(&owner).unwrap().attachment_generation = Some(2);
        apply_live_events(&indexer, 41, vec![event(2)], &mut runtime).await;
        assert_eq!(primary.dump_events().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn incorrectly_attributed_live_event_withdraws_projection_eligibility() {
        let selected = owner(4);
        let other = owner(5);
        let worker = WorkerWithDpRank::new(17, 3);
        let mut runtime = HashMap::from([(
            selected,
            OwnerRuntime {
                publisher_id: Some(41),
                recovered_cursor: 0,
                attachment_generation: Some(2),
                last_worker: Some(worker),
                ready: true,
                recovery_required: false,
            },
        )]);
        let event = RouterEvent::with_cache_owner(
            worker.worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(7)],
                }),
                dp_rank: worker.dp_rank,
            },
            StorageTier::HostPinned,
            other,
        );

        assert!(apply_live_events(&Indexer::None, 41, vec![event], &mut runtime).await);
        let state = runtime.get(&selected).unwrap();
        assert!(!state.ready);
        assert!(state.recovery_required);
        assert_eq!(state.recovered_cursor, 0);
    }
}
