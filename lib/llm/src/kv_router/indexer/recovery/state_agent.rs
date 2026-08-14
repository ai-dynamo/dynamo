// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Version-aware KV state-source selection and router recovery.
//!
//! This controller lives at one router/indexer boundary. It owns source-mode
//! selection and publishes one aggregate CacheOwner projection; router-core
//! remains unaware of discovery and attachment lifecycles.

use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use anyhow::{Context, Result};
use dynamo_kv_router::{
    identity::CacheOwnerId,
    indexer::{KvStateAgentIdentity, KvStateProtocolVersion, WorkerKvQueryResponse},
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
use tokio::sync::{oneshot, watch};
use tokio_util::sync::CancellationToken;

use crate::discovery::kv_state_agent::{
    KV_STATE_ATTACHMENT_TOPIC_V2, KV_STATE_EVENT_TOPIC_V2, KV_STATE_SOURCE_TOPIC_V2,
    KvStateAttachmentAdvertisement, KvStateSourceAdvertisement,
};
use crate::{
    discovery::{KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus},
    kv_router::Indexer,
};

use super::RuntimeWorkerQueryTransport;

const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
const RECOVERY_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy)]
struct OwnerRuntime {
    publisher_id: u64,
    recovered_cursor: u64,
    attachment_generation: Option<u64>,
    last_worker: Option<WorkerWithDpRank>,
    ready: bool,
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

    let (recognized_tx, recognized_rx) = watch::channel(HashSet::new());
    let legacy_membership = filtered_legacy_membership(
        membership.clone(),
        recognized_rx,
        cancellation_token.child_token(),
    );
    let (completion_tx, completion_rx) = oneshot::channel();
    component.drt().runtime().secondary().spawn(async move {
        run_state_agent_router(
            indexer,
            membership,
            component,
            endpoint,
            expected_block_size,
            sources,
            recognized_tx,
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

#[allow(clippy::too_many_arguments)]
async fn run_state_agent_router(
    indexer: Indexer,
    mut membership: KvSourceMembershipWatch,
    component: Component,
    endpoint: EndpointId,
    expected_block_size: u32,
    mut source_stream: dynamo_runtime::discovery::DiscoveryStream,
    recognized_tx: watch::Sender<HashSet<WorkerWithDpRank>>,
    cancel: CancellationToken,
) {
    let mut attachment_stream: Option<dynamo_runtime::discovery::DiscoveryStream> = None;
    let mut event_stream: Option<
        dynamo_runtime::transports::event_plane::TypedEventSubscriber<Vec<RouterEvent>>,
    > = None;
    let mut transport: Option<RuntimeWorkerQueryTransport> = None;
    let mut sources = HashMap::new();
    let mut attachments = HashMap::new();
    let mut runtime: HashMap<CacheOwnerId, OwnerRuntime> = HashMap::new();

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
        };
        let mut reconcile = false;
        match input {
            Input::Cancelled => break,
            Input::Membership => {
                membership.borrow_and_update();
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
                if apply_live_events(&indexer, envelope.publisher_id, events, &mut runtime).await {
                    publish_projection(&indexer, &runtime);
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
        }
        if reconcile {
            let membership_snapshot = membership.borrow().clone();
            if !sources.is_empty() && transport.is_none() {
                match start_state_agent_consumers(&component, &endpoint, cancel.child_token()).await
                {
                    Ok((attachments, events, state_transport)) => {
                        attachment_stream = Some(attachments);
                        event_stream = Some(events);
                        transport = Some(state_transport);
                    }
                    Err(error) => {
                        tracing::warn!(%error, %endpoint, "Failed to activate V2 KV state consumers");
                        indexer.set_residency_projection(ResidencyProjection::default());
                        recognized_tx
                            .send_replace(membership_snapshot.sources.keys().copied().collect());
                        continue;
                    }
                }
            }
            let Some(transport) = transport.as_ref() else {
                continue;
            };
            if let Err(error) = reconcile_state_sources(
                &indexer,
                transport,
                membership_snapshot,
                &endpoint,
                expected_block_size,
                &sources,
                &attachments,
                &mut runtime,
                &recognized_tx,
            )
            .await
            {
                tracing::warn!(%error, %endpoint, "KV state-source reconciliation failed closed");
            }
        }
    }
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
async fn reconcile_state_sources(
    indexer: &Indexer,
    transport: &RuntimeWorkerQueryTransport,
    membership: KvSourceMembershipView,
    endpoint: &EndpointId,
    expected_block_size: u32,
    sources: &HashMap<u64, KvStateSourceAdvertisement>,
    attachments: &HashMap<u64, KvStateAttachmentAdvertisement>,
    runtime: &mut HashMap<CacheOwnerId, OwnerRuntime>,
    recognized_tx: &watch::Sender<HashSet<WorkerWithDpRank>>,
) -> Result<()> {
    let live_workers: HashSet<_> = membership.sources.keys().copied().collect();
    let mut source_by_owner: HashMap<CacheOwnerId, Vec<&KvStateSourceAdvertisement>> =
        HashMap::new();
    for (discovery_id, source) in sources {
        if source.publisher_id != *discovery_id
            || source.protocol_version != KvStateProtocolVersion::V2
            || source.event_topic != KV_STATE_EVENT_TOPIC_V2
            || source.kv_state_endpoint != *endpoint
            || source.kv_block_size != expected_block_size
            || source.ingress_protocol
                != crate::discovery::kv_state_agent::KvStateIngressProtocol::VllmResidencyV1
            || source.cache_owner_id.pool().indexer_domain() != source.indexer_domain_id
        {
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
        if attachment.publisher_id != *discovery_id
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
    let mut provisional_recognized = HashSet::new();
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
            if let Some(state) = runtime.get_mut(&owner) {
                // No overlapping source incarnation is authoritative. Fence
                // live traffic until one exact source wins and recovers.
                state.publisher_id = 0;
                state.attachment_generation = None;
                state.ready = false;
            }
            tracing::warn!(%owner, "Ambiguous KV state-source incarnations; retaining exact ownership unprojected");
            continue;
        }
        let Some(source) = matching_sources.first().copied() else {
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
            && (previous.publisher_id != source.publisher_id
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
            .filter(|value| value.publisher_id == source.publisher_id)
            .map_or(0, |value| value.recovered_cursor);

        let expected = KvStateAgentIdentity {
            cache_owner_id: owner,
            publisher_id: source.publisher_id,
            protocol_version: source.protocol_version,
        };
        let status = match transport
            .query_status(
                attachment.map_or(0, |value| value.worker.worker_id),
                source.global_dp_rank,
                source.recovery_control_target.clone(),
                expected.clone(),
                expected_generation,
                CONTROL_TIMEOUT,
            )
            .await
        {
            Ok(status) => status,
            Err(error) => {
                tracing::warn!(%owner, %error, "KV state-agent status handshake failed");
                runtime.insert(
                    owner,
                    OwnerRuntime {
                        publisher_id: source.publisher_id,
                        recovered_cursor: previous_cursor,
                        attachment_generation: expected_generation,
                        last_worker: recognized_worker,
                        ready: false,
                    },
                );
                continue;
            }
        };
        let needs_recovery = previous.is_none_or(|value| {
            value.publisher_id != source.publisher_id
                || value.attachment_generation != expected_generation
                || value.recovered_cursor
                    < attachment.map_or(status.outbound_cursor, |value| {
                        value.ready_at_outbound_cursor
                    })
        });
        let mut recovered_cursor = previous_cursor;
        if needs_recovery {
            if previous.is_some_and(|value| value.publisher_id != source.publisher_id) {
                clear_cache_owner(indexer, owner, 0).await?;
            }
            if let Some(worker) = recognized_worker {
                clear_worker(indexer, worker, owner).await?;
            }
            let response = transport
                .query_state_agent_recovery(
                    attachment.map_or(0, |value| value.worker.worker_id),
                    source.global_dp_rank,
                    source.recovery_control_target.clone(),
                    expected.clone(),
                    expected_generation,
                    RECOVERY_TIMEOUT,
                )
                .await;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    tracing::warn!(%owner, %error, "KV state-agent recovery query failed");
                    runtime.insert(
                        owner,
                        OwnerRuntime {
                            publisher_id: source.publisher_id,
                            recovered_cursor,
                            attachment_generation: expected_generation,
                            last_worker: recognized_worker,
                            ready: false,
                        },
                    );
                    continue;
                }
            };
            recovered_cursor = match apply_recovery_response(
                indexer,
                owner,
                recognized_worker,
                &expected,
                expected_generation,
                response,
            )
            .await
            {
                Ok(cursor) => cursor,
                Err(error) => {
                    tracing::warn!(%owner, %error, "KV state-agent recovery response was rejected");
                    runtime.insert(
                        owner,
                        OwnerRuntime {
                            publisher_id: source.publisher_id,
                            recovered_cursor,
                            attachment_generation: expected_generation,
                            last_worker: recognized_worker,
                            ready: false,
                        },
                    );
                    continue;
                }
            };
        }
        let post_status = if needs_recovery {
            match transport
                .query_status(
                    attachment.map_or(0, |value| value.worker.worker_id),
                    source.global_dp_rank,
                    source.recovery_control_target.clone(),
                    expected,
                    expected_generation,
                    CONTROL_TIMEOUT,
                )
                .await
            {
                Ok(status) => status,
                Err(error) => {
                    tracing::warn!(%owner, %error, "KV state-agent post-recovery status check failed");
                    runtime.insert(
                        owner,
                        OwnerRuntime {
                            publisher_id: source.publisher_id,
                            recovered_cursor,
                            attachment_generation: expected_generation,
                            last_worker: recognized_worker,
                            ready: false,
                        },
                    );
                    continue;
                }
            }
        } else {
            status
        };
        let ready = attachment.is_some_and(|attachment| {
            attachment.cache_readable
                && live_workers.contains(&attachment.worker)
                && post_status.cache_owner_ready
                && post_status.identity.publisher_id == source.publisher_id
                && post_status
                    .attachment
                    .as_ref()
                    .is_some_and(|status_attachment| {
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
            owner,
            OwnerRuntime {
                publisher_id: source.publisher_id,
                recovered_cursor,
                attachment_generation: expected_generation,
                last_worker: recognized_worker,
                ready,
            },
        );
    }

    let mut recognized = provisional_recognized;
    recognized.extend(runtime.values().filter_map(|value| value.last_worker));
    recognized_tx.send_replace(recognized);
    publish_projection(indexer, runtime);
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
        .find(|(_, state)| state.publisher_id == publisher_id)
    else {
        return false;
    };
    let was_ready = state.ready;
    for event in events {
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
        let valid = event
            .resolved_residency_domain()
            .is_ok_and(|domain| match domain {
                ResidencyDomain::CacheOwner => event.state_source == Some(*owner),
                ResidencyDomain::Worker => {
                    state.attachment_generation.is_some()
                        && state.last_worker.is_some_and(|worker| {
                            worker.worker_id == event.worker_id
                                && worker.dp_rank == event.event.dp_rank
                        })
                }
            });
        if !valid {
            tracing::warn!(
                publisher_id,
                event_id,
                "Ignoring incorrectly attributed KV state event"
            );
            state.ready = false;
            state.recovered_cursor = event_id;
            continue;
        }
        let is_clear = matches!(event.event.data, KvCacheEventData::Cleared);
        if let Err(error) = indexer.try_apply_event(event).await {
            tracing::warn!(publisher_id, event_id, %error, "Failed to apply advisory KV state event");
            if is_clear {
                state.ready = false;
            }
        }
        state.recovered_cursor = event_id;
    }
    was_ready != state.ready
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
            recovery_expected: HashMap::from([(worker, false)]),
        };
        let filtered = suppress_recognized(view, &HashSet::from([worker]));
        assert_eq!(filtered.status(&worker), Some(&KvSourceStatus::Suppressed));
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
                publisher_id: 41,
                recovered_cursor: 0,
                attachment_generation: None,
                last_worker: Some(worker),
                ready: false,
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
                publisher_id: 41,
                recovered_cursor: 0,
                attachment_generation: Some(2),
                last_worker: Some(worker),
                ready: true,
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
        assert_eq!(state.recovered_cursor, 1);
    }
}
