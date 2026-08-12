// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in vLLM KV state-agent foundation.
//!
//! One instance owns exactly one stable DP slot, one local recovery index, one
//! semantic event publisher, and one outbound cursor. Hosting is currently in
//! the vLLM handler process, but none of the durable identities derive from the
//! handler's DRT connection ID.
//!
//! This is a structural foundation, not restart persistence: the current vLLM
//! engine monitor terminates the handler and this agent on engine death.
//!
//! Cross-incarnation KVCC continuity is not implemented by this foundation.
//! For now, startup assumes a sunny-day stream: every committed KVCC residency
//! transition is delivered in order to this state-agent incarnation.
//!
//! Duplicate delivery is recoverable: router ownership mutations are
//! replay-convergent, and the future replay path will deduplicate stable KVCC
//! journal cursors before publisher-side refcount bookkeeping. Missing a
//! committed transition is not recoverable from the live stream alone and must
//! keep CacheOwner unready.
//!
//! TODO(#13044): require replacement KVCC/vLLM ingress to provide at-least-once
//! delivery across the incarnation boundary using an overlapping journal suffix
//! followed atomically by live events. A detected gap or replay-window miss must
//! fail closed and require a fuller CacheOwner resynchronization.

use std::time::Duration;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex as StdMutex},
};

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use dynamo_kv_router::{
    identity::{CacheOwnerId, CanonicalIdentityMaterial, ExplicitIdentityMap, StableDpSlotId},
    indexer::{
        KvIndexerMetrics, KvStateAgentIdentity, KvStateAgentStatus, KvStateAttachmentStatus,
        KvStateProtocolVersion, LocalKvIndexer,
    },
    protocols::{
        KvCacheEvent, KvCacheEventData, PlacementEvent, ResidencyDomain, RouterEvent, StorageTier,
        WorkerWithDpRank,
    },
    zmq_wire::{KvEventSourceKind, ZmqEventNormalizer},
};
use dynamo_runtime::{
    component::{Component, Endpoint, StartedEndpoint},
    discovery::{DiscoveryInstance, DiscoverySpec, EventScope},
    protocols::EndpointId,
    traits::DistributedRuntimeProvider,
    transports::event_plane::EventPublisher,
};
use futures::StreamExt;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::{
    discovery::{
        KvEventSource as DiscoveredKvEventSource,
        kv_state_agent::{
            KV_STATE_ATTACHMENT_TOPIC_V2, KV_STATE_EVENT_TOPIC_V2, KV_STATE_SOURCE_TOPIC_V2,
            KvStateAttachmentAdvertisement, KvStateSourceAdvertisement,
        },
    },
    kv_router::{
        KV_EVENT_SUBJECT, WORKER_KV_INDEXER_BUFFER_SIZE,
        indexer::start_worker_kv_query_endpoint_with_status,
    },
    utils::zmq::{connect_sub_socket, multipart_message},
};

use super::{
    dedup::EventDedupFilter,
    sinks::{EventPlanePublisher, RouterEventBatchSink},
};

const CONTROL_QUEUE_CAPACITY: usize = 16;
const INGRESS_QUEUE_CAPACITY: usize = 64;
const MAX_INGRESS_EVENTS: usize = 128;
const MAX_INGRESS_BLOCKS: usize = 8_192;
const MAX_ZMQ_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;
const PUBLISH_ATTEMPTS: usize = 3;
/// Out-of-band contract for the one vLLM-originated raw stream.
///
/// Framework-only mode preserves current vLLM compatibility. CacheOwner
/// transitions are accepted only on the explicitly versioned mode, so an old
/// raw listener cannot silently reinterpret them as Worker residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvStateAgentRawMode {
    FrameworkV1,
    ResidencyV2,
}

/// Resolve an operator-supplied durable slot name once on the control path.
pub fn resolve_stable_dp_slot_id(explicit_slot: &str) -> Result<StableDpSlotId> {
    let explicit = ExplicitIdentityMap::new(BTreeMap::from([(
        "slot".to_string(),
        explicit_slot.to_string(),
    )]))?;
    let material = CanonicalIdentityMaterial::stable_dp_slot(&explicit);
    let digest = blake3::hash(material.bytes()).as_bytes()[..16]
        .try_into()
        .expect("BLAKE3 output is 32 bytes");
    Ok(StableDpSlotId::new(digest, material.source()))
}

#[derive(Debug, Clone)]
pub struct KvStateAgentSlotConfig {
    pub cache_owner_id: CacheOwnerId,
    pub worker: WorkerWithDpRank,
}

#[derive(Debug, Clone)]
pub struct KvStateAgentVllmSource {
    pub endpoint: String,
    pub topic: String,
    pub image_token_id: Option<u32>,
    pub raw_mode: KvStateAgentRawMode,
}

#[derive(Clone)]
pub struct KvStateAgentConfig {
    pub endpoint: Endpoint,
    pub kv_state_endpoint: EndpointId,
    pub slot: KvStateAgentSlotConfig,
    pub kv_block_size: u32,
    pub vllm_source: Option<KvStateAgentVllmSource>,
}

#[derive(Debug, Clone)]
struct Attachment {
    generation: u64,
    worker: WorkerWithDpRank,
    vllm_zmq_endpoint: String,
    ready: bool,
    cache_readable: bool,
    ready_at_outbound_cursor: u64,
}

impl Attachment {
    fn status(&self) -> KvStateAttachmentStatus {
        KvStateAttachmentStatus {
            generation: self.generation,
            worker: self.worker,
            ready: self.ready,
            cache_readable: self.cache_readable,
            ready_at_outbound_cursor: self.ready_at_outbound_cursor,
        }
    }
}

#[derive(Debug)]
struct IngressBatch {
    generation: u64,
    source_cursor: u64,
    chunk_index: u32,
    final_chunk: bool,
    events: Vec<PlacementEvent>,
    cache_owner_fault: Option<&'static str>,
}

#[derive(Default)]
struct SourceCursorState {
    completed: Option<u64>,
    active: Option<(u64, u32)>,
}

impl SourceCursorState {
    fn observe(&mut self, batch: &IngressBatch) -> bool {
        if batch.chunk_index == 0 {
            if self.active.is_some()
                || self
                    .completed
                    .is_some_and(|previous| previous.checked_add(1) != Some(batch.source_cursor))
            {
                return false;
            }
        } else if self.active != Some((batch.source_cursor, batch.chunk_index - 1)) {
            return false;
        }

        if batch.final_chunk {
            self.completed = Some(batch.source_cursor);
            self.active = None;
        } else {
            self.active = Some((batch.source_cursor, batch.chunk_index));
        }
        true
    }
}

enum ControlCommand {
    Attach {
        attachment: Attachment,
        response: oneshot::Sender<Result<Attachment>>,
    },
    BeginDetach {
        generation: u64,
        response: oneshot::Sender<Result<()>>,
    },
    CompleteDetach {
        generation: u64,
        response: oneshot::Sender<Result<u64>>,
    },
    Quarantine {
        generation: u64,
        reason: &'static str,
        response: Option<oneshot::Sender<()>>,
    },
    AbortAttach {
        generation: u64,
    },
    Shutdown,
}

struct Coordinator<P> {
    publisher: P,
    local_indexer: Arc<LocalKvIndexer>,
    identity: KvStateAgentIdentity,
    slot_dp_rank: u32,
    status: Arc<ArcSwap<KvStateAgentStatus>>,
    control_rx: mpsc::Receiver<ControlCommand>,
    ingress_rx: mpsc::Receiver<IngressBatch>,
    attachment: Option<Attachment>,
    ingress_generation: Option<u64>,
    next_outbound_id: u64,
    next_attachment_generation: u64,
    source_cursor: SourceCursorState,
    dedup: EventDedupFilter,
    worker_domain_failed: bool,
    cache_owner_domain_failed: bool,
    source_failed: bool,
}

impl<P: RouterEventBatchSink + 'static> Coordinator<P> {
    async fn run(mut self, cancel: CancellationToken) {
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                command = self.control_rx.recv() => {
                    let Some(command) = command else { break };
                    if self.handle_control(command).await {
                        break;
                    }
                }
                batch = self.ingress_rx.recv() => {
                    let Some(batch) = batch else { continue };
                    if let Err(error) = self.handle_ingress(batch).await {
                        tracing::error!(%error, "KV state-agent ingress failed");
                    }
                }
            }
        }
    }

    async fn handle_control(&mut self, command: ControlCommand) -> bool {
        match command {
            ControlCommand::Attach {
                mut attachment,
                response,
            } => {
                let result = self.attach(&mut attachment).await.map(|()| attachment);
                let _ = response.send(result);
                false
            }
            ControlCommand::BeginDetach {
                generation,
                response,
            } => {
                let result = self.begin_detach(generation);
                let _ = response.send(result);
                false
            }
            ControlCommand::CompleteDetach {
                generation,
                response,
            } => {
                let result = self.complete_detach(generation).await;
                let _ = response.send(result);
                false
            }
            ControlCommand::Quarantine {
                generation,
                reason,
                response,
            } => {
                self.quarantine(generation, reason);
                if let Some(response) = response {
                    let _ = response.send(());
                }
                false
            }
            ControlCommand::AbortAttach { generation } => {
                if self
                    .attachment
                    .as_ref()
                    .is_some_and(|attachment| attachment.generation == generation)
                {
                    self.attachment = None;
                    self.ingress_generation = None;
                    self.source_cursor = SourceCursorState::default();
                    self.publish_status();
                }
                false
            }
            ControlCommand::Shutdown => true,
        }
    }

    async fn attach(&mut self, attachment: &mut Attachment) -> Result<()> {
        if self.worker_domain_failed {
            anyhow::bail!("Worker domain is retired after a local/publish failure");
        }
        if self.attachment.is_some() {
            anyhow::bail!("KV state agent already has an attachment");
        }

        attachment.generation = self.next_attachment_generation;
        self.next_attachment_generation = self
            .next_attachment_generation
            .checked_add(1)
            .context("attachment generation exhausted")?;

        attachment.ready = false;
        self.attachment = Some(attachment.clone());
        self.publish_status();

        let barrier_cursor = match self
            .apply_and_publish_reset(attachment.worker, ResidencyDomain::Worker)
            .await
        {
            Ok(cursor) => cursor,
            Err(error) => {
                self.worker_domain_failed = true;
                self.quarantine(attachment.generation, "attach reset transaction failed");
                return Err(error)
                    .context("failed to establish Worker reset barrier before attach");
            }
        };
        attachment.ready_at_outbound_cursor = barrier_cursor;
        self.ingress_generation = Some(attachment.generation);
        attachment.ready = true;
        // NOTE: Resetting this per-attachment cursor does not prove KVCC continuity.
        // The foundation assumes the replacement stream delivered every prior
        // committed transition. Future reattachment must prove at-least-once
        // cross-incarnation delivery before CacheOwner becomes ready.
        self.source_cursor = SourceCursorState::default();
        self.attachment = Some(attachment.clone());
        self.publish_status();
        Ok(())
    }

    fn begin_detach(&mut self, generation: u64) -> Result<()> {
        let attachment = self
            .attachment
            .as_mut()
            .context("KV state agent is not attached")?;
        if attachment.generation != generation {
            anyhow::bail!("stale attachment generation {generation}");
        }
        attachment.ready = false;
        self.publish_status();
        Ok(())
    }

    async fn complete_detach(&mut self, generation: u64) -> Result<u64> {
        let attachment = self
            .attachment
            .as_ref()
            .context("KV state agent is not attached")?
            .clone();
        if attachment.generation != generation || attachment.ready {
            anyhow::bail!("attachment must be unready before detach cleanup");
        }
        self.ingress_generation = None;

        // Worker mutations queued for this attachment may be discarded because the
        // ordered Worker reset supersedes them. CacheOwner mutations are post-commit
        // residency truth and must be drained or fail CacheOwner closed.
        self.drain_detaching_cache_owner(generation).await;

        let cursor = match self
            .apply_and_publish_reset(attachment.worker, ResidencyDomain::Worker)
            .await
        {
            Ok(cursor) => cursor,
            Err(error) => {
                self.worker_domain_failed = true;
                self.publish_status();
                return Err(error).context("failed to publish final Worker detach barrier");
            }
        };
        self.attachment = None;
        self.source_cursor = SourceCursorState::default();
        self.publish_status();
        Ok(cursor)
    }

    fn quarantine(&mut self, generation: u64, reason: &'static str) {
        let Some(attachment) = self.attachment.as_mut() else {
            return;
        };
        if attachment.generation != generation {
            return;
        }
        attachment.ready = false;
        tracing::warn!(generation, reason, "Quarantining vLLM KV ingress");
        self.publish_status();
    }

    async fn handle_ingress(&mut self, batch: IngressBatch) -> Result<()> {
        let valid_generation = self.ingress_generation == Some(batch.generation)
            && !self.worker_domain_failed
            && !self.source_failed;
        if !valid_generation {
            anyhow::bail!(
                "ingress is closed for attachment generation {}",
                batch.generation
            );
        }

        if batch.events.len() > MAX_INGRESS_EVENTS
            || event_block_count(&batch.events) > MAX_INGRESS_BLOCKS
        {
            self.fail_source(batch.generation, "oversized ingress envelope");
            anyhow::bail!("oversized ingress envelope");
        }

        if !self.source_cursor.observe(&batch) {
            self.fail_source(batch.generation, "raw source cursor gap or regression");
            anyhow::bail!("raw source cursor gap or regression");
        }

        if let Some(reason) = batch.cache_owner_fault {
            self.fail_cache_owner(batch.generation, reason);
        }

        let events = batch
            .events
            .into_iter()
            .filter(|event| {
                event.placement.residency_domain != ResidencyDomain::CacheOwner
                    || !self.cache_owner_domain_failed
            })
            .collect();
        if let Err(error) = self.apply_and_publish_chunk(events).await {
            tracing::error!(%error, "KV state-agent transaction failed");
            let reservation = self.reserve_locally_applied_event_ids();
            self.source_failed = true;
            self.worker_domain_failed = true;
            self.cache_owner_domain_failed = true;
            if let Some(attachment) = self.attachment.as_mut()
                && attachment.generation == batch.generation
            {
                attachment.ready = false;
            }
            self.publish_status();
            reservation.context("failed to reserve locally applied state-agent event IDs")?;
            return Err(error);
        }
        Ok(())
    }

    async fn drain_detaching_cache_owner(&mut self, generation: u64) {
        while let Ok(batch) = self.ingress_rx.try_recv() {
            if batch.generation != generation {
                tracing::warn!(
                    queued_generation = batch.generation,
                    generation,
                    "Discarding ingress from a non-current attachment generation"
                );
                continue;
            }

            if !self.source_cursor.observe(&batch) {
                self.fail_cache_owner(generation, "raw source cursor gap during detach");
                continue;
            }
            if let Some(reason) = batch.cache_owner_fault {
                self.fail_cache_owner(generation, reason);
            }

            let cache_owner_events = batch
                .events
                .into_iter()
                .filter(|event| {
                    event.placement.residency_domain == ResidencyDomain::CacheOwner
                        && !self.cache_owner_domain_failed
                })
                .collect();
            let result = self.apply_and_publish_chunk(cache_owner_events).await;
            if let Err(error) = &result {
                tracing::error!(%error, "Failed to drain CacheOwner residency during detach");
                self.cache_owner_domain_failed = true;
                // Successful local applications are already present in the recovery
                // buffer. Reserve their IDs before the authoritative Worker barrier so
                // recovery can close any publication gap without an ID collision.
                if let Err(error) = self.reserve_locally_applied_event_ids() {
                    tracing::error!(%error, "Failed to reserve drained state-agent event IDs");
                    self.source_failed = true;
                    self.worker_domain_failed = true;
                    self.publish_status();
                    continue;
                }
                self.publish_status();
            }
        }
        if self.source_cursor.active.is_some() {
            self.fail_cache_owner(generation, "incomplete raw source batch during detach");
        }
    }

    fn fail_source(&mut self, generation: u64, reason: &'static str) {
        self.source_failed = true;
        self.worker_domain_failed = true;
        self.cache_owner_domain_failed = true;
        self.quarantine(generation, reason);
    }

    fn fail_cache_owner(&mut self, generation: u64, reason: &'static str) {
        self.cache_owner_domain_failed = true;
        tracing::error!(generation, reason, "Quarantining CacheOwner residency");
        self.publish_status();
    }

    fn reserve_locally_applied_event_ids(&mut self) -> Result<()> {
        let next_local_id = self
            .local_indexer
            .current_event_id()
            .checked_add(1)
            .context("state-agent outbound cursor exhausted")?;
        self.next_outbound_id = next_local_id.max(self.next_outbound_id);
        Ok(())
    }

    async fn apply_and_publish_chunk(
        &mut self,
        placement_events: Vec<PlacementEvent>,
    ) -> Result<()> {
        let attachment_worker = self
            .attachment
            .as_ref()
            .context("KV state agent has no framework attachment")?
            .worker;
        let event_worker_id = attachment_worker.worker_id;
        let mut next_outbound_id = self.next_outbound_id;
        let mut output = Vec::with_capacity(placement_events.len());

        for placement_event in placement_events {
            let domain = placement_event.placement.residency_domain;
            let mut event = placement_event.event;
            if event.dp_rank != self.slot_dp_rank {
                anyhow::bail!(
                    "ingress rank {} does not match stable slot rank {}",
                    event.dp_rank,
                    self.slot_dp_rank
                );
            }
            let tier = placement_event.placement.tier;
            event.data = match event.data {
                KvCacheEventData::Removed(data) => {
                    let Some(filtered) =
                        self.dedup
                            .filter_remove_in_domain(event.dp_rank, tier, domain, data)
                    else {
                        continue;
                    };
                    KvCacheEventData::Removed(filtered)
                }
                KvCacheEventData::Stored(data) => {
                    self.dedup
                        .track_store_in_domain(event.dp_rank, tier, domain, &data);
                    KvCacheEventData::Stored(data)
                }
                KvCacheEventData::Cleared => {
                    self.dedup.clear_rank_domain(event.dp_rank, domain);
                    KvCacheEventData::Cleared
                }
            };
            event.event_id = next_outbound_id;
            next_outbound_id = next_outbound_id
                .checked_add(1)
                .context("state-agent outbound cursor exhausted")?;

            let router_event = match domain {
                ResidencyDomain::Worker => RouterEvent::with_residency_domain(
                    event_worker_id,
                    event,
                    tier,
                    ResidencyDomain::Worker,
                )
                .with_state_source(self.identity.cache_owner_id),
                ResidencyDomain::CacheOwner => RouterEvent::with_cache_owner(
                    event_worker_id,
                    event,
                    tier,
                    self.identity.cache_owner_id,
                ),
            };
            self.local_indexer
                .apply_event_with_buffer(router_event.clone())
                .await
                .context("local residency application failed")?;
            output.push(router_event);
        }

        self.publish_exact(&output).await?;
        self.next_outbound_id = next_outbound_id;
        self.publish_status();
        Ok(())
    }

    async fn apply_and_publish_reset(
        &mut self,
        worker: WorkerWithDpRank,
        domain: ResidencyDomain,
    ) -> Result<u64> {
        let placement = dynamo_kv_router::protocols::Placement {
            owner: dynamo_kv_router::protocols::PlacementOwner::LocalWorker(worker),
            tier: StorageTier::Device,
            residency_domain: domain,
        };
        self.apply_and_publish_chunk(vec![PlacementEvent::new(
            placement,
            KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Cleared,
                dp_rank: worker.dp_rank,
            },
        )])
        .await?;
        Ok(self
            .next_outbound_id
            .checked_sub(1)
            .expect("outbound IDs start at one"))
    }

    async fn publish_exact(&self, events: &[RouterEvent]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        let mut last_error = None;
        for attempt in 0..PUBLISH_ATTEMPTS {
            match self.publisher.publish_events(events).await {
                Ok(()) => return Ok(()),
                Err(error) => last_error = Some(error),
            }
            if attempt + 1 < PUBLISH_ATTEMPTS {
                tokio::time::sleep(Duration::from_millis(10 * (attempt as u64 + 1))).await;
            }
        }
        Err(last_error.expect("at least one publish attempt"))
            .context("event publication remained uncertain after bounded retries")
    }

    fn publish_status(&self) {
        self.status.store(Arc::new(KvStateAgentStatus {
            identity: self.identity.clone(),
            attachment: self.attachment.as_ref().map(Attachment::status),
            cache_owner_ready: !self.cache_owner_domain_failed,
            outbound_cursor: self
                .next_outbound_id
                .checked_sub(1)
                .expect("next outbound ID is initialized to one"),
        }));
    }
}

/// Handle for one per-slot state agent.
pub struct KvStateAgent {
    component: Component,
    kv_state_endpoint: EndpointId,
    kv_block_size: u32,
    slot_dp_rank: u32,
    raw_mode: KvStateAgentRawMode,
    event_topic: &'static str,
    recovery_target: dynamo_runtime::component::Instance,
    identity: KvStateAgentIdentity,
    control_tx: mpsc::Sender<ControlCommand>,
    ingress_tx: mpsc::Sender<IngressBatch>,
    status: Arc<ArcSwap<KvStateAgentStatus>>,
    cancel: CancellationToken,
    coordinator: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    vllm_listener: Mutex<Option<VllmListener>>,
    recovery_endpoint: Mutex<Option<StartedEndpoint>>,
    lifecycle: Mutex<()>,
    event_source: Arc<Mutex<Option<DiscoveryInstance>>>,
    persistent_source: Arc<Mutex<Option<DiscoveryInstance>>>,
    attachment_source: Arc<Mutex<Option<DiscoveryInstance>>>,
    attachment: Mutex<Option<Attachment>>,
}

impl KvStateAgent {
    pub async fn start(config: KvStateAgentConfig) -> Result<Self> {
        if config.kv_block_size == 0 {
            anyhow::bail!("kv_block_size cannot be zero");
        }
        let component = config.endpoint.component().clone();
        let cancel = CancellationToken::new();
        let local_indexer = Arc::new(LocalKvIndexer::new(
            cancel.child_token(),
            config.kv_block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            WORKER_KV_INDEXER_BUFFER_SIZE,
        ));

        let raw_mode = config
            .vllm_source
            .as_ref()
            .map_or(KvStateAgentRawMode::FrameworkV1, |source| source.raw_mode);
        let event_topic = match raw_mode {
            KvStateAgentRawMode::FrameworkV1 => KV_EVENT_SUBJECT,
            // NOTE: ResidencyV2 is intentionally producer/discovery infrastructure only
            // in this foundation. Current routers still consume the functional
            // FrameworkV1 `kv-events` path; no router subscribes to this v2 topic yet.
            // Enabling ResidencyV2 must not be interpreted as end-to-end CacheOwner
            // routing support.
            //
            // TODO(#13044): activate this topic only with the version-aware source
            // watch, fenced recovery controller, and aggregate owner projection.
            KvStateAgentRawMode::ResidencyV2 => KV_STATE_EVENT_TOPIC_V2,
        };
        let event_publisher = EventPublisher::for_endpoint_id(
            config.endpoint.drt(),
            &config.kv_state_endpoint,
            event_topic,
        )
        .await
        .context("failed to create state-agent event publisher")?;
        let publisher_id = event_publisher.publisher_id();
        let identity = KvStateAgentIdentity {
            cache_owner_id: config.slot.cache_owner_id,
            publisher_id,
            protocol_version: KvStateProtocolVersion::V2,
        };
        let status = Arc::new(ArcSwap::from_pointee(KvStateAgentStatus {
            identity: identity.clone(),
            attachment: None,
            cache_owner_ready: raw_mode == KvStateAgentRawMode::ResidencyV2,
            outbound_cursor: 0,
        }));

        let recovery_endpoint = start_worker_kv_query_endpoint_with_status(
            component.clone(),
            publisher_id,
            config.slot.worker.worker_id,
            config.slot.worker.dp_rank,
            local_indexer.clone(),
            Some(status.clone()),
        )
        .await
        .context("failed to start state-agent recovery/control endpoint")?;

        let recovery_target = recovery_endpoint.instance().clone();
        let source_ad = DiscoveredKvEventSource {
            kv_state_endpoint: config.kv_state_endpoint.clone(),
            worker: config.slot.worker,
            publisher_id,
            recovery_target: Some(recovery_target.clone()),
        };
        let event_source = match register_advertisement(
            &component,
            EventScope::Endpoint {
                endpoint: config.kv_state_endpoint.clone(),
            },
            event_topic,
            publisher_id,
            &source_ad,
        )
        .await
        {
            Ok(instance) => instance,
            Err(error) => {
                let _ = recovery_endpoint.shutdown().await;
                return Err(error).context("failed to advertise state-agent event source");
            }
        };

        let persistent_ad = KvStateSourceAdvertisement {
            cache_owner_id: identity.cache_owner_id,
            publisher_id,
            protocol_version: identity.protocol_version,
            event_topic: event_topic.to_string(),
            recovery_control_target: recovery_target.clone(),
        };
        let persistent_source = match register_advertisement(
            &component,
            component_scope(&component),
            KV_STATE_SOURCE_TOPIC_V2,
            publisher_id,
            &persistent_ad,
        )
        .await
        {
            Ok(instance) => instance,
            Err(error) => {
                let _ = component.drt().discovery().unregister(event_source).await;
                let _ = recovery_endpoint.shutdown().await;
                return Err(error).context("failed to advertise persistent state source");
            }
        };

        let (control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
        let (ingress_tx, ingress_rx) = mpsc::channel(INGRESS_QUEUE_CAPACITY);
        let coordinator_cancel = cancel.child_token();
        let coordinator = component.drt().runtime().secondary().spawn(
            Coordinator {
                publisher: EventPlanePublisher(event_publisher),
                local_indexer,
                identity: identity.clone(),
                slot_dp_rank: config.slot.worker.dp_rank,
                status: status.clone(),
                control_rx,
                ingress_rx,
                attachment: None,
                ingress_generation: None,
                next_outbound_id: 1,
                next_attachment_generation: 1,
                source_cursor: SourceCursorState::default(),
                dedup: EventDedupFilter::new(),
                worker_domain_failed: false,
                cache_owner_domain_failed: raw_mode != KvStateAgentRawMode::ResidencyV2,
                source_failed: false,
            }
            .run(coordinator_cancel),
        );

        let framework_endpoint = config
            .vllm_source
            .as_ref()
            .map(|source| source.endpoint.clone())
            .unwrap_or_else(|| "internal://vllm-ingress".to_string());
        let attachment = Attachment {
            generation: 0,
            worker: config.slot.worker,
            vllm_zmq_endpoint: framework_endpoint,
            ready: false,
            cache_readable: false,
            ready_at_outbound_cursor: 0,
        };
        let attachment = match send_attach(&control_tx, attachment).await {
            Ok(attachment) => attachment,
            Err(error) => {
                cancel.cancel();
                let _ = coordinator.await;
                let _ = component
                    .drt()
                    .discovery()
                    .unregister(persistent_source)
                    .await;
                let _ = component.drt().discovery().unregister(event_source).await;
                let _ = recovery_endpoint.shutdown().await;
                return Err(error).context("failed to establish initial engine attachment");
            }
        };

        let attachment_ad = attachment_advertisement(&identity, &recovery_target, &attachment);
        let attachment_source = match register_advertisement(
            &component,
            component_scope(&component),
            KV_STATE_ATTACHMENT_TOPIC_V2,
            attachment_discovery_id(publisher_id, attachment.generation),
            &attachment_ad,
        )
        .await
        {
            Ok(instance) => instance,
            Err(error) => {
                cancel.cancel();
                let _ = coordinator.await;
                let _ = component
                    .drt()
                    .discovery()
                    .unregister(persistent_source)
                    .await;
                let _ = component.drt().discovery().unregister(event_source).await;
                let _ = recovery_endpoint.shutdown().await;
                return Err(error).context("failed to advertise ready engine attachment");
            }
        };

        let event_source = Arc::new(Mutex::new(Some(event_source)));
        let persistent_source = Arc::new(Mutex::new(Some(persistent_source)));
        let attachment_source = Arc::new(Mutex::new(Some(attachment_source)));
        let vllm_listener = config.vllm_source.map(|source| {
            start_vllm_listener(
                component.clone(),
                source,
                config.slot.worker,
                attachment.generation,
                config.kv_block_size,
                ingress_tx.clone(),
                control_tx.clone(),
                event_source.clone(),
                attachment_source.clone(),
                cancel.child_token(),
            )
        });

        let agent = Self {
            component,
            kv_state_endpoint: config.kv_state_endpoint,
            kv_block_size: config.kv_block_size,
            slot_dp_rank: config.slot.worker.dp_rank,
            raw_mode,
            event_topic,
            recovery_target,
            identity,
            control_tx,
            ingress_tx,
            status,
            cancel,
            coordinator: StdMutex::new(Some(coordinator)),
            vllm_listener: Mutex::new(vllm_listener),
            recovery_endpoint: Mutex::new(Some(recovery_endpoint)),
            lifecycle: Mutex::new(()),
            event_source,
            persistent_source,
            attachment_source,
            attachment: Mutex::new(Some(attachment)),
        };

        Ok(agent)
    }

    pub fn identity(&self) -> &KvStateAgentIdentity {
        &self.identity
    }

    pub fn status(&self) -> Arc<KvStateAgentStatus> {
        self.status.load_full()
    }

    pub async fn quarantine_vllm(&self, generation: u64, reason: &'static str) -> Result<()> {
        let (response, received) = oneshot::channel();
        self.control_tx
            .send(ControlCommand::Quarantine {
                generation,
                reason,
                response: Some(response),
            })
            .await
            .context("state-agent coordinator is closed")?;
        received
            .await
            .context("state-agent quarantine acknowledgement was dropped")
    }

    /// Attach a new engine incarnation after the prior attachment has fully detached.
    ///
    /// The API accepts a different WorkerId even though the initial in-process
    /// vLLM host does not yet survive its handler lifecycle.
    // TODO(#13044): make attach construction cancellation-safe when a host exists
    // that can survive engine replacement. The current Python host creates one
    // attachment and terminates with the vLLM handler.
    pub async fn attach(
        &self,
        worker: WorkerWithDpRank,
        vllm_source: KvStateAgentVllmSource,
        cache_readable: bool,
    ) -> Result<KvStateAttachmentStatus> {
        let _lifecycle = self.lifecycle.lock().await;
        if worker.dp_rank != self.slot_dp_rank {
            anyhow::bail!("attachment DP rank does not match the stable state-agent slot");
        }
        if vllm_source.raw_mode != self.raw_mode {
            anyhow::bail!("raw KV protocol mode is immutable for one state-agent publisher");
        }
        if self.attachment.lock().await.is_some()
            || self.attachment_source.lock().await.is_some()
            || self.event_source.lock().await.is_some()
            || self.vllm_listener.lock().await.is_some()
        {
            anyhow::bail!("previous engine attachment has not fully detached");
        }

        let attachment = send_attach(
            &self.control_tx,
            Attachment {
                generation: 0,
                worker,
                vllm_zmq_endpoint: vllm_source.endpoint.clone(),
                ready: false,
                cache_readable,
                ready_at_outbound_cursor: 0,
            },
        )
        .await?;

        let event_ad = DiscoveredKvEventSource {
            kv_state_endpoint: self.kv_state_endpoint.clone(),
            worker,
            publisher_id: self.identity.publisher_id,
            recovery_target: Some(self.recovery_target.clone()),
        };
        let event_source = match register_advertisement(
            &self.component,
            EventScope::Endpoint {
                endpoint: self.kv_state_endpoint.clone(),
            },
            self.event_topic,
            self.identity.publisher_id,
            &event_ad,
        )
        .await
        {
            Ok(source) => source,
            Err(error) => {
                abort_attach(&self.control_tx, attachment.generation).await;
                return Err(error).context("failed to advertise reattached state-agent source");
            }
        };

        let attachment_ad =
            attachment_advertisement(&self.identity, &self.recovery_target, &attachment);
        let attachment_source = match register_advertisement(
            &self.component,
            component_scope(&self.component),
            KV_STATE_ATTACHMENT_TOPIC_V2,
            attachment_discovery_id(self.identity.publisher_id, attachment.generation),
            &attachment_ad,
        )
        .await
        {
            Ok(source) => source,
            Err(error) => {
                let _ = self
                    .component
                    .drt()
                    .discovery()
                    .unregister(event_source)
                    .await;
                abort_attach(&self.control_tx, attachment.generation).await;
                return Err(error).context("failed to advertise reattached engine attachment");
            }
        };

        *self.event_source.lock().await = Some(event_source);
        *self.attachment_source.lock().await = Some(attachment_source);
        *self.attachment.lock().await = Some(attachment.clone());
        *self.vllm_listener.lock().await = Some(start_vllm_listener(
            self.component.clone(),
            vllm_source,
            worker,
            attachment.generation,
            self.kv_block_size,
            self.ingress_tx.clone(),
            self.control_tx.clone(),
            self.event_source.clone(),
            self.attachment_source.clone(),
            self.cancel.child_token(),
        ));
        Ok(attachment.status())
    }

    pub async fn detach(&self, generation: u64) -> Result<u64> {
        let _lifecycle = self.lifecycle.lock().await;
        let (response, received) = oneshot::channel();
        self.control_tx
            .send(ControlCommand::BeginDetach {
                generation,
                response,
            })
            .await
            .context("state-agent coordinator is closed")?;
        received
            .await
            .context("state-agent detach begin was dropped")??;

        unregister_required(
            &self.component,
            &self.event_source,
            "state-agent event source",
        )
        .await?;
        unregister_required(
            &self.component,
            &self.attachment_source,
            "attachment advertisement",
        )
        .await?;

        if let Some(listener) = self.vllm_listener.lock().await.take() {
            listener.stop().await;
        }
        let (response, received) = oneshot::channel();
        self.control_tx
            .send(ControlCommand::CompleteDetach {
                generation,
                response,
            })
            .await
            .context("state-agent coordinator is closed")?;
        let cursor = received
            .await
            .context("state-agent detach completion was dropped")??;

        *self.attachment.lock().await = None;
        Ok(cursor)
    }

    pub async fn shutdown(&self) -> Result<()> {
        let _lifecycle = self.lifecycle.lock().await;
        let _ = self.control_tx.send(ControlCommand::Shutdown).await;
        self.cancel.cancel();
        if let Some(listener) = self.vllm_listener.lock().await.take() {
            listener.stop().await;
        }
        let coordinator = self.coordinator.lock().unwrap().take();
        if let Some(coordinator) = coordinator {
            let _ = coordinator.await;
        }

        unregister_if_present(&self.component, &self.attachment_source).await;
        unregister_if_present(&self.component, &self.persistent_source).await;
        unregister_if_present(&self.component, &self.event_source).await;
        if let Some(endpoint) = self.recovery_endpoint.lock().await.take() {
            endpoint.shutdown().await?;
        }
        Ok(())
    }
}

impl Drop for KvStateAgent {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Ok(mut coordinator) = self.coordinator.lock()
            && let Some(coordinator) = coordinator.take()
        {
            coordinator.abort();
        }
    }
}

async fn send_attach(
    control_tx: &mpsc::Sender<ControlCommand>,
    attachment: Attachment,
) -> Result<Attachment> {
    let (response, received) = oneshot::channel();
    control_tx
        .send(ControlCommand::Attach {
            attachment,
            response,
        })
        .await
        .context("state-agent coordinator is closed")?;
    received.await.context("state-agent attach was dropped")?
}

async fn abort_attach(control_tx: &mpsc::Sender<ControlCommand>, generation: u64) {
    let _ = control_tx
        .send(ControlCommand::AbortAttach { generation })
        .await;
}

fn attachment_advertisement(
    identity: &KvStateAgentIdentity,
    recovery_target: &dynamo_runtime::component::Instance,
    attachment: &Attachment,
) -> KvStateAttachmentAdvertisement {
    KvStateAttachmentAdvertisement {
        cache_owner_id: identity.cache_owner_id,
        publisher_id: identity.publisher_id,
        protocol_version: identity.protocol_version,
        recovery_control_target: recovery_target.clone(),
        attachment_generation: attachment.generation,
        worker: attachment.worker,
        vllm_zmq_endpoint: attachment.vllm_zmq_endpoint.clone(),
        cache_readable: attachment.cache_readable,
        ready_at_outbound_cursor: attachment.ready_at_outbound_cursor,
    }
}

fn component_scope(component: &Component) -> EventScope {
    EventScope::Component {
        namespace: component.namespace().name(),
        component: component.name().to_string(),
    }
}

async fn register_advertisement<T: serde::Serialize>(
    component: &Component,
    scope: EventScope,
    topic: &str,
    publisher_id: u64,
    advertisement: &T,
) -> Result<DiscoveryInstance> {
    let metadata = serde_json::to_value(advertisement)?;
    component
        .drt()
        .discovery()
        .register(DiscoverySpec::EventSource {
            scope,
            topic: topic.to_string(),
            publisher_id,
            metadata,
        })
        .await
}

async fn unregister_if_present(
    component: &Component,
    instance: &Arc<Mutex<Option<DiscoveryInstance>>>,
) {
    let current = instance.lock().await.clone();
    let Some(current) = current else {
        return;
    };
    if let Err(error) = component
        .drt()
        .discovery()
        .unregister(current.clone())
        .await
    {
        tracing::warn!(%error, "Failed to unregister KV state-agent advertisement");
        return;
    }
    let mut instance = instance.lock().await;
    if instance.as_ref() == Some(&current) {
        *instance = None;
    }
}

async fn unregister_required(
    component: &Component,
    instance: &Arc<Mutex<Option<DiscoveryInstance>>>,
    description: &str,
) -> Result<()> {
    let current = instance.lock().await.clone();
    let Some(current) = current else {
        return Ok(());
    };
    component
        .drt()
        .discovery()
        .unregister(current.clone())
        .await
        .with_context(|| format!("failed to remove exact {description}"))?;
    let mut instance = instance.lock().await;
    if instance.as_ref() == Some(&current) {
        *instance = None;
    }
    Ok(())
}

fn event_block_count(events: &[PlacementEvent]) -> usize {
    events
        .iter()
        .map(|event| match &event.event.data {
            KvCacheEventData::Stored(data) => data.blocks.len(),
            KvCacheEventData::Removed(data) => data.block_hashes.len(),
            KvCacheEventData::Cleared => 0,
        })
        .sum()
}

fn attachment_discovery_id(publisher_id: u64, generation: u64) -> u64 {
    const JSON_SAFE_MASK: u64 = (1u64 << 53) - 1;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"dynamo/kv-state-attachment-record/v1");
    hasher.update(&publisher_id.to_be_bytes());
    hasher.update(&generation.to_be_bytes());
    let digest = hasher.finalize();
    let value = u64::from_be_bytes(
        digest.as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 digest has eight prefix bytes"),
    );
    (value & JSON_SAFE_MASK).max(1)
}

struct VllmListener {
    cancel: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

impl VllmListener {
    async fn stop(self) {
        self.cancel.cancel();
        let mut handle = self.handle;
        if tokio::time::timeout(Duration::from_secs(2), &mut handle)
            .await
            .is_err()
        {
            tracing::warn!("Timed out joining framework KV listener");
            handle.abort();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn start_vllm_listener(
    component: Component,
    source: KvStateAgentVllmSource,
    worker: WorkerWithDpRank,
    generation: u64,
    kv_block_size: u32,
    ingress_tx: mpsc::Sender<IngressBatch>,
    control_tx: mpsc::Sender<ControlCommand>,
    event_source: Arc<Mutex<Option<DiscoveryInstance>>>,
    attachment_source: Arc<Mutex<Option<DiscoveryInstance>>>,
    cancel: CancellationToken,
) -> VllmListener {
    let listener_cancel = cancel.child_token();
    let task_cancel = listener_cancel.clone();
    let handle = component.drt().runtime().secondary().spawn(async move {
        let result = run_vllm_listener(
            &source,
            worker,
            generation,
            kv_block_size,
            ingress_tx,
            task_cancel.clone(),
        )
        .await;
        if !task_cancel.is_cancelled() {
            if let Err(error) = result {
                tracing::error!(
                    endpoint = %source.endpoint,
                    generation,
                    %error,
                    "vLLM KV listener failed"
                );
            }
            let _ = control_tx
                .send(ControlCommand::Quarantine {
                    generation,
                    reason: "framework listener terminated",
                    response: None,
                })
                .await;
            // New consumers also validate the endpoint status, but exact
            // withdrawal lets legacy Worker-only routers fail closed without
            // conflating transport interruption with a CacheOwner clear.
            unregister_if_present(&component, &attachment_source).await;
            unregister_if_present(&component, &event_source).await;
        }
    });
    VllmListener {
        cancel: listener_cancel,
        handle,
    }
}

async fn run_vllm_listener(
    source: &KvStateAgentVllmSource,
    worker: WorkerWithDpRank,
    generation: u64,
    kv_block_size: u32,
    ingress_tx: mpsc::Sender<IngressBatch>,
    cancel: CancellationToken,
) -> Result<()> {
    let mut framework_normalizer =
        ZmqEventNormalizer::new(kv_block_size).with_image_token_id(source.image_token_id);
    let mut cache_owner_normalizer =
        ZmqEventNormalizer::new(kv_block_size).with_image_token_id(source.image_token_id);
    let mut socket = connect_sub_socket(&source.endpoint, Some(&source.topic)).await?;

    loop {
        let frames = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            frames = socket.next() => frames,
        };
        let frames = match frames {
            Some(Ok(frames)) => multipart_message(frames),
            Some(Err(error)) => return Err(error).context("ZMQ receive failed"),
            None => anyhow::bail!("ZMQ stream ended"),
        };
        if frames.len() != 3 {
            anyhow::bail!("expected three ZMQ frames, received {}", frames.len());
        }
        let payload = &frames[2];
        if payload.len() > MAX_ZMQ_PAYLOAD_BYTES {
            anyhow::bail!("ZMQ payload exceeds {MAX_ZMQ_PAYLOAD_BYTES} bytes");
        }
        let sequence: [u8; 8] = frames[1]
            .as_slice()
            .try_into()
            .context("ZMQ sequence must contain eight bytes")?;
        let source_cursor = u64::from_be_bytes(sequence);
        let batch = dynamo_kv_router::zmq_wire::decode_event_batch(payload)
            .context("failed to decode framework KV event batch")?;
        let dp_rank = batch.data_parallel_rank.unwrap_or(0).cast_unsigned();
        if dp_rank != worker.dp_rank {
            anyhow::bail!(
                "vLLM batch rank {dp_rank} does not match configured rank {}",
                worker.dp_rank
            );
        }

        let NormalizedRawBatch {
            events,
            cache_owner_fault,
        } = normalize_raw_batch(
            batch.events,
            worker,
            kv_block_size,
            source.raw_mode,
            &mut framework_normalizer,
            &mut cache_owner_normalizer,
        );

        let chunks = bounded_ingress_chunks(events)?;
        let final_chunk_index = chunks.len().saturating_sub(1);
        for (chunk_index, events) in chunks.into_iter().enumerate() {
            let envelope = IngressBatch {
                generation,
                source_cursor,
                chunk_index: u32::try_from(chunk_index)
                    .context("vLLM batch has too many chunks")?,
                final_chunk: chunk_index == final_chunk_index,
                events,
                cache_owner_fault: (chunk_index == 0).then_some(cache_owner_fault).flatten(),
            };
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return Ok(()),
                result = ingress_tx.send(envelope) => {
                    result.context("state-agent vLLM queue closed")?;
                }
            }
        }
    }
}

struct NormalizedRawBatch {
    events: Vec<PlacementEvent>,
    cache_owner_fault: Option<&'static str>,
}

fn normalize_raw_batch(
    raw_events: Vec<dynamo_kv_router::zmq_wire::RawKvEvent>,
    worker: WorkerWithDpRank,
    kv_block_size: u32,
    raw_mode: KvStateAgentRawMode,
    framework_normalizer: &mut ZmqEventNormalizer,
    cache_owner_normalizer: &mut ZmqEventNormalizer,
) -> NormalizedRawBatch {
    // KVCC is the placement authority. vLLM only enriches committed KVCC
    // transitions with canonical routing metadata and serializes them on this
    // versioned stream. Any failure on that path quarantines CacheOwner below;
    // silently omitting a KVCC transition would make retained ownership lie.
    let mut events = Vec::with_capacity(raw_events.len().min(MAX_INGRESS_EVENTS));
    let mut cache_owner_fault = None;
    for raw_event in raw_events {
        let source_kind = match raw_event.source_kind() {
            Ok(source_kind) => source_kind,
            Err(_) => {
                cache_owner_fault.get_or_insert("unknown raw source kind");
                continue;
            }
        };
        if source_kind == KvEventSourceKind::Kvcc && raw_mode != KvStateAgentRawMode::ResidencyV2 {
            cache_owner_fault.get_or_insert("KVCC event on framework-only raw protocol");
            continue;
        }
        if source_kind == KvEventSourceKind::Kvcc
            && raw_event
                .block_size()
                .is_some_and(|size| size != kv_block_size as usize)
        {
            cache_owner_fault.get_or_insert("KVCC block size is incompatible");
            continue;
        }

        let normalizer = match source_kind {
            KvEventSourceKind::Framework => &mut *framework_normalizer,
            KvEventSourceKind::Kvcc => &mut *cache_owner_normalizer,
        };
        let raw_event = match normalizer.preprocess_residency_with_reason(raw_event, worker) {
            Ok(raw_event) => raw_event,
            Err(reason) if source_kind == KvEventSourceKind::Kvcc => {
                tracing::warn!(?reason, "KVCC event enrichment failed");
                cache_owner_fault.get_or_insert("KVCC event enrichment failed");
                continue;
            }
            Err(_) => continue,
        };
        let Some(mut event) = normalizer.normalize_preprocessed(raw_event, 0, worker) else {
            if source_kind == KvEventSourceKind::Kvcc {
                cache_owner_fault.get_or_insert("KVCC event canonicalization failed");
            }
            continue;
        };
        match source_kind {
            KvEventSourceKind::Framework => {
                event.placement.residency_domain = ResidencyDomain::Worker;
            }
            KvEventSourceKind::Kvcc => {
                if !matches!(
                    (&event.event.data, event.placement.tier),
                    (KvCacheEventData::Cleared, _)
                        | (_, StorageTier::HostPinned | StorageTier::Disk)
                ) {
                    cache_owner_fault.get_or_insert("unsupported KVCC storage medium");
                    continue;
                }
                event.placement.residency_domain = ResidencyDomain::CacheOwner;
            }
        }
        events.push(event);
    }
    NormalizedRawBatch {
        events,
        cache_owner_fault,
    }
}

fn bounded_ingress_chunks(events: Vec<PlacementEvent>) -> Result<Vec<Vec<PlacementEvent>>> {
    if events.is_empty() {
        return Ok(vec![Vec::new()]);
    }
    let mut chunks = Vec::new();
    let mut current = Vec::new();
    let mut current_blocks = 0usize;
    for event in events {
        let blocks = event_block_count(std::slice::from_ref(&event));
        if blocks > MAX_INGRESS_BLOCKS {
            anyhow::bail!("one ingress event exceeds the block bound");
        }
        if !current.is_empty()
            && (current.len() == MAX_INGRESS_EVENTS
                || current_blocks.saturating_add(blocks) > MAX_INGRESS_BLOCKS)
        {
            chunks.push(std::mem::take(&mut current));
            current_blocks = 0;
        }
        current_blocks = current_blocks.saturating_add(blocks);
        current.push(event);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Mutex as StdMutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    use dynamo_kv_router::{
        identity::{
            CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId, RoutingScopeId,
            StableDpSlotId,
        },
        indexer::WorkerKvQueryResponse,
        protocols::{
            ExternalSequenceBlockHash, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
            Placement, PlacementOwner,
        },
        zmq_wire::{BlockHashValue, RawKvEvent},
    };

    use super::*;

    #[derive(Clone, Default)]
    struct RecordingSink {
        events: Arc<StdMutex<Vec<RouterEvent>>>,
        publications: Arc<AtomicUsize>,
        fail: Arc<AtomicBool>,
    }

    impl RouterEventBatchSink for RecordingSink {
        async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
            self.publications.fetch_add(1, Ordering::Relaxed);
            if self.fail.load(Ordering::Relaxed) {
                anyhow::bail!("injected state-agent publication failure");
            }
            self.events.lock().unwrap().extend_from_slice(events);
            Ok(())
        }
    }

    fn cache_owner_id() -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        )
    }

    fn store(
        worker: WorkerWithDpRank,
        domain: ResidencyDomain,
        parent: Option<u64>,
        block: u64,
        local: u64,
    ) -> PlacementEvent {
        PlacementEvent::new(
            Placement {
                owner: PlacementOwner::LocalWorker(worker),
                tier: StorageTier::HostPinned,
                residency_domain: domain,
            },
            KvCacheEvent {
                event_id: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent.map(ExternalSequenceBlockHash),
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(block),
                        tokens_hash: LocalBlockHash(local),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: worker.dp_rank,
            },
        )
    }

    fn raw_store(medium: Option<&str>, source_kind: Option<&str>, block: u64) -> RawKvEvent {
        RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(block)],
            parent_block_hash: None,
            token_ids: vec![10, 11, 12, 13],
            block_size: 4,
            medium: medium.map(str::to_owned),
            lora_name: None,
            cache_namespace: None,
            block_mm_infos: None,
            is_eagle: Some(false),
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
            locality: None,
            source_kind: source_kind.map(str::to_owned),
        }
    }

    #[test]
    fn raw_source_kind_maps_owners_and_quarantines_unsupported_kvcc() {
        let worker = WorkerWithDpRank::new(17, 3);
        let mut framework = ZmqEventNormalizer::new(4);
        let mut cache_owner = ZmqEventNormalizer::new(4);
        let normalized = normalize_raw_batch(
            vec![
                raw_store(None, None, 101),
                raw_store(Some("CPU"), Some("framework"), 102),
                raw_store(Some("CPU_PINNED"), Some("kvcc"), 103),
                raw_store(Some("STORAGE"), Some("kvcc"), 104),
                RawKvEvent::AllBlocksCleared {
                    source_kind: Some("kvcc".to_string()),
                },
            ],
            worker,
            4,
            KvStateAgentRawMode::ResidencyV2,
            &mut framework,
            &mut cache_owner,
        );
        assert_eq!(normalized.cache_owner_fault, None);
        assert_eq!(
            normalized
                .events
                .iter()
                .map(|event| (event.placement.residency_domain, event.placement.tier))
                .collect::<Vec<_>>(),
            vec![
                (ResidencyDomain::Worker, StorageTier::Device),
                (ResidencyDomain::Worker, StorageTier::HostPinned),
                (ResidencyDomain::CacheOwner, StorageTier::HostPinned),
                (ResidencyDomain::CacheOwner, StorageTier::Disk),
                (ResidencyDomain::CacheOwner, StorageTier::Device),
            ]
        );

        let normalized = normalize_raw_batch(
            vec![
                raw_store(Some("GPU"), Some("kvcc"), 105),
                raw_store(None, None, 106),
            ],
            worker,
            4,
            KvStateAgentRawMode::ResidencyV2,
            &mut framework,
            &mut cache_owner,
        );
        assert_eq!(
            normalized.cache_owner_fault,
            Some("unsupported KVCC storage medium")
        );
        assert_eq!(normalized.events.len(), 1);
        assert_eq!(
            normalized.events[0].placement.residency_domain,
            ResidencyDomain::Worker
        );

        assert_eq!(
            ZmqEventNormalizer::new(4)
                .preprocess_with_reason(raw_store(Some("CPU"), Some("kvcc"), 107), worker)
                .unwrap_err(),
            dynamo_kv_router::zmq_wire::ZmqEventFilterReason::UnsupportedSourceKind
        );
    }

    async fn send_ingress(
        tx: &mpsc::Sender<IngressBatch>,
        generation: u64,
        source_cursor: u64,
        events: Vec<PlacementEvent>,
    ) -> Result<()> {
        tx.send(IngressBatch {
            generation,
            source_cursor,
            chunk_index: 0,
            final_chunk: true,
            events,
            cache_owner_fault: None,
        })
        .await
        .context("state-agent test ingress queue closed")
    }

    async fn detach(control: &mpsc::Sender<ControlCommand>, generation: u64) -> Result<u64> {
        let (response, received) = oneshot::channel();
        control
            .send(ControlCommand::BeginDetach {
                generation,
                response,
            })
            .await
            .unwrap();
        received.await.unwrap()?;
        let (response, received) = oneshot::channel();
        control
            .send(ControlCommand::CompleteDetach {
                generation,
                response,
            })
            .await
            .unwrap();
        received.await.unwrap()
    }

    async fn quarantine(
        control: &mpsc::Sender<ControlCommand>,
        generation: u64,
        reason: &'static str,
    ) {
        let (response, received) = oneshot::channel();
        control
            .send(ControlCommand::Quarantine {
                generation,
                reason,
                response: Some(response),
            })
            .await
            .unwrap();
        received.await.unwrap();
    }

    #[tokio::test]
    async fn one_stream_lifecycle_preserves_cache_owner_across_worker_reattach() {
        let owner = cache_owner_id();
        let first_worker = WorkerWithDpRank::new(17, 3);
        let replacement_worker = WorkerWithDpRank::new(29, 3);
        let identity = KvStateAgentIdentity {
            cache_owner_id: owner,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
        };
        let status = Arc::new(ArcSwap::from_pointee(KvStateAgentStatus {
            identity: identity.clone(),
            attachment: None,
            cache_owner_ready: true,
            outbound_cursor: 0,
        }));
        let cancel = CancellationToken::new();
        let local_indexer = Arc::new(LocalKvIndexer::new(
            cancel.child_token(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            32,
        ));
        let sink = RecordingSink::default();
        let output = sink.events.clone();
        let publications = sink.publications.clone();
        let (control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
        let (ingress_tx, ingress_rx) = mpsc::channel(INGRESS_QUEUE_CAPACITY);
        let task = tokio::spawn(
            Coordinator {
                publisher: sink,
                local_indexer: local_indexer.clone(),
                identity: identity.clone(),
                slot_dp_rank: 3,
                status: status.clone(),
                control_rx,
                ingress_rx,
                attachment: None,
                ingress_generation: None,
                next_outbound_id: 1,
                next_attachment_generation: 1,
                source_cursor: SourceCursorState::default(),
                dedup: EventDedupFilter::new(),
                worker_domain_failed: false,
                cache_owner_domain_failed: false,
                source_failed: false,
            }
            .run(cancel.child_token()),
        );

        let first = send_attach(
            &control_tx,
            Attachment {
                generation: 0,
                worker: first_worker,
                vllm_zmq_endpoint: "tcp://framework-7".to_string(),
                ready: false,
                cache_readable: true,
                ready_at_outbound_cursor: 0,
            },
        )
        .await
        .unwrap();
        assert_eq!(first.ready_at_outbound_cursor, 1);
        assert_eq!(first.generation, 1);

        send_ingress(
            &ingress_tx,
            first.generation,
            1,
            vec![
                store(first_worker, ResidencyDomain::Worker, None, 101, 11),
                store(first_worker, ResidencyDomain::CacheOwner, None, 101, 11),
            ],
        )
        .await
        .unwrap();

        send_ingress(
            &ingress_tx,
            first.generation,
            2,
            vec![store(
                first_worker,
                ResidencyDomain::CacheOwner,
                Some(101),
                102,
                12,
            )],
        )
        .await
        .unwrap();
        quarantine(&control_tx, first.generation, "test interruption").await;
        assert!(!status.load().attachment.as_ref().unwrap().ready);
        assert_eq!(detach(&control_tx, first.generation).await.unwrap(), 5);

        let replacement = send_attach(
            &control_tx,
            Attachment {
                generation: 0,
                worker: replacement_worker,
                vllm_zmq_endpoint: "tcp://framework-8".to_string(),
                ready: false,
                cache_readable: true,
                ready_at_outbound_cursor: 0,
            },
        )
        .await
        .unwrap();
        assert_eq!(replacement.ready_at_outbound_cursor, 6);
        assert_eq!(replacement.generation, 2);
        assert_eq!(
            status.load().attachment.as_ref().unwrap().worker,
            replacement_worker
        );

        let events = output.lock().unwrap().clone();
        assert_eq!(
            events
                .iter()
                .map(|event| event.event.event_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4, 5, 6]
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| {
                    event.resolved_residency_domain() == Ok(ResidencyDomain::CacheOwner)
                })
                .count(),
            2
        );
        assert_eq!(
            publications.load(Ordering::Relaxed),
            5,
            "each ingress chunk is published once rather than once per event"
        );

        let WorkerKvQueryResponse::TreeDump { events, .. } =
            local_indexer.get_events_in_id_range(None, None).await
        else {
            panic!("expected a complete local snapshot");
        };
        assert_eq!(
            events
                .iter()
                .filter(|event| event.state_source == Some(owner))
                .count(),
            2,
            "Worker detach and replacement must preserve exact CacheOwner state"
        );

        control_tx.send(ControlCommand::Shutdown).await.unwrap();
        task.await.unwrap();
    }

    #[tokio::test]
    async fn detach_accepts_cache_owner_ingress_after_readiness_withdrawal() {
        let owner = cache_owner_id();
        let worker = WorkerWithDpRank::new(17, 3);
        let identity = KvStateAgentIdentity {
            cache_owner_id: owner,
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
        };
        let status = Arc::new(ArcSwap::from_pointee(KvStateAgentStatus {
            identity: identity.clone(),
            attachment: None,
            cache_owner_ready: true,
            outbound_cursor: 0,
        }));
        let cancel = CancellationToken::new();
        let local_indexer = Arc::new(LocalKvIndexer::new(
            cancel,
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            32,
        ));
        let sink = RecordingSink::default();
        let output = sink.events.clone();
        let (_control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
        let (_ingress_tx, ingress_rx) = mpsc::channel(INGRESS_QUEUE_CAPACITY);
        let mut coordinator = Coordinator {
            publisher: sink,
            local_indexer: local_indexer.clone(),
            identity,
            slot_dp_rank: 3,
            status,
            control_rx,
            ingress_rx,
            attachment: None,
            ingress_generation: None,
            next_outbound_id: 1,
            next_attachment_generation: 1,
            source_cursor: SourceCursorState::default(),
            dedup: EventDedupFilter::new(),
            worker_domain_failed: false,
            cache_owner_domain_failed: false,
            source_failed: false,
        };
        let mut attachment = Attachment {
            generation: 0,
            worker,
            vllm_zmq_endpoint: "tcp://framework".to_string(),
            ready: false,
            cache_readable: true,
            ready_at_outbound_cursor: 0,
        };
        coordinator.attach(&mut attachment).await.unwrap();

        coordinator.begin_detach(attachment.generation).unwrap();
        coordinator
            .handle_ingress(IngressBatch {
                generation: attachment.generation,
                source_cursor: 1,
                chunk_index: 0,
                final_chunk: true,
                events: vec![
                    store(worker, ResidencyDomain::Worker, None, 101, 11),
                    store(worker, ResidencyDomain::CacheOwner, None, 201, 21),
                ],
                cache_owner_fault: None,
            })
            .await
            .unwrap();
        assert_eq!(
            coordinator
                .complete_detach(attachment.generation)
                .await
                .unwrap(),
            4
        );

        let events = output.lock().unwrap().clone();
        assert_eq!(
            events
                .iter()
                .map(|event| (
                    event.event.event_id,
                    event.resolved_residency_domain().unwrap(),
                    matches!(event.event.data, KvCacheEventData::Cleared),
                ))
                .collect::<Vec<_>>(),
            vec![
                (1, ResidencyDomain::Worker, true),
                (2, ResidencyDomain::Worker, false),
                (3, ResidencyDomain::CacheOwner, false),
                (4, ResidencyDomain::Worker, true),
            ]
        );
        let WorkerKvQueryResponse::TreeDump { events, .. } =
            local_indexer.get_events_in_id_range(None, None).await
        else {
            panic!("expected full snapshot")
        };
        assert_eq!(
            events
                .iter()
                .filter(|event| event.state_source == Some(owner))
                .count(),
            1
        );

        let mut replacement = Attachment {
            generation: 0,
            worker,
            vllm_zmq_endpoint: "tcp://framework-2".to_string(),
            ready: false,
            cache_readable: true,
            ready_at_outbound_cursor: 0,
        };
        coordinator.attach(&mut replacement).await.unwrap();
        coordinator
            .handle_ingress(IngressBatch {
                generation: replacement.generation,
                source_cursor: 1,
                chunk_index: 0,
                final_chunk: false,
                events: vec![store(worker, ResidencyDomain::CacheOwner, None, 301, 31)],
                cache_owner_fault: None,
            })
            .await
            .unwrap();
        coordinator.begin_detach(replacement.generation).unwrap();
        coordinator
            .complete_detach(replacement.generation)
            .await
            .unwrap();
        assert!(
            !coordinator.status.load().cache_owner_ready,
            "an incomplete raw batch must fail CacheOwner closed"
        );
    }

    #[tokio::test]
    async fn failed_chunk_reserves_locally_applied_event_ids() {
        let worker = WorkerWithDpRank::new(17, 3);
        let identity = KvStateAgentIdentity {
            cache_owner_id: cache_owner_id(),
            publisher_id: 41,
            protocol_version: KvStateProtocolVersion::V2,
        };
        let status = Arc::new(ArcSwap::from_pointee(KvStateAgentStatus {
            identity: identity.clone(),
            attachment: None,
            cache_owner_ready: true,
            outbound_cursor: 0,
        }));
        let local_indexer = Arc::new(LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            32,
        ));
        let sink = RecordingSink::default();
        let fail_publication = sink.fail.clone();
        let (_control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
        let (_ingress_tx, ingress_rx) = mpsc::channel(INGRESS_QUEUE_CAPACITY);
        let mut coordinator = Coordinator {
            publisher: sink,
            local_indexer: local_indexer.clone(),
            identity,
            slot_dp_rank: 3,
            status: status.clone(),
            control_rx,
            ingress_rx,
            attachment: None,
            ingress_generation: None,
            next_outbound_id: 1,
            next_attachment_generation: 1,
            source_cursor: SourceCursorState::default(),
            dedup: EventDedupFilter::new(),
            worker_domain_failed: false,
            cache_owner_domain_failed: false,
            source_failed: false,
        };
        let mut attachment = Attachment {
            generation: 0,
            worker,
            vllm_zmq_endpoint: "tcp://framework".to_string(),
            ready: false,
            cache_readable: true,
            ready_at_outbound_cursor: 0,
        };
        coordinator.attach(&mut attachment).await.unwrap();
        let stale_event = store(worker, ResidencyDomain::Worker, None, 99, 9);
        assert!(
            coordinator
                .handle_ingress(IngressBatch {
                    generation: attachment.generation + 1,
                    source_cursor: 1,
                    chunk_index: 0,
                    final_chunk: true,
                    events: vec![stale_event; MAX_INGRESS_EVENTS + 1],
                    cache_owner_fault: None,
                })
                .await
                .is_err()
        );
        assert!(status.load().attachment.as_ref().unwrap().ready);

        fail_publication.store(true, Ordering::Relaxed);
        assert!(
            coordinator
                .handle_ingress(IngressBatch {
                    generation: attachment.generation,
                    source_cursor: 1,
                    chunk_index: 0,
                    final_chunk: true,
                    events: vec![store(worker, ResidencyDomain::Worker, None, 101, 11)],
                    cache_owner_fault: None,
                })
                .await
                .is_err()
        );
        assert_eq!(local_indexer.current_event_id(), 2);
        assert_eq!(status.load().outbound_cursor, 2);
        assert!(!status.load().cache_owner_ready);
    }
}
