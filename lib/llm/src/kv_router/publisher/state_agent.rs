// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Externalizable vLLM KV state agent.
//!
//! One instance owns exactly one stable DP slot, one local recovery index, one
//! semantic event publisher, and one outbound cursor. A standalone host may
//! multiplex these per-slot agents on one DRT without coupling their durable
//! identity to its connection ID.
//!
//! Cross-incarnation KVCR continuity is not implemented by this foundation.
//! CacheOwner routing is advisory: live sequence gaps are reported but do not
//! claim that every committed KVCR residency transition reached this agent.
//!
//! CacheOwner events bypass Worker refcount bookkeeping because KVCR guarantees
//! at most one logical residency per (CacheOwner, tier, block hash). This avoids
//! amplifying replayed Stores, but does not make arbitrary duplicate Removes or
//! out-of-order delivery replay-convergent. Missing a committed transition is
//! not recoverable from the live stream alone.
//!
//! TODO(#13044): require replacement KVCR/vLLM ingress to provide at-least-once
//! delivery across the incarnation boundary using an overlapping journal suffix
//! followed atomically by live events. The later authoritative mode must keep
//! CacheOwner unready after a detected gap or replay-window miss until a fuller
//! resynchronization completes.

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
        KvCacheEvent, KvCacheEventData, PlacementEvent, ResidencyDomain, ResidencyOwner,
        RouterEvent, RouterHintSourceMetadata, StorageTier, WorkerWithDpRank,
    },
    zmq_wire::{KvEventOwnership, ZmqEventNormalizer},
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
    discovery::kv_state_agent::{
        KV_STATE_ATTACHMENT_TOPIC_V2, KV_STATE_EVENT_TOPIC_V2, KV_STATE_SOURCE_TOPIC_V2,
        KvStateAttachmentAdvertisement, KvStateIngressProtocol, KvStateSourceAdvertisement,
        attachment_record_id,
    },
    kv_router::{
        WORKER_KV_INDEXER_BUFFER_SIZE, indexer::start_worker_kv_query_endpoint_with_status,
        metrics::kv_publisher_metrics,
    },
    utils::zmq::{SubSocket, connect_sub_socket, multipart_message},
};

use super::{
    DEFAULT_MAX_BATCH_BLOCKS,
    batching::PlacementEventCoalescer,
    dedup::{EventDedupFilter, EventDedupPolicy},
    sinks::{EventPlanePublisher, RouterEventBatchSink, admit_local_event},
    zmq_listener::{DecodedZmqKvBatch, decode_zmq_kv_batch},
};

const CONTROL_QUEUE_CAPACITY: usize = 16;
const MAX_INGRESS_EVENTS: usize = 128;
const MAX_INGRESS_BLOCKS: usize = 8_192;
const MAX_ZMQ_PAYLOAD_BYTES: usize = 8 * 1024 * 1024;

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
    pub global_dp_rank: u32,
    pub router_hint_source: Option<RouterHintSourceMetadata>,
}

#[derive(Debug, Clone)]
pub struct KvStateAgentVllmSource {
    pub endpoint: String,
    pub topic: String,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub ingress_protocol: KvStateIngressProtocol,
}

#[derive(Debug, Clone)]
pub struct KvStateAgentAttachmentConfig {
    pub producer_instance: dynamo_runtime::component::Instance,
    pub intent_incarnation: u64,
    pub worker: WorkerWithDpRank,
    pub vllm_source: KvStateAgentVllmSource,
}

#[derive(Clone)]
pub struct KvStateAgentConfig {
    pub endpoint: Endpoint,
    pub kv_state_endpoint: EndpointId,
    pub slot: KvStateAgentSlotConfig,
    pub kv_block_size: u32,
    pub ingress_protocol: KvStateIngressProtocol,
}

#[derive(Debug, Clone)]
struct Attachment {
    generation: u64,
    producer_instance: dynamo_runtime::component::Instance,
    intent_incarnation: u64,
    worker: WorkerWithDpRank,
    vllm_zmq_endpoint: String,
    raw_topic: String,
    ingress_protocol: KvStateIngressProtocol,
    ready: bool,
    ready_at_outbound_cursor: u64,
}

impl Attachment {
    fn status(&self) -> KvStateAttachmentStatus {
        KvStateAttachmentStatus {
            generation: self.generation,
            worker: self.worker,
            ready: self.ready,
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
    source_fault: Option<&'static str>,
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

    fn accept_after_loss(&mut self, batch: &IngressBatch) {
        if batch.final_chunk {
            self.completed = Some(batch.source_cursor);
            self.active = None;
        } else {
            self.active = Some((batch.source_cursor, batch.chunk_index));
        }
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
        response: oneshot::Sender<()>,
    },
    AbortAttach {
        generation: u64,
        response: oneshot::Sender<()>,
    },
    Shutdown {
        response: oneshot::Sender<()>,
    },
}

struct Coordinator<P> {
    publisher: P,
    local_indexer: Arc<LocalKvIndexer>,
    identity: KvStateAgentIdentity,
    slot_dp_rank: u32,
    status: Arc<ArcSwap<KvStateAgentStatus>>,
    withdrawal_tx: mpsc::UnboundedSender<u64>,
    control_rx: mpsc::Receiver<ControlCommand>,
    ingress_rx: mpsc::UnboundedReceiver<IngressBatch>,
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
                let current = self
                    .attachment
                    .as_ref()
                    .is_some_and(|attachment| attachment.generation == generation);
                self.quarantine(generation, reason);
                if current {
                    self.request_attachment_withdrawal(generation);
                }
                let _ = response.send(());
                false
            }
            ControlCommand::AbortAttach {
                generation,
                response,
            } => {
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
                let _ = response.send(());
                false
            }
            ControlCommand::Shutdown { response } => {
                let _ = response.send(());
                true
            }
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
        // Resetting this per-attachment cursor does not prove KVCR continuity.
        // CacheOwner remains advisory; authoritative cross-incarnation replay is
        // separate from attachment readiness.
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
        self.ingress_generation = None;
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

        self.observe_source_cursor_lossy(&batch, "live ingress");

        if let Some(reason) = batch.source_fault {
            self.fail_source(batch.generation, reason);
            anyhow::bail!("incompatible raw source: {reason}");
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
        self.apply_and_publish_chunk(events).await
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

            self.observe_source_cursor_lossy(&batch, "detach drain");
            if let Some(reason) = batch.source_fault {
                self.fail_source(generation, reason);
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
            if let Err(error) = self.apply_and_publish_chunk(cache_owner_events).await {
                tracing::error!(%error, "Failed to drain CacheOwner residency during detach");
            }
        }
        if self.source_cursor.active.is_some() {
            tracing::warn!(
                generation,
                "Detaching after an incomplete raw source batch; CacheOwner remains advisory"
            );
            if let Some(metrics) = kv_publisher_metrics() {
                metrics.increment_engines_dropped_events(1);
            }
            self.source_cursor = SourceCursorState::default();
        }
    }

    fn observe_source_cursor_lossy(&mut self, batch: &IngressBatch, phase: &'static str) {
        if self.source_cursor.observe(batch) {
            return;
        }
        let previous_cursor = self.source_cursor.completed;
        let missing = previous_cursor
            .and_then(|previous| batch.source_cursor.checked_sub(previous.checked_add(1)?))
            .filter(|missing| *missing > 0)
            .unwrap_or(1);
        tracing::warn!(
            generation = batch.generation,
            source_cursor = batch.source_cursor,
            ?previous_cursor,
            chunk_index = batch.chunk_index,
            phase,
            "Raw KV source cursor gap or regression; continuing with advisory residency"
        );
        if let Some(metrics) = kv_publisher_metrics() {
            metrics.increment_engines_dropped_events(missing);
        }
        self.source_cursor.accept_after_loss(batch);
    }

    fn fail_source(&mut self, generation: u64, reason: &'static str) {
        self.source_failed = true;
        self.worker_domain_failed = true;
        self.cache_owner_domain_failed = true;
        self.quarantine(generation, reason);
        self.request_attachment_withdrawal(generation);
    }

    fn fail_cache_owner(&mut self, generation: u64, reason: &'static str) {
        self.cache_owner_domain_failed = true;
        self.ingress_generation = None;
        self.quarantine(generation, reason);
        self.request_attachment_withdrawal(generation);
        tracing::error!(generation, reason, "CacheOwner residency failed closed");
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
        let mut coalescer = PlacementEventCoalescer::new(DEFAULT_MAX_BATCH_BLOCKS);
        let mut coalesced = Vec::with_capacity(placement_events.len());
        for placement_event in placement_events {
            let exact_owner = match placement_event.placement.residency_domain {
                ResidencyDomain::Worker => ResidencyOwner::Worker(attachment_worker),
                ResidencyDomain::CacheOwner => {
                    ResidencyOwner::CacheOwner(self.identity.cache_owner_id)
                }
            };
            let key = (
                exact_owner,
                self.identity.cache_owner_id,
                placement_event.placement.tier,
            );
            coalesced.extend(coalescer.push(key, placement_event).into_iter().flatten());
        }
        coalesced.extend(coalescer.flush());
        let event_worker_id = attachment_worker.worker_id;
        let mut output = Vec::with_capacity(coalesced.len());
        let mut cleared_domains = Vec::new();
        let mut failed_barrier = None;

        for placement_event in coalesced {
            let domain = placement_event.placement.residency_domain;
            let dedup_policy = match domain {
                ResidencyDomain::Worker => EventDedupPolicy::RefCounted,
                // This state agent creates CacheOwner events only from KVCR ownership.
                ResidencyDomain::CacheOwner => EventDedupPolicy::SetLike,
            };
            let mut event = placement_event.event;
            if event.dp_rank != self.slot_dp_rank {
                self.fail_source(
                    self.attachment
                        .as_ref()
                        .expect("attachment was checked above")
                        .generation,
                    "ingress rank does not match stable slot",
                );
                anyhow::bail!(
                    "ingress rank {} does not match stable slot rank {}",
                    event.dp_rank,
                    self.slot_dp_rank
                );
            }
            let tier = placement_event.placement.tier;
            event.data = match event.data {
                KvCacheEventData::Removed(data) => {
                    let Some(filtered) = self.dedup.filter_remove_in_domain(
                        event.dp_rank,
                        tier,
                        domain,
                        dedup_policy,
                        data,
                    ) else {
                        continue;
                    };
                    KvCacheEventData::Removed(filtered)
                }
                KvCacheEventData::Stored(data) => {
                    self.dedup.track_store_in_domain(
                        event.dp_rank,
                        tier,
                        domain,
                        dedup_policy,
                        &data,
                    );
                    KvCacheEventData::Stored(data)
                }
                KvCacheEventData::Cleared => {
                    self.dedup
                        .clear_rank_domain(event.dp_rank, domain, dedup_policy);
                    KvCacheEventData::Cleared
                }
            };
            event.event_id = self.next_outbound_id;
            let Some(next_outbound_id) = self.next_outbound_id.checked_add(1) else {
                self.fail_source(
                    self.attachment
                        .as_ref()
                        .expect("attachment was checked above")
                        .generation,
                    "state-agent outbound cursor exhausted",
                );
                anyhow::bail!("state-agent outbound cursor exhausted");
            };
            self.next_outbound_id = next_outbound_id;

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
            // NOTE: Stored/Removed intentionally use the legacy lossy contract: queue
            // admission and one best-effort publish, with no completion acknowledgement.
            // Cleared is stronger and completes every affected tier before publication.
            // Do not add per-event completion or listener stop-and-wait without changing
            // this invariant and revalidating ingestion throughput.
            if let Err(error) = admit_local_event(Some(&self.local_indexer), &router_event).await {
                if matches!(router_event.event.data, KvCacheEventData::Cleared) {
                    self.fail_domain(
                        self.attachment
                            .as_ref()
                            .expect("attachment was checked above")
                            .generation,
                        domain,
                        "local reset barrier failed",
                    );
                    failed_barrier =
                        Some(anyhow::Error::new(error).context("local reset barrier failed"));
                    break;
                }
                tracing::warn!(
                    worker_id = event_worker_id,
                    dp_rank = router_event.event.dp_rank,
                    ?domain,
                    %error,
                    "Failed to admit ordinary residency event locally; continuing lossy stream"
                );
            }
            if matches!(router_event.event.data, KvCacheEventData::Cleared) {
                cleared_domains.push(domain);
            }
            output.push(router_event);
        }

        if !output.is_empty()
            && let Err(error) = self.publisher.publish_events(&output).await
        {
            if cleared_domains.is_empty() {
                tracing::warn!(
                    attempted_event_count = output.len(),
                    %error,
                    "Failed to publish ordinary state-agent events; continuing lossy stream"
                );
            } else {
                let generation = self
                    .attachment
                    .as_ref()
                    .expect("attachment was checked above")
                    .generation;
                for domain in cleared_domains {
                    self.fail_domain(generation, domain, "reset barrier publication failed");
                }
                self.publish_status();
                return Err(error).context("failed to publish reset barrier");
            }
        }
        self.publish_status();
        if let Some(error) = failed_barrier {
            return Err(error);
        }
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

    fn fail_domain(&mut self, generation: u64, domain: ResidencyDomain, reason: &'static str) {
        match domain {
            ResidencyDomain::Worker => {
                self.worker_domain_failed = true;
                self.ingress_generation = None;
                self.quarantine(generation, reason);
                self.request_attachment_withdrawal(generation);
            }
            ResidencyDomain::CacheOwner => self.fail_cache_owner(generation, reason),
        }
    }

    fn request_attachment_withdrawal(&self, generation: u64) {
        if self.withdrawal_tx.send(generation).is_err() {
            tracing::warn!(
                generation,
                "KV state-agent attachment withdrawal supervisor is closed"
            );
        }
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
    ingress_protocol: KvStateIngressProtocol,
    recovery_target: dynamo_runtime::component::Instance,
    identity: KvStateAgentIdentity,
    control_tx: mpsc::Sender<ControlCommand>,
    ingress_tx: mpsc::UnboundedSender<IngressBatch>,
    status: Arc<ArcSwap<KvStateAgentStatus>>,
    cancel: CancellationToken,
    coordinator: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    withdrawal_supervisor: StdMutex<Option<tokio::task::JoinHandle<()>>>,
    vllm_listener: Mutex<Option<VllmListener>>,
    recovery_endpoint: Mutex<Option<StartedEndpoint>>,
    lifecycle: Mutex<()>,
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
        let cancel = component.drt().primary_token().child_token();
        let local_indexer = Arc::new(LocalKvIndexer::new(
            cancel.child_token(),
            config.kv_block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            WORKER_KV_INDEXER_BUFFER_SIZE,
        ));

        if config.ingress_protocol != KvStateIngressProtocol::VllmResidencyV1 {
            anyhow::bail!("external state-agent slots require vllm_residency_v1 ingress");
        }
        let event_publisher = EventPublisher::for_endpoint_id(
            config.endpoint.drt(),
            &config.kv_state_endpoint,
            KV_STATE_EVENT_TOPIC_V2,
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
            cache_owner_ready: true,
            outbound_cursor: 0,
        }));

        let recovery_endpoint = start_worker_kv_query_endpoint_with_status(
            component.clone(),
            publisher_id,
            0,
            config.slot.global_dp_rank,
            local_indexer.clone(),
            Some(status.clone()),
        )
        .await
        .context("failed to start state-agent recovery/control endpoint")?;

        let recovery_target = recovery_endpoint.instance().clone();
        let persistent_ad = KvStateSourceAdvertisement {
            cache_owner_id: identity.cache_owner_id,
            global_dp_rank: config.slot.global_dp_rank,
            kv_state_endpoint: config.kv_state_endpoint.clone(),
            indexer_domain_id: identity.cache_owner_id.pool().indexer_domain(),
            kv_block_size: config.kv_block_size,
            ingress_protocol: config.ingress_protocol,
            publisher_id,
            protocol_version: identity.protocol_version,
            event_topic: KV_STATE_EVENT_TOPIC_V2.to_string(),
            recovery_control_target: recovery_target.clone(),
            router_hint_source: config.slot.router_hint_source.clone(),
        };
        let persistent_source = match register_advertisement(
            &component,
            EventScope::Endpoint {
                endpoint: config.kv_state_endpoint.clone(),
            },
            KV_STATE_SOURCE_TOPIC_V2,
            publisher_id,
            &persistent_ad,
        )
        .await
        {
            Ok(instance) => instance,
            Err(error) => {
                let _ = recovery_endpoint.shutdown().await;
                return Err(error).context("failed to advertise persistent state source");
            }
        };

        let persistent_source = Arc::new(Mutex::new(Some(persistent_source)));
        let attachment_source = Arc::new(Mutex::new(None));
        let (withdrawal_tx, withdrawal_rx) = mpsc::unbounded_channel();
        let withdrawal_supervisor =
            component
                .drt()
                .runtime()
                .secondary()
                .spawn(run_attachment_withdrawal_supervisor(
                    component.clone(),
                    publisher_id,
                    attachment_source.clone(),
                    withdrawal_rx,
                    cancel.child_token(),
                ));

        let (control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
        let (ingress_tx, ingress_rx) = mpsc::unbounded_channel();
        let coordinator_cancel = cancel.child_token();
        let coordinator = component.drt().runtime().secondary().spawn(
            Coordinator {
                publisher: EventPlanePublisher(event_publisher),
                local_indexer,
                identity: identity.clone(),
                slot_dp_rank: config.slot.global_dp_rank,
                status: status.clone(),
                withdrawal_tx,
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
            .run(coordinator_cancel),
        );

        let agent = Self {
            component,
            kv_state_endpoint: config.kv_state_endpoint,
            kv_block_size: config.kv_block_size,
            slot_dp_rank: config.slot.global_dp_rank,
            ingress_protocol: config.ingress_protocol,
            recovery_target,
            identity,
            control_tx,
            ingress_tx,
            status,
            cancel,
            coordinator: StdMutex::new(Some(coordinator)),
            withdrawal_supervisor: StdMutex::new(Some(withdrawal_supervisor)),
            vllm_listener: Mutex::new(None),
            recovery_endpoint: Mutex::new(Some(recovery_endpoint)),
            lifecycle: Mutex::new(()),
            persistent_source,
            attachment_source,
            attachment: Mutex::new(None),
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
        let _lifecycle = self.lifecycle.lock().await;
        let (response, received) = oneshot::channel();
        self.control_tx
            .send(ControlCommand::Quarantine {
                generation,
                reason,
                response,
            })
            .await
            .context("state-agent coordinator is closed")?;
        received
            .await
            .context("state-agent quarantine acknowledgement was dropped")?;
        unregister_required(
            &self.component,
            &self.attachment_source,
            "attachment advertisement",
            &self.cancel,
        )
        .await?;
        if let Some(listener) = self.vllm_listener.lock().await.take() {
            listener.stop().await;
        }
        Ok(())
    }

    /// Attach a new engine incarnation after the prior attachment has fully detached.
    ///
    /// The API accepts a different WorkerId at the stable slot's global rank.
    /// Publishing the attachment advertisement commits listener readiness: it
    /// happens only after the raw event listener is established. Projection is
    /// separately recovery-gated; do not add another readability phase.
    pub async fn attach(
        &self,
        config: KvStateAgentAttachmentConfig,
    ) -> Result<KvStateAttachmentStatus> {
        let _lifecycle = self.lifecycle.lock().await;
        let KvStateAgentAttachmentConfig {
            producer_instance,
            intent_incarnation,
            worker,
            vllm_source,
        } = config;
        if worker.dp_rank != self.slot_dp_rank {
            anyhow::bail!("attachment DP rank does not match the stable state-agent slot");
        }
        if vllm_source.ingress_protocol != self.ingress_protocol {
            anyhow::bail!("raw KV protocol mode is immutable for one state-agent publisher");
        }
        if self.attachment.lock().await.is_some()
            || self.attachment_source.lock().await.is_some()
            || self.vllm_listener.lock().await.is_some()
        {
            anyhow::bail!("previous engine attachment has not fully detached");
        }

        let attachment = send_attach(
            &self.control_tx,
            Attachment {
                generation: 0,
                producer_instance,
                intent_incarnation,
                worker,
                vllm_zmq_endpoint: vllm_source.endpoint.clone(),
                raw_topic: vllm_source.topic.clone(),
                ingress_protocol: vllm_source.ingress_protocol,
                ready: false,
                ready_at_outbound_cursor: 0,
            },
        )
        .await?;

        let listener = match start_vllm_listener(
            self.component.clone(),
            vllm_source,
            worker,
            attachment.generation,
            self.kv_block_size,
            self.ingress_tx.clone(),
            self.control_tx.clone(),
            self.status.clone(),
            self.cancel.child_token(),
        )
        .await
        {
            Ok(listener) => listener,
            Err(error) => {
                abort_attach(&self.control_tx, attachment.generation).await;
                return Err(error).context("failed to start the attached raw KV listener");
            }
        };

        let attachment_ad =
            attachment_advertisement(&self.identity, &self.recovery_target, &attachment);
        let attachment_source = match register_advertisement(
            &self.component,
            EventScope::Endpoint {
                endpoint: self.kv_state_endpoint.clone(),
            },
            KV_STATE_ATTACHMENT_TOPIC_V2,
            attachment_record_id(self.identity.publisher_id, attachment.generation),
            &attachment_ad,
        )
        .await
        {
            Ok(source) => source,
            Err(error) => {
                if let Err(rollback_error) =
                    rollback_started_attach(&self.control_tx, listener, attachment.generation).await
                {
                    tracing::error!(%rollback_error, "Failed to roll back active KV attachment");
                }
                return Err(error).context("failed to advertise reattached engine attachment");
            }
        };

        if !self
            .status()
            .attachment
            .as_ref()
            .is_some_and(|current| current.ready && current.generation == attachment.generation)
        {
            let _ = self
                .component
                .drt()
                .discovery()
                .unregister(attachment_source)
                .await;
            rollback_started_attach(&self.control_tx, listener, attachment.generation)
                .await
                .context("failed to roll back terminated KV attachment")?;
            anyhow::bail!("raw KV listener terminated before attachment advertisement completed");
        }

        *self.attachment_source.lock().await = Some(attachment_source);
        *self.attachment.lock().await = Some(attachment.clone());
        *self.vllm_listener.lock().await = Some(listener);
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
            &self.attachment_source,
            "attachment advertisement",
            &self.cancel,
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

        if let Some(generation) = self
            .status()
            .attachment
            .as_ref()
            .map(|attachment| attachment.generation)
        {
            let (response, received) = oneshot::channel();
            if self
                .control_tx
                .send(ControlCommand::Quarantine {
                    generation,
                    reason: "state-agent shutdown",
                    response,
                })
                .await
                .is_ok()
            {
                let _ = received.await;
            }
        }
        // Withdraw readiness before stopping ingress or the coordinator. The
        // persistent source is removed later, after coordination is quiescent.
        unregister_if_present(&self.component, &self.attachment_source).await;
        if let Some(listener) = self.vllm_listener.lock().await.take() {
            listener.stop().await;
        }

        let (response, received) = oneshot::channel();
        if self
            .control_tx
            .send(ControlCommand::Shutdown { response })
            .await
            .is_ok()
        {
            let _ = received.await;
        }
        self.cancel.cancel();
        let coordinator = self.coordinator.lock().unwrap().take();
        if let Some(coordinator) = coordinator {
            let _ = coordinator.await;
        }
        let withdrawal_supervisor = self.withdrawal_supervisor.lock().unwrap().take();
        if let Some(withdrawal_supervisor) = withdrawal_supervisor {
            let _ = withdrawal_supervisor.await;
        }

        *self.attachment.lock().await = None;
        unregister_if_present(&self.component, &self.persistent_source).await;
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
        if let Ok(mut supervisor) = self.withdrawal_supervisor.lock()
            && let Some(supervisor) = supervisor.take()
        {
            supervisor.abort();
        }
    }
}

async fn run_attachment_withdrawal_supervisor(
    component: Component,
    publisher_id: u64,
    attachment_source: Arc<Mutex<Option<DiscoveryInstance>>>,
    mut withdrawals: mpsc::UnboundedReceiver<u64>,
    cancel: CancellationToken,
) {
    loop {
        let generation = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            generation = withdrawals.recv() => generation,
        };
        let Some(generation) = generation else {
            break;
        };
        let mut delay = Duration::from_millis(100);
        loop {
            match unregister_attachment_generation(
                &component,
                &attachment_source,
                publisher_id,
                generation,
            )
            .await
            {
                Ok(()) => break,
                Err(error) => {
                    tracing::warn!(generation, %error, ?delay, "Retrying failed KV state attachment withdrawal");
                }
            }
            tokio::select! {
                biased;
                _ = cancel.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
            delay = delay.saturating_mul(2).min(Duration::from_secs(5));
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
    let (response, received) = oneshot::channel();
    if control_tx
        .send(ControlCommand::AbortAttach {
            generation,
            response,
        })
        .await
        .is_ok()
    {
        let _ = received.await;
    }
}

async fn rollback_started_attach(
    control_tx: &mpsc::Sender<ControlCommand>,
    listener: VllmListener,
    generation: u64,
) -> Result<()> {
    let (response, received) = oneshot::channel();
    let begin = async {
        control_tx
            .send(ControlCommand::BeginDetach {
                generation,
                response,
            })
            .await
            .context("state-agent coordinator is closed")?;
        received
            .await
            .context("state-agent detach begin was dropped")?
    }
    .await;
    listener.stop().await;
    begin?;

    let (response, received) = oneshot::channel();
    control_tx
        .send(ControlCommand::CompleteDetach {
            generation,
            response,
        })
        .await
        .context("state-agent coordinator is closed")?;
    received
        .await
        .context("state-agent detach completion was dropped")??;
    Ok(())
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
        producer_instance: attachment.producer_instance.clone(),
        intent_incarnation: attachment.intent_incarnation,
        worker: attachment.worker,
        ingress_protocol: attachment.ingress_protocol,
        raw_zmq_endpoint: attachment.vllm_zmq_endpoint.clone(),
        raw_topic: attachment.raw_topic.clone(),
        ready_at_outbound_cursor: attachment.ready_at_outbound_cursor,
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

async fn unregister_attachment_generation(
    component: &Component,
    instance: &Arc<Mutex<Option<DiscoveryInstance>>>,
    publisher_id: u64,
    generation: u64,
) -> Result<()> {
    let expected_id = attachment_record_id(publisher_id, generation);
    let current = instance.lock().await.clone();
    let Some(current) = current else {
        return Ok(());
    };
    let matches_generation = matches!(
        &current,
        DiscoveryInstance::EventSource {
            publisher_id: record_id,
            topic,
            ..
        } if *record_id == expected_id && topic == KV_STATE_ATTACHMENT_TOPIC_V2
    );
    if !matches_generation {
        return Ok(());
    }
    component
        .drt()
        .discovery()
        .unregister(current.clone())
        .await
        .context("failed to withdraw failed KV state attachment")?;
    let mut instance = instance.lock().await;
    if instance.as_ref() == Some(&current) {
        *instance = None;
    }
    Ok(())
}

async fn unregister_required(
    component: &Component,
    instance: &Arc<Mutex<Option<DiscoveryInstance>>>,
    description: &str,
    cancel: &CancellationToken,
) -> Result<()> {
    let current = instance.lock().await.clone();
    let Some(current) = current else {
        return Ok(());
    };
    let mut delay = Duration::from_millis(100);
    loop {
        match component
            .drt()
            .discovery()
            .unregister(current.clone())
            .await
        {
            Ok(()) => break,
            Err(error) => {
                tracing::warn!(%error, ?delay, description, "Retrying required discovery withdrawal");
            }
        }
        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                anyhow::bail!("state-agent lease ended before {description} was withdrawn");
            }
            _ = tokio::time::sleep(delay) => {}
        }
        delay = delay.saturating_mul(2).min(Duration::from_secs(5));
    }
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
async fn start_vllm_listener(
    component: Component,
    source: KvStateAgentVllmSource,
    worker: WorkerWithDpRank,
    generation: u64,
    kv_block_size: u32,
    ingress_tx: mpsc::UnboundedSender<IngressBatch>,
    control_tx: mpsc::Sender<ControlCommand>,
    status: Arc<ArcSwap<KvStateAgentStatus>>,
    cancel: CancellationToken,
) -> Result<VllmListener> {
    let socket = connect_sub_socket(&source.endpoint, Some(&source.topic)).await?;
    let listener_cancel = cancel.child_token();
    let task_cancel = listener_cancel.clone();
    let handle = component.drt().runtime().secondary().spawn(async move {
        let result = run_vllm_listener(VllmListenerTask {
            source: &source,
            worker,
            generation,
            kv_block_size,
            ingress_tx,
            socket,
            status,
            cancel: task_cancel.clone(),
        })
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
            let (response, received) = oneshot::channel();
            let command = ControlCommand::Quarantine {
                generation,
                reason: "framework listener terminated",
                response,
            };
            if control_tx.send(command).await.is_ok() {
                let _ = received.await;
            }
            // The acknowledged quarantine enqueues a generation-fenced
            // withdrawal retained by the agent supervisor until discovery
            // accepts it or this source lease ends.
        }
    });
    Ok(VllmListener {
        cancel: listener_cancel,
        handle,
    })
}

struct VllmListenerTask<'a> {
    source: &'a KvStateAgentVllmSource,
    worker: WorkerWithDpRank,
    generation: u64,
    kv_block_size: u32,
    ingress_tx: mpsc::UnboundedSender<IngressBatch>,
    socket: SubSocket,
    status: Arc<ArcSwap<KvStateAgentStatus>>,
    cancel: CancellationToken,
}

async fn run_vllm_listener(task: VllmListenerTask<'_>) -> Result<()> {
    let VllmListenerTask {
        source,
        worker,
        generation,
        kv_block_size,
        ingress_tx,
        mut socket,
        status,
        cancel,
    } = task;
    let mut framework_normalizer = ZmqEventNormalizer::new(kv_block_size)
        .with_image_token_id(source.image_token_id)
        .with_video_token_id(source.video_token_id);
    let mut cache_owner_normalizer = ZmqEventNormalizer::new(kv_block_size)
        .with_image_token_id(source.image_token_id)
        .with_video_token_id(source.video_token_id);
    loop {
        if !status
            .load()
            .attachment
            .as_ref()
            .is_some_and(|attachment| attachment.generation == generation && attachment.ready)
        {
            return Ok(());
        }
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
        if frames
            .get(2)
            .is_some_and(|payload| payload.len() > MAX_ZMQ_PAYLOAD_BYTES)
        {
            anyhow::bail!("ZMQ payload exceeds {MAX_ZMQ_PAYLOAD_BYTES} bytes");
        }
        let DecodedZmqKvBatch {
            source_cursor,
            batch,
        } = decode_zmq_kv_batch(frames).context("failed to decode framework KV event batch")?;
        let dp_rank = batch.data_parallel_rank.unwrap_or(0).cast_unsigned();
        if dp_rank != worker.dp_rank {
            anyhow::bail!(
                "vLLM batch rank {dp_rank} does not match configured rank {}",
                worker.dp_rank
            );
        }

        let NormalizedRawBatch {
            events,
            source_fault,
            cache_owner_fault,
        } = normalize_raw_batch(
            batch.events,
            worker,
            kv_block_size,
            source.ingress_protocol,
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
                source_fault: (chunk_index == 0).then_some(source_fault).flatten(),
                cache_owner_fault: (chunk_index == 0).then_some(cache_owner_fault).flatten(),
            };
            if cancel.is_cancelled() {
                return Ok(());
            }
            admit_ingress(&ingress_tx, envelope)?;
        }
    }
}

fn admit_ingress(
    ingress_tx: &mpsc::UnboundedSender<IngressBatch>,
    envelope: IngressBatch,
) -> Result<()> {
    ingress_tx
        .send(envelope)
        .context("state-agent ingress coordinator is closed")
}

struct NormalizedRawBatch {
    events: Vec<PlacementEvent>,
    source_fault: Option<&'static str>,
    cache_owner_fault: Option<&'static str>,
}

fn normalize_raw_batch(
    raw_events: Vec<dynamo_kv_router::zmq_wire::RawKvEvent>,
    worker: WorkerWithDpRank,
    kv_block_size: u32,
    ingress_protocol: KvStateIngressProtocol,
    framework_normalizer: &mut ZmqEventNormalizer,
    cache_owner_normalizer: &mut ZmqEventNormalizer,
) -> NormalizedRawBatch {
    // KVCR is the placement authority. vLLM only enriches committed KVCR
    // transitions with canonical routing metadata and serializes them on this
    // versioned stream. Any failure on that path quarantines CacheOwner below;
    // silently omitting a KVCR transition would make retained ownership lie.
    let mut events = Vec::with_capacity(raw_events.len().min(MAX_INGRESS_EVENTS));
    let mut source_fault = None;
    let mut cache_owner_fault = None;
    for raw_event in raw_events {
        let ownership = match raw_event.ownership() {
            Ok(ownership) => ownership,
            Err(_) => {
                source_fault.get_or_insert("unknown raw event ownership");
                continue;
            }
        };
        if ownership == KvEventOwnership::Kvcr
            && ingress_protocol != KvStateIngressProtocol::VllmResidencyV1
        {
            source_fault.get_or_insert("KVCR event on framework-only raw protocol");
            continue;
        }
        if ownership == KvEventOwnership::Kvcr
            && !matches!(
                raw_event.medium(),
                Some("CPU" | "CPU_PINNED" | "STORAGE" | "DISK")
            )
            && !matches!(
                raw_event,
                dynamo_kv_router::zmq_wire::RawKvEvent::AllBlocksCleared { .. }
            )
        {
            cache_owner_fault.get_or_insert("KVCR ownership has an unsupported storage medium");
            continue;
        }
        if ownership == KvEventOwnership::Kvcr
            && raw_event
                .block_size()
                .is_some_and(|size| size != kv_block_size as usize)
        {
            cache_owner_fault.get_or_insert("KVCR block size is incompatible");
            continue;
        }

        let normalizer = match ownership {
            KvEventOwnership::Framework => &mut *framework_normalizer,
            KvEventOwnership::Kvcr => &mut *cache_owner_normalizer,
        };
        let raw_event = match normalizer.preprocess_residency_with_reason(raw_event, worker) {
            Ok(raw_event) => raw_event,
            Err(reason) if ownership == KvEventOwnership::Kvcr => {
                tracing::warn!(?reason, "KVCR event enrichment failed");
                cache_owner_fault.get_or_insert("KVCR event enrichment failed");
                continue;
            }
            Err(_) => continue,
        };
        let Some(mut event) = normalizer.normalize_preprocessed(raw_event, 0, worker) else {
            if ownership == KvEventOwnership::Kvcr {
                cache_owner_fault.get_or_insert("KVCR event canonicalization failed");
            }
            continue;
        };
        match ownership {
            KvEventOwnership::Framework => {
                event.placement.residency_domain = ResidencyDomain::Worker;
            }
            KvEventOwnership::Kvcr => {
                if !matches!(
                    (&event.event.data, event.placement.tier),
                    (KvCacheEventData::Cleared, _)
                        | (_, StorageTier::HostPinned | StorageTier::Disk)
                ) {
                    cache_owner_fault.get_or_insert("unsupported KVCR storage medium");
                    continue;
                }
                event.placement.residency_domain = ResidencyDomain::CacheOwner;
            }
        }
        events.push(event);
    }
    NormalizedRawBatch {
        events,
        source_fault,
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
    use std::{
        collections::HashSet,
        sync::{
            Mutex as StdMutex,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
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
        published: Arc<tokio::sync::Notify>,
    }

    impl RecordingSink {
        async fn wait_for_publications(&self, count: usize) {
            loop {
                let published = self.published.notified();
                if self.publications.load(Ordering::Relaxed) >= count {
                    return;
                }
                published.await;
            }
        }
    }

    impl RouterEventBatchSink for RecordingSink {
        async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
            self.publications.fetch_add(1, Ordering::Relaxed);
            if self.fail.load(Ordering::Relaxed) {
                anyhow::bail!("injected state-agent publication failure");
            }
            self.events.lock().unwrap().extend_from_slice(events);
            self.published.notify_one();
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

    fn test_attachment(worker: WorkerWithDpRank, endpoint: &str) -> Attachment {
        Attachment {
            generation: 0,
            producer_instance: dynamo_runtime::component::Instance {
                component: "producer".to_string(),
                endpoint: "raw-kv".to_string(),
                namespace: "test".to_string(),
                instance_id: worker.worker_id,
                transport: dynamo_runtime::component::TransportType::Tcp(
                    "tcp://127.0.0.1:1234".to_string(),
                ),
                device_type: None,
                request_plane_codec: None,
            },
            intent_incarnation: worker.worker_id,
            worker,
            vllm_zmq_endpoint: endpoint.to_string(),
            raw_topic: "kv-events-v2".to_string(),
            ingress_protocol: KvStateIngressProtocol::VllmResidencyV1,
            ready: false,
            ready_at_outbound_cursor: 0,
        }
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

    fn raw_store(medium: Option<&str>, ownership: Option<&str>, block: u64) -> RawKvEvent {
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
            ownership: ownership.map(str::to_owned),
        }
    }

    #[test]
    fn raw_ownership_maps_domains_and_quarantines_unsupported_kvcr() {
        let worker = WorkerWithDpRank::new(17, 3);
        let mut framework = ZmqEventNormalizer::new(4);
        let mut cache_owner = ZmqEventNormalizer::new(4);
        let normalized = normalize_raw_batch(
            vec![
                raw_store(None, None, 101),
                raw_store(Some("CPU"), None, 102),
                raw_store(Some("CPU_PINNED"), Some("kvcr"), 103),
                raw_store(Some("STORAGE"), Some("kvcr"), 104),
                RawKvEvent::AllBlocksCleared {
                    ownership: Some("kvcr".to_string()),
                },
            ],
            worker,
            4,
            KvStateIngressProtocol::VllmResidencyV1,
            &mut framework,
            &mut cache_owner,
        );
        assert_eq!(normalized.source_fault, None);
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
                raw_store(Some("GPU"), Some("kvcr"), 105),
                raw_store(None, None, 106),
            ],
            worker,
            4,
            KvStateIngressProtocol::VllmResidencyV1,
            &mut framework,
            &mut cache_owner,
        );
        assert_eq!(
            normalized.cache_owner_fault,
            Some("KVCR ownership has an unsupported storage medium")
        );
        assert_eq!(normalized.source_fault, None);
        assert_eq!(normalized.events.len(), 1);
        assert_eq!(
            normalized.events[0].placement.residency_domain,
            ResidencyDomain::Worker
        );

        assert_eq!(
            ZmqEventNormalizer::new(4)
                .preprocess_with_reason(raw_store(Some("CPU"), Some("kvcr"), 107), worker)
                .unwrap_err(),
            dynamo_kv_router::zmq_wire::ZmqEventFilterReason::UnsupportedOwnership
        );

        let normalized = normalize_raw_batch(
            vec![raw_store(Some("CPU"), Some("future_owner"), 108)],
            worker,
            4,
            KvStateIngressProtocol::VllmResidencyV1,
            &mut framework,
            &mut cache_owner,
        );
        assert_eq!(normalized.source_fault, Some("unknown raw event ownership"));
    }

    #[tokio::test]
    async fn production_named_map_ownership_reaches_exact_local_recovery_owners() {
        let wire = (
            1.0f64,
            vec![
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [101],
                    "parent_block_hash": null,
                    "token_ids": [10, 11, 12, 13],
                    "block_size": 4,
                    "medium": "CPU_PINNED",
                    "locality": "LOCAL",
                    "ownership": "kvcr"
                }),
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [102, 202],
                    "parent_block_hash": 101,
                    "token_ids": [14, 15, 16, 17, 18, 19, 20, 21],
                    "block_size": 4,
                    "medium": "STORAGE",
                    "locality": "LOCAL",
                    "ownership": "kvcr"
                }),
                serde_json::json!({
                    "type": "BlockRemoved",
                    "block_hashes": [102],
                    "medium": "STORAGE",
                    "locality": "LOCAL",
                    "ownership": "kvcr"
                }),
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [105],
                    "parent_block_hash": null,
                    "token_ids": [22, 23, 24, 25],
                    "block_size": 4,
                    "medium": "CPU_PINNED",
                    "locality": "LOCAL",
                    "ownership": "kvcr"
                }),
                serde_json::json!({
                    "type": "BlockRemoved",
                    "block_hashes": [101],
                    "medium": "CPU_PINNED",
                    "locality": "LOCAL",
                    "ownership": "kvcr"
                }),
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [106],
                    "parent_block_hash": null,
                    "token_ids": [26, 27, 28, 29],
                    "block_size": 4,
                    "medium": "CPU",
                    "locality": "LOCAL"
                }),
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [104],
                    "parent_block_hash": null,
                    "token_ids": [30, 31, 32, 33],
                    "block_size": 4,
                    "medium": "GPU"
                }),
            ],
            Some(3i32),
        );
        let payload = rmp_serde::to_vec_named(&wire).unwrap();
        let batch = dynamo_kv_router::zmq_wire::decode_event_batch(&payload).unwrap();
        let worker = WorkerWithDpRank::new(17, 3);
        let mut framework = ZmqEventNormalizer::new(4);
        let mut cache_owner = ZmqEventNormalizer::new(4);
        let normalized = normalize_raw_batch(
            batch.events,
            worker,
            4,
            KvStateIngressProtocol::VllmResidencyV1,
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
                (ResidencyDomain::CacheOwner, StorageTier::HostPinned),
                (ResidencyDomain::CacheOwner, StorageTier::Disk),
                (ResidencyDomain::CacheOwner, StorageTier::Disk),
                (ResidencyDomain::CacheOwner, StorageTier::HostPinned),
                (ResidencyDomain::CacheOwner, StorageTier::HostPinned),
                (ResidencyDomain::Worker, StorageTier::HostPinned),
                (ResidencyDomain::Worker, StorageTier::Device),
            ]
        );

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
        let (_control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
        let (_ingress_tx, ingress_rx) = mpsc::unbounded_channel();
        let mut coordinator = Coordinator {
            publisher: RecordingSink::default(),
            local_indexer: local_indexer.clone(),
            identity,
            slot_dp_rank: 3,
            status,
            withdrawal_tx: mpsc::unbounded_channel().0,
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
        let mut attachment = test_attachment(worker, "tcp://framework");
        coordinator.attach(&mut attachment).await.unwrap();
        coordinator
            .handle_ingress(IngressBatch {
                generation: attachment.generation,
                source_cursor: 1,
                chunk_index: 0,
                final_chunk: true,
                events: normalized.events,
                source_fault: normalized.source_fault,
                cache_owner_fault: normalized.cache_owner_fault,
            })
            .await
            .unwrap();
        let WorkerKvQueryResponse::TreeDump { events, .. } =
            local_indexer.get_events_in_id_range(None, None).await
        else {
            panic!("expected a complete local snapshot")
        };
        let cache_owner_tiers = events
            .iter()
            .filter_map(|event| {
                (event.state_source == Some(cache_owner_id())).then_some(event.storage_tier)
            })
            .collect::<HashSet<_>>();
        assert_eq!(
            cache_owner_tiers,
            HashSet::from([StorageTier::HostPinned, StorageTier::Disk])
        );

        let boundary_wire = (
            2.0f64,
            vec![
                serde_json::json!({
                    "type": "BlockStored",
                    "block_hashes": [108],
                    "parent_block_hash": null,
                    "token_ids": [34, 35, 36, 37],
                    "block_size": 4,
                    "medium": "GPU"
                }),
                serde_json::json!({
                    "type": "AllBlocksCleared"
                }),
                serde_json::json!({
                    "type": "AllBlocksCleared",
                    "ownership": "kvcr"
                }),
            ],
            Some(3i32),
        );
        let payload = rmp_serde::to_vec_named(&boundary_wire).unwrap();
        let batch = dynamo_kv_router::zmq_wire::decode_event_batch(&payload).unwrap();
        let normalized = normalize_raw_batch(
            batch.events,
            worker,
            4,
            KvStateIngressProtocol::VllmResidencyV1,
            &mut framework,
            &mut cache_owner,
        );
        assert_eq!(normalized.source_fault, None);
        assert_eq!(normalized.cache_owner_fault, None);
        assert_eq!(
            normalized
                .events
                .iter()
                .map(|event| event.placement.residency_domain)
                .collect::<Vec<_>>(),
            vec![
                ResidencyDomain::Worker,
                ResidencyDomain::Worker,
                ResidencyDomain::CacheOwner,
            ]
        );
    }

    fn send_ingress(
        tx: &mpsc::UnboundedSender<IngressBatch>,
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
            source_fault: None,
            cache_owner_fault: None,
        })
        .context("state-agent test ingress queue closed")
    }

    #[test]
    fn listener_admission_does_not_wait_for_coordinator_progress() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        for source_cursor in 1..=1_024 {
            admit_ingress(
                &tx,
                IngressBatch {
                    generation: 7,
                    source_cursor,
                    chunk_index: 0,
                    final_chunk: true,
                    events: Vec::new(),
                    source_fault: None,
                    cache_owner_fault: None,
                },
            )
            .unwrap();
        }

        assert_eq!(rx.try_recv().unwrap().source_cursor, 1);
        assert_eq!(rx.len(), 1_023);
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
                response,
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
        let sink_observer = sink.clone();
        let (control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_CAPACITY);
        let (ingress_tx, ingress_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(
            Coordinator {
                publisher: sink,
                local_indexer: local_indexer.clone(),
                identity: identity.clone(),
                slot_dp_rank: 3,
                status: status.clone(),
                withdrawal_tx: mpsc::unbounded_channel().0,
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
            test_attachment(first_worker, "tcp://framework-7"),
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
        .unwrap();
        sink_observer.wait_for_publications(3).await;
        quarantine(&control_tx, first.generation, "test interruption").await;
        assert!(!status.load().attachment.as_ref().unwrap().ready);
        assert_eq!(detach(&control_tx, first.generation).await.unwrap(), 5);

        let replacement = send_attach(
            &control_tx,
            test_attachment(replacement_worker, "tcp://framework-8"),
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
            events[1].residency_owner().unwrap(),
            ResidencyOwner::Worker(first_worker)
        );
        assert_eq!(events[1].state_source, Some(owner));
        assert_eq!(
            events[2].residency_owner().unwrap(),
            ResidencyOwner::CacheOwner(owner)
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

        let (response, received) = oneshot::channel();
        control_tx
            .send(ControlCommand::Shutdown { response })
            .await
            .unwrap();
        received.await.unwrap();
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
        let (_ingress_tx, ingress_rx) = mpsc::unbounded_channel();
        let mut coordinator = Coordinator {
            publisher: sink,
            local_indexer: local_indexer.clone(),
            identity,
            slot_dp_rank: 3,
            status,
            withdrawal_tx: mpsc::unbounded_channel().0,
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
        let mut attachment = test_attachment(worker, "tcp://framework");
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
                source_fault: None,
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

        let mut replacement = test_attachment(worker, "tcp://framework-2");
        coordinator.attach(&mut replacement).await.unwrap();
        coordinator
            .handle_ingress(IngressBatch {
                generation: replacement.generation,
                source_cursor: 1,
                chunk_index: 0,
                final_chunk: false,
                events: vec![store(worker, ResidencyDomain::CacheOwner, None, 301, 31)],
                source_fault: None,
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
            coordinator.status.load().cache_owner_ready,
            "live-stream continuity remains advisory until journal replay is implemented"
        );
    }

    #[tokio::test]
    async fn lossy_sequence_and_publication_failures_do_not_reuse_outbound_ids() {
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
        let (_ingress_tx, ingress_rx) = mpsc::unbounded_channel();
        let mut coordinator = Coordinator {
            publisher: sink,
            local_indexer: local_indexer.clone(),
            identity,
            slot_dp_rank: 3,
            status: status.clone(),
            withdrawal_tx: mpsc::unbounded_channel().0,
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
        let mut attachment = test_attachment(worker, "tcp://framework");
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
                    source_fault: None,
                    cache_owner_fault: None,
                })
                .await
                .is_err()
        );
        assert!(status.load().attachment.as_ref().unwrap().ready);

        fail_publication.store(true, Ordering::Relaxed);
        coordinator
            .handle_ingress(IngressBatch {
                generation: attachment.generation,
                source_cursor: 1,
                chunk_index: 0,
                final_chunk: true,
                events: vec![store(worker, ResidencyDomain::Worker, None, 101, 11)],
                source_fault: None,
                cache_owner_fault: None,
            })
            .await
            .unwrap();
        assert_eq!(local_indexer.current_event_id(), 2);
        assert_eq!(status.load().outbound_cursor, 2);
        assert!(status.load().cache_owner_ready);
        assert!(status.load().attachment.as_ref().unwrap().ready);

        fail_publication.store(false, Ordering::Relaxed);
        coordinator
            .handle_ingress(IngressBatch {
                generation: attachment.generation,
                source_cursor: 3,
                chunk_index: 0,
                final_chunk: true,
                events: vec![store(worker, ResidencyDomain::Worker, Some(101), 102, 12)],
                source_fault: None,
                cache_owner_fault: None,
            })
            .await
            .unwrap();
        assert_eq!(status.load().outbound_cursor, 3);
        assert!(status.load().attachment.as_ref().unwrap().ready);
    }
}
