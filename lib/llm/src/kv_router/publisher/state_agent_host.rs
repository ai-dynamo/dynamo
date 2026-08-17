// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone multi-slot host for persistent KV state agents.

use std::{collections::HashMap, sync::Arc};

use anyhow::{Context, Result};
use arc_swap::ArcSwap;
use async_trait::async_trait;
use dynamo_kv_router::identity::CacheOwnerId;
use dynamo_runtime::{
    component::{Component, Endpoint, StartedEndpoint},
    discovery::{
        DiscoveryEvent, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, EventScope,
        EventSourceQuery,
    },
    metrics::MetricsHierarchy,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
        network::Ingress,
    },
    stream,
    traits::DistributedRuntimeProvider,
};
use futures::StreamExt;
use prometheus::{IntCounter, IntGaugeVec};
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::discovery::kv_state_agent::{
    KV_STATE_ATTACHMENT_INTENT_TOPIC_V2, KV_STATE_HOST_TOPIC_V2, KvStateAttachmentIntent,
    KvStateHostAdvertisement, KvStateHostControlRequest, KvStateHostStatus, KvStateIngressProtocol,
};

use super::state_agent::{
    KvStateAgent, KvStateAgentAttachmentConfig, KvStateAgentConfig, KvStateAgentSlotConfig,
    KvStateAgentVllmSource,
};

pub const DEFAULT_KV_STATE_AGENT_MAX_SLOTS: usize = 8;

#[derive(Clone)]
pub struct KvStateAgentHostConfig {
    pub endpoint: Endpoint,
    pub max_slots: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotLifecycle {
    Active,
    Detached,
    Failed,
}

struct HostedSlot {
    intent_incarnation: Option<u64>,
    producer_instance: Option<dynamo_runtime::component::Instance>,
    lifecycle: SlotLifecycle,
    agent: Option<Arc<KvStateAgent>>,
    global_dp_rank: u32,
    kv_state_endpoint: dynamo_runtime::protocols::EndpointId,
    indexer_domain_id: dynamo_kv_router::identity::IndexerDomainId,
    kv_block_size: u32,
    ingress_protocol: KvStateIngressProtocol,
}

impl HostedSlot {
    fn matches(&self, intent: &KvStateAttachmentIntent) -> bool {
        self.global_dp_rank == intent.worker.dp_rank
            && self.kv_state_endpoint == intent.kv_state_endpoint
            && self.indexer_domain_id == intent.indexer_domain_id
            && self.kv_block_size == intent.kv_block_size
            && self.ingress_protocol == intent.ingress_protocol
    }
}

struct HostMetrics {
    slots: IntGaugeVec,
    capacity_rejected: IntCounter,
}

impl HostMetrics {
    fn new(component: &Component) -> Result<Self> {
        let metrics = component.metrics();
        let slots = metrics
            .create_intgaugevec(
                "kv_state_agent_slots",
                "KV state-agent slots by lifecycle state",
                &["state"],
                &[],
            )
            .context("failed to register KV state-agent slot metrics")?;
        let capacity_rejected = metrics
            .create_intcounter(
                "kv_state_agent_capacity_rejected_total",
                "Attachment intents rejected because the host slot limit was reached",
                &[],
            )
            .context("failed to register KV state-agent capacity metric")?;
        Ok(Self {
            slots,
            capacity_rejected,
        })
    }

    fn publish(&self, status: &KvStateHostStatus) {
        self.slots
            .with_label_values(&["total"])
            .set(status.total_slots as i64);
        self.slots
            .with_label_values(&["active"])
            .set(status.active_slots as i64);
        self.slots
            .with_label_values(&["detached"])
            .set(status.detached_slots as i64);
        self.slots
            .with_label_values(&["failed"])
            .set(status.failed_slots as i64);
    }
}

struct HostControlEngine {
    status: Arc<ArcSwap<KvStateHostStatus>>,
    slots: Arc<Mutex<HashMap<CacheOwnerId, HostedSlot>>>,
}

#[async_trait]
impl AsyncEngine<SingleIn<KvStateHostControlRequest>, ManyOut<KvStateHostStatus>, anyhow::Error>
    for HostControlEngine
{
    async fn generate(
        &self,
        request: SingleIn<KvStateHostControlRequest>,
    ) -> Result<ManyOut<KvStateHostStatus>> {
        let (request, context) = request.into_parts();
        match request {
            KvStateHostControlRequest::Status => {}
            KvStateHostControlRequest::SetCacheReadable {
                cache_owner_id,
                producer_instance,
                intent_incarnation,
                readable,
            } => {
                // NOTE: The Dynamo namespace/DRT control plane is a trusted
                // component boundary. These identities fence the current
                // attachment; they are not authentication secrets. A future
                // untrusted control plane requires transport authentication,
                // not a capability copied through discovery.
                let (agent, generation) = {
                    let slots = self.slots.lock().await;
                    let slot = slots
                        .get(&cache_owner_id)
                        .context("KV state-agent slot is not present")?;
                    if slot.lifecycle != SlotLifecycle::Active
                        || slot.intent_incarnation != Some(intent_incarnation)
                        || slot.producer_instance.as_ref() != Some(producer_instance.as_ref())
                    {
                        anyhow::bail!(
                            "readability update does not match the active attachment intent"
                        );
                    }
                    let agent = slot
                        .agent
                        .clone()
                        .context("active slot has no state agent")?;
                    let generation = agent
                        .status()
                        .attachment
                        .as_ref()
                        .context("active slot has no engine attachment")?
                        .generation;
                    (agent, generation)
                };
                agent
                    .set_cache_readable(generation, readable)
                    .await
                    .context("failed to update CacheOwner readability")?;
            }
        }
        Ok(ResponseStream::new(
            Box::pin(stream::iter(vec![(*self.status.load_full()).clone()])),
            context.context(),
        ))
    }
}

/// One process-level host sharing a DRT and Tokio runtime across stable slots.
pub struct KvStateAgentHost {
    component: Component,
    status: Arc<ArcSwap<KvStateHostStatus>>,
    cancel: CancellationToken,
    supervisor: Mutex<Option<tokio::task::JoinHandle<()>>>,
    control_endpoint: Mutex<Option<StartedEndpoint>>,
    host_advertisement: Arc<Mutex<Option<DiscoveryInstance>>>,
    terminated: CancellationToken,
}

impl KvStateAgentHost {
    pub async fn start(config: KvStateAgentHostConfig) -> Result<Arc<Self>> {
        if config.max_slots == 0 {
            anyhow::bail!("max_slots must be greater than zero");
        }

        let component = config.endpoint.component().clone();
        let slots = Arc::new(Mutex::new(HashMap::new()));
        let status = Arc::new(ArcSwap::from_pointee(KvStateHostStatus {
            healthy: true,
            ..Default::default()
        }));
        let control = config
            .endpoint
            .endpoint_builder()
            .handler(Ingress::for_engine(Arc::new(HostControlEngine {
                status: status.clone(),
                slots: slots.clone(),
            }))?)
            .graceful_shutdown(true)
            .start_with_registration()
            .await
            .context("failed to start KV state-agent host control endpoint")?;
        let control_target = control.instance().clone();
        let discovery_id = host_discovery_id(component.drt().connection_id());
        let host_ad = KvStateHostAdvertisement {
            protocol_version: dynamo_kv_router::indexer::KvStateProtocolVersion::V2,
            host_instance: control_target.clone(),
            control_target: control_target.clone(),
            max_slots: config.max_slots,
        };
        let host_advertisement = match component
            .drt()
            .discovery()
            .register(DiscoverySpec::EventSource {
                scope: component_scope(&component),
                topic: KV_STATE_HOST_TOPIC_V2.to_string(),
                publisher_id: discovery_id,
                metadata: serde_json::to_value(host_ad)?,
            })
            .await
        {
            Ok(instance) => instance,
            Err(error) => {
                let _ = control.shutdown().await;
                return Err(error).context("failed to advertise KV state-agent host");
            }
        };

        let watch_cancel = CancellationToken::new();
        let stream = match component
            .drt()
            .discovery()
            .list_and_watch(
                DiscoveryQuery::EventSources(EventSourceQuery::topic(
                    component.namespace().name(),
                    component.name(),
                    KV_STATE_ATTACHMENT_INTENT_TOPIC_V2,
                )),
                Some(watch_cancel.clone()),
            )
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                let _ = component
                    .drt()
                    .discovery()
                    .unregister(host_advertisement)
                    .await;
                let _ = control.shutdown().await;
                return Err(error).context("failed to watch KV state-agent attachment intents");
            }
        };

        let metrics = match HostMetrics::new(&component) {
            Ok(metrics) => metrics,
            Err(error) => {
                let _ = component
                    .drt()
                    .discovery()
                    .unregister(host_advertisement)
                    .await;
                let _ = control.shutdown().await;
                return Err(error);
            }
        };
        let host_advertisement = Arc::new(Mutex::new(Some(host_advertisement)));
        let cancel = CancellationToken::new();
        let terminated = CancellationToken::new();
        let supervisor_cancel = cancel.clone();
        let supervisor_component = component.clone();
        let supervisor_status = status.clone();
        let supervisor_slots = slots.clone();
        let supervisor_host_advertisement = host_advertisement.clone();
        let supervisor_terminated = terminated.clone();
        let supervisor = component.drt().runtime().secondary().spawn(async move {
            run_host_supervisor(
                supervisor_component,
                control_target,
                config.max_slots,
                stream,
                supervisor_slots,
                supervisor_status,
                metrics,
                supervisor_cancel,
                supervisor_host_advertisement,
            )
            .await;
            watch_cancel.cancel();
            supervisor_terminated.cancel();
        });

        Ok(Arc::new(Self {
            component,
            status,
            cancel,
            supervisor: Mutex::new(Some(supervisor)),
            control_endpoint: Mutex::new(Some(control)),
            host_advertisement,
            terminated,
        }))
    }

    pub fn status(&self) -> Arc<KvStateHostStatus> {
        self.status.load_full()
    }

    pub async fn wait_terminated(&self) {
        self.terminated.cancelled().await;
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.cancel.cancel();
        if let Some(supervisor) = self.supervisor.lock().await.take() {
            let _ = supervisor.await;
        }
        let unregister_result = match self.host_advertisement.lock().await.take() {
            Some(advertisement) => self
                .component
                .drt()
                .discovery()
                .unregister(advertisement)
                .await
                .context("failed to remove KV state-agent host advertisement"),
            None => Ok(()),
        };
        let endpoint_result = match self.control_endpoint.lock().await.take() {
            Some(endpoint) => endpoint.shutdown().await,
            None => Ok(()),
        };
        unregister_result?;
        endpoint_result?;
        Ok(())
    }
}

impl Drop for KvStateAgentHost {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_host_supervisor(
    component: Component,
    host_instance: dynamo_runtime::component::Instance,
    max_slots: usize,
    mut stream: dynamo_runtime::discovery::DiscoveryStream,
    slots: Arc<Mutex<HashMap<CacheOwnerId, HostedSlot>>>,
    status: Arc<ArcSwap<KvStateHostStatus>>,
    metrics: HostMetrics,
    cancel: CancellationToken,
    host_advertisement: Arc<Mutex<Option<DiscoveryInstance>>>,
) {
    let mut intents: HashMap<CacheOwnerId, HashMap<u64, KvStateAttachmentIntent>> = HashMap::new();
    let mut intent_owners: HashMap<u64, CacheOwnerId> = HashMap::new();
    let mut capacity_rejected_total = 0u64;
    let mut watch_ended = false;

    loop {
        let event = tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            event = stream.next() => event,
        };
        let Some(event) = event else {
            tracing::error!("KV state-agent attachment-intent watch ended");
            watch_ended = true;
            break;
        };
        let owner =
            match reconcile_intent_event(event, &host_instance, &mut intents, &mut intent_owners) {
                Ok(owner) => owner,
                Err(error) => {
                    tracing::warn!(%error, "Ignoring invalid KV state-agent attachment intent");
                    continue;
                }
            };
        let Some(owner) = owner else {
            continue;
        };
        let mut slots_guard = slots.lock().await;
        if let Err(error) = reconcile_owner(
            &component,
            owner,
            &intents,
            &mut slots_guard,
            max_slots,
            &mut capacity_rejected_total,
            &metrics,
        )
        .await
        {
            tracing::error!(%owner, %error, "Failed to reconcile KV state-agent slot");
        }
        publish_host_status(
            &slots_guard,
            capacity_rejected_total,
            true,
            &status,
            &metrics,
        );
    }

    let mut slots = slots.lock().await;
    publish_host_status(&slots, capacity_rejected_total, false, &status, &metrics);
    for slot in slots.values_mut() {
        if let Some(agent) = slot.agent.take()
            && let Err(error) = agent.shutdown().await
        {
            tracing::warn!(%error, "Failed to stop hosted KV state agent");
        }
    }
    drop(slots);
    if watch_ended {
        let current = host_advertisement.lock().await.clone();
        if let Some(current) = current {
            if let Err(error) = component
                .drt()
                .discovery()
                .unregister(current.clone())
                .await
            {
                tracing::error!(%error, "Failed to withdraw unhealthy KV state-agent host advertisement");
            } else {
                let mut advertisement = host_advertisement.lock().await;
                if advertisement.as_ref() == Some(&current) {
                    *advertisement = None;
                }
            }
        }
    }
}

fn reconcile_intent_event(
    event: Result<DiscoveryEvent>,
    host_instance: &dynamo_runtime::component::Instance,
    intents: &mut HashMap<CacheOwnerId, HashMap<u64, KvStateAttachmentIntent>>,
    intent_owners: &mut HashMap<u64, CacheOwnerId>,
) -> Result<Option<CacheOwnerId>> {
    let event = event?;
    let (scope, topic, publisher_id, metadata) = match event {
        DiscoveryEvent::Added(DiscoveryInstance::EventSource {
            scope,
            topic,
            publisher_id,
            metadata,
        }) => (scope, topic, publisher_id, metadata),
        DiscoveryEvent::Removed(dynamo_runtime::discovery::DiscoveryInstanceId::EventSource(
            id,
        )) => {
            if id.scope
                != (EventScope::Component {
                    namespace: host_instance.namespace.clone(),
                    component: host_instance.component.clone(),
                })
                || id.topic != KV_STATE_ATTACHMENT_INTENT_TOPIC_V2
            {
                return Ok(None);
            }
            let Some(owner) = intent_owners.remove(&id.publisher_id) else {
                return Ok(None);
            };
            if let Some(owner_intents) = intents.get_mut(&owner) {
                owner_intents.remove(&id.publisher_id);
                if owner_intents.is_empty() {
                    intents.remove(&owner);
                }
            }
            return Ok(Some(owner));
        }
        DiscoveryEvent::Added(_)
        | DiscoveryEvent::Removed(_)
        | DiscoveryEvent::ModelTaintsUpdated(_) => return Ok(None),
    };
    if scope
        != (EventScope::Component {
            namespace: host_instance.namespace.clone(),
            component: host_instance.component.clone(),
        })
        || topic != KV_STATE_ATTACHMENT_INTENT_TOPIC_V2
    {
        anyhow::bail!("attachment intent has an unexpected scope or topic");
    }
    let intent: KvStateAttachmentIntent = serde_json::from_value(metadata)?;
    validate_intent(&intent, host_instance, publisher_id)?;
    let owner = intent.cache_owner_id;
    if let Some(previous_owner) = intent_owners.get(&publisher_id)
        && *previous_owner != owner
    {
        anyhow::bail!("intent discovery identity changed its CacheOwnerId");
    }
    let owner_intents = intents.entry(owner).or_default();
    if owner_intents
        .get(&publisher_id)
        .is_some_and(|previous| previous != &intent)
    {
        anyhow::bail!("intent incarnation changed its immutable descriptor");
    }
    owner_intents.insert(publisher_id, intent);
    intent_owners.insert(publisher_id, owner);
    Ok(Some(owner))
}

fn validate_intent(
    intent: &KvStateAttachmentIntent,
    host_instance: &dynamo_runtime::component::Instance,
    publisher_id: u64,
) -> Result<()> {
    if &intent.target_host != host_instance {
        anyhow::bail!("attachment intent targets another host incarnation");
    }
    if intent.intent_incarnation != publisher_id {
        anyhow::bail!("intent incarnation does not match its discovery identity");
    }
    if intent.kv_block_size == 0 {
        anyhow::bail!("attachment intent has zero KV block size");
    }
    if intent.cache_owner_id.pool().indexer_domain() != intent.indexer_domain_id {
        anyhow::bail!("attachment intent indexer domain disagrees with CacheOwnerId");
    }
    if intent.ingress_protocol != KvStateIngressProtocol::VllmResidencyV1 {
        anyhow::bail!("unsupported state-agent raw ingress protocol");
    }
    if intent.raw_zmq_endpoint.is_empty() {
        anyhow::bail!("attachment intent must supply a resolved raw endpoint");
    }
    if !intent.raw_zmq_endpoint.starts_with("tcp://") {
        anyhow::bail!("attachment intent raw endpoint must use the tcp:// scheme");
    }
    Ok(())
}

fn slot_endpoint_name(owner: CacheOwnerId) -> String {
    let pool = owner.pool();
    let domain = pool.indexer_domain();
    format!(
        "slot_{}_{}_{}_{}",
        domain.cache_semantics(),
        domain.routing_scope(),
        pool.dc_id(),
        owner.slot()
    )
}

#[allow(clippy::too_many_arguments)]
async fn reconcile_owner(
    component: &Component,
    owner: CacheOwnerId,
    intents: &HashMap<CacheOwnerId, HashMap<u64, KvStateAttachmentIntent>>,
    slots: &mut HashMap<CacheOwnerId, HostedSlot>,
    max_slots: usize,
    capacity_rejected_total: &mut u64,
    metrics: &HostMetrics,
) -> Result<()> {
    let matching = intents.get(&owner);
    let intent = match matching.map(HashMap::len).unwrap_or(0) {
        0 => {
            if let Some(slot) = slots.get_mut(&owner) {
                detach_slot(slot).await?;
                if slot.agent.is_none() {
                    slots.remove(&owner);
                }
            }
            return Ok(());
        }
        1 => matching.and_then(|items| items.values().next()).cloned(),
        _ => {
            if let Some(slot) = slots.get_mut(&owner) {
                detach_slot(slot).await?;
                slot.lifecycle = SlotLifecycle::Failed;
            }
            tracing::warn!(%owner, "Ambiguous attachment intents; slot remains detached");
            return Ok(());
        }
    };
    let intent = intent.expect("one intent was counted");

    if !slots.contains_key(&owner) {
        // NOTE: Detached agents deliberately retain CacheOwner/index/recovery
        // state and count against max_slots. This is a durable-slot resource
        // budget, not an active-attachment count. Failed construction remains
        // reserved only until its matching intent disappears.
        // TODO(#13044): require explicit decommission/migration authority
        // before reclaiming a durable owner from a long-lived host.
        if slots.len() >= max_slots {
            *capacity_rejected_total = capacity_rejected_total.saturating_add(1);
            metrics.capacity_rejected.inc();
            tracing::warn!(%owner, max_slots, "Rejecting KV state-agent intent at capacity");
            return Ok(());
        }
        let slot_shape = HostedSlot {
            intent_incarnation: Some(intent.intent_incarnation),
            producer_instance: Some(intent.producer_instance.clone()),
            lifecycle: SlotLifecycle::Failed,
            agent: None,
            global_dp_rank: intent.worker.dp_rank,
            kv_state_endpoint: intent.kv_state_endpoint.clone(),
            indexer_domain_id: intent.indexer_domain_id,
            kv_block_size: intent.kv_block_size,
            ingress_protocol: intent.ingress_protocol,
        };
        slots.insert(owner, slot_shape);
        let agent = match KvStateAgent::start(KvStateAgentConfig {
            endpoint: component.endpoint(slot_endpoint_name(owner)),
            kv_state_endpoint: intent.kv_state_endpoint.clone(),
            slot: KvStateAgentSlotConfig {
                cache_owner_id: owner,
                global_dp_rank: intent.worker.dp_rank,
            },
            kv_block_size: intent.kv_block_size,
            ingress_protocol: intent.ingress_protocol,
        })
        .await
        {
            Ok(agent) => Arc::new(agent),
            Err(error) => {
                // Keep the failed reservation until this exact intent disappears.
                // It remains capacity-accounted and avoids implicit construction
                // retries under an unchanged lease-owned request.
                return Err(error).context("failed to construct stable state-agent slot");
            }
        };
        slots.get_mut(&owner).expect("inserted above").agent = Some(agent);
    }

    let slot = slots.get_mut(&owner).expect("slot exists");
    if !slot.matches(&intent) {
        slot.lifecycle = SlotLifecycle::Failed;
        anyhow::bail!("attachment intent conflicts with immutable stable slot properties");
    }
    if slot.intent_incarnation == Some(intent.intent_incarnation)
        && slot.lifecycle == SlotLifecycle::Active
    {
        return Ok(());
    }
    detach_slot(slot).await?;
    let agent = slot
        .agent
        .as_ref()
        .context("failed slot has no state agent")?;
    agent
        .attach(KvStateAgentAttachmentConfig {
            producer_instance: intent.producer_instance.clone(),
            intent_incarnation: intent.intent_incarnation,
            worker: intent.worker,
            vllm_source: KvStateAgentVllmSource {
                endpoint: intent.raw_zmq_endpoint.clone(),
                topic: intent.raw_topic.clone(),
                image_token_id: intent.image_token_id,
                ingress_protocol: intent.ingress_protocol,
            },
        })
        .await
        .context("failed to attach raw KV source")?;
    slot.intent_incarnation = Some(intent.intent_incarnation);
    slot.producer_instance = Some(intent.producer_instance);
    slot.lifecycle = SlotLifecycle::Active;
    Ok(())
}

async fn detach_slot(slot: &mut HostedSlot) -> Result<()> {
    slot.intent_incarnation = None;
    slot.producer_instance = None;
    slot.lifecycle = SlotLifecycle::Failed;
    let Some(agent) = slot.agent.as_ref() else {
        return Ok(());
    };
    if let Some(attachment) = agent.status().attachment.as_ref() {
        agent.detach(attachment.generation).await?;
    }
    slot.lifecycle = SlotLifecycle::Detached;
    Ok(())
}

fn publish_host_status(
    slots: &HashMap<CacheOwnerId, HostedSlot>,
    capacity_rejected_total: u64,
    healthy: bool,
    destination: &Arc<ArcSwap<KvStateHostStatus>>,
    metrics: &HostMetrics,
) {
    let status = KvStateHostStatus {
        healthy,
        total_slots: slots.len(),
        active_slots: slots
            .values()
            .filter(|slot| slot.lifecycle == SlotLifecycle::Active)
            .count(),
        detached_slots: slots
            .values()
            .filter(|slot| slot.lifecycle == SlotLifecycle::Detached)
            .count(),
        failed_slots: slots
            .values()
            .filter(|slot| slot.lifecycle == SlotLifecycle::Failed)
            .count(),
        capacity_rejected_total,
        error: (!healthy).then(|| "KV state-agent host supervisor is not running".to_string()),
    };
    metrics.publish(&status);
    destination.store(Arc::new(status));
}

fn component_scope(component: &Component) -> EventScope {
    EventScope::Component {
        namespace: component.namespace().name(),
        component: component.name().to_string(),
    }
}

fn host_discovery_id(instance_id: u64) -> u64 {
    const JSON_SAFE_MASK: u64 = (1u64 << 53) - 1;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"dynamo/kv-state-host/v2");
    hasher.update(&instance_id.to_be_bytes());
    let value = u64::from_be_bytes(
        hasher.finalize().as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 digest has eight prefix bytes"),
    );
    (value & JSON_SAFE_MASK).max(1)
}

#[cfg(test)]
mod tests {
    use dynamo_kv_router::identity::{
        CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId,
        RoutingScopeId, StableDpSlotId,
    };
    use dynamo_kv_router::protocols::WorkerWithDpRank;
    use dynamo_runtime::component::{Instance, TransportType};
    use dynamo_runtime::discovery::DiscoveryInstanceId;
    use dynamo_runtime::protocols::EndpointId;

    use super::*;

    fn instance(component: &str, instance_id: u64) -> Instance {
        Instance {
            namespace: "ns".to_string(),
            component: component.to_string(),
            endpoint: "control".to_string(),
            instance_id,
            transport: TransportType::Tcp(format!("tcp://127.0.0.1:{}", 10_000 + instance_id)),
            device_type: None,
            request_plane_codec: None,
        }
    }

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

    fn intent(
        host: &Instance,
        producer: Instance,
        owner: CacheOwnerId,
        worker: WorkerWithDpRank,
        incarnation: u64,
    ) -> KvStateAttachmentIntent {
        KvStateAttachmentIntent {
            target_host: host.clone(),
            producer_instance: producer,
            intent_incarnation: incarnation,
            cache_owner_id: owner,
            worker,
            kv_state_endpoint: EndpointId::from("ns.backend.generate"),
            indexer_domain_id: owner.pool().indexer_domain(),
            kv_block_size: 64,
            ingress_protocol: KvStateIngressProtocol::VllmResidencyV1,
            raw_zmq_endpoint: format!("tcp://127.0.0.1:{}", 20_000 + worker.dp_rank),
            raw_topic: "kv-events-residency-v1".to_string(),
            image_token_id: None,
        }
    }

    fn added(
        host: &Instance,
        publisher_id: u64,
        intent: &KvStateAttachmentIntent,
    ) -> DiscoveryEvent {
        DiscoveryEvent::Added(DiscoveryInstance::EventSource {
            scope: EventScope::Component {
                namespace: host.namespace.clone(),
                component: host.component.clone(),
            },
            topic: KV_STATE_ATTACHMENT_INTENT_TOPIC_V2.to_string(),
            publisher_id,
            metadata: serde_json::to_value(intent).unwrap(),
        })
    }

    #[test]
    fn producer_owned_global_rank_slices_do_not_collide() {
        let host = instance("kv_state_agent", 1);
        let leader_worker_id = 17;
        let first = intent(
            &host,
            instance("sglang_node_a", 11),
            owner(4),
            WorkerWithDpRank::new(leader_worker_id, 4),
            101,
        );
        let second = intent(
            &host,
            instance("sglang_node_b", 12),
            owner(7),
            WorkerWithDpRank::new(leader_worker_id, 7),
            202,
        );
        let mut intents = HashMap::new();
        let mut intent_owners = HashMap::new();
        assert_eq!(
            reconcile_intent_event(
                Ok(added(&host, 101, &first)),
                &host,
                &mut intents,
                &mut intent_owners,
            )
            .unwrap(),
            Some(first.cache_owner_id)
        );
        assert_eq!(
            reconcile_intent_event(
                Ok(added(&host, 202, &second)),
                &host,
                &mut intents,
                &mut intent_owners,
            )
            .unwrap(),
            Some(second.cache_owner_id)
        );
        assert_eq!(intents.len(), 2);

        let removed = DiscoveryEvent::Removed(DiscoveryInstanceId::EventSource(
            dynamo_runtime::discovery::EventSourceInstanceId {
                scope: EventScope::Component {
                    namespace: host.namespace.clone(),
                    component: host.component.clone(),
                },
                topic: KV_STATE_ATTACHMENT_INTENT_TOPIC_V2.to_string(),
                publisher_id: 101,
            },
        ));
        assert_eq!(
            reconcile_intent_event(Ok(removed), &host, &mut intents, &mut intent_owners,).unwrap(),
            Some(first.cache_owner_id)
        );
        assert!(!intents.contains_key(&first.cache_owner_id));
        assert_eq!(
            intents[&second.cache_owner_id][&202].worker,
            WorkerWithDpRank::new(leader_worker_id, 7)
        );
    }

    #[test]
    fn changed_intent_incarnation_does_not_replace_the_accepted_descriptor() {
        let host = instance("kv_state_agent", 1);
        let accepted = intent(
            &host,
            instance("vllm", 11),
            owner(4),
            WorkerWithDpRank::new(17, 4),
            101,
        );
        let mut changed = accepted.clone();
        changed.raw_topic = "changed-topic".to_string();
        let mut intents = HashMap::new();
        let mut intent_owners = HashMap::new();

        reconcile_intent_event(
            Ok(added(&host, 101, &accepted)),
            &host,
            &mut intents,
            &mut intent_owners,
        )
        .unwrap();
        assert!(
            reconcile_intent_event(
                Ok(added(&host, 101, &changed)),
                &host,
                &mut intents,
                &mut intent_owners,
            )
            .is_err()
        );
        assert_eq!(intents[&accepted.cache_owner_id][&101], accepted);
        assert_eq!(intent_owners[&101], accepted.cache_owner_id);

        let moved = intent(
            &host,
            instance("vllm", 11),
            owner(5),
            WorkerWithDpRank::new(17, 4),
            101,
        );
        assert!(
            reconcile_intent_event(
                Ok(added(&host, 101, &moved)),
                &host,
                &mut intents,
                &mut intent_owners,
            )
            .is_err()
        );
        assert!(!intents.contains_key(&moved.cache_owner_id));
        assert_eq!(intent_owners[&101], accepted.cache_owner_id);
    }

    #[test]
    fn intent_rejects_non_tcp_raw_endpoint() {
        let host = instance("kv_state_agent", 1);
        let mut invalid = intent(
            &host,
            instance("vllm", 11),
            owner(4),
            WorkerWithDpRank::new(17, 4),
            101,
        );
        invalid.raw_zmq_endpoint = "ipc:///tmp/kv-events".to_string();

        let error = validate_intent(&invalid, &host, 101).unwrap_err();
        assert!(error.to_string().contains("tcp://"));
    }

    #[test]
    fn slot_endpoint_name_includes_pool_identity() {
        let first = owner(4);
        let second = CacheOwnerId::new(
            PoolId::new(first.pool().indexer_domain(), DcId::new(4)),
            first.slot(),
        );

        assert_ne!(slot_endpoint_name(first), slot_endpoint_name(second));
    }
}
