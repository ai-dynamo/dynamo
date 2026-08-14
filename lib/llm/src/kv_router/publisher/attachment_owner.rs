// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lease-owned attachment intents for an external KV state-agent host.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result};
use dynamo_kv_router::{
    identity::{CacheOwnerId, IndexerDomainId},
    indexer::KvStateProtocolVersion,
    protocols::{WorkerId, WorkerWithDpRank},
};
use dynamo_runtime::{
    component::{Endpoint, Instance, build_transport_type},
    discovery::{
        Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, EventScope, EventSourceQuery,
    },
    pipeline::{AddressedPushRouter, AddressedRequest, AsyncEngine, ManyOut, SingleIn},
    protocols::{EndpointId, maybe_error::MaybeError},
    traits::DistributedRuntimeProvider,
};
use futures::StreamExt;
use rand::TryRngCore;
use tokio::{
    sync::{Mutex, oneshot},
    time::Duration,
};
use tokio_util::sync::CancellationToken;

use crate::discovery::kv_state_agent::{
    KV_STATE_ATTACHMENT_INTENT_TOPIC_V2, KV_STATE_HOST_TOPIC_V2, KvStateAttachmentIntent,
    KvStateHostAdvertisement, KvStateHostControlRequest, KvStateHostStatus, KvStateIngressProtocol,
};

const HOST_COMPONENT: &str = "kv_state_agent";
const JSON_SAFE_MASK: u64 = (1u64 << 53) - 1;

#[derive(Debug, Clone)]
pub struct KvStateAttachmentDescriptor {
    pub cache_owner_id: CacheOwnerId,
    pub worker: WorkerWithDpRank,
    pub kv_state_endpoint: EndpointId,
    pub indexer_domain_id: IndexerDomainId,
    pub kv_block_size: u32,
    pub ingress_protocol: KvStateIngressProtocol,
    pub raw_zmq_endpoint: String,
    pub raw_topic: String,
    pub image_token_id: Option<u32>,
}

/// Process-scoped owner of every state-agent intent emitted by one backend worker.
pub struct KvStateAttachmentOwner {
    component: dynamo_runtime::component::Component,
    producer_instance: Instance,
    host: KvStateHostAdvertisement,
    intents: HashMap<u32, KvStateAttachmentIntent>,
    registrations: Arc<Mutex<Vec<DiscoveryInstance>>>,
    addressed: Arc<AddressedPushRouter>,
    closed: AtomicBool,
}

impl KvStateAttachmentOwner {
    pub async fn start(
        endpoint: Endpoint,
        worker_id: WorkerId,
        descriptors: Vec<KvStateAttachmentDescriptor>,
    ) -> Result<Arc<Self>> {
        validate_descriptors(&descriptors)?;
        let component = endpoint.component().clone();
        let host = discover_single_host(&component).await?;
        let producer_instance = producer_instance(&endpoint).await?;
        let addressed = AddressedPushRouter::from_runtime_provider(&component).await?;

        let intents = materialize_intents(&host, &producer_instance, worker_id, descriptors)?;

        let owner = Arc::new(Self {
            component,
            producer_instance,
            host,
            intents,
            registrations: Arc::new(Mutex::new(Vec::new())),
            addressed,
            closed: AtomicBool::new(false),
        });
        register_transaction(owner.clone()).await?;
        Ok(owner)
    }

    pub fn managed_ranks(&self) -> impl Iterator<Item = u32> + '_ {
        self.intents.keys().copied()
    }

    pub async fn set_cache_readable(&self, global_dp_rank: u32, readable: bool) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            anyhow::bail!("KV state attachment owner is closed");
        }
        let intent = self
            .intents
            .get(&global_dp_rank)
            .context("global DP rank is not managed by this attachment owner")?;
        let request = KvStateHostControlRequest::SetCacheReadable {
            cache_owner_id: intent.cache_owner_id,
            producer_instance: Box::new(self.producer_instance.clone()),
            intent_incarnation: intent.intent_incarnation,
            readable,
        };
        self.query_host(request).await?;
        Ok(())
    }

    pub async fn close(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        let registrations = std::mem::take(&mut *self.registrations.lock().await);
        let discovery = self.component.drt().discovery();
        let lease = self.component.drt().primary_token();
        let (finished, result) = oneshot::channel();
        self.component
            .drt()
            .runtime()
            .secondary()
            .spawn(async move {
                cleanup_registrations_required(discovery, registrations, lease).await;
                let _ = finished.send(());
            });
        result
            .await
            .context("state-agent intent cleanup task terminated")
    }

    async fn query_host(&self, request: KvStateHostControlRequest) -> Result<KvStateHostStatus> {
        let addressed = SingleIn::new(request).map(|request| {
            AddressedRequest::for_instance(request, self.host.control_target.clone())
        });
        let mut responses: ManyOut<KvStateHostStatus> = self
            .addressed
            .generate(addressed)
            .await
            .context("failed to call KV state-agent host control endpoint")?;
        let status = responses
            .next()
            .await
            .context("KV state-agent host returned no control response")?;
        if let Some(error) = status.err() {
            return Err(error).context("KV state-agent host rejected control request");
        }
        Ok(status)
    }
}

fn materialize_intents(
    host: &KvStateHostAdvertisement,
    producer_instance: &Instance,
    worker_id: WorkerId,
    descriptors: Vec<KvStateAttachmentDescriptor>,
) -> Result<HashMap<u32, KvStateAttachmentIntent>> {
    let mut intents = HashMap::with_capacity(descriptors.len());
    for descriptor in descriptors {
        let intent_incarnation = random_discovery_id()?;
        let worker = WorkerWithDpRank {
            worker_id,
            dp_rank: descriptor.worker.dp_rank,
        };
        if worker != descriptor.worker {
            anyhow::bail!("descriptor WorkerId disagrees with the process worker identity");
        }
        intents.insert(
            worker.dp_rank,
            KvStateAttachmentIntent {
                target_host: host.host_instance.clone(),
                producer_instance: producer_instance.clone(),
                intent_incarnation,
                cache_owner_id: descriptor.cache_owner_id,
                worker,
                kv_state_endpoint: descriptor.kv_state_endpoint,
                indexer_domain_id: descriptor.indexer_domain_id,
                kv_block_size: descriptor.kv_block_size,
                ingress_protocol: descriptor.ingress_protocol,
                raw_zmq_endpoint: descriptor.raw_zmq_endpoint,
                raw_topic: descriptor.raw_topic,
                image_token_id: descriptor.image_token_id,
            },
        );
    }
    Ok(intents)
}

fn validate_descriptors(descriptors: &[KvStateAttachmentDescriptor]) -> Result<()> {
    if descriptors.is_empty() {
        anyhow::bail!("state-agent mode requires at least one global DP rank");
    }
    let mut ranks = HashSet::with_capacity(descriptors.len());
    let mut owners = HashSet::with_capacity(descriptors.len());
    for descriptor in descriptors {
        if !ranks.insert(descriptor.worker.dp_rank) {
            anyhow::bail!("duplicate global DP rank in state-agent descriptors");
        }
        if !owners.insert(descriptor.cache_owner_id) {
            anyhow::bail!("CacheOwnerId must be unique per managed global DP rank");
        }
        if descriptor.kv_block_size == 0 {
            anyhow::bail!("state-agent descriptor has zero KV block size");
        }
        if descriptor.cache_owner_id.pool().indexer_domain() != descriptor.indexer_domain_id {
            anyhow::bail!("CacheOwnerId and indexer domain disagree");
        }
        if descriptor.ingress_protocol != KvStateIngressProtocol::VllmResidencyV1 {
            anyhow::bail!("unsupported state-agent raw ingress protocol");
        }
        if descriptor.raw_zmq_endpoint.is_empty() {
            anyhow::bail!("state-agent descriptor has an empty raw endpoint");
        }
    }
    Ok(())
}

async fn discover_single_host(
    component: &dynamo_runtime::component::Component,
) -> Result<KvStateHostAdvertisement> {
    let instances = component
        .drt()
        .discovery()
        .list(DiscoveryQuery::EventSources(EventSourceQuery::topic(
            component.namespace().name(),
            HOST_COMPONENT,
            KV_STATE_HOST_TOPIC_V2,
        )))
        .await
        .context("failed to discover KV state-agent host")?;
    let mut hosts = Vec::new();
    for instance in instances {
        let DiscoveryInstance::EventSource {
            topic, metadata, ..
        } = instance
        else {
            continue;
        };
        if topic != KV_STATE_HOST_TOPIC_V2 {
            continue;
        }
        let host: KvStateHostAdvertisement = serde_json::from_value(metadata)?;
        if host.protocol_version == KvStateProtocolVersion::V2
            && host.host_instance == host.control_target
        {
            hosts.push(host);
        }
    }
    let [host] = hosts.as_slice() else {
        anyhow::bail!(
            "state-agent mode requires exactly one live V2 host, found {}",
            hosts.len()
        );
    };
    Ok(host.clone())
}

async fn producer_instance(endpoint: &Endpoint) -> Result<Instance> {
    let endpoint_id = endpoint.id();
    let connection_id = endpoint.drt().connection_id();
    Ok(Instance {
        namespace: endpoint_id.namespace,
        component: endpoint_id.component,
        endpoint: endpoint_id.name,
        instance_id: connection_id,
        transport: build_transport_type(endpoint, &endpoint.id(), connection_id).await?,
        device_type: None,
        request_plane_codec: None,
    })
}

async fn register_transaction(owner: Arc<KvStateAttachmentOwner>) -> Result<()> {
    let (finished, result) = oneshot::channel();
    let discovery = owner.component.drt().discovery();
    let host = owner.host.clone();
    let intents = owner.intents.values().cloned().collect();
    let registrations = owner.registrations.clone();
    let lease = owner.component.drt().primary_token();
    owner
        .component
        .drt()
        .runtime()
        .secondary()
        .spawn(async move {
            run_registration_transaction(discovery, host, intents, registrations, lease, finished)
                .await;
        });
    result
        .await
        .context("state-agent intent transaction task terminated")?
}

async fn run_registration_transaction(
    discovery: Arc<dyn Discovery>,
    host: KvStateHostAdvertisement,
    mut intents: Vec<KvStateAttachmentIntent>,
    registrations: Arc<Mutex<Vec<DiscoveryInstance>>>,
    lease: CancellationToken,
    finished: oneshot::Sender<Result<()>>,
) {
    let outcome = register_all(
        discovery.clone(),
        &host,
        &mut intents,
        registrations.clone(),
        lease.clone(),
    )
    .await;
    if finished.send(outcome).is_err() {
        let registrations = std::mem::take(&mut *registrations.lock().await);
        cleanup_registrations_required(discovery, registrations, lease).await;
    }
}

async fn register_all(
    discovery: Arc<dyn Discovery>,
    host: &KvStateHostAdvertisement,
    intents: &mut [KvStateAttachmentIntent],
    registrations: Arc<Mutex<Vec<DiscoveryInstance>>>,
    lease: CancellationToken,
) -> Result<()> {
    intents.sort_unstable_by_key(|intent| intent.worker.dp_rank);
    for intent in intents.iter().cloned() {
        let expected = DiscoveryInstance::EventSource {
            scope: EventScope::Component {
                namespace: host.host_instance.namespace.clone(),
                component: host.host_instance.component.clone(),
            },
            topic: KV_STATE_ATTACHMENT_INTENT_TOPIC_V2.to_string(),
            publisher_id: intent.intent_incarnation,
            metadata: serde_json::to_value(&intent)?,
        };
        let registered = discovery
            .register(DiscoverySpec::EventSource {
                scope: match &expected {
                    DiscoveryInstance::EventSource { scope, .. } => scope.clone(),
                    _ => unreachable!(),
                },
                topic: KV_STATE_ATTACHMENT_INTENT_TOPIC_V2.to_string(),
                publisher_id: intent.intent_incarnation,
                metadata: serde_json::to_value(intent)?,
            })
            .await;
        match registered {
            Ok(registered) => registrations.lock().await.push(registered),
            Err(error) => {
                let mut rollback = std::mem::take(&mut *registrations.lock().await);
                rollback.push(expected);
                cleanup_registrations_required(discovery.clone(), rollback, lease.clone()).await;
                return Err(error).context("failed to register state-agent attachment intents");
            }
        }
    }
    Ok(())
}

async fn cleanup_registrations_required(
    discovery: Arc<dyn Discovery>,
    registrations: Vec<DiscoveryInstance>,
    lease: CancellationToken,
) {
    for registration in registrations.into_iter().rev() {
        let mut delay = Duration::from_millis(100);
        loop {
            let result = tokio::select! {
                biased;
                _ = lease.cancelled() => return,
                result = discovery.unregister(registration.clone()) => result,
            };
            match result {
                Ok(()) => break,
                Err(error) => {
                    tracing::warn!(%error, ?delay, "Retrying required attachment-intent withdrawal");
                }
            }
            tokio::select! {
                biased;
                _ = lease.cancelled() => return,
                _ = tokio::time::sleep(delay) => {}
            }
            delay = delay.saturating_mul(2).min(Duration::from_secs(5));
        }
    }
}

fn random_discovery_id() -> Result<u64> {
    let mut rng = rand::rngs::OsRng;
    let value = rng
        .try_next_u64()
        .map_err(|error| anyhow::anyhow!("failed to generate intent identity: {error}"))?;
    Ok((value & JSON_SAFE_MASK).max(1))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, PoolId, RoutingScopeId, StableDpSlotId,
    };
    use dynamo_runtime::{
        component::TransportType,
        discovery::{DiscoveryStream, MockDiscovery, SharedMockRegistry},
    };
    use tokio::sync::Notify;
    use tokio_util::sync::CancellationToken;

    use super::*;

    struct ControlledDiscovery {
        inner: MockDiscovery,
        registration_count: AtomicUsize,
        fail_at: Option<usize>,
        pause_at: Option<usize>,
        registration_paused: Notify,
        continue_registration: Notify,
    }

    impl ControlledDiscovery {
        fn new(fail_at: Option<usize>, pause_at: Option<usize>) -> Arc<Self> {
            Arc::new(Self {
                inner: MockDiscovery::new(Some(1), SharedMockRegistry::new()),
                registration_count: AtomicUsize::new(0),
                fail_at,
                pause_at,
                registration_paused: Notify::new(),
                continue_registration: Notify::new(),
            })
        }
    }

    #[async_trait]
    impl Discovery for ControlledDiscovery {
        fn instance_id(&self) -> u64 {
            self.inner.instance_id()
        }

        async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
            let current = self.registration_count.fetch_add(1, Ordering::AcqRel) + 1;
            if self.pause_at == Some(current) {
                self.registration_paused.notify_one();
                self.continue_registration.notified().await;
            }
            if self.fail_at == Some(current) {
                anyhow::bail!("injected registration failure at {current}");
            }
            self.inner.register_internal(spec).await
        }

        async fn unregister(&self, instance: DiscoveryInstance) -> Result<()> {
            self.inner.unregister(instance).await
        }

        async fn list(&self, query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
            self.inner.list(query).await
        }

        async fn list_and_watch(
            &self,
            query: DiscoveryQuery,
            cancel_token: Option<CancellationToken>,
        ) -> Result<DiscoveryStream> {
            self.inner.list_and_watch(query, cancel_token).await
        }
    }

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

    fn cache_owner(slot: u8) -> CacheOwnerId {
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

    fn host() -> KvStateHostAdvertisement {
        let host = instance("kv_state_agent", 1);
        KvStateHostAdvertisement {
            protocol_version: KvStateProtocolVersion::V2,
            host_instance: host.clone(),
            control_target: host,
            max_slots: 8,
        }
    }

    fn descriptors() -> Vec<KvStateAttachmentDescriptor> {
        [4, 7]
            .into_iter()
            .map(|rank| {
                let cache_owner_id = cache_owner(rank as u8);
                KvStateAttachmentDescriptor {
                    cache_owner_id,
                    worker: WorkerWithDpRank::new(17, rank),
                    kv_state_endpoint: EndpointId::from("ns.backend.generate"),
                    indexer_domain_id: cache_owner_id.pool().indexer_domain(),
                    kv_block_size: 64,
                    ingress_protocol: KvStateIngressProtocol::VllmResidencyV1,
                    raw_zmq_endpoint: format!("tcp://producer.example:{}", 20_000 + rank),
                    raw_topic: String::new(),
                    image_token_id: None,
                }
            })
            .collect()
    }

    fn intent_query() -> DiscoveryQuery {
        DiscoveryQuery::EventSources(EventSourceQuery::topic(
            "ns",
            "kv_state_agent",
            KV_STATE_ATTACHMENT_INTENT_TOPIC_V2,
        ))
    }

    #[test]
    fn generic_owner_preserves_producer_worker_rank_and_resolved_uri() {
        let producer = instance("sglang_node_b", 11);
        let intents = materialize_intents(&host(), &producer, 17, descriptors()).unwrap();
        let selected = &intents[&7];
        assert_eq!(selected.producer_instance, producer);
        assert_eq!(selected.worker, WorkerWithDpRank::new(17, 7));
        assert_eq!(selected.raw_zmq_endpoint, "tcp://producer.example:20007");
        assert_eq!(selected.raw_topic, "");
    }

    #[tokio::test]
    async fn kth_registration_failure_rolls_back_every_intent() {
        let discovery = ControlledDiscovery::new(Some(2), None);
        let registrations = Arc::new(Mutex::new(Vec::new()));
        let mut intents =
            materialize_intents(&host(), &instance("producer", 11), 17, descriptors())
                .unwrap()
                .into_values()
                .collect::<Vec<_>>();
        assert!(
            register_all(
                discovery.clone(),
                &host(),
                &mut intents,
                registrations.clone(),
                CancellationToken::new(),
            )
            .await
            .is_err()
        );
        assert!(registrations.lock().await.is_empty());
        assert!(discovery.list(intent_query()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn cancelled_registration_transaction_rolls_back_every_intent() {
        let discovery = ControlledDiscovery::new(None, Some(2));
        let registrations = Arc::new(Mutex::new(Vec::new()));
        let intents = materialize_intents(&host(), &instance("producer", 11), 17, descriptors())
            .unwrap()
            .into_values()
            .collect();
        let (finished, result) = oneshot::channel();
        let task = tokio::spawn(run_registration_transaction(
            discovery.clone(),
            host(),
            intents,
            registrations.clone(),
            CancellationToken::new(),
            finished,
        ));
        discovery.registration_paused.notified().await;
        drop(result);
        discovery.continue_registration.notify_one();
        task.await.unwrap();

        assert!(registrations.lock().await.is_empty());
        assert!(discovery.list(intent_query()).await.unwrap().is_empty());
    }
}
