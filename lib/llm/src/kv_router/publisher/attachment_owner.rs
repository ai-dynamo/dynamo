// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lease-owned attachment intents for an external KV state-agent host.

use std::{
    collections::{HashMap, HashSet},
    ffi::OsString,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use anyhow::{Context, Result};
use dynamo_kv_router::{
    identity::{CacheOwnerId, IndexerDomainId},
    indexer::KvStateProtocolVersion,
    protocols::{RouterHintSourceMetadata, WorkerId, WorkerWithDpRank},
};
use dynamo_runtime::{
    component::{Endpoint, Instance, build_transport_type},
    config::environment_names::llm::DYN_KV_STATE_AGENT_HOST_DISCOVERY_TIMEOUT_SECS as HOST_DISCOVERY_TIMEOUT_ENV,
    discovery::{
        Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, EventScope, EventSourceQuery,
    },
    protocols::EndpointId,
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
    KvStateHostAdvertisement, KvStateIngressProtocol,
};

const HOST_COMPONENT: &str = "kv_state_agent";
const DEFAULT_HOST_DISCOVERY_TIMEOUT: Duration = Duration::from_secs(30);
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
    pub video_token_id: Option<u32>,
    pub router_hint_source: Option<RouterHintSourceMetadata>,
}

/// Process-scoped owner of every state-agent intent emitted by one backend worker.
pub struct KvStateAttachmentOwner {
    component: dynamo_runtime::component::Component,
    host: KvStateHostAdvertisement,
    intents: HashMap<u32, KvStateAttachmentIntent>,
    registrations: Arc<Mutex<Vec<DiscoveryInstance>>>,
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

        let intents = materialize_intents(&host, &producer_instance, worker_id, descriptors)?;

        let owner = Arc::new(Self {
            component,
            host,
            intents,
            registrations: Arc::new(Mutex::new(Vec::new())),
            closed: AtomicBool::new(false),
        });
        register_transaction(owner.clone()).await?;
        Ok(owner)
    }

    pub fn managed_ranks(&self) -> impl Iterator<Item = u32> + '_ {
        self.intents.keys().copied()
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
                video_token_id: descriptor.video_token_id,
                router_hint_source: descriptor.router_hint_source,
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

/// Resolve the single live V2 state-agent host for this component's namespace.
async fn discover_single_host(
    component: &dynamo_runtime::component::Component,
) -> Result<KvStateHostAdvertisement> {
    let namespace = component.namespace().name();
    let discovery = component.drt().discovery();
    let shutdown = component.drt().primary_token();
    await_single_host(
        discovery.as_ref(),
        &namespace,
        host_discovery_timeout(|key| std::env::var_os(key)),
        &shutdown,
    )
    .await
}

/// Startup wait for host discovery from the env knob; invalid values warn and use the default.
fn host_discovery_timeout(mut get_env: impl FnMut(&str) -> Option<OsString>) -> Duration {
    match get_env(HOST_DISCOVERY_TIMEOUT_ENV) {
        None => DEFAULT_HOST_DISCOVERY_TIMEOUT,
        Some(raw) => match raw.to_string_lossy().trim().parse::<u64>() {
            Ok(secs) => Duration::from_secs(secs),
            Err(_) => {
                tracing::warn!(
                    env = HOST_DISCOVERY_TIMEOUT_ENV,
                    value = %raw.to_string_lossy(),
                    default_secs = DEFAULT_HOST_DISCOVERY_TIMEOUT.as_secs(),
                    "invalid state-agent host discovery timeout; using default"
                );
                DEFAULT_HOST_DISCOVERY_TIMEOUT
            }
        },
    }
}

/// Wait up to `timeout` for exactly one live V2 state-agent host.
///
/// The `list` snapshot stays authoritative: two or more hosts fail closed
/// immediately (misconfiguration, not a race), and the watch stream is only a
/// wake-up to re-list while no host is advertised yet — a watch event is
/// never trusted directly because initial-snapshot events are
/// indistinguishable from later additions. A zero timeout preserves the
/// original single-snapshot behavior; cancelling `shutdown` aborts the wait.
async fn await_single_host(
    discovery: &dyn Discovery,
    namespace: &str,
    timeout: Duration,
    shutdown: &CancellationToken,
) -> Result<KvStateHostAdvertisement> {
    // NOTE: V2 assumes one host per control scope that remains alive for the
    // deployment run. This wait covers only start ordering: source mode is
    // immutable for a worker lifecycle, so a worker that races the host's
    // advertisement would otherwise serve without KV routing until it
    // restarts. TODO(#13044): watch and reselect the host after startup when
    // rolling host replacement and Kubernetes lifecycle are supported.
    if timeout.is_zero() {
        return match list_single_host(discovery, namespace).await? {
            Some(host) => Ok(host),
            None => Err(exactly_one_host_error(0)),
        };
    }
    // The deadline is fixed before any backend I/O: the initial list, watch
    // setup, and every re-list all run under it.
    let expiry = host_discovery_expiry(timeout);
    let watch_cancel = shutdown.child_token();
    // Cancelled on every exit path; without this the discovery watch task
    // outlives the wait for the rest of the process.
    let _watch_guard = watch_cancel.clone().drop_guard();
    let wait = async {
        if let Some(host) = list_single_host(discovery, namespace).await? {
            return Ok(host);
        }
        tracing::info!(
            namespace,
            timeout_secs = timeout.as_secs(),
            "KV state-agent host is not advertised yet; waiting"
        );
        let mut events = discovery
            .list_and_watch(host_query(namespace), Some(watch_cancel.clone()))
            .await
            .context("failed to watch for the KV state-agent host")?;
        while let Some(event) = events.next().await {
            event.context("KV state-agent host discovery watch failed")?;
            if let Some(host) = list_single_host(discovery, namespace).await? {
                return Ok(host);
            }
        }
        anyhow::bail!("KV state-agent host discovery watch ended before a host appeared")
    };
    tokio::select! {
        biased;
        _ = watch_cancel.cancelled() => {
            anyhow::bail!("runtime shutdown before the KV state-agent host appeared")
        }
        result = tokio::time::timeout_at(expiry, wait) => match result {
            Ok(result) => result,
            Err(_elapsed) => anyhow::bail!(
                "no live V2 KV state-agent host appeared within {}s \
                 (topic {KV_STATE_HOST_TOPIC_V2} on {namespace}/{HOST_COMPONENT}; \
                 set {HOST_DISCOVERY_TIMEOUT_ENV} to adjust the startup wait)",
                timeout.as_secs()
            ),
        },
    }
}

/// Absolute deadline for the wait; a span too large for the clock is
/// effectively unbounded, not a panic.
fn host_discovery_expiry(timeout: Duration) -> tokio::time::Instant {
    let now = tokio::time::Instant::now();
    now.checked_add(timeout)
        .unwrap_or_else(|| now + Duration::from_secs(u32::MAX.into()))
}

/// Shared so the zero-timeout path stays byte-identical to the legacy error message.
fn exactly_one_host_error(found: usize) -> anyhow::Error {
    anyhow::anyhow!("state-agent mode requires exactly one live V2 host, found {found}")
}

/// The one discovery query the snapshot and the watch must share.
fn host_query(namespace: &str) -> DiscoveryQuery {
    DiscoveryQuery::EventSources(EventSourceQuery::topic(
        namespace,
        HOST_COMPONENT,
        KV_STATE_HOST_TOPIC_V2,
    ))
}

/// One authoritative snapshot: `None` while no host is advertised, an error on more than one.
async fn list_single_host(
    discovery: &dyn Discovery,
    namespace: &str,
) -> Result<Option<KvStateHostAdvertisement>> {
    let instances = discovery
        .list(host_query(namespace))
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
        let host: KvStateHostAdvertisement = match serde_json::from_value(metadata) {
            Ok(host) => host,
            Err(error) => {
                tracing::warn!(%error, "Skipping undecodable KV state-agent host advertisement");
                continue;
            }
        };
        if host.protocol_version == KvStateProtocolVersion::V2
            && host.host_instance == host.control_target
        {
            hosts.push(host);
        }
    }
    if hosts.len() > 1 {
        return Err(exactly_one_host_error(hosts.len()));
    }
    Ok(hosts.pop())
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
                Err(error) if is_missing_registration(&error) => break,
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

fn is_missing_registration(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        matches!(
            cause.downcast_ref::<dynamo_runtime::storage::kv::StoreError>(),
            Some(dynamo_runtime::storage::kv::StoreError::MissingKey(_))
        )
    })
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
            let target = instance.id();
            let present = self
                .inner
                .list(intent_query())
                .await?
                .iter()
                .any(|candidate| candidate.id() == target);
            if !present {
                return Err(dynamo_runtime::storage::kv::StoreError::MissingKey(format!(
                    "{target:?}"
                ))
                .into());
            }
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
                    image_token_id: Some(99),
                    video_token_id: Some(100),
                    router_hint_source: None,
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
        assert_eq!(selected.image_token_id, Some(99));
        assert_eq!(selected.video_token_id, Some(100));
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

    /// A fresh in-memory discovery backend with nothing advertised.
    fn mock_discovery() -> Arc<dyn Discovery> {
        Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()))
    }

    /// Advertise a host on the V2 topic the way `state_agent_host` would.
    async fn register_host(
        discovery: &dyn Discovery,
        publisher_id: u64,
        advertisement: &KvStateHostAdvertisement,
    ) {
        discovery
            .register(DiscoverySpec::EventSource {
                scope: EventScope::Component {
                    namespace: "ns".to_string(),
                    component: HOST_COMPONENT.to_string(),
                },
                topic: KV_STATE_HOST_TOPIC_V2.to_string(),
                publisher_id,
                metadata: serde_json::to_value(advertisement).unwrap(),
            })
            .await
            .unwrap();
    }

    #[test]
    fn host_discovery_timeout_parses_and_defaults() {
        assert_eq!(
            host_discovery_timeout(|_| None),
            DEFAULT_HOST_DISCOVERY_TIMEOUT
        );
        assert_eq!(
            host_discovery_timeout(|_| Some(OsString::from("5"))),
            Duration::from_secs(5)
        );
        assert_eq!(
            host_discovery_timeout(|_| Some(OsString::from(" 0 "))),
            Duration::ZERO
        );
        assert_eq!(
            host_discovery_timeout(|_| Some(OsString::from("not-a-number"))),
            DEFAULT_HOST_DISCOVERY_TIMEOUT
        );
    }

    #[tokio::test]
    async fn advertised_host_is_returned_without_waiting() {
        let discovery = mock_discovery();
        register_host(discovery.as_ref(), 41, &host()).await;
        let found = await_single_host(
            discovery.as_ref(),
            "ns",
            Duration::ZERO,
            &CancellationToken::new(),
        )
        .await
        .unwrap();
        assert_eq!(found.host_instance, host().host_instance);
    }

    /// A late advertisement must wake the watch and be found by the re-list.
    #[tokio::test]
    async fn host_advertised_during_the_wait_is_discovered() {
        let discovery = mock_discovery();
        let register = discovery.clone();
        let registration = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            register_host(register.as_ref(), 41, &host()).await;
        });
        // The outer timeout bounds the test if registration fails or the
        // watch never wakes.
        let found = tokio::time::timeout(
            Duration::from_secs(10),
            await_single_host(
                discovery.as_ref(),
                "ns",
                Duration::from_secs(30),
                &CancellationToken::new(),
            ),
        )
        .await
        .expect("discovery must resolve well before the outer deadline")
        .unwrap();
        registration.await.unwrap();
        assert_eq!(found.host_instance, host().host_instance);
    }

    #[tokio::test]
    async fn oversized_timeout_expiry_saturates_instead_of_panicking() {
        let expiry = host_discovery_expiry(Duration::from_secs(u64::MAX));
        assert!(expiry > tokio::time::Instant::now());
    }

    /// Runtime shutdown must unblock the wait instead of running out the timeout.
    #[tokio::test]
    async fn shutdown_cancels_the_wait_before_the_timeout() {
        let discovery = mock_discovery();
        let shutdown = CancellationToken::new();
        let cancel = shutdown.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            cancel.cancel();
        });
        let started = tokio::time::Instant::now();
        let error = tokio::time::timeout(
            Duration::from_secs(10),
            await_single_host(discovery.as_ref(), "ns", Duration::from_secs(30), &shutdown),
        )
        .await
        .expect("cancellation must resolve the wait")
        .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(format!("{error:#}").contains("shutdown"), "{error:#}");
    }

    #[tokio::test]
    async fn missing_host_fails_with_a_diagnosable_error_after_the_deadline() {
        let discovery = mock_discovery();
        let error = await_single_host(
            discovery.as_ref(),
            "ns",
            Duration::from_millis(200),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains("no live V2 KV state-agent host appeared"),
            "{message}"
        );
        assert!(message.contains(HOST_DISCOVERY_TIMEOUT_ENV), "{message}");
    }

    #[tokio::test]
    async fn zero_timeout_preserves_the_single_snapshot_failure() {
        let discovery = mock_discovery();
        let error = await_single_host(
            discovery.as_ref(),
            "ns",
            Duration::ZERO,
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("exactly one live V2 host, found 0"));
    }

    /// Ambiguity is misconfiguration: two hosts must bail without waiting.
    #[tokio::test]
    async fn two_advertised_hosts_fail_closed_before_the_deadline() {
        let discovery = mock_discovery();
        register_host(discovery.as_ref(), 41, &host()).await;
        let second_instance = instance("kv_state_agent", 2);
        let second = KvStateHostAdvertisement {
            protocol_version: KvStateProtocolVersion::V2,
            host_instance: second_instance.clone(),
            control_target: second_instance,
            max_slots: 8,
        };
        // Distinct publisher ids: a same-id re-register resolves to the
        // existing instance and would silently collapse this into one host.
        register_host(discovery.as_ref(), 42, &second).await;
        let started = tokio::time::Instant::now();
        let error = await_single_host(
            discovery.as_ref(),
            "ns",
            Duration::from_secs(30),
            &CancellationToken::new(),
        )
        .await
        .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(format!("{error:#}").contains("exactly one live V2 host, found 2"));
    }
}
