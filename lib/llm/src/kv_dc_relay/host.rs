// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DC-scoped KV-cache Relay with one serialized CKF actor per physical pool.
//!
//! Dynamo discovery and worker-local recovery feed endpoint actors. The actors'
//! exact member ownership is authoritative; each materialization publishes one
//! physical CKF layout through the pool catalog subscription boundary.
//!
//! NOTE: One serialized actor per endpoint pool is the current measured choice, not a claim that
//! it scales indefinitely. A worker-partitioned, multi-issuer Mooncake comparison found the
//! attempted striped concurrent producer slower with worse tail admission latency. Rerun the
//! dedicated Relay campaign before changing this ownership model; further producer optimization
//! will likely be needed for substantially larger DC-scale pools.

#[cfg(feature = "kv-dc-relay-wan")]
use std::collections::HashSet;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "ckf-diagnostics")]
use std::sync::atomic::Ordering;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::CkfFailureAction;
#[cfg(feature = "kv-dc-relay-wan")]
use dynamo_kv_router::protocols::ActiveLoad;
use dynamo_kv_router::protocols::{DpRank, KvCacheEventError, WorkerId};
use dynamo_runtime::component::Component;
#[cfg(feature = "kv-dc-relay-wan")]
use dynamo_runtime::component::Instance;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::traits::DistributedRuntimeProvider;
#[cfg(feature = "kv-dc-relay-wan")]
use dynamo_runtime::transports::event_plane::EventSubscriber;
use parking_lot::Mutex;
use rand::TryRngCore;
use serde::Serialize;
use tokio::sync::{RwLock, Semaphore, watch};
use tokio::task::{JoinHandle, JoinSet};
use tokio_util::sync::CancellationToken;

use super::actor::{ActorFault, DEFAULT_FAULT_CAPACITY, KvDcRelayHandle, KvDcRelayRecoveryTarget};
use super::discovery::{
    DcMembershipView, DcMembershipWatch, EndpointMembership, KvCacheDomainKey,
    KvDcRelayDiscoveryConfig, MaterializationConflict,
};
use super::identity::{CanonicalModelRegistration, DcPoolCatalog, DcRelayIdentity, WorkerRole};
use super::pool_registry::{
    PoolActorConfig, PoolAttachRequest, PoolAttachment, PoolRegistry, PoolRetirementMode,
    drain_faults_while,
};
#[cfg(feature = "kv-dc-relay-wan")]
use super::pool_registry::{PoolPublicationConfig, PoolWanFacts};
#[cfg(feature = "kv-dc-relay-wan")]
use super::publication_hub::PublicationHubConfig;
use super::resolution::stable_dc_id;
#[cfg(feature = "kv-dc-relay-wan")]
use super::topology::TopologyPublisher;
#[cfg(feature = "kv-dc-relay-wan")]
use super::transport::{KvDcRelayTransport, WanPublicationSource};
#[cfg(feature = "kv-dc-relay-wan")]
use super::transport_config::KvDcRelayTransportConfig;
use crate::discovery::{
    KvSourceMembershipCoordinator, KvSourceMembershipView, KvSourceMembershipWatch,
};
#[cfg(feature = "kv-dc-relay-wan")]
use crate::kv_router::KV_METRICS_SUBJECT;
#[cfg(feature = "ckf-diagnostics")]
use crate::kv_router::indexer::WorkerQueryHealthSnapshot;
use crate::kv_router::indexer::{
    DEFAULT_RECOVERY_ATTEMPT_TIMEOUT, RecoverySupervisor, TargetFaultDisposition,
    start_target_subscriber,
};
use crate::local_model::runtime_config::ModelRuntimeConfig;

pub const DEFAULT_EXPECTED_UNIQUE_BLOCKS: usize = 1_048_576;
const DEFAULT_RECOVERY_FETCH_CONCURRENCY: usize = 16;
const DEFAULT_PUBLICATION_THRESHOLD: usize = 16;
const DEFAULT_PUBLICATION_DELAY: Duration = Duration::from_millis(1);

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KvDcRelayError {
    #[error("KV DC Relay is shutting down")]
    ShuttingDown,
    #[error("KV DC Relay actor stopped before completing an accepted command")]
    ActorStopped,
    #[cfg(feature = "ckf-diagnostics")]
    #[error("unknown or inactive serving endpoint {0}")]
    UnknownEndpoint(String),
    #[error("invalid tree dump for worker {worker_id} rank {dp_rank}: {message}")]
    InvalidTreeDump {
        worker_id: WorkerId,
        dp_rank: DpRank,
        message: String,
    },
    #[error(transparent)]
    Build(#[from] dynamo_kv_router::indexer::cuckoo::CkfBuildError),
    #[error(transparent)]
    Event(#[from] KvCacheEventError),
    #[error("KV DC Relay publisher requires a replacement snapshot: {0}")]
    Publisher(String),
}

#[derive(Debug, Clone)]
pub struct KvDcRelayProducerConfig {
    pub expected_unique_blocks: usize,
    pub publication_threshold: usize,
    pub publication_delay_ms: u64,
    pub recovery_attempt_timeout_ms: u64,
}

impl Default for KvDcRelayProducerConfig {
    fn default() -> Self {
        Self {
            expected_unique_blocks: DEFAULT_EXPECTED_UNIQUE_BLOCKS,
            publication_threshold: DEFAULT_PUBLICATION_THRESHOLD,
            publication_delay_ms: DEFAULT_PUBLICATION_DELAY.as_millis() as u64,
            recovery_attempt_timeout_ms: DEFAULT_RECOVERY_ATTEMPT_TIMEOUT.as_millis() as u64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KvDcRelayConfig {
    pub discovery: KvDcRelayDiscoveryConfig,
    pub producer: KvDcRelayProducerConfig,
    #[cfg(feature = "kv-dc-relay-wan")]
    pub transport: Option<KvDcRelayTransportConfig>,
}

impl Default for KvDcRelayConfig {
    fn default() -> Self {
        Self {
            discovery: KvDcRelayDiscoveryConfig {
                watch_all: true,
                ..Default::default()
            },
            producer: KvDcRelayProducerConfig::default(),
            #[cfg(feature = "kv-dc-relay-wan")]
            transport: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ActorPublicationConfig {
    threshold: usize,
    delay: Duration,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayStats {
    pub identity: KvDcRelayIdentityStats,
    pub endpoints: Vec<KvDcRelayEndpointStats>,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayIdentityStats {
    pub dc_id: String,
    pub drt_instance_id: u64,
    pub relay_incarnation: u64,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayEndpointStats {
    pub serving_endpoint: String,
    pub lifecycle: String,
    pub layout_generation: u64,
    pub cache_domain: Option<KvDcRelayCacheDomainStats>,
    pub membership_conflicts: Vec<String>,
    pub models: Vec<String>,
    pub aliases: Vec<String>,
    pub roles: Vec<String>,
    pub aggregation: Option<KvDcRelayAggregationStats>,
    pub publication: Option<KvDcRelayPublicationStats>,
    pub recovery: KvDcRelayRecoveryStats,
    pub memory: Option<KvDcRelayMemoryStats>,
    pub actor: KvDcRelayActorStats,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayCacheDomainStats {
    pub model_artifact: String,
    pub kv_block_size: u32,
    pub event_hash_format: u16,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayMemberStats {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub blocks: usize,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayAggregationStats {
    pub members: Vec<KvDcRelayMemberStats>,
    pub contribution_count: usize,
    pub unique_block_count: usize,
    pub unknown_removals: u64,
    pub parent_not_found: u64,
    pub capacity_failures: u64,
    pub occupied_bucket_count: usize,
    pub occupied_slot_count: usize,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayPublicationStats {
    pub sequence: u64,
    pub pending_events: usize,
    pub publication_count: u64,
    pub unchanged_publication_count: u64,
    pub physical_touches: u64,
    pub distinct_touched_buckets: u64,
    pub emitted_images: u64,
    pub net_reverted_buckets: u64,
    pub reset_count: u64,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Default, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayRecoveryStats {
    pub degraded_resets: u64,
    pub rebuild_count: u64,
    pub rebuild_ns: u64,
    pub rebuild_max_ns: u64,
    pub worker_count: usize,
    pub rank_count: usize,
    pub recovering_rank_count: usize,
    pub pending_live_event_count: usize,
    pub discovered_endpoint_count: usize,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayMemoryStats {
    pub filter_bytes: usize,
    pub dirty_tracking_bytes: usize,
    pub source_lineage_capacity: usize,
    pub canonical_owner_capacity: usize,
    pub insertion_scratch_capacity: usize,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Default, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayActorStats {
    pub mailbox_depth: usize,
    pub mailbox_capacity: usize,
    pub mailbox_wait_ns: u64,
    pub mailbox_max_wait_ns: u64,
    pub active_command: Option<String>,
    pub active_command_age_ms: Option<u64>,
    pub shutting_down: bool,
    pub faulted: bool,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayHealth {
    pub healthy: bool,
    pub shutting_down: bool,
    pub host_last_error: Option<String>,
    pub endpoint_count: usize,
    pub active_endpoint_count: usize,
    pub fenced_endpoint_count: usize,
    #[cfg(feature = "kv-dc-relay-wan")]
    pub wan_enabled: bool,
    #[cfg(feature = "kv-dc-relay-wan")]
    pub wan_serving: bool,
    #[cfg(feature = "kv-dc-relay-wan")]
    pub wan_bound_address: Option<String>,
    #[cfg(feature = "kv-dc-relay-wan")]
    pub wan_last_error: Option<String>,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayDiagnosticSnapshot {
    pub drt_instance_id: u64,
    pub relay_incarnation: u64,
    pub dc_id: String,
    pub serving_endpoint: String,
    pub layout_generation: u64,
    pub sequence: u64,
    pub member_count: usize,
    pub contribution_count: usize,
    pub unique_block_count: usize,
    pub format_version: u16,
    pub seed: u64,
    pub bucket_count: usize,
    pub fingerprint_bits: u8,
    pub slots_per_bucket: u8,
    pub buckets: Vec<u64>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotLifecycle {
    Discovered,
    Starting,
    Active,
    Fenced,
    Draining,
    Lightweight,
}

impl SlotLifecycle {
    #[cfg(feature = "ckf-diagnostics")]
    fn as_str(self) -> &'static str {
        match self {
            Self::Discovered => "discovered",
            Self::Starting => "starting",
            Self::Active => "active",
            Self::Fenced => "fenced",
            Self::Draining => "draining",
            Self::Lightweight => "lightweight",
        }
    }
}

#[derive(Clone)]
struct EndpointSlotStatus {
    lifecycle: SlotLifecycle,
    layout_generation: u64,
    membership: Option<EndpointMembership>,
    actor: Option<KvDcRelayHandle>,
    #[cfg(test)]
    settled_membership_generation: Option<u64>,
    #[cfg(feature = "ckf-diagnostics")]
    recovery: WorkerQueryHealthSnapshot,
}

impl Default for EndpointSlotStatus {
    fn default() -> Self {
        Self {
            lifecycle: SlotLifecycle::Lightweight,
            layout_generation: 0,
            membership: None,
            actor: None,
            #[cfg(test)]
            settled_membership_generation: None,
            #[cfg(feature = "ckf-diagnostics")]
            recovery: WorkerQueryHealthSnapshot::default(),
        }
    }
}

type SharedEndpointStatus = Arc<RwLock<EndpointSlotStatus>>;

struct EndpointSlotTask {
    metadata: watch::Sender<Option<EndpointMembership>>,
    status: SharedEndpointStatus,
    cancel: CancellationToken,
    task: JoinHandle<()>,
}

struct EndpointPoolRuntime {
    attachment: PoolAttachment,
    recovery: RecoverySupervisor<KvDcRelayRecoveryTarget>,
    binding: ActorBinding,
    registrations: Vec<CanonicalModelRegistration>,
    roles: Vec<WorkerRole>,
    #[cfg(feature = "kv-dc-relay-wan")]
    wan: Option<EndpointWanRuntime>,
}

#[cfg(feature = "kv-dc-relay-wan")]
struct EndpointWanRuntime {
    load_runtime_configs: HashMap<WorkerId, ModelRuntimeConfig>,
    load_cancel: CancellationToken,
    load_task: JoinHandle<()>,
}

#[cfg(feature = "kv-dc-relay-wan")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WanFactsMode {
    Disabled,
    Enabled,
}

#[cfg(feature = "kv-dc-relay-wan")]
impl WanFactsMode {
    const fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

#[cfg(feature = "kv-dc-relay-wan")]
struct EndpointAvailabilityWatch {
    routable: watch::Receiver<Vec<WorkerId>>,
    discovered: watch::Receiver<Vec<Instance>>,
}

#[cfg(feature = "kv-dc-relay-wan")]
impl EndpointAvailabilityWatch {
    async fn changed(&mut self) -> Result<(), watch::error::RecvError> {
        tokio::select! {
            result = self.routable.changed() => result,
            result = self.discovered.changed() => result,
        }
    }

    fn live_workers(&self) -> HashSet<WorkerId> {
        let discovered = self
            .discovered
            .borrow()
            .iter()
            .map(Instance::id)
            .collect::<HashSet<_>>();
        self.routable
            .borrow()
            .iter()
            .copied()
            .filter(|worker_id| discovered.contains(worker_id))
            .collect()
    }
}

#[derive(Default)]
struct HostTerminalState {
    last_error: Mutex<Option<String>>,
}

impl HostTerminalState {
    fn record(&self, reason: String) {
        let mut last_error = self.last_error.lock();
        if last_error.is_none() {
            *last_error = Some(reason);
        }
    }

    fn last_error(&self) -> Option<String> {
        self.last_error.lock().clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActorBinding {
    domain: KvCacheDomainKey,
    kv_state_endpoint: EndpointId,
}

fn desired_actor_binding(
    membership: &EndpointMembership,
    source_view: Option<&KvSourceMembershipView>,
) -> Option<ActorBinding> {
    if !membership.is_materializable() {
        return None;
    }
    let domain = membership.domain.as_ref()?;
    let source_view = source_view?;
    let kv_state_endpoint = source_view.resolved_kv_state_endpoint()?;
    source_view
        .sources
        .values()
        .any(|source| source.active_source().is_some())
        .then(|| ActorBinding {
            domain: domain.clone(),
            kv_state_endpoint: kv_state_endpoint.clone(),
        })
}
const MAX_PENDING_SOURCE_FAULTS: usize = DEFAULT_FAULT_CAPACITY;

enum ProducerFenceTrigger {
    Fault(ActorFault),
    PendingOverflow(ActorFault),
}

enum PendingActorAction {
    Fault(ActorFault),
    ProducerFence(ProducerFenceTrigger),
}

#[derive(Default)]
struct PendingActorFaults {
    producer_fence: Option<ProducerFenceTrigger>,
    source_faults: HashMap<(WorkerId, DpRank), ActorFault>,
    source_order: VecDeque<(WorkerId, DpRank)>,
}

impl PendingActorFaults {
    fn push(&mut self, fault: ActorFault) {
        if self.producer_fence.is_some() {
            return;
        }

        match fault.disposition.action {
            CkfFailureAction::ContinueCapacityOmission => {}
            CkfFailureAction::ReportResourceFailure | CkfFailureAction::RejectSource => {
                self.push_source_fault(fault);
            }
            CkfFailureAction::FenceAndRebuildProducer => {
                self.install_producer_fence(ProducerFenceTrigger::Fault(fault));
            }
            CkfFailureAction::DeactivateAndSnapshot | CkfFailureAction::RetrySnapshot => {
                unreachable!("consumer-lane disposition cannot originate from Relay actor")
            }
        }
    }

    fn push_source_fault(&mut self, fault: ActorFault) {
        let key = (fault.worker_id, fault.dp_rank);
        if let Some(current) = self.source_faults.get_mut(&key) {
            if fault.publisher_id != current.publisher_id
                || is_stronger_source_fault(fault.disposition.action, current.disposition.action)
            {
                *current = fault;
            }
            return;
        }

        if self.source_faults.len() >= MAX_PENDING_SOURCE_FAULTS {
            self.install_producer_fence(ProducerFenceTrigger::PendingOverflow(fault));
            return;
        }

        self.source_order.push_back(key);
        self.source_faults.insert(key, fault);
    }

    fn install_producer_fence(&mut self, trigger: ProducerFenceTrigger) {
        self.source_faults.clear();
        self.source_order.clear();
        self.producer_fence = Some(trigger);
    }

    fn drain_ready(&mut self, receiver: &mut tokio::sync::mpsc::Receiver<ActorFault>) {
        while self.producer_fence.is_none() {
            let Ok(fault) = receiver.try_recv() else {
                break;
            };
            self.push(fault);
        }
    }

    fn pop_front(&mut self) -> Option<PendingActorAction> {
        if let Some(trigger) = self.producer_fence.take() {
            return Some(PendingActorAction::ProducerFence(trigger));
        }
        while let Some(key) = self.source_order.pop_front() {
            if let Some(fault) = self.source_faults.remove(&key) {
                return Some(PendingActorAction::Fault(fault));
            }
        }
        None
    }

    fn take_producer_fence(&mut self) -> Option<ProducerFenceTrigger> {
        self.producer_fence.take()
    }

    fn clear(&mut self) {
        self.producer_fence = None;
        self.source_faults.clear();
        self.source_order.clear();
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        usize::from(self.producer_fence.is_some()) + self.source_faults.len()
    }
}

fn is_stronger_source_fault(candidate: CkfFailureAction, current: CkfFailureAction) -> bool {
    matches!(
        (current, candidate),
        (
            CkfFailureAction::ReportResourceFailure,
            CkfFailureAction::RejectSource
        )
    )
}

/// DC-wide Relay host. It is intentionally not scoped to a model, namespace, or endpoint.
pub struct KvDcRelay {
    #[cfg(feature = "ckf-diagnostics")]
    dc_id: Arc<str>,
    #[cfg(any(feature = "ckf-diagnostics", feature = "kv-dc-relay-wan"))]
    relay_identity: DcRelayIdentity,
    cancel: CancellationToken,
    membership: Mutex<Option<DcMembershipWatch>>,
    supervisor: Mutex<Option<JoinHandle<()>>>,
    terminal: Arc<HostTerminalState>,
    statuses: Arc<RwLock<HashMap<EndpointId, SharedEndpointStatus>>>,
    pools: Arc<PoolRegistry>,
    #[cfg(feature = "kv-dc-relay-wan")]
    transport: Mutex<Option<KvDcRelayTransport>>,
}

impl KvDcRelay {
    pub async fn start(
        component: Component,
        dc_id: String,
        config: KvDcRelayConfig,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!dc_id.is_empty(), "KV DC Relay dc_id must not be empty");
        anyhow::ensure!(
            dc_id.trim() == dc_id,
            "KV DC Relay dc_id must not contain leading or trailing whitespace"
        );
        config.discovery.validate()?;
        anyhow::ensure!(
            config.producer.expected_unique_blocks != 0,
            "KV DC Relay expected_unique_blocks must be positive"
        );
        anyhow::ensure!(
            config.producer.publication_threshold != 0,
            "KV DC Relay publication_threshold must be positive"
        );
        anyhow::ensure!(
            config.producer.publication_delay_ms != 0,
            "KV DC Relay publication_delay_ms must be positive"
        );
        anyhow::ensure!(
            config.producer.recovery_attempt_timeout_ms != 0,
            "KV DC Relay recovery_attempt_timeout_ms must be positive"
        );
        #[cfg(feature = "kv-dc-relay-wan")]
        let transport_config = config.transport.clone();
        #[cfg(feature = "kv-dc-relay-wan")]
        let wan_facts_mode = if transport_config.is_some() {
            WanFactsMode::Enabled
        } else {
            WanFactsMode::Disabled
        };
        #[cfg(feature = "kv-dc-relay-wan")]
        if let Some(transport) = transport_config.as_ref() {
            transport.validate()?;
        }
        let publication = ActorPublicationConfig {
            threshold: config.producer.publication_threshold,
            delay: Duration::from_millis(config.producer.publication_delay_ms),
        };
        let relay_identity =
            DcRelayIdentity::new(component.drt().connection_id(), new_relay_incarnation()?);
        let cancel = component.drt().child_token();
        let membership = DcMembershipWatch::start(
            component.drt().discovery(),
            config.discovery,
            cancel.clone(),
        )
        .await?;
        let membership_rx = membership.subscribe();
        let statuses = Arc::new(RwLock::new(HashMap::new()));
        let dc_id: Arc<str> = Arc::from(dc_id);
        let ckf_dc_id = stable_dc_id(dc_id.as_ref());
        let actor_config = PoolActorConfig {
            expected_unique_blocks: config.producer.expected_unique_blocks,
            publication_threshold: publication.threshold,
            publication_delay: publication.delay,
        };
        #[cfg(feature = "kv-dc-relay-wan")]
        let pools = Arc::new(match transport_config.as_ref() {
            Some(transport) => PoolRegistry::new_with_publication_config(
                relay_identity,
                actor_config,
                PoolPublicationConfig {
                    hub: PublicationHubConfig {
                        queue_capacity: transport.publication_queue_capacity,
                        queue_bytes: transport.publication_queue_bytes,
                        max_subscribers: transport.max_subscribers_per_pool,
                        max_delta_images: super::protocol::wire::images::max_delta_images(),
                        encoding_permits: Arc::new(Semaphore::new(
                            transport.publication_encoding_concurrency,
                        )),
                        #[cfg(test)]
                        initialization_gate: None,
                    },
                    max_initialized_pool_hubs: transport.max_initialized_pool_hubs,
                },
            ),
            None => PoolRegistry::new(relay_identity, actor_config),
        });
        #[cfg(not(feature = "kv-dc-relay-wan"))]
        let pools = Arc::new(PoolRegistry::new(relay_identity, actor_config));
        #[cfg(feature = "kv-dc-relay-wan")]
        let topology = {
            let mut initial_view = membership_rx.borrow().clone();
            reject_duplicate_live_pools(&mut initial_view, ckf_dc_id);
            Arc::new(TopologyPublisher::new(initial_view, &pools.catalog()))
        };
        #[cfg(feature = "kv-dc-relay-wan")]
        let transport = if let Some(transport_config) = transport_config {
            let source = WanPublicationSource::new(
                component.clone(),
                pools.clone(),
                topology.clone(),
                relay_identity,
                cancel.clone(),
            );
            match KvDcRelayTransport::start(source, transport_config).await {
                Ok(transport) => Some(transport),
                Err(error) => {
                    cancel.cancel();
                    membership.shutdown().await;
                    pools.shutdown().await;
                    return Err(error);
                }
            }
        } else {
            None
        };
        let terminal = Arc::new(HostTerminalState::default());
        let host = tokio::spawn(run_host_supervisor(
            component,
            ckf_dc_id,
            membership_rx,
            statuses.clone(),
            pools.clone(),
            Duration::from_millis(config.producer.recovery_attempt_timeout_ms),
            cancel.child_token(),
            cancel.clone(),
            terminal.clone(),
            #[cfg(feature = "kv-dc-relay-wan")]
            wan_facts_mode,
            #[cfg(feature = "kv-dc-relay-wan")]
            topology.clone(),
        ));
        let supervisor = tokio::spawn(supervise_host_task(
            host,
            cancel.clone(),
            terminal.clone(),
            pools.clone(),
        ));
        Ok(Self {
            #[cfg(feature = "ckf-diagnostics")]
            dc_id,
            #[cfg(any(feature = "ckf-diagnostics", feature = "kv-dc-relay-wan"))]
            relay_identity,
            cancel,
            membership: Mutex::new(Some(membership)),
            supervisor: Mutex::new(Some(supervisor)),
            terminal,
            statuses,
            pools,
            #[cfg(feature = "kv-dc-relay-wan")]
            transport: Mutex::new(transport),
        })
    }

    pub fn pool_catalog(&self) -> DcPoolCatalog {
        self.pools.catalog()
    }

    pub fn watch_pool_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.pools.watch_catalog()
    }

    #[cfg(feature = "kv-dc-relay-wan")]
    pub const fn relay_identity(&self) -> DcRelayIdentity {
        self.relay_identity
    }

    #[cfg(feature = "ckf-diagnostics")]
    pub async fn stats(&self) -> Result<KvDcRelayStats, KvDcRelayError> {
        let statuses: Vec<_> = self
            .statuses
            .read()
            .await
            .iter()
            .map(|(endpoint, status)| (endpoint.clone(), status.clone()))
            .collect();
        let mut endpoints = Vec::with_capacity(statuses.len());
        for (endpoint, status) in statuses {
            endpoints.push(endpoint_stats(endpoint, status).await?);
        }
        endpoints
            .sort_unstable_by(|left, right| left.serving_endpoint.cmp(&right.serving_endpoint));
        Ok(KvDcRelayStats {
            identity: KvDcRelayIdentityStats {
                dc_id: self.dc_id.to_string(),
                drt_instance_id: self.relay_identity.drt_instance_id(),
                relay_incarnation: self.relay_identity.relay_incarnation(),
            },
            endpoints,
        })
    }

    #[cfg(feature = "ckf-diagnostics")]
    pub async fn diagnostic_snapshot(
        &self,
        endpoint: &EndpointId,
    ) -> Result<KvDcRelayDiagnosticSnapshot, KvDcRelayError> {
        let status = self
            .statuses
            .read()
            .await
            .get(endpoint)
            .cloned()
            .ok_or_else(|| KvDcRelayError::UnknownEndpoint(endpoint.to_string()))?;
        let status = status.read().await;
        let handle = status
            .actor
            .clone()
            .ok_or_else(|| KvDcRelayError::UnknownEndpoint(endpoint.to_string()))?;
        let layout_generation = status.layout_generation;
        drop(status);
        let actor_snapshot = handle.snapshot().await?;
        let format = actor_snapshot.identity.format();
        let aggregation = actor_snapshot.stats.aggregation();
        Ok(KvDcRelayDiagnosticSnapshot {
            drt_instance_id: self.relay_identity.drt_instance_id(),
            relay_incarnation: self.relay_identity.relay_incarnation(),
            dc_id: self.dc_id.to_string(),
            serving_endpoint: endpoint.to_string(),
            layout_generation,
            sequence: actor_snapshot.sequence,
            member_count: aggregation.member_count(),
            contribution_count: aggregation.contribution_count(),
            unique_block_count: aggregation.unique_block_count(),
            format_version: format.format_version(),
            seed: format.seed(),
            bucket_count: format.bucket_count(),
            fingerprint_bits: format.fingerprint_bits(),
            slots_per_bucket: format.slots_per_bucket(),
            buckets: actor_snapshot.buckets.into_vec(),
        })
    }

    /// Force every materialized endpoint to publish its pending cadence tail.
    pub async fn flush(&self) -> Result<(), KvDcRelayError> {
        let statuses: Vec<_> = self.statuses.read().await.values().cloned().collect();
        for status in statuses {
            let handle = status.read().await.actor.clone();
            if let Some(handle) = handle {
                handle.flush().await?;
            }
        }
        Ok(())
    }

    pub async fn health(&self) -> KvDcRelayHealth {
        let statuses: Vec<_> = self.statuses.read().await.values().cloned().collect();
        let mut active_endpoint_count = 0;
        let mut fenced_endpoint_count = 0;
        for status in &statuses {
            match status.read().await.lifecycle {
                SlotLifecycle::Active => active_endpoint_count += 1,
                SlotLifecycle::Fenced => fenced_endpoint_count += 1,
                _ => {}
            }
        }
        #[cfg(feature = "kv-dc-relay-wan")]
        let transport_health = self
            .transport
            .lock()
            .as_ref()
            .map(KvDcRelayTransport::health)
            .unwrap_or_default();
        #[cfg(feature = "kv-dc-relay-wan")]
        let transport_healthy = !transport_health.enabled
            || (transport_health.serving && transport_health.last_error.is_none());
        let host_last_error = self.terminal.last_error();
        KvDcRelayHealth {
            healthy: !self.cancel.is_cancelled()
                && host_last_error.is_none()
                && fenced_endpoint_count == 0
                && {
                    #[cfg(feature = "kv-dc-relay-wan")]
                    {
                        transport_healthy
                    }
                    #[cfg(not(feature = "kv-dc-relay-wan"))]
                    {
                        true
                    }
                },
            shutting_down: self.cancel.is_cancelled(),
            host_last_error,
            endpoint_count: statuses.len(),
            active_endpoint_count,
            fenced_endpoint_count,
            #[cfg(feature = "kv-dc-relay-wan")]
            wan_enabled: transport_health.enabled,
            #[cfg(feature = "kv-dc-relay-wan")]
            wan_serving: transport_health.serving,
            #[cfg(feature = "kv-dc-relay-wan")]
            wan_bound_address: transport_health
                .bound_address
                .map(|address| address.to_string()),
            #[cfg(feature = "kv-dc-relay-wan")]
            wan_last_error: transport_health.last_error,
        }
    }

    pub async fn wait_for_shutdown(&self) {
        self.cancel.cancelled().await;
    }

    pub async fn shutdown(&self) -> Result<(), KvDcRelayError> {
        self.cancel.cancel();
        #[cfg(feature = "kv-dc-relay-wan")]
        let transport = self.transport.lock().take();
        #[cfg(feature = "kv-dc-relay-wan")]
        if let Some(transport) = transport {
            transport.shutdown().await;
        }
        let supervisor = self.supervisor.lock().take();
        if let Some(supervisor) = supervisor
            && let Err(error) = supervisor.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV DC Relay host supervisor failed during shutdown");
        }
        let membership = self.membership.lock().take();
        if let Some(membership) = membership {
            membership.shutdown().await;
        }
        Ok(())
    }
}

impl Drop for KvDcRelay {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

fn new_relay_incarnation() -> anyhow::Result<u64> {
    let random_id = rand::rngs::OsRng
        .try_next_u64()
        .map_err(|error| anyhow::anyhow!("failed to generate Relay incarnation: {error}"))?;
    Ok(random_id & (i64::MAX as u64))
}

#[allow(clippy::too_many_arguments)]
async fn run_host_supervisor(
    component: Component,
    ckf_dc_id: dynamo_kv_router::DcId,
    mut membership_rx: watch::Receiver<DcMembershipView>,
    statuses: Arc<RwLock<HashMap<EndpointId, SharedEndpointStatus>>>,
    pools: Arc<PoolRegistry>,
    recovery_attempt_timeout: Duration,
    cancel: CancellationToken,
    fatal_cancel: CancellationToken,
    terminal: Arc<HostTerminalState>,
    #[cfg(feature = "kv-dc-relay-wan")] wan_facts_mode: WanFactsMode,
    #[cfg(feature = "kv-dc-relay-wan")] topology: Arc<TopologyPublisher>,
) -> anyhow::Result<()> {
    let recovery_fetch_permit = Arc::new(Semaphore::new(DEFAULT_RECOVERY_FETCH_CONCURRENCY));
    let mut slots: HashMap<EndpointId, EndpointSlotTask> = HashMap::new();
    let mut retired_slots = JoinSet::new();
    #[cfg(feature = "kv-dc-relay-wan")]
    let mut catalog_rx = pools.watch_catalog();
    #[cfg(feature = "kv-dc-relay-wan")]
    let mut next_slot_incarnation = 1u64;

    let outcome = 'supervisor: loop {
        let mut view = membership_rx.borrow_and_update().clone();
        reject_duplicate_live_pools(&mut view, ckf_dc_id);
        #[cfg(feature = "kv-dc-relay-wan")]
        {
            topology.replace_membership(view.clone());
            topology.replace_catalog(&catalog_rx.borrow_and_update());
        }
        for (endpoint, membership) in view.endpoints.iter() {
            let slot = match slots.entry(endpoint.clone()) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let (metadata, metadata_rx) = watch::channel(None);
                    let status = Arc::new(RwLock::new(EndpointSlotStatus::default()));
                    let slot_cancel = cancel.child_token();
                    #[cfg(feature = "kv-dc-relay-wan")]
                    let slot_incarnation =
                        match allocate_slot_incarnation(&mut next_slot_incarnation) {
                            Ok(slot_incarnation) => {
                                topology.claim_availability(endpoint.clone(), slot_incarnation);
                                slot_incarnation
                            }
                            Err(error) => {
                                record_host_failure(&fatal_cancel, &terminal, error.to_string());
                                break 'supervisor Err(error);
                            }
                        };
                    let task = tokio::spawn(run_endpoint_slot(
                        component.clone(),
                        ckf_dc_id,
                        endpoint.clone(),
                        metadata_rx,
                        status.clone(),
                        Arc::new(Semaphore::new(1)),
                        recovery_fetch_permit.clone(),
                        pools.clone(),
                        recovery_attempt_timeout,
                        slot_cancel.clone(),
                        #[cfg(feature = "kv-dc-relay-wan")]
                        slot_incarnation,
                        #[cfg(feature = "kv-dc-relay-wan")]
                        wan_facts_mode,
                        #[cfg(feature = "kv-dc-relay-wan")]
                        topology.clone(),
                    ));
                    entry.insert(EndpointSlotTask {
                        metadata,
                        status,
                        cancel: slot_cancel,
                        task,
                    })
                }
            };
            publish_endpoint_metadata_if_changed(&slot.metadata, membership);
        }
        retire_departed_endpoint_slots(&view, &mut slots, &mut retired_slots);
        *statuses.write().await = slots
            .iter()
            .map(|(endpoint, slot)| (endpoint.clone(), slot.status.clone()))
            .collect();

        loop {
            let catalog_changed = async {
                #[cfg(feature = "kv-dc-relay-wan")]
                {
                    return catalog_rx.changed().await;
                }
                #[cfg(not(feature = "kv-dc-relay-wan"))]
                std::future::pending::<Result<(), watch::error::RecvError>>().await
            };
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break 'supervisor Ok(()),
                changed = membership_rx.changed() => {
                    if changed.is_err() {
                        if cancel.is_cancelled() {
                            break 'supervisor Ok(());
                        }
                        let reason = "KV DC Relay membership watch closed unexpectedly".to_string();
                        record_host_failure(&fatal_cancel, &terminal, reason.clone());
                        break 'supervisor Err(anyhow::anyhow!(reason));
                    }
                    break;
                }
                changed = catalog_changed => {
                    #[cfg(feature = "kv-dc-relay-wan")]
                    {
                        if changed.is_err() {
                            if cancel.is_cancelled() {
                                break 'supervisor Ok(());
                            }
                            let reason = "KV DC Relay pool catalog watch closed unexpectedly".to_string();
                            record_host_failure(&fatal_cancel, &terminal, reason.clone());
                            break 'supervisor Err(anyhow::anyhow!(reason));
                        }
                        topology.replace_catalog(&catalog_rx.borrow_and_update());
                        continue;
                    }
                    #[cfg(not(feature = "kv-dc-relay-wan"))]
                    let _ = changed;
                }
                retired = retired_slots.join_next(), if !retired_slots.is_empty() => {
                    report_retired_endpoint_slot(retired);
                }
            }
        }
    };

    for (endpoint, slot) in slots {
        slot.cancel.cancel();
        drop(slot.metadata);
        report_endpoint_slot_exit(endpoint, slot.task.await);
    }
    while let Some(retired) = retired_slots.join_next().await {
        report_retired_endpoint_slot(Some(retired));
    }
    pools.shutdown().await;
    outcome
}

#[cfg(feature = "kv-dc-relay-wan")]
fn allocate_slot_incarnation(next: &mut u64) -> anyhow::Result<u64> {
    let incarnation = *next;
    *next = incarnation
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("KV DC Relay endpoint-slot incarnation space exhausted"))?;
    Ok(incarnation)
}

async fn supervise_host_task(
    host: JoinHandle<anyhow::Result<()>>,
    cancel: CancellationToken,
    terminal: Arc<HostTerminalState>,
    pools: Arc<PoolRegistry>,
) {
    let result = host.await;
    if cancel.is_cancelled() {
        return;
    }
    let reason = match result {
        Ok(Ok(())) => "KV DC Relay host stopped unexpectedly".to_string(),
        Ok(Err(error)) => error.to_string(),
        Err(error) => format!("KV DC Relay host task failed: {error}"),
    };
    record_host_failure(&cancel, &terminal, reason);
    pools.shutdown().await;
}

fn record_host_failure(cancel: &CancellationToken, terminal: &HostTerminalState, reason: String) {
    tracing::error!(error = %reason, "KV DC Relay host failed");
    terminal.record(reason);
    cancel.cancel();
}

fn reject_duplicate_live_pools(view: &mut DcMembershipView, dc_id: dynamo_kv_router::DcId) {
    let mut owners: HashMap<PoolId, Vec<EndpointId>> = HashMap::new();
    for (endpoint, membership) in view.endpoints.iter() {
        if !membership.is_materializable() {
            continue;
        }
        let Some(domain) = &membership.domain else {
            continue;
        };
        owners
            .entry(PoolId::new(domain.id, dc_id))
            .or_default()
            .push(endpoint.clone());
    }

    for (pool_id, mut endpoints) in owners {
        if endpoints.len() < 2 {
            continue;
        }
        endpoints.sort_unstable_by_key(ToString::to_string);
        tracing::error!(
            %pool_id,
            endpoints = ?endpoints,
            "multiple live serving endpoints resolve to one CKF pool; fencing all colliding endpoints"
        );
        let memberships = Arc::make_mut(&mut view.endpoints);
        for endpoint in &endpoints {
            let Some(membership) = memberships.get_mut(endpoint) else {
                continue;
            };
            membership
                .conflicts
                .push(MaterializationConflict::Endpoint {
                    endpoint: endpoint.clone(),
                    reason: format!("pool {pool_id} is claimed by multiple serving endpoints"),
                });
        }
    }
}

fn inactive_slot_lifecycle(membership: Option<&EndpointMembership>) -> SlotLifecycle {
    match membership {
        None => SlotLifecycle::Lightweight,
        Some(membership) if !membership.conflicts.is_empty() => SlotLifecycle::Fenced,
        Some(_) => SlotLifecycle::Discovered,
    }
}

fn publish_endpoint_metadata_if_changed(
    sender: &watch::Sender<Option<EndpointMembership>>,
    membership: &EndpointMembership,
) {
    sender.send_if_modified(|current| {
        if current.as_ref() == Some(membership) {
            return false;
        }
        *current = Some(membership.clone());
        true
    });
}

fn retire_departed_endpoint_slots(
    view: &DcMembershipView,
    slots: &mut HashMap<EndpointId, EndpointSlotTask>,
    retired_slots: &mut JoinSet<(EndpointId, Result<(), tokio::task::JoinError>)>,
) {
    let departed: Vec<_> = slots
        .keys()
        .filter(|endpoint| !view.endpoints.contains_key(*endpoint))
        .cloned()
        .collect();
    for endpoint in departed {
        let Some(slot) = slots.remove(&endpoint) else {
            continue;
        };
        slot.cancel.cancel();
        drop(slot.metadata);
        retired_slots.spawn(async move {
            let result = slot.task.await;
            (endpoint, result)
        });
    }
}

type RetiredEndpointSlot =
    Result<(EndpointId, Result<(), tokio::task::JoinError>), tokio::task::JoinError>;

fn report_retired_endpoint_slot(retired: Option<RetiredEndpointSlot>) {
    match retired {
        Some(Ok((endpoint, result))) => report_endpoint_slot_exit(endpoint, result),
        Some(Err(error)) if !error.is_cancelled() => {
            tracing::warn!(%error, "KV DC Relay endpoint retirement monitor failed");
        }
        Some(Err(_)) | None => {}
    }
}

fn report_endpoint_slot_exit(endpoint: EndpointId, result: Result<(), tokio::task::JoinError>) {
    if let Err(error) = result
        && !error.is_cancelled()
    {
        tracing::warn!(%endpoint, %error, "KV DC Relay endpoint slot failed");
    }
}

fn report_actor_fault(endpoint: &EndpointId, fault: &ActorFault) {
    tracing::error!(
        %endpoint,
        publisher_id = fault.publisher_id,
        worker_id = fault.worker_id,
        dp_rank = fault.dp_rank,
        event_id = ?fault.event_id,
        category = ?fault.category,
        error = %fault.message,
        "KV DC Relay actor failed an admitted mutation"
    );
}

fn report_producer_fence_trigger(endpoint: &EndpointId, trigger: &ProducerFenceTrigger) {
    match trigger {
        ProducerFenceTrigger::Fault(fault) => report_actor_fault(endpoint, fault),
        ProducerFenceTrigger::PendingOverflow(fault) => tracing::error!(
            %endpoint,
            publisher_id = fault.publisher_id,
            worker_id = fault.worker_id,
            dp_rank = fault.dp_rank,
            event_id = ?fault.event_id,
            category = ?fault.category,
            action = ?fault.disposition.action,
            error = %fault.message,
            pending_capacity = MAX_PENDING_SOURCE_FAULTS,
            "KV DC Relay pending source faults exceeded their bound; fencing producer"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_endpoint_slot(
    component: Component,
    ckf_dc_id: dynamo_kv_router::DcId,
    endpoint: EndpointId,
    mut metadata_rx: watch::Receiver<Option<EndpointMembership>>,
    status: SharedEndpointStatus,
    rebuild_permit: Arc<Semaphore>,
    recovery_fetch_permit: Arc<Semaphore>,
    pools: Arc<PoolRegistry>,
    recovery_attempt_timeout: Duration,
    cancel: CancellationToken,
    #[cfg(feature = "kv-dc-relay-wan")] slot_incarnation: u64,
    #[cfg(feature = "kv-dc-relay-wan")] wan_facts_mode: WanFactsMode,
    #[cfg(feature = "kv-dc-relay-wan")] topology: Arc<TopologyPublisher>,
) {
    let mut config_tx: Option<watch::Sender<HashMap<WorkerId, ModelRuntimeConfig>>> = None;
    let mut source_watch: Option<KvSourceMembershipWatch> = None;
    let mut runtime: Option<EndpointPoolRuntime> = None;
    #[cfg(feature = "kv-dc-relay-wan")]
    let mut instance_rx = None;
    #[cfg(feature = "kv-dc-relay-wan")]
    let mut availability_retry_delay = Duration::from_millis(100);
    let mut layout_generation = 0u64;
    let mut retry_binding: Option<ActorBinding> = None;
    let mut retry_delay = Duration::from_millis(100);
    let mut start_failures = 0u64;
    let mut registration_refresh_failures = 0u64;
    let mut role_refresh_failures = 0u64;
    let mut pending_faults = PendingActorFaults::default();

    loop {
        let membership = metadata_rx.borrow_and_update().clone();
        #[cfg(test)]
        let membership_generation = membership.as_ref().map(|membership| membership.generation);
        {
            let mut current = status.write().await;
            #[cfg(test)]
            if current.settled_membership_generation != membership_generation {
                current.settled_membership_generation = None;
            }
            current.membership = membership.clone();
            current.layout_generation = layout_generation;
            if runtime.is_none() {
                current.lifecycle = inactive_slot_lifecycle(membership.as_ref());
            }
        }

        #[cfg(feature = "kv-dc-relay-wan")]
        if wan_facts_mode.is_enabled() && membership.is_some() && instance_rx.is_none() {
            match instance_availability_watch(&component, &endpoint).await {
                Ok(receiver) => {
                    // Both runtime watchers carry an initial snapshot. An empty snapshot is an
                    // authoritative observation that the endpoint currently has no live workers.
                    topology.replace_availability(
                        endpoint.clone(),
                        slot_incarnation,
                        Some(receiver.live_workers()),
                    );
                    instance_rx = Some(receiver);
                    availability_retry_delay = Duration::from_millis(100);
                }
                Err(error) => {
                    topology.replace_availability(endpoint.clone(), slot_incarnation, None);
                    tracing::debug!(
                        %endpoint,
                        %error,
                        retry_ms = availability_retry_delay.as_millis(),
                        "Failed to observe routable endpoint instances for Relay readiness"
                    );
                }
            }
        }

        if let Some(membership) = &membership {
            if let Some(sender) = &config_tx {
                sender.send_if_modified(|current| {
                    if current == &membership.runtime_configs {
                        return false;
                    }
                    current.clone_from(&membership.runtime_configs);
                    true
                });
            } else {
                let (sender, configs) = watch::channel(membership.runtime_configs.clone());
                let coordinator = KvSourceMembershipCoordinator::start(
                    endpoint.clone(),
                    configs,
                    component.drt().discovery(),
                );
                source_watch = Some(coordinator.subscribe());
                config_tx = Some(sender);
            }
        }

        let source_view = source_watch.as_ref().map(|watch| watch.borrow().clone());
        let source_binding_pending = membership.as_ref().zip(source_view.as_ref()).is_some_and(
            |(membership, source_view)| {
                !source_view.matches_binding_inputs(&membership.runtime_configs)
            },
        );
        let desired_binding = membership
            .as_ref()
            .and_then(|membership| desired_actor_binding(membership, source_view.as_ref()));
        if retry_binding.as_ref() != desired_binding.as_ref() {
            retry_binding = desired_binding.clone();
            retry_delay = Duration::from_millis(100);
            start_failures = 0;
            registration_refresh_failures = 0;
            role_refresh_failures = 0;
        }

        let binding_changed = runtime
            .as_ref()
            .is_some_and(|active| Some(&active.binding) != desired_binding.as_ref());
        if binding_changed || membership.is_none() {
            if let Some(active) = runtime.take() {
                let lifecycle = inactive_slot_lifecycle(membership.as_ref());
                status.write().await.lifecycle = if lifecycle == SlotLifecycle::Fenced {
                    SlotLifecycle::Fenced
                } else {
                    SlotLifecycle::Draining
                };
                if lifecycle == SlotLifecycle::Fenced {
                    fence_endpoint_pool(active, &pools).await;
                } else {
                    stop_endpoint_pool(active, &pools).await;
                }
                pending_faults.clear();
                let mut current = status.write().await;
                current.actor = None;
                current.lifecycle = lifecycle;
            }
            if membership.is_none() {
                config_tx = None;
                source_watch = None;
                #[cfg(feature = "kv-dc-relay-wan")]
                {
                    instance_rx = None;
                    availability_retry_delay = Duration::from_millis(100);
                }
                let mut current = status.write().await;
                current.lifecycle = SlotLifecycle::Lightweight;
                #[cfg(feature = "ckf-diagnostics")]
                {
                    current.recovery = WorkerQueryHealthSnapshot::default();
                }
            }
        }

        if let (Some(active), Some(membership)) = (runtime.as_mut(), membership.as_ref())
            && !membership.registrations.is_empty()
            && membership.conflicts.is_empty()
            && active.registrations != membership.registrations
        {
            match refresh_pool_registrations(
                &pools,
                &mut active.attachment,
                &active.binding,
                desired_binding.as_ref(),
                source_binding_pending,
                &membership.registrations,
            )
            .await
            {
                Ok(RegistrationRefresh::Skipped) => {}
                Ok(RegistrationRefresh::Published) => {
                    registration_refresh_failures = 0;
                    active.registrations.clone_from(&membership.registrations);
                }
                Err(error) => {
                    registration_refresh_failures = registration_refresh_failures.saturating_add(1);
                    if registration_refresh_failures == 1 {
                        tracing::warn!(
                            %endpoint,
                            %error,
                            "Failed to refresh KV DC Relay model bindings"
                        );
                    } else {
                        tracing::debug!(
                            %endpoint,
                            %error,
                            registration_refresh_failures,
                            "KV DC Relay model binding refresh failed again"
                        );
                    }
                }
            }
        }

        if let (Some(active), Some(membership)) = (runtime.as_mut(), membership.as_ref())
            && !source_binding_pending
            && Some(&active.binding) == desired_binding.as_ref()
            && !membership.registrations.is_empty()
            && membership.conflicts.is_empty()
            && active.roles != membership.roles
        {
            match pools.replace_roles(&mut active.attachment, membership.roles.clone()) {
                Ok(()) => {
                    role_refresh_failures = 0;
                    active.roles.clone_from(&membership.roles);
                }
                Err(error) => {
                    role_refresh_failures = role_refresh_failures.saturating_add(1);
                    if role_refresh_failures == 1 {
                        tracing::warn!(
                            %endpoint,
                            %error,
                            "Failed to refresh KV DC Relay pool roles"
                        );
                    } else {
                        tracing::debug!(
                            %endpoint,
                            %error,
                            role_refresh_failures,
                            "KV DC Relay pool role refresh failed again"
                        );
                    }
                }
            }
        }

        #[cfg(feature = "kv-dc-relay-wan")]
        if let (Some(active), Some(membership)) = (runtime.as_mut(), membership.as_ref())
            && !source_binding_pending
            && Some(&active.binding) == desired_binding.as_ref()
            && let Some(wan) = active.wan.as_mut()
            && wan.load_runtime_configs != membership.runtime_configs
        {
            match pools.replace_load_capacity(
                active.attachment.pool_id,
                active.attachment.layout_generation,
                &membership.runtime_configs,
            ) {
                Ok(true) => wan
                    .load_runtime_configs
                    .clone_from(&membership.runtime_configs),
                Ok(false) => {}
                Err(error) => tracing::warn!(
                    %endpoint,
                    %error,
                    "Failed to refresh KV DC Relay pool load capacity"
                ),
            }
        }

        if runtime.is_none()
            && !source_binding_pending
            && let (Some(binding), Some(membership), Some(membership_watch)) = (
                desired_binding.clone(),
                membership.clone(),
                source_watch.clone(),
            )
        {
            status.write().await.lifecycle = SlotLifecycle::Starting;
            match start_endpoint_pool(
                component.clone(),
                ckf_dc_id,
                endpoint.clone(),
                binding.clone(),
                membership.registrations.clone(),
                membership.roles.clone(),
                #[cfg(feature = "kv-dc-relay-wan")]
                wan_facts_mode.is_enabled().then(|| PoolWanFacts {
                    runtime_configs: membership.runtime_configs.clone(),
                }),
                membership_watch,
                rebuild_permit.clone(),
                recovery_fetch_permit.clone(),
                pools.clone(),
                recovery_attempt_timeout,
                cancel.child_token(),
            )
            .await
            {
                Ok(candidate)
                    if metadata_rx.borrow().as_ref() == Some(&membership)
                        && source_watch.as_ref().and_then(|watch| {
                            watch.borrow().resolved_kv_state_endpoint().cloned()
                        }) == Some(binding.kv_state_endpoint.clone()) =>
                {
                    retry_delay = Duration::from_millis(100);
                    start_failures = 0;
                    registration_refresh_failures = 0;
                    role_refresh_failures = 0;
                    layout_generation = candidate.attachment.layout_generation;
                    let mut current = status.write().await;
                    current.layout_generation = layout_generation;
                    current.actor = Some(candidate.attachment.handle.clone());
                    current.lifecycle = SlotLifecycle::Active;
                    runtime = Some(candidate);
                }
                Ok(candidate) => {
                    stop_endpoint_pool(candidate, &pools).await;
                }
                Err(error) => {
                    start_failures = start_failures.saturating_add(1);
                    if start_failures == 1 {
                        tracing::error!(%endpoint, %error, "Failed to materialize KV DC Relay endpoint actor");
                    } else {
                        tracing::debug!(%endpoint, %error, start_failures, retry_ms = retry_delay.as_millis(), "KV DC Relay endpoint actor retry failed");
                    }
                    let mut current = status.write().await;
                    current.lifecycle = SlotLifecycle::Fenced;
                    current.actor = None;
                }
            }
        }

        #[cfg(test)]
        {
            status.write().await.settled_membership_generation = membership_generation;
        }

        enum SlotInput {
            Metadata,
            Source,
            SourceClosed,
            #[cfg(feature = "kv-dc-relay-wan")]
            Instance,
            #[cfg(feature = "kv-dc-relay-wan")]
            AvailabilityRetry,
            Fault(PendingActorAction),
            PoolUnavailable,
            Health,
            Retry,
            Cancelled,
        }
        #[cfg(feature = "kv-dc-relay-wan")]
        let availability_input = async {
            if let Some(instance_rx) = instance_rx.as_mut() {
                return if instance_rx.changed().await.is_ok() {
                    SlotInput::Instance
                } else {
                    SlotInput::AvailabilityRetry
                };
            }
            if wan_facts_mode.is_enabled() && membership.is_some() {
                tokio::time::sleep(availability_retry_delay).await;
                return SlotInput::AvailabilityRetry;
            }
            std::future::pending().await
        };
        #[cfg(not(feature = "kv-dc-relay-wan"))]
        let availability_input = std::future::pending::<SlotInput>();
        if let Some(active) = runtime.as_mut() {
            pending_faults.drain_ready(&mut active.attachment.faults);
        }
        let pool_cancel = runtime
            .as_ref()
            .map(|active| active.attachment.pool_cancel.clone());
        let input = tokio::select! {
            _ = cancel.cancelled() => SlotInput::Cancelled,
            changed = metadata_rx.changed() => {
                if changed.is_ok() { SlotInput::Metadata } else { SlotInput::Cancelled }
            }
            changed = async {
                let Some(source_watch) = source_watch.as_mut() else {
                    return std::future::pending().await;
                };
                source_watch.changed().await
            } => {
                if changed.is_ok() { SlotInput::Source } else { SlotInput::SourceClosed }
            }
            availability = availability_input => availability,
            fault = async {
                if let Some(fault) = pending_faults.pop_front() {
                    return Some(fault);
                }
                let Some(runtime) = runtime.as_mut() else {
                    return std::future::pending().await;
                };
                runtime
                    .attachment
                    .faults
                    .recv()
                    .await
                    .map(PendingActorAction::Fault)
            } => {
                match fault {
                    Some(fault) => SlotInput::Fault(fault),
                    None => SlotInput::PoolUnavailable,
                }
            }
            _ = async {
                let Some(pool_cancel) = pool_cancel.as_ref() else {
                    return std::future::pending().await;
                };
                pool_cancel.cancelled().await
            } => SlotInput::PoolUnavailable,
            _ = diagnostic_tick(), if runtime.is_some() => SlotInput::Health,
            _ = tokio::time::sleep(retry_delay), if runtime.is_none() && desired_binding.is_some() => SlotInput::Retry,
        };
        match input {
            SlotInput::Metadata | SlotInput::Source | SlotInput::Health => {}
            #[cfg(feature = "kv-dc-relay-wan")]
            SlotInput::Instance => {
                topology.replace_availability(
                    endpoint.clone(),
                    slot_incarnation,
                    instance_rx
                        .as_ref()
                        .map(EndpointAvailabilityWatch::live_workers),
                );
            }
            #[cfg(feature = "kv-dc-relay-wan")]
            SlotInput::AvailabilityRetry => {
                instance_rx = None;
                topology.replace_availability(endpoint.clone(), slot_incarnation, None);
                availability_retry_delay = availability_retry_delay
                    .saturating_mul(2)
                    .min(Duration::from_secs(5));
            }
            SlotInput::SourceClosed => {
                tracing::debug!(%endpoint, "KV source membership watch closed; rebinding");
                config_tx = None;
                source_watch = None;
            }
            SlotInput::Retry => {
                retry_delay = retry_delay.saturating_mul(2).min(Duration::from_secs(5));
            }
            SlotInput::PoolUnavailable => {
                tracing::warn!(%endpoint, "KV DC Relay pool generation became unavailable; restarting pool actor");
                status.write().await.lifecycle = SlotLifecycle::Fenced;
                if let Some(active) = runtime.take() {
                    fence_endpoint_pool(active, &pools).await;
                }
                pending_faults.clear();
                let mut current = status.write().await;
                current.actor = None;
            }
            SlotInput::Fault(action) => {
                let mut retirement_mode = None;
                match action {
                    PendingActorAction::ProducerFence(trigger) => {
                        report_producer_fence_trigger(&endpoint, &trigger);
                        retirement_mode = Some(PoolRetirementMode::Fenced);
                    }
                    PendingActorAction::Fault(fault) => {
                        report_actor_fault(&endpoint, &fault);
                        match fault.disposition.action {
                            CkfFailureAction::ContinueCapacityOmission => {}
                            CkfFailureAction::ReportResourceFailure => {
                                if let Some(active) = runtime.as_mut() {
                                    let client = active.recovery.client().clone();
                                    match collect_pending_while(
                                        &mut active.attachment.faults,
                                        &mut pending_faults,
                                        client.handle_target_fault(
                                            fault.worker_id,
                                            fault.dp_rank,
                                            fault.publisher_id,
                                            false,
                                        ),
                                    )
                                    .await
                                    {
                                        FaultCollection::Completed(disposition) => {
                                            retirement_mode =
                                                target_fault_retirement_mode(disposition);
                                        }
                                        FaultCollection::ProducerFence(trigger) => {
                                            report_producer_fence_trigger(&endpoint, &trigger);
                                            retirement_mode = Some(PoolRetirementMode::Fenced);
                                        }
                                    }
                                }
                            }
                            CkfFailureAction::RejectSource => {
                                if let Some(active) = runtime.as_mut() {
                                    let client = active.recovery.client().clone();
                                    if let FaultCollection::ProducerFence(trigger) =
                                        collect_pending_while(
                                            &mut active.attachment.faults,
                                            &mut pending_faults,
                                            client.reject_source(
                                                fault.worker_id,
                                                fault.dp_rank,
                                                fault.publisher_id,
                                            ),
                                        )
                                        .await
                                    {
                                        report_producer_fence_trigger(&endpoint, &trigger);
                                        retirement_mode = Some(PoolRetirementMode::Fenced);
                                    }
                                }
                            }
                            CkfFailureAction::FenceAndRebuildProducer => {
                                retirement_mode = Some(PoolRetirementMode::Fenced);
                            }
                            CkfFailureAction::DeactivateAndSnapshot
                            | CkfFailureAction::RetrySnapshot => {
                                unreachable!(
                                    "consumer-lane disposition cannot originate from Relay actor"
                                )
                            }
                        }
                    }
                }
                if let Some(mode) = retirement_mode {
                    status.write().await.lifecycle = SlotLifecycle::Fenced;
                    if let Some(active) = runtime.take() {
                        retire_endpoint_pool(active, &pools, mode).await;
                    }
                    pending_faults.clear();
                    status.write().await.actor = None;
                }
            }
            SlotInput::Cancelled => break,
        }

        #[cfg(feature = "ckf-diagnostics")]
        if let Some(active) = &runtime {
            status.write().await.recovery = active.recovery.client().health_snapshot().await;
        }
    }

    if let Some(active) = runtime {
        status.write().await.lifecycle = SlotLifecycle::Draining;
        stop_endpoint_pool(active, &pools).await;
    }
    let mut current = status.write().await;
    current.actor = None;
    current.lifecycle = SlotLifecycle::Lightweight;
}

#[cfg(feature = "kv-dc-relay-wan")]
async fn instance_availability_watch(
    component: &Component,
    endpoint: &EndpointId,
) -> anyhow::Result<EndpointAvailabilityWatch> {
    let endpoint = component
        .drt()
        .namespace(&endpoint.namespace)?
        .component(&endpoint.component)?
        .endpoint(&endpoint.name);
    let client = endpoint.client().await?;
    Ok(EndpointAvailabilityWatch {
        routable: client.instance_avail_watcher(),
        discovered: client.instance_source.as_ref().clone(),
    })
}

async fn diagnostic_tick() {
    #[cfg(feature = "ckf-diagnostics")]
    tokio::time::sleep(Duration::from_secs(1)).await;
    #[cfg(not(feature = "ckf-diagnostics"))]
    std::future::pending::<()>().await;
}

#[allow(clippy::too_many_arguments)]
async fn start_endpoint_pool(
    component: Component,
    ckf_dc_id: dynamo_kv_router::DcId,
    endpoint: EndpointId,
    binding: ActorBinding,
    registrations: Vec<CanonicalModelRegistration>,
    roles: Vec<WorkerRole>,
    #[cfg(feature = "kv-dc-relay-wan")] wan_facts: Option<PoolWanFacts>,
    membership_watch: KvSourceMembershipWatch,
    rebuild_permit: Arc<Semaphore>,
    recovery_fetch_permit: Arc<Semaphore>,
    pools: Arc<PoolRegistry>,
    recovery_attempt_timeout: Duration,
    cancel: CancellationToken,
) -> anyhow::Result<EndpointPoolRuntime> {
    let attachment = pools
        .attach(PoolAttachRequest {
            pool_id: PoolId::new(binding.domain.id, ckf_dc_id),
            endpoint: endpoint.clone(),
            registrations: registrations.clone(),
            query_semantics: binding.domain.query_semantics,
            roles: roles.clone(),
            #[cfg(feature = "kv-dc-relay-wan")]
            wan_facts: wan_facts.clone(),
        })
        .await?;
    let initial_recoveries = membership_watch
        .borrow()
        .sources
        .iter()
        .filter_map(|(worker, status)| {
            status
                .active_source()
                .is_some_and(|source| source.recovery_target.is_some())
                .then_some(*worker)
        })
        .collect();
    let target = KvDcRelayRecoveryTarget::new(
        attachment.handle.clone(),
        rebuild_permit,
        initial_recoveries,
        recovery_attempt_timeout,
    );
    #[cfg(feature = "kv-dc-relay-wan")]
    let load_cancel = wan_facts.as_ref().map(|_| cancel.child_token());
    let recovery = match start_target_subscriber(
        component.clone(),
        endpoint.clone(),
        target,
        membership_watch,
        "kv-dc-relay".to_string(),
        "kv_dc_relay",
        recovery_fetch_permit,
        recovery_attempt_timeout,
        cancel,
    )
    .await
    {
        Ok(recovery) => recovery,
        Err(error) => {
            // `detach` owns the actor fault receiver and is cancellation-sensitive; this
            // endpoint-slot task is joined by the host, so keep the await inline and drive it to
            // completion before returning.
            let _ = pools.detach(attachment).await;
            return Err(error);
        }
    };
    #[cfg(feature = "kv-dc-relay-wan")]
    let wan = wan_facts.map(|facts| {
        let load_cancel = load_cancel.expect("WAN facts create a load cancellation token");
        let load_task = start_load_collector(
            component,
            endpoint,
            attachment.pool_id,
            attachment.layout_generation,
            pools,
            load_cancel.clone(),
        );
        EndpointWanRuntime {
            load_runtime_configs: facts.runtime_configs,
            load_cancel,
            load_task,
        }
    });
    Ok(EndpointPoolRuntime {
        attachment,
        recovery,
        binding,
        registrations,
        roles,
        #[cfg(feature = "kv-dc-relay-wan")]
        wan,
    })
}

async fn stop_endpoint_pool(active: EndpointPoolRuntime, pools: &PoolRegistry) {
    retire_endpoint_pool(active, pools, PoolRetirementMode::Graceful).await;
}

async fn fence_endpoint_pool(active: EndpointPoolRuntime, pools: &PoolRegistry) {
    retire_endpoint_pool(active, pools, PoolRetirementMode::Fenced).await;
}

async fn retire_endpoint_pool(
    active: EndpointPoolRuntime,
    pools: &PoolRegistry,
    mode: PoolRetirementMode,
) {
    let EndpointPoolRuntime {
        attachment,
        recovery,
        binding,
        #[cfg(feature = "kv-dc-relay-wan")]
        wan,
        ..
    } = active;
    let PoolAttachment {
        pool_id,
        layout_generation,
        handle,
        mut faults,
        ..
    } = attachment;
    let actor_teardown = async {
        match mode {
            PoolRetirementMode::Graceful => {
                recovery.shutdown().await;
                handle.shutdown().await
            }
            PoolRetirementMode::Fenced => {
                let ((), result) = tokio::join!(recovery.shutdown(), handle.fence());
                result
            }
        }
    };
    #[cfg(feature = "kv-dc-relay-wan")]
    let teardown = async {
        if let Some(wan) = wan {
            wan.load_cancel.cancel();
            let (result, load_result) = tokio::join!(actor_teardown, wan.load_task);
            if let Err(error) = load_result
                && !error.is_cancelled()
            {
                tracing::warn!(%error, %pool_id, "KV DC Relay load collector failed during pool retirement");
            }
            result
        } else {
            actor_teardown.await
        }
    };
    #[cfg(not(feature = "kv-dc-relay-wan"))]
    let teardown = actor_teardown;
    let result = withdraw_drain_and_remove_pool(
        pools,
        pool_id,
        layout_generation,
        mode,
        &mut faults,
        teardown,
    )
    .await;
    if let Err(error) = result {
        tracing::warn!(
            %error,
            %pool_id,
            endpoint = %binding.kv_state_endpoint,
            ?mode,
            "Failed to retire KV DC Relay pool actor"
        );
    }
}

async fn withdraw_drain_and_remove_pool<T>(
    pools: &PoolRegistry,
    pool_id: PoolId,
    layout_generation: u64,
    mode: PoolRetirementMode,
    faults: &mut tokio::sync::mpsc::Receiver<ActorFault>,
    teardown: impl Future<Output = T>,
) -> T {
    if !pools.withdraw(pool_id, layout_generation, mode).await {
        tracing::warn!(
            %pool_id,
            layout_generation,
            ?mode,
            "KV DC Relay pool generation was already absent during retirement"
        );
    }
    let result = drain_faults_while(pool_id, faults, teardown).await;
    pools.remove(pool_id, layout_generation).await;
    result
}

#[cfg(feature = "kv-dc-relay-wan")]
fn start_load_collector(
    component: Component,
    endpoint: EndpointId,
    pool_id: PoolId,
    layout_generation: u64,
    pools: Arc<PoolRegistry>,
    cancel: CancellationToken,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let collector_cancel = cancel.clone();
        let collector_pools = pools.clone();
        let collector = tokio::spawn(run_load_collector(
            component,
            endpoint,
            pool_id,
            layout_generation,
            collector_pools,
            collector_cancel,
        ));
        let result = collector.await;
        if cancel.is_cancelled() {
            return;
        }
        let reason = match result {
            Ok(()) => "ActiveLoad collector stopped unexpectedly".to_string(),
            Err(error) => format!("ActiveLoad collector task failed: {error}"),
        };
        pools.fence_generation(pool_id, layout_generation, &reason);
    })
}

enum FaultCollection<T> {
    Completed(T),
    /// The source-local future was dropped; the caller must retire the producer generation.
    ProducerFence(ProducerFenceTrigger),
}

async fn collect_pending_while<T>(
    receiver: &mut tokio::sync::mpsc::Receiver<ActorFault>,
    pending: &mut PendingActorFaults,
    future: impl Future<Output = T>,
) -> FaultCollection<T> {
    tokio::pin!(future);
    loop {
        tokio::select! {
            result = &mut future => return FaultCollection::Completed(result),
            item = receiver.recv() => match item {
                Some(fault) => {
                    pending.push(fault);
                    if let Some(trigger) = pending.take_producer_fence() {
                        return FaultCollection::ProducerFence(trigger);
                    }
                }
                None => return FaultCollection::Completed(future.await),
            },
        }
    }
}

#[cfg(feature = "kv-dc-relay-wan")]
async fn run_load_collector(
    component: Component,
    endpoint: EndpointId,
    pool_id: PoolId,
    layout_generation: u64,
    pools: Arc<PoolRegistry>,
    cancel: CancellationToken,
) {
    let mut retry = LoadRetryBackoff::default();
    loop {
        let subscriber =
            EventSubscriber::for_endpoint_id(component.drt(), &endpoint, KV_METRICS_SUBJECT).await;
        let mut subscriber = match subscriber {
            Ok(subscriber) => subscriber.typed::<ActiveLoad>(),
            Err(error) => {
                let failure = retry.failed();
                if failure.first {
                    tracing::warn!(%endpoint, %pool_id, layout_generation, %error, retry_ms = failure.delay.as_millis(), "Failed to subscribe to KV DC Relay ActiveLoad stream");
                } else {
                    tracing::debug!(%endpoint, %pool_id, layout_generation, %error, failures = failure.count, retry_ms = failure.delay.as_millis(), "KV DC Relay ActiveLoad subscription retry failed");
                }
                tokio::select! {
                    _ = cancel.cancelled() => return,
                    _ = tokio::time::sleep(failure.delay) => {}
                }
                continue;
            }
        };
        loop {
            let event = tokio::select! {
                _ = cancel.cancelled() => return,
                event = subscriber.next() => event,
            };
            match event {
                Some(Ok((_envelope, load))) => {
                    retry.succeeded();
                    if !pools.observe_load(pool_id, layout_generation, load) {
                        tracing::debug!(%endpoint, %pool_id, layout_generation, "Ignoring ActiveLoad outside the pool generation's expected ranks");
                    }
                }
                Some(Err(error)) => {
                    pools.clear_load_observations(pool_id, layout_generation);
                    let failure = retry.failed();
                    if failure.first {
                        tracing::warn!(%endpoint, %pool_id, layout_generation, %error, retry_ms = failure.delay.as_millis(), "KV DC Relay ActiveLoad stream failed; resubscribing");
                    } else {
                        tracing::debug!(%endpoint, %pool_id, layout_generation, %error, failures = failure.count, retry_ms = failure.delay.as_millis(), "KV DC Relay ActiveLoad stream failed again; resubscribing");
                    }
                    tokio::select! {
                        _ = cancel.cancelled() => return,
                        _ = tokio::time::sleep(failure.delay) => {}
                    }
                    break;
                }
                None => {
                    pools.clear_load_observations(pool_id, layout_generation);
                    let failure = retry.failed();
                    if failure.first {
                        tracing::warn!(%endpoint, %pool_id, layout_generation, retry_ms = failure.delay.as_millis(), "KV DC Relay ActiveLoad stream closed; resubscribing");
                    } else {
                        tracing::debug!(%endpoint, %pool_id, layout_generation, failures = failure.count, retry_ms = failure.delay.as_millis(), "KV DC Relay ActiveLoad stream closed again; resubscribing");
                    }
                    tokio::select! {
                        _ = cancel.cancelled() => return,
                        _ = tokio::time::sleep(failure.delay) => {}
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(feature = "kv-dc-relay-wan")]
const LOAD_RETRY_INITIAL: Duration = Duration::from_millis(100);
#[cfg(feature = "kv-dc-relay-wan")]
const LOAD_RETRY_MAX: Duration = Duration::from_secs(5);

#[cfg(feature = "kv-dc-relay-wan")]
struct LoadRetryBackoff {
    failures: u64,
    next_delay: Duration,
}

#[cfg(feature = "kv-dc-relay-wan")]
impl Default for LoadRetryBackoff {
    fn default() -> Self {
        Self {
            failures: 0,
            next_delay: LOAD_RETRY_INITIAL,
        }
    }
}

#[cfg(feature = "kv-dc-relay-wan")]
impl LoadRetryBackoff {
    fn failed(&mut self) -> LoadRetryFailure {
        self.failures = self.failures.saturating_add(1);
        let failure = LoadRetryFailure {
            count: self.failures,
            delay: self.next_delay,
            first: self.failures == 1,
        };
        self.next_delay = self.next_delay.saturating_mul(2).min(LOAD_RETRY_MAX);
        failure
    }

    fn succeeded(&mut self) {
        self.failures = 0;
        self.next_delay = LOAD_RETRY_INITIAL;
    }
}

#[cfg(feature = "kv-dc-relay-wan")]
struct LoadRetryFailure {
    count: u64,
    delay: Duration,
    first: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegistrationRefresh {
    Skipped,
    Published,
}

async fn refresh_pool_registrations(
    pools: &PoolRegistry,
    attachment: &mut PoolAttachment,
    active_binding: &ActorBinding,
    desired_binding: Option<&ActorBinding>,
    source_binding_pending: bool,
    desired_registrations: &[CanonicalModelRegistration],
) -> anyhow::Result<RegistrationRefresh> {
    if source_binding_pending || Some(active_binding) != desired_binding {
        return Ok(RegistrationRefresh::Skipped);
    }

    pools
        .replace_registrations(attachment, desired_registrations.to_vec())
        .await?;
    Ok(RegistrationRefresh::Published)
}

fn target_fault_retirement_mode(disposition: TargetFaultDisposition) -> Option<PoolRetirementMode> {
    (disposition == TargetFaultDisposition::Fenced).then_some(PoolRetirementMode::Fenced)
}

#[cfg(feature = "ckf-diagnostics")]
async fn endpoint_stats(
    endpoint: EndpointId,
    status: SharedEndpointStatus,
) -> Result<KvDcRelayEndpointStats, KvDcRelayError> {
    let status = status.read().await.clone();
    let actor_stats = status.actor.as_ref().map(actor_health).unwrap_or_default();
    let (aggregation, publication, memory) = if let Some(actor) = &status.actor {
        let (stats, sequence, members) = actor.state_stats().await?;
        let aggregation = stats.aggregation();
        let publication = stats.publication();
        let memory = stats.memory();
        (
            Some(KvDcRelayAggregationStats {
                members: members
                    .into_iter()
                    .map(|(source, blocks)| KvDcRelayMemberStats {
                        worker_id: source.worker_id,
                        dp_rank: source.dp_rank,
                        blocks,
                    })
                    .collect(),
                contribution_count: aggregation.contribution_count(),
                unique_block_count: aggregation.unique_block_count(),
                unknown_removals: aggregation.unknown_removals(),
                parent_not_found: aggregation.parent_not_found(),
                capacity_failures: aggregation.capacity_failures(),
                occupied_bucket_count: aggregation.occupied_bucket_count(),
                occupied_slot_count: aggregation.occupied_slot_count(),
            }),
            Some(KvDcRelayPublicationStats {
                sequence,
                pending_events: publication.pending_events(),
                publication_count: actor
                    .diagnostics
                    .0
                    .counters
                    .publications
                    .load(Ordering::Relaxed),
                unchanged_publication_count: actor
                    .diagnostics
                    .0
                    .counters
                    .unchanged_publications
                    .load(Ordering::Relaxed),
                physical_touches: publication.physical_touches(),
                distinct_touched_buckets: publication.distinct_touched_buckets(),
                emitted_images: publication.emitted_images(),
                net_reverted_buckets: publication.net_reverted_buckets(),
                reset_count: 0,
            }),
            Some(KvDcRelayMemoryStats {
                filter_bytes: memory.filter_bytes(),
                dirty_tracking_bytes: memory.dirty_tracking_bytes(),
                source_lineage_capacity: memory.source_lineage_capacity(),
                canonical_owner_capacity: memory.canonical_owner_capacity(),
                insertion_scratch_capacity: memory.insertion_scratch_capacity(),
            }),
        )
    } else {
        (None, None, None)
    };
    let membership = status.membership;
    Ok(KvDcRelayEndpointStats {
        serving_endpoint: endpoint.to_string(),
        lifecycle: status.lifecycle.as_str().to_string(),
        layout_generation: status.layout_generation,
        cache_domain: membership
            .as_ref()
            .and_then(|membership| membership.domain.as_ref())
            .map(cache_domain_stats),
        membership_conflicts: membership
            .as_ref()
            .map(|membership| {
                membership
                    .conflicts
                    .iter()
                    .map(|conflict| format!("{conflict:?}"))
                    .collect()
            })
            .unwrap_or_default(),
        models: membership
            .as_ref()
            .map(|membership| membership.models.clone())
            .unwrap_or_default(),
        aliases: membership
            .as_ref()
            .map(|membership| membership.aliases.clone())
            .unwrap_or_default(),
        roles: membership
            .as_ref()
            .map(|membership| {
                membership
                    .roles
                    .iter()
                    .map(|role| role.as_str().to_string())
                    .collect()
            })
            .unwrap_or_default(),
        aggregation,
        publication,
        recovery: KvDcRelayRecoveryStats {
            degraded_resets: status.actor.as_ref().map_or(0, |actor| {
                actor
                    .diagnostics
                    .0
                    .counters
                    .degraded_resets
                    .load(Ordering::Relaxed)
            }),
            rebuild_count: status.actor.as_ref().map_or(0, |actor| {
                actor
                    .diagnostics
                    .0
                    .counters
                    .rebuild_count
                    .load(Ordering::Relaxed)
            }),
            rebuild_ns: status.actor.as_ref().map_or(0, |actor| {
                actor
                    .diagnostics
                    .0
                    .counters
                    .rebuild_ns
                    .load(Ordering::Relaxed)
            }),
            rebuild_max_ns: status.actor.as_ref().map_or(0, |actor| {
                actor
                    .diagnostics
                    .0
                    .counters
                    .rebuild_max_ns
                    .load(Ordering::Relaxed)
            }),
            worker_count: status.recovery.worker_count,
            rank_count: status.recovery.rank_count,
            recovering_rank_count: status.recovery.recovering_rank_count,
            pending_live_event_count: status.recovery.pending_live_event_count,
            discovered_endpoint_count: status.recovery.discovered_endpoint_count,
        },
        memory,
        actor: actor_stats,
    })
}

#[cfg(feature = "ckf-diagnostics")]
fn cache_domain_stats(domain: &KvCacheDomainKey) -> KvDcRelayCacheDomainStats {
    KvDcRelayCacheDomainStats {
        model_artifact: domain.diagnostic_model_artifact.clone(),
        kv_block_size: domain.query_semantics.kv_block_size(),
        event_hash_format: domain.query_semantics.hash_format().identity_version(),
    }
}

#[cfg(feature = "ckf-diagnostics")]
fn actor_health(handle: &KvDcRelayHandle) -> KvDcRelayActorStats {
    let activity = handle.diagnostics.0.activity.lock();
    KvDcRelayActorStats {
        mailbox_depth: handle.mailbox_depth(),
        mailbox_capacity: handle.mailbox_capacity(),
        mailbox_wait_ns: handle
            .diagnostics
            .0
            .counters
            .mailbox_wait_ns
            .load(Ordering::Relaxed),
        mailbox_max_wait_ns: handle
            .diagnostics
            .0
            .counters
            .mailbox_max_wait_ns
            .load(Ordering::Relaxed),
        active_command: activity.active_command.map(str::to_string),
        active_command_age_ms: activity
            .active_since
            .map(|started| started.elapsed().as_millis().min(u64::MAX as u128) as u64),
        shutting_down: activity.shutting_down,
        faulted: activity.last_error.is_some(),
        last_error: activity.last_error.clone(),
    }
}
#[cfg(test)]
mod tests {
    use dynamo_kv_router::{
        identity::{CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, RoutingScopeId},
        indexer::cuckoo::{CkfFailureDisposition, CkfFailurePoint},
        protocols::{KV_EVENT_SUBJECT, WorkerWithDpRank},
    };
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::{DiscoveryInstance, DiscoverySpec, EventTransportKind},
        distributed::DistributedConfig,
        transports::event_plane::{EventPublisher, EventScope},
    };

    use super::super::actor::ActorFaultCategory;
    use super::*;
    use crate::discovery::{KvEventSource, KvSourceMembership};

    fn membership(endpoint: &str, domain: KvCacheDomainKey) -> EndpointMembership {
        let endpoint = EndpointId::from(endpoint);
        let namespace = endpoint.namespace.clone();
        EndpointMembership {
            endpoint,
            generation: 1,
            domain: Some(domain),
            namespace,
            registrations: vec![CanonicalModelRegistration::new(
                super::super::identity::CanonicalModelId::new("llama").unwrap(),
                Vec::new(),
            )],
            models: vec!["llama".to_string()],
            aliases: Vec::new(),
            roles: vec![WorkerRole::Aggregated],
            runtime_configs: HashMap::new(),
            worker_topology: HashMap::new(),
            adapters: HashMap::new(),
            conflicts: Vec::new(),
        }
    }

    fn domain(seed: u8, artifact: &str) -> KvCacheDomainKey {
        KvCacheDomainKey {
            id: IndexerDomainId::new(
                CacheSemanticsId::new([seed; 16], IdentitySource::Explicit),
                RoutingScopeId::new([seed.wrapping_add(1); 16], IdentitySource::Explicit),
            ),
            diagnostic_model_artifact: artifact.to_string(),
            query_semantics: query_semantics(),
        }
    }

    fn query_semantics() -> super::super::identity::KvQuerySemantics {
        super::super::identity::KvQuerySemantics::new(
            64,
            super::super::identity::KvQueryHashFormat::DynamoStandardV1,
        )
        .unwrap()
    }

    fn registry() -> PoolRegistry {
        PoolRegistry::new(
            DcRelayIdentity::new(11, 7),
            PoolActorConfig {
                expected_unique_blocks: 32,
                publication_threshold: 1,
                publication_delay: Duration::from_millis(1),
            },
        )
    }

    fn actor_fault(
        worker_id: WorkerId,
        dp_rank: DpRank,
        publisher_id: u64,
        event_id: u64,
        disposition: CkfFailureDisposition,
    ) -> ActorFault {
        let category = match disposition.action {
            CkfFailureAction::ReportResourceFailure => ActorFaultCategory::Resource,
            CkfFailureAction::RejectSource => ActorFaultCategory::SourceProtocol,
            CkfFailureAction::ContinueCapacityOmission
            | CkfFailureAction::FenceAndRebuildProducer => ActorFaultCategory::ProducerInvariant,
            CkfFailureAction::DeactivateAndSnapshot | CkfFailureAction::RetrySnapshot => {
                unreachable!("consumer fault cannot be used by Relay host tests")
            }
        };
        ActorFault {
            worker_id,
            dp_rank,
            publisher_id,
            event_id: Some(event_id),
            category,
            disposition,
            message: format!("fault {event_id}"),
        }
    }

    async fn test_component(name: &str) -> Component {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        drt.namespace(format!("kv-dc-relay-{name}"))
            .unwrap()
            .component("relay")
            .unwrap()
    }

    async fn register_live_source(
        component: &Component,
        endpoint: &EndpointId,
        worker: WorkerWithDpRank,
    ) -> (EventPublisher, DiscoveryInstance) {
        let publisher = EventPublisher::for_endpoint_id_with_transport(
            component.drt(),
            endpoint,
            KV_EVENT_SUBJECT,
            EventTransportKind::Zmq,
        )
        .await
        .unwrap();
        let source = crate::discovery::KvEventSource {
            kv_state_endpoint: endpoint.clone(),
            worker,
            publisher_id: publisher.publisher_id(),
            recovery_target: None,
        };
        let instance = component
            .drt()
            .discovery()
            .register(DiscoverySpec::EventSource {
                scope: EventScope::Endpoint {
                    endpoint: endpoint.clone(),
                },
                topic: KV_EVENT_SUBJECT.to_string(),
                publisher_id: publisher.publisher_id(),
                metadata: serde_json::to_value(source).unwrap(),
            })
            .await
            .unwrap();
        (publisher, instance)
    }

    fn projected_membership(
        endpoint: &EndpointId,
        worker_id: WorkerId,
        model: &str,
        artifact: &str,
        kv_state_endpoint: EndpointId,
    ) -> EndpointMembership {
        projected_membership_with_metadata(
            endpoint,
            worker_id,
            model,
            artifact,
            kv_state_endpoint,
            None,
            Vec::new(),
            None,
        )
    }

    fn projected_membership_with_role(
        endpoint: &EndpointId,
        worker_id: WorkerId,
        model: &str,
        artifact: &str,
        kv_state_endpoint: EndpointId,
        worker_type: crate::worker_type::WorkerType,
    ) -> EndpointMembership {
        projected_membership_with_metadata(
            endpoint,
            worker_id,
            model,
            artifact,
            kv_state_endpoint,
            Some(worker_type),
            Vec::new(),
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn projected_membership_with_metadata(
        endpoint: &EndpointId,
        worker_id: WorkerId,
        model: &str,
        artifact: &str,
        kv_state_endpoint: EndpointId,
        worker_type: Option<crate::worker_type::WorkerType>,
        aliases: Vec<String>,
        context_length: Option<u32>,
    ) -> EndpointMembership {
        let mut card = crate::model_card::ModelDeploymentCard::with_name_only(model);
        card.source_path = Some(artifact.to_string());
        card.kv_cache_block_size = 64;
        card.worker_type = worker_type;
        card.aliases = aliases;
        card.runtime_config = ModelRuntimeConfig {
            context_length,
            data_parallel_start_rank: 0,
            data_parallel_size: 1,
            enable_local_indexer: true,
            kv_state_endpoint: Some(kv_state_endpoint),
            ..ModelRuntimeConfig::default()
        };
        let instance = DiscoveryInstance::Model {
            namespace: endpoint.namespace.clone(),
            component: endpoint.component.clone(),
            endpoint: endpoint.name.clone(),
            instance_id: worker_id,
            card_json: serde_json::to_value(card).unwrap(),
            model_suffix: None,
        };
        super::super::discovery::project_instances_for_test(vec![instance])
            .endpoints
            .get(endpoint)
            .cloned()
            .unwrap()
    }

    async fn wait_for_catalog(
        receiver: &mut watch::Receiver<DcPoolCatalog>,
        predicate: impl Fn(&DcPoolCatalog) -> bool,
    ) -> DcPoolCatalog {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let catalog = receiver.borrow_and_update().clone();
                if predicate(&catalog) {
                    return catalog;
                }
                receiver.changed().await.unwrap();
            }
        })
        .await
        .expect("Relay catalog transition timed out")
    }

    async fn wait_for_slot_settled(
        status: &SharedEndpointStatus,
        membership_generation: u64,
    ) -> EndpointSlotStatus {
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let current = status.read().await.clone();
                if current.settled_membership_generation == Some(membership_generation) {
                    return current;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("endpoint slot did not settle")
    }

    #[tokio::test]
    async fn departed_endpoint_slots_are_cancelled_before_reaping() {
        let endpoint = membership("prod.backend.generate", domain(1, "meta/llama")).endpoint;
        let (metadata, _metadata_rx) = watch::channel(None);
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let task = tokio::spawn(async move { task_cancel.cancelled().await });
        let mut slots = HashMap::from([(
            endpoint.clone(),
            EndpointSlotTask {
                metadata,
                status: Arc::new(RwLock::new(EndpointSlotStatus::default())),
                cancel: cancel.clone(),
                task,
            },
        )]);
        let mut retired_slots = JoinSet::new();

        retire_departed_endpoint_slots(
            &DcMembershipView::default(),
            &mut slots,
            &mut retired_slots,
        );

        assert!(slots.is_empty());
        assert!(cancel.is_cancelled());
        let (retired_slot, result) = retired_slots.join_next().await.unwrap().unwrap();
        assert_eq!(retired_slot, endpoint);
        result.unwrap();
    }

    #[tokio::test]
    async fn repeated_relay_start_separates_drt_identity_from_relay_incarnation() {
        let component = test_component("incarnation").await;
        let first = KvDcRelay::start(
            component.clone(),
            "test-dc".to_string(),
            KvDcRelayConfig::default(),
        )
        .await
        .unwrap();
        #[cfg(feature = "kv-dc-relay-wan")]
        {
            let health = first.health().await;
            assert!(!health.wan_enabled);
            assert!(!health.wan_serving);
        }
        let first_identity = first.pool_catalog().identity();
        first.shutdown().await.unwrap();

        let second = KvDcRelay::start(component, "test-dc".to_string(), KvDcRelayConfig::default())
            .await
            .unwrap();
        let second_identity = second.pool_catalog().identity();
        second.shutdown().await.unwrap();

        assert_eq!(
            first_identity.drt_instance_id(),
            second_identity.drt_instance_id()
        );
        assert_ne!(
            first_identity.relay_incarnation(),
            second_identity.relay_incarnation()
        );
    }

    #[tokio::test]
    async fn unexpected_host_return_cancels_relay_and_records_terminal_error() {
        let cancel = CancellationToken::new();
        let terminal = Arc::new(HostTerminalState::default());
        let pools = Arc::new(registry());

        supervise_host_task(
            tokio::spawn(async { Ok(()) }),
            cancel.clone(),
            terminal.clone(),
            pools.clone(),
        )
        .await;

        let relay = KvDcRelay {
            #[cfg(feature = "ckf-diagnostics")]
            dc_id: Arc::from("test-dc"),
            #[cfg(any(feature = "ckf-diagnostics", feature = "kv-dc-relay-wan"))]
            relay_identity: DcRelayIdentity::new(11, 7),
            cancel,
            membership: Mutex::new(None),
            supervisor: Mutex::new(None),
            terminal,
            statuses: Arc::new(RwLock::new(HashMap::new())),
            pools,
            #[cfg(feature = "kv-dc-relay-wan")]
            transport: Mutex::new(None),
        };
        tokio::time::timeout(Duration::from_millis(100), relay.wait_for_shutdown())
            .await
            .expect("host failure must wake Relay shutdown waiters");
        let health = relay.health().await;
        assert!(!health.healthy);
        assert_eq!(
            health.host_last_error.as_deref(),
            Some("KV DC Relay host stopped unexpectedly")
        );
    }

    #[tokio::test]
    async fn host_panic_cancels_relay_and_preserves_the_reason() {
        let cancel = CancellationToken::new();
        let terminal = Arc::new(HostTerminalState::default());
        let pools = Arc::new(registry());
        let host = tokio::spawn(async {
            panic!("injected host panic");
            #[allow(unreachable_code)]
            Ok(())
        });

        supervise_host_task(host, cancel.clone(), terminal.clone(), pools).await;

        assert!(cancel.is_cancelled());
        let reason = terminal.last_error().expect("terminal host reason");
        assert!(reason.contains("injected host panic"));
    }

    #[tokio::test]
    async fn normal_host_cancellation_records_no_terminal_error() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let terminal = Arc::new(HostTerminalState::default());
        supervise_host_task(
            tokio::spawn(async { Ok(()) }),
            cancel,
            terminal.clone(),
            Arc::new(registry()),
        )
        .await;

        assert!(terminal.last_error().is_none());
    }

    #[tokio::test]
    async fn duplicate_pool_owners_are_all_fenced_and_make_health_unhealthy() {
        let domain = domain(1, "meta/llama");
        let first = membership("prod.backend.fast", domain.clone());
        let second = membership("prod.backend.slow", domain);
        let mut view = DcMembershipView {
            endpoints: Arc::new(HashMap::from([
                (first.endpoint.clone(), first),
                (second.endpoint.clone(), second),
            ])),
        };

        reject_duplicate_live_pools(&mut view, DcId::new(7));
        assert!(view.endpoints.values().all(|membership| {
            !membership.conflicts.is_empty()
                && inactive_slot_lifecycle(Some(membership)) == SlotLifecycle::Fenced
        }));

        let statuses = Arc::new(RwLock::new(
            view.endpoints
                .iter()
                .map(|(endpoint, membership)| {
                    let status = EndpointSlotStatus {
                        lifecycle: inactive_slot_lifecycle(Some(membership)),
                        membership: Some(membership.clone()),
                        ..EndpointSlotStatus::default()
                    };
                    (endpoint.clone(), Arc::new(RwLock::new(status)))
                })
                .collect(),
        ));
        let relay = KvDcRelay {
            #[cfg(feature = "ckf-diagnostics")]
            dc_id: Arc::from("test-dc"),
            #[cfg(any(feature = "ckf-diagnostics", feature = "kv-dc-relay-wan"))]
            relay_identity: DcRelayIdentity::new(11, 7),
            cancel: CancellationToken::new(),
            membership: Mutex::new(None),
            supervisor: Mutex::new(None),
            terminal: Arc::new(HostTerminalState::default()),
            statuses,
            pools: Arc::new(registry()),
            #[cfg(feature = "kv-dc-relay-wan")]
            transport: Mutex::new(None),
        };

        let health = relay.health().await;
        assert!(!health.healthy);
        assert_eq!(health.fenced_endpoint_count, 2);
        assert_eq!(health.active_endpoint_count, 0);
    }

    #[tokio::test]
    async fn binding_change_never_publishes_new_model_with_old_producer() {
        let registry = registry();
        let mut catalog_rx = registry.watch_catalog();
        let old_domain = domain(1, "meta/llama");
        let new_domain = domain(3, "meta/llama-v2");
        let old_binding = ActorBinding {
            domain: old_domain.clone(),
            kv_state_endpoint: EndpointId::from("prod.backend.old-kv"),
        };
        let desired_binding = ActorBinding {
            domain: new_domain,
            kv_state_endpoint: EndpointId::from("prod.backend.new-kv"),
        };
        let mut old_attachment = registry
            .attach(PoolAttachRequest {
                pool_id: PoolId::new(old_domain.id, DcId::new(7)),
                endpoint: EndpointId::from("prod.backend.generate"),
                registrations: vec![CanonicalModelRegistration::new(
                    super::super::identity::CanonicalModelId::new("llama").unwrap(),
                    Vec::new(),
                )],
                query_semantics: query_semantics(),
                roles: vec![WorkerRole::Aggregated],
                #[cfg(feature = "kv-dc-relay-wan")]
                wan_facts: Some(PoolWanFacts {
                    runtime_configs: HashMap::new(),
                }),
            })
            .await
            .unwrap();
        catalog_rx.changed().await.unwrap();
        let old_catalog = catalog_rx.borrow_and_update().clone();
        let old_producer = old_catalog.pools()[0].producer();
        let new_registrations = vec![CanonicalModelRegistration::new(
            super::super::identity::CanonicalModelId::new("llama-v2").unwrap(),
            Vec::new(),
        )];

        assert_eq!(
            refresh_pool_registrations(
                &registry,
                &mut old_attachment,
                &old_binding,
                Some(&desired_binding),
                false,
                &new_registrations,
            )
            .await
            .unwrap(),
            RegistrationRefresh::Skipped
        );
        assert!(!catalog_rx.has_changed().unwrap());
        assert_eq!(
            old_catalog.pools()[0].registrations()[0].model().as_str(),
            "llama"
        );

        registry.detach(old_attachment).await.unwrap();
        catalog_rx.changed().await.unwrap();
        let withdrawn_catalog = catalog_rx.borrow_and_update().clone();
        assert!(withdrawn_catalog.pools().is_empty());

        let new_attachment = registry
            .attach(PoolAttachRequest {
                pool_id: PoolId::new(desired_binding.domain.id, DcId::new(7)),
                endpoint: EndpointId::from("prod.backend.generate"),
                registrations: new_registrations,
                query_semantics: query_semantics(),
                roles: vec![WorkerRole::Aggregated],
                #[cfg(feature = "kv-dc-relay-wan")]
                wan_facts: Some(PoolWanFacts {
                    runtime_configs: HashMap::new(),
                }),
            })
            .await
            .unwrap();
        catalog_rx.changed().await.unwrap();
        let new_catalog = catalog_rx.borrow_and_update().clone();
        assert_ne!(new_catalog.pools()[0].producer(), old_producer);
        assert_eq!(
            new_catalog.pools()[0].registrations()[0].model().as_str(),
            "llama-v2"
        );

        for catalog in [&old_catalog, &withdrawn_catalog, &new_catalog] {
            assert!(!catalog.pools().iter().any(|descriptor| {
                descriptor.producer() == old_producer
                    && descriptor
                        .registrations()
                        .iter()
                        .any(|registration| registration.model().as_str() == "llama-v2")
            }));
        }

        registry.detach(new_attachment).await.unwrap();
    }

    #[tokio::test]
    async fn mdc_binding_transition_never_publishes_new_model_with_old_producer() {
        let component = test_component("mdc-transition").await;
        let worker_id = component.drt().connection_id();
        let worker = WorkerWithDpRank::new(worker_id, 0);
        let serving_endpoint = EndpointId::from("relay-test.backend.generate");
        let old_kv_endpoint = EndpointId::from("relay-test.backend.kv-old");
        let new_kv_endpoint = EndpointId::from("relay-test.backend.kv-new");
        let (_old_publisher, _old_instance) =
            register_live_source(&component, &old_kv_endpoint, worker).await;
        let (_new_publisher, _new_instance) =
            register_live_source(&component, &new_kv_endpoint, worker).await;
        let old_membership = projected_membership(
            &serving_endpoint,
            worker_id,
            "llama",
            "meta/llama",
            old_kv_endpoint,
        );
        let new_membership = projected_membership(
            &serving_endpoint,
            worker_id,
            "llama-v2",
            "meta/llama-v2",
            new_kv_endpoint,
        );
        let endpoint = old_membership.endpoint.clone();
        let (metadata_tx, metadata_rx) = watch::channel(Some(old_membership));
        let status = Arc::new(RwLock::new(EndpointSlotStatus::default()));
        let registry = Arc::new(registry());
        let mut catalog_rx = registry.watch_catalog();
        let slot_cancel = CancellationToken::new();
        #[cfg(feature = "kv-dc-relay-wan")]
        let topology = Arc::new(TopologyPublisher::new(
            DcMembershipView::default(),
            &registry.catalog(),
        ));
        let slot = tokio::spawn(run_endpoint_slot(
            component,
            DcId::new(7),
            endpoint,
            metadata_rx,
            status,
            Arc::new(Semaphore::new(1)),
            Arc::new(Semaphore::new(1)),
            registry.clone(),
            Duration::from_secs(1),
            slot_cancel.clone(),
            #[cfg(feature = "kv-dc-relay-wan")]
            1,
            #[cfg(feature = "kv-dc-relay-wan")]
            WanFactsMode::Disabled,
            #[cfg(feature = "kv-dc-relay-wan")]
            topology,
        ));

        let old_catalog = wait_for_catalog(&mut catalog_rx, |catalog| {
            catalog
                .pools()
                .iter()
                .any(|descriptor| descriptor.registrations()[0].model().as_str() == "llama")
        })
        .await;
        let old_producer = old_catalog.pools()[0].producer();

        metadata_tx.send_replace(Some(new_membership));
        let new_catalog =
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    catalog_rx.changed().await.unwrap();
                    let catalog = catalog_rx.borrow_and_update().clone();
                    assert!(!catalog.pools().iter().any(|descriptor| {
                        descriptor.producer() == old_producer
                            && descriptor
                                .registrations()
                                .iter()
                                .any(|registration| registration.model().as_str() == "llama-v2")
                    }));
                    if catalog.pools().iter().any(|descriptor| {
                        descriptor.registrations()[0].model().as_str() == "llama-v2"
                    }) {
                        return catalog;
                    }
                }
            })
            .await
            .expect("replacement Relay catalog generation timed out");
        assert_ne!(new_catalog.pools()[0].producer(), old_producer);

        slot_cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), slot)
            .await
            .expect("endpoint slot shutdown timed out")
            .unwrap();
    }

    #[tokio::test]
    async fn alias_and_context_change_refreshes_the_active_pool_catalog() {
        let component = test_component("metadata-transition").await;
        let worker_id = component.drt().connection_id();
        let worker = WorkerWithDpRank::new(worker_id, 0);
        let serving_endpoint = EndpointId::from("relay-test.backend.generate");
        let kv_endpoint = EndpointId::from("relay-test.backend.kv");
        let (_publisher, _source_instance) =
            register_live_source(&component, &kv_endpoint, worker).await;
        let old_membership = projected_membership_with_metadata(
            &serving_endpoint,
            worker_id,
            "llama",
            "meta/llama",
            kv_endpoint.clone(),
            None,
            vec!["old-alias".to_string()],
            Some(4096),
        );
        let new_membership = projected_membership_with_metadata(
            &serving_endpoint,
            worker_id,
            "llama",
            "meta/llama",
            kv_endpoint,
            None,
            vec!["new-alias".to_string()],
            Some(8192),
        );
        let endpoint = old_membership.endpoint.clone();
        let (metadata_tx, metadata_rx) = watch::channel(Some(old_membership));
        let status = Arc::new(RwLock::new(EndpointSlotStatus::default()));
        let registry = Arc::new(registry());
        let mut catalog_rx = registry.watch_catalog();
        let slot_cancel = CancellationToken::new();
        #[cfg(feature = "kv-dc-relay-wan")]
        let topology = Arc::new(TopologyPublisher::new(
            DcMembershipView::default(),
            &registry.catalog(),
        ));
        let slot = tokio::spawn(run_endpoint_slot(
            component,
            DcId::new(7),
            endpoint,
            metadata_rx,
            status,
            Arc::new(Semaphore::new(1)),
            Arc::new(Semaphore::new(1)),
            registry.clone(),
            Duration::from_secs(1),
            slot_cancel.clone(),
            #[cfg(feature = "kv-dc-relay-wan")]
            1,
            #[cfg(feature = "kv-dc-relay-wan")]
            WanFactsMode::Disabled,
            #[cfg(feature = "kv-dc-relay-wan")]
            topology,
        ));

        let old_catalog = wait_for_catalog(&mut catalog_rx, |catalog| {
            catalog.pools().iter().any(|descriptor| {
                descriptor.registrations()[0]
                    .aliases()
                    .iter()
                    .any(|alias| alias.as_str() == "old-alias")
            })
        })
        .await;
        let producer = old_catalog.pools()[0].producer();
        #[cfg(feature = "kv-dc-relay-wan")]
        assert!(registry.load_snapshots().is_empty());

        metadata_tx.send_replace(Some(new_membership));
        let new_catalog = wait_for_catalog(&mut catalog_rx, |catalog| {
            catalog.pools().iter().any(|descriptor| {
                descriptor.producer() == producer
                    && descriptor.registrations()[0]
                        .aliases()
                        .iter()
                        .any(|alias| alias.as_str() == "new-alias")
            })
        })
        .await;
        assert!(new_catalog.pools().iter().all(|descriptor| {
            descriptor.registrations().iter().all(|registration| {
                registration
                    .aliases()
                    .iter()
                    .all(|alias| alias.as_str() != "old-alias")
            })
        }));

        slot_cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), slot)
            .await
            .expect("endpoint slot shutdown timed out")
            .unwrap();
    }

    #[test]
    fn encode_pool_materializes_only_after_an_active_kv_source_appears() {
        let endpoint = EndpointId::from("production.encoder.generate");
        let cache_domain = domain(1, "vision-language");
        let worker = WorkerWithDpRank::new(1, 0);
        let mut membership = membership("production.encoder.generate", cache_domain.clone());
        membership.roles = vec![WorkerRole::Encode];
        membership
            .runtime_configs
            .insert(worker.worker_id, ModelRuntimeConfig::default());

        let mut sources = KvSourceMembership::new();
        let missing = sources.view(&endpoint, &membership.runtime_configs);
        assert_eq!(desired_actor_binding(&membership, Some(&missing)), None);

        sources
            .add(KvEventSource {
                kv_state_endpoint: endpoint.clone(),
                worker,
                publisher_id: 41,
                recovery_target: None,
            })
            .unwrap();
        let active = sources.view(&endpoint, &membership.runtime_configs);
        assert_eq!(
            desired_actor_binding(&membership, Some(&active)),
            Some(ActorBinding {
                domain: cache_domain,
                kv_state_endpoint: endpoint,
            })
        );
    }

    #[tokio::test]
    async fn endpoint_slot_materializes_and_dematerializes_with_its_kv_source() {
        let component = test_component("source-transition").await;
        let worker_id = component.drt().connection_id();
        let worker = WorkerWithDpRank::new(worker_id, 0);
        let serving_endpoint = EndpointId::from("relay-test.backend.generate");
        let kv_endpoint = EndpointId::from("relay-test.backend.kv");
        let membership = projected_membership(
            &serving_endpoint,
            worker_id,
            "llama",
            "meta/llama",
            kv_endpoint.clone(),
        );
        let membership_generation = membership.generation;
        let endpoint = membership.endpoint.clone();
        let (_metadata_tx, metadata_rx) = watch::channel(Some(membership));
        let status = Arc::new(RwLock::new(EndpointSlotStatus::default()));
        let registry = Arc::new(registry());
        let mut catalog_rx = registry.watch_catalog();
        let slot_cancel = CancellationToken::new();
        #[cfg(feature = "kv-dc-relay-wan")]
        let topology = Arc::new(TopologyPublisher::new(
            DcMembershipView::default(),
            &registry.catalog(),
        ));
        let slot = tokio::spawn(run_endpoint_slot(
            component.clone(),
            DcId::new(7),
            endpoint,
            metadata_rx,
            status.clone(),
            Arc::new(Semaphore::new(1)),
            Arc::new(Semaphore::new(1)),
            registry.clone(),
            Duration::from_secs(1),
            slot_cancel.clone(),
            #[cfg(feature = "kv-dc-relay-wan")]
            1,
            #[cfg(feature = "kv-dc-relay-wan")]
            WanFactsMode::Disabled,
            #[cfg(feature = "kv-dc-relay-wan")]
            topology,
        ));

        let settled = wait_for_slot_settled(&status, membership_generation).await;
        assert_eq!(settled.lifecycle, SlotLifecycle::Discovered);
        assert!(settled.actor.is_none());
        assert!(catalog_rx.borrow_and_update().pools().is_empty());

        let (_publisher, source_instance) =
            register_live_source(&component, &kv_endpoint, worker).await;
        let materialized = wait_for_catalog(&mut catalog_rx, |catalog| {
            catalog
                .pools()
                .iter()
                .any(|descriptor| descriptor.serving_endpoint() == &serving_endpoint)
        })
        .await;
        assert_eq!(materialized.pools().len(), 1);

        component
            .drt()
            .discovery()
            .unregister(source_instance)
            .await
            .unwrap();
        let withdrawn =
            wait_for_catalog(&mut catalog_rx, |catalog| catalog.pools().is_empty()).await;
        assert!(withdrawn.pools().is_empty());

        slot_cancel.cancel();
        tokio::time::timeout(Duration::from_secs(5), slot)
            .await
            .expect("endpoint slot shutdown timed out")
            .unwrap();
    }

    #[tokio::test]
    async fn prefill_and_decode_sources_materialize_and_withdraw_independent_pools() {
        let component = test_component("pd-source-transition").await;
        let prefill_worker_id = component.drt().connection_id();
        let decode_worker_id = prefill_worker_id.wrapping_add(1);
        let prefill_worker = WorkerWithDpRank::new(prefill_worker_id, 0);
        let decode_worker = WorkerWithDpRank::new(decode_worker_id, 0);
        let prefill_endpoint = EndpointId::from("relay-test.prefill.generate");
        let decode_endpoint = EndpointId::from("relay-test.decode.generate");
        let prefill_kv_endpoint = EndpointId::from("relay-test.prefill.kv");
        let decode_kv_endpoint = EndpointId::from("relay-test.decode.kv");
        let (_prefill_publisher, prefill_source) =
            register_live_source(&component, &prefill_kv_endpoint, prefill_worker).await;
        let (_decode_publisher, decode_source) =
            register_live_source(&component, &decode_kv_endpoint, decode_worker).await;
        let prefill_membership = projected_membership_with_role(
            &prefill_endpoint,
            prefill_worker_id,
            "llama",
            "meta/llama",
            prefill_kv_endpoint,
            crate::worker_type::WorkerType::Prefill,
        );
        let decode_membership = projected_membership_with_role(
            &decode_endpoint,
            decode_worker_id,
            "llama",
            "meta/llama",
            decode_kv_endpoint,
            crate::worker_type::WorkerType::Decode,
        );
        let prefill_endpoint_id = prefill_membership.endpoint.clone();
        let decode_endpoint_id = decode_membership.endpoint.clone();
        let (_prefill_metadata_tx, prefill_metadata_rx) = watch::channel(Some(prefill_membership));
        let (_decode_metadata_tx, decode_metadata_rx) = watch::channel(Some(decode_membership));
        let registry = Arc::new(registry());
        let mut catalog_rx = registry.watch_catalog();
        let rebuild_permit = Arc::new(Semaphore::new(2));
        let recovery_fetch_permit = Arc::new(Semaphore::new(2));
        let prefill_cancel = CancellationToken::new();
        let decode_cancel = CancellationToken::new();
        #[cfg(feature = "kv-dc-relay-wan")]
        let topology = Arc::new(TopologyPublisher::new(
            DcMembershipView::default(),
            &registry.catalog(),
        ));
        let prefill_slot = tokio::spawn(run_endpoint_slot(
            component.clone(),
            DcId::new(7),
            prefill_endpoint_id,
            prefill_metadata_rx,
            Arc::new(RwLock::new(EndpointSlotStatus::default())),
            rebuild_permit.clone(),
            recovery_fetch_permit.clone(),
            registry.clone(),
            Duration::from_secs(1),
            prefill_cancel.clone(),
            #[cfg(feature = "kv-dc-relay-wan")]
            1,
            #[cfg(feature = "kv-dc-relay-wan")]
            WanFactsMode::Disabled,
            #[cfg(feature = "kv-dc-relay-wan")]
            topology.clone(),
        ));
        let decode_slot = tokio::spawn(run_endpoint_slot(
            component.clone(),
            DcId::new(7),
            decode_endpoint_id,
            decode_metadata_rx,
            Arc::new(RwLock::new(EndpointSlotStatus::default())),
            rebuild_permit,
            recovery_fetch_permit,
            registry.clone(),
            Duration::from_secs(1),
            decode_cancel.clone(),
            #[cfg(feature = "kv-dc-relay-wan")]
            2,
            #[cfg(feature = "kv-dc-relay-wan")]
            WanFactsMode::Disabled,
            #[cfg(feature = "kv-dc-relay-wan")]
            topology,
        ));

        let materialized = wait_for_catalog(&mut catalog_rx, |catalog| {
            catalog.pools().len() == 2
                && catalog
                    .pools()
                    .iter()
                    .any(|pool| pool.serving_endpoint() == &prefill_endpoint)
                && catalog
                    .pools()
                    .iter()
                    .any(|pool| pool.serving_endpoint() == &decode_endpoint)
        })
        .await;
        let prefill_pool = materialized
            .pools()
            .iter()
            .find(|pool| pool.serving_endpoint() == &prefill_endpoint)
            .unwrap();
        let decode_pool = materialized
            .pools()
            .iter()
            .find(|pool| pool.serving_endpoint() == &decode_endpoint)
            .unwrap();
        for pool in [prefill_pool, decode_pool] {
            assert_eq!(pool.registrations().len(), 1);
            assert_eq!(pool.registrations()[0].model().as_str(), "llama");
        }
        assert_ne!(prefill_pool.pool_id(), decode_pool.pool_id());
        assert_eq!(prefill_pool.pool_roles(), [WorkerRole::Prefill]);
        assert_eq!(decode_pool.pool_roles(), [WorkerRole::Decode]);
        let prefill_pool_id = prefill_pool.pool_id();

        component
            .drt()
            .discovery()
            .unregister(decode_source)
            .await
            .unwrap();
        let decode_withdrawn = wait_for_catalog(&mut catalog_rx, |catalog| {
            catalog.pools().len() == 1 && catalog.pools()[0].serving_endpoint() == &prefill_endpoint
        })
        .await;
        assert_eq!(decode_withdrawn.pools()[0].pool_id(), prefill_pool_id);
        assert_eq!(
            decode_withdrawn.pools()[0].pool_roles(),
            [WorkerRole::Prefill]
        );

        component
            .drt()
            .discovery()
            .unregister(prefill_source)
            .await
            .unwrap();
        wait_for_catalog(&mut catalog_rx, |catalog| catalog.pools().is_empty()).await;
        prefill_cancel.cancel();
        decode_cancel.cancel();
        for slot in [prefill_slot, decode_slot] {
            tokio::time::timeout(Duration::from_secs(5), slot)
                .await
                .expect("endpoint slot shutdown timed out")
                .unwrap();
        }
    }

    #[tokio::test]
    async fn pool_is_withdrawn_before_recovery_teardown_completes() {
        let registry = Arc::new(registry());
        let attachment = registry
            .attach(PoolAttachRequest {
                pool_id: PoolId::new(domain(1, "meta/llama").id, DcId::new(7)),
                endpoint: EndpointId::from("prod.backend.generate"),
                registrations: vec![CanonicalModelRegistration::new(
                    super::super::identity::CanonicalModelId::new("llama").unwrap(),
                    Vec::new(),
                )],
                query_semantics: query_semantics(),
                roles: vec![WorkerRole::Aggregated],
                #[cfg(feature = "kv-dc-relay-wan")]
                wan_facts: Some(PoolWanFacts {
                    runtime_configs: HashMap::new(),
                }),
            })
            .await
            .unwrap();
        let PoolAttachment {
            pool_id,
            layout_generation,
            handle,
            mut faults,
            ..
        } = attachment;
        let (teardown_started, teardown_started_rx) = tokio::sync::oneshot::channel();
        let (release_teardown, release_teardown_rx) = tokio::sync::oneshot::channel();
        let retirement_registry = registry.clone();
        let retirement = tokio::spawn(async move {
            withdraw_drain_and_remove_pool(
                &retirement_registry,
                pool_id,
                layout_generation,
                PoolRetirementMode::Graceful,
                &mut faults,
                async move {
                    let _ = teardown_started.send(());
                    let _ = release_teardown_rx.await;
                    handle.shutdown().await
                },
            )
            .await
        });

        teardown_started_rx.await.unwrap();
        assert!(registry.catalog().pools().is_empty());
        assert_eq!(registry.pool_count().await, 1);

        release_teardown.send(()).unwrap();
        retirement.await.unwrap().unwrap();
        assert_eq!(registry.pool_count().await, 0);
    }

    #[test]
    fn pending_faults_coalesce_duplicate_and_latest_publisher_actions() {
        let resource = CkfFailurePoint::PrecommitAllocationFailure.disposition();
        let reject = CkfFailurePoint::SourceProtocolFailure.disposition();
        let mut pending = PendingActorFaults::default();

        for event_id in 0..1_000 {
            pending.push(actor_fault(1, 0, 1, event_id, resource));
        }
        pending.push(actor_fault(1, 0, 1, 1_000, reject));
        pending.push(actor_fault(1, 0, 1, 1_001, resource));

        assert_eq!(pending.len(), 1);
        let Some(PendingActorAction::Fault(fault)) = pending.pop_front() else {
            panic!("strongest source fault was not retained");
        };
        assert_eq!(fault.disposition.action, CkfFailureAction::RejectSource);
        assert!(pending.pop_front().is_none());

        pending.push(actor_fault(1, 0, 1, 1_002, reject));
        pending.push(actor_fault(1, 0, 2, 1_003, resource));
        let Some(PendingActorAction::Fault(fault)) = pending.pop_front() else {
            panic!("latest publisher fault was not retained");
        };
        assert_eq!(fault.publisher_id, 2);
        assert_eq!(
            fault.disposition.action,
            CkfFailureAction::ReportResourceFailure
        );
    }

    #[test]
    fn pending_fault_overflow_fails_safe_with_a_producer_fence() {
        let resource = CkfFailurePoint::PrecommitAllocationFailure.disposition();
        let mut pending = PendingActorFaults::default();

        for worker_id in 0..MAX_PENDING_SOURCE_FAULTS as u64 {
            pending.push(actor_fault(worker_id, 0, 1, worker_id, resource));
        }
        assert_eq!(pending.len(), MAX_PENDING_SOURCE_FAULTS);

        pending.push(actor_fault(
            MAX_PENDING_SOURCE_FAULTS as u64,
            0,
            1,
            MAX_PENDING_SOURCE_FAULTS as u64,
            resource,
        ));
        assert_eq!(pending.len(), 1);
        assert!(matches!(
            pending.pop_front(),
            Some(PendingActorAction::ProducerFence(
                ProducerFenceTrigger::PendingOverflow(_)
            ))
        ));
        assert!(pending.pop_front().is_none());
    }

    #[test]
    fn producer_fence_supersedes_varied_pending_source_faults() {
        let resource = CkfFailurePoint::PrecommitAllocationFailure.disposition();
        let reject = CkfFailurePoint::SourceProtocolFailure.disposition();
        let fence = CkfFailurePoint::PrewriteInvariantMismatch.disposition();
        let mut pending = PendingActorFaults::default();

        pending.push(actor_fault(1, 0, 1, 1, resource));
        pending.push(actor_fault(1, 0, 1, 2, resource));
        pending.push(actor_fault(2, 0, 1, 3, reject));
        pending.push(actor_fault(3, 0, 1, 4, resource));
        pending.push(actor_fault(1, 0, 1, 5, fence));
        pending.push(actor_fault(4, 0, 1, 6, resource));

        assert_eq!(pending.len(), 1);
        let Some(PendingActorAction::ProducerFence(ProducerFenceTrigger::Fault(fault))) =
            pending.pop_front()
        else {
            panic!("producer fence did not supersede weaker source faults");
        };
        assert_eq!(
            fault.disposition.action,
            CkfFailureAction::FenceAndRebuildProducer
        );
        assert!(pending.pop_front().is_none());
    }

    #[tokio::test]
    async fn producer_fence_interrupts_inflight_fault_recovery() {
        let resource = CkfFailurePoint::PrecommitAllocationFailure.disposition();
        let fence = CkfFailurePoint::PrewriteInvariantMismatch.disposition();
        let (sender, mut receiver) = tokio::sync::mpsc::channel(DEFAULT_FAULT_CAPACITY);
        let (recovery_started, recovery_started_rx) = tokio::sync::oneshot::channel();
        let (release_recovery, release_recovery_rx) = tokio::sync::oneshot::channel::<()>();
        let collection = tokio::spawn(async move {
            let mut pending = PendingActorFaults::default();
            let outcome = collect_pending_while(&mut receiver, &mut pending, async move {
                let _ = recovery_started.send(());
                let _ = release_recovery_rx.await;
                TargetFaultDisposition::Recovering
            })
            .await;
            (outcome, pending)
        });

        recovery_started_rx.await.unwrap();
        for event_id in 0..100 {
            sender
                .send(actor_fault(1, 0, 1, event_id, resource))
                .await
                .unwrap();
        }
        sender.send(actor_fault(1, 0, 1, 100, fence)).await.unwrap();

        let (outcome, pending) = tokio::time::timeout(Duration::from_secs(1), collection)
            .await
            .expect("producer fence did not interrupt fault recovery")
            .unwrap();
        assert!(matches!(
            outcome,
            FaultCollection::ProducerFence(ProducerFenceTrigger::Fault(_))
        ));
        assert_eq!(pending.len(), 0);
        assert!(release_recovery.send(()).is_err());
    }

    #[tokio::test]
    async fn fenced_target_disposition_withdraws_the_pool_generation() {
        let registry = registry();
        let domain = domain(1, "meta/llama");
        let attachment = registry
            .attach(PoolAttachRequest {
                pool_id: PoolId::new(domain.id, DcId::new(7)),
                endpoint: EndpointId::from("prod.backend.generate"),
                registrations: vec![CanonicalModelRegistration::new(
                    super::super::identity::CanonicalModelId::new("llama").unwrap(),
                    Vec::new(),
                )],
                query_semantics: query_semantics(),
                roles: vec![WorkerRole::Aggregated],
                #[cfg(feature = "kv-dc-relay-wan")]
                wan_facts: Some(PoolWanFacts {
                    runtime_configs: HashMap::new(),
                }),
            })
            .await
            .unwrap();
        let mode = target_fault_retirement_mode(TargetFaultDisposition::Fenced).unwrap();
        assert!(
            registry
                .withdraw(attachment.pool_id, attachment.layout_generation, mode)
                .await
        );
        assert!(registry.catalog().pools().is_empty());

        let PoolAttachment {
            pool_id,
            layout_generation,
            handle,
            mut faults,
            ..
        } = attachment;
        drain_faults_while(pool_id, &mut faults, handle.fence())
            .await
            .unwrap();
        assert!(registry.remove(pool_id, layout_generation).await);
    }
}
