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

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "ckf-diagnostics")]
use std::sync::atomic::Ordering;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::CkfFailureAction;
use dynamo_kv_router::protocols::{DpRank, KvCacheEventError, WorkerId};
use dynamo_runtime::component::Component;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::traits::DistributedRuntimeProvider;
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
use super::identity::{CanonicalModelRegistration, DcPoolCatalog, DcRelayIdentity};
use super::pool_registry::{
    PoolActorConfig, PoolAttachRequest, PoolAttachment, PoolRegistry, PoolRetirementMode,
    drain_faults_while,
};
use super::resolution::stable_dc_id;
use crate::discovery::{KvSourceMembershipCoordinator, KvSourceMembershipWatch};
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
pub struct KvDcRelayConfig {
    pub namespace_filter: Option<String>,
    pub endpoint_prefix: Option<String>,
    pub publication_threshold: usize,
    pub publication_delay_ms: u64,
    pub recovery_attempt_timeout_ms: u64,
}

impl Default for KvDcRelayConfig {
    fn default() -> Self {
        Self {
            namespace_filter: None,
            endpoint_prefix: None,
            publication_threshold: DEFAULT_PUBLICATION_THRESHOLD,
            publication_delay_ms: DEFAULT_PUBLICATION_DELAY.as_millis() as u64,
            recovery_attempt_timeout_ms: DEFAULT_RECOVERY_ATTEMPT_TIMEOUT.as_millis() as u64,
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
    pub member_set_capacity: usize,
    pub refcount_capacity: usize,
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
    pub endpoint_count: usize,
    pub active_endpoint_count: usize,
    pub fenced_endpoint_count: usize,
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
            #[cfg(feature = "ckf-diagnostics")]
            recovery: WorkerQueryHealthSnapshot::default(),
        }
    }
}

type SharedEndpointStatus = Arc<RwLock<EndpointSlotStatus>>;

struct EndpointSlotTask {
    metadata: watch::Sender<Option<EndpointMembership>>,
    status: SharedEndpointStatus,
    task: JoinHandle<()>,
}

struct EndpointPoolRuntime {
    attachment: PoolAttachment,
    recovery: RecoverySupervisor<KvDcRelayRecoveryTarget>,
    binding: ActorBinding,
    registrations: Vec<CanonicalModelRegistration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActorBinding {
    domain: KvCacheDomainKey,
    kv_state_endpoint: EndpointId,
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
    #[cfg(feature = "ckf-diagnostics")]
    relay_identity: DcRelayIdentity,
    cancel: CancellationToken,
    membership: Mutex<Option<DcMembershipWatch>>,
    supervisor: Mutex<Option<JoinHandle<()>>>,
    statuses: Arc<RwLock<HashMap<EndpointId, SharedEndpointStatus>>>,
    pools: Arc<PoolRegistry>,
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
        anyhow::ensure!(
            config.publication_threshold != 0,
            "KV DC Relay publication_threshold must be positive"
        );
        anyhow::ensure!(
            config.publication_delay_ms != 0,
            "KV DC Relay publication_delay_ms must be positive"
        );
        anyhow::ensure!(
            config.recovery_attempt_timeout_ms != 0,
            "KV DC Relay recovery_attempt_timeout_ms must be positive"
        );
        let publication = ActorPublicationConfig {
            threshold: config.publication_threshold,
            delay: Duration::from_millis(config.publication_delay_ms),
        };
        let discovery = KvDcRelayDiscoveryConfig {
            watch_all: config.namespace_filter.is_none(),
            namespaces: config.namespace_filter.into_iter().collect(),
            endpoint_prefixes: config.endpoint_prefix.into_iter().collect(),
        };
        let cancel = component.drt().child_token();
        let membership =
            DcMembershipWatch::start(component.drt().discovery(), discovery, cancel.clone())
                .await?;
        let membership_rx = membership.subscribe();
        let statuses = Arc::new(RwLock::new(HashMap::new()));
        let dc_id: Arc<str> = Arc::from(dc_id);
        let relay_identity =
            DcRelayIdentity::new(component.drt().connection_id(), new_relay_incarnation()?);
        let ckf_dc_id = stable_dc_id(dc_id.as_ref());
        let pools = Arc::new(PoolRegistry::new(
            relay_identity,
            PoolActorConfig {
                expected_unique_blocks: DEFAULT_EXPECTED_UNIQUE_BLOCKS,
                publication_threshold: publication.threshold,
                publication_delay: publication.delay,
            },
        ));
        let supervisor = tokio::spawn(run_host_supervisor(
            component,
            ckf_dc_id,
            membership_rx,
            statuses.clone(),
            pools.clone(),
            Duration::from_millis(config.recovery_attempt_timeout_ms),
            cancel.child_token(),
        ));
        Ok(Self {
            #[cfg(feature = "ckf-diagnostics")]
            dc_id,
            #[cfg(feature = "ckf-diagnostics")]
            relay_identity,
            cancel,
            membership: Mutex::new(Some(membership)),
            supervisor: Mutex::new(Some(supervisor)),
            statuses,
            pools,
        })
    }

    pub fn pool_catalog(&self) -> DcPoolCatalog {
        self.pools.catalog()
    }

    pub fn watch_pool_catalog(&self) -> watch::Receiver<DcPoolCatalog> {
        self.pools.watch_catalog()
    }

    #[cfg(feature = "ckf-diagnostics")]
    pub async fn stats(&self) -> Result<KvDcRelayStats, KvDcRelayError> {
        let statuses: Vec<_> = self
            .statuses
            .read()
            .await
            .iter()
            .map(|(slot_id, status)| (slot_id.clone(), status.clone()))
            .collect();
        let mut endpoints = Vec::with_capacity(statuses.len());
        for (slot_id, status) in statuses {
            endpoints.push(endpoint_stats(slot_id, status).await?);
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
        KvDcRelayHealth {
            healthy: !self.cancel.is_cancelled() && fenced_endpoint_count == 0,
            shutting_down: self.cancel.is_cancelled(),
            endpoint_count: statuses.len(),
            active_endpoint_count,
            fenced_endpoint_count,
        }
    }

    pub async fn shutdown(&self) -> Result<(), KvDcRelayError> {
        self.cancel.cancel();
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
) {
    let recovery_fetch_permit = Arc::new(Semaphore::new(DEFAULT_RECOVERY_FETCH_CONCURRENCY));
    let mut slots: HashMap<EndpointId, EndpointSlotTask> = HashMap::new();
    let mut retired_slots = JoinSet::new();

    loop {
        let mut view = membership_rx.borrow_and_update().clone();
        reject_duplicate_live_pools(&mut view, ckf_dc_id);
        for (slot_id, membership) in view.endpoints.iter() {
            let slot = slots.entry(slot_id.clone()).or_insert_with(|| {
                let (metadata, metadata_rx) = watch::channel(None);
                let status = Arc::new(RwLock::new(EndpointSlotStatus::default()));
                let task = tokio::spawn(run_endpoint_slot(
                    component.clone(),
                    ckf_dc_id,
                    slot_id.clone(),
                    metadata_rx,
                    status.clone(),
                    Arc::new(Semaphore::new(1)),
                    recovery_fetch_permit.clone(),
                    pools.clone(),
                    recovery_attempt_timeout,
                    cancel.child_token(),
                ));
                EndpointSlotTask {
                    metadata,
                    status,
                    task,
                }
            });
            publish_endpoint_metadata_if_changed(&slot.metadata, membership);
        }
        retire_departed_endpoint_slots(&view, &mut slots, &mut retired_slots);
        *statuses.write().await = slots
            .iter()
            .map(|(slot_id, slot)| (slot_id.clone(), slot.status.clone()))
            .collect();

        tokio::select! {
            _ = cancel.cancelled() => break,
            changed = membership_rx.changed() => {
                if changed.is_err() {
                    break;
                }
            }
            retired = retired_slots.join_next(), if !retired_slots.is_empty() => {
                report_retired_endpoint_slot(retired);
            }
        }
    }

    for (slot_id, slot) in slots {
        drop(slot.metadata);
        report_endpoint_slot_exit(slot_id, slot.task.await);
    }
    while let Some(retired) = retired_slots.join_next().await {
        report_retired_endpoint_slot(Some(retired));
    }
    pools.shutdown().await;
}

fn reject_duplicate_live_pools(view: &mut DcMembershipView, dc_id: dynamo_kv_router::DcId) {
    let mut owners: HashMap<PoolId, Vec<EndpointId>> = HashMap::new();
    for (endpoint, membership) in view.endpoints.iter() {
        if !membership.conflicts.is_empty() {
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
        .filter(|slot_id| !view.endpoints.contains_key(*slot_id))
        .cloned()
        .collect();
    for slot_id in departed {
        let Some(slot) = slots.remove(&slot_id) else {
            continue;
        };
        drop(slot.metadata);
        retired_slots.spawn(async move {
            let result = slot.task.await;
            (slot_id, result)
        });
    }
}

type RetiredEndpointSlot =
    Result<(EndpointId, Result<(), tokio::task::JoinError>), tokio::task::JoinError>;

fn report_retired_endpoint_slot(retired: Option<RetiredEndpointSlot>) {
    match retired {
        Some(Ok((slot_id, result))) => report_endpoint_slot_exit(slot_id, result),
        Some(Err(error)) if !error.is_cancelled() => {
            tracing::warn!(%error, "KV DC Relay endpoint retirement monitor failed");
        }
        Some(Err(_)) | None => {}
    }
}

fn report_endpoint_slot_exit(slot_id: EndpointId, result: Result<(), tokio::task::JoinError>) {
    if let Err(error) = result
        && !error.is_cancelled()
    {
        tracing::warn!(endpoint = %slot_id, %error, "KV DC Relay endpoint slot failed");
    }
}

fn report_actor_fault(endpoint: &EndpointId, fault: &ActorFault) {
    tracing::error!(
        %endpoint,
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
            worker_id = fault.worker_id,
            dp_rank = fault.dp_rank,
            publisher_id = fault.publisher_id,
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
    slot_id: EndpointId,
    mut metadata_rx: watch::Receiver<Option<EndpointMembership>>,
    status: SharedEndpointStatus,
    rebuild_permit: Arc<Semaphore>,
    recovery_fetch_permit: Arc<Semaphore>,
    pools: Arc<PoolRegistry>,
    recovery_attempt_timeout: Duration,
    cancel: CancellationToken,
) {
    let endpoint = slot_id.clone();
    let mut config_tx: Option<watch::Sender<HashMap<WorkerId, ModelRuntimeConfig>>> = None;
    let mut source_watch: Option<KvSourceMembershipWatch> = None;
    let mut runtime: Option<EndpointPoolRuntime> = None;
    let mut layout_generation = 0u64;
    let mut retry_binding: Option<ActorBinding> = None;
    let mut retry_delay = Duration::from_millis(100);
    let mut start_failures = 0u64;
    let mut registration_refresh_failures = 0u64;
    let mut pending_faults = PendingActorFaults::default();

    loop {
        let membership = metadata_rx.borrow_and_update().clone();
        {
            let mut current = status.write().await;
            current.membership = membership.clone();
            current.layout_generation = layout_generation;
            if runtime.is_none() {
                current.lifecycle = inactive_slot_lifecycle(membership.as_ref());
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
        let desired_binding = membership.as_ref().and_then(|membership| {
            if !membership.is_materializable() {
                return None;
            }
            let domain = membership.domain.clone()?;
            let kv_state_endpoint = source_view.as_ref()?.resolved_kv_state_endpoint()?.clone();
            source_view
                .as_ref()?
                .sources
                .values()
                .any(|source| source.active_source().is_some())
                .then_some(ActorBinding {
                    domain,
                    kv_state_endpoint,
                })
        });
        if retry_binding.as_ref() != desired_binding.as_ref() {
            retry_binding = desired_binding.clone();
            retry_delay = Duration::from_millis(100);
            start_failures = 0;
            registration_refresh_failures = 0;
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
                let mut current = status.write().await;
                current.lifecycle = SlotLifecycle::Lightweight;
                #[cfg(feature = "ckf-diagnostics")]
                {
                    current.recovery = WorkerQueryHealthSnapshot::default();
                }
            }
        }

        if let (Some(active), Some(membership)) = (runtime.as_mut(), membership.as_ref())
            && membership.is_materializable()
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

        enum SlotInput {
            Metadata,
            Source,
            SourceClosed,
            Fault(PendingActorAction),
            PoolUnavailable,
            Health,
            Retry,
            Cancelled,
        }
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
    Ok(EndpointPoolRuntime {
        attachment,
        recovery,
        binding,
        registrations,
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
        ..
    } = active;
    let PoolAttachment {
        pool_id,
        layout_generation,
        handle,
        mut faults,
        ..
    } = attachment;
    let teardown = async {
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
    slot_id: EndpointId,
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
                member_set_capacity: memory.member_set_capacity(),
                refcount_capacity: memory.refcount_capacity(),
                insertion_scratch_capacity: memory.insertion_scratch_capacity(),
            }),
        )
    } else {
        (None, None, None)
    };
    let membership = status.membership;
    Ok(KvDcRelayEndpointStats {
        serving_endpoint: slot_id.to_string(),
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
            .map(|membership| membership.roles.clone())
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
        kv_block_size: domain.kv_block_size,
        event_hash_format: domain.event_hash_format,
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

    fn membership(endpoint: &str, domain: KvCacheDomainKey) -> EndpointMembership {
        let endpoint = EndpointId::from(endpoint);
        EndpointMembership {
            endpoint,
            generation: 1,
            domain: Some(domain),
            registrations: vec![CanonicalModelRegistration::new(
                super::super::identity::CanonicalModelId::new("llama").unwrap(),
                Vec::new(),
            )],
            models: vec!["llama".to_string()],
            aliases: Vec::new(),
            roles: Vec::new(),
            runtime_configs: HashMap::new(),
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
            kv_block_size: 64,
            event_hash_format: 1,
        }
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
    ) -> EventPublisher {
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
        component
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
        publisher
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
            Vec::new(),
            None,
        )
    }

    fn projected_membership_with_metadata(
        endpoint: &EndpointId,
        worker_id: WorkerId,
        model: &str,
        artifact: &str,
        kv_state_endpoint: EndpointId,
        aliases: Vec<String>,
        context_length: Option<u32>,
    ) -> EndpointMembership {
        let mut card = crate::model_card::ModelDeploymentCard::with_name_only(model);
        card.source_path = Some(artifact.to_string());
        card.kv_cache_block_size = 64;
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

    #[tokio::test]
    async fn departed_endpoint_slots_are_reaped_instead_of_parked() {
        let slot_id = EndpointId::from("prod.backend.generate");
        let (metadata, mut metadata_rx) = watch::channel(None);
        let task = tokio::spawn(async move { while metadata_rx.changed().await.is_ok() {} });
        let mut slots = HashMap::from([(
            slot_id.clone(),
            EndpointSlotTask {
                metadata,
                status: Arc::new(RwLock::new(EndpointSlotStatus::default())),
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
        let (retired_slot, result) = retired_slots.join_next().await.unwrap().unwrap();
        assert_eq!(retired_slot, slot_id);
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
            !membership.is_materializable()
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
            #[cfg(feature = "ckf-diagnostics")]
            relay_identity: DcRelayIdentity::new(11, 7),
            cancel: CancellationToken::new(),
            membership: Mutex::new(None),
            supervisor: Mutex::new(None),
            statuses,
            pools: Arc::new(registry()),
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
        let _old_publisher = register_live_source(&component, &old_kv_endpoint, worker).await;
        let _new_publisher = register_live_source(&component, &new_kv_endpoint, worker).await;
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
        let (metadata_tx, metadata_rx) = watch::channel(Some(old_membership));
        let status = Arc::new(RwLock::new(EndpointSlotStatus::default()));
        let registry = Arc::new(registry());
        let mut catalog_rx = registry.watch_catalog();
        let slot_cancel = CancellationToken::new();
        let slot = tokio::spawn(run_endpoint_slot(
            component,
            DcId::new(7),
            serving_endpoint,
            metadata_rx,
            status,
            Arc::new(Semaphore::new(1)),
            Arc::new(Semaphore::new(1)),
            registry,
            Duration::from_secs(1),
            slot_cancel.clone(),
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
        let _publisher = register_live_source(&component, &kv_endpoint, worker).await;
        let old_membership = projected_membership_with_metadata(
            &serving_endpoint,
            worker_id,
            "llama",
            "meta/llama",
            kv_endpoint.clone(),
            vec!["old-alias".to_string()],
            Some(4096),
        );
        let new_membership = projected_membership_with_metadata(
            &serving_endpoint,
            worker_id,
            "llama",
            "meta/llama",
            kv_endpoint,
            vec!["new-alias".to_string()],
            Some(8192),
        );
        let (metadata_tx, metadata_rx) = watch::channel(Some(old_membership));
        let status = Arc::new(RwLock::new(EndpointSlotStatus::default()));
        let registry = Arc::new(registry());
        let mut catalog_rx = registry.watch_catalog();
        let slot_cancel = CancellationToken::new();
        let slot = tokio::spawn(run_endpoint_slot(
            component,
            DcId::new(7),
            serving_endpoint,
            metadata_rx,
            status,
            Arc::new(Semaphore::new(1)),
            Arc::new(Semaphore::new(1)),
            registry,
            Duration::from_secs(1),
            slot_cancel.clone(),
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
            pending.push(actor_fault(1, 0, 205, event_id, resource));
        }
        pending.push(actor_fault(1, 0, 205, 1_000, reject));
        pending.push(actor_fault(1, 0, 205, 1_001, resource));

        assert_eq!(pending.len(), 1);
        let Some(PendingActorAction::Fault(fault)) = pending.pop_front() else {
            panic!("strongest source fault was not retained");
        };
        assert_eq!(fault.disposition.action, CkfFailureAction::RejectSource);
        assert!(pending.pop_front().is_none());

        pending.push(actor_fault(1, 0, 205, 1_002, reject));
        pending.push(actor_fault(1, 0, 100, 1_003, resource));
        let Some(PendingActorAction::Fault(fault)) = pending.pop_front() else {
            panic!("latest publisher fault was not retained");
        };
        assert_eq!(fault.publisher_id, 100);
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
