// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DC-scoped KV-cache Relay with one serialized CKF actor per serving endpoint.
//!
//! Dynamo discovery and worker-local recovery feed endpoint actors. The actors'
//! exact member ownership is authoritative; each materialization publishes one
//! physical CKF layout for a future Relay-to-global-router adapter.
//! The subscription seam remains crate-private: a standalone/WAN publisher API
//! requires delivery cursors and recovery semantics and is intentionally deferred.
//!
//! NOTE: One serialized actor per endpoint pool is the current measured choice, not a claim that
//! it scales indefinitely. A worker-partitioned, multi-issuer Mooncake comparison found the
//! attempted striped concurrent producer slower with worse tail admission latency. Rerun the
//! dedicated Relay campaign before changing this ownership model; further producer optimization
//! will likely be needed for substantially larger DC-scale pools.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "ckf-diagnostics")]
use std::sync::atomic::AtomicU64;
#[cfg(feature = "ckf-diagnostics")]
use std::sync::atomic::Ordering;
#[cfg(feature = "ckf-diagnostics")]
use std::time::Instant;

use dynamo_kv_router::identity::PoolId;
use dynamo_kv_router::indexer::cuckoo::{CkfConfig, CkfFailureAction};
use dynamo_kv_router::protocols::{DpRank, KvCacheEventError, WorkerId};
use dynamo_runtime::component::Component;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::{RwLock, Semaphore, mpsc, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use super::actor::{ActorFault, KvDcRelayHandle, KvDcRelayRecoveryTarget, StreamScope};
use super::discovery::{
    DcDiscoveryFilter, DcMembershipView, DcMembershipWatch, EndpointMembership, KvCacheDomainKey,
};
use super::resolution::{EndpointLocator, PoolBinding, stable_dc_id};
use crate::discovery::{KvSourceMembershipCoordinator, KvSourceMembershipWatch};
#[cfg(feature = "ckf-diagnostics")]
use crate::kv_router::indexer::WorkerQueryHealthSnapshot;
use crate::kv_router::indexer::{
    DEFAULT_RECOVERY_ATTEMPT_TIMEOUT, RecoverySupervisor, TargetFaultDisposition,
    start_target_subscriber,
};

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
    #[error(
        "stale source epoch for worker {worker_id} rank {dp_rank}: current {current}, received {received}"
    )]
    StaleSourceEpoch {
        worker_id: WorkerId,
        dp_rank: DpRank,
        current: u64,
        received: u64,
    },
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
    pub process_incarnation: u64,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayEndpointStats {
    pub serving_endpoint: String,
    pub lifecycle: String,
    pub layout_generation: u64,
    pub cache_domain: Option<KvDcRelayCacheDomainStats>,
    pub compatibility_conflict: bool,
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
    pub process_incarnation: u64,
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

struct EndpointActorRuntime {
    handle: KvDcRelayHandle,
    recovery: RecoverySupervisor<KvDcRelayRecoveryTarget>,
    faults: mpsc::Receiver<ActorFault>,
    binding: ActorBinding,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActorBinding {
    domain: KvCacheDomainKey,
    kv_state_endpoint: EndpointId,
}

/// DC-wide Relay host. It is intentionally not scoped to a model, namespace, or endpoint.
pub struct KvDcRelay {
    #[cfg(feature = "ckf-diagnostics")]
    dc_id: Arc<str>,
    #[cfg(feature = "ckf-diagnostics")]
    process_incarnation: u64,
    cancel: CancellationToken,
    membership: Mutex<Option<DcMembershipWatch>>,
    supervisor: Mutex<Option<JoinHandle<()>>>,
    statuses: Arc<RwLock<HashMap<EndpointId, SharedEndpointStatus>>>,
}

impl KvDcRelay {
    pub async fn start(
        component: Component,
        dc_id: String,
        config: KvDcRelayConfig,
    ) -> anyhow::Result<Self> {
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
        let cancel = component.drt().child_token();
        let membership = DcMembershipWatch::start(
            component.drt().discovery(),
            DcDiscoveryFilter {
                namespace: config.namespace_filter,
                endpoint_prefix: config.endpoint_prefix,
            },
            cancel.clone(),
        )
        .await?;
        let membership_rx = membership.subscribe();
        let statuses = Arc::new(RwLock::new(HashMap::new()));
        let dc_id: Arc<str> = Arc::from(dc_id);
        let process_incarnation = component.drt().connection_id();
        let supervisor = tokio::spawn(run_host_supervisor(
            component,
            dc_id.clone(),
            process_incarnation,
            membership_rx,
            statuses.clone(),
            publication,
            Duration::from_millis(config.recovery_attempt_timeout_ms),
            cancel.child_token(),
        ));
        Ok(Self {
            #[cfg(feature = "ckf-diagnostics")]
            dc_id,
            #[cfg(feature = "ckf-diagnostics")]
            process_incarnation,
            cancel,
            membership: Mutex::new(Some(membership)),
            supervisor: Mutex::new(Some(supervisor)),
            statuses,
        })
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
                process_incarnation: self.process_incarnation,
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
            process_incarnation: self.process_incarnation,
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

#[allow(clippy::too_many_arguments)]
async fn run_host_supervisor(
    component: Component,
    dc_id: Arc<str>,
    process_incarnation: u64,
    mut membership_rx: watch::Receiver<DcMembershipView>,
    statuses: Arc<RwLock<HashMap<EndpointId, SharedEndpointStatus>>>,
    publication: ActorPublicationConfig,
    recovery_attempt_timeout: Duration,
    cancel: CancellationToken,
) {
    let recovery_fetch_permit = Arc::new(Semaphore::new(DEFAULT_RECOVERY_FETCH_CONCURRENCY));
    let mut slots: HashMap<EndpointId, EndpointSlotTask> = HashMap::new();
    let ckf_dc_id = stable_dc_id(dc_id.as_ref());

    loop {
        let mut view = membership_rx.borrow_and_update().clone();
        reject_duplicate_live_pools(&mut view, ckf_dc_id);
        for (endpoint, membership) in &view.endpoints {
            let slot = slots.entry(endpoint.clone()).or_insert_with(|| {
                let (metadata, metadata_rx) = watch::channel(None);
                let status = Arc::new(RwLock::new(EndpointSlotStatus::default()));
                let task = tokio::spawn(run_endpoint_slot(
                    component.clone(),
                    dc_id.clone(),
                    process_incarnation,
                    endpoint.clone(),
                    metadata_rx,
                    status.clone(),
                    Arc::new(Semaphore::new(1)),
                    recovery_fetch_permit.clone(),
                    publication,
                    recovery_attempt_timeout,
                    cancel.child_token(),
                ));
                EndpointSlotTask {
                    metadata,
                    status,
                    task,
                }
            });
            slot.metadata.send_replace(Some(membership.clone()));
        }
        for (endpoint, slot) in &slots {
            if !view.endpoints.contains_key(endpoint) {
                slot.metadata.send_replace(None);
            }
        }
        *statuses.write().await = slots
            .iter()
            .map(|(endpoint, slot)| (endpoint.clone(), slot.status.clone()))
            .collect();

        tokio::select! {
            _ = cancel.cancelled() => break,
            changed = membership_rx.changed() => {
                if changed.is_err() {
                    break;
                }
            }
        }
    }

    drop(
        slots
            .values()
            .map(|slot| slot.metadata.clone())
            .collect::<Vec<_>>(),
    );
    for (_, slot) in slots {
        drop(slot.metadata);
        if let Err(error) = slot.task.await
            && !error.is_cancelled()
        {
            tracing::warn!(%error, "KV DC Relay endpoint slot failed during shutdown");
        }
    }
}

fn reject_duplicate_live_pools(view: &mut DcMembershipView, dc_id: dynamo_kv_router::DcId) {
    let mut owners: HashMap<PoolId, Vec<EndpointId>> = HashMap::new();
    for (endpoint, membership) in &view.endpoints {
        if membership.compatibility_conflict {
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

    for (pool_id, endpoints) in owners {
        if endpoints.len() < 2 {
            continue;
        }
        tracing::error!(
            %pool_id,
            endpoints = ?endpoints,
            "multiple live serving endpoints resolve to one CKF pool; fencing all colliding endpoints"
        );
        for endpoint in endpoints {
            if let Some(membership) = view.endpoints.get_mut(&endpoint) {
                membership.compatibility_conflict = true;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_endpoint_slot(
    component: Component,
    dc_id: Arc<str>,
    process_incarnation: u64,
    endpoint: EndpointId,
    mut metadata_rx: watch::Receiver<Option<EndpointMembership>>,
    status: SharedEndpointStatus,
    rebuild_permit: Arc<Semaphore>,
    recovery_fetch_permit: Arc<Semaphore>,
    publication: ActorPublicationConfig,
    recovery_attempt_timeout: Duration,
    cancel: CancellationToken,
) {
    let mut config_tx: Option<
        watch::Sender<HashMap<WorkerId, crate::local_model::runtime_config::ModelRuntimeConfig>>,
    > = None;
    let mut source_watch: Option<KvSourceMembershipWatch> = None;
    let mut actor: Option<EndpointActorRuntime> = None;
    let mut layout_generation = 0u64;

    loop {
        let membership = metadata_rx.borrow_and_update().clone();
        {
            let mut current = status.write().await;
            current.membership = membership.clone();
            current.layout_generation = layout_generation;
            if membership.is_some() && current.lifecycle == SlotLifecycle::Lightweight {
                current.lifecycle = SlotLifecycle::Discovered;
            }
        }

        if let Some(membership) = &membership {
            if let Some(sender) = &config_tx {
                sender.send_replace(membership.runtime_configs.clone());
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
        let desired_binding = membership.as_ref().and_then(|membership| {
            if membership.compatibility_conflict {
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

        let binding_changed = actor
            .as_ref()
            .is_some_and(|active| Some(&active.binding) != desired_binding.as_ref());
        if binding_changed || membership.is_none() {
            if let Some(active) = actor.take() {
                status.write().await.lifecycle = SlotLifecycle::Draining;
                stop_endpoint_actor(active).await;
                let mut current = status.write().await;
                current.actor = None;
                if membership.is_some() {
                    current.lifecycle = SlotLifecycle::Discovered;
                }
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

        if actor.is_none()
            && let (Some(binding), Some(membership), Some(membership_watch)) = (
                desired_binding.clone(),
                membership.clone(),
                source_watch.clone(),
            )
        {
            status.write().await.lifecycle = SlotLifecycle::Starting;
            let candidate_generation = membership.generation;
            layout_generation = layout_generation.saturating_add(1);
            match start_endpoint_actor(
                component.clone(),
                dc_id.clone(),
                process_incarnation,
                endpoint.clone(),
                layout_generation,
                binding.clone(),
                membership_watch,
                rebuild_permit.clone(),
                recovery_fetch_permit.clone(),
                publication,
                recovery_attempt_timeout,
                cancel.child_token(),
            )
            .await
            {
                Ok(candidate)
                    if metadata_rx
                        .borrow()
                        .as_ref()
                        .is_some_and(|current| current.generation == candidate_generation)
                        && source_watch.as_ref().and_then(|watch| {
                            watch.borrow().resolved_kv_state_endpoint().cloned()
                        }) == Some(binding.kv_state_endpoint.clone()) =>
                {
                    let mut current = status.write().await;
                    current.layout_generation = layout_generation;
                    current.actor = Some(candidate.handle.clone());
                    current.lifecycle = SlotLifecycle::Active;
                    actor = Some(candidate);
                }
                Ok(candidate) => {
                    stop_endpoint_actor(candidate).await;
                }
                Err(error) => {
                    tracing::error!(%endpoint, %error, "Failed to materialize KV DC Relay endpoint actor");
                    let mut current = status.write().await;
                    current.lifecycle = SlotLifecycle::Fenced;
                    current.actor = None;
                }
            }
        }

        if membership
            .as_ref()
            .is_some_and(|membership| membership.compatibility_conflict)
        {
            status.write().await.lifecycle = SlotLifecycle::Fenced;
        }

        enum SlotInput {
            Metadata,
            Source,
            Fault(ActorFault),
            ActorExited,
            Health,
            Cancelled,
        }
        let input = tokio::select! {
            _ = cancel.cancelled() => SlotInput::Cancelled,
            changed = metadata_rx.changed() => {
                if changed.is_ok() { SlotInput::Metadata } else { SlotInput::Cancelled }
            }
            changed = async { source_watch.as_mut().expect("guarded source watch").changed().await }, if source_watch.is_some() => {
                if changed.is_ok() { SlotInput::Source } else { SlotInput::Metadata }
            }
            fault = async { actor.as_mut().expect("guarded actor").faults.recv().await }, if actor.is_some() => {
                match fault {
                    Some(fault) => SlotInput::Fault(fault),
                    None => SlotInput::ActorExited,
                }
            }
            _ = diagnostic_tick(), if actor.is_some() => SlotInput::Health,
        };
        match input {
            SlotInput::Metadata | SlotInput::Source | SlotInput::Health => {}
            SlotInput::ActorExited => {
                tracing::error!(%endpoint, "KV DC Relay actor exited unexpectedly; rebuilding its producer generation");
                status.write().await.lifecycle = SlotLifecycle::Fenced;
                if let Some(active) = actor.take() {
                    stop_endpoint_actor(active).await;
                }
                let mut current = status.write().await;
                current.actor = None;
            }
            SlotInput::Fault(fault) => {
                tracing::error!(
                    %endpoint,
                    worker_id = fault.worker_id,
                    dp_rank = fault.dp_rank,
                    event_id = ?fault.event_id,
                    category = ?fault.category,
                    error = %fault.message,
                    "KV DC Relay actor failed an admitted mutation"
                );
                match fault.disposition.action {
                    CkfFailureAction::ContinueCapacityOmission => {}
                    CkfFailureAction::ReportResourceFailure => {
                        if let Some(active) = actor.as_ref() {
                            let disposition = active
                                .recovery
                                .client()
                                .handle_target_fault(
                                    fault.worker_id,
                                    fault.dp_rank,
                                    fault.source_epoch,
                                    false,
                                )
                                .await;
                            if disposition == TargetFaultDisposition::Fenced {
                                active
                                    .recovery
                                    .client()
                                    .reject_source(
                                        fault.worker_id,
                                        fault.dp_rank,
                                        fault.source_epoch,
                                    )
                                    .await;
                                status.write().await.lifecycle = SlotLifecycle::Fenced;
                            }
                        }
                    }
                    CkfFailureAction::RejectSource => {
                        if let Some(active) = actor.as_ref() {
                            active
                                .recovery
                                .client()
                                .reject_source(fault.worker_id, fault.dp_rank, fault.source_epoch)
                                .await;
                        }
                    }
                    CkfFailureAction::FenceAndRebuildProducer => {
                        // The producer's exact state is suspect. Retire its publisher and source
                        // bindings before the slot loop constructs a fresh layout generation.
                        status.write().await.lifecycle = SlotLifecycle::Fenced;
                        if let Some(active) = actor.take() {
                            fence_endpoint_actor(active).await;
                        }
                        status.write().await.actor = None;
                    }
                    CkfFailureAction::DeactivateAndSnapshot | CkfFailureAction::RetrySnapshot => {
                        unreachable!("consumer-lane disposition cannot originate from Relay actor")
                    }
                }
            }
            SlotInput::Cancelled => break,
        }

        #[cfg(feature = "ckf-diagnostics")]
        if let Some(active) = &actor {
            status.write().await.recovery = active.recovery.client().health_snapshot().await;
        }
    }

    if let Some(active) = actor {
        status.write().await.lifecycle = SlotLifecycle::Draining;
        stop_endpoint_actor(active).await;
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
async fn start_endpoint_actor(
    component: Component,
    dc_id: Arc<str>,
    process_incarnation: u64,
    endpoint: EndpointId,
    layout_generation: u64,
    binding: ActorBinding,
    membership_watch: KvSourceMembershipWatch,
    rebuild_permit: Arc<Semaphore>,
    recovery_fetch_permit: Arc<Semaphore>,
    publication: ActorPublicationConfig,
    recovery_attempt_timeout: Duration,
    cancel: CancellationToken,
) -> anyhow::Result<EndpointActorRuntime> {
    let mut config = CkfConfig::new(DEFAULT_EXPECTED_UNIQUE_BLOCKS);
    config.publish_every_n_events = publication.threshold;
    let ckf_dc_id = stable_dc_id(dc_id.as_ref());
    let scope = StreamScope {
        process_incarnation,
        layout_generation,
        pool_binding: PoolBinding::new(
            PoolId::new(binding.domain.id, ckf_dc_id),
            EndpointLocator::new(ckf_dc_id, endpoint.clone()),
            Some(EndpointLocator::new(
                ckf_dc_id,
                binding.kv_state_endpoint.clone(),
            )),
        ),
    };
    let (handle, faults) =
        KvDcRelayHandle::spawn_with_publication_delay(config, scope, publication.delay)?;
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
        handle.clone(),
        rebuild_permit,
        initial_recoveries,
        recovery_attempt_timeout,
    );
    let recovery = match start_target_subscriber(
        component,
        endpoint,
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
            let _ = handle.shutdown().await;
            return Err(error);
        }
    };
    Ok(EndpointActorRuntime {
        handle,
        recovery,
        faults,
        binding,
    })
}

async fn stop_endpoint_actor(active: EndpointActorRuntime) {
    active.recovery.shutdown().await;
    if let Err(error) = active.handle.shutdown().await {
        tracing::warn!(%error, endpoint = %active.handle.scope.pool_binding.serving_endpoint().endpoint_id(), "Failed to drain KV DC Relay endpoint actor");
    }
}

async fn fence_endpoint_actor(active: EndpointActorRuntime) {
    // Stop publication first. Recovery shutdown may attempt rank resets, but a producer whose
    // exact state is suspect must not emit another apparently valid delta while being retired.
    if let Err(error) = active.handle.fence().await {
        tracing::warn!(%error, endpoint = %active.handle.scope.pool_binding.serving_endpoint().endpoint_id(), "Failed to fence KV DC Relay endpoint actor cleanly");
    }
    active.recovery.shutdown().await;
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
                    .map(|(worker, blocks)| KvDcRelayMemberStats {
                        worker_id: worker.worker_id,
                        dp_rank: worker.dp_rank,
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
        serving_endpoint: endpoint.to_string(),
        lifecycle: status.lifecycle.as_str().to_string(),
        layout_generation: status.layout_generation,
        cache_domain: membership
            .as_ref()
            .and_then(|membership| membership.domain.as_ref())
            .map(cache_domain_stats),
        compatibility_conflict: membership
            .as_ref()
            .is_some_and(|membership| membership.compatibility_conflict),
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
    use dynamo_kv_router::identity::{
        CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, RoutingScopeId,
    };

    use super::*;
    use crate::kv_dc_relay::resolution::ResolvedIndexerDomain;

    fn membership(endpoint: &str, domain: ResolvedIndexerDomain) -> EndpointMembership {
        EndpointMembership {
            endpoint: EndpointId::from(endpoint),
            generation: 1,
            domain: Some(domain),
            compatibility_conflict: false,
            models: Vec::new(),
            aliases: Vec::new(),
            roles: Vec::new(),
            runtime_configs: HashMap::new(),
        }
    }

    #[test]
    fn simultaneous_endpoints_cannot_own_one_pool() {
        let domain_id = IndexerDomainId::new(
            CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
            RoutingScopeId::new([2; 16], IdentitySource::Explicit),
        );
        let domain = ResolvedIndexerDomain {
            id: domain_id,
            diagnostic_model_artifact: "model".to_string(),
            kv_block_size: 512,
            event_hash_format: 1,
        };
        let first = membership("ns/router/first", domain.clone());
        let second = membership("ns/router/second", domain);
        let mut view = DcMembershipView {
            endpoints: HashMap::from([
                (first.endpoint.clone(), first),
                (second.endpoint.clone(), second),
            ]),
        };

        reject_duplicate_live_pools(&mut view, DcId::new(7));

        assert!(
            view.endpoints
                .values()
                .all(|membership| membership.compatibility_conflict)
        );
    }
}
