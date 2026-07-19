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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dynamo_kv_router::indexer::cuckoo::{
    CacheDomainId, CacheDomainIdentity, CkfConfig, CkfFailureAction, CkfFailureDisposition,
    CkfFailurePoint, DcCkfDelta, DcCkfDeltaSink, DcCkfPublisher, DcCkfSnapshot, DcCkfState,
    DcCkfStats, DcId as CkfDcId, EndpointId as CkfEndpointId, LaneLease, ProducerIdentity,
    PublisherEmitOutcome,
};
use dynamo_kv_router::protocols::{
    DpRank, ExternalSequenceBlockHash, KvCacheEventData, KvCacheEventError, RouterEvent,
    StorageTier, WorkerId, WorkerWithDpRank,
};
use dynamo_runtime::component::Component;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, RwLock, Semaphore, broadcast, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::discovery::{KvSourceMembershipCoordinator, KvSourceMembershipWatch};
use crate::kv_dc_relay_discovery::{
    DcDiscoveryFilter, DcMembershipView, DcMembershipWatch, EndpointMembership, KvCacheDomainKey,
};
use crate::kv_router::indexer::{
    RecoveryResetReason, RecoverySupervisor, RecoveryTarget, SourceEpoch, TargetFaultDisposition,
    WorkerQueryHealthSnapshot, start_target_subscriber,
};

pub const DEFAULT_EXPECTED_UNIQUE_BLOCKS: usize = 1_048_576;
const DEFAULT_MAILBOX_CAPACITY: usize = 256;
const DEFAULT_PENDING_BLOCK_PERMITS: usize = 65_536;
const DEFAULT_PUBLICATION_CAPACITY: usize = 64;
const DEFAULT_FAULT_CAPACITY: usize = 16;
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
}

impl Default for KvDcRelayConfig {
    fn default() -> Self {
        Self {
            namespace_filter: None,
            endpoint_prefix: None,
            publication_threshold: DEFAULT_PUBLICATION_THRESHOLD,
            publication_delay_ms: DEFAULT_PUBLICATION_DELAY.as_millis() as u64,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ActorPublicationConfig {
    threshold: usize,
    delay: Duration,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayStats {
    pub identity: KvDcRelayIdentityStats,
    pub endpoints: Vec<KvDcRelayEndpointStats>,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayIdentityStats {
    pub dc_id: String,
    pub process_incarnation: u64,
}

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

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayCacheDomainStats {
    pub model_artifact: String,
    pub kv_block_size: u32,
    pub event_hash_format: u16,
}

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayMemberStats {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub blocks: usize,
}

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

#[derive(Debug, Clone, Serialize)]
#[non_exhaustive]
pub struct KvDcRelayMemoryStats {
    pub filter_bytes: usize,
    pub dirty_tracking_bytes: usize,
    pub member_set_capacity: usize,
    pub refcount_capacity: usize,
    pub insertion_scratch_capacity: usize,
}

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

#[derive(Debug)]
// NOTE: This is the intentional producer snapshot/delta seam for the future non-local adapter.
// It remains crate-private until that transport has delivery cursors and recovery semantics.
#[allow(dead_code)]
pub(crate) struct DcCkfSubscription {
    pub(crate) snapshot: DcCkfSnapshot,
    pub(crate) deltas: broadcast::Receiver<DcCkfDelta>,
}

#[derive(Debug, Default)]
struct ActorCounters {
    mailbox_wait_ns: AtomicU64,
    mailbox_max_wait_ns: AtomicU64,
    degraded_resets: AtomicU64,
    publications: AtomicU64,
    unchanged_publications: AtomicU64,
    rebuild_count: AtomicU64,
    rebuild_ns: AtomicU64,
    rebuild_max_ns: AtomicU64,
}

#[derive(Debug, Default)]
struct ActorActivity {
    active_command: Option<&'static str>,
    active_since: Option<Instant>,
    shutting_down: bool,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActorFaultCategory {
    Resource,
    SourceProtocol,
    ProducerInvariant,
}

#[derive(Debug)]
struct ActorFault {
    worker_id: WorkerId,
    dp_rank: DpRank,
    source_epoch: SourceEpoch,
    event_id: Option<u64>,
    category: ActorFaultCategory,
    disposition: CkfFailureDisposition,
    message: String,
}

// NOTE: Recovery is selected from the state whose commit became uncertain—not merely from the
// error's name.
fn event_failure_point(error: KvCacheEventError) -> CkfFailurePoint {
    match error {
        KvCacheEventError::CapacityExhausted => CkfFailurePoint::BoundedRelocationFailure,
        KvCacheEventError::AllocationFailed => CkfFailurePoint::PrecommitAllocationFailure,
        KvCacheEventError::OwnershipDegreeOverflow
        | KvCacheEventError::ParentBlockNotFound
        | KvCacheEventError::BlockNotFound
        | KvCacheEventError::InvalidBlockSequence => CkfFailurePoint::SourceProtocolFailure,
        KvCacheEventError::IndexerInvariantViolation => CkfFailurePoint::PrewriteInvariantMismatch,
        _ => CkfFailurePoint::SourceProtocolFailure,
    }
}

fn actor_fault_category(disposition: CkfFailureDisposition) -> ActorFaultCategory {
    match disposition.action {
        CkfFailureAction::ReportResourceFailure => ActorFaultCategory::Resource,
        CkfFailureAction::RejectSource => ActorFaultCategory::SourceProtocol,
        CkfFailureAction::FenceAndRebuildProducer | CkfFailureAction::ContinueCapacityOmission => {
            ActorFaultCategory::ProducerInvariant
        }
        CkfFailureAction::DeactivateAndSnapshot | CkfFailureAction::RetrySnapshot => {
            unreachable!("consumer-lane disposition cannot originate from a producer event")
        }
    }
}

#[derive(Debug, Clone)]
struct StreamScope {
    process_incarnation: u64,
    serving_endpoint: EndpointId,
    layout_generation: u64,
    cache_domain: CacheDomainIdentity,
    ckf_dc_id: CkfDcId,
    ckf_endpoint_id: CkfEndpointId,
}

#[derive(Debug, Clone)]
struct BroadcastDeltaSink {
    sender: broadcast::Sender<DcCkfDelta>,
}

impl DcCkfDeltaSink for BroadcastDeltaSink {
    type Error = broadcast::error::SendError<DcCkfDelta>;

    fn enqueue(&mut self, delta: DcCkfDelta) -> Result<(), Self::Error> {
        self.sender.send(delta).map(|_| ())
    }
}

#[derive(Debug, Clone)]
pub(crate) struct KvDcRelayHandle {
    sender: mpsc::Sender<ActorCommand>,
    payload_permits: Arc<Semaphore>,
    counters: Arc<ActorCounters>,
    activity: Arc<Mutex<ActorActivity>>,
    scope: StreamScope,
}

impl KvDcRelayHandle {
    #[cfg(test)]
    fn spawn(
        config: CkfConfig,
        scope: StreamScope,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        Self::spawn_with_capacity_and_delay(
            config,
            scope,
            DEFAULT_MAILBOX_CAPACITY,
            DEFAULT_PUBLICATION_DELAY,
        )
    }

    fn spawn_with_publication_delay(
        config: CkfConfig,
        scope: StreamScope,
        publication_delay: Duration,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        Self::spawn_with_capacity_and_delay(
            config,
            scope,
            DEFAULT_MAILBOX_CAPACITY,
            publication_delay,
        )
    }

    #[cfg(test)]
    fn spawn_with_capacity(
        config: CkfConfig,
        scope: StreamScope,
        capacity: usize,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        Self::spawn_with_capacity_and_delay(config, scope, capacity, DEFAULT_PUBLICATION_DELAY)
    }

    fn spawn_with_capacity_and_delay(
        config: CkfConfig,
        scope: StreamScope,
        capacity: usize,
        publication_delay: Duration,
    ) -> Result<(Self, mpsc::Receiver<ActorFault>), KvDcRelayError> {
        let state = DcCkfState::new(config)?;
        let (sender, receiver) = mpsc::channel(capacity);
        let timer_sender = sender.clone();
        let publication_timer_armed = Arc::new(AtomicBool::new(false));
        let timer_armed = Arc::clone(&publication_timer_armed);
        tokio::spawn(async move {
            let mut timer = tokio::time::interval(publication_delay);
            timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            timer.tick().await;
            loop {
                timer.tick().await;
                if !timer_armed.swap(false, Ordering::AcqRel) {
                    continue;
                }
                match timer_sender.try_send(ActorCommand::PublishTimer) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        timer_armed.store(true, Ordering::Release);
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => return,
                }
            }
        });
        let (publication_tx, _) = broadcast::channel(DEFAULT_PUBLICATION_CAPACITY);
        let identity = ProducerIdentity::new(
            scope.cache_domain,
            scope.ckf_dc_id,
            scope.ckf_endpoint_id,
            scope.process_incarnation,
            scope.layout_generation,
            state.format(),
        );
        let publisher = DcCkfPublisher::new(
            identity,
            0,
            BroadcastDeltaSink {
                sender: publication_tx.clone(),
            },
        );
        let (fault_tx, fault_rx) = mpsc::channel(DEFAULT_FAULT_CAPACITY);
        let counters = Arc::new(ActorCounters::default());
        let activity = Arc::new(Mutex::new(ActorActivity::default()));
        tokio::spawn(run_actor(
            state,
            publisher,
            receiver,
            publication_timer_armed,
            fault_tx,
            counters.clone(),
            activity.clone(),
        ));
        Ok((
            Self {
                sender,
                payload_permits: Arc::new(Semaphore::new(DEFAULT_PENDING_BLOCK_PERMITS)),
                counters,
                activity,
                scope,
            },
            fault_rx,
        ))
    }

    async fn submit<T>(
        &self,
        make_command: impl FnOnce(oneshot::Sender<Result<T, KvDcRelayError>>) -> ActorCommand,
    ) -> Result<T, KvDcRelayError> {
        let (response_tx, response_rx) = oneshot::channel();
        let wait_started = Instant::now();
        self.sender
            .send(make_command(response_tx))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        self.record_mailbox_wait(wait_started);
        response_rx
            .await
            .map_err(|_| KvDcRelayError::ActorStopped)?
    }

    pub(crate) async fn admit_event(
        &self,
        source_epoch: SourceEpoch,
        event: RouterEvent,
    ) -> Result<(), KvDcRelayError> {
        let weight = event_payload_weight(&event).min(DEFAULT_PENDING_BLOCK_PERMITS) as u32;
        let wait_started = Instant::now();
        let permit = self
            .payload_permits
            .clone()
            .acquire_many_owned(weight.max(1))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        self.sender
            .send(ActorCommand::Apply {
                source_epoch,
                event,
                _payload_permit: permit,
            })
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        self.record_mailbox_wait(wait_started);
        Ok(())
    }

    async fn replace_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> Result<(), KvDcRelayError> {
        let weight = events
            .iter()
            .map(event_payload_weight)
            .fold(0usize, usize::saturating_add)
            .min(DEFAULT_PENDING_BLOCK_PERMITS) as u32;
        let permit = self
            .payload_permits
            .clone()
            .acquire_many_owned(weight.max(1))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        self.submit(|response| ActorCommand::ReplaceRank {
            source_epoch,
            worker_id,
            dp_rank,
            events,
            _payload_permit: permit,
            response,
        })
        .await
    }

    async fn reset_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        degraded: bool,
    ) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::ResetRank {
            source_epoch,
            worker_id,
            dp_rank,
            degraded,
            response,
        })
        .await
    }

    async fn flush(&self) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::Flush { response })
            .await
    }

    async fn snapshot(&self) -> Result<ActorSnapshot, KvDcRelayError> {
        self.submit(|response| ActorCommand::Snapshot { response })
            .await
    }

    async fn state_stats(
        &self,
    ) -> Result<(DcCkfStats, u64, Vec<(WorkerWithDpRank, usize)>), KvDcRelayError> {
        self.submit(|response| ActorCommand::Stats { response })
            .await
    }

    // Kept with `DcCkfSubscription` as the crate-private producer boundary described above.
    #[allow(dead_code)]
    pub(crate) async fn subscribe(
        &self,
        lease: LaneLease,
    ) -> Result<DcCkfSubscription, KvDcRelayError> {
        let subscription = self
            .submit(|response| ActorCommand::Subscribe { lease, response })
            .await?;
        Ok(DcCkfSubscription {
            snapshot: subscription.snapshot,
            deltas: subscription.deltas,
        })
    }

    async fn shutdown(&self) -> Result<(), KvDcRelayError> {
        self.submit(|response| ActorCommand::Shutdown { response })
            .await
    }

    fn mailbox_depth(&self) -> usize {
        self.sender
            .max_capacity()
            .saturating_sub(self.sender.capacity())
    }

    fn record_mailbox_wait(&self, started: Instant) {
        let waited = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.counters
            .mailbox_wait_ns
            .fetch_add(waited, Ordering::Relaxed);
        self.counters
            .mailbox_max_wait_ns
            .fetch_max(waited, Ordering::Relaxed);
    }
}

#[derive(Clone)]
struct KvDcRelayRecoveryTarget {
    handle: KvDcRelayHandle,
    rebuild_permit: Arc<Semaphore>,
}

impl RecoveryTarget for KvDcRelayRecoveryTarget {
    async fn admit_event(
        &self,
        source_epoch: SourceEpoch,
        event: RouterEvent,
    ) -> anyhow::Result<()> {
        self.handle
            .admit_event(source_epoch, event)
            .await
            .map_err(Into::into)
    }

    async fn replace_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> anyhow::Result<()> {
        let _permit = self
            .rebuild_permit
            .acquire()
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        self.handle
            .replace_rank(source_epoch, worker_id, dp_rank, events)
            .await
            .map_err(Into::into)
    }

    async fn reset_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        reason: RecoveryResetReason,
    ) -> anyhow::Result<()> {
        self.handle
            .reset_rank(
                source_epoch,
                worker_id,
                dp_rank,
                reason == RecoveryResetReason::TreeDumpFailed,
            )
            .await
            .map_err(Into::into)
    }
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
    recovery: WorkerQueryHealthSnapshot,
}

impl Default for EndpointSlotStatus {
    fn default() -> Self {
        Self {
            lifecycle: SlotLifecycle::Lightweight,
            layout_generation: 0,
            membership: None,
            actor: None,
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
    dc_id: Arc<str>,
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
            cancel.child_token(),
        ));
        Ok(Self {
            dc_id,
            process_incarnation,
            cancel,
            membership: Mutex::new(Some(membership)),
            supervisor: Mutex::new(Some(supervisor)),
            statuses,
        })
    }

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

async fn run_host_supervisor(
    component: Component,
    dc_id: Arc<str>,
    process_incarnation: u64,
    mut membership_rx: watch::Receiver<DcMembershipView>,
    statuses: Arc<RwLock<HashMap<EndpointId, SharedEndpointStatus>>>,
    publication: ActorPublicationConfig,
    cancel: CancellationToken,
) {
    let rebuild_permit = Arc::new(Semaphore::new(1));
    let recovery_fetch_permit = Arc::new(Semaphore::new(DEFAULT_RECOVERY_FETCH_CONCURRENCY));
    let mut slots: HashMap<EndpointId, EndpointSlotTask> = HashMap::new();

    loop {
        let view = membership_rx.borrow_and_update().clone();
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
                    rebuild_permit.clone(),
                    recovery_fetch_permit.clone(),
                    publication,
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
                current.recovery = WorkerQueryHealthSnapshot::default();
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
                    None => SlotInput::Metadata,
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(1)), if actor.is_some() => SlotInput::Health,
        };
        match input {
            SlotInput::Metadata | SlotInput::Source | SlotInput::Health => {}
            SlotInput::Fault(fault) => {
                let Some(active) = actor.as_ref() else {
                    continue;
                };
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
                    CkfFailureAction::ReportResourceFailure
                    | CkfFailureAction::RejectSource
                    | CkfFailureAction::DeactivateAndSnapshot
                    | CkfFailureAction::RetrySnapshot => {
                        // These failures are already observable at their known commit boundary.
                        // They must not be converted into an exact-state producer rebuild.
                    }
                    CkfFailureAction::FenceAndRebuildProducer => {
                        let disposition = active
                            .recovery
                            .client()
                            .handle_target_fault(
                                fault.worker_id,
                                fault.dp_rank,
                                fault.source_epoch,
                                true,
                            )
                            .await;
                        if disposition == TargetFaultDisposition::Fenced {
                            status.write().await.lifecycle = SlotLifecycle::Fenced;
                        }
                    }
                }
            }
            SlotInput::Cancelled => break,
        }

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
    cancel: CancellationToken,
) -> anyhow::Result<EndpointActorRuntime> {
    let mut config = CkfConfig::new(DEFAULT_EXPECTED_UNIQUE_BLOCKS);
    config.publish_every_n_events = publication.threshold;
    let scope = StreamScope {
        process_incarnation,
        cache_domain: CacheDomainIdentity::new(
            CacheDomainId::new(stable_identity_key(
                b"kv-cache-domain",
                &[binding.domain.model_artifact.as_str()],
            )),
            binding.domain.kv_block_size,
            binding.domain.event_hash_format,
        ),
        ckf_dc_id: CkfDcId::new(stable_identity_key(b"dc", &[dc_id.as_ref()])),
        ckf_endpoint_id: CkfEndpointId::new(stable_identity_key(
            b"endpoint",
            &[
                endpoint.namespace.as_str(),
                endpoint.component.as_str(),
                endpoint.name.as_str(),
            ],
        )),
        serving_endpoint: endpoint.clone(),
        layout_generation,
    };
    let (handle, faults) =
        KvDcRelayHandle::spawn_with_publication_delay(config, scope, publication.delay)?;
    let target = KvDcRelayRecoveryTarget {
        handle: handle.clone(),
        rebuild_permit,
    };
    let recovery = match start_target_subscriber(
        component,
        endpoint,
        target,
        membership_watch,
        "kv-dc-relay".to_string(),
        "kv_dc_relay",
        recovery_fetch_permit,
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

/// Convert cold external identity strings to compact wire keys before they enter CKF state.
/// String-keyed discovery maps retain randomized hashing; the mutation/query hot paths see only
/// these numeric identities.
fn stable_identity_key(domain: &[u8], parts: &[&str]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(&(part.len() as u64).to_le_bytes());
        hasher.update(part.as_bytes());
    }
    let bytes = hasher.finalize();
    u64::from_le_bytes(
        bytes.as_bytes()[..8]
            .try_into()
            .expect("BLAKE3 output is 32 bytes"),
    )
}

async fn stop_endpoint_actor(active: EndpointActorRuntime) {
    active.recovery.shutdown().await;
    if let Err(error) = active.handle.shutdown().await {
        tracing::warn!(%error, endpoint = %active.handle.scope.serving_endpoint, "Failed to drain KV DC Relay endpoint actor");
    }
}

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
                publication_count: actor.counters.publications.load(Ordering::Relaxed),
                unchanged_publication_count: actor
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
                actor.counters.degraded_resets.load(Ordering::Relaxed)
            }),
            rebuild_count: status.actor.as_ref().map_or(0, |actor| {
                actor.counters.rebuild_count.load(Ordering::Relaxed)
            }),
            rebuild_ns: status
                .actor
                .as_ref()
                .map_or(0, |actor| actor.counters.rebuild_ns.load(Ordering::Relaxed)),
            rebuild_max_ns: status.actor.as_ref().map_or(0, |actor| {
                actor.counters.rebuild_max_ns.load(Ordering::Relaxed)
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

fn cache_domain_stats(domain: &KvCacheDomainKey) -> KvDcRelayCacheDomainStats {
    KvDcRelayCacheDomainStats {
        model_artifact: domain.model_artifact.clone(),
        kv_block_size: domain.kv_block_size,
        event_hash_format: domain.event_hash_format,
    }
}

fn actor_health(handle: &KvDcRelayHandle) -> KvDcRelayActorStats {
    let activity = handle.activity.lock();
    KvDcRelayActorStats {
        mailbox_depth: handle.mailbox_depth(),
        mailbox_capacity: handle.sender.max_capacity(),
        mailbox_wait_ns: handle.counters.mailbox_wait_ns.load(Ordering::Relaxed),
        mailbox_max_wait_ns: handle.counters.mailbox_max_wait_ns.load(Ordering::Relaxed),
        active_command: activity.active_command.map(str::to_string),
        active_command_age_ms: activity
            .active_since
            .map(|started| started.elapsed().as_millis().min(u64::MAX as u128) as u64),
        shutting_down: activity.shutting_down,
        faulted: activity.last_error.is_some(),
        last_error: activity.last_error.clone(),
    }
}

fn clear_actor_activity(activity: &Mutex<ActorActivity>) {
    let mut active = activity.lock();
    active.active_command = None;
    active.active_since = None;
}

type ActorStatsResult = Result<(DcCkfStats, u64, Vec<(WorkerWithDpRank, usize)>), KvDcRelayError>;

enum ActorCommand {
    Apply {
        source_epoch: SourceEpoch,
        event: RouterEvent,
        _payload_permit: OwnedSemaphorePermit,
    },
    ReplaceRank {
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
        _payload_permit: OwnedSemaphorePermit,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    ResetRank {
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        degraded: bool,
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    Flush {
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    PublishTimer,
    Snapshot {
        response: oneshot::Sender<Result<ActorSnapshot, KvDcRelayError>>,
    },
    Subscribe {
        lease: LaneLease,
        response: oneshot::Sender<Result<ActorSubscription, KvDcRelayError>>,
    },
    Stats {
        response: oneshot::Sender<ActorStatsResult>,
    },
    Shutdown {
        response: oneshot::Sender<Result<(), KvDcRelayError>>,
    },
    #[cfg(test)]
    Pause {
        entered: oneshot::Sender<()>,
        release: oneshot::Receiver<()>,
    },
}

struct ActorSnapshot {
    identity: ProducerIdentity,
    sequence: u64,
    buckets: Box<[u64]>,
    stats: DcCkfStats,
}

struct ActorSubscription {
    snapshot: DcCkfSnapshot,
    deltas: broadcast::Receiver<DcCkfDelta>,
}

impl ActorCommand {
    fn kind(&self) -> &'static str {
        match self {
            Self::Apply { .. } => "apply_event",
            Self::ReplaceRank { .. } => "replace_rank",
            Self::ResetRank { .. } => "reset_rank",
            Self::Flush { .. } => "flush",
            Self::PublishTimer => "publish_timer",
            Self::Snapshot { .. } => "snapshot",
            Self::Subscribe { .. } => "subscribe",
            Self::Stats { .. } => "stats",
            Self::Shutdown { .. } => "shutdown",
            #[cfg(test)]
            Self::Pause { .. } => "test_pause",
        }
    }
}

async fn run_actor(
    mut state: DcCkfState,
    mut publisher: DcCkfPublisher<BroadcastDeltaSink>,
    mut receiver: mpsc::Receiver<ActorCommand>,
    publication_timer_armed: Arc<AtomicBool>,
    fault_tx: mpsc::Sender<ActorFault>,
    counters: Arc<ActorCounters>,
    activity: Arc<Mutex<ActorActivity>>,
) {
    let mut source_epochs = FxHashMap::<WorkerWithDpRank, SourceEpoch>::default();
    let mut capacity_omission_events = 0u64;
    let mut shutdown_response = None;
    while let Some(command) = receiver.recv().await {
        let kind = command.kind();
        {
            let mut active = activity.lock();
            active.active_command = Some(kind);
            active.active_since = Some(Instant::now());
        }
        match command {
            ActorCommand::Apply {
                source_epoch,
                event,
                ..
            } => {
                let worker_id = event.worker_id;
                let dp_rank = event.event.dp_rank;
                let event_id = event.event.event_id;
                let key = WorkerWithDpRank::new(worker_id, dp_rank);
                let current_epoch = source_epochs.get(&key).copied();
                if current_epoch.is_some_and(|current| source_epoch < current) {
                    tracing::debug!(
                        worker_id,
                        dp_rank,
                        event_id,
                        source_epoch = source_epoch.get(),
                        current_epoch = current_epoch.expect("guarded current epoch").get(),
                        "Dropping an admitted KV mutation from a superseded source epoch"
                    );
                    clear_actor_activity(activity.as_ref());
                    continue;
                }
                if let Some(current_epoch) = current_epoch
                    && source_epoch > current_epoch
                {
                    let disposition = CkfFailurePoint::SourceProtocolFailure.disposition();
                    let message = format!(
                        "source epoch advanced from {} to {} without a reset or replacement barrier",
                        current_epoch.get(),
                        source_epoch.get()
                    );
                    activity.lock().last_error = Some(message.clone());
                    if fault_tx
                        .send(ActorFault {
                            worker_id,
                            dp_rank,
                            source_epoch,
                            event_id: Some(event_id),
                            category: actor_fault_category(disposition),
                            disposition,
                            message,
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                    clear_actor_activity(activity.as_ref());
                    continue;
                }
                source_epochs.entry(key).or_insert(source_epoch);
                let outcome = state.apply_event(event);
                let first_error = outcome.first_error().copied();
                let publication_boundary = outcome.publication_boundary();
                if outcome.unknown_removals() != 0 {
                    tracing::warn!(
                        worker_id,
                        dp_rank,
                        event_id,
                        unknown_removals = outcome.unknown_removals(),
                        "Ignoring KV DC Relay removals not owned by this worker/rank"
                    );
                }
                if let Some(batch) = outcome.into_publication() {
                    if let Err(error) = publish_batch(batch, &mut publisher, &counters) {
                        activity.lock().last_error = Some(error.to_string());
                    }
                } else if publication_boundary {
                    counters
                        .unchanged_publications
                        .fetch_add(1, Ordering::Relaxed);
                }
                if publication_boundary {
                    publication_timer_armed.store(false, Ordering::Release);
                } else if state.pending_event_count() == 1 {
                    publication_timer_armed.store(true, Ordering::Release);
                }
                if let Some(error) = first_error {
                    let disposition = event_failure_point(error).disposition();
                    let message = error.to_string();
                    if disposition.action == CkfFailureAction::ContinueCapacityOmission {
                        // NOTE: A bounded relocation miss is a pre-commit lossy-index omission.
                        // Do not turn it into a lifecycle fault: the affected block is unchanged,
                        // successful sibling blocks remain committed, a later Store may retry, and
                        // an omitted hash's Remove remains a safe no-op.
                        capacity_omission_events = capacity_omission_events.saturating_add(1);
                        if capacity_omission_events == 1 {
                            tracing::warn!(
                                worker_id,
                                dp_rank,
                                event_id,
                                capacity_omission_events,
                                "KV DC Relay omitted a capacity-exhausted mutation; service continues"
                            );
                        } else if capacity_omission_events.is_power_of_two() {
                            tracing::debug!(
                                worker_id,
                                dp_rank,
                                event_id,
                                capacity_omission_events,
                                "KV DC Relay continues after repeated capacity omissions"
                            );
                        }
                        clear_actor_activity(activity.as_ref());
                        continue;
                    }
                    let category = actor_fault_category(disposition);
                    activity.lock().last_error = Some(message.clone());
                    if fault_tx
                        .send(ActorFault {
                            worker_id,
                            dp_rank,
                            source_epoch,
                            event_id: Some(event_id),
                            category,
                            disposition,
                            message,
                        })
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
            ActorCommand::ReplaceRank {
                source_epoch,
                worker_id,
                dp_rank,
                events,
                _payload_permit: _,
                response,
            } => {
                let key = WorkerWithDpRank::new(worker_id, dp_rank);
                if let Some(current) = source_epochs.get(&key).copied()
                    && source_epoch < current
                {
                    let _ = response.send(Err(KvDcRelayError::StaleSourceEpoch {
                        worker_id,
                        dp_rank,
                        current: current.get(),
                        received: source_epoch.get(),
                    }));
                    clear_actor_activity(activity.as_ref());
                    continue;
                }
                let rebuild_started = Instant::now();
                let result = replacement_hashes(worker_id, dp_rank, events)
                    .and_then(|hashes| {
                        state
                            .replace_rank(WorkerWithDpRank::new(worker_id, dp_rank), hashes)
                            .map_err(Into::into)
                    })
                    .and_then(|publication| {
                        if let Some(batch) = publication {
                            publish_batch(batch, &mut publisher, &counters)?;
                        } else {
                            counters
                                .unchanged_publications
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(())
                    });
                if result.is_ok() {
                    source_epochs.insert(key, source_epoch);
                }
                let elapsed = rebuild_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                counters.rebuild_count.fetch_add(1, Ordering::Relaxed);
                counters.rebuild_ns.fetch_add(elapsed, Ordering::Relaxed);
                counters
                    .rebuild_max_ns
                    .fetch_max(elapsed, Ordering::Relaxed);
                // Replacement is built off-side. A pre-swap failure leaves the old generation
                // unchanged, so its strong response carries the barrier error without an
                // asynchronous producer-recovery fault.
                let _ = response.send(result);
            }
            ActorCommand::ResetRank {
                source_epoch,
                worker_id,
                dp_rank,
                degraded,
                response,
            } => {
                let key = WorkerWithDpRank::new(worker_id, dp_rank);
                if let Some(current) = source_epochs.get(&key).copied()
                    && source_epoch < current
                {
                    let _ = response.send(Err(KvDcRelayError::StaleSourceEpoch {
                        worker_id,
                        dp_rank,
                        current: current.get(),
                        received: source_epoch.get(),
                    }));
                    clear_actor_activity(activity.as_ref());
                    continue;
                }
                let mut removal = state.remove_rank(key);
                if let Err(error) = removal {
                    // Clear may have committed earlier hashes while remaining exact. Retry the
                    // still-tracked suffix once; the strong acknowledgement reports failure if
                    // progress cannot be completed.
                    tracing::warn!(
                        worker_id,
                        dp_rank,
                        source_epoch = source_epoch.get(),
                        %error,
                        "Retrying the remaining tracked hashes after a partial rank reset"
                    );
                    removal = state.remove_rank(key);
                }
                let result = removal
                    .map_err(KvDcRelayError::from)
                    .and_then(|publication| {
                        if degraded {
                            counters.degraded_resets.fetch_add(1, Ordering::Relaxed);
                        }
                        if let Some(batch) = publication {
                            publish_batch(batch, &mut publisher, &counters)?;
                        } else {
                            counters
                                .unchanged_publications
                                .fetch_add(1, Ordering::Relaxed);
                        }
                        Ok(())
                    });
                if result.is_ok() {
                    source_epochs.insert(key, source_epoch);
                }
                let _ = response.send(result);
            }
            ActorCommand::Flush { response } => {
                let result = publish_pending(&mut state, &mut publisher, &counters);
                let _ = response.send(result);
            }
            ActorCommand::PublishTimer => {
                if state.has_pending_publication()
                    && let Err(error) = publish_pending(&mut state, &mut publisher, &counters)
                {
                    activity.lock().last_error = Some(error.to_string());
                }
            }
            ActorCommand::Snapshot { response } => {
                let result = diagnostic_barrier_snapshot(&mut state, &mut publisher, &counters);
                let _ = response.send(result);
            }
            ActorCommand::Subscribe { lease, response } => {
                let result = publisher
                    .snapshot_after_barrier(&mut state, lease)
                    .map_err(|error| KvDcRelayError::Publisher(format!("{error:?}")))
                    .map(|snapshot| {
                        // The actor cannot process a continuation mutation until this command
                        // returns. Subscribe after any old-lease tail so the new receiver starts
                        // exactly after snapshot sequence N.
                        let deltas = publisher.sink().sender.subscribe();
                        ActorSubscription { snapshot, deltas }
                    });
                let _ = response.send(result);
            }
            ActorCommand::Stats { response } => {
                let _ = response.send(Ok((
                    state.stats(),
                    publisher.last_sequence(),
                    state.member_counts(),
                )));
            }
            ActorCommand::Shutdown { response } => {
                if shutdown_response.is_some() {
                    let _ = response.send(Err(KvDcRelayError::ShuttingDown));
                } else {
                    receiver.close();
                    activity.lock().shutting_down = true;
                    shutdown_response = Some(response);
                }
            }
            #[cfg(test)]
            ActorCommand::Pause { entered, release } => {
                let _ = entered.send(());
                let _ = release.await;
            }
        }
        clear_actor_activity(activity.as_ref());
    }

    if let Err(error) = publish_pending(&mut state, &mut publisher, &counters) {
        activity.lock().last_error = Some(error.to_string());
    }
    publisher.retire_lease();
    drop(fault_tx);
    if let Some(response) = shutdown_response {
        let _ = response.send(Ok(()));
    }
}

fn replacement_hashes(
    worker_id: WorkerId,
    dp_rank: DpRank,
    events: Vec<RouterEvent>,
) -> Result<FxHashSet<ExternalSequenceBlockHash>, KvDcRelayError> {
    let mut hashes = FxHashSet::default();
    for event in events {
        if event.worker_id != worker_id || event.event.dp_rank != dp_rank {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id,
                dp_rank,
                message: "event identity does not match replacement rank".to_string(),
            });
        }
        if event.storage_tier != StorageTier::Device {
            continue;
        }
        let KvCacheEventData::Stored(store) = event.event.data else {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id,
                dp_rank,
                message: "tree dump contains a non-Stored event".to_string(),
            });
        };
        hashes.try_reserve(store.blocks.len()).map_err(|_| {
            KvDcRelayError::Build(
                dynamo_kv_router::indexer::cuckoo::CkfBuildError::AllocationFailed,
            )
        })?;
        hashes.extend(store.blocks.into_iter().map(|block| block.block_hash));
    }
    Ok(hashes)
}

fn publish_batch(
    batch: dynamo_kv_router::indexer::cuckoo::DcCkfPublicationBatch,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    counters: &ActorCounters,
) -> Result<(), KvDcRelayError> {
    match publisher.publish(batch) {
        Ok(PublisherEmitOutcome::Published { .. }) => {
            counters.publications.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        Ok(PublisherEmitOutcome::NoSubscriber { .. }) => {
            counters
                .unchanged_publications
                .fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        Err(error) => Err(KvDcRelayError::Publisher(format!("{error:?}"))),
    }
}

fn publish_pending(
    state: &mut DcCkfState,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    counters: &ActorCounters,
) -> Result<(), KvDcRelayError> {
    let Some(batch) = state.flush() else {
        return Ok(());
    };
    publish_batch(batch, publisher, counters)
}

fn diagnostic_barrier_snapshot(
    state: &mut DcCkfState,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    counters: &ActorCounters,
) -> Result<ActorSnapshot, KvDcRelayError> {
    let (publication, buckets) = state.barrier_snapshot()?;
    if let Some(batch) = publication {
        publish_batch(batch, publisher, counters)?;
    }
    Ok(ActorSnapshot {
        identity: publisher.identity(),
        sequence: publisher.last_sequence(),
        buckets,
        stats: state.stats(),
    })
}

fn event_payload_weight(event: &RouterEvent) -> usize {
    match &event.event.data {
        KvCacheEventData::Stored(store) => store.blocks.len().max(1),
        KvCacheEventData::Removed(remove) => remove.block_hashes.len().max(1),
        KvCacheEventData::Cleared => 1,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_kv_router::indexer::cuckoo::{CkfCommitState, CkfFailureDomain, ConsumerInstanceId};
    use dynamo_kv_router::protocols::{
        KvCacheEvent, KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    };

    use super::*;

    fn scope(name: &str) -> StreamScope {
        let endpoint = format!("ns.worker.{name}");
        StreamScope {
            process_incarnation: 1,
            serving_endpoint: EndpointId::from(endpoint.as_str()),
            layout_generation: 1,
            cache_domain: CacheDomainIdentity::new(CacheDomainId::new(1), 512, 1),
            ckf_dc_id: CkfDcId::new(2),
            ckf_endpoint_id: CkfEndpointId::new(3),
        }
    }

    fn lease(epoch: u64) -> LaneLease {
        LaneLease::new(ConsumerInstanceId::new(4), 0, epoch)
    }

    fn stored(worker: WorkerWithDpRank, event_id: u64, hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            worker.worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: hashes
                        .iter()
                        .copied()
                        .map(|hash| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(hash),
                            tokens_hash: LocalBlockHash(hash),
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
                dp_rank: worker.dp_rank,
            },
        )
    }

    async fn pause_actor(handle: &KvDcRelayHandle) -> oneshot::Sender<()> {
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        handle
            .sender
            .send(ActorCommand::Pause {
                entered: entered_tx,
                release: release_rx,
            })
            .await
            .unwrap();
        entered_rx.await.unwrap();
        release_tx
    }

    #[tokio::test]
    async fn admission_completes_before_a_paused_actor_applies_the_event() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn_with_capacity(CkfConfig::new(32), scope("admit"), 4).unwrap();
        let release = pause_actor(&handle).await;

        tokio::time::timeout(
            Duration::from_millis(50),
            handle.admit_event(SourceEpoch::new(0), stored(worker, 1, &[1])),
        )
        .await
        .expect("queue admission should not await CKF mutation")
        .unwrap();
        assert_eq!(handle.mailbox_depth(), 1);

        release.send(()).unwrap();
        handle.flush().await.unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 1);
    }

    #[tokio::test]
    async fn bounded_mailbox_backpressures_before_admission_without_dropping_commands() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn_with_capacity(CkfConfig::new(32), scope("backpressure"), 1)
                .unwrap();
        let release = pause_actor(&handle).await;
        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1]))
            .await
            .unwrap();

        let second_handle = handle.clone();
        let mut second = tokio::spawn(async move {
            second_handle
                .admit_event(SourceEpoch::new(0), stored(worker, 2, &[2]))
                .await
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut second)
                .await
                .is_err(),
            "the second command should wait for bounded mailbox capacity"
        );

        release.send(()).unwrap();
        second.await.unwrap().unwrap();
        handle.flush().await.unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 2);
    }

    #[tokio::test]
    async fn subscriber_gets_snapshot_then_one_atomic_replacement_delta() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("replace")).unwrap();
        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1, 2]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();
        let base_sequence = subscription.snapshot.sequence();

        handle
            .replace_rank(
                SourceEpoch::new(0),
                worker.worker_id,
                worker.dp_rank,
                vec![stored(worker, 0, &[3, 4])],
            )
            .await
            .unwrap();
        let delta = subscription.deltas.recv().await.unwrap();
        assert_eq!(delta.base_sequence(), base_sequence);
        assert_eq!(delta.sequence(), base_sequence + 1);
    }

    #[tokio::test]
    async fn shutdown_drains_admitted_events_and_rejects_new_admission() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn_with_capacity(CkfConfig::new(32), scope("shutdown"), 4).unwrap();
        let release = pause_actor(&handle).await;
        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1]))
            .await
            .unwrap();
        let shutdown_handle = handle.clone();
        let shutdown = tokio::spawn(async move { shutdown_handle.shutdown().await });
        release.send(()).unwrap();

        shutdown.await.unwrap().unwrap();
        assert!(matches!(
            handle
                .admit_event(SourceEpoch::new(0), stored(worker, 2, &[2]))
                .await,
            Err(KvDcRelayError::ShuttingDown)
        ));
    }

    #[tokio::test]
    async fn cadence_advances_on_duplicate_events_without_acknowledging_mutation() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let (handle, _faults) = KvDcRelayHandle::spawn(config, scope("cadence")).unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();

        for event_id in 1..=15 {
            handle
                .admit_event(SourceEpoch::new(0), stored(worker, event_id, &[7]))
                .await
                .unwrap();
        }
        let (stats, sequence, _) = handle.state_stats().await.unwrap();
        assert_eq!(sequence, 0);
        assert_eq!(stats.publication().pending_events(), 15);
        assert!(subscription.deltas.try_recv().is_err());

        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 16, &[7]))
            .await
            .unwrap();
        let delta = subscription.deltas.recv().await.unwrap();
        assert_eq!(delta.base_sequence(), 0);
        assert_eq!(delta.sequence(), 1);
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.publication().pending_events(), 0);
        assert_eq!(stats.aggregation().unique_block_count(), 1);
    }

    #[tokio::test]
    async fn publication_timer_emits_a_sparse_tail_without_flush() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let (handle, _faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            scope("timer"),
            Duration::from_millis(1),
        )
        .unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();

        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[7]))
            .await
            .unwrap();
        let delta = tokio::time::timeout(Duration::from_millis(100), subscription.deltas.recv())
            .await
            .expect("the 1 ms timer must publish a sparse dirty tail")
            .unwrap();

        assert_eq!((delta.base_sequence(), delta.sequence()), (0, 1));
        assert_eq!(handle.state_stats().await.unwrap().1, 1);
    }

    #[tokio::test]
    async fn replacement_subscription_starts_after_the_old_lease_tail() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let (handle, _faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            scope("subscription-tail"),
            Duration::from_secs(10),
        )
        .unwrap();
        let mut old = handle.subscribe(lease(1)).await.unwrap();
        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[7]))
            .await
            .unwrap();

        let mut replacement = handle.subscribe(lease(2)).await.unwrap();
        let old_tail = old.deltas.recv().await.unwrap();
        assert_eq!(old_tail.lease(), lease(1));
        assert_eq!(replacement.snapshot.sequence(), old_tail.sequence());
        assert!(replacement.deltas.try_recv().is_err());

        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 2, &[8]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let continuation = replacement.deltas.recv().await.unwrap();
        assert_eq!(continuation.lease(), lease(2));
        assert_eq!(
            continuation.base_sequence(),
            replacement.snapshot.sequence()
        );
        assert_eq!(continuation.sequence(), replacement.snapshot.sequence() + 1);
    }

    #[tokio::test]
    async fn capacity_omission_is_observable_without_a_lifecycle_fault() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(1);
        config.max_kicks = 1;
        let (handle, mut faults) = KvDcRelayHandle::spawn(config, scope("fault")).unwrap();
        let hashes: Vec<_> = (1..=32).collect();

        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &hashes))
            .await
            .unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert!(stats.aggregation().unique_block_count() > 0);
        assert!(stats.aggregation().capacity_failures() > 0);

        assert!(!actor_health(&handle).faulted);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "a pre-commit capacity omission must not enter lifecycle fault handling"
        );

        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 2, &[1]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        assert!(!actor_health(&handle).faulted);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "later work must remain live without delayed capacity lifecycle faults"
        );
    }

    #[tokio::test]
    async fn failed_replacement_returns_barrier_error_without_replaying_or_faulting() {
        let worker = WorkerWithDpRank::new(1, 0);
        let foreign = WorkerWithDpRank::new(2, 0);
        let (handle, mut faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("barrier-fault")).unwrap();

        assert!(
            handle
                .replace_rank(
                    SourceEpoch::new(0),
                    worker.worker_id,
                    worker.dp_rank,
                    vec![stored(foreign, 1, &[1])],
                )
                .await
                .is_err()
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "a replacement build failure before swap leaves the old generation unchanged"
        );
    }

    #[test]
    fn event_failures_keep_commit_domains_distinct() {
        let capacity = event_failure_point(KvCacheEventError::CapacityExhausted).disposition();
        assert_eq!(capacity.action, CkfFailureAction::ContinueCapacityOmission);
        assert_eq!(capacity.domain, CkfFailureDomain::ProducerCore);
        assert_eq!(capacity.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(capacity.recovery_domain, None);

        let allocation = event_failure_point(KvCacheEventError::AllocationFailed).disposition();
        assert_eq!(allocation.action, CkfFailureAction::ReportResourceFailure);
        assert_eq!(allocation.commit, CkfCommitState::KnownUnchanged);

        let source = event_failure_point(KvCacheEventError::OwnershipDegreeOverflow).disposition();
        assert_eq!(source.action, CkfFailureAction::RejectSource);
        assert_eq!(source.commit, CkfCommitState::KnownUnchanged);

        let invariant =
            event_failure_point(KvCacheEventError::IndexerInvariantViolation).disposition();
        assert_eq!(invariant.action, CkfFailureAction::FenceAndRebuildProducer);
        assert_eq!(invariant.commit, CkfCommitState::KnownUnchanged);
        assert_eq!(
            invariant.recovery_domain,
            Some(CkfFailureDomain::ProducerCore)
        );
    }

    #[tokio::test]
    async fn stale_source_epoch_cannot_mutate_or_fault_a_replacement_rank() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, mut faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("stale-epoch")).unwrap();

        handle
            .admit_event(SourceEpoch::new(1), stored(worker, 1, &[1]))
            .await
            .unwrap();
        handle
            .reset_rank(SourceEpoch::new(2), worker.worker_id, worker.dp_rank, false)
            .await
            .unwrap();

        handle
            .admit_event(SourceEpoch::new(1), stored(worker, 2, &[2]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let (stats, _, _) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 0);

        assert!(matches!(
            handle
                .reset_rank(SourceEpoch::new(1), worker.worker_id, worker.dp_rank, false)
                .await,
            Err(KvDcRelayError::StaleSourceEpoch {
                current: 2,
                received: 1,
                ..
            })
        ));
        assert!(matches!(
            handle
                .replace_rank(
                    SourceEpoch::new(1),
                    worker.worker_id,
                    worker.dp_rank,
                    vec![stored(worker, 3, &[3])],
                )
                .await,
            Err(KvDcRelayError::StaleSourceEpoch {
                current: 2,
                received: 1,
                ..
            })
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(20), faults.recv())
                .await
                .is_err(),
            "stale traffic must not fault the replacement epoch"
        );
    }
}
