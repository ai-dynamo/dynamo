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

#[cfg(any(test, feature = "ckf-diagnostics"))]
use dynamo_kv_router::indexer::cuckoo::DcCkfStats;
#[cfg(feature = "ckf-diagnostics")]
use dynamo_kv_router::indexer::cuckoo::PublisherEmitOutcome;
use dynamo_kv_router::indexer::cuckoo::{
    CacheDomainId, CacheDomainIdentity, CkfConfig, CkfFailureAction, CkfFailureDisposition,
    CkfFailurePoint, DcCkfDelta, DcCkfDeltaSink, DcCkfPublisher, DcCkfSnapshot, DcCkfState,
    DcId as CkfDcId, EndpointId as CkfEndpointId, LaneLease, ProducerIdentity,
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
#[cfg(feature = "ckf-diagnostics")]
use crate::kv_router::indexer::WorkerQueryHealthSnapshot;
use crate::kv_router::indexer::{
    DEFAULT_RECOVERY_ATTEMPT_TIMEOUT, RecoveryResetReason, RecoverySupervisor, RecoveryTarget,
    SourceEpoch, TargetFaultDisposition, start_target_subscriber,
};

pub const DEFAULT_EXPECTED_UNIQUE_BLOCKS: usize = 1_048_576;
const DEFAULT_MAILBOX_CAPACITY: usize = 256;
const DEFAULT_PENDING_BLOCK_PERMITS: usize = 65_536;
const DEFAULT_PUBLICATION_CAPACITY: usize = 64;
const DEFAULT_FAULT_CAPACITY: usize = 16;
const DEFAULT_RECOVERY_FETCH_CONCURRENCY: usize = 16;
const DEFAULT_PUBLICATION_THRESHOLD: usize = 16;
const DEFAULT_PUBLICATION_DELAY: Duration = Duration::from_millis(1);
const RECOVERY_REBUILD_BATCH_WINDOW: Duration = Duration::from_millis(5);

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

#[derive(Debug)]
// NOTE: This is the intentional producer snapshot/delta seam for the future non-local adapter.
// It remains crate-private until that transport has delivery cursors and recovery semantics.
#[allow(dead_code)]
pub(crate) struct DcCkfSubscription {
    pub(crate) snapshot: DcCkfSnapshot,
    pub(crate) deltas: broadcast::Receiver<DcCkfDelta>,
}

// NOTE: `dynamo-llm` enables the router's general metrics feature in production. Keep these
// pull-only diagnostics on a separate feature so ordinary commands do not acquire an activity
// mutex, read the clock, or update mailbox/publication atomics.
#[cfg(feature = "ckf-diagnostics")]
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

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Default)]
struct ActorActivity {
    active_command: Option<&'static str>,
    active_since: Option<Instant>,
    shutting_down: bool,
    last_error: Option<String>,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Default)]
struct ActorDiagnostics {
    counters: ActorCounters,
    activity: Mutex<ActorActivity>,
}

#[cfg(feature = "ckf-diagnostics")]
#[derive(Debug, Clone, Default)]
struct ActorDiagnosticsHandle(Arc<ActorDiagnostics>);

#[cfg(not(feature = "ckf-diagnostics"))]
#[derive(Debug, Clone, Default)]
struct ActorDiagnosticsHandle;

#[cfg(feature = "ckf-diagnostics")]
impl ActorDiagnosticsHandle {
    fn new() -> Self {
        Self::default()
    }

    fn start_command(&self, command: &ActorCommand) {
        let mut activity = self.0.activity.lock();
        activity.active_command = Some(command.kind());
        activity.active_since = Some(Instant::now());
    }

    fn finish_command(&self) {
        let mut activity = self.0.activity.lock();
        activity.active_command = None;
        activity.active_since = None;
    }

    fn record_error(&self, error: &impl std::fmt::Display) {
        self.0.activity.lock().last_error = Some(error.to_string());
    }

    fn record_shutdown(&self) {
        self.0.activity.lock().shutting_down = true;
    }

    fn record_mailbox_wait(&self, started: Instant) {
        let waited = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.0
            .counters
            .mailbox_wait_ns
            .fetch_add(waited, Ordering::Relaxed);
        self.0
            .counters
            .mailbox_max_wait_ns
            .fetch_max(waited, Ordering::Relaxed);
    }

    fn record_publish_outcome(&self, outcome: &PublisherEmitOutcome) {
        match outcome {
            PublisherEmitOutcome::Published { .. } => {
                self.0.counters.publications.fetch_add(1, Ordering::Relaxed);
            }
            PublisherEmitOutcome::NoSubscriber { .. } => {
                self.record_no_publication();
            }
        }
    }

    fn record_no_publication(&self) {
        self.0
            .counters
            .unchanged_publications
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_degraded_reset(&self) {
        self.0
            .counters
            .degraded_resets
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_rebuild(&self, started: Instant) {
        let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.0
            .counters
            .rebuild_count
            .fetch_add(1, Ordering::Relaxed);
        self.0
            .counters
            .rebuild_ns
            .fetch_add(elapsed, Ordering::Relaxed);
        self.0
            .counters
            .rebuild_max_ns
            .fetch_max(elapsed, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "ckf-diagnostics"))]
impl ActorDiagnosticsHandle {
    fn new() -> Self {
        Self
    }

    #[inline(always)]
    fn start_command(&self, _command: &ActorCommand) {}

    #[inline(always)]
    fn finish_command(&self) {}

    #[inline(always)]
    fn record_error(&self, _error: &impl std::fmt::Display) {}

    #[inline(always)]
    fn record_shutdown(&self) {}

    #[inline(always)]
    fn record_publish_outcome<T>(&self, _outcome: &T) {}

    #[inline(always)]
    fn record_no_publication(&self) {}

    #[inline(always)]
    fn record_degraded_reset(&self) {}
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

struct CancelOnDrop(CancellationToken);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
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
    fence: CancellationToken,
    stopped: CancellationToken,
    #[cfg(feature = "ckf-diagnostics")]
    diagnostics: ActorDiagnosticsHandle,
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
        let diagnostics = ActorDiagnosticsHandle::new();
        let fence = CancellationToken::new();
        let stopped = CancellationToken::new();
        tokio::spawn(run_actor(
            state,
            publisher,
            receiver,
            publication_delay,
            fault_tx,
            diagnostics.clone(),
            fence.clone(),
            stopped.clone(),
        ));
        Ok((
            Self {
                sender,
                payload_permits: Arc::new(Semaphore::new(DEFAULT_PENDING_BLOCK_PERMITS)),
                fence,
                stopped,
                #[cfg(feature = "ckf-diagnostics")]
                diagnostics,
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
        #[cfg(feature = "ckf-diagnostics")]
        let wait_started = Instant::now();
        self.sender
            .send(make_command(response_tx))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        #[cfg(feature = "ckf-diagnostics")]
        self.diagnostics.record_mailbox_wait(wait_started);
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
        #[cfg(feature = "ckf-diagnostics")]
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
        #[cfg(feature = "ckf-diagnostics")]
        self.diagnostics.record_mailbox_wait(wait_started);
        Ok(())
    }

    async fn replace_ranks(
        &self,
        replacements: Vec<RankReplacement>,
    ) -> Result<(), KvDcRelayError> {
        let weight = replacements
            .iter()
            .flat_map(|replacement| &replacement.events)
            .map(event_payload_weight)
            .fold(0usize, usize::saturating_add)
            .min(DEFAULT_PENDING_BLOCK_PERMITS) as u32;
        let permit = self
            .payload_permits
            .clone()
            .acquire_many_owned(weight.max(1))
            .await
            .map_err(|_| KvDcRelayError::ShuttingDown)?;
        self.submit(|response| ActorCommand::ReplaceRanks {
            replacements,
            _payload_permit: permit,
            response,
        })
        .await
    }

    #[cfg(test)]
    async fn replace_rank(
        &self,
        source_epoch: SourceEpoch,
        worker_id: WorkerId,
        dp_rank: DpRank,
        events: Vec<RouterEvent>,
    ) -> Result<(), KvDcRelayError> {
        self.replace_ranks(vec![RankReplacement {
            source_epoch,
            worker_id,
            dp_rank,
            events,
        }])
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

    #[cfg(feature = "ckf-diagnostics")]
    async fn snapshot(&self) -> Result<ActorSnapshot, KvDcRelayError> {
        self.submit(|response| ActorCommand::Snapshot { response })
            .await
    }

    #[cfg(any(test, feature = "ckf-diagnostics"))]
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

    async fn fence(&self) -> Result<(), KvDcRelayError> {
        self.fence.cancel();
        self.stopped.cancelled().await;
        Ok(())
    }

    #[cfg(any(test, feature = "ckf-diagnostics"))]
    fn mailbox_depth(&self) -> usize {
        self.sender
            .max_capacity()
            .saturating_sub(self.sender.capacity())
    }
}

#[derive(Debug)]
struct RankReplacement {
    source_epoch: SourceEpoch,
    worker_id: WorkerId,
    dp_rank: DpRank,
    events: Vec<RouterEvent>,
}

struct PendingRankReplacement {
    replacement: RankReplacement,
    response: oneshot::Sender<Result<(), String>>,
}

struct RankReplacementBatcher {
    state: tokio::sync::Mutex<RankReplacementBatchState>,
    initial_deadline: Duration,
}

#[derive(Default)]
struct RankReplacementBatchState {
    pending: Vec<PendingRankReplacement>,
    flush_scheduled: bool,
    initial_timer_scheduled: bool,
    initial_expected: Option<FxHashSet<WorkerWithDpRank>>,
    initial_completed: FxHashSet<WorkerWithDpRank>,
}

#[derive(Clone)]
struct KvDcRelayRecoveryTarget {
    handle: KvDcRelayHandle,
    rebuild_permit: Arc<Semaphore>,
    replacement_batcher: Arc<RankReplacementBatcher>,
}

impl KvDcRelayRecoveryTarget {
    async fn flush_replacement_batch(self, wait_for_quiet: bool) {
        if wait_for_quiet {
            let mut observed = 0usize;
            loop {
                tokio::time::sleep(RECOVERY_REBUILD_BATCH_WINDOW).await;
                let current = self.replacement_batcher.state.lock().await.pending.len();
                if current == observed {
                    break;
                }
                observed = current;
            }
        }
        let pending = {
            let mut state = self.replacement_batcher.state.lock().await;
            state.flush_scheduled = false;
            std::mem::take(&mut state.pending)
        };
        let (replacements, responses): (Vec<_>, Vec<_>) = pending
            .into_iter()
            .map(|pending| (pending.replacement, pending.response))
            .unzip();
        let batch_result = match self.rebuild_permit.acquire().await {
            Ok(_permit) => self
                .handle
                .replace_ranks(replacements)
                .await
                .map_err(|error| error.to_string()),
            Err(_) => Err(KvDcRelayError::ShuttingDown.to_string()),
        };
        for response_tx in responses {
            let response = match &batch_result {
                Ok(()) => Ok(()),
                Err(error) => Err(error.clone()),
            };
            let _ = response_tx.send(response);
        }
    }

    async fn expire_initial_recovery_batch(self) {
        tokio::time::sleep(self.replacement_batcher.initial_deadline).await;
        let schedule_flush = {
            let mut state = self.replacement_batcher.state.lock().await;
            let initial_open = state.initial_expected.take().is_some();
            if initial_open {
                state.initial_completed.clear();
            }
            if !initial_open || state.pending.is_empty() || state.flush_scheduled {
                false
            } else {
                state.flush_scheduled = true;
                true
            }
        };
        if schedule_flush {
            self.flush_replacement_batch(false).await;
        }
    }

    fn new_replacement_batcher(
        expected: FxHashSet<WorkerWithDpRank>,
        initial_deadline: Duration,
    ) -> Arc<RankReplacementBatcher> {
        Arc::new(RankReplacementBatcher {
            state: tokio::sync::Mutex::new(RankReplacementBatchState {
                initial_expected: (!expected.is_empty()).then_some(expected),
                ..RankReplacementBatchState::default()
            }),
            initial_deadline,
        })
    }

    async fn mark_initial_complete(&self, member: WorkerWithDpRank) {
        let schedule_flush = {
            let mut state = self.replacement_batcher.state.lock().await;
            let Some(expected) = state.initial_expected.as_ref() else {
                return;
            };
            if !expected.contains(&member) {
                return;
            }
            state.initial_completed.insert(member);
            let complete = state
                .initial_expected
                .as_ref()
                .is_some_and(|expected| expected.is_subset(&state.initial_completed));
            if !complete {
                return;
            }
            state.initial_expected = None;
            state.initial_completed.clear();
            if state.pending.is_empty() || state.flush_scheduled {
                false
            } else {
                state.flush_scheduled = true;
                true
            }
        };
        if schedule_flush {
            tokio::spawn(self.clone().flush_replacement_batch(false));
        }
    }
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
        let (response, result) = oneshot::channel();
        let member = WorkerWithDpRank::new(worker_id, dp_rank);
        let (schedule_flush, schedule_deadline, wait_for_quiet) = {
            let mut state = self.replacement_batcher.state.lock().await;
            state.pending.push(PendingRankReplacement {
                replacement: RankReplacement {
                    source_epoch,
                    worker_id,
                    dp_rank,
                    events,
                },
                response,
            });
            let initial = state
                .initial_expected
                .as_ref()
                .is_some_and(|expected| expected.contains(&member));
            if initial {
                state.initial_completed.insert(member);
            }
            let initial_complete = initial
                && state
                    .initial_expected
                    .as_ref()
                    .is_some_and(|expected| expected.is_subset(&state.initial_completed));
            if initial_complete {
                state.initial_expected = None;
                state.initial_completed.clear();
            }
            let initial_wave_open = state.initial_expected.is_some();
            let schedule_deadline = initial
                && !initial_complete
                && !std::mem::replace(&mut state.initial_timer_scheduled, true);
            if state.flush_scheduled || initial_wave_open {
                (false, schedule_deadline, false)
            } else {
                state.flush_scheduled = true;
                (true, schedule_deadline, !initial)
            }
        };
        if schedule_flush {
            tokio::spawn(self.clone().flush_replacement_batch(wait_for_quiet));
        }
        if schedule_deadline {
            tokio::spawn(self.clone().expire_initial_recovery_batch());
        }
        result
            .await
            .map_err(|_| anyhow::anyhow!("rank replacement batch coordinator stopped"))?
            .map_err(anyhow::Error::msg)
    }

    async fn complete_initial_recovery(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.mark_initial_complete(WorkerWithDpRank::new(worker_id, dp_rank))
            .await;
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
    let target = KvDcRelayRecoveryTarget {
        handle: handle.clone(),
        rebuild_permit,
        replacement_batcher: KvDcRelayRecoveryTarget::new_replacement_batcher(
            initial_recoveries,
            recovery_attempt_timeout,
        ),
    };
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

async fn fence_endpoint_actor(active: EndpointActorRuntime) {
    // Stop publication first. Recovery shutdown may attempt rank resets, but a producer whose
    // exact state is suspect must not emit another apparently valid delta while being retired.
    if let Err(error) = active.handle.fence().await {
        tracing::warn!(%error, endpoint = %active.handle.scope.serving_endpoint, "Failed to fence KV DC Relay endpoint actor cleanly");
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
        model_artifact: domain.model_artifact.clone(),
        kv_block_size: domain.kv_block_size,
        event_hash_format: domain.event_hash_format,
    }
}

#[cfg(feature = "ckf-diagnostics")]
fn actor_health(handle: &KvDcRelayHandle) -> KvDcRelayActorStats {
    let activity = handle.diagnostics.0.activity.lock();
    KvDcRelayActorStats {
        mailbox_depth: handle.mailbox_depth(),
        mailbox_capacity: handle.sender.max_capacity(),
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

#[cfg(any(test, feature = "ckf-diagnostics"))]
type ActorStatsResult = Result<(DcCkfStats, u64, Vec<(WorkerWithDpRank, usize)>), KvDcRelayError>;

enum ActorCommand {
    Apply {
        source_epoch: SourceEpoch,
        event: RouterEvent,
        _payload_permit: OwnedSemaphorePermit,
    },
    ReplaceRanks {
        replacements: Vec<RankReplacement>,
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
    #[cfg(feature = "ckf-diagnostics")]
    Snapshot {
        response: oneshot::Sender<Result<ActorSnapshot, KvDcRelayError>>,
    },
    Subscribe {
        lease: LaneLease,
        response: oneshot::Sender<Result<ActorSubscription, KvDcRelayError>>,
    },
    #[cfg(any(test, feature = "ckf-diagnostics"))]
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

#[cfg(feature = "ckf-diagnostics")]
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
    #[cfg(feature = "ckf-diagnostics")]
    fn kind(&self) -> &'static str {
        match self {
            Self::Apply { .. } => "apply_event",
            Self::ReplaceRanks { .. } => "replace_ranks",
            Self::ResetRank { .. } => "reset_rank",
            Self::Flush { .. } => "flush",
            Self::Snapshot { .. } => "snapshot",
            Self::Subscribe { .. } => "subscribe",
            #[cfg(any(test, feature = "ckf-diagnostics"))]
            Self::Stats { .. } => "stats",
            Self::Shutdown { .. } => "shutdown",
            #[cfg(test)]
            Self::Pause { .. } => "test_pause",
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_actor(
    mut state: DcCkfState,
    mut publisher: DcCkfPublisher<BroadcastDeltaSink>,
    mut receiver: mpsc::Receiver<ActorCommand>,
    publication_delay: Duration,
    fault_tx: mpsc::Sender<ActorFault>,
    diagnostics: ActorDiagnosticsHandle,
    fence: CancellationToken,
    stopped: CancellationToken,
) {
    let _stopped_guard = CancelOnDrop(stopped);
    let mut source_epochs = FxHashMap::<WorkerWithDpRank, SourceEpoch>::default();
    let mut capacity_omission_events = 0u64;
    let mut shutdown_response = None;
    let mut discard_tail = false;
    let mut publication_deadline = None;
    loop {
        let command = if let Some(deadline) = publication_deadline {
            tokio::select! {
                biased;
                _ = fence.cancelled() => {
                    discard_tail = true;
                    break;
                }
                command = receiver.recv() => command,
                _ = tokio::time::sleep_until(deadline) => {
                    publication_deadline = None;
                    if state.has_pending_publication()
                        && let Err(error) = publish_pending(&mut state, &mut publisher, &diagnostics)
                    {
                        diagnostics.record_error(&error);
                    }
                    continue;
                }
            }
        } else {
            tokio::select! {
                biased;
                _ = fence.cancelled() => {
                    discard_tail = true;
                    break;
                }
                command = receiver.recv() => command,
            }
        };
        let Some(command) = command else {
            break;
        };
        diagnostics.start_command(&command);
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
                    diagnostics.finish_command();
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
                    diagnostics.record_error(&message);
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
                    diagnostics.finish_command();
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
                    if let Err(error) = publish_batch(batch, &mut publisher, &diagnostics) {
                        diagnostics.record_error(&error);
                    }
                } else if publication_boundary {
                    diagnostics.record_no_publication();
                }
                if publication_boundary {
                    publication_deadline = None;
                } else if state.pending_event_count() == 1 && publication_deadline.is_none() {
                    publication_deadline = Some(tokio::time::Instant::now() + publication_delay);
                }
                if let Some(error) = first_error {
                    let disposition = event_failure_point(error).disposition();
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
                        diagnostics.finish_command();
                        continue;
                    }
                    let message = error.to_string();
                    let category = actor_fault_category(disposition);
                    diagnostics.record_error(&message);
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
            ActorCommand::ReplaceRanks {
                replacements,
                _payload_permit: _,
                response,
            } => {
                let stale = replacements.iter().find_map(|replacement| {
                    let key = WorkerWithDpRank::new(replacement.worker_id, replacement.dp_rank);
                    let current = source_epochs.get(&key).copied()?;
                    (replacement.source_epoch < current).then_some((replacement, current))
                });
                if let Some((replacement, current)) = stale {
                    let _ = response.send(Err(KvDcRelayError::StaleSourceEpoch {
                        worker_id: replacement.worker_id,
                        dp_rank: replacement.dp_rank,
                        current: current.get(),
                        received: replacement.source_epoch.get(),
                    }));
                    diagnostics.finish_command();
                    continue;
                }
                #[cfg(feature = "ckf-diagnostics")]
                let rebuild_started = Instant::now();
                let mut committed_epochs = Vec::with_capacity(replacements.len());
                let result = replacement_batch_hashes(replacements, &mut committed_epochs)
                    .and_then(|hashes| state.replace_ranks(hashes).map_err(Into::into))
                    .and_then(|publication| {
                        if let Some(batch) = publication {
                            publish_batch(batch, &mut publisher, &diagnostics)?;
                        } else {
                            diagnostics.record_no_publication();
                        }
                        Ok(())
                    });
                if result.is_ok() {
                    source_epochs.extend(committed_epochs);
                }
                #[cfg(feature = "ckf-diagnostics")]
                diagnostics.record_rebuild(rebuild_started);
                // The whole cold-start batch is built off-side. A pre-swap failure leaves every
                // prior rank unchanged; the strong responses all observe the same atomic result.
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
                    diagnostics.finish_command();
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
                            diagnostics.record_degraded_reset();
                        }
                        if let Some(batch) = publication {
                            publish_batch(batch, &mut publisher, &diagnostics)?;
                        } else {
                            diagnostics.record_no_publication();
                        }
                        Ok(())
                    });
                if result.is_ok() {
                    source_epochs.insert(key, source_epoch);
                }
                let _ = response.send(result);
            }
            ActorCommand::Flush { response } => {
                let result = publish_pending(&mut state, &mut publisher, &diagnostics);
                let _ = response.send(result);
            }
            #[cfg(feature = "ckf-diagnostics")]
            ActorCommand::Snapshot { response } => {
                let result = diagnostic_barrier_snapshot(&mut state, &mut publisher, &diagnostics);
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
            #[cfg(any(test, feature = "ckf-diagnostics"))]
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
                    diagnostics.record_shutdown();
                    shutdown_response = Some(response);
                }
            }
            #[cfg(test)]
            ActorCommand::Pause { entered, release } => {
                let _ = entered.send(());
                let _ = release.await;
            }
        }
        if !state.has_pending_publication() {
            publication_deadline = None;
        }
        diagnostics.finish_command();
    }

    if !discard_tail && let Err(error) = publish_pending(&mut state, &mut publisher, &diagnostics) {
        diagnostics.record_error(&error);
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

fn replacement_batch_hashes(
    replacements: Vec<RankReplacement>,
    committed_epochs: &mut Vec<(WorkerWithDpRank, SourceEpoch)>,
) -> Result<FxHashMap<WorkerWithDpRank, FxHashSet<ExternalSequenceBlockHash>>, KvDcRelayError> {
    let mut hashes_by_rank = FxHashMap::default();
    for replacement in replacements {
        let member = WorkerWithDpRank::new(replacement.worker_id, replacement.dp_rank);
        if hashes_by_rank.contains_key(&member) {
            return Err(KvDcRelayError::InvalidTreeDump {
                worker_id: replacement.worker_id,
                dp_rank: replacement.dp_rank,
                message: "replacement batch contains the same rank more than once".to_string(),
            });
        }
        let hashes = replacement_hashes(
            replacement.worker_id,
            replacement.dp_rank,
            replacement.events,
        )?;
        hashes_by_rank.insert(member, hashes);
        committed_epochs.push((member, replacement.source_epoch));
    }
    Ok(hashes_by_rank)
}

fn publish_batch(
    batch: dynamo_kv_router::indexer::cuckoo::DcCkfPublicationBatch,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    diagnostics: &ActorDiagnosticsHandle,
) -> Result<(), KvDcRelayError> {
    let outcome = publisher
        .publish(batch)
        .map_err(|error| KvDcRelayError::Publisher(format!("{error:?}")))?;
    diagnostics.record_publish_outcome(&outcome);
    Ok(())
}

fn publish_pending(
    state: &mut DcCkfState,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    diagnostics: &ActorDiagnosticsHandle,
) -> Result<(), KvDcRelayError> {
    let Some(batch) = state.flush() else {
        return Ok(());
    };
    publish_batch(batch, publisher, diagnostics)
}

#[cfg(feature = "ckf-diagnostics")]
fn diagnostic_barrier_snapshot(
    state: &mut DcCkfState,
    publisher: &mut DcCkfPublisher<BroadcastDeltaSink>,
    diagnostics: &ActorDiagnosticsHandle,
) -> Result<ActorSnapshot, KvDcRelayError> {
    let (publication, buckets) = state.barrier_snapshot()?;
    if let Some(batch) = publication {
        publish_batch(batch, publisher, diagnostics)?;
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

    #[cfg(feature = "ckf-diagnostics")]
    #[tokio::test]
    async fn diagnostic_feature_exposes_rich_actor_and_snapshot_state() {
        let worker = WorkerWithDpRank::new(1, 0);
        let (handle, _faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("diagnostics")).unwrap();

        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1, 2]))
            .await
            .unwrap();
        handle.flush().await.unwrap();
        let snapshot = handle.snapshot().await.unwrap();

        assert_eq!(snapshot.stats.aggregation().unique_block_count(), 2);
        assert_eq!(
            snapshot.buckets.len(),
            snapshot.identity.format().bucket_count()
        );
        assert_eq!(
            actor_health(&handle).mailbox_capacity,
            DEFAULT_MAILBOX_CAPACITY
        );
        assert!(
            handle
                .diagnostics
                .0
                .counters
                .unchanged_publications
                .load(Ordering::Relaxed)
                > 0
        );
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
    async fn initial_rank_recoveries_share_one_transactional_pool_rebuild() {
        let first = WorkerWithDpRank::new(1, 0);
        let second = WorkerWithDpRank::new(2, 0);
        let expected = [first, second].into_iter().collect();
        let (handle, _faults) =
            KvDcRelayHandle::spawn(CkfConfig::new(32), scope("batch-replace")).unwrap();
        let target = KvDcRelayRecoveryTarget {
            handle: handle.clone(),
            rebuild_permit: Arc::new(Semaphore::new(1)),
            replacement_batcher: KvDcRelayRecoveryTarget::new_replacement_batcher(
                expected,
                Duration::from_millis(100),
            ),
        };

        let first_target = target.clone();
        let mut first_replacement = tokio::spawn(async move {
            first_target
                .replace_rank(
                    SourceEpoch::new(1),
                    first.worker_id,
                    first.dp_rank,
                    vec![stored(first, 0, &[1, 2])],
                )
                .await
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut first_replacement)
                .await
                .is_err(),
            "the first cold-start rank must wait for the recovery wave"
        );
        target
            .replace_rank(
                SourceEpoch::new(1),
                second.worker_id,
                second.dp_rank,
                vec![stored(second, 0, &[3])],
            )
            .await
            .unwrap();
        first_replacement.await.unwrap().unwrap();

        let (stats, _, members) = handle.state_stats().await.unwrap();
        assert_eq!(stats.aggregation().unique_block_count(), 3);
        assert_eq!(members.len(), 2);
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
    async fn producer_fence_retires_stream_without_publishing_uncertain_tail() {
        let worker = WorkerWithDpRank::new(1, 0);
        let mut config = CkfConfig::new(32);
        config.publish_every_n_events = 16;
        let (handle, _faults) = KvDcRelayHandle::spawn_with_publication_delay(
            config,
            scope("fence"),
            Duration::from_secs(10),
        )
        .unwrap();
        let mut subscription = handle.subscribe(lease(1)).await.unwrap();
        handle
            .admit_event(SourceEpoch::new(0), stored(worker, 1, &[1]))
            .await
            .unwrap();

        handle.fence().await.unwrap();
        assert!(matches!(
            subscription.deltas.recv().await,
            Err(broadcast::error::RecvError::Closed)
        ));
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
