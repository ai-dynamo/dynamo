// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The shared selection core: one `SelectionCore` per process holding the
//! partitions, their catalog, and the reservation index. Selection itself is
//! in `run`, reservations in `reservations`, worker membership in `workers`.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use dynamo_tokens::SequenceHash;
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::config::{CacheHitWeights, RouterConfig, Validated};
use crate::identity::RoutingPartitionId;
use crate::kv_hints::{
    KvHint, KvHintAction, KvSourceLocationsPayload, KvTransferCandidateSource, KvTransferCandidates,
};
use crate::overlap_summary::MooncakeOverlapSummary;
use crate::protocols::{
    LocalBlockHash, PrefillLoadHint, SharedCacheHits, WorkerAffinityTarget, WorkerConfigLike,
    WorkerId, WorkerWithDpRank,
};
use crate::queue::SchedulerBookingDescriptor;
use crate::replication::{
    AFFINITY_EVENT_CHANNEL_CAPACITY, AffinityMeshSink, HostReplicaSyncFactory, ReplicaInbox,
    ReplicaSyncConfig, SchedulerLoadSink, ScopedReplicaEvent, ScopedSequencePublisher,
    setup_scoped_replica_sync,
};
use crate::selector::{WorkerSelectionPolicy, WorkerSelector};
use crate::shared_cache::SharedKvCache;
use crate::tracking_hash::{TrackingHashContext, TrackingHashScope};
use crate::{
    KvSchedulerError, LocalScheduler, OverlapAnalysis, OverlapSignals, OverloadedWorkerProvider,
    PotentialLoad, PrefillLoadEstimator, RejectThresholdsProvider, ScheduleMode, ScheduleRequest,
    SessionContext, TieredOverlapRefresher, WorkerAvailabilityProvider, effective_prefill_tokens,
    prefill_load_hint_from_effective_tokens,
};
use dynamo_kv_indexer::backend::{Indexer, IndexerPolicy};
use dynamo_kv_indexer::registry::IndexRegistry;
use dynamo_kv_indexer::{LowerTierQueryOptions, RoutingDecisionHashes, TieredMatchDetails};
use dynamo_slot_tracker::{
    ActiveSequencesMultiWorker, LifecycleMutationOutcome, ReplicaRequestLeaseObserver,
    ReplicaWorkerPolicy, SequenceRequest, SequenceTrackerOptions, active_request_expiry_duration,
};
use dynamo_worker_selection::WorkerInputs;
use dynamo_worker_selection::lora_filter::{LoraWorkerFilter, narrow_allowed_worker_ids_by_lora};

mod hint;
mod operation;
mod queries;
mod reservations;
mod run;
#[cfg(test)]
mod tests;
mod workers;

use reservations::ReservationIndex;

pub use operation::{
    LookupTimings, Selected, SelectionAdmission, SelectionOperation, SelectionOutcome,
    SelectionRun, SessionBinding,
};

use super::catalog::WorkerCatalog;
use super::error::SelectionError;
use super::ingress::{KvEventIngress, LocalIndexIngress};
use super::input::{PromptView, TrackingHashInput};
use super::pending::{PendingSelection, SelectionCache, SelectionCacheConfig};
use super::types::{
    ModelLoadResponse, OverlapScoresRequest, OverlapScoresResponse, PotentialLoadsRequest,
    ReadyResponse, ReservationRequest, ReservationResponse, SelectAndReserveRequest, SelectRequest,
    SelectResponse, SelectionWorkerConfig, SelectionWorkerLoad, WorkerCatalogRecord,
    WorkerLifecycle, WorkerPatchRequest, WorkerRequest,
};
use crate::replication::AffinityBindingEvent;
use crate::types::WorkerSelectionPolicyFactory;
use dynamo_router_protocol::worker_type::WorkerType;
use dynamo_session_affinity::{
    AcquireStep, AffinityError, AffinityLease, Hold, SessionAffinity, SessionAffinityConfig,
};

pub type SelectionScheduler = LocalScheduler<
    ScopedSequencePublisher,
    SelectionWorkerConfig,
    WorkerSelectionPolicy,
    TieredOverlapRefresher<Indexer>,
>;

/// Handle to one partition's scheduler and indexer for an embedding host that
/// drives scheduling directly (bypassing the request-shaped `select` API).
#[derive(Clone)]
pub struct SelectionPartition(Arc<SelectionEntry>);

impl SelectionPartition {
    pub fn key(&self) -> &RoutingPartitionId {
        &self.0.key
    }

    pub fn scheduler(&self) -> &SelectionScheduler {
        &self.0.scheduler
    }

    pub fn indexer(&self) -> &Indexer {
        &self.0.indexer
    }

    /// Return this partition's affinity table, initialized with `config`.
    pub fn session_affinity(
        &self,
        config: SessionAffinityConfig,
    ) -> Result<SessionAffinity, SelectionError> {
        self.0.session_affinity(config).cloned()
    }
}

struct SelectionEntry {
    key: RoutingPartitionId,
    block_size: u32,
    is_eagle: bool,
    indexer: Indexer,
    workers_tx: watch::Sender<HashMap<WorkerId, SelectionWorkerConfig>>,
    /// Whether a worker in `workers_tx` consumes router hints, set on publish.
    hint_capable_workers: AtomicBool,
    /// Whether the selection policy scores fetchable prefixes (`KV_TRANSFER`).
    scores_kv_transfer: bool,
    scheduler: SelectionScheduler,
    replica_inbox: Option<ReplicaInbox>,
    affinity: OnceCell<SessionAffinity>,
    replica_config: Option<ReplicaSyncConfig>,
    affinity_replication: Option<(u64, Arc<dyn dynamo_session_affinity::AffinityReplicaSink>)>,
}

impl SelectionEntry {
    fn session_affinity(
        &self,
        config: SessionAffinityConfig,
    ) -> Result<&SessionAffinity, SelectionError> {
        let table = self
            .affinity
            .get_or_try_init(|| -> Result<_, SelectionError> {
                let table = SessionAffinity::with_config(config).map_err(affinity_error)?;
                if let Some(config) = &self.replica_config
                    && let Some(sink) = config.affinity_sink(&self.key)
                {
                    table.enable_replication(config.process_id(), sink);
                } else if let Some((writer_id, sink)) = &self.affinity_replication {
                    table.enable_replication(*writer_id, sink.clone());
                }
                Ok(table)
            })?;
        if table.ttl() != config.ttl || table.mode() != config.mode {
            return Err(SelectionError::Conflict(format!(
                "session affinity config mismatch for {}: existing=({:?}, {:?}) requested=({:?}, {:?})",
                self.key,
                table.ttl(),
                table.mode(),
                config.ttl,
                config.mode
            )));
        }
        Ok(table)
    }
}

/// What an embedding host supplies to every partition the core creates,
/// grouped by purpose. Each group defaults to the standalone service's
/// behavior, so a host overrides only the groups it owns.
#[derive(Clone, Default)]
pub struct SelectionHost {
    pub load: HostLoad,
    pub cache: HostCache,
    pub eligibility: HostEligibility,
    pub telemetry: HostTelemetry,
    pub replication: HostReplication,
}

/// Load signals the host knows and the partition scheduler does not.
#[derive(Clone, Default)]
pub struct HostLoad {
    pub prefill_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    /// Workers to shed from selection (the host's overload detector).
    pub overloaded_workers: Option<OverloadedWorkerProvider>,
    /// Workers the host can currently reach; others are never selected.
    pub available_workers: Option<WorkerAvailabilityProvider>,
    /// Live per-rank limits past which workers are shed from selection.
    pub reject_thresholds: Option<RejectThresholdsProvider>,
}

/// Where a partition's KV knowledge comes from.
#[derive(Clone, Default)]
pub struct HostCache {
    /// Queried alongside the indexer for prompts that carry `token_ids`; a
    /// failed lookup is logged and selection proceeds without shared hits.
    pub shared: Option<Arc<dyn SharedKvCache>>,
    pub index: KvIndexSource,
}

/// Where a partition's KV index comes from and who feeds it.
#[derive(Clone)]
pub enum KvIndexSource {
    /// The ingress builds each partition's index and feeds it with worker KV
    /// events; it also decides what metadata a worker needs to be schedulable
    /// and what happens to the index when a worker leaves.
    Owned(Arc<dyn KvEventIngress>),
}

impl Default for KvIndexSource {
    fn default() -> Self {
        Self::Owned(Arc::new(LocalIndexIngress))
    }
}

/// Host-owned narrowing of the candidate set.
#[derive(Clone, Default)]
pub struct HostEligibility {
    /// Narrows candidates to the workers that can serve the request's LoRA
    /// adapter, strictly within the caller's `allowed_worker_ids`.
    pub lora_worker_filter: Option<Arc<dyn LoraWorkerFilter>>,
}

/// Scheduler state the host consumes.
#[derive(Clone, Default)]
pub struct HostTelemetry {
    /// Receives each partition's scheduler-owned load snapshots (active decode
    /// blocks and prefill tokens per worker) for metrics and overload detection.
    pub scheduler_load: Option<Arc<dyn SchedulerLoadSink>>,
}

/// Replica-sync transport the host carries for partitions this core does not
/// mesh itself (ignored when the service runs its own ZMQ replica sync).
#[derive(Clone)]
pub struct HostReplication {
    pub channels: Option<HostReplicaSyncFactory>,
    /// Owns request expiry when supplied; the partition then expires only through lifecycle events.
    pub request_leases: Option<Arc<dyn ReplicaRequestLeaseObserver>>,
    /// Whether peer events may create accounting before catalog discovery.
    /// This does not make an undiscovered worker eligible for selection.
    pub replica_worker_policy: ReplicaWorkerPolicy,
}

impl Default for HostReplication {
    fn default() -> Self {
        Self {
            channels: None,
            request_leases: None,
            replica_worker_policy: ReplicaWorkerPolicy::RequireRegistered,
        }
    }
}

type SelectionEntries = RwLock<HashMap<RoutingPartitionId, Arc<OnceCell<Arc<SelectionEntry>>>>>;

pub struct SelectionCore {
    catalog: WorkerCatalog,
    /// Serializes catalog commits and the corresponding ingress changes. Never held by selection.
    catalog_updates: tokio::sync::Mutex<()>,
    entries: Arc<SelectionEntries>,
    /// Lock order: `entries` before `reservation_index`, never nested the other way.
    reservation_index: Arc<ReservationIndex>,
    /// Sweep task is started lazily from the first `ensure_entry`, which always
    /// runs inside the host runtime; construction itself may not.
    reservation_sweep_started: OnceCell<()>,
    /// Whether this core subscribes to worker KV events itself. False when
    /// events are disabled, when the primary indexer is a remote service that
    /// workers publish to directly, or when the embedding host feeds events.
    listens_for_kv_events: bool,
    index_registry: Arc<IndexRegistry>,
    config: Validated<RouterConfig>,
    worker_selection_policy_factory: WorkerSelectionPolicyFactory,
    host: SelectionHost,
    worker_type: WorkerType,
    cancel_token: CancellationToken,
    replica_config: Option<ReplicaSyncConfig>,
    /// Booking inputs captured by `select`, keyed by `selection_id`, so a later
    /// `create_reservation` can replay them without re-sending the prompt.
    selection_cache: SelectionCache,
    tracking_hash: Arc<TrackingHashContext>,
    session_affinity: Option<SessionAffinityConfig>,
    /// Writer id and publisher for session bindings the host carries to its peers.
    session_binding_tx: Option<(u64, mpsc::Sender<AffinityBindingEvent>)>,
    /// Worker ids whose upsert fails with `Internal` before any catalog
    /// mutation, so membership tests can exercise per-worker error paths.
    #[cfg(test)]
    pub(super) fail_upsert_for: parking_lot::Mutex<std::collections::HashSet<WorkerId>>,
    /// Scheduler-config publishes that changed a partition's worker map.
    #[cfg(test)]
    pub(super) publish_count: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    pub(super) after_affinity_invalidation: Option<Arc<dyn Fn() + Send + Sync>>,
}

fn affinity_error(error: AffinityError) -> SelectionError {
    match error {
        AffinityError::InvalidArgument(message) => SelectionError::BadRequest(message),
        AffinityError::ResourceExhausted(message) => SelectionError::NotReady(message),
        AffinityError::Dropped => SelectionError::Internal(error.to_string()),
    }
}

impl SelectionCore {
    fn entry(&self, key: &RoutingPartitionId) -> Option<Arc<SelectionEntry>> {
        self.entries
            .read()
            .get(key)
            .and_then(|entry| entry.get().cloned())
    }

    fn initialized_entries(&self) -> Vec<Arc<SelectionEntry>> {
        self.entries
            .read()
            .values()
            .filter_map(|entry| entry.get().cloned())
            .collect()
    }

    pub fn partition(&self, key: &RoutingPartitionId) -> Option<SelectionPartition> {
        self.entry(key).map(SelectionPartition)
    }

    /// The kernel constructor: every input is a resolved slice or a hook the host owns.
    /// `SelectionServiceBuilder` and the local-core helpers in the composed router call it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Validated<RouterConfig>,
        indexer_threads: usize,
        cancel_token: CancellationToken,
        replica_config: Option<ReplicaSyncConfig>,
        worker_selection_policy_factory: WorkerSelectionPolicyFactory,
        host: SelectionHost,
        worker_type: WorkerType,
        signal_indexer_ready: bool,
        cache_config: SelectionCacheConfig,
        tracking_hash: Arc<TrackingHashContext>,
        indexer_policy: IndexerPolicy,
        session_affinity: Option<SessionAffinityConfig>,
    ) -> Self {
        let cancel_token = cancel_token.child_token();
        // Selection entries and their schedulers retain these indexers across worker updates.
        let index_registry = Arc::new(
            IndexRegistry::new(
                indexer_threads,
                Arc::new(dynamo_kv_indexer::KvIndexerMetrics::new_unregistered()),
            )
            .with_retained_indexers(),
        );
        let listens_for_kv_events = config.use_kv_events;
        index_registry.set_indexer_policy(indexer_policy);
        if signal_indexer_ready {
            index_registry.signal_ready();
        }
        {
            let KvIndexSource::Owned(ingress) = &host.cache.index;
            ingress.attach(&index_registry, cancel_token.clone());
        }
        Self {
            catalog: WorkerCatalog::default(),
            catalog_updates: tokio::sync::Mutex::new(()),
            entries: Arc::new(RwLock::new(HashMap::new())),
            reservation_index: Arc::new(RwLock::new(HashMap::new())),
            reservation_sweep_started: OnceCell::new(),
            listens_for_kv_events,
            index_registry,
            config,
            worker_selection_policy_factory,
            host,
            worker_type,
            cancel_token,
            replica_config,
            selection_cache: SelectionCache::new(&cache_config),
            tracking_hash,
            session_affinity,
            session_binding_tx: None,
            #[cfg(test)]
            fail_upsert_for: parking_lot::Mutex::default(),
            #[cfg(test)]
            publish_count: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            after_affinity_invalidation: None,
        }
    }

    /// Export session bindings through a bounded channel for the host to carry.
    pub fn with_session_binding_publisher(
        mut self,
        writer_id: u64,
    ) -> (Self, mpsc::Receiver<AffinityBindingEvent>) {
        let (tx, rx) = mpsc::channel(AFFINITY_EVENT_CHANNEL_CAPACITY);
        self.session_binding_tx = Some((writer_id, tx));
        (self, rx)
    }

    fn session_binding_sink(
        &self,
        partition: &RoutingPartitionId,
    ) -> Option<(u64, Arc<dyn dynamo_session_affinity::AffinityReplicaSink>)> {
        self.session_binding_tx.as_ref().map(|(writer_id, tx)| {
            let sink =
                AffinityMeshSink::new(partition.clone(), tx.clone(), self.cancel_token.clone());
            (
                *writer_id,
                Arc::new(sink) as Arc<dyn dynamo_session_affinity::AffinityReplicaSink>,
            )
        })
    }

    /// Cancel core-scoped tasks (KV-event listeners, scheduling, replica sync,
    /// periodic expiry) without cancelling the parent token. In-flight and
    /// queued selections then fail fast.
    ///
    /// The KV indexer thread pool is owned by the registry and released when
    /// this `SelectionCore` is dropped. Idempotent.
    pub fn shutdown(&self) {
        self.cancel_token.cancel();
    }

    pub fn ensure_running(&self) -> Result<(), SelectionError> {
        if self.cancel_token.is_cancelled() {
            return Err(SelectionError::NotReady(
                "selection service is shutting down".to_string(),
            ));
        }
        Ok(())
    }

    /// The core's own cancellation token (a child of the one it was built with), for tests.
    #[cfg(any(test, feature = "testing"))]
    pub fn cancel_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// The per-partition indexers this core schedules over.
    pub fn index_registry(&self) -> &Arc<IndexRegistry> {
        &self.index_registry
    }

    pub fn signal_indexer_ready(&self) {
        self.index_registry.signal_ready();
    }

    pub fn dispatch_replica_event(&self, envelope: ScopedReplicaEvent) {
        let (key, block_size, event) = envelope.into_parts();
        if self
            .replica_config
            .as_ref()
            .is_some_and(|config| config.is_self_event(&event))
        {
            return;
        }

        let Some(entry) = self.entry(&key) else {
            tracing::trace!(%key, "Dropping replica event for unknown selector entry");
            return;
        };
        if let Some(inbox) = &entry.replica_inbox {
            inbox.deliver(&key, entry.block_size, block_size, event);
        }
    }

    /// Apply a session binding a replica published.
    pub fn dispatch_affinity_event(&self, event: AffinityBindingEvent) {
        let Some(entry) = self.entry(&event.partition) else {
            tracing::trace!(
                key = %event.partition,
                "Dropping session affinity replica update for unknown selector entry"
            );
            return;
        };
        let Some(table) = entry.affinity.get() else {
            tracing::trace!(
                key = %event.partition,
                "Dropping session affinity replica update: no affinity table"
            );
            return;
        };
        if self
            .replica_config
            .as_ref()
            .is_some_and(|config| config.process_id() == event.writer_id)
        {
            return;
        }
        table.observe_replica_sequence(event.sequence);
        if self
            .catalog
            .get(event.worker_id)
            .is_none_or(|record| record.key() != event.partition)
        {
            tracing::trace!(
                key = %event.partition,
                worker_id = event.worker_id,
                "Dropping session affinity replica update: worker not in partition"
            );
            return;
        }
        let (target, version, worker_id) = (event.target(), event.version(), event.worker_id);
        let outcome = table.apply_replica_update(event.session_id, target, version);
        tracing::trace!(
            worker_id,
            ?outcome,
            "Applied session affinity replica update"
        );
    }

    fn ready_entry(&self, key: &RoutingPartitionId) -> Result<Arc<SelectionEntry>, SelectionError> {
        let Some(entry) = self.entry(key) else {
            return Err(self.not_ready(key));
        };
        if !self.catalog.has_schedulable_for_key(key) {
            return Err(self.not_ready(key));
        }
        Ok(entry)
    }

    /// Only the failure path pays for the catalog-wide count.
    fn not_ready(&self, key: &RoutingPartitionId) -> SelectionError {
        if self.catalog.schedulable_count() == 0 {
            SelectionError::NotReady("no schedulable workers are available".to_string())
        } else {
            SelectionError::NotReady(format!("no schedulable workers for {key}"))
        }
    }
}

fn tracking_scope(entry: &SelectionEntry) -> TrackingHashScope<'_> {
    TrackingHashScope {
        partition: entry.key.as_ref(),
        block_size: entry.block_size,
    }
}

impl Drop for SelectionCore {
    fn drop(&mut self) {
        self.shutdown();
    }
}
