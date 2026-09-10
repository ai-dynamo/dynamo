// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_tokens::SequenceHash;
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use tokio::sync::{mpsc, watch};
use tokio_util::sync::CancellationToken;

use crate::identity::RoutingPartitionId;
use crate::indexer::{
    LowerTierQueryOptions, RoutingDecisionHashes, SharedKvCache, TieredMatchDetails,
};
use crate::kv_hints::{
    KvHint, KvHintAction, KvSourceLocationsPayload, KvTransferCandidateSource, KvTransferCandidates,
};
#[cfg(test)]
use crate::protocols::ActiveSequenceEventData;
use crate::protocols::{
    ActiveSequenceEvent, LocalBlockHash, PrefillLoadHint, RoutingConstraints, SharedCacheHits,
    WorkerAffinityTarget, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use crate::scheduling::config::RouterConfigOverride;
use crate::scheduling::queue::SchedulerBookingDescriptor;
use crate::scheduling::selector::WorkerSelectionPolicy;
use crate::scheduling::{
    KvSchedulerError, LocalScheduler, LoraWorkerFilter, OverlapAnalysis, OverlapSignals,
    OverloadedWorkerProvider, PotentialLoad, PrefillLoadEstimator, ScheduleMode, ScheduleRequest,
    SessionContext, TieredOverlapRefresher, WorkerAvailabilityProvider, effective_prefill_tokens,
    narrow_allowed_worker_ids_by_lora, prefill_load_hint_from_effective_tokens,
};
use crate::sequences::{
    ActiveSequencesMultiWorker, LifecycleMutationOutcome, ReplicaRequestLeaseObserver,
    ReplicaWorkerPolicy, SequenceRequest, SequenceTrackerOptions, active_request_expiry_duration,
};
#[cfg(test)]
use crate::services::common::replica_sync::HostReplicaChannels;
use crate::services::common::replica_sync::{
    HostReplicaSyncFactory, ReplicaSyncConfig, SchedulerLoadSink, ScopedReplicaEvent,
    ScopedSequencePublisher, setup_scoped_replica_sync,
};
use crate::services::indexer::backend::{Indexer, IndexerPolicy};
use crate::services::indexer::recovery;
use crate::services::indexer::registry::WorkerRegistry;
use crate::services::overlap::MooncakeOverlapSummary;
use crate::tracking_hash::{TrackingHashContext, TrackingHashScope};

use super::affinity::{AffinityError, AffinityLease, Hold, SessionAffinity, SessionAffinityConfig};
use super::catalog::WorkerCatalog;
use super::error::SelectionError;
use super::ingress::{KvEventIngress, ZmqDirectIngress};
use super::input::{PromptRequest, TrackingHashInput};
use super::pending::{PendingSelection, SelectionCache, SelectionCacheConfig};
use super::types::{
    ModelLoadResponse, OverlapScoresRequest, OverlapScoresResponse, PotentialLoadsRequest,
    ReadyResponse, ReservationRequest, ReservationResponse, SelectAndReserveRequest, SelectRequest,
    SelectResponse, SelectionWorkerConfig, SelectionWorkerLoad, WorkerCatalogRecord,
    WorkerLifecycle, WorkerPatchRequest, WorkerRequest,
};
use crate::WorkerSelectionPolicyFactory;
use crate::WorkerType;
use crate::indexer::KvRouterError;
use crate::services::common::replica_sync::AffinityBindingEvent;

/// The scheduler type every partition runs.
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

    pub fn block_size(&self) -> u32 {
        self.0.block_size
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
    scheduler: SelectionScheduler,
    replica_tx: Option<mpsc::Sender<ActiveSequenceEvent>>,
    affinity: OnceCell<SessionAffinity>,
    replica_config: Option<ReplicaSyncConfig>,
}

struct PreparedSelectionInputs {
    block_hashes: Vec<LocalBlockHash>,
    sequence_hashes: Vec<SequenceHash>,
    isl_tokens: usize,
    overlap: OverlapSignals,
    shared_cache_hits: Option<SharedCacheHits>,
    kv_transfer_candidates: Option<KvTransferCandidates>,
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
                }
                Ok(table)
            })?;
        if table.ttl() != config.ttl {
            return Err(SelectionError::Conflict(format!(
                "session affinity TTL mismatch for {}",
                self.key
            )));
        }
        Ok(table)
    }
}

struct SelectionOperation {
    key: RoutingPartitionId,
    selection_id: Option<String>,
    prompt: PromptRequest,
    router_config_override: Option<RouterConfigOverride>,
    expected_output_tokens: Option<u32>,
    priority_jump: f64,
    strict_priority: u32,
    policy_class: Option<String>,
    session_context: Option<SessionContext>,
    affinity_target: Option<WorkerAffinityTarget>,
    pinned_worker: Option<WorkerWithDpRank>,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    routing_constraints: RoutingConstraints,
    /// Skip queue admission and return the chosen worker's load snapshot.
    advisory: bool,
}

/// Resolved inputs for booking a reservation, shared by the cached and explicit
/// `create_reservation` paths.
struct ReservationBooking {
    key: RoutingPartitionId,
    selection_id: String,
    worker: WorkerWithDpRank,
    sequence_hashes: Vec<SequenceHash>,
    prefill_load_hint: Option<PrefillLoadHint>,
    expected_output_tokens: Option<u32>,
    track_prefill_tokens: bool,
    lora_name: Option<String>,
    /// Public block hashes to record into an approximate indexer once booked.
    routing_hashes: Option<Vec<LocalBlockHash>>,
}

/// What an embedding host supplies to every partition the core creates,
/// grouped by purpose. Each group defaults to the standalone service's
/// behavior, so a host overrides only the groups it owns: the frontend
/// `KvRouter` feeds load and availability from its request client, points
/// overlap refresh at its own index, and carries replica sync on its own
/// transport.
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
    /// and what happens to the index when a worker leaves. Defaults to
    /// [`ZmqDirectIngress`]; the frontend supplies its runtime-backed ingress.
    Owned(Arc<dyn KvEventIngress>),
    /// A standalone indexer at this base URL serves the primary index; this
    /// core does not subscribe to worker KV events.
    Remote(String),
}

impl Default for KvIndexSource {
    fn default() -> Self {
        Self::Owned(Arc::new(ZmqDirectIngress))
    }
}

impl std::fmt::Debug for KvIndexSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(_) => formatter.write_str("Owned"),
            Self::Remote(url) => formatter.debug_tuple("Remote").field(url).finish(),
        }
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
#[derive(Clone, Default)]
pub struct HostReplication {
    pub channels: Option<HostReplicaSyncFactory>,
    /// Owns request expiry when supplied; the partition then expires only through lifecycle events.
    pub request_leases: Option<Arc<dyn ReplicaRequestLeaseObserver>>,
}

impl std::fmt::Debug for SelectionHost {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SelectionHost")
            .field("prefill_estimator", &self.load.prefill_estimator.is_some())
            .field(
                "overloaded_workers",
                &self.load.overloaded_workers.is_some(),
            )
            .field("available_workers", &self.load.available_workers.is_some())
            .field("shared_cache", &self.cache.shared.is_some())
            .field("index", &self.cache.index)
            .field(
                "lora_worker_filter",
                &self.eligibility.lora_worker_filter.is_some(),
            )
            .field("scheduler_load", &self.telemetry.scheduler_load.is_some())
            .field("replication", &self.replication.channels.is_some())
            .finish()
    }
}

#[derive(Debug, Clone)]
pub struct SelectionServiceConfig {
    pub port: u16,
    pub threads: usize,
    pub indexer_peers: Vec<String>,
    /// Base URL of a standalone indexer that serves the primary KV index.
    /// When set, this service does not listen for worker KV events and
    /// `indexer_peers` recovery is skipped.
    pub remote_indexer_url: Option<String>,
    pub replica_sync_port: Option<u16>,
    pub replica_sync_peers: Vec<String>,
    pub kv_router_config: crate::config::KvRouterConfig,
    pub selection_cache: SelectionCacheConfig,
    /// Session stickiness TTL; `None` disables session affinity.
    pub session_affinity_ttl: Option<Duration>,
}

type SelectionEntries = RwLock<HashMap<RoutingPartitionId, Arc<OnceCell<Arc<SelectionEntry>>>>>;

/// `selection_id` -> the booking it holds.
///
/// Lifecycle calls (`prefill_complete`, `free`, `add_output_block`) arrive with
/// only a selection id; the index resolves them to one partition and one
/// scheduler booking, so they never touch a booking made by a later request
/// that reused the id. An id is claimed here before its booking is made and
/// installed once the booking is final; a claim (`booking == None`) rejects a
/// concurrent booking of the same id and is invisible to lifecycle calls.
/// Bookings mirrored from replica peers are indexed by
/// [`ReservationIndexObserver`].
type ReservationIndex = RwLock<HashMap<String, Reservation>>;

struct Reservation {
    partition: RoutingPartitionId,
    booking: Option<SchedulerBookingDescriptor>,
    _affinity_lease: Option<AffinityLease>,
}

/// Exclusive ownership of a selection id while its booking is in flight.
/// Dropping the claim without `install` releases the id.
struct ReservationClaim<'a> {
    index: &'a ReservationIndex,
    selection_id: String,
    armed: bool,
}

impl ReservationClaim<'_> {
    fn install(mut self, reservation: Reservation) {
        self.armed = false;
        self.index
            .write()
            .insert(std::mem::take(&mut self.selection_id), reservation);
    }
}

impl Drop for ReservationClaim<'_> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let mut index = self.index.write();
        if index
            .get(&self.selection_id)
            .is_some_and(|reservation| reservation.booking.is_none())
        {
            index.remove(&self.selection_id);
        }
    }
}

/// Keeps the reservation index exact for bookings replicated from peers, which
/// never pass through this core's booking paths, and forwards to the host's
/// own observer.
struct ReservationIndexObserver {
    index: Arc<ReservationIndex>,
    partition: RoutingPartitionId,
    host: Option<Arc<dyn ReplicaRequestLeaseObserver>>,
}

impl ReplicaRequestLeaseObserver for ReservationIndexObserver {
    fn admitted(&self, booking: SchedulerBookingDescriptor) {
        {
            let mut index = self.index.write();
            // A peer only admits an id this partition's scheduler does not hold,
            // so an existing row for it is a stale booking (expired, not yet
            // swept) or a claim whose local booking will now fail; the mirror
            // replaces both. A row from another partition is left alone.
            let own_row = index
                .get(&booking.request_id)
                .is_none_or(|reservation| reservation.partition == self.partition);
            if own_row {
                index.insert(
                    booking.request_id.clone(),
                    Reservation {
                        partition: self.partition.clone(),
                        booking: Some(booking.clone()),
                        _affinity_lease: None,
                    },
                );
            }
        }
        if let Some(host) = &self.host {
            host.admitted(booking);
        }
    }

    fn progressed(&self, booking: &SchedulerBookingDescriptor) {
        if let Some(host) = &self.host {
            host.progressed(booking);
        }
    }

    fn completed(&self, booking: &SchedulerBookingDescriptor) {
        forget_reservation_if(&self.index, &self.partition, booking);
        if let Some(host) = &self.host {
            host.completed(booking);
        }
    }
}

/// Remove the index entry for `booking` only if it still describes it.
fn forget_reservation_if(
    index: &ReservationIndex,
    partition: &RoutingPartitionId,
    booking: &SchedulerBookingDescriptor,
) {
    let mut index = index.write();
    if index.get(&booking.request_id).is_some_and(|reservation| {
        reservation.partition == *partition && reservation.booking.as_ref() == Some(booking)
    }) {
        index.remove(&booking.request_id);
    }
}

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
    indexer_registry: Arc<WorkerRegistry>,
    kv_router_config: crate::config::KvRouterConfig,
    worker_selection_policy_factory: Option<WorkerSelectionPolicyFactory>,
    host: SelectionHost,
    worker_type: WorkerType,
    cancel_token: CancellationToken,
    replica_config: Option<ReplicaSyncConfig>,
    /// Booking inputs captured by `select`, keyed by `selection_id`, so a later
    /// `create_reservation` can replay them without re-sending the prompt.
    selection_cache: SelectionCache,
    tracking_hash: Arc<TrackingHashContext>,
    session_affinity: Option<SessionAffinityConfig>,
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

    /// Create a local selector and report invalid tracking configuration.
    pub fn try_new_local(
        kv_router_config: crate::config::KvRouterConfig,
        indexer_threads: usize,
        cancel_token: CancellationToken,
        cache_config: SelectionCacheConfig,
    ) -> anyhow::Result<Self> {
        kv_router_config
            .validate_config()
            .map_err(anyhow::Error::msg)?;
        let tracking_hash = Arc::new(TrackingHashContext::from_config(&kv_router_config)?);
        let indexer_policy = IndexerPolicy::from_router_config(&kv_router_config)?;
        Ok(Self::new_inner(
            kv_router_config,
            indexer_threads,
            cancel_token,
            None,
            None,
            SelectionHost::default(),
            WorkerType::Aggregated,
            true,
            cache_config,
            tracking_hash,
            indexer_policy,
            None,
        ))
    }

    /// Scheduler and indexer handle for `key`, once a worker has been upserted
    /// into that partition.
    pub fn partition(&self, key: &RoutingPartitionId) -> Option<SelectionPartition> {
        self.entry(key).map(SelectionPartition)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new_inner(
        kv_router_config: crate::config::KvRouterConfig,
        indexer_threads: usize,
        cancel_token: CancellationToken,
        replica_config: Option<ReplicaSyncConfig>,
        worker_selection_policy_factory: Option<WorkerSelectionPolicyFactory>,
        host: SelectionHost,
        worker_type: WorkerType,
        signal_indexer_ready: bool,
        cache_config: SelectionCacheConfig,
        tracking_hash: Arc<TrackingHashContext>,
        indexer_policy: IndexerPolicy,
        session_affinity: Option<SessionAffinityConfig>,
    ) -> Self {
        let cancel_token = cancel_token.child_token();
        let indexer_registry = Arc::new(
            WorkerRegistry::new_with_cancel_token(indexer_threads, cancel_token.clone())
                .retain_empty_partitions(),
        );
        let listens_for_kv_events = kv_router_config.use_kv_events && !indexer_policy.is_remote();
        indexer_registry.set_indexer_policy(indexer_policy);
        if signal_indexer_ready {
            indexer_registry.signal_ready();
        }
        Self {
            catalog: WorkerCatalog::default(),
            catalog_updates: tokio::sync::Mutex::new(()),
            entries: Arc::new(RwLock::new(HashMap::new())),
            reservation_index: Arc::new(RwLock::new(HashMap::new())),
            reservation_sweep_started: OnceCell::new(),
            listens_for_kv_events,
            indexer_registry,
            kv_router_config,
            worker_selection_policy_factory,
            host,
            worker_type,
            cancel_token,
            replica_config,
            selection_cache: SelectionCache::new(&cache_config),
            tracking_hash,
            session_affinity,
        }
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

    fn ensure_running(&self) -> Result<(), SelectionError> {
        if self.cancel_token.is_cancelled() {
            return Err(SelectionError::NotReady(
                "selection service is shutting down".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) async fn recover_indexer_from_peers(
        &self,
        peers: &[String],
    ) -> anyhow::Result<bool> {
        recovery::recover_from_peers(peers, &self.indexer_registry).await
    }

    pub(crate) fn signal_indexer_ready(&self) {
        self.indexer_registry.signal_ready();
    }

    pub(crate) async fn dump_indexer_events(&self) -> serde_json::Value {
        crate::services::indexer::server::dump_registry(&self.indexer_registry).await
    }

    pub(crate) fn dispatch_replica_event(&self, envelope: ScopedReplicaEvent) {
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
        if entry.block_size != block_size {
            tracing::debug!(
                %key,
                expected_block_size = entry.block_size,
                received_block_size = block_size,
                "Dropping selector replica event with mismatched block size"
            );
            return;
        }
        let Some(replica_tx) = &entry.replica_tx else {
            return;
        };
        match replica_tx.try_send(event) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(event)) => {
                tracing::trace!(
                    %key,
                    request_id = %event.request_id,
                    "Selector replica subscriber channel full; dropping event"
                );
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::debug!(%key, "Selector replica subscriber channel closed");
            }
        }
    }

    pub async fn upsert_worker(
        &self,
        req: WorkerRequest,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        self.ensure_running()?;
        let mut record = WorkerCatalogRecord::new(req);
        // Partition policy factories may construct independently. Only committing
        // membership and reconciling ingress needs the catalog mutation lock.
        self.prepare_worker(&mut record)?;
        let _update = self.catalog_updates.lock().await;
        self.ensure_running()?;
        let previous = self.catalog.get(record.worker_id);
        self.reconcile_worker(record, previous).await
    }

    pub async fn patch_worker(
        &self,
        worker_id: WorkerId,
        patch: WorkerPatchRequest,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let _update = self.catalog_updates.lock().await;
        self.ensure_running()?;
        let previous = self
            .catalog
            .get(worker_id)
            .ok_or_else(|| SelectionError::NotFound(format!("worker {worker_id} not found")))?;
        let mut record = previous.clone();
        record.apply_patch(patch);
        self.prepare_worker(&mut record)?;
        self.reconcile_worker(record, Some(previous)).await
    }

    pub async fn delete_worker(
        &self,
        worker_id: WorkerId,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let _update = self.catalog_updates.lock().await;
        let Some(previous) = self.catalog.get(worker_id) else {
            return Err(SelectionError::NotFound(format!(
                "worker {worker_id} not found"
            )));
        };
        let key = previous.key();
        self.catalog
            .set_lifecycle(worker_id, WorkerLifecycle::Draining, Vec::new());
        self.publish_scheduler_config(&key)?;
        self.cleanup_indexer_registration(&previous).await;
        let record = self
            .catalog
            .set_lifecycle(worker_id, WorkerLifecycle::Unschedulable, Vec::new())
            .ok_or_else(|| SelectionError::NotFound(format!("worker {worker_id} not found")))?;
        self.publish_scheduler_config(&key)?;
        Ok(record)
    }

    pub fn list_workers(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<WorkerCatalogRecord> {
        self.catalog.list(model_name, routing_group)
    }

    pub fn ready(&self) -> ReadyResponse {
        let schedulable_workers = self.catalog.schedulable_count();
        let workers = self.catalog.list(None, None);
        ReadyResponse {
            ready: !self.cancel_token.is_cancelled() && schedulable_workers > 0,
            schedulable_workers,
            workers,
        }
    }

    fn prepare_worker(&self, record: &mut WorkerCatalogRecord) -> Result<(), SelectionError> {
        let queueing_enabled = self
            .kv_router_config
            .queueing_enabled(Some(&record.model_name))
            .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
        record.not_schedulable_reasons = record.missing_schedulable_metadata(queueing_enabled);
        if let Some(ingress) = self.ingress() {
            record
                .not_schedulable_reasons
                .extend(ingress.missing_metadata(record));
        }
        if record.not_schedulable_reasons.is_empty()
            && let Err(error) = self.ensure_entry(record)
        {
            record
                .not_schedulable_reasons
                .push(format!("reconciliation failed: {error}"));
        }

        Ok(())
    }

    async fn reconcile_worker(
        &self,
        mut record: WorkerCatalogRecord,
        previous: Option<WorkerCatalogRecord>,
    ) -> Result<WorkerCatalogRecord, SelectionError> {
        let previous = previous.filter(|old| old.lifecycle == WorkerLifecycle::Schedulable);
        let previous = if let Some(old) = previous.as_ref()
            && (old.key() != record.key() || !record.not_schedulable_reasons.is_empty())
        {
            self.catalog
                .set_lifecycle(old.worker_id, WorkerLifecycle::Draining, Vec::new());
            self.publish_scheduler_config(&old.key())?;
            self.cleanup_indexer_registration(old).await;
            None
        } else {
            previous
        };

        if record.not_schedulable_reasons.is_empty()
            && let Some(ingress) = self.ingress()
            && let Err(error) = ingress
                .reconcile(&self.indexer_registry, previous.as_ref(), &record)
                .await
        {
            self.cleanup_indexer_registration(&record).await;
            record
                .not_schedulable_reasons
                .push(format!("reconciliation failed: {error}"));
        }
        record.lifecycle = if record.not_schedulable_reasons.is_empty() {
            WorkerLifecycle::Schedulable
        } else {
            WorkerLifecycle::Incomplete
        };
        // Readers see only committed metadata. A valid capacity/topology update preserves
        // live bookings on ranks present in both the old and new snapshots.
        self.catalog.replace(record.clone());
        self.publish_scheduler_config(&record.key())?;
        Ok(record)
    }

    fn ensure_entry(
        &self,
        record: &WorkerCatalogRecord,
    ) -> Result<Arc<SelectionEntry>, SelectionError> {
        let block_size = record
            .block_size
            .ok_or_else(|| SelectionError::BadRequest("block_size is required".to_string()))?;
        self.ensure_entry_for(record.key(), block_size, record.is_eagle.unwrap_or(false))
    }

    /// Create the partition scheduler and indexer for `key` before any worker
    /// registers, so an embedding host can hold the scheduler handle from
    /// construction. Idempotent; a later worker with a different block size or
    /// eagle setting is rejected at reconciliation.
    pub fn ensure_partition(
        &self,
        key: RoutingPartitionId,
        block_size: u32,
        is_eagle: bool,
    ) -> Result<SelectionPartition, SelectionError> {
        self.ensure_running()?;
        if block_size == 0 {
            return Err(SelectionError::BadRequest(
                "block_size must be greater than 0".to_string(),
            ));
        }
        self.ensure_entry_for(key, block_size, is_eagle)
            .map(SelectionPartition)
    }

    fn ensure_entry_for(
        &self,
        key: RoutingPartitionId,
        block_size: u32,
        is_eagle: bool,
    ) -> Result<Arc<SelectionEntry>, SelectionError> {
        self.reservation_sweep_started.get_or_init(|| {
            spawn_reservation_index_sweep(
                Arc::clone(&self.entries),
                Arc::clone(&self.reservation_index),
                self.cancel_token.child_token(),
            );
        });

        let entry_cell = { self.entries.read().get(&key).cloned() };
        let entry_cell = entry_cell.unwrap_or_else(|| {
            self.entries
                .write()
                .entry(key.clone())
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        });
        let entry = entry_cell
            .get_or_try_init(|| -> Result<Arc<SelectionEntry>, SelectionError> {
                let (workers_tx, workers_rx) = watch::channel(HashMap::new());
                let host_replica = self
                    .host
                    .replication
                    .channels
                    .as_ref()
                    .and_then(|factory| factory(&key));
                let scoped_replica_sync = setup_scoped_replica_sync(
                    self.replica_config.as_ref(),
                    &key,
                    block_size,
                    host_replica,
                );
                let worker_label = self.worker_type.as_str();
                let slots = Arc::new(ActiveSequencesMultiWorker::new_with_options(
                    scoped_replica_sync
                        .publisher
                        .with_load_sink(self.host.telemetry.scheduler_load.clone()),
                    block_size as usize,
                    HashMap::new(),
                    scoped_replica_sync.enabled,
                    scoped_replica_sync.process_id,
                    worker_label,
                    SequenceTrackerOptions {
                        replica_worker_policy: ReplicaWorkerPolicy::RequireRegistered,
                        expiry_duration: self
                            .host
                            .replication
                            .request_leases
                            .is_none()
                            .then(active_request_expiry_duration),
                    },
                ));
                slots.set_replica_request_lease_observer(Arc::new(ReservationIndexObserver {
                    index: Arc::clone(&self.reservation_index),
                    partition: key.clone(),
                    host: self.host.replication.request_leases.clone(),
                }));
                let replica_tx = scoped_replica_sync.channel.map(|(replica_tx, subscriber)| {
                    slots.start_replica_sync(subscriber, self.cancel_token.child_token());
                    replica_tx
                });
                if self.host.replication.request_leases.is_none() {
                    slots.start_periodic_force_expiry_across_all_workers(
                        self.cancel_token.child_token(),
                    );
                }

                let indexer = match &self.host.cache.index {
                    KvIndexSource::Owned(ingress) => {
                        ingress.open(&self.indexer_registry, &key, block_size)
                    }
                    KvIndexSource::Remote(_) => self
                        .indexer_registry
                        .get_or_create_indexer(key.clone(), block_size),
                };
                let overlap_refresh = indexer.supports_overlap_refresh().then(|| {
                    Arc::new(TieredOverlapRefresher::new(
                        indexer.clone(),
                        self.kv_router_config.clone(),
                        block_size,
                    ))
                });
                let selector = self.worker_selection_policy_factory.as_ref().map_or_else(
                    || WorkerSelectionPolicy::default(self.kv_router_config.clone(), worker_label),
                    |factory| factory(&self.kv_router_config, self.worker_type, key.as_ref()),
                );
                let profile = self
                    .kv_router_config
                    .policy_profile(Some(&key.model_name))
                    .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
                let scheduler = LocalScheduler::new(
                    slots,
                    workers_rx,
                    profile,
                    block_size,
                    selector,
                    self.host.load.prefill_estimator.clone(),
                    overlap_refresh,
                    // Standalone selection has no router Client snapshot, so
                    // these stay `None` unless an embedding host injects them.
                    self.host.load.overloaded_workers.clone(),
                    self.host.load.available_workers.clone(),
                    self.kv_router_config.router_queue_recheck_interval(),
                    self.kv_router_config.router_track_prefill_tokens,
                    self.cancel_token.child_token(),
                    worker_label,
                    true,
                );
                Ok(Arc::new(SelectionEntry {
                    key: key.clone(),
                    block_size,
                    is_eagle,
                    indexer,
                    workers_tx,
                    scheduler,
                    replica_tx,
                    affinity: OnceCell::new(),
                    replica_config: self.replica_config.clone(),
                }))
            })?
            .clone();
        if entry.block_size != block_size {
            return Err(SelectionError::Conflict(format!(
                "block_size mismatch for {key}: existing={} requested={block_size}",
                entry.block_size
            )));
        }
        if entry.is_eagle != is_eagle {
            return Err(SelectionError::Conflict(format!(
                "is_eagle mismatch for {key}: existing={} requested={is_eagle}",
                entry.is_eagle
            )));
        }
        if let Some(config) = self.session_affinity {
            entry.session_affinity(config)?;
        }
        Ok(entry)
    }

    /// The ingress feeding core-owned indexes, when this core listens for KV events.
    fn ingress(&self) -> Option<&dyn KvEventIngress> {
        match &self.host.cache.index {
            KvIndexSource::Owned(ingress) if self.listens_for_kv_events => Some(ingress.as_ref()),
            _ => None,
        }
    }

    async fn cleanup_indexer_registration(&self, record: &WorkerCatalogRecord) {
        if let KvIndexSource::Owned(ingress) = &self.host.cache.index {
            ingress.detach(&self.indexer_registry, record).await;
            return;
        }

        let key = record.key();
        let indexer = self
            .indexer_registry
            .get_indexer(&key)
            .map(|entry| entry.indexer.clone());
        if let Some(indexer) = indexer {
            indexer.remove_worker(record.worker_id).await;
        }
    }

    fn publish_scheduler_config(&self, key: &RoutingPartitionId) -> Result<(), SelectionError> {
        let Some(entry) = self.entry(key) else {
            return Ok(());
        };
        let workers = self.catalog.scheduler_configs_for_key(key);
        // Lifecycle transitions between non-schedulable states publish the same
        // map; skipping them saves the scheduler a wake and a full map clone.
        entry.workers_tx.send_if_modified(|current| {
            if *current == workers {
                false
            } else {
                *current = workers;
                true
            }
        });
        Ok(())
    }

    fn ready_entry(&self, key: &RoutingPartitionId) -> Result<Arc<SelectionEntry>, SelectionError> {
        if self.catalog.schedulable_count() == 0 {
            return Err(SelectionError::NotReady(
                "no schedulable workers are available".to_string(),
            ));
        }

        let Some(entry) = self.entry(key) else {
            return Err(SelectionError::NotReady(format!(
                "no schedulable workers for {key}"
            )));
        };
        if !self.catalog.has_schedulable_for_key(key) {
            return Err(SelectionError::NotReady(format!(
                "no schedulable workers for {key}"
            )));
        }
        Ok(entry)
    }

    pub async fn select(&self, req: SelectRequest) -> Result<SelectResponse, SelectionError> {
        self.select_with_policy_class(req, None).await
    }

    pub async fn select_with_policy_class(
        &self,
        mut req: SelectRequest,
        policy_class: Option<String>,
    ) -> Result<SelectResponse, SelectionError> {
        let session_context = req.take_session_context();
        self.schedule_selection(
            SelectionOperation {
                key: RoutingPartitionId::new(req.model_name, req.routing_group),
                selection_id: req.selection_id,
                prompt: req.prompt,
                router_config_override: req.router_config_override,
                expected_output_tokens: req.expected_output_tokens,
                priority_jump: req.priority_jump.unwrap_or_default(),
                strict_priority: req.strict_priority.unwrap_or(0),
                policy_class,
                session_context,
                affinity_target: req.affinity_target,
                pinned_worker: req.pinned_worker,
                allowed_worker_ids: req.allowed_worker_ids,
                routing_constraints: req.routing_constraints,
                advisory: req.advisory,
            },
            false,
        )
        .await
    }

    pub async fn select_and_reserve(
        &self,
        req: SelectAndReserveRequest,
    ) -> Result<SelectResponse, SelectionError> {
        self.select_and_reserve_with_policy_class(req, None).await
    }

    pub async fn select_and_reserve_with_policy_class(
        &self,
        mut req: SelectAndReserveRequest,
        policy_class: Option<String>,
    ) -> Result<SelectResponse, SelectionError> {
        let session_context = req.take_session_context();
        let selection_id = req
            .selection_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        self.schedule_selection(
            SelectionOperation {
                key: RoutingPartitionId::new(req.model_name, req.routing_group),
                selection_id: Some(selection_id),
                prompt: req.prompt,
                router_config_override: req.router_config_override,
                expected_output_tokens: req.expected_output_tokens,
                priority_jump: req.priority_jump.unwrap_or_default(),
                strict_priority: req.strict_priority.unwrap_or(0),
                policy_class,
                session_context,
                affinity_target: req.affinity_target,
                pinned_worker: req.pinned_worker,
                allowed_worker_ids: req.allowed_worker_ids,
                routing_constraints: req.routing_constraints,
                advisory: false,
            },
            true,
        )
        .await
    }

    async fn schedule_selection(
        &self,
        operation: SelectionOperation,
        book: bool,
    ) -> Result<SelectResponse, SelectionError> {
        let SelectionOperation {
            key,
            selection_id,
            prompt,
            router_config_override,
            expected_output_tokens,
            priority_jump,
            strict_priority,
            policy_class,
            session_context,
            affinity_target,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            advisory,
        } = operation;
        self.ensure_running()?;

        let entry = self.ready_entry(&key)?;
        let claim = match selection_id.as_deref() {
            Some(selection_id) if book => Some(self.claim_reservation(selection_id, &key)?),
            _ => None,
        };

        // Session stickiness: a bound session steers selection (exclusive for
        // the default selector); a new session is bound to the worker chosen.
        // An explicit affinity target or pin from the caller wins.
        let table = entry.affinity.get();
        let session_id = table
            .and(session_context.as_ref())
            .filter(|_| affinity_target.is_none() && pinned_worker.is_none())
            .map(|context| context.session_id().to_string());
        let mut affinity_hold = None;
        let affinity_target = match (session_id.as_deref(), table) {
            (Some(session_id), Some(table)) if book => {
                affinity_hold = self.hold_session(table, session_id, &key).await?;
                affinity_hold.as_ref().and_then(Hold::target)
            }
            (Some(session_id), Some(table)) => table
                .query_target(session_id, None)
                .map_err(affinity_error)?
                .map(|target| WorkerAffinityTarget::new(target.worker_id, target.dp_rank)),
            _ => affinity_target,
        };
        // Router hints are attached to bookings only, and only when a worker in
        // this partition can consume them and the indexer can retain the
        // matched chain (local, event-driven, no approximate writes).
        let retain_kv_transfer_chain = book
            && entry.indexer.supports_kv_transfer_chain_retention()
            && self.catalog.has_router_hint_capable_workers(&key);
        let PreparedSelectionInputs {
            block_hashes,
            sequence_hashes,
            isl_tokens,
            overlap,
            shared_cache_hits,
            kv_transfer_candidates,
        } = self
            .prepare_selection_inputs(
                &entry,
                &prompt,
                self.kv_router_config
                    .assume_kv_reuse(router_config_override.as_ref()),
                true,
                retain_kv_transfer_chain,
            )
            .await?;
        // The queue lease frees a booking whose response is never consumed
        // (the caller dropped this future after the actor booked).
        let mode = if book {
            ScheduleMode::TrackedWithLifecycle {
                request_id: selection_id.clone().ok_or_else(|| {
                    SelectionError::Internal(
                        "booked selection did not include a selection ID".to_string(),
                    )
                })?,
            }
        } else {
            ScheduleMode::QueryOnly {
                request_id: selection_id.clone(),
            }
        };
        let track_prefill_tokens = router_config_override
            .as_ref()
            .and_then(|cfg| cfg.track_prefill_tokens)
            .unwrap_or(self.kv_router_config.router_track_prefill_tokens);
        // `select` (book == false) with a selection_id caches the booking inputs
        // so a follow-up `create_reservation` can replay them by that id.
        let cached_inputs = (!book).then(|| selection_id.clone()).flatten().map(|id| {
            (
                id,
                sequence_hashes.clone(),
                prompt.lora_name.clone(),
                track_prefill_tokens,
            )
        });
        let allowed_worker_ids = match self.host.eligibility.lora_worker_filter.as_deref() {
            Some(filter) => narrow_allowed_worker_ids_by_lora(
                filter,
                prompt.lora_name.as_deref(),
                allowed_worker_ids,
                pinned_worker.as_ref(),
                || self.catalog.schedulable_worker_ids_for_key(&key),
            ),
            None => allowed_worker_ids,
        };
        let response_sequence_hashes =
            book.then(|| sequence_hashes.iter().map(|hash| *hash as i64).collect());
        let response_isl_tokens = book.then_some(isl_tokens);
        let response_track_prefill_tokens = book.then_some(track_prefill_tokens);
        // Bookings (now, or later via the pending-selection cache) are recorded
        // into an approximate indexer; keep the public hashes for that.
        let routing_hashes = (entry.indexer.records_routing_decisions()
            && (book || cached_inputs.is_some()))
        .then(|| block_hashes.clone());
        let schedule_request = ScheduleRequest {
            mode,
            token_seq: Some(sequence_hashes),
            block_hashes: Some(block_hashes),
            isl_tokens,
            overlap,
            kv_transfer_candidates,
            retain_kv_transfer_chain,
            router_config_override,
            lora_name: prompt.lora_name,
            priority_jump,
            strict_priority,
            policy_class,
            session_context,
            expected_output_tokens,
            affinity_target,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            shared_cache_hits,
        };
        // `lease` guards the booking until it is installed below: any early
        // return or drop before then frees it.
        let (response, advisory_load, lease) = tokio::select! {
            biased;
            _ = self.cancel_token.cancelled() => {
                return Err(SelectionError::Scheduler(KvSchedulerError::SubscriberShutdown));
            }
            result = async {
                if advisory {
                    entry
                        .scheduler
                        .select_without_admission(schedule_request)
                        .await
                        .map(|advisory| {
                            (advisory.response, Some(advisory.selected_worker_load), None)
                        })
                } else {
                    entry
                        .scheduler
                        .schedule_request_with_lease(schedule_request)
                        .await
                        .map(|(admitted, lease)| (admitted.response, None, lease))
                }
            } => result?,
        };
        let Some(endpoint) = self
            .catalog
            .schedulable_endpoint(response.best_worker.worker_id, &key)
        else {
            return Err(SelectionError::Internal(format!(
                "selected worker {} is no longer schedulable",
                response.best_worker.worker_id
            )));
        };
        let overlap = MooncakeOverlapSummary::from_selected_worker_tiers(
            &response.selected_worker_tiers,
            entry.block_size,
        );

        let effective_prefill = effective_prefill_tokens(isl_tokens, response.cached_tokens);
        let potential_decode_blocks = response.potential_decode_blocks as u64;
        let total_kv_blocks = advisory_load
            .and_then(|load| load.total_kv_blocks.map(|blocks| blocks as u64))
            .or_else(|| {
                self.catalog
                    .total_kv_blocks(response.best_worker.worker_id, &key)
            });
        let decode_busy = self
            .kv_router_config
            .conditional_disagg_decode_busy_threshold
            .zip(total_kv_blocks)
            .map(|(threshold, total_kv_blocks)| {
                potential_decode_blocks as f64 > threshold * total_kv_blocks as f64
            });
        let kv_hint = if retain_kv_transfer_chain {
            transfer_hint_for_selection(
                &self.catalog.scheduler_configs_for_key(&key),
                response.best_worker,
                response.target_cached_prefix_blocks,
                response.kv_transfer_candidates.as_ref(),
            )
            .map(|payload| {
                KvHint::new(
                    selection_id.as_deref().unwrap_or_default(),
                    vec![KvHintAction::fetch("a1", payload)],
                )
            })
        } else {
            None
        };
        let worker_load = advisory_load.map(|load| SelectionWorkerLoad {
            active_prefill_tokens: load.active_prefill_tokens,
            prefill_token_capacity: load.prefill_token_capacity,
            total_kv_blocks,
            prefill_busy: self
                .kv_router_config
                .conditional_disagg_prefill_busy_threshold
                .map(|threshold| load.prefill_load_exceeds(threshold)),
        });

        if let Some(claim) = claim {
            let Some(lease) = lease else {
                return Err(SelectionError::Internal(
                    "booked selection has no lifecycle lease".to_string(),
                ));
            };
            // A rejected affinity commit returns while the lease is still armed,
            // so the booking is freed and nothing below is recorded.
            let affinity_lease = match (affinity_hold, session_id.as_deref(), table) {
                (Some(hold), Some(session_id), Some(table)) => Some(
                    self.commit_session(table, hold, session_id, response.best_worker, &key)
                        .await?,
                ),
                _ => None,
            };
            if let Some(hashes) = routing_hashes.clone() {
                self.record_routing_decision(&entry, response.best_worker, hashes)
                    .await;
            }
            claim.install(Reservation {
                partition: key.clone(),
                booking: Some(lease.commit().ok_or_else(missing_booking)?),
                _affinity_lease: affinity_lease,
            });
        }

        if let Some((cache_id, sequence_hashes, lora_name, track_prefill_tokens)) = cached_inputs {
            self.selection_cache.insert(
                cache_id,
                PendingSelection {
                    key: key.clone(),
                    worker: response.best_worker,
                    sequence_hashes,
                    isl_tokens,
                    effective_prefill_tokens: effective_prefill,
                    expected_output_tokens,
                    track_prefill_tokens,
                    lora_name,
                    routing_hashes,
                },
                Instant::now(),
            );
        }

        Ok(SelectResponse {
            selection_id,
            sequence_hashes: response_sequence_hashes,
            isl_tokens: response_isl_tokens,
            track_prefill_tokens: response_track_prefill_tokens,
            model_name: key.model_name,
            routing_group: key.routing_group,
            worker_id: response.best_worker.worker_id,
            dp_rank: response.best_worker.dp_rank,
            endpoint,
            block_size: entry.block_size,
            overlap,
            effective_prefill_tokens: effective_prefill,
            potential_decode_blocks,
            decode_busy,
            worker_load,
            kv_hint,
        })
    }

    pub async fn create_reservation(
        &self,
        req: ReservationRequest,
    ) -> Result<ReservationResponse, SelectionError> {
        self.ensure_running()?;

        let key = RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone());

        // Explicit form: book on the given worker under selection_id, discarding
        // any cached selection for the id so a later replay can't book stale state.
        if let Some(worker_id) = req.worker_id {
            self.selection_cache.discard(&key, &req.selection_id);
            return self.reserve_explicit(key, worker_id, req).await;
        }

        // Replay form: peek, book, and consume only once the booking lands. A
        // failure leaves the entry for a retry; concurrent replays of the same
        // id collide at the scheduler, so they can't double-book.
        let Some((pending, generation)) =
            self.selection_cache
                .peek(&key, &req.selection_id, Instant::now())
        else {
            return Err(SelectionError::NotFound(format!(
                "no pending selection {} for {key} (expired, already used, \
                 or never selected)",
                req.selection_id
            )));
        };
        let response = self.book_cached_selection(pending, &req).await?;
        self.selection_cache
            .remove(&key, &req.selection_id, generation);
        Ok(response)
    }

    /// Book a reservation replaying what the matching `select` captured; request
    /// fields other than the ids are ignored.
    async fn book_cached_selection(
        &self,
        pending: Arc<PendingSelection>,
        req: &ReservationRequest,
    ) -> Result<ReservationResponse, SelectionError> {
        let (entry, endpoint, prefill_load_hint) = self.resolve_cached_booking(&pending)?;
        let track_prefill_tokens = pending.track_prefill_tokens;
        self.finalize_reservation(
            entry,
            endpoint,
            ReservationBooking {
                key: pending.key.clone(),
                selection_id: req.selection_id.clone(),
                worker: pending.worker,
                sequence_hashes: pending.sequence_hashes.clone(),
                prefill_load_hint: track_prefill_tokens.then_some(prefill_load_hint),
                expected_output_tokens: pending.expected_output_tokens,
                track_prefill_tokens,
                lora_name: pending.lora_name.clone(),
                routing_hashes: pending.routing_hashes.clone(),
            },
        )
        .await
    }

    /// Resolve everything a cached booking needs (ready entry, schedulable
    /// endpoint, prefill hint), so the only fallible step left in
    /// `finalize_reservation` is the scheduler call.
    fn resolve_cached_booking(
        &self,
        pending: &PendingSelection,
    ) -> Result<(Arc<SelectionEntry>, String, PrefillLoadHint), SelectionError> {
        let entry = self.ready_entry(&pending.key)?;
        // Validate the full worker/rank against current topology; a rank a PATCH
        // removed during the window is rejected (the entry stays for a retry).
        let endpoint = self
            .catalog
            .schedulable_worker_endpoint(pending.worker, &pending.key)
            .ok_or_else(|| {
                SelectionError::NotFound(format!(
                    "schedulable worker {} (dp_rank {}) not found for {}",
                    pending.worker.worker_id, pending.worker.dp_rank, pending.key
                ))
            })?;
        let prefill_load_hint = prefill_load_hint_from_effective_tokens(
            pending.isl_tokens,
            pending.effective_prefill_tokens,
        )
        .map_err(|error| SelectionError::BadRequest(error.to_string()))?;
        Ok((entry, endpoint, prefill_load_hint))
    }

    fn schedulable_endpoint(
        &self,
        worker_id: WorkerId,
        key: &RoutingPartitionId,
    ) -> Result<String, SelectionError> {
        self.catalog
            .schedulable_endpoint(worker_id, key)
            .ok_or_else(|| {
                SelectionError::NotFound(format!(
                    "schedulable worker {worker_id} not found for {key}"
                ))
            })
    }

    /// Book a reservation from a self-contained request (explicit worker_id and prompt).
    async fn reserve_explicit(
        &self,
        key: RoutingPartitionId,
        worker_id: WorkerId,
        req: ReservationRequest,
    ) -> Result<ReservationResponse, SelectionError> {
        let entry = self.ready_entry(&key)?;
        let normalized = req.prompt.normalize_for_reservation(
            entry.is_eagle,
            TrackingHashInput {
                context: &self.tracking_hash,
                scope: tracking_scope(&entry),
                assume_kv_reuse: self
                    .kv_router_config
                    .assume_kv_reuse(req.router_config_override.as_ref()),
            },
        )?;
        let prefill_load_hint = req
            .effective_prefill_tokens
            .map(|tokens| {
                prefill_load_hint_from_effective_tokens(normalized.isl_tokens, tokens)
                    .map_err(|error| SelectionError::BadRequest(error.to_string()))
            })
            .transpose()?;
        let worker = WorkerWithDpRank::new(worker_id, req.dp_rank.unwrap_or(0));
        let endpoint = self.schedulable_endpoint(worker.worker_id, &key)?;
        let track_prefill_tokens = req.track_prefill_tokens.unwrap_or_else(|| {
            req.effective_prefill_tokens.is_some()
                || req
                    .router_config_override
                    .as_ref()
                    .and_then(|cfg| cfg.track_prefill_tokens)
                    .unwrap_or(self.kv_router_config.router_track_prefill_tokens)
        });
        // Hash-only reservations (sequence hashes without block hashes) carry
        // nothing an indexer can key on; recording is skipped for them.
        let can_record = entry.indexer.records_routing_decisions()
            && (req.prompt.token_ids.is_some() || req.prompt.block_hashes.is_some());
        let routing_hashes = can_record
            .then(|| {
                req.prompt
                    .block_hashes_for_indexer(entry.block_size, entry.is_eagle)
            })
            .transpose()?;

        self.finalize_reservation(
            entry,
            endpoint,
            ReservationBooking {
                key,
                selection_id: req.selection_id,
                worker,
                sequence_hashes: normalized.sequence_hashes,
                prefill_load_hint,
                expected_output_tokens: req.expected_output_tokens,
                track_prefill_tokens,
                lora_name: req.prompt.lora_name,
                routing_hashes,
            },
        )
        .await
    }

    /// Register the booking with the scheduler. All fallible resolution happens
    /// in the caller; the scheduler add here is the last step that can fail, and
    /// the cached path leaves its selection in place (to retry) if it does.
    async fn finalize_reservation(
        &self,
        entry: Arc<SelectionEntry>,
        endpoint: String,
        booking: ReservationBooking,
    ) -> Result<ReservationResponse, SelectionError> {
        let ReservationBooking {
            key,
            selection_id,
            worker,
            sequence_hashes,
            prefill_load_hint,
            expected_output_tokens,
            track_prefill_tokens,
            lora_name,
            routing_hashes,
        } = booking;

        let claim = self.claim_reservation(&selection_id, &key)?;
        // Strict booking: never lazily recreate a worker/rank removed since the
        // reservation was resolved. The lease frees the booking if this future
        // is dropped before `install`.
        let lease = entry
            .scheduler
            .add_request_if_registered_guarded(SequenceRequest {
                request_id: selection_id.clone(),
                token_sequence: Some(sequence_hashes),
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
                worker,
                lora_name,
            })?;
        if let Some(hashes) = routing_hashes {
            self.record_routing_decision(&entry, worker, hashes).await;
        }
        claim.install(Reservation {
            partition: key.clone(),
            booking: Some(lease.commit().ok_or_else(missing_booking)?),
            _affinity_lease: None,
        });

        Ok(ReservationResponse {
            selection_id,
            model_name: key.model_name,
            routing_group: key.routing_group,
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            endpoint,
        })
    }

    /// Take `selection_id` for a booking about to be made in `key`.
    fn claim_reservation(
        &self,
        selection_id: &str,
        key: &RoutingPartitionId,
    ) -> Result<ReservationClaim<'_>, SelectionError> {
        let mut index = self.reservation_index.write();
        if index.contains_key(selection_id) {
            return Err(SelectionError::Conflict(format!(
                "selection {selection_id} is already reserved or being reserved"
            )));
        }
        index.insert(
            selection_id.to_string(),
            Reservation {
                partition: key.clone(),
                booking: None,
                _affinity_lease: None,
            },
        );
        Ok(ReservationClaim {
            index: &self.reservation_index,
            selection_id: selection_id.to_string(),
            armed: true,
        })
    }

    /// Hold `session_id` for a booking, re-initializing a binding whose worker
    /// this partition can no longer schedule. `None` when the table is full: a
    /// router-side limit, not a client fault, so the request routes unpinned.
    async fn hold_session(
        &self,
        table: &SessionAffinity,
        session_id: &str,
        key: &RoutingPartitionId,
    ) -> Result<Option<Hold>, SelectionError> {
        loop {
            let acquired = tokio::select! {
                _ = self.cancel_token.cancelled() => {
                    return Err(SelectionError::Scheduler(KvSchedulerError::SubscriberShutdown));
                }
                result = table.acquire(session_id, None) => result,
            };
            match acquired {
                Ok(Hold::Bound { target, mut lease })
                    if !self.catalog.is_schedulable(target.worker_id, key) =>
                {
                    tracing::debug!(
                        session_id,
                        worker_id = target.worker_id,
                        "session affinity target is not schedulable; re-initializing"
                    );
                    lease.invalidate();
                }
                Ok(hold) => return Ok(Some(hold)),
                Err(AffinityError::ResourceExhausted(_)) => {
                    tracing::debug!(
                        session_id,
                        "affinity table full; routing without session affinity"
                    );
                    return Ok(None);
                }
                Err(error) => return Err(affinity_error(error)),
            }
        }
    }

    /// Bind the held session to `dispatched`. A `Hard` rejection whose bound
    /// worker departed after [`Self::hold_session`] checked it is not a client
    /// fault: the binding is re-initialized on the dispatched worker instead.
    async fn commit_session(
        &self,
        table: &SessionAffinity,
        hold: Hold,
        session_id: &str,
        dispatched: WorkerWithDpRank,
        key: &RoutingPartitionId,
    ) -> Result<AffinityLease, SelectionError> {
        let bound = hold.target();
        let dispatched = WorkerAffinityTarget::new(dispatched.worker_id, Some(dispatched.dp_rank));
        match table.commit(hold, dispatched) {
            Ok(lease) => Ok(lease),
            Err(error) => {
                let departed =
                    bound.is_some_and(|target| !self.catalog.is_schedulable(target.worker_id, key));
                if !departed {
                    return Err(affinity_error(error));
                }
                // `commit` invalidated the stale binding; the next hold initializes.
                match self.hold_session(table, session_id, key).await? {
                    Some(hold) => table.commit(hold, dispatched).map_err(affinity_error),
                    None => Err(affinity_error(error)),
                }
            }
        }
    }

    /// Apply a session binding a replica published.
    pub(crate) fn dispatch_affinity_event(&self, event: AffinityBindingEvent) {
        let Some(entry) = self.entry(&event.partition) else {
            return;
        };
        let Some(table) = entry.affinity.get() else {
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
            return;
        }
        let (target, version, worker_id) = (event.target(), event.version(), event.worker_id);
        let outcome = table.apply_replica_update(event.session_id, target, version);
        tracing::trace!(
            worker_id,
            ?outcome,
            "applied session affinity replica update"
        );
    }

    /// Record a booked routing decision into the partition's approximate
    /// indexer (side or primary). The booking already landed, so a failure
    /// here only costs predicted cache credit; it is logged, not returned.
    async fn record_routing_decision(
        &self,
        entry: &SelectionEntry,
        worker: WorkerWithDpRank,
        block_hashes: Vec<LocalBlockHash>,
    ) {
        if block_hashes.is_empty() {
            return;
        }
        if let Err(error) = entry
            .indexer
            .record_routing_decision(
                worker,
                RoutingDecisionHashes::from_local_hashes(block_hashes),
            )
            .await
        {
            tracing::warn!(
                %error,
                key = %entry.key,
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                "Failed to record routing decision into approximate indexer"
            );
        }
    }

    /// The partition and booking a lifecycle call on `selection_id` may touch.
    /// `None` for unknown ids and for ids whose booking is still in flight.
    fn indexed_booking(
        &self,
        selection_id: &str,
    ) -> Option<(Arc<SelectionEntry>, SchedulerBookingDescriptor)> {
        // Release the index guard before taking `entries`: the sweep takes
        // `entries` then `reservation_index`, so nesting here would invert the
        // lock order.
        let (partition, booking) = {
            let index = self.reservation_index.read();
            let reservation = index.get(selection_id)?;
            (reservation.partition.clone(), reservation.booking.clone()?)
        };
        Some((self.entry(&partition)?, booking))
    }

    fn reservation_not_found(selection_id: &str) -> SelectionError {
        SelectionError::NotFound(format!("reservation {selection_id} not found"))
    }

    pub async fn prefill_complete(&self, selection_id: &str) -> Result<(), SelectionError> {
        let Some((entry, booking)) = self.indexed_booking(selection_id) else {
            return Err(Self::reservation_not_found(selection_id));
        };
        match entry
            .scheduler
            .mark_prefill_completed_if_booking(&booking)
            .await?
        {
            LifecycleMutationOutcome::Applied => Ok(()),
            // Already marked: still publish the ordered completion so peers
            // that missed the first event converge.
            LifecycleMutationOutcome::NoChange
                if entry
                    .scheduler
                    .publish_prefill_completed_if_booking(&booking) =>
            {
                Ok(())
            }
            LifecycleMutationOutcome::NoChange => {
                forget_reservation_if(&self.reservation_index, &entry.key, &booking);
                Err(Self::reservation_not_found(selection_id))
            }
        }
    }

    pub async fn free_reservation(&self, selection_id: &str) -> Result<(), SelectionError> {
        let Some((entry, booking)) = self.indexed_booking(selection_id) else {
            return Err(Self::reservation_not_found(selection_id));
        };
        let outcome = entry.scheduler.free_if_booking(&booking).await?;
        forget_reservation_if(&self.reservation_index, &entry.key, &booking);
        match outcome {
            LifecycleMutationOutcome::Applied => Ok(()),
            LifecycleMutationOutcome::NoChange => Err(Self::reservation_not_found(selection_id)),
        }
    }

    pub fn add_output_block(
        &self,
        selection_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SelectionError> {
        if let Some(frac) = decay_fraction
            && !(0.0..=1.0).contains(&frac)
        {
            return Err(SelectionError::BadRequest(
                "decay_fraction must be between 0.0 and 1.0".to_string(),
            ));
        }

        let Some((entry, booking)) = self.indexed_booking(selection_id) else {
            return Err(Self::reservation_not_found(selection_id));
        };
        match entry
            .scheduler
            .add_output_block_if_booking_sync(&booking, decay_fraction)?
        {
            LifecycleMutationOutcome::Applied => Ok(()),
            LifecycleMutationOutcome::NoChange => {
                forget_reservation_if(&self.reservation_index, &entry.key, &booking);
                Err(Self::reservation_not_found(selection_id))
            }
        }
    }

    pub fn loads(
        &self,
        model_name: Option<&str>,
        routing_group: Option<&str>,
    ) -> Vec<ModelLoadResponse> {
        let entries = self.initialized_entries();
        let mut loads = Vec::new();
        for entry in entries {
            if model_name.is_some_and(|model_name| entry.key.model_name != model_name)
                || routing_group
                    .is_some_and(|routing_group| entry.key.routing_group != routing_group)
            {
                continue;
            }
            loads.push(ModelLoadResponse {
                model_name: entry.key.model_name.clone(),
                routing_group: entry.key.routing_group.clone(),
                loads: entry
                    .scheduler
                    .get_potential_loads(None, 0, HashMap::new(), false),
                pending_count: entry.scheduler.pending_count(),
                pending_isl_tokens: entry.scheduler.pending_isl_tokens(),
            });
        }
        loads.sort_by(|a, b| {
            (&a.model_name, &a.routing_group).cmp(&(&b.model_name, &b.routing_group))
        });
        loads
    }

    pub async fn potential_loads(
        &self,
        req: PotentialLoadsRequest,
    ) -> Result<Vec<PotentialLoad>, SelectionError> {
        let key = RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone());
        let entry = self.ready_entry(&key)?;
        let prepared = self
            .prepare_selection_inputs(
                &entry,
                &req.prompt,
                self.kv_router_config
                    .assume_kv_reuse(req.router_config_override.as_ref()),
                false,
                false,
            )
            .await?;
        let track_prefill_tokens = req
            .router_config_override
            .as_ref()
            .and_then(|cfg| cfg.track_prefill_tokens)
            .unwrap_or(self.kv_router_config.router_track_prefill_tokens);
        Ok(entry.scheduler.get_potential_loads(
            Some(prepared.sequence_hashes),
            prepared.isl_tokens,
            prepared.overlap.effective_cached_tokens,
            track_prefill_tokens,
        ))
    }

    pub async fn overlap_scores(
        &self,
        req: OverlapScoresRequest,
    ) -> Result<OverlapScoresResponse, SelectionError> {
        let key = RoutingPartitionId::new(req.model_name.clone(), req.routing_group.clone());
        let entry = self.ready_entry(&key)?;
        let block_hashes = req
            .prompt
            .block_hashes_for_indexer(entry.block_size, entry.is_eagle)?;
        let num_blocks = block_hashes.len();
        let tiered = entry
            .indexer
            .find_tiered_matches(block_hashes)
            .await
            .map_err(|error| SelectionError::Internal(error.to_string()))?;
        let schedulable_workers = self.schedulable_worker_ranks(&key);
        Ok(
            OverlapAnalysis::new(&self.kv_router_config, entry.block_size, &tiered)
                .scores_response(
                    req.router_config_override.as_ref(),
                    num_blocks,
                    schedulable_workers,
                    false,
                    None,
                    None,
                ),
        )
    }

    /// Normalize the prompt and gather cache signals. The indexer lookup and
    /// the optional shared-cache lookup run concurrently; the shared cache is
    /// consulted only when `query_shared_cache` is set, a shared cache is
    /// attached, and the prompt carries raw `token_ids`.
    async fn prepare_selection_inputs(
        &self,
        entry: &SelectionEntry,
        prompt: &PromptRequest,
        assume_kv_reuse: bool,
        query_shared_cache: bool,
        retain_kv_transfer_chain: bool,
    ) -> Result<PreparedSelectionInputs, SelectionError> {
        let normalized = prompt.normalize_for_selection(
            entry.is_eagle,
            TrackingHashInput {
                context: &self.tracking_hash,
                scope: tracking_scope(entry),
                assume_kv_reuse,
            },
        )?;
        let indexer_lookup = async {
            if normalized.block_hashes.is_empty() {
                Ok(TieredMatchDetails::default())
            } else {
                entry
                    .indexer
                    .find_tiered_matches_with_options(
                        normalized.block_hashes.clone(),
                        LowerTierQueryOptions {
                            retain_kv_transfer_chain,
                        },
                    )
                    .await
                    .map_err(|error| match error {
                        KvRouterError::IndexerOffline => {
                            SelectionError::NotReady(error.to_string())
                        }
                        other => SelectionError::Internal(other.to_string()),
                    })
            }
        };
        let shared_cache = query_shared_cache
            .then_some(self.host.cache.shared.as_deref())
            .flatten()
            .zip(prompt.token_ids.as_deref());
        let shared_cache_lookup = async {
            let (shared_cache, tokens) = shared_cache?;
            match shared_cache
                .check_blocks(tokens, entry.block_size, prompt.cache_namespace.as_deref())
                .await
            {
                Ok(hits) => Some(hits),
                Err(error) => {
                    tracing::warn!(%error, "Shared cache query failed, ignoring");
                    None
                }
            }
        };
        let (tiered, shared_cache_hits) = tokio::join!(indexer_lookup, shared_cache_lookup);
        let tiered = tiered?;
        let overlap =
            OverlapAnalysis::new(&self.kv_router_config, entry.block_size, &tiered).signals();
        let kv_transfer_candidates = retain_kv_transfer_chain
            .then(|| tiered.kv_transfer_candidates().cloned())
            .flatten();
        drop(tiered);
        Ok(PreparedSelectionInputs {
            block_hashes: normalized.block_hashes,
            sequence_hashes: normalized.sequence_hashes,
            isl_tokens: normalized.isl_tokens,
            overlap,
            shared_cache_hits,
            kv_transfer_candidates,
        })
    }

    fn schedulable_worker_ranks(&self, key: &RoutingPartitionId) -> Vec<WorkerWithDpRank> {
        let configs = self.catalog.scheduler_configs_for_key(key);
        let mut workers = Vec::new();
        for (worker_id, config) in configs {
            let start = config.data_parallel_start_rank;
            let end = start.saturating_add(config.data_parallel_size);
            for dp_rank in start..end {
                workers.push(WorkerWithDpRank::new(worker_id, dp_rank));
            }
        }
        workers
    }
}

/// Drop index entries whose booking no longer exists in its partition
/// scheduler (expired by the periodic force-expiry, or freed through a path
/// that bypassed this core). Claims are left for their owner. Returns the
/// number of entries removed.
fn sweep_reservation_index(entries: &SelectionEntries, index: &ReservationIndex) -> usize {
    let entries = entries.read();
    let mut index = index.write();
    let before = index.len();
    index.retain(|_, reservation| {
        let Some(booking) = &reservation.booking else {
            return true;
        };
        entries
            .get(&reservation.partition)
            .and_then(|cell| cell.get())
            .is_some_and(|entry| entry.scheduler.has_booking(booking))
    });
    before - index.len()
}

fn missing_booking() -> SelectionError {
    SelectionError::Internal("booking lease holds no booking".to_string())
}

fn spawn_reservation_index_sweep(
    entries: Arc<SelectionEntries>,
    index: Arc<ReservationIndex>,
    cancel_token: CancellationToken,
) {
    let period = crate::sequences::active_request_expiry_duration();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(period);
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        interval.tick().await;
        loop {
            tokio::select! {
                _ = cancel_token.cancelled() => break,
                _ = interval.tick() => {
                    let removed = sweep_reservation_index(&entries, &index);
                    if removed > 0 {
                        tracing::debug!(removed, "Swept stale selection reservation index entries");
                    }
                }
            }
        }
    });
}

/// Pick the best router-hint source for `target`: a same-role worker (or
/// cache owner) holding a longer root-aligned prefix than the target's own
/// `target_cached_prefix_blocks`, with a non-empty control endpoint. Mirrors the
/// frontend `KvRouter::router_hint_for_selection`.
fn transfer_hint_for_selection(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    target: WorkerWithDpRank,
    target_cached_prefix_blocks: u32,
    candidates: Option<&KvTransferCandidates>,
) -> Option<KvSourceLocationsPayload> {
    let candidates = candidates?;
    let target_config = configs.get(&target.worker_id)?;
    let target_metadata = target_config.kv_hint_transfer_metadata_for_dp_rank(target.dp_rank)?;

    let prefix_blocks_to_beat = usize::try_from(target_cached_prefix_blocks).unwrap_or(usize::MAX);
    let (source, block_hashes) =
        candidates.best_source(prefix_blocks_to_beat, |source| match source {
            KvTransferCandidateSource::Worker(worker) => {
                worker != target
                    && configs.get(&worker.worker_id).is_some_and(|config| {
                        config
                            .kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)
                            .is_some_and(|source_metadata| {
                                source_metadata.worker_type == target_metadata.worker_type
                                    && source_metadata
                                        .source_control_endpoint
                                        .is_some_and(|endpoint| !endpoint.is_empty())
                            })
                    })
            }
            KvTransferCandidateSource::CacheOwner(owner) => candidates
                .routing_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.router_hint_source(owner))
                .is_some_and(|source| {
                    source.attached_worker != Some(target)
                        && source.metadata.worker_type == target_metadata.worker_type
                        && !source.metadata.source_control_endpoint.is_empty()
                }),
        })?;
    let source_control_endpoint = match source {
        KvTransferCandidateSource::Worker(worker) => configs
            .get(&worker.worker_id)?
            .kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)?
            .source_control_endpoint?
            .to_string(),
        KvTransferCandidateSource::CacheOwner(owner) => candidates
            .routing_snapshot
            .as_ref()?
            .router_hint_source(owner)?
            .metadata
            .source_control_endpoint
            .clone(),
    };
    if block_hashes.is_empty() {
        return None;
    }
    Some(KvSourceLocationsPayload {
        source_control_endpoint,
        block_hashes,
    })
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

#[cfg(test)]
mod tests {
    use super::super::affinity::SessionAffinityMode;
    use super::*;
    use crate::protocols::StorageTier;
    use crate::services::indexer::backend::test_util::store_event;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::sleep;
    use std::time::Duration;

    fn test_config(use_kv_events: bool) -> crate::config::KvRouterConfig {
        crate::config::KvRouterConfig {
            use_kv_events,
            router_queue_threshold: None,
            ..Default::default()
        }
    }

    fn worker(worker_id: WorkerId) -> WorkerRequest {
        WorkerRequest {
            worker_id,
            model_name: "model".to_string(),
            routing_group: "default".to_string(),
            endpoint: Some(format!("http://worker-{worker_id}:8000")),
            kv_events_endpoint: None,
            kv_events_endpoints: HashMap::new(),
            replay_endpoint: None,
            block_size: Some(4),
            data_parallel_start_rank: None,
            data_parallel_size: None,
            max_num_batched_tokens: Some(1024),
            total_kv_blocks: None,
            stable_routing_id: None,
            is_eagle: None,
            taints: HashSet::new(),
            topology_domains: HashMap::new(),
            kv_transfer_domain: None,
            kv_transfer_enforcement: None,
            kv_transfer_preferred_weight: None,
            router_hint_worker_type: None,
            router_hint_source_control_endpoints: HashMap::new(),
        }
    }

    fn worker_with_kv_events(worker_id: WorkerId) -> WorkerRequest {
        WorkerRequest {
            kv_events_endpoint: Some("tcp://127.0.0.1:5557".to_string()),
            ..worker(worker_id)
        }
    }

    fn prompt() -> PromptRequest {
        PromptRequest {
            token_ids: Some(vec![1, 2, 3, 4]),
            mm_routing_info: None,
            block_mm_infos: None,
            block_hashes: None,
            sequence_hashes: None,
            isl_tokens: None,
            lora_name: None,
            cache_namespace: None,
            is_eagle: None,
        }
    }

    fn select_request() -> SelectRequest {
        SelectRequest {
            model_name: "model".to_string(),
            routing_group: "default".to_string(),
            selection_id: None,
            prompt: prompt(),
            router_config_override: None,
            expected_output_tokens: None,
            priority_jump: None,
            strict_priority: None,
            session_id: None,
            session_context: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            advisory: false,
        }
    }

    fn reserve_request(selection_id: &str) -> SelectAndReserveRequest {
        SelectAndReserveRequest {
            model_name: "model".to_string(),
            routing_group: "default".to_string(),
            selection_id: Some(selection_id.to_string()),
            prompt: prompt(),
            router_config_override: None,
            expected_output_tokens: None,
            priority_jump: None,
            strict_priority: None,
            session_id: None,
            session_context: None,
            affinity_target: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
        }
    }

    async fn wait_until(what: &str, mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !condition() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("timed out waiting for {what}"));
    }

    async fn wait_for_pending_selection(core: &SelectionCore) {
        wait_until("pending selection", || {
            core.loads(Some("model"), Some("default"))[0].pending_count == 1
        })
        .await;
    }

    fn assert_shutdown_error(error: SelectionError) {
        assert!(matches!(
            error,
            SelectionError::NotReady(message)
                if message == "selection service is shutting down"
        ));
    }

    #[test]
    fn parent_cancel_cancels_core() {
        let parent = CancellationToken::new();
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            parent.clone(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");

        assert!(!core.cancel_token.is_cancelled());
        parent.cancel();
        assert!(core.cancel_token.is_cancelled());
    }

    #[tokio::test]
    async fn selection_setup_uses_worker_type_label() {
        for (worker_type, expected_label) in [
            (WorkerType::Prefill, "prefill"),
            (WorkerType::Decode, "decode"),
            (WorkerType::Encode, "encode"),
            (WorkerType::Aggregated, "aggregated"),
        ] {
            let config = test_config(false);
            let tracking_hash = Arc::new(
                TrackingHashContext::from_config(&config)
                    .expect("valid tracking hash configuration"),
            );
            let core = SelectionCore::new_inner(
                config,
                1,
                CancellationToken::new(),
                None,
                None,
                SelectionHost::default(),
                worker_type,
                true,
                SelectionCacheConfig::default(),
                tracking_hash,
                IndexerPolicy::from_router_config(&test_config(false)).expect("indexer policy"),
                None,
            );

            core.upsert_worker(worker(1)).await.expect("worker upsert");
            let entry = core
                .entry(&RoutingPartitionId::new("model", "default"))
                .expect("selection entry");
            assert_eq!(
                entry.scheduler.worker_type(),
                expected_label,
                "{worker_type}"
            );
        }
    }

    fn core_with_host(host: SelectionHost) -> SelectionCore {
        core_with_host_and_policy(host, None)
    }

    fn core_with_host_and_policy(
        host: SelectionHost,
        policy_factory: Option<WorkerSelectionPolicyFactory>,
    ) -> SelectionCore {
        let config = test_config(false);
        let tracking_hash = Arc::new(
            TrackingHashContext::from_config(&config).expect("valid tracking hash configuration"),
        );
        let indexer_policy = IndexerPolicy::from_router_config(&config).expect("indexer policy");
        SelectionCore::new_inner(
            config,
            1,
            CancellationToken::new(),
            None,
            policy_factory,
            host,
            WorkerType::Aggregated,
            true,
            SelectionCacheConfig::default(),
            tracking_hash,
            indexer_policy,
            None,
        )
    }

    async fn wait_for_overlap(
        core: &SelectionCore,
        request: impl Fn() -> SelectRequest,
    ) -> SelectResponse {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let response = core.select(request()).await.expect("select");
                if response.overlap.longest_matched > 0 {
                    return response;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("approximate indexer never credited the booked prompt")
    }

    #[tokio::test]
    async fn bookings_populate_the_approximate_primary_without_kv_events() {
        // use_kv_events=false: the primary is approximate and bookings feed it.
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");

        // Query-only selection records nothing.
        let first = core.select(select_request()).await.expect("select");
        assert_eq!(first.overlap.longest_matched, 0);

        // select_and_reserve records the routed prefix for the chosen worker.
        let booked = core
            .select_and_reserve(reserve_request("booked"))
            .await
            .expect("reserve");
        let credited = wait_for_overlap(&core, || {
            let mut request = select_request();
            request.allowed_worker_ids = Some(HashSet::from([booked.worker_id]));
            request
        })
        .await;
        assert_eq!(credited.worker_id, booked.worker_id);
        assert_eq!(credited.overlap.longest_matched, 4);

        // The cached replay path records too, on a different prompt.
        let prompt_b = || PromptRequest {
            token_ids: Some(vec![5, 6, 7, 8]),
            ..PromptRequest::default()
        };
        let mut request = select_request();
        request.prompt = prompt_b();
        request.selection_id = Some("cached".to_string());
        request.allowed_worker_ids = Some(HashSet::from([2]));
        core.select(request).await.expect("select");
        core.create_reservation(ReservationRequest {
            model_name: "model".to_string(),
            routing_group: "default".to_string(),
            selection_id: "cached".to_string(),
            worker_id: None,
            dp_rank: None,
            prompt: PromptRequest::default(),
            router_config_override: None,
            expected_output_tokens: None,
            effective_prefill_tokens: None,
            track_prefill_tokens: None,
        })
        .await
        .expect("cached reservation");
        let credited = wait_for_overlap(&core, || {
            let mut request = select_request();
            request.prompt = prompt_b();
            request.allowed_worker_ids = Some(HashSet::from([2]));
            request
        })
        .await;
        assert_eq!(credited.worker_id, 2);

        // And the explicit reservation form.
        let prompt_c = || PromptRequest {
            token_ids: Some(vec![9, 10, 11, 12]),
            ..PromptRequest::default()
        };
        core.create_reservation(ReservationRequest {
            model_name: "model".to_string(),
            routing_group: "default".to_string(),
            selection_id: "explicit".to_string(),
            worker_id: Some(1),
            dp_rank: None,
            prompt: prompt_c(),
            router_config_override: None,
            expected_output_tokens: None,
            effective_prefill_tokens: None,
            track_prefill_tokens: None,
        })
        .await
        .expect("explicit reservation");
        let credited = wait_for_overlap(&core, || {
            let mut request = select_request();
            request.prompt = prompt_c();
            request.allowed_worker_ids = Some(HashSet::from([1]));
            request
        })
        .await;
        assert_eq!(credited.worker_id, 1);
    }

    #[tokio::test]
    async fn unreachable_remote_indexer_is_reported_not_ready() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let base_url = format!("http://{}", listener.local_addr().expect("addr"));
        drop(listener);
        let config = test_config(true);
        let tracking_hash = Arc::new(
            TrackingHashContext::from_config(&config).expect("valid tracking hash configuration"),
        );
        let indexer_policy = IndexerPolicy::from_router_config(&config)
            .expect("indexer policy")
            .with_remote_indexer(base_url)
            .expect("remote policy");
        let core = SelectionCore::new_inner(
            config,
            1,
            CancellationToken::new(),
            None,
            None,
            SelectionHost::default(),
            WorkerType::Aggregated,
            true,
            SelectionCacheConfig::default(),
            tracking_hash,
            indexer_policy,
            None,
        );
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        assert!(matches!(
            core.select(select_request()).await,
            Err(SelectionError::NotReady(_))
        ));
    }

    #[tokio::test]
    async fn remote_indexer_serves_selection_without_local_kv_listeners() {
        use crate::indexer::KvIndexerInterface;
        use crate::protocols::{BlockHashOptions, StorageTier, compute_block_hash_for_seq};
        use crate::services::indexer::registry::WorkerRegistry;
        use crate::services::indexer::server::spawn_test_indexer_server;

        // The standalone indexer holds worker 2's cache for the test prompt.
        let key = RoutingPartitionId::new("model", "default");
        let served = Arc::new(WorkerRegistry::new(1));
        let served_indexer = served.get_or_create_indexer(key.clone(), 4);
        let hashes: Vec<u64> =
            compute_block_hash_for_seq(&[1, 2, 3, 4], 4, BlockHashOptions::default())
                .into_iter()
                .map(|hash| hash.0)
                .collect();
        served_indexer
            .apply_event_routed(store_event(2, 0, 1, &[], &hashes, StorageTier::Device))
            .await
            .unwrap();
        if let Indexer::Single { primary, .. } = &served_indexer {
            let _ = primary.flush().await;
        }
        let (base_url, server) = spawn_test_indexer_server(served).await;

        // use_kv_events=true, but the primary is remote: workers need no
        // kv_events endpoints and no ZMQ listener is started here.
        let config = test_config(true);
        let tracking_hash = Arc::new(
            TrackingHashContext::from_config(&config).expect("valid tracking hash configuration"),
        );
        let indexer_policy = IndexerPolicy::from_router_config(&config)
            .expect("indexer policy")
            .with_remote_indexer(base_url)
            .expect("remote policy");
        let core = SelectionCore::new_inner(
            config,
            1,
            CancellationToken::new(),
            None,
            None,
            SelectionHost::default(),
            WorkerType::Aggregated,
            true,
            SelectionCacheConfig::default(),
            tracking_hash,
            indexer_policy,
            None,
        );
        assert!(!core.listens_for_kv_events);
        for worker_id in [1, 2] {
            let record = core
                .upsert_worker(worker(worker_id))
                .await
                .expect("worker upsert");
            assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable, "{record:?}");
        }

        let response = core.select(select_request()).await.expect("select");
        assert_eq!(
            response.worker_id, 2,
            "remote cache credit steers selection"
        );
        assert_eq!(response.overlap.longest_matched, 4);

        core.delete_worker(2).await.expect("delete worker");
        server.abort();
    }

    #[tokio::test]
    async fn booked_selection_attaches_router_hint_from_a_better_source() {
        use crate::indexer::KvIndexerInterface;
        use crate::protocols::{BlockHashOptions, StorageTier, compute_block_hash_for_seq};

        let core = SelectionCore::try_new_local(
            test_config(true),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        for worker_id in [1, 2] {
            let mut request = worker_with_kv_events(worker_id);
            request.router_hint_worker_type = Some("decode".to_string());
            request.router_hint_source_control_endpoints =
                HashMap::from([(0, format!("tcp://worker-{worker_id}:9000"))]);
            core.upsert_worker(request).await.expect("worker upsert");
        }
        let key = RoutingPartitionId::new("model", "default");
        let entry = core.entry(&key).expect("entry");
        assert!(entry.indexer.supports_kv_transfer_chain_retention());

        // Worker 1 holds both blocks of the prompt; worker 2 holds nothing.
        let tokens: Vec<u32> = (1..=8).collect();
        let hashes: Vec<u64> = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())
            .into_iter()
            .map(|hash| hash.0)
            .collect();
        assert_eq!(hashes.len(), 2);
        entry
            .indexer
            .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
            .await
            .unwrap();
        if let Indexer::Single { primary, .. } = &entry.indexer {
            let _ = primary.flush().await;
        }
        let prompt = || PromptRequest {
            token_ids: Some(tokens.clone()),
            ..PromptRequest::default()
        };

        // Booking on worker 2: worker 1 is a same-role source with a longer prefix.
        let mut request = reserve_request("to-worker-2");
        request.prompt = prompt();
        request.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
        let response = core.select_and_reserve(request).await.expect("reserve");
        let hint = response.kv_hint.expect("router hint for worker 2");
        assert_eq!(hint.message_id, "to-worker-2");
        assert_eq!(hint.actions[0].action_type, "kv.fetch");
        let payload: KvSourceLocationsPayload =
            serde_json::from_value(serde_json::to_value(&hint.actions[0].payload).unwrap())
                .unwrap();
        assert_eq!(payload.source_control_endpoint, "tcp://worker-1:9000");
        assert_eq!(payload.block_hashes.len(), 2);

        // Booking on worker 1 itself: nothing holds a longer prefix.
        let mut request = reserve_request("to-worker-1");
        request.prompt = prompt();
        request.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        let response = core.select_and_reserve(request).await.expect("reserve");
        assert!(response.kv_hint.is_none());

        // Query-only selections never carry a hint.
        let mut request = select_request();
        request.prompt = prompt();
        request.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
        let response = core.select(request).await.expect("select");
        assert!(response.kv_hint.is_none());
    }

    #[tokio::test]
    async fn router_hint_needs_capable_workers() {
        use crate::indexer::KvIndexerInterface;
        use crate::protocols::{BlockHashOptions, StorageTier, compute_block_hash_for_seq};

        let core = SelectionCore::try_new_local(
            test_config(true),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        for worker_id in [1, 2] {
            core.upsert_worker(worker_with_kv_events(worker_id))
                .await
                .expect("worker upsert");
        }
        let entry = core
            .entry(&RoutingPartitionId::new("model", "default"))
            .expect("entry");
        let hashes: Vec<u64> =
            compute_block_hash_for_seq(&[1, 2, 3, 4], 4, BlockHashOptions::default())
                .into_iter()
                .map(|hash| hash.0)
                .collect();
        entry
            .indexer
            .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
            .await
            .unwrap();
        if let Indexer::Single { primary, .. } = &entry.indexer {
            let _ = primary.flush().await;
        }
        let mut request = reserve_request("plain");
        request.pinned_worker = Some(WorkerWithDpRank::new(2, 0));
        let response = core.select_and_reserve(request).await.expect("reserve");
        assert!(response.kv_hint.is_none());
    }

    #[tokio::test]
    async fn event_driven_indexer_does_not_record_bookings() {
        let core = SelectionCore::try_new_local(
            test_config(true),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker_with_kv_events(1))
            .await
            .expect("worker upsert");
        core.select_and_reserve(reserve_request("booked"))
            .await
            .expect("reserve");
        core.free_reservation("booked").await.expect("free");
        for _ in 0..3 {
            let response = core.select(select_request()).await.expect("select");
            assert_eq!(response.overlap.longest_matched, 0);
            tokio::task::yield_now().await;
        }
    }

    /// Picker that records what worker selection saw and always takes row 0.
    struct CapturingPicker {
        observed: Arc<parking_lot::Mutex<Vec<SelectionObservation>>>,
    }

    #[derive(Debug, Clone)]
    struct SelectionObservation {
        session_context: Option<SessionContext>,
        shared_beyond_device_blocks: Vec<u32>,
    }

    impl crate::scheduling::selector::WorkerPicker for CapturingPicker {
        fn required_worker_inputs(&self) -> crate::scheduling::selector::WorkerInputs {
            crate::scheduling::selector::WorkerInputs::CACHE
        }

        fn pick(
            &mut self,
            context: &crate::scheduling::selector::WorkerSelectionContext<'_>,
            input: crate::scheduling::selector::WorkerInputView<'_>,
        ) -> Result<usize, crate::scheduling::WorkerSelectionPolicyError> {
            self.observed.lock().push(SelectionObservation {
                session_context: context.session_context().cloned(),
                shared_beyond_device_blocks: input
                    .cache()
                    .expect("CACHE inputs requested")
                    .iter()
                    .map(|cache| cache.shared_beyond_device_blocks())
                    .collect(),
            });
            Ok(0)
        }
    }

    fn capturing_policy_factory() -> (
        WorkerSelectionPolicyFactory,
        Arc<parking_lot::Mutex<Vec<SelectionObservation>>>,
    ) {
        let observed = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let factory_observed = Arc::clone(&observed);
        let factory: WorkerSelectionPolicyFactory =
            Arc::new(move |config, worker_type, _partition| {
                WorkerSelectionPolicy::new(
                    config.clone(),
                    worker_type.as_str(),
                    Vec::new(),
                    Box::new(CapturingPicker {
                        observed: Arc::clone(&factory_observed),
                    }),
                )
            });
        (factory, observed)
    }

    type SharedCacheCalls = Arc<parking_lot::Mutex<Vec<(Vec<u32>, u32, Option<String>)>>>;

    /// Shared cache that reports every block as a hit and records each query.
    struct RecordingSharedCache {
        calls: SharedCacheCalls,
    }

    #[async_trait::async_trait]
    impl SharedKvCache for RecordingSharedCache {
        async fn check_blocks(
            &self,
            tokens: &[u32],
            block_size: u32,
            cache_namespace: Option<&str>,
        ) -> Result<SharedCacheHits, crate::indexer::KvRouterError> {
            self.calls.lock().push((
                tokens.to_vec(),
                block_size,
                cache_namespace.map(str::to_string),
            ));
            let blocks = (tokens.len() / block_size as usize) as u32;
            Ok(SharedCacheHits::from_hits(&vec![true; blocks as usize]))
        }
    }

    struct OnlyWorkerForLora {
        worker_id: WorkerId,
    }

    impl LoraWorkerFilter for OnlyWorkerForLora {
        fn filter_worker_ids_for_lora(
            &self,
            _lora_name: &str,
            available: &[WorkerId],
        ) -> Vec<WorkerId> {
            available
                .iter()
                .copied()
                .filter(|id| *id == self.worker_id)
                .collect()
        }
    }

    #[tokio::test]
    async fn injected_lora_filter_narrows_candidates() {
        let core = core_with_host(SelectionHost {
            eligibility: HostEligibility {
                lora_worker_filter: Some(Arc::new(OnlyWorkerForLora { worker_id: 2 })),
            },
            ..SelectionHost::default()
        });
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");

        // LoRA request: only the filter's worker is eligible.
        for _ in 0..4 {
            let mut request = select_request();
            request.prompt.lora_name = Some("adapter-a".to_string());
            let response = core.select(request).await.expect("select");
            assert_eq!(response.worker_id, 2);
        }

        // The filter never widens the caller's allow-set: an allow-set that
        // excludes the filter's worker is preserved as-is.
        let mut request = select_request();
        request.prompt.lora_name = Some("adapter-a".to_string());
        request.allowed_worker_ids = Some(HashSet::from([1]));
        let response = core.select(request).await.expect("select");
        assert_eq!(response.worker_id, 1);

        // A pinned worker inside the universe survives the filter.
        let mut request = select_request();
        request.prompt.lora_name = Some("adapter-a".to_string());
        request.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        let response = core.select(request).await.expect("select");
        assert_eq!(response.worker_id, 1);
    }

    #[tokio::test]
    async fn shared_cache_hits_reach_worker_selection() {
        let calls: SharedCacheCalls = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let (factory, observed) = capturing_policy_factory();
        let core = core_with_host_and_policy(
            SelectionHost {
                cache: HostCache {
                    shared: Some(Arc::new(RecordingSharedCache {
                        calls: Arc::clone(&calls),
                    })),
                    ..HostCache::default()
                },
                ..SelectionHost::default()
            },
            Some(factory),
        );
        core.upsert_worker(worker(1)).await.expect("worker upsert");

        let mut request = select_request();
        request.prompt.cache_namespace = Some("tenant-a".to_string());
        core.select(request).await.expect("select");

        assert_eq!(
            calls.lock().as_slice(),
            &[(vec![1, 2, 3, 4], 4, Some("tenant-a".to_string()))]
        );
        let observations = observed.lock().clone();
        assert_eq!(observations.len(), 1);
        // One block in the prompt, no device overlap, so the whole prompt is a
        // shared-cache hit beyond the device prefix.
        assert_eq!(observations[0].shared_beyond_device_blocks, vec![1]);

        // Load projection does not consult the shared cache.
        core.potential_loads(PotentialLoadsRequest {
            model_name: "model".to_string(),
            routing_group: "default".to_string(),
            prompt: prompt(),
            router_config_override: None,
        })
        .await
        .expect("potential loads");
        assert_eq!(calls.lock().len(), 1);

        // Prompts without raw tokens cannot be checked against the shared cache.
        let mut request = select_request();
        request.prompt = PromptRequest {
            token_ids: None,
            block_hashes: Some(vec![11]),
            sequence_hashes: Some(vec![101]),
            isl_tokens: Some(4),
            ..PromptRequest::default()
        };
        core.select(request).await.expect("select");
        assert_eq!(calls.lock().len(), 1);
        assert_eq!(observed.lock()[1].shared_beyond_device_blocks, vec![0]);
    }

    #[tokio::test]
    async fn session_context_reaches_worker_selection() {
        use super::super::types::SelectionSessionContext;

        let (factory, observed) = capturing_policy_factory();
        let core = core_with_host_and_policy(SelectionHost::default(), Some(factory));
        core.upsert_worker(worker(1)).await.expect("worker upsert");

        let mut request = select_request();
        request.session_id = Some("ignored-legacy".to_string());
        request.session_context = Some(SelectionSessionContext {
            session_id: "child-session".to_string(),
            parent_session_id: Some("root-session".to_string()),
            session_final: Some(true),
            input_trigger: Some(super::super::types::SelectionInputTrigger::ToolResult),
        });
        core.select(request).await.expect("select");

        let mut request = reserve_request("legacy-session-reservation");
        request.session_id = Some("legacy-only".to_string());
        core.select_and_reserve(request)
            .await
            .expect("select and reserve");

        let observations = observed.lock();
        let context = observations[0]
            .session_context
            .as_ref()
            .expect("structured session context");
        assert_eq!(context.session_id(), "child-session");
        assert_eq!(context.parent_session_id(), Some("root-session"));
        assert_eq!(context.session_final(), Some(true));
        assert_eq!(
            context.input_trigger(),
            Some(crate::scheduling::WorkerSelectionInputTrigger::ToolResult)
        );

        let legacy = observations[1]
            .session_context
            .as_ref()
            .expect("legacy session context");
        assert_eq!(legacy.session_id(), "legacy-only");
        assert_eq!(legacy.parent_session_id(), None);
    }

    #[tokio::test]
    async fn full_affinity_table_routes_without_pinning() {
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let entry = core
            .entry(&RoutingPartitionId::new("model", "default"))
            .expect("entry");
        let table = SessionAffinity::new_with_limits(Duration::from_secs(60), 1, 256)
            .expect("affinity table");
        assert!(entry.affinity.set(table).is_ok());

        for (selection_id, session_id) in [("first", "s1"), ("second", "s2")] {
            let mut request = reserve_request(selection_id);
            request.session_id = Some(session_id.to_string());
            core.select_and_reserve(request)
                .await
                .expect("a full affinity table must not fail selection");
        }
        assert_eq!(
            core.reservation_index
                .read()
                .values()
                .filter(|r| r._affinity_lease.is_some())
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn injected_availability_provider_restricts_selection() {
        let available: Arc<parking_lot::Mutex<Option<Arc<HashSet<WorkerId>>>>> =
            Arc::new(parking_lot::Mutex::new(None));
        let provider_state = Arc::clone(&available);
        let core = core_with_host(SelectionHost {
            load: HostLoad {
                available_workers: Some(Arc::new(move || provider_state.lock().clone())),
                ..HostLoad::default()
            },
            ..SelectionHost::default()
        });
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");

        for only in [1, 2] {
            *available.lock() = Some(Arc::new(HashSet::from([only])));
            for _ in 0..4 {
                let response = core.select(select_request()).await.expect("select");
                assert_eq!(response.worker_id, only);
            }
        }
    }

    #[tokio::test]
    async fn injected_overload_provider_excludes_worker() {
        let core = core_with_host(SelectionHost {
            load: HostLoad {
                overloaded_workers: Some(Arc::new(|| Some(HashSet::from([1])))),
                ..HostLoad::default()
            },
            ..SelectionHost::default()
        });
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");

        for _ in 0..4 {
            let response = core.select(select_request()).await.expect("select");
            assert_eq!(response.worker_id, 2);
        }
    }

    #[test]
    fn shutdown_keeps_parent_alive() {
        let parent = CancellationToken::new();
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            parent.clone(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");

        core.shutdown();

        assert!(core.cancel_token.is_cancelled());
        assert!(!parent.is_cancelled());
    }

    #[tokio::test]
    async fn multi_rank_worker_with_replay_endpoint_is_incomplete() {
        let core = SelectionCore::try_new_local(
            test_config(true),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");

        let record = core
            .upsert_worker(WorkerRequest {
                data_parallel_size: Some(2),
                kv_events_endpoints: HashMap::from([
                    (0, "tcp://127.0.0.1:5557".to_string()),
                    (1, "tcp://127.0.0.1:5558".to_string()),
                ]),
                replay_endpoint: Some("tcp://127.0.0.1:5600".to_string()),
                ..worker(1)
            })
            .await
            .expect("worker upsert");
        assert_eq!(record.lifecycle, WorkerLifecycle::Incomplete, "{record:?}");
        assert!(!core.indexer_registry.has_worker(1));
    }

    #[tokio::test]
    async fn reupsert_recreates_a_listener_lost_to_a_cancelled_update() {
        use crate::indexer::KvIndexerInterface;

        let core = SelectionCore::try_new_local(
            test_config(true),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker_with_kv_events(1))
            .await
            .expect("worker upsert");
        let entry = core
            .entry(&RoutingPartitionId::new("model", "default"))
            .expect("entry");
        let hashes = [11u64, 12];
        entry
            .indexer
            .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
            .await
            .unwrap();
        if let Indexer::Single { primary, .. } = &entry.indexer {
            let _ = primary.flush().await;
        }
        // An endpoint update cancelled right after removing the listener leaves
        // the catalog record schedulable, the listener gone, and its blocks indexed.
        core.indexer_registry.forget_listener(1, 0);

        let record = core
            .upsert_worker(worker_with_kv_events(1))
            .await
            .expect("worker re-upsert");
        assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable, "{record:?}");
        assert!(core.indexer_registry.has_listener(1, 0));
        // The purge is queued to the indexer thread; wait for it to land.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let overlap = entry
                    .indexer
                    .find_matches(hashes.iter().copied().map(LocalBlockHash).collect())
                    .await
                    .expect("find matches");
                if overlap.scores.is_empty() {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("stale blocks survived the re-upsert");
    }

    #[tokio::test]
    async fn shutdown_cancels_listeners() {
        let parent = CancellationToken::new();
        let core = SelectionCore::try_new_local(
            test_config(true),
            1,
            parent,
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");

        let record = core
            .upsert_worker(worker_with_kv_events(1))
            .await
            .expect("worker upsert");
        assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
        assert_eq!(core.indexer_registry.listener_cancelled(1, 0), Some(false));

        core.shutdown();
        assert_eq!(core.indexer_registry.listener_cancelled(1, 0), Some(true));
    }

    #[tokio::test]
    async fn upsert_moves_global_worker_id_between_routing_groups() {
        let core = SelectionCore::try_new_local(
            test_config(true),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        let mut group_a = worker_with_kv_events(1);
        group_a.routing_group = "group-a".to_string();
        core.upsert_worker(group_a).await.expect("group A upsert");
        assert_eq!(
            core.indexer_registry
                .list_filtered(Some("model"), Some("group-a"))
                .len(),
            1
        );

        let mut group_b = worker_with_kv_events(1);
        group_b.routing_group = "group-b".to_string();
        core.upsert_worker(group_b).await.expect("group B upsert");

        assert!(core.list_workers(Some("model"), Some("group-a")).is_empty());
        assert_eq!(core.list_workers(Some("model"), Some("group-b")).len(), 1);
        assert!(
            core.indexer_registry
                .list_filtered(Some("model"), Some("group-a"))
                .is_empty()
        );
        assert_eq!(
            core.indexer_registry
                .list_filtered(Some("model"), Some("group-b"))
                .len(),
            1
        );

        let mut select_a = select_request();
        select_a.routing_group = "group-a".to_string();
        assert!(matches!(
            core.select(select_a).await,
            Err(SelectionError::NotReady(_))
        ));
        let mut select_b = select_request();
        select_b.routing_group = "group-b".to_string();
        assert_eq!(core.select(select_b).await.unwrap().worker_id, 1);

        core.delete_worker(1).await.expect("delete group B worker");
        assert!(
            core.indexer_registry
                .list_filtered(Some("model"), Some("group-b"))
                .is_empty()
        );
    }

    #[tokio::test]
    async fn shutdown_reports_not_ready_and_rejects_new_work() {
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        assert!(core.ready().ready);

        core.shutdown();

        let ready = core.ready();
        assert!(!ready.ready);
        assert_eq!(ready.schedulable_workers, 1);

        let upsert_error = core
            .upsert_worker(worker(2))
            .await
            .expect_err("upsert should fail after shutdown");
        assert_shutdown_error(upsert_error);

        let patch = serde_json::from_value(serde_json::json!({
            "endpoint": "http://worker-1:9000"
        }))
        .expect("worker patch");
        let patch_error = core
            .patch_worker(1, patch)
            .await
            .expect_err("patch should fail after shutdown");
        assert_shutdown_error(patch_error);

        let select_error = core
            .select(select_request())
            .await
            .expect_err("selection should fail after shutdown");
        assert_shutdown_error(select_error);

        let reservation_error = core
            .create_reservation(ReservationRequest {
                model_name: "model".to_string(),
                routing_group: "default".to_string(),
                selection_id: "res-after-shutdown".to_string(),
                worker_id: Some(1),
                dp_rank: None,
                prompt: prompt(),
                router_config_override: None,
                expected_output_tokens: None,
                effective_prefill_tokens: None,
                track_prefill_tokens: None,
            })
            .await
            .expect_err("reservation should fail after shutdown");
        assert_shutdown_error(reservation_error);

        assert_eq!(core.list_workers(None, None).len(), 1);
        assert_eq!(core.loads(None, None).len(), 1);
        let deleted = core
            .delete_worker(1)
            .await
            .expect("delete should remain available after shutdown");
        assert_eq!(deleted.lifecycle, WorkerLifecycle::Unschedulable);
    }

    #[tokio::test]
    async fn queued_selection_errors_on_shutdown() {
        let mut config = test_config(false);
        config.router_queue_threshold = Some(0.0);
        let core = Arc::new(
            SelectionCore::try_new_local(
                config,
                1,
                CancellationToken::new(),
                SelectionCacheConfig::default(),
            )
            .expect("valid test config"),
        );

        let record = core.upsert_worker(worker(1)).await.expect("worker upsert");
        assert_eq!(record.lifecycle, WorkerLifecycle::Schedulable);
        core.select_and_reserve(reserve_request("res-a"))
            .await
            .expect("initial reservation");

        let queued_core = core.clone();
        let queued = tokio::spawn(async move { queued_core.select(select_request()).await });
        wait_for_pending_selection(&core).await;

        core.shutdown();
        let err = tokio::time::timeout(Duration::from_secs(1), queued)
            .await
            .expect("queued selection timed out")
            .expect("queued selection task panicked")
            .expect_err("queued selection should fail");

        assert!(matches!(
            err,
            SelectionError::Scheduler(KvSchedulerError::SubscriberShutdown)
        ));
    }

    #[tokio::test]
    async fn booking_is_freed_when_selected_worker_drained_while_queued() {
        let mut config = test_config(false);
        config.router_queue_threshold = Some(0.0);
        let core = Arc::new(
            SelectionCore::try_new_local(
                config,
                1,
                CancellationToken::new(),
                SelectionCacheConfig::default(),
            )
            .expect("valid test config"),
        );
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let key = RoutingPartitionId::new("model", "default");
        let entry = core.entry(&key).expect("entry");
        core.select_and_reserve(reserve_request("res-a"))
            .await
            .expect("initial reservation");

        let queued_core = core.clone();
        let queued = tokio::spawn(async move {
            queued_core
                .select_and_reserve(reserve_request("queued"))
                .await
        });
        wait_for_pending_selection(&core).await;
        // First half of `delete_worker`: the worker drains while the request waits.
        core.catalog
            .set_lifecycle(1, WorkerLifecycle::Draining, Vec::new());
        core.free_reservation("res-a").await.expect("free res-a");

        let err = tokio::time::timeout(Duration::from_secs(2), queued)
            .await
            .expect("queued selection timed out")
            .expect("task panicked")
            .expect_err("drained worker is not schedulable");
        assert!(
            matches!(&err, SelectionError::Internal(m) if m.contains("no longer schedulable")),
            "{err:?}"
        );
        wait_until("booking release", || !entry.scheduler.has_request("queued")).await;
    }

    #[tokio::test]
    async fn dropped_selection_future_frees_its_booking() {
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let key = RoutingPartitionId::new("model", "default");
        let entry = core.entry(&key).expect("entry");

        // Drive the selection by hand so the actor's response is delivered but
        // never consumed: poll until the booking exists, then drop the future.
        let mut selection = Box::pin(core.select_and_reserve(reserve_request("dropped")));
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        wait_until("scheduler booking", || {
            if entry.scheduler.has_request("dropped") {
                return true;
            }
            assert!(
                selection.as_mut().poll(&mut context).is_pending(),
                "selection completed before the booking was observed"
            );
            false
        })
        .await;
        assert!(
            core.reservation_index
                .read()
                .get("dropped")
                .is_some_and(|reservation| reservation.booking.is_none()),
            "the id is claimed while its booking is in flight"
        );
        drop(selection);

        wait_until("booking release", || {
            !entry.scheduler.has_request("dropped")
        })
        .await;
        assert!(core.reservation_index.read().get("dropped").is_none());
    }

    #[tokio::test]
    async fn same_selection_id_in_two_partitions_is_a_conflict() {
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        for (worker_id, routing_group) in [(1, "group-a"), (2, "group-b")] {
            let mut request = worker(worker_id);
            request.routing_group = routing_group.to_string();
            core.upsert_worker(request).await.expect("worker upsert");
        }
        let mut first = reserve_request("shared");
        first.routing_group = "group-a".to_string();
        core.select_and_reserve(first).await.expect("first booking");

        let mut second = reserve_request("shared");
        second.routing_group = "group-b".to_string();
        let err = core
            .select_and_reserve(second)
            .await
            .expect_err("a live id cannot be booked again");
        assert!(matches!(err, SelectionError::Conflict(_)), "{err:?}");

        let (entry, _) = core
            .indexed_booking("shared")
            .expect("first booking indexed");
        assert_eq!(entry.key.routing_group, "group-a");
        assert!(entry.scheduler.has_request("shared"));
        assert!(
            !core
                .entry(&RoutingPartitionId::new("model", "group-b"))
                .expect("entry")
                .scheduler
                .has_request("shared")
        );
    }

    #[tokio::test]
    async fn explicit_reservation_of_a_live_id_is_a_conflict() {
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.select_and_reserve(reserve_request("taken"))
            .await
            .expect("booking");
        let err = core
            .create_reservation(ReservationRequest {
                model_name: "model".to_string(),
                routing_group: "default".to_string(),
                selection_id: "taken".to_string(),
                worker_id: Some(1),
                dp_rank: None,
                prompt: prompt(),
                router_config_override: None,
                expected_output_tokens: None,
                effective_prefill_tokens: None,
                track_prefill_tokens: None,
            })
            .await
            .expect_err("explicit booking of a live id");
        assert!(matches!(err, SelectionError::Conflict(_)), "{err:?}");
    }

    #[test]
    fn index_observer_replaces_stale_and_claimed_rows_of_its_partition() {
        use crate::scheduling::AttemptId;
        let index = Arc::new(RwLock::new(HashMap::new()));
        let partition = RoutingPartitionId::new("model", "default");
        let observer = ReservationIndexObserver {
            index: Arc::clone(&index),
            partition: partition.clone(),
            host: None,
        };
        let booking = |attempt: u64| SchedulerBookingDescriptor {
            request_id: "shared".to_string(),
            worker: WorkerWithDpRank::new(1, 0),
            attempt_id: AttemptId::new(attempt),
        };
        let row = |partition: &RoutingPartitionId, booking| Reservation {
            partition: partition.clone(),
            booking,
            _affinity_lease: None,
        };

        // A stale row (its booking expired before the sweep ran) yields to the mirror.
        index
            .write()
            .insert("shared".to_string(), row(&partition, Some(booking(1))));
        observer.admitted(booking(2));
        assert_eq!(index.read()["shared"].booking, Some(booking(2)));

        // A completion for the replaced booking leaves the live mirror alone.
        observer.completed(&booking(1));
        assert!(index.read().contains_key("shared"));
        observer.completed(&booking(2));
        assert!(!index.read().contains_key("shared"));

        // A claim whose local booking is about to fail also yields, and dropping
        // that claim keeps the mirror.
        let claim = ReservationClaim {
            index: &index,
            selection_id: "shared".to_string(),
            armed: true,
        };
        index
            .write()
            .insert("shared".to_string(), row(&partition, None));
        observer.admitted(booking(3));
        drop(claim);
        assert_eq!(index.read()["shared"].booking, Some(booking(3)));

        // Another partition's row is never replaced.
        let other = RoutingPartitionId::new("model", "other");
        index
            .write()
            .insert("shared".to_string(), row(&other, Some(booking(4))));
        observer.admitted(booking(5));
        assert_eq!(index.read()["shared"].booking, Some(booking(4)));
    }

    #[tokio::test]
    async fn free_of_an_in_flight_reservation_is_not_found() {
        let mut config = test_config(false);
        config.router_queue_threshold = Some(0.0);
        let core = Arc::new(
            SelectionCore::try_new_local(
                config,
                1,
                CancellationToken::new(),
                SelectionCacheConfig::default(),
            )
            .expect("valid test config"),
        );
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.select_and_reserve(reserve_request("res-a"))
            .await
            .expect("initial reservation");
        let queued_core = core.clone();
        let queued = tokio::spawn(async move {
            queued_core
                .select_and_reserve(reserve_request("queued"))
                .await
        });
        wait_for_pending_selection(&core).await;

        // A free racing the in-flight booking neither frees nor evicts it.
        let err = core
            .free_reservation("queued")
            .await
            .expect_err("in-flight id is not a reservation yet");
        assert!(matches!(err, SelectionError::NotFound(_)), "{err:?}");
        assert!(core.reservation_index.read().contains_key("queued"));

        core.free_reservation("res-a").await.expect("free res-a");
        tokio::time::timeout(Duration::from_secs(2), queued)
            .await
            .expect("queued selection timed out")
            .expect("task panicked")
            .expect("queued selection books");
        core.free_reservation("queued").await.expect("free queued");
        assert!(core.reservation_index.read().is_empty());
    }

    #[tokio::test]
    async fn prefill_complete_is_idempotent_for_a_live_booking() {
        let mut config = test_config(false);
        config.router_track_prefill_tokens = true;
        let core = SelectionCore::try_new_local(
            config,
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.select_and_reserve(reserve_request("live"))
            .await
            .expect("booking");
        core.prefill_complete("live").await.expect("first mark");
        core.prefill_complete("live")
            .await
            .expect("a repeated mark on a live booking is not an error");
        let (entry, _) = core.indexed_booking("live").expect("still indexed");
        assert!(entry.scheduler.has_request("live"));
        core.free_reservation("live").await.expect("free");
    }

    #[tokio::test]
    async fn mirrored_replica_bookings_are_indexed_until_freed() {
        let (outbound_tx, _outbound_rx) = mpsc::channel(16);
        let (inbound_tx, inbound_rx) = mpsc::channel(16);
        let channels = parking_lot::Mutex::new(Some(HostReplicaChannels {
            outbound: Some(outbound_tx),
            inbound_tx: inbound_tx.clone(),
            inbound_rx,
            process_id: 7,
        }));
        let core = core_with_host(SelectionHost {
            replication: HostReplication {
                channels: Some(Arc::new(move |_| channels.lock().take())),
                ..HostReplication::default()
            },
            ..SelectionHost::default()
        });
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        // Replica events for a worker the scheduler has not registered yet are
        // dropped; the first local booking registers it.
        core.select_and_reserve(reserve_request("warm"))
            .await
            .expect("warm booking");
        core.free_reservation("warm").await.expect("free warm");
        let entry = core
            .entry(&RoutingPartitionId::new("model", "default"))
            .expect("entry");
        let peer_event = |request_id: &str, data| ActiveSequenceEvent {
            request_id: request_id.to_string(),
            worker: WorkerWithDpRank::new(1, 0),
            data,
            router_id: 99,
            lora_name: None,
        };
        let add = |request_id: &str| {
            peer_event(
                request_id,
                ActiveSequenceEventData::AddRequest {
                    token_sequence: Some(vec![1, 2]),
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                },
            )
        };

        inbound_tx.send(add("peer-a")).await.expect("send");
        inbound_tx.send(add("peer-b")).await.expect("send");
        wait_until("mirrored bookings indexed", || {
            core.indexed_booking("peer-a").is_some() && core.indexed_booking("peer-b").is_some()
        })
        .await;

        // A lifecycle call on a mirrored booking resolves through the index.
        core.free_reservation("peer-a")
            .await
            .expect("free mirrored");
        assert!(!entry.scheduler.has_request("peer-a"));
        assert!(core.indexed_booking("peer-a").is_none());

        // The peer freeing its own booking removes the mirror.
        inbound_tx
            .send(peer_event("peer-b", ActiveSequenceEventData::Free))
            .await
            .expect("send");
        wait_until("mirror removed", || {
            core.indexed_booking("peer-b").is_none()
        })
        .await;
        assert!(!entry.scheduler.has_request("peer-b"));
    }

    #[tokio::test]
    async fn lifecycle_operations_resolve_through_the_index() {
        let mut config = test_config(false);
        config.router_track_prefill_tokens = true;
        let core = SelectionCore::try_new_local(
            config,
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");

        for (worker_id, routing_group) in [(1, "group-a"), (2, "group-b")] {
            let mut request = worker(worker_id);
            request.routing_group = routing_group.to_string();
            core.upsert_worker(request).await.expect("worker upsert");
        }

        let entries = core.initialized_entries();
        assert_eq!(entries.len(), 2);
        let target = &entries[1];
        let target_group = target.key.routing_group.clone();
        let target_worker = *target
            .workers_tx
            .borrow()
            .keys()
            .next()
            .expect("target worker");

        let mut request = reserve_request("later-entry-reservation");
        request.routing_group = target_group.clone();
        core.select_and_reserve(request)
            .await
            .expect("reserve in later entry");

        let load = || {
            core.loads(Some("model"), Some(&target_group))[0]
                .loads
                .iter()
                .find(|load| load.worker_id == target_worker)
                .expect("target load")
                .potential_prefill_tokens
        };
        assert_eq!(load(), 4);

        core.prefill_complete("later-entry-reservation")
            .await
            .expect("complete prefill in later entry");
        assert_eq!(load(), 0);

        core.free_reservation("later-entry-reservation")
            .await
            .expect("free reservation in later entry");
        assert!(matches!(
            core.add_output_block("later-entry-reservation", None),
            Err(SelectionError::NotFound(_))
        ));
    }

    #[tokio::test]
    async fn advisory_select_reports_worker_load_and_busy_evaluation() {
        let mut config = test_config(false);
        config.conditional_disagg_prefill_busy_threshold = Some(0.5);
        config.conditional_disagg_decode_busy_threshold = Some(0.0);
        let core = SelectionCore::try_new_local(
            config,
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        let mut request = worker(1);
        request.total_kv_blocks = Some(1000);
        core.upsert_worker(request).await.expect("worker upsert");

        // Admitted (queued) select: decode evaluation comes from the catalog's
        // total_kv_blocks; no load snapshot is taken.
        let response = core.select(select_request()).await.expect("select");
        assert!(response.potential_decode_blocks > 0);
        assert_eq!(
            response.decode_busy,
            Some(true),
            "threshold 0.0 is always exceeded"
        );
        assert!(response.worker_load.is_none());

        // Advisory select: same decode evaluation plus the projected load.
        let mut request = select_request();
        request.advisory = true;
        let response = core.select(request).await.expect("advisory select");
        assert!(response.potential_decode_blocks > 0);
        assert_eq!(response.decode_busy, Some(true));
        let load = response.worker_load.expect("advisory load");
        assert_eq!(load.total_kv_blocks, Some(1000));
        assert_eq!(load.prefill_token_capacity, 1024);
        assert_eq!(load.active_prefill_tokens, 0);
        assert_eq!(load.prefill_busy, Some(false));

        // Advisory selection does not book.
        assert!(core.reservation_index.read().is_empty());
        assert_eq!(
            core.loads(Some("model"), Some("default"))[0].loads[0].potential_prefill_tokens,
            0
        );
    }

    #[tokio::test]
    async fn busy_evaluation_is_absent_without_thresholds_or_capacity() {
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        let mut request = select_request();
        request.advisory = true;
        let response = core.select(request).await.expect("advisory select");
        assert_eq!(response.decode_busy, None);
        let load = response.worker_load.expect("advisory load");
        assert_eq!(load.total_kv_blocks, None);
        assert_eq!(load.prefill_busy, None);
    }

    #[tokio::test]
    async fn reservation_index_tracks_bookings_until_freed() {
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        for (worker_id, routing_group) in [(1, "group-a"), (2, "group-b")] {
            let mut request = worker(worker_id);
            request.routing_group = routing_group.to_string();
            core.upsert_worker(request).await.expect("worker upsert");
        }
        let key_b = RoutingPartitionId::new("model", "group-b");

        // select_and_reserve records the booking's partition.
        let mut request = reserve_request("booked");
        request.routing_group = "group-b".to_string();
        core.select_and_reserve(request).await.expect("reserve");
        assert_eq!(
            core.reservation_index
                .read()
                .get("booked")
                .map(|r| &r.partition),
            Some(&key_b)
        );
        assert_eq!(
            core.indexed_booking("booked")
                .expect("indexed booking")
                .0
                .key,
            key_b
        );

        // The explicit reservation path records too.
        let mut request = select_request();
        request.routing_group = "group-b".to_string();
        request.selection_id = Some("cached".to_string());
        core.select(request).await.expect("select");
        assert!(core.reservation_index.read().get("cached").is_none());
        core.create_reservation(ReservationRequest {
            model_name: "model".to_string(),
            routing_group: "group-b".to_string(),
            selection_id: "cached".to_string(),
            worker_id: None,
            dp_rank: None,
            prompt: PromptRequest::default(),
            router_config_override: None,
            expected_output_tokens: None,
            effective_prefill_tokens: None,
            track_prefill_tokens: None,
        })
        .await
        .expect("cached reservation");
        assert_eq!(
            core.reservation_index
                .read()
                .get("cached")
                .map(|r| &r.partition),
            Some(&key_b)
        );

        // Lifecycle calls still resolve, and free drops the index entry.
        core.prefill_complete("booked")
            .await
            .expect("prefill complete");
        core.free_reservation("booked").await.expect("free");
        assert!(core.reservation_index.read().get("booked").is_none());
        core.free_reservation("cached").await.expect("free");
        assert!(core.reservation_index.read().is_empty());

        // Unknown ids fall back to the full scan and stay unindexed.
        assert!(matches!(
            core.prefill_complete("never-booked").await,
            Err(SelectionError::NotFound(_))
        ));
        assert!(core.reservation_index.read().is_empty());
    }

    #[tokio::test]
    async fn reservation_index_sweep_drops_bookings_released_out_of_band() {
        let core = SelectionCore::try_new_local(
            test_config(false),
            1,
            CancellationToken::new(),
            SelectionCacheConfig::default(),
        )
        .expect("valid test config");
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.select_and_reserve(reserve_request("live"))
            .await
            .expect("reserve live");
        core.select_and_reserve(reserve_request("stale"))
            .await
            .expect("reserve stale");
        assert_eq!(core.reservation_index.read().len(), 2);
        assert_eq!(
            sweep_reservation_index(&core.entries, &core.reservation_index),
            0
        );

        // Release directly through the scheduler, as force-expiry would.
        let entry = core
            .entry(&RoutingPartitionId::new("model", "default"))
            .expect("entry");
        entry.scheduler.free("stale").await.expect("scheduler free");

        assert_eq!(
            sweep_reservation_index(&core.entries, &core.reservation_index),
            1
        );
        let index = core.reservation_index.read();
        assert_eq!(index.len(), 1);
        assert!(index.contains_key("live"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lifecycle_lookup_does_not_nest_reservation_index_inside_entries() {
        // Three parties: a sweep holding `entries` and wanting `reservation_index`,
        // a lifecycle call, and a partition creation queued on `entries.write()`.
        let core = Arc::new(
            SelectionCore::try_new_local(
                test_config(false),
                1,
                CancellationToken::new(),
                SelectionCacheConfig::default(),
            )
            .expect("valid test config"),
        );
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.select_and_reserve(reserve_request("live"))
            .await
            .expect("reserve live");
        let deadline = Instant::now() + Duration::from_secs(5);

        let sweep_entries = core.entries.read();
        let writer = {
            let core = Arc::clone(&core);
            std::thread::spawn(move || {
                core.entries
                    .write()
                    .entry(RoutingPartitionId::new("other", "default"))
                    .or_insert_with(|| Arc::new(OnceCell::new()));
            })
        };
        while core.entries.try_read().is_some() {
            assert!(Instant::now() < deadline, "partition writer never queued");
            std::thread::yield_now();
        }
        let lifecycle_started = Arc::new(AtomicBool::new(false));
        let lifecycle = {
            let core = Arc::clone(&core);
            let started = Arc::clone(&lifecycle_started);
            std::thread::spawn(move || {
                started.store(true, Ordering::Release);
                core.indexed_booking("live").is_some()
            })
        };
        while !lifecycle_started.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline, "lifecycle thread never started");
            std::thread::yield_now();
        }
        sleep(Duration::from_millis(50));

        // With the index guard held across `entries.read()` this never acquires.
        drop(
            core.reservation_index
                .try_write_for(Duration::from_secs(2))
                .expect("reservation index must not be held by a blocked lifecycle call"),
        );
        drop(sweep_entries);
        writer.join().expect("partition writer");
        assert!(lifecycle.join().expect("lifecycle lookup"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queued_selection_returns_refreshed_overlap_snapshot() {
        let mut config = test_config(false);
        config.router_queue_threshold = Some(0.0);
        let core = Arc::new(
            SelectionCore::try_new_local(
                config,
                1,
                CancellationToken::new(),
                SelectionCacheConfig::default(),
            )
            .expect("valid test config"),
        );

        for worker_id in [1, 2] {
            let mut request = worker(worker_id);
            request.max_num_batched_tokens = Some(8);
            core.upsert_worker(request).await.expect("worker upsert");
        }
        let key = RoutingPartitionId::new("model", "default");
        let entry = core.entry(&key).expect("entry");
        entry
            .indexer
            .apply_event_routed(store_event(1, 0, 1, &[], &[11], StorageTier::Device))
            .await
            .unwrap();
        entry.indexer.dump_events().await.expect("flush indexer");

        for worker_id in [1, 2] {
            core.create_reservation(ReservationRequest {
                model_name: "model".to_string(),
                routing_group: "default".to_string(),
                selection_id: format!("occupy-{worker_id}"),
                worker_id: Some(worker_id),
                dp_rank: Some(0),
                prompt: PromptRequest {
                    token_ids: None,
                    mm_routing_info: None,
                    block_mm_infos: None,
                    block_hashes: None,
                    sequence_hashes: Some(vec![1, 2]),
                    isl_tokens: Some(8),
                    lora_name: None,
                    cache_namespace: None,
                    is_eagle: None,
                },
                router_config_override: None,
                expected_output_tokens: None,
                effective_prefill_tokens: Some(8),
                track_prefill_tokens: None,
            })
            .await
            .expect("occupy worker");
        }

        let queued_core = Arc::clone(&core);
        let queued = tokio::spawn(async move {
            queued_core
                .select_and_reserve(SelectAndReserveRequest {
                    model_name: "model".to_string(),
                    routing_group: "default".to_string(),
                    selection_id: Some("refresh-selection".to_string()),
                    prompt: PromptRequest {
                        token_ids: None,
                        mm_routing_info: None,
                        block_mm_infos: None,
                        block_hashes: Some(vec![11, 12]),
                        sequence_hashes: Some(vec![101, 102]),
                        isl_tokens: Some(8),
                        lora_name: None,
                        cache_namespace: None,
                        is_eagle: None,
                    },
                    router_config_override: None,
                    expected_output_tokens: None,
                    priority_jump: None,
                    strict_priority: None,
                    session_id: None,
                    session_context: None,
                    affinity_target: None,
                    pinned_worker: None,
                    allowed_worker_ids: None,
                    routing_constraints: RoutingConstraints::default(),
                })
                .await
        });
        wait_for_pending_selection(&core).await;

        entry
            .indexer
            .apply_event_routed(store_event(2, 0, 1, &[], &[11, 12], StorageTier::Device))
            .await
            .unwrap();
        entry.indexer.dump_events().await.expect("flush indexer");
        // Freeze time only after async setup so background timers cannot expire the fixture.
        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(11)).await;
        core.free_reservation("occupy-2")
            .await
            .expect("release worker 2");

        let response = queued.await.expect("selection task").expect("selection");
        assert_eq!(response.worker_id, 2);
        assert_eq!(response.effective_prefill_tokens, 0);
        assert_eq!(response.overlap.gpu, 8);
        assert_eq!(response.overlap.cpu, 8);
        assert_eq!(response.overlap.disk, 8);
        assert_eq!(response.overlap.dp, HashMap::from([("0".to_string(), 8)]));
    }

    fn core_with_session_affinity_mode(mode: SessionAffinityMode) -> SelectionCore {
        let config = test_config(false);
        let tracking_hash = Arc::new(
            TrackingHashContext::from_config(&config).expect("valid tracking hash configuration"),
        );
        let indexer_policy = IndexerPolicy::from_router_config(&config).expect("indexer policy");
        SelectionCore::new_inner(
            config,
            1,
            CancellationToken::new(),
            None,
            None,
            SelectionHost::default(),
            WorkerType::Aggregated,
            true,
            SelectionCacheConfig::default(),
            tracking_hash,
            indexer_policy,
            Some(SessionAffinityConfig::new(Duration::from_secs(10)).with_mode(mode)),
        )
    }

    fn core_with_session_affinity() -> SelectionCore {
        core_with_session_affinity_mode(SessionAffinityMode::Hard)
    }

    fn bound_worker(core: &SelectionCore, session_id: &str) -> Option<WorkerId> {
        core.entry(&RoutingPartitionId::new("model", "default"))
            .and_then(|entry| entry.affinity.get().cloned())
            .and_then(|table| table.query_target(session_id, None).expect("query"))
            .map(|target| target.worker_id)
    }

    fn session_reservation(selection_id: &str, session_id: &str) -> SelectAndReserveRequest {
        let mut request = reserve_request(selection_id);
        request.session_id = Some(session_id.to_string());
        request
    }

    #[tokio::test]
    async fn departed_session_worker_reinitializes_the_session() {
        let core = core_with_session_affinity();
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");
        let first = core
            .select_and_reserve(session_reservation("r1", "s"))
            .await
            .expect("first booking");
        core.free_reservation("r1").await.expect("free");
        core.delete_worker(first.worker_id).await.expect("delete");

        let second = core
            .select_and_reserve(session_reservation("r2", "s"))
            .await
            .expect("session must move off a departed worker");
        assert_ne!(second.worker_id, first.worker_id);
        assert_eq!(bound_worker(&core, "s"), Some(second.worker_id));
    }

    #[tokio::test]
    async fn session_worker_departing_after_the_hold_reinitializes_the_session() {
        let core = core_with_session_affinity();
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");
        let first = core
            .select_and_reserve(session_reservation("r1", "s"))
            .await
            .expect("first booking");
        core.free_reservation("r1").await.expect("free");
        let key = RoutingPartitionId::new("model", "default");

        // One poll takes the hold (the bound worker still passes the check)
        // and hands the request to the scheduler actor, which has not run yet.
        let mut second = Box::pin(core.select_and_reserve(session_reservation("r2", "s")));
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        assert!(second.as_mut().poll(&mut context).is_pending());
        core.catalog
            .set_lifecycle(first.worker_id, WorkerLifecycle::Draining, Vec::new());
        core.publish_scheduler_config(&key).expect("publish");

        let second = second
            .await
            .expect("a departure after the hold is not a client fault");
        assert_ne!(second.worker_id, first.worker_id);
        assert_eq!(bound_worker(&core, "s"), Some(second.worker_id));
    }

    #[tokio::test]
    async fn concurrent_holds_on_a_departed_worker_both_land_on_the_replacement() {
        let table = SessionAffinity::new(Duration::from_secs(60)).expect("affinity table");
        let departed = WorkerAffinityTarget::new(1, Some(0));
        let replacement = WorkerAffinityTarget::new(2, Some(0));
        let Hold::Initialize(init) = table.acquire("s", None).await.expect("acquire") else {
            panic!("fresh session must initialize");
        };
        drop(
            table
                .commit(Hold::Initialize(init), departed)
                .expect("bind"),
        );
        let (
            Hold::Bound {
                lease: mut first, ..
            },
            Hold::Bound {
                lease: mut second, ..
            },
        ) = (
            table.acquire("s", None).await.expect("first hold"),
            table.acquire("s", None).await.expect("second hold"),
        )
        else {
            panic!("both requests hold the departed binding");
        };

        // Both requests notice the departure, as `hold_session` does: the first
        // invalidation drops the binding, the second is a no-op release.
        first.invalidate();
        second.invalidate();
        let leases = [
            table.commit(
                table.acquire("s", None).await.expect("re-acquire"),
                replacement,
            ),
            table.commit(
                table.acquire("s", None).await.expect("re-acquire"),
                replacement,
            ),
        ];
        for lease in leases {
            drop(lease.expect("both requests bind to the replacement"));
        }
        assert_eq!(
            table.query_target("s", None).expect("query"),
            Some(replacement)
        );
    }

    #[tokio::test]
    async fn hard_mode_rejects_dispatch_away_from_a_live_binding() {
        let core = core_with_session_affinity();
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");
        let first = core
            .select_and_reserve(session_reservation("r1", "s"))
            .await
            .expect("first booking");
        core.free_reservation("r1").await.expect("free");
        let other = if first.worker_id == 1 { 2 } else { 1 };

        // Steering cannot reach the bound worker, so selection lands elsewhere.
        let mut request = session_reservation("r2", "s");
        request.allowed_worker_ids = Some(HashSet::from([other]));
        let err = core
            .select_and_reserve(request)
            .await
            .expect_err("hard affinity rejects a dispatch away from the binding");
        assert!(matches!(err, SelectionError::BadRequest(_)), "{err:?}");
        let entry = core
            .entry(&RoutingPartitionId::new("model", "default"))
            .expect("entry");
        wait_until("rejected booking release", || {
            !entry.scheduler.has_request("r2")
        })
        .await;
        assert!(core.reservation_index.read().is_empty());
        assert_eq!(bound_worker(&core, "s"), None, "stale binding is dropped");

        // The session's next request re-initializes.
        let mut request = session_reservation("r3", "s");
        request.allowed_worker_ids = Some(HashSet::from([other]));
        let third = core.select_and_reserve(request).await.expect("rebind");
        assert_eq!(third.worker_id, other);
        assert_eq!(bound_worker(&core, "s"), Some(other));
    }

    #[tokio::test]
    async fn soft_mode_follows_the_dispatch() {
        let core = core_with_session_affinity_mode(SessionAffinityMode::Soft);
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");
        let first = core
            .select_and_reserve(session_reservation("r1", "s"))
            .await
            .expect("first booking");
        core.free_reservation("r1").await.expect("free");
        let other = if first.worker_id == 1 { 2 } else { 1 };

        let mut request = session_reservation("r2", "s");
        request.allowed_worker_ids = Some(HashSet::from([other]));
        let second = core.select_and_reserve(request).await.expect("soft rebind");
        assert_eq!(second.worker_id, other);
        assert_eq!(bound_worker(&core, "s"), Some(other));
    }

    #[tokio::test]
    async fn session_stays_on_its_first_worker_across_bookings() {
        let core = core_with_session_affinity();
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");

        let first = core
            .select_and_reserve(session_reservation("r1", "chat-a"))
            .await
            .expect("first booking");
        for index in 0..4 {
            let response = core
                .select_and_reserve(session_reservation(&format!("r-{index}"), "chat-a"))
                .await
                .expect("booking");
            assert_eq!(response.worker_id, first.worker_id, "session must stay put");
            core.free_reservation(&format!("r-{index}"))
                .await
                .expect("free");
        }
        // A read-only select sees the binding too.
        let mut advisory = select_request();
        advisory.session_id = Some("chat-a".to_string());
        let response = core.select(advisory).await.expect("select");
        assert_eq!(response.worker_id, first.worker_id);
        core.free_reservation("r1").await.expect("free");
    }

    #[tokio::test]
    async fn replicated_binding_steers_a_new_session_and_frees_with_the_booking() {
        let core = core_with_session_affinity();
        core.upsert_worker(worker(1)).await.expect("worker upsert");
        core.upsert_worker(worker(2)).await.expect("worker upsert");

        core.dispatch_affinity_event(AffinityBindingEvent {
            partition: RoutingPartitionId::new("model", "default"),
            session_id: "chat-b".to_string(),
            worker_id: 2,
            dp_rank: Some(0),
            sequence: 1,
            writer_id: 99,
        });
        let response = core
            .select_and_reserve(session_reservation("r2", "chat-b"))
            .await
            .expect("booking");
        assert_eq!(response.worker_id, 2);
        assert!(
            core.reservation_index
                .read()
                .get("r2")
                .unwrap()
                ._affinity_lease
                .is_some()
        );
        core.free_reservation("r2").await.expect("free");
        assert!(core.reservation_index.read().is_empty());
    }

    #[tokio::test]
    async fn expired_booking_releases_affinity_lease() {
        let core = core_with_session_affinity();
        core.upsert_worker(worker(1)).await.unwrap();
        core.select_and_reserve(session_reservation("abandoned", "session"))
            .await
            .unwrap();
        let key = RoutingPartitionId::new("model", "default");
        core.entry(&key)
            .unwrap()
            .scheduler
            .free("abandoned")
            .await
            .unwrap();
        assert_eq!(
            sweep_reservation_index(&core.entries, &core.reservation_index),
            1
        );
        tokio::time::pause();
        tokio::time::advance(Duration::from_secs(11)).await;
        assert_eq!(
            core.entry(&RoutingPartitionId::new("model", "default"))
                .unwrap()
                .affinity
                .get()
                .unwrap()
                .query_target("session", None)
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn partition_sessions_keep_independent_bindings() {
        let core = core_with_session_affinity();
        core.upsert_worker(worker(1)).await.unwrap();
        let first = core
            .select_and_reserve(session_reservation("first", "shared-session"))
            .await
            .unwrap();
        core.upsert_worker(worker(2)).await.unwrap();
        let mut other = worker(3);
        other.routing_group = "other".to_string();
        core.upsert_worker(other).await.unwrap();
        let mut request = session_reservation("other", "shared-session");
        request.routing_group = "other".to_string();
        assert_eq!(core.select_and_reserve(request).await.unwrap().worker_id, 3);
        assert_eq!(
            core.entry(&RoutingPartitionId::new("model", "default"))
                .unwrap()
                .affinity
                .get()
                .unwrap()
                .query_target("shared-session", None)
                .unwrap()
                .unwrap()
                .worker_id,
            first.worker_id
        );
        let again = core
            .select_and_reserve(session_reservation("again", "shared-session"))
            .await
            .unwrap();
        assert_eq!(again.worker_id, first.worker_id);
    }

    #[tokio::test]
    async fn rejoined_worker_feeds_partition_index() {
        use crate::protocols::{BlockHashOptions, compute_block_hash_for_seq};
        for update_only in [false, true] {
            let core = SelectionCore::try_new_local(
                test_config(true),
                1,
                CancellationToken::new(),
                SelectionCacheConfig::default(),
            )
            .unwrap();
            let request = worker_with_kv_events(1);
            core.upsert_worker(request.clone()).await.unwrap();
            let key = RoutingPartitionId::new("model", "default");
            let partition = core.partition(&key).unwrap();
            let tokens: Vec<u32> = (1..=8).collect();
            let hashes: Vec<u64> =
                compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default())
                    .into_iter()
                    .map(|hash| hash.0)
                    .collect();
            let indexer = core
                .indexer_registry
                .get_indexer(&key)
                .unwrap()
                .indexer
                .clone();
            indexer
                .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
                .await
                .unwrap();
            indexer.dump_events().await.unwrap();
            if !update_only {
                core.delete_worker(1).await.unwrap();
            }
            let mut request = request;
            request.total_kv_blocks = Some(2048);
            core.upsert_worker(request).await.unwrap();
            let retained = partition.indexer().dump_events().await.unwrap();
            if update_only {
                assert!(
                    !retained.is_empty(),
                    "metadata update cleared cached blocks"
                );
            } else {
                assert!(retained.is_empty(), "removed worker retained cached blocks");
                let current = core
                    .indexer_registry
                    .get_indexer(&key)
                    .unwrap()
                    .indexer
                    .clone();
                current
                    .apply_event_routed(store_event(1, 0, 1, &[], &hashes, StorageTier::Device))
                    .await
                    .unwrap();
                current.dump_events().await.unwrap();
            }
            let mut request = select_request();
            request.prompt = PromptRequest {
                token_ids: Some(tokens),
                ..PromptRequest::default()
            };
            assert_eq!(
                core.select(request).await.unwrap().overlap.longest_matched,
                8,
                "update_only={update_only}"
            );
            assert_eq!(
                partition.indexer().dump_events().await.unwrap().len(),
                indexer.dump_events().await.unwrap().len()
            );
        }
    }
    #[tokio::test]
    async fn metadata_update_preserves_live_booking() {
        struct SuspendedDetach(tokio::sync::Notify);
        #[async_trait::async_trait]
        impl KvEventIngress for SuspendedDetach {
            fn open(
                &self,
                registry: &WorkerRegistry,
                key: &RoutingPartitionId,
                block_size: u32,
            ) -> Indexer {
                registry.get_or_create_indexer(key.clone(), block_size)
            }
            async fn detach(&self, _registry: &WorkerRegistry, _record: &WorkerCatalogRecord) {
                self.0.notify_one();
                std::future::pending().await
            }
        }
        let ingress = Arc::new(SuspendedDetach(tokio::sync::Notify::new()));
        let core = core_with_host(SelectionHost {
            cache: HostCache {
                index: KvIndexSource::Owned(ingress.clone()),
                shared: None,
            },
            ..SelectionHost::default()
        });
        core.upsert_worker(worker(1)).await.unwrap();
        core.select_and_reserve(reserve_request("live"))
            .await
            .unwrap();
        let entry = core
            .entry(&RoutingPartitionId::new("model", "default"))
            .unwrap();
        let mut updated = worker(1);
        updated.max_num_batched_tokens = Some(2048);
        tokio::select! {
            result = core.upsert_worker(updated) => { result.unwrap(); },
            _ = ingress.0.notified() => {
                // A capacity update must not temporarily withdraw this worker.
                assert!(entry.scheduler.has_request("live"));
                panic!("capacity update detached the worker");
            }
        }
        assert!(entry.scheduler.has_request("live"));
        core.free_reservation("live").await.unwrap();
    }
    #[tokio::test(start_paused = true)]
    async fn host_lease_manager_owns_expiry() {
        use crate::scheduling::queue::SchedulerBookingDescriptor;
        struct HostLeases;
        impl ReplicaRequestLeaseObserver for HostLeases {
            fn admitted(&self, _: SchedulerBookingDescriptor) {}
            fn progressed(&self, _: &SchedulerBookingDescriptor) {}
            fn completed(&self, _: &SchedulerBookingDescriptor) {}
        }
        for host_owned in [false, true] {
            let core = core_with_host(SelectionHost {
                replication: HostReplication {
                    request_leases: host_owned
                        .then(|| Arc::new(HostLeases) as Arc<dyn ReplicaRequestLeaseObserver>),
                    ..HostReplication::default()
                },
                ..SelectionHost::default()
            });
            core.upsert_worker(worker(1)).await.unwrap();
            core.select_and_reserve(reserve_request("live"))
                .await
                .unwrap();
            tokio::task::yield_now().await;
            tokio::time::advance(active_request_expiry_duration() * 3).await;
            tokio::task::yield_now().await;
            let entry = core
                .entry(&RoutingPartitionId::new("model", "default"))
                .unwrap();
            assert_eq!(entry.scheduler.has_request("live"), host_owned);
            if host_owned {
                core.free_reservation("live").await.unwrap();
                assert!(!entry.scheduler.has_request("live"));
            }
        }
    }

    #[tokio::test]
    async fn catalog_updates_commit_in_order_without_exposing_candidates() {
        struct PausedUpdate {
            entered: tokio::sync::Notify,
            release: tokio::sync::Notify,
        }
        #[async_trait::async_trait]
        impl KvEventIngress for PausedUpdate {
            fn open(
                &self,
                registry: &WorkerRegistry,
                key: &RoutingPartitionId,
                block_size: u32,
            ) -> Indexer {
                registry.get_or_create_indexer(key.clone(), block_size)
            }
            async fn reconcile(
                &self,
                _: &WorkerRegistry,
                previous: Option<&WorkerCatalogRecord>,
                _: &WorkerCatalogRecord,
            ) -> Result<(), SelectionError> {
                if previous.is_some() {
                    self.entered.notify_one();
                    self.release.notified().await;
                }
                Ok(())
            }
        }
        let ingress = Arc::new(PausedUpdate {
            entered: tokio::sync::Notify::new(),
            release: tokio::sync::Notify::new(),
        });
        let mut core = core_with_host(SelectionHost {
            cache: HostCache {
                index: KvIndexSource::Owned(ingress.clone()),
                shared: None,
            },
            ..SelectionHost::default()
        });
        core.listens_for_kv_events = true;
        core.upsert_worker(worker(1)).await.unwrap();
        core.select_and_reserve(reserve_request("live"))
            .await
            .unwrap();
        let mut updated = worker(1);
        updated.max_num_batched_tokens = Some(2048);
        let update = core.upsert_worker(updated);
        tokio::pin!(update);
        tokio::select! {
            _ = &mut update => panic!("update should pause before committing"),
            _ = ingress.entered.notified() => {}
        }
        let committed = core.catalog.get(1).unwrap();
        assert_eq!(committed.lifecycle, WorkerLifecycle::Schedulable);
        assert_eq!(committed.max_num_batched_tokens, Some(1024));
        let entry = core.entry(&committed.key()).unwrap();
        assert!(entry.scheduler.has_request("live"));
        let deletion = core.delete_worker(1);
        tokio::pin!(deletion);
        assert!(
            std::future::Future::poll(
                deletion.as_mut(),
                &mut std::task::Context::from_waker(std::task::Waker::noop())
            )
            .is_pending()
        );
        ingress.release.notify_one();
        assert_eq!(update.await.unwrap().max_num_batched_tokens, Some(2048));
        assert!(entry.scheduler.has_request("live"));
        assert_eq!(
            deletion.await.unwrap().lifecycle,
            WorkerLifecycle::Unschedulable
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            while entry.scheduler.has_request("live") {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn affinity_configuration_rejects_invalid_or_conflicting_ttl() {
        use super::super::service::SelectionServiceBuilder;
        for ttl in [
            Duration::ZERO,
            Duration::from_secs(super::super::affinity::MAX_SESSION_AFFINITY_TTL_SECS + 1),
        ] {
            let result = SelectionServiceBuilder::new(
                test_config(false),
                WorkerType::Aggregated,
                Default::default(),
            )
            .session_affinity(ttl)
            .build()
            .await;
            assert!(result.is_err());
        }
        let core = core_with_session_affinity();
        core.upsert_worker(worker(1)).await.unwrap();
        let partition = core
            .partition(&RoutingPartitionId::new("model", "default"))
            .unwrap();
        assert!(matches!(
            partition.session_affinity(SessionAffinityConfig::new(Duration::from_secs(20))),
            Err(SelectionError::Conflict(_))
        ));
    }
}
