// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use crossbeam_queue::SegQueue;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::config::RouterQueuePolicy;
use super::filter::RoutingEligibility;
use super::overlap::SelectedWorkerTierSnapshot;
use super::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, read_overlap_refresh_after, refresh_overlap,
};
use super::policy_config::{PolicyClassConfig, PolicyProfile};
use super::policy_queue::{
    DeadlineStage, PolicyQueue, PolicyQueueEntry, QueueArrival, QueueDeadlineExceeded,
    QueueSnapshot,
};
use super::prefill_load::{PrefillLoadEstimator, effective_prefill_tokens};
use super::queue_admission::{
    AdmissionHandoff, AdmissionRequestOutcome, QueueAdmissionEvent, QueueAdmissionId,
    QueueAdmissionLayer, QueueAdmissionWorker, QueueAdmissionWorkerSnapshot,
};
use super::selector::{DefaultWorkerSelector, WorkerSelector};
use super::types::{
    AdvisorySchedulingResponse, AdvisoryWorkerLoad, KvSchedulerError, NonMaxOverlapSelection,
    NonMaxOverlapSelectionObserver, OverloadedWorkerProvider, SchedulingContext, SchedulingRequest,
    SchedulingResponse, WorkerAvailabilityProvider, WorkerPlacement,
};
use crate::protocols::{
    LocalBlockHash, PrefillLoadHint, WorkerConfigLike, WorkerId, WorkerSelectionResult,
    WorkerWithDpRank,
};
use crate::sequences::topology::WorkerDpRange;
use crate::sequences::{ActiveSequencesMultiWorker, SequencePublisher, SequenceRequest};

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

const ADMISSION_CHANNEL_CAPACITY: usize = 65_536;

struct ClassQueueCounters {
    pending_count: AtomicUsize,
    pending_isl_tokens: AtomicUsize,
    pending_cached_tokens: AtomicUsize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassQueueStats {
    pub pending_count: usize,
    pub pending_isl_tokens: usize,
    pub pending_cached_tokens: usize,
}

/// One request the scheduler owns between router arrival and dispatch.
///
/// This is the actor's handoff payload, so it carries only neutral request
/// facts and no layer reads another's fields. The admission layer may retain it
/// while a policy defers the request. After the handoff, `should_queue` decides
/// what becomes of it: direct dispatch consumes it outright, or a policy class
/// stores it until its deficit-round-robin turn. `admission_id` is the identity
/// the admission layer reports this request's lifecycle under, and is absent for
/// a bypassed or unmanaged request.
struct QueuedRequest {
    request: SchedulingRequest,
    /// Monotonic router arrival. A class that stores this request derives its
    /// fixed-SLO deadline from this instant, so time spent deferred is spent
    /// from the same latency budget; a request dispatched directly gets no
    /// deadline from it.
    arrival_at: Instant,
    block_hashes: Option<Vec<LocalBlockHash>>,
    admission_id: Option<QueueAdmissionId>,
}

struct SelectedWorkerForRequest {
    selection: WorkerSelectionResult,
    selected_worker_tiers: SelectedWorkerTierSnapshot,
    selected_worker_load: AdvisoryWorkerLoad,
    non_max_overlap_selection: Option<NonMaxOverlapSelection>,
}

/// What the router's direct-versus-queued decision did with one request the
/// queue-admission layer handed over.
enum ScheduleOutcome {
    /// Sent straight to worker selection: the request's class had no runnable
    /// backlog and an eligible worker was free, so it never entered queue
    /// storage and never met a class limit, the queue Admission gate, or DRR.
    Admitted { owns_lifecycle: bool },
    /// Stored in its policy class's queue to wait for a deficit-round-robin turn.
    Queued,
    /// Answered with an error before it could run.
    Rejected,
}

fn non_max_overlap_selection<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    request: &SchedulingRequest,
    eligibility: RoutingEligibility<'_>,
    selected_worker: WorkerWithDpRank,
    selected_overlap_blocks: f64,
) -> Option<NonMaxOverlapSelection> {
    if eligibility.pinned_worker().is_some() {
        return None;
    }

    let mut highest_overlap = None;
    for (&worker, &overlap_blocks) in &request.overlap.effective_overlap_blocks {
        if overlap_blocks <= selected_overlap_blocks
            || eligibility.validate_worker_rank(workers, worker).is_err()
        {
            continue;
        }
        let is_better = highest_overlap.is_none_or(
            |(current_worker, current_overlap): (WorkerWithDpRank, f64)| {
                overlap_blocks > current_overlap
                    || (overlap_blocks == current_overlap && worker < current_worker)
            },
        );
        if is_better {
            highest_overlap = Some((worker, overlap_blocks));
        }
    }

    let (highest_overlap_worker, highest_overlap_blocks) = highest_overlap?;
    (highest_overlap_blocks > selected_overlap_blocks).then_some(NonMaxOverlapSelection {
        selected_worker,
        highest_overlap_worker,
        highest_overlap_blocks,
        selected_overlap_blocks,
    })
}

/// The policy-class name carried in request metadata, with an empty name
/// normalized to "no preference".
#[inline]
fn requested_policy_class(request: &SchedulingRequest) -> Option<&str> {
    request
        .policy_class
        .as_deref()
        .filter(|name| !name.is_empty())
}

/// How far past `deadline` `now` is, or `None` when the request is still inside
/// its class SLO or the class has no SLO.
#[inline]
fn overdue_by(deadline: Option<Instant>, now: Instant) -> Option<Duration> {
    deadline
        .filter(|deadline| now > *deadline)
        .map(|deadline| now.saturating_duration_since(deadline))
}

fn target_cached_prefix_blocks(request: &SchedulingRequest, target: WorkerWithDpRank) -> u32 {
    let device = request
        .overlap
        .tier_overlap_blocks
        .device
        .get(&target)
        .copied()
        .unwrap_or(0);
    let lower_tier = request
        .overlap
        .tier_overlap_blocks
        .host_pinned
        .get(&target)
        .copied()
        .unwrap_or(0);
    u32::try_from(device.saturating_add(lower_tier)).unwrap_or(u32::MAX)
}

#[allow(clippy::large_enum_variant)]
enum AdmissionCommand {
    Enqueue {
        request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        /// Monotonic router arrival, captured before the bounded actor-channel
        /// wait so channel backlog counts against the request's class SLO.
        arrival_at: Instant,
        lease: Option<Box<RequestLifecycleLease>>,
        ack_tx: oneshot::Sender<Option<Box<RequestLifecycleLease>>>,
    },
    SelectWithoutAdmission {
        request: SchedulingRequest,
        resp_tx: oneshot::Sender<Result<AdvisorySchedulingResponse, KvSchedulerError>>,
    },
    Update {
        worker: Option<WorkerWithDpRank>,
        finished: Option<(String, Option<WorkerWithDpRank>, AdmissionRequestOutcome)>,
        reconcile_admission: bool,
        ack_tx: oneshot::Sender<bool>,
    },
    Cleanup,
}

#[derive(Debug, PartialEq, Eq)]
struct AdmissionCleanupEntry {
    request_id: String,
}

#[derive(Default)]
struct AdmissionCleanup {
    dirty: SegQueue<AdmissionCleanupEntry>,
    pending: AtomicBool,
}

impl AdmissionCleanup {
    fn enqueue(&self, cleanup: AdmissionCleanupEntry) -> bool {
        self.dirty.push(cleanup);
        !self.pending.swap(true, AtomicOrdering::AcqRel)
    }

    fn drain(&self) -> Vec<AdmissionCleanupEntry> {
        if !self.pending.load(AtomicOrdering::Acquire) {
            return Vec::new();
        }

        // Drain to a quiescent point to preserve the coalesced-wake handoff. A large burst of
        // active lease drops can delay actor commands; any bounded or interleaved drain
        // must preserve wake correctness and be benchmarked.
        let mut dirty = Vec::new();
        loop {
            while let Some(cleanup) = self.dirty.pop() {
                dirty.push(cleanup);
            }
            self.pending.store(false, AtomicOrdering::Release);
            if self.dirty.is_empty() {
                return dirty;
            }
            self.pending.store(true, AtomicOrdering::Release);
        }
    }
}

/// Single-owner cleanup lease for one scheduler-tracked request.
pub(crate) struct RequestLifecycleLease {
    cleanup: Arc<AdmissionCleanup>,
    actor_tx: mpsc::Sender<AdmissionCommand>,
    request_id: Option<String>,
}

impl std::fmt::Debug for RequestLifecycleLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestLifecycleLease")
            .field("request_id", &self.request_id)
            .finish_non_exhaustive()
    }
}

impl RequestLifecycleLease {
    pub(crate) fn disarm(&mut self) {
        self.request_id = None;
    }
}

impl Drop for RequestLifecycleLease {
    fn drop(&mut self) {
        let Some(request_id) = self.request_id.take() else {
            return;
        };
        if self.cleanup.enqueue(AdmissionCleanupEntry { request_id }) {
            let _ = self.actor_tx.try_send(AdmissionCommand::Cleanup);
        }
    }
}

struct SchedulerQueueActor<
    P: SequencePublisher,
    C: WorkerConfigLike,
    Sel: WorkerSelector<C>,
    RF: OverlapScoresRefresh,
> {
    pending: PolicyQueue<QueuedRequest>,
    cleanup: Arc<AdmissionCleanup>,
    queueing_enabled: bool,
    profile: PolicyProfile,
    pending_count: Arc<AtomicUsize>,
    pending_isl_tokens: Arc<AtomicUsize>,
    class_counters: Arc<Vec<ClassQueueCounters>>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    /// Epoch for the fallback profile's arrival-offset ordering score.
    start_time: Instant,
    block_size: u32,
    selector: Sel,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    overlap_scores_refresh: Option<Arc<RF>>,
    overlap_refresh_after: Option<Duration>,
    overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    available_worker_provider: Option<WorkerAvailabilityProvider>,
    /// The queue-admission layer, when a worker-selection policy attached one.
    /// It is independent of `pending`: the actor coordinates the two and is the
    /// only place they meet.
    admission: Option<QueueAdmissionLayer<QueuedRequest>>,
    /// Request IDs whose lifecycle ownership the actor released while processing
    /// the current command. Only the `Enqueue` arm reads it, and it clears the
    /// set before it starts, so this never grows past one command's releases.
    released_lifecycle_request_ids: HashSet<String>,
    admission_worker_snapshot: QueueAdmissionWorkerSnapshot,
    admission_overloaded_worker_ids: Option<Arc<HashSet<WorkerId>>>,
    admission_available_worker_ids: Option<Arc<HashSet<WorkerId>>>,
    non_max_overlap_selection_observer: Arc<OnceLock<NonMaxOverlapSelectionObserver>>,
}

/// Queue that gates scheduling requests behind a capacity check.
/// When all workers exceed `threshold_frac` utilisation the request is parked in `pending`.
/// When capacity frees up (`update()`), pending requests are scheduled in priority order.
/// If queueing is disabled (threshold_frac is None), requests are scheduled immediately.
pub struct SchedulerQueue<
    P: SequencePublisher,
    C: WorkerConfigLike,
    Sel: WorkerSelector<C> = DefaultWorkerSelector,
    RF: OverlapScoresRefresh = NoopOverlapScoresRefresh,
> {
    admission_tx: mpsc::Sender<AdmissionCommand>,
    cleanup: Arc<AdmissionCleanup>,
    /// Number of requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_count: Arc<AtomicUsize>,
    /// Sum of `isl_tokens` for requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_isl_tokens: Arc<AtomicUsize>,
    class_counters: Arc<Vec<ClassQueueCounters>>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    queueing_enabled: bool,
    has_admission_policy: bool,
    admission_reconcile_interval: Option<Duration>,
    supports_overlap_refresh: bool,
    non_max_overlap_selection_observer: Arc<OnceLock<NonMaxOverlapSelectionObserver>>,
    _marker: PhantomData<fn() -> (Sel, RF)>,
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, Sel, RF>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_overlap_refresh(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        block_size: u32,
        selector: Sel,
        queue_policy: RouterQueuePolicy,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
    ) -> Self {
        let profile = PolicyProfile::synthetic(threshold_frac, queue_policy);
        Self::new_with_policy_profile(
            slots,
            workers_with_configs,
            profile,
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overloaded_worker_provider,
            available_worker_provider,
        )
        .expect("synthetic policy profile does not require admission policies")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_policy_profile(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        profile: PolicyProfile,
        block_size: u32,
        selector: Sel,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
    ) -> Result<Self, KvSchedulerError> {
        Self::new_with_policy_profile_and_capacity(
            slots,
            workers_with_configs,
            profile,
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overloaded_worker_provider,
            available_worker_provider,
            ADMISSION_CHANNEL_CAPACITY,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_policy_profile_and_capacity(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        profile: PolicyProfile,
        block_size: u32,
        mut selector: Sel,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
        admission_channel_capacity: usize,
    ) -> Result<Self, KvSchedulerError> {
        let admission = selector
            .take_admission_policy()
            .map(QueueAdmissionLayer::new);
        let admission_reconcile_interval = admission
            .as_ref()
            .and_then(QueueAdmissionLayer::reconcile_interval);
        let has_admission_policy = admission.is_some();
        let pending = PolicyQueue::new(profile.clone());
        let queueing_enabled = profile
            .classes()
            .iter()
            .any(PolicyClassConfig::queueing_enabled)
            || has_admission_policy;
        for class in profile.classes() {
            tracing::info!(
                policy_class = class.name,
                ordering = %class.ordering,
                quantum = class.quantum,
                prefill_busy_threshold = ?class.prefill_busy_threshold,
                prefill_busy_threshold_frac = ?class.prefill_busy_threshold_frac,
                "Router policy class configured"
            );
        }
        let overlap_refresh_after = if overlap_scores_refresh.is_some() {
            let configured = read_overlap_refresh_after();
            match configured {
                Some(d) => tracing::info!(
                    "Router queue overlap-score refresh enabled after {:.1}s wait",
                    d.as_secs_f64()
                ),
                None => tracing::info!(
                    "Router queue overlap-score refresh disabled via DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS"
                ),
            }
            configured
        } else {
            None
        };
        let pending_count = Arc::new(AtomicUsize::new(0));
        let pending_isl_tokens = Arc::new(AtomicUsize::new(0));
        let class_counters = Arc::new(
            profile
                .classes()
                .iter()
                .map(|_| ClassQueueCounters {
                    pending_count: AtomicUsize::new(0),
                    pending_isl_tokens: AtomicUsize::new(0),
                    pending_cached_tokens: AtomicUsize::new(0),
                })
                .collect(),
        );
        let (admission_tx, admission_rx) = mpsc::channel(admission_channel_capacity);
        let cleanup = Arc::new(AdmissionCleanup::default());
        let non_max_overlap_selection_observer = Arc::new(OnceLock::new());
        let actor = SchedulerQueueActor {
            pending,
            cleanup: Arc::clone(&cleanup),
            queueing_enabled,
            profile,
            pending_count: Arc::clone(&pending_count),
            pending_isl_tokens: Arc::clone(&pending_isl_tokens),
            class_counters: Arc::clone(&class_counters),
            slots: Arc::clone(&slots),
            workers_with_configs: workers_with_configs.clone(),
            start_time: Instant::now(),
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overlap_refresh_after,
            overloaded_worker_provider,
            available_worker_provider,
            admission,
            released_lifecycle_request_ids: HashSet::new(),
            admission_worker_snapshot: QueueAdmissionWorkerSnapshot::new(0, Vec::new()),
            admission_overloaded_worker_ids: None,
            admission_available_worker_ids: None,
            non_max_overlap_selection_observer: Arc::clone(&non_max_overlap_selection_observer),
        };
        tokio::spawn(actor.run(admission_rx));
        Ok(Self {
            admission_tx,
            cleanup,
            pending_count,
            pending_isl_tokens,
            class_counters,
            slots,
            workers_with_configs,
            queueing_enabled,
            has_admission_policy,
            admission_reconcile_interval,
            supports_overlap_refresh: overlap_refresh_after.is_some(),
            non_max_overlap_selection_observer,
            _marker: PhantomData,
        })
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
> SchedulerQueue<P, C, Sel, NoopOverlapScoresRefresh>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        block_size: u32,
        selector: Sel,
        queue_policy: RouterQueuePolicy,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    ) -> Self {
        Self::new_with_overlap_refresh(
            slots,
            workers_with_configs,
            threshold_frac,
            block_size,
            selector,
            queue_policy,
            prefill_load_estimator,
            None,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_overload_provider(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        block_size: u32,
        selector: Sel,
        queue_policy: RouterQueuePolicy,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    ) -> Self {
        Self::new_with_overlap_refresh(
            slots,
            workers_with_configs,
            threshold_frac,
            block_size,
            selector,
            queue_policy,
            prefill_load_estimator,
            None,
            overloaded_worker_provider,
            None,
        )
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, Sel, RF>
{
    /// Register externally-provided workers in the slot tracker.
    ///
    /// Looks up DP rank/size from the discovery watch channel; defaults to
    /// `(0, 1)` for workers not yet known to discovery.
    pub fn register_workers(&self, worker_ids: &std::collections::HashSet<u64>) {
        let discovery_workers = self.workers_with_configs.borrow();
        for &worker_id in worker_ids {
            let (dp_start, dp_size) = discovery_workers
                .get(&worker_id)
                .map(|runtime_config| {
                    (
                        runtime_config.data_parallel_start_rank(),
                        runtime_config.data_parallel_size(),
                    )
                })
                .unwrap_or((0, 1));
            let range = WorkerDpRange::new(worker_id, dp_start, dp_size);
            if let Err(error) = self.slots.upsert_worker(range) {
                tracing::warn!(worker_id, %error, "Invalid externally-provided worker topology");
            }
        }
    }

    /// Install the observer for admitted selections that sacrifice KV overlap.
    ///
    /// Returns `false` when an observer is already installed.
    pub fn set_non_max_overlap_selection_observer(
        &self,
        observer: NonMaxOverlapSelectionObserver,
    ) -> bool {
        self.non_max_overlap_selection_observer
            .set(observer)
            .is_ok()
    }

    /// Enqueue a new request.
    /// If queueing is disabled or workers have capacity, schedule immediately.
    /// Otherwise park in the pending heap.
    pub async fn enqueue(&self, request: SchedulingRequest) {
        self.enqueue_with_block_hashes(request, None).await;
    }

    pub async fn enqueue_with_block_hashes(
        &self,
        request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) {
        let _ = self
            .enqueue_with_block_hashes_and_lease(request, block_hashes, None)
            .await;
    }

    pub(crate) async fn enqueue_with_block_hashes_and_lease(
        &self,
        mut request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        lease: Option<Box<RequestLifecycleLease>>,
    ) -> Option<Box<RequestLifecycleLease>> {
        // Router arrival is observed once, here, before this request can wait
        // behind a saturated actor channel. Everything downstream derives the
        // class deadline from this instant and never re-reads the clock for it.
        let arrival_at = Instant::now();
        if self.queueing_enabled && lease.is_none() && request.mode.lifecycle_request_id().is_some()
        {
            request.respond(Err(KvSchedulerError::BookingFailed(
                "admission-managed requests must be scheduled through LocalScheduler".to_string(),
            )));
            return None;
        }

        let eligibility = request.eligibility();

        if let Err(error) = eligibility.validate_pinned_worker_allowed() {
            request.respond(Err(error));
            return None;
        }

        let (ack_tx, ack_rx) = oneshot::channel();
        let command = AdmissionCommand::Enqueue {
            request,
            block_hashes: self.prepare_block_hashes_for_refresh(block_hashes),
            arrival_at,
            lease,
            ack_tx,
        };

        if let Err(error) = self.admission_tx.send(command).await {
            let AdmissionCommand::Enqueue { mut request, .. } = error.0 else {
                return None;
            };
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
            return None;
        }

        match ack_rx.await {
            Ok(lease) => lease,
            Err(_) => {
                tracing::warn!("scheduler queue actor dropped enqueue acknowledgement");
                None
            }
        }
    }

    pub(crate) fn new_request_lifecycle_lease(
        &self,
        request_id: Option<&str>,
    ) -> Option<Box<RequestLifecycleLease>> {
        if !self.queueing_enabled {
            return None;
        }
        request_id?;
        Some(Box::new(RequestLifecycleLease {
            cleanup: Arc::clone(&self.cleanup),
            actor_tx: self.admission_tx.clone(),
            request_id: None,
        }))
    }

    /// Select a worker from current scheduler state without entering admission.
    ///
    /// This is for advisory policy probes that must not wait in the router
    /// queue and must not book active scheduler state.
    pub async fn select_without_admission(
        &self,
        request: SchedulingRequest,
    ) -> Result<AdvisorySchedulingResponse, KvSchedulerError> {
        request.eligibility().validate_pinned_worker_allowed()?;

        let (resp_tx, resp_rx) = oneshot::channel();
        let command = AdmissionCommand::SelectWithoutAdmission { request, resp_tx };
        if self.admission_tx.send(command).await.is_err() {
            return Err(KvSchedulerError::SubscriberShutdown);
        }

        resp_rx
            .await
            .map_err(|_| KvSchedulerError::SubscriberShutdown)?
    }

    /// Called on prefill_complete/free. Drains pending requests while workers have capacity.
    /// Each scheduled request updates active_tokens via add_request, so the prefill-busy check
    /// sees fresh state on the next iteration.
    pub async fn update(&self) {
        let _ = self.update_after(None, None, true).await;
    }

    pub(crate) async fn update_worker(&self, worker: WorkerWithDpRank) {
        let _ = self.update_after(Some(worker), None, false).await;
    }

    pub(crate) async fn complete_request(
        &self,
        request_id: &str,
        update_worker: Option<WorkerWithDpRank>,
        expected_worker: Option<WorkerWithDpRank>,
        context_tokens: Option<usize>,
    ) -> bool {
        self.update_after(
            update_worker,
            Some((
                request_id.to_owned(),
                expected_worker,
                AdmissionRequestOutcome::Completed { context_tokens },
            )),
            false,
        )
        .await
    }

    pub(crate) async fn abort_request(
        &self,
        request_id: &str,
        update_worker: Option<WorkerWithDpRank>,
        expected_worker: Option<WorkerWithDpRank>,
    ) -> bool {
        self.update_after(
            update_worker,
            Some((
                request_id.to_owned(),
                expected_worker,
                AdmissionRequestOutcome::Aborted,
            )),
            false,
        )
        .await
    }

    async fn update_after(
        &self,
        worker: Option<WorkerWithDpRank>,
        finished: Option<(String, Option<WorkerWithDpRank>, AdmissionRequestOutcome)>,
        reconcile_admission: bool,
    ) -> bool {
        if !self.queueing_enabled {
            return false;
        }

        let (ack_tx, ack_rx) = oneshot::channel();
        if self
            .admission_tx
            .send(AdmissionCommand::Update {
                worker,
                finished,
                reconcile_admission,
                ack_tx,
            })
            .await
            .is_ok()
        {
            return ack_rx.await.unwrap_or(false);
        }
        false
    }

    /// Number of requests currently parked in the pending queue (lock-free).
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(AtomicOrdering::Relaxed)
    }

    /// Sum of `isl_tokens` for requests currently parked in the pending queue (lock-free).
    pub fn pending_isl_tokens(&self) -> usize {
        self.pending_isl_tokens.load(AtomicOrdering::Relaxed)
    }

    pub fn class_queue_stats(&self, class_index: usize) -> Option<ClassQueueStats> {
        let counters = self.class_counters.get(class_index)?;
        Some(ClassQueueStats {
            pending_count: counters.pending_count.load(AtomicOrdering::Relaxed),
            pending_isl_tokens: counters.pending_isl_tokens.load(AtomicOrdering::Relaxed),
            pending_cached_tokens: counters.pending_cached_tokens.load(AtomicOrdering::Relaxed),
        })
    }

    pub fn supports_overlap_refresh(&self) -> bool {
        self.supports_overlap_refresh
    }

    pub(crate) fn has_admission_policy(&self) -> bool {
        self.has_admission_policy
    }

    pub(crate) fn admission_reconcile_interval(&self) -> Option<Duration> {
        self.admission_reconcile_interval
    }

    fn prepare_block_hashes_for_refresh(
        &self,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) -> Option<Vec<LocalBlockHash>> {
        if !self.supports_overlap_refresh {
            return None;
        }
        block_hashes.filter(|hashes| !hashes.is_empty())
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, Sel, RF>
{
    async fn run(mut self, mut rx: mpsc::Receiver<AdmissionCommand>) {
        let mut commands_since_cleanup = 0usize;
        while let Some(command) = rx.recv().await {
            // Lifecycle releases are only meaningful to the command that
            // observes them, so the record never outlives one command.
            self.released_lifecycle_request_ids.clear();
            let drain_cleanup = self.queueing_enabled && {
                commands_since_cleanup += 1;
                let drain_cleanup = rx.is_empty() || commands_since_cleanup == 256;
                if drain_cleanup {
                    commands_since_cleanup = 0;
                }
                drain_cleanup
            };
            match command {
                AdmissionCommand::Enqueue {
                    request,
                    block_hashes,
                    arrival_at,
                    mut lease,
                    ack_tx,
                } => {
                    let request_id = lease
                        .as_ref()
                        .and_then(|_| request.mode.tracked_request_id().map(str::to_owned));
                    let (enqueue_ready, owns_lifecycle) =
                        self.handle_enqueue(request, block_hashes, arrival_at);
                    let made_ready = enqueue_ready | (drain_cleanup && self.drain_cleanup());
                    if made_ready {
                        self.handle_update(None, false).await;
                    }
                    // Arm the lease only after scheduling and any drain above,
                    // and only if this request still owns its lifecycle. Either
                    // route can dispatch it and then lose it to a failed
                    // selection, booking, or response delivery; a lease armed
                    // for a request the actor already released would let its
                    // cleanup kill a later attempt that reused the ID.
                    if let Some(lease) = lease.as_mut()
                        && owns_lifecycle
                        && let Some(request_id) = request_id
                        && !self.released_lifecycle_request_ids.contains(&request_id)
                    {
                        lease.request_id = Some(request_id);
                    }
                    let _ = ack_tx.send(lease);
                }
                AdmissionCommand::SelectWithoutAdmission { request, resp_tx } => {
                    let result = self.select_without_admission_inner(request, Instant::now());
                    let _ = resp_tx.send(result);
                }
                AdmissionCommand::Update {
                    mut worker,
                    finished,
                    reconcile_admission,
                    ack_tx,
                } => {
                    let terminal_update = finished.is_some();
                    let mut finished_handled = false;
                    if let Some((request_id, expected_worker, outcome)) = finished {
                        let (handled, booked_worker) =
                            self.handle_admission_finished(&request_id, expected_worker, outcome);
                        if handled {
                            worker = worker.or(booked_worker);
                            finished_handled = true;
                        }
                    }
                    if !terminal_update || worker.is_some() || finished_handled {
                        self.handle_update(worker, reconcile_admission).await;
                    }
                    if drain_cleanup && self.drain_cleanup() {
                        self.handle_update(None, false).await;
                    }
                    let _ = ack_tx.send(finished_handled);
                }
                AdmissionCommand::Cleanup => {
                    if self.drain_cleanup() {
                        self.handle_update(None, false).await;
                    }
                }
            }
        }
        self.drain_cleanup();

        let class_counters = Arc::clone(&self.class_counters);
        for entry in self.pending.drain() {
            let class_index = entry.class_index();
            let snapshot = entry.snapshot();
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            let counters = &class_counters[class_index];
            counters.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            counters
                .pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            counters
                .pending_cached_tokens
                .fetch_sub(snapshot.cached_tokens, AtomicOrdering::Relaxed);

            let mut request = entry.into_payload().request;
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
        }

        // Work the admission layer still holds never reached a policy class, so
        // it carries no queue accounting to reverse; it only needs an answer.
        if let Some(admission) = self.admission.take() {
            for queued in admission.into_deferred() {
                let mut request = queued.request;
                request.respond(Err(KvSchedulerError::SubscriberShutdown));
            }
        }
    }

    /// Run one arriving request through the queue-admission layer and, unless
    /// that layer retains it, into the router's scheduling decision.
    ///
    /// `arrival_at` is the router-acceptance instant carried on the command. A
    /// request the router admits directly never gets a queue deadline at all;
    /// for one that has to wait, this instant is the only basis for it, so
    /// actor-channel backlog, deferral, wake-up, and a long queue wait all count
    /// against the class SLO. Load-decay math still reads the current clock.
    ///
    /// Returns whether runnable work is now available and whether the caller's
    /// lease owns this request's lifecycle.
    fn handle_enqueue(
        &mut self,
        mut request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        arrival_at: Instant,
    ) -> (bool, bool) {
        // Queue admission runs first and unconditionally: it sits entirely above
        // policy classes, so it must not be skipped, reordered, or short-circuited
        // by anything the class layer owns — including an unknown class name,
        // which is the class layer's rejection to make at the handoff below.
        let handoff = match self.admit_to_admission_layer(&mut request) {
            Ok(handoff) => handoff,
            Err(()) => return (false, false),
        };
        let queued = QueuedRequest {
            request,
            arrival_at,
            block_hashes,
            admission_id: handoff.admission_id(),
        };
        if let AdmissionHandoff::Defer(id) = handoff {
            // Held above policy classes: no class, no deadline, no class limit,
            // no class statistics, and no pending accounting until the policy
            // releases it.
            tracing::debug!(
                request_id = queued.request.mode.request_id().unwrap_or("unknown"),
                "queue admission deferred request"
            );
            self.admission
                .as_mut()
                .expect("a deferral comes from the admission layer")
                .defer(id, queued);
            return (false, true);
        }

        let mut woken = Vec::new();
        let outcome = self.schedule_handoff_into(queued, &mut woken);
        // Only work released by a lifecycle event can need a drain here. A fresh
        // arrival that was queued is queued precisely because it cannot run yet,
        // and one that was admitted has already been dispatched.
        let made_ready = self.admit_woken(woken);
        match outcome {
            ScheduleOutcome::Admitted { owns_lifecycle } => (made_ready, owns_lifecycle),
            ScheduleOutcome::Queued => (made_ready, true),
            ScheduleOutcome::Rejected => (made_ready, false),
        }
    }

    /// Offer one arriving request to the queue-admission layer.
    ///
    /// `Err` means the layer refused the request and it has already been
    /// answered. A request with no lifecycle identity, or a scheduler with no
    /// admission policy, passes straight through as an unmanaged handoff.
    fn admit_to_admission_layer(
        &mut self,
        request: &mut SchedulingRequest,
    ) -> Result<AdmissionHandoff, ()> {
        if self.admission.is_none() {
            return Ok(AdmissionHandoff::Admit(None));
        }
        let Some(request_id) = request.mode.lifecycle_request_id().map(str::to_owned) else {
            return Ok(AdmissionHandoff::Admit(None));
        };
        if !self
            .admission
            .as_mut()
            .expect("checked as some")
            .begin_lifecycle(&request_id)
        {
            request.respond(Err(KvSchedulerError::BookingFailed(format!(
                "request ID {request_id} is already managed by queue admission"
            ))));
            return Err(());
        }

        self.refresh_admission_worker_snapshot();
        let workers = self.workers_with_configs.borrow();
        let routing_eligibility = request.eligibility();
        let is_worker_eligible = |worker| {
            routing_eligibility
                .validate_worker_rank(&workers, worker)
                .is_ok()
        };
        Ok(self.admission.as_mut().expect("checked as some").decide(
            &request_id,
            request.isl_tokens,
            request.session_context.as_ref(),
            &self.admission_worker_snapshot,
            request.pinned_worker,
            request.allowed_worker_ids.as_ref(),
            request.routing_constraints.has_hard_constraints(),
            &is_worker_eligible,
        ))
    }

    /// The one scheduling decision every request handed over by the queue-admission
    /// layer reaches.
    ///
    /// A bypassed request, a request the policy made ready, and a request the
    /// policy woke from deferral all arrive here and are indistinguishable from
    /// this point on. This decision is the router's, not the admission policy's
    /// and not the class queue's: it selects the policy class and then chooses
    /// between direct admission and queue storage exactly as the router always
    /// has — a class that has queueing enabled and either a runnable backlog of
    /// its own or no unbusy eligible worker stores the request; anything else
    /// goes straight to worker selection.
    ///
    /// The clock is read here, at the handoff itself. A custom admission
    /// policy's callback runs before this point and can take arbitrarily long,
    /// so a time sampled before that callback could let work that is already
    /// past its class deadline through the queue Admission gate.
    ///
    /// Work released by a lifecycle event this produces is appended to `woken`
    /// rather than scheduled here, which keeps wake chains iterative.
    fn schedule_handoff_into(
        &mut self,
        queued: QueuedRequest,
        woken: &mut Vec<QueuedRequest>,
    ) -> ScheduleOutcome {
        let now = Instant::now();
        let Some(class_index) = self.resolve_class_index(&queued.request) else {
            let policy_class = requested_policy_class(&queued.request)
                .unwrap_or_default()
                .to_string();
            tracing::debug!(policy_class, "rejecting unknown router policy class");
            let mut request = queued.request;
            self.abort_lifecycle_into(&request, woken);
            request.respond(Err(KvSchedulerError::UnknownPolicyClass { policy_class }));
            return ScheduleOutcome::Rejected;
        };

        let class = self.profile.class(class_index);
        let should_queue = self.should_queue(class_index, class, || {
            self.all_workers_prefill_busy(class, queued.request.eligibility(), now)
        });
        if !should_queue {
            // Direct admission. The request never enters queue storage, so it
            // spends no class limit, takes no deficit-round-robin turn, and does
            // not pass the queue Admission gate.
            let QueuedRequest {
                request,
                admission_id,
                ..
            } = queued;
            let owns_lifecycle = self.admit_one_into(request, now, admission_id, woken);
            return ScheduleOutcome::Admitted { owns_lifecycle };
        }

        // Resolve the whole scheduling key once, including this request's
        // absolute class deadline. The Admission gate and the queue entry both
        // read this value; neither recomputes `arrival + slo`.
        let arrival = QueueArrival::new(
            queued.arrival_at,
            queued
                .arrival_at
                .saturating_duration_since(self.start_time)
                .as_secs_f64(),
            queued.request.priority_jump,
            queued.request.strict_priority,
            class,
        );

        // Storage is required, so this is the class-queue Admission gate.
        // Requests admitted directly above never reach it.
        //
        // Read the clock again here rather than reusing `now`: class resolution
        // and the eligible-worker busy scan run in between, and the scan is
        // O(workers). A request that crossed its deadline during that work has
        // missed it, and admitting it would put already-late work into storage.
        let admission_now = Instant::now();
        if let Some(overdue) = overdue_by(arrival.deadline(), admission_now) {
            let error = self.deadline_error(class_index, DeadlineStage::Admission, overdue);
            let mut request = queued.request;
            self.abort_lifecycle_into(&request, woken);
            tracing::debug!(
                request_id = request.mode.request_id().unwrap_or("unknown"),
                policy_class = %error.policy_class,
                overdue_ms = error.overdue_ms,
                "rejecting request past its policy class deadline before queue storage"
            );
            request.respond(Err(KvSchedulerError::QueueDeadlineExceeded(error)));
            return ScheduleOutcome::Rejected;
        }

        let snapshot = self.snapshot_for(&queued.request);
        tracing::debug!(
            policy_class = self.profile.class(class_index).name,
            "queueing request"
        );
        let placement = queued
            .request
            .pinned_worker
            .map_or(WorkerPlacement::Any, WorkerPlacement::Exact);
        let worker_count = self.workers_with_configs.borrow().len();
        if let Err((rejection, queued)) = self.pending.enqueue(
            class_index,
            worker_count,
            snapshot,
            arrival,
            placement,
            queued,
        ) {
            let mut request = queued.request;
            self.abort_lifecycle_into(&request, woken);
            request.respond(Err(KvSchedulerError::QueueRejected(rejection)));
            return ScheduleOutcome::Rejected;
        }
        self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
        self.pending_isl_tokens
            .fetch_add(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        self.add_class_counters(class_index, snapshot);
        ScheduleOutcome::Queued
    }

    fn should_queue(
        &self,
        class_index: usize,
        class: &PolicyClassConfig,
        all_workers_busy: impl FnOnce() -> bool,
    ) -> bool {
        // Preserve backlog anti-bypass and lazily avoid worker scans when an
        // earlier condition already decides admission.
        class.queueing_enabled() && (self.pending.has_backlog(class_index) || all_workers_busy())
    }

    /// Push every payload the admission layer released through the same router
    /// scheduling decision, iteratively.
    ///
    /// A released request can be admitted directly, queued, or rejected, and a
    /// rejection reports one terminal abort that the policy may answer by
    /// releasing more work, so the batch doubles as the worklist. It terminates
    /// because deferred storage only ever shrinks during the pass.
    ///
    /// Returns whether any of them became runnable queued work, which is the
    /// only outcome that gives a later poll something new to find.
    fn admit_woken(&mut self, woken: Vec<QueuedRequest>) -> bool {
        let mut made_ready = false;
        let mut batch = woken;
        let mut next = Vec::new();
        while !batch.is_empty() {
            for queued in batch.drain(..) {
                made_ready |= matches!(
                    self.schedule_handoff_into(queued, &mut next),
                    ScheduleOutcome::Queued
                );
            }
            std::mem::swap(&mut batch, &mut next);
        }
        made_ready
    }

    /// The policy class this request names, or `None` for "no preference".
    ///
    /// Flat classes resolve from the requested name alone. Only an absent or
    /// empty value means "no preference"; every other value is matched exactly,
    /// so a padded name is unknown rather than silently normalized onto a class
    /// it does not spell.
    fn resolve_class_index(&self, request: &SchedulingRequest) -> Option<usize> {
        self.profile
            .resolve_class_index(requested_policy_class(request))
    }

    fn snapshot_for(&self, request: &SchedulingRequest) -> QueueSnapshot {
        let workers = self.workers_with_configs.borrow();
        Self::snapshot_for_with(request, &workers)
    }

    fn snapshot_for_with(
        request: &SchedulingRequest,
        workers: &HashMap<WorkerId, C>,
    ) -> QueueSnapshot {
        // Cache overlap is sampled once and reused for queue limits, DRR cost,
        // and counters. Class resolution does not use it.
        let context = SchedulingContext::new(request, workers);
        QueueSnapshot::new(request.isl_tokens, context.best_cached_tokens())
    }

    fn refresh_admission_worker_snapshot(&mut self) {
        let topology_changed = self.admission_worker_snapshot.generation() == 0
            || self.workers_with_configs.has_changed().unwrap_or(false);
        let overloaded_worker_ids = self
            .overloaded_worker_provider
            .as_ref()
            .and_then(|provider| provider());
        let available_worker_ids = self
            .available_worker_provider
            .as_ref()
            .and_then(|provider| provider());
        let overloaded_unchanged = match (
            self.admission_overloaded_worker_ids.as_ref(),
            overloaded_worker_ids.as_ref(),
        ) {
            (Some(previous), Some(current)) => {
                Arc::ptr_eq(previous, current) || previous.as_ref() == current.as_ref()
            }
            (None, None) => true,
            _ => false,
        };
        let available_unchanged = match (
            self.admission_available_worker_ids.as_ref(),
            available_worker_ids.as_ref(),
        ) {
            (Some(previous), Some(current)) => {
                Arc::ptr_eq(previous, current) || previous.as_ref() == current.as_ref()
            }
            (None, None) => true,
            _ => false,
        };
        let availability_changed = !overloaded_unchanged || !available_unchanged;

        if !topology_changed && !availability_changed {
            return;
        }

        let mut admission_workers = if topology_changed {
            let workers = self.workers_with_configs.borrow_and_update();
            let mut admission_workers = Vec::new();
            for (&worker_id, config) in workers.iter() {
                let capacity_tokens =
                    config
                        .total_kv_blocks()
                        .filter(|blocks| *blocks > 0)
                        .map(|blocks| {
                            let tokens = blocks
                                .saturating_mul(u64::from(self.block_size))
                                .saturating_add(
                                    config.native_offloading_capacity_tokens().unwrap_or(0),
                                );
                            usize::try_from(tokens).unwrap_or(usize::MAX)
                        });
                let start = config.data_parallel_start_rank();
                let end = start.saturating_add(config.data_parallel_size());
                for dp_rank in start..end {
                    admission_workers.push(QueueAdmissionWorker::new(
                        WorkerWithDpRank::new(worker_id, dp_rank),
                        capacity_tokens,
                        true,
                    ));
                }
            }
            admission_workers
        } else {
            self.admission_worker_snapshot.workers().to_vec()
        };

        for worker in &mut admission_workers {
            let worker_id = worker.worker().worker_id;
            let available = available_worker_ids
                .as_ref()
                .is_none_or(|workers| workers.contains(&worker_id))
                && overloaded_worker_ids
                    .as_ref()
                    .is_none_or(|workers| !workers.contains(&worker_id));
            *worker =
                QueueAdmissionWorker::new(worker.worker(), worker.capacity_tokens(), available);
        }

        let generation = self
            .admission_worker_snapshot
            .generation()
            .wrapping_add(1)
            .max(1);
        self.admission_worker_snapshot =
            QueueAdmissionWorkerSnapshot::new(generation, admission_workers);
        self.admission_overloaded_worker_ids = overloaded_worker_ids;
        self.admission_available_worker_ids = available_worker_ids;
    }

    /// Reject every request past its class deadline, exactly once each: reverse
    /// queue accounting, report one terminal lifecycle event for
    /// admission-managed work, and answer the caller.
    ///
    /// Aborting managed work can release deferred requests. Those repeat the
    /// actor's ordinary scheduling decision, so what happens to one that aged
    /// while it was parked depends on that decision and not on the fact that it
    /// was parked: it meets the Admission gate only if the decision requires
    /// storage, and a stale release can still be admitted directly when
    /// capacity is idle. Either way there is no wake-specific stage.
    fn reject_expired(
        &mut self,
        expired: &mut Vec<PolicyQueueEntry<QueuedRequest>>,
        now: Instant,
        stage: DeadlineStage,
    ) -> bool {
        let mut woken = Vec::new();
        for entry in expired.drain(..) {
            self.reject_one_expired(entry, now, stage, &mut woken);
        }
        self.admit_woken(woken)
    }

    /// Reject one expired request: reverse its queue accounting, report one
    /// terminal lifecycle event for admission-managed work, and answer it.
    fn reject_one_expired(
        &mut self,
        entry: PolicyQueueEntry<QueuedRequest>,
        now: Instant,
        stage: DeadlineStage,
        woken: &mut Vec<QueuedRequest>,
    ) {
        let class_index = entry.class_index();
        self.subtract_pending_counters(class_index, entry.snapshot());
        let overdue = overdue_by(entry.deadline(), now).unwrap_or(Duration::ZERO);
        let error = self.deadline_error(class_index, stage, overdue);
        let queued = entry.into_payload();
        let mut request = queued.request;
        self.abort_lifecycle_into(&request, woken);
        tracing::debug!(
            request_id = request.mode.request_id().unwrap_or("unknown"),
            policy_class = %error.policy_class,
            %stage,
            overdue_ms = error.overdue_ms,
            "rejecting request past its policy class deadline"
        );
        request.respond(Err(KvSchedulerError::QueueDeadlineExceeded(error)));
    }

    /// Release one rejected request's admission lifecycle state.
    ///
    /// Lifecycle ownership, not the admission ID, decides cleanup: a bypassed
    /// request carries no admission ID but still holds lifecycle state. Work
    /// the abort releases is appended to `woken` rather than admitted here,
    /// which keeps wake chains iterative in the caller.
    fn abort_lifecycle_into(
        &mut self,
        request: &SchedulingRequest,
        woken: &mut Vec<QueuedRequest>,
    ) {
        let Some(request_id) = request.mode.lifecycle_request_id() else {
            return;
        };
        self.release_lifecycle_into(request_id, woken);
    }

    /// Release one request's admission lifecycle state and admit anything the
    /// resulting terminal abort released.
    fn abort_admission_request(&mut self, request_id: &str) -> bool {
        let mut woken = Vec::new();
        self.release_lifecycle_into(request_id, &mut woken);
        self.admit_woken(woken)
    }

    /// Record that the actor gave up lifecycle ownership of `request_id` and
    /// release the admission layer's state for it.
    ///
    /// The record is what keeps a lease honest. An arrival takes lifecycle
    /// ownership, but scheduling it can lose that ownership again inside the
    /// same actor command — worker selection, booking, and response delivery can
    /// all fail, on the direct route as readily as from a drain — and the
    /// `Enqueue` arm arms its lease only afterwards. Handing back a lease armed
    /// for a request the actor already released would let the cleanup it
    /// eventually fires kill a later attempt that reused the ID.
    fn release_lifecycle_into(&mut self, request_id: &str, woken: &mut Vec<QueuedRequest>) {
        self.released_lifecycle_request_ids
            .insert(request_id.to_owned());
        if let Some(admission) = self.admission.as_mut() {
            admission.abort(request_id, woken);
        }
    }

    fn deadline_error(
        &self,
        class_index: usize,
        stage: DeadlineStage,
        overdue: Duration,
    ) -> QueueDeadlineExceeded {
        let class = self.profile.class(class_index);
        QueueDeadlineExceeded {
            policy_class: class.name.clone(),
            stage,
            slo_ms: class.slo().unwrap_or_default().as_millis() as u64,
            overdue_ms: overdue.as_millis() as u64,
        }
    }

    fn drain_cleanup(&mut self) -> bool {
        let dirty = self.cleanup.drain();
        if dirty.is_empty() {
            return false;
        }

        let unmanaged_request_ids: HashSet<&str> = dirty
            .iter()
            .map(|cleanup| cleanup.request_id.as_str())
            .collect();
        let is_abandoned = |queued: &QueuedRequest| {
            queued
                .request
                .mode
                .tracked_request_id()
                .is_some_and(|request_id| unmanaged_request_ids.contains(request_id))
        };

        // Retract every abandoned request from both layers, and reverse its
        // queue and class accounting, before any lifecycle abort reaches the
        // policy. An abort can release deferred work straight into class
        // admission, and that work must be measured against the accounting this
        // cleanup leaves behind rather than the stale counts it found: a class
        // at its limit only because of requests this pass is retracting would
        // otherwise reject the very work the cancellation made room for.
        if let Some(admission) = self.admission.as_mut() {
            admission.retain_deferred(|queued| !is_abandoned(queued));
        }
        let mut removed_ready_head = false;
        for class_index in 0..self.profile.classes().len() {
            let (removed, class_head_removed) =
                self.pending.take_if_in_class(class_index, is_abandoned);
            removed_ready_head |= class_head_removed;
            for entry in removed {
                self.subtract_pending_counters(class_index, entry.snapshot());
            }
        }

        let mut made_ready = false;
        for cleanup in &dirty {
            let request_id = &cleanup.request_id;
            if self.slots.request_worker(request_id).is_some() {
                if let Err(error) = self.slots.free(request_id, Instant::now()) {
                    tracing::error!(%request_id, %error, "Failed to release dropped scheduler booking");
                }
                made_ready = true;
            }
        }
        for cleanup in &dirty {
            made_ready |= self.abort_admission_request(&cleanup.request_id);
        }
        made_ready || (removed_ready_head && self.has_dispatchable_ready_head())
    }

    /// Relay one terminal response-path outcome to the admission layer.
    ///
    /// Returns whether the layer owned the request and, when it did, the worker
    /// it was booked on.
    fn handle_admission_finished(
        &mut self,
        request_id: &str,
        expected_worker: Option<WorkerWithDpRank>,
        outcome: AdmissionRequestOutcome,
    ) -> (bool, Option<WorkerWithDpRank>) {
        let mut woken = Vec::new();
        let Some(admission) = self.admission.as_mut() else {
            return (false, None);
        };
        let Some(worker) = admission.finish(request_id, expected_worker, outcome, &mut woken)
        else {
            return (false, None);
        };
        // The request is handled either way: whether the event released other
        // work is a separate signal and must not mask the acknowledgement.
        self.admit_woken(woken);
        (true, worker)
    }

    fn has_dispatchable_ready_head(&self) -> bool {
        let active_tokens = self.slots.active_tokens(Instant::now());
        let configs = self.workers_with_configs.borrow();
        self.pending.any_ready_head(|_, class, queued| {
            !Self::all_workers_prefill_busy_with(
                &active_tokens,
                &configs,
                class,
                queued.request.eligibility(),
            )
        })
    }

    fn subtract_pending_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
        self.pending_isl_tokens
            .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        self.subtract_class_counters(class_index, snapshot);
    }

    /// Re-poll every class head after scheduler state changed.
    ///
    /// Each class holds one due-time queue and only its head is tested, so
    /// there is no per-worker readiness index to refresh. `_worker` names the
    /// worker whose capacity changed and is deliberately unused: an
    /// undispatchable head is simply retested on this global re-poll.
    async fn handle_update(
        &mut self,
        _worker: Option<WorkerWithDpRank>,
        reconcile_admission: bool,
    ) {
        if reconcile_admission && self.admission.is_some() {
            self.refresh_admission_worker_snapshot();
            let mut woken = Vec::new();
            // Take the worker-snapshot borrow only for the event itself, so it
            // has ended before released work reaches the scheduling decision.
            self.admission.as_mut().expect("checked as some").on_event(
                QueueAdmissionEvent::Reconcile {
                    snapshot: &self.admission_worker_snapshot,
                },
                &mut woken,
            );
            // Whatever this callback released repeats the ordinary scheduling
            // decision, which reads the clock for itself after the callback
            // returned. A slow reconcile is therefore charged to the class SLO
            // of anything the decision sends to queue storage, while a release
            // the decision can dispatch outright is unaffected by it.
            self.admit_woken(woken);
        }
        if !self.pending.has_ready() {
            return;
        }

        // Continuation draining stays actor-local; never self-send through the
        // bounded command channel while processing an update.
        let mut expired = Vec::new();
        loop {
            let decay_now = Instant::now();
            let active_tokens = self.slots.active_tokens(decay_now);
            let popped = {
                let configs = self.workers_with_configs.borrow();
                self.pending
                    .pop_next(decay_now, &mut expired, |_, class, queued| {
                        // TODO: This preserves head-of-line blocking within each policy
                        // class. A blocked constrained head can stall later entries in
                        // that class until a bounded non-HOL policy is introduced.
                        !Self::all_workers_prefill_busy_with(
                            &active_tokens,
                            &configs,
                            class,
                            queued.request.eligibility(),
                        )
                    })
            };
            // Every polled class shed its expired heads above; rejecting them can
            // wake deferred work, so an empty poll is only final once nothing new
            // became runnable.
            let woke_deferred = !expired.is_empty()
                && self.reject_expired(&mut expired, decay_now, DeadlineStage::Dispatch);
            let Some(mut popped) = popped else {
                if woke_deferred {
                    continue;
                }
                break;
            };
            let snapshot = popped.snapshot();
            let current_pending_count = self.pending_count.load(AtomicOrdering::Relaxed);
            debug_assert!(
                current_pending_count > 0,
                "pending_count underflow on queue drain"
            );
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            let current_pending_isl_tokens = self.pending_isl_tokens.load(AtomicOrdering::Relaxed);
            debug_assert!(
                current_pending_isl_tokens >= snapshot.raw_isl_tokens,
                "pending_isl_tokens underflow: pending={} request_isl_tokens={}",
                current_pending_isl_tokens,
                snapshot.raw_isl_tokens
            );
            self.pending_isl_tokens
                .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
            self.subtract_class_counters(popped.class_index(), snapshot);
            let queued = popped.payload_mut();
            // NOTE: Overlap refresh is expected to be very short. We intentionally
            // accept load crossing the class threshold during this await: busy
            // thresholds guide admission, not reservation. This differs from main
            // to avoid reversing counters, heap state, and charged DRR credit.
            let refreshed = refresh_overlap(
                self.overlap_scores_refresh.as_deref(),
                self.overlap_refresh_after,
                queued.block_hashes.as_deref(),
                queued.request.retain_router_hint_chain,
                queued.arrival_at,
                decay_now,
            )
            .await;
            let wait_ms = queued.arrival_at.elapsed().as_millis() as u64;
            if let Some(snapshot) = refreshed {
                tracing::info!(
                    request_id = queued.request.mode.request_id().unwrap_or("unknown"),
                    wait_ms,
                    "refreshed overlap scores after long queue wait"
                );
                queued.request.overlap = snapshot.overlap;
                queued.request.router_hint_candidates = if queued.request.retain_router_hint_chain {
                    snapshot.router_hint_candidates
                } else {
                    None
                };
            }
            let admit_now = Instant::now();
            let class_index = popped.class_index();
            let class = self.profile.class(class_index);
            let queued = popped.into_payload();
            let admission_id = queued.admission_id;
            let request = queued.request;
            tracing::debug!(
                policy_class = class.name,
                "scheduling request from pending queue"
            );
            self.admit_one(request, admit_now, admission_id);
        }
    }

    fn select_worker_for_request(
        &self,
        request: &mut SchedulingRequest,
        decay_now: Instant,
    ) -> Result<SelectedWorkerForRequest, KvSchedulerError> {
        request.worker_loads = self
            .slots
            .project_worker_loads(request.token_seq.as_deref(), decay_now);

        {
            let workers = self.workers_with_configs.borrow();
            let overloaded_worker_ids = self
                .overloaded_worker_provider
                .as_ref()
                .and_then(|provider| provider());
            let available_worker_ids = self
                .available_worker_provider
                .as_ref()
                .and_then(|provider| provider());
            let eligibility = request
                .eligibility_with_overloaded(overloaded_worker_ids.as_deref())
                .with_available_workers(available_worker_ids.as_deref());
            self.selector
                .select_worker(&workers, request, eligibility, self.block_size)
                .map(|selection| {
                    let non_max_overlap_selection = if request.mode.is_tracked()
                        && self.non_max_overlap_selection_observer.get().is_some()
                    {
                        non_max_overlap_selection(
                            &workers,
                            request,
                            eligibility,
                            selection.worker,
                            selection.effective_overlap_blocks,
                        )
                    } else {
                        None
                    };
                    let config = workers
                        .get(&selection.worker.worker_id)
                        .expect("selected worker config must exist");
                    let selected_worker_tiers = request
                        .overlap
                        .selected_worker_tiers(selection.worker, config);
                    let worker_load = request.worker_load_for(selection.worker);
                    let selected_worker_load = AdvisoryWorkerLoad {
                        active_prefill_tokens: worker_load.active_prefill_tokens,
                        prefill_token_capacity: config
                            .max_num_batched_tokens()
                            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS)
                            as usize,
                        total_kv_blocks: config.total_kv_blocks().map(|blocks| blocks as usize),
                    };
                    SelectedWorkerForRequest {
                        selection,
                        selected_worker_tiers,
                        selected_worker_load,
                        non_max_overlap_selection,
                    }
                })
        }
    }

    fn select_without_admission_inner(
        &self,
        mut request: SchedulingRequest,
        decay_now: Instant,
    ) -> Result<AdvisorySchedulingResponse, KvSchedulerError> {
        let selected = self.select_worker_for_request(&mut request, decay_now)?;
        let target_cached_prefix_blocks =
            target_cached_prefix_blocks(&request, selected.selection.worker);

        Ok(AdvisorySchedulingResponse {
            selected_worker_load: selected.selected_worker_load,
            response: SchedulingResponse {
                best_worker: selected.selection.worker,
                effective_overlap_blocks: selected.selection.effective_overlap_blocks,
                cached_tokens: selected.selection.cached_tokens,
                selected_worker_tiers: selected.selected_worker_tiers,
                target_cached_prefix_blocks,
                router_hint_candidates: request.router_hint_candidates.take(),
                potential_decode_blocks: selected.selection.potential_decode_blocks,
            },
        })
    }

    /// [`Self::admit_one_into`] for the queued route: the drain in
    /// [`Self::handle_update`] calls this for a request that waited and won its
    /// deficit-round-robin turn, and it owns the wake pass for whatever that
    /// dispatch releases.
    ///
    /// The direct route does not come through here. It is already inside a wake
    /// pass of its own, so [`Self::schedule_handoff_into`] calls
    /// [`Self::admit_one_into`] and lets its caller drive the released work.
    fn admit_one(
        &mut self,
        request: SchedulingRequest,
        decay_now: Instant,
        admission_id: Option<QueueAdmissionId>,
    ) -> (bool, bool) {
        let mut woken = Vec::new();
        let owns_lifecycle = self.admit_one_into(request, decay_now, admission_id, &mut woken);
        (self.admit_woken(woken), owns_lifecycle)
    }

    /// The terminal dispatch pipeline both of the router's routes share, and
    /// the only place worker selection happens: compute projected load ->
    /// select worker -> book tracked state -> respond. Neither route is
    /// privileged over the other; they differ only in what decided to call it.
    ///
    /// Work released by the lifecycle events this dispatch reports is appended
    /// to `woken` instead of being scheduled here. A released request may itself
    /// dispatch directly and release more work, so recursing would make the
    /// chain's depth the number of deferred requests; the caller keeps it
    /// iterative. Returns only whether *this* request owns its lifecycle —
    /// whatever the wake released is the caller's to report.
    fn admit_one_into(
        &mut self,
        mut request: SchedulingRequest,
        decay_now: Instant,
        admission_id: Option<QueueAdmissionId>,
        woken: &mut Vec<QueuedRequest>,
    ) -> bool {
        let selected = match self.select_worker_for_request(&mut request, decay_now) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                self.abort_lifecycle_into(&request, woken);
                request.respond(Err(e));
                return false;
            }
        };

        let target_cached_prefix_blocks =
            target_cached_prefix_blocks(&request, selected.selection.worker);
        let response = SchedulingResponse {
            best_worker: selected.selection.worker,
            effective_overlap_blocks: selected.selection.effective_overlap_blocks,
            cached_tokens: selected.selection.cached_tokens,
            selected_worker_tiers: selected.selected_worker_tiers,
            target_cached_prefix_blocks,
            router_hint_candidates: request.router_hint_candidates.take(),
            potential_decode_blocks: selected.selection.potential_decode_blocks,
        };
        let non_max_overlap_selection = selected.non_max_overlap_selection;

        if !request.mode.is_tracked() {
            debug_assert!(admission_id.is_none());
            request.respond(Ok(response));
            return false;
        }

        let request_id = request
            .mode
            .tracked_request_id()
            .expect("tracked mode always has a request ID")
            .to_string();

        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            selected.selection.cached_tokens,
            request.track_prefill_tokens,
        );

        let sequence_request = SequenceRequest {
            request_id: request_id.clone(),
            token_sequence: request.token_seq.take(),
            track_prefill_tokens: request.track_prefill_tokens,
            expected_output_tokens: request.expected_output_tokens,
            prefill_load_hint,
            worker: selected.selection.worker,
            lora_name: request.lora_name.take(),
        };
        let worker = selected.selection.worker;
        let owns_lifecycle = self.book_and_respond(
            request,
            sequence_request,
            response,
            non_max_overlap_selection,
        );
        if owns_lifecycle {
            self.record_admission_dispatch_into(&request_id, admission_id, worker, woken);
        } else {
            // Response delivery lost its race, so the booking rolled back. The
            // request owns lifecycle state whether or not the policy managed it.
            self.release_lifecycle_into(&request_id, woken);
        }
        owns_lifecycle
    }

    /// Tell the admission layer which worker this request was booked on.
    ///
    /// The booking is recorded for every request the layer tracks, including
    /// bypassed work; `admission_id` decides only whether the policy is told.
    /// Anything the report releases is appended to `woken` for the caller.
    fn record_admission_dispatch_into(
        &mut self,
        request_id: &str,
        admission_id: Option<QueueAdmissionId>,
        worker: WorkerWithDpRank,
        woken: &mut Vec<QueuedRequest>,
    ) {
        let Some(admission) = self.admission.as_mut() else {
            return;
        };
        admission.dispatched(request_id, admission_id, worker, woken);
    }

    /// A closed receiver means the actor-owned request was abandoned before
    /// booking, so there is nothing to install. Otherwise booking precedes the
    /// response: once delivery succeeds, the response channel no longer tracks
    /// request lifetime and the caller must install its RAII cleanup owner. If
    /// delivery loses that race, roll back the booking here.
    fn book_and_respond(
        &self,
        mut request: SchedulingRequest,
        sequence_request: SequenceRequest,
        response: SchedulingResponse,
        non_max_overlap_selection: Option<NonMaxOverlapSelection>,
    ) -> bool {
        if request.response_is_closed() {
            tracing::debug!(
                request_id = %sequence_request.request_id,
                "Skipping scheduler booking for cancelled request"
            );
            return false;
        }

        let request_id = sequence_request.request_id.clone();
        if let Err(error) = self.slots.add_request(sequence_request, Instant::now()) {
            tracing::warn!(%request_id, %error, "Failed to book scheduler state");
            request.respond(Err(KvSchedulerError::BookingFailed(error.to_string())));
            return false;
        }

        if request.respond(Ok(response)) {
            if let Some(selection) = non_max_overlap_selection {
                self.dispatch_non_max_overlap_selection(request_id, selection);
            }
            return true;
        }

        tracing::debug!(%request_id, "Rolling back undelivered scheduler booking");
        if let Err(error) = self.slots.free(&request_id, Instant::now()) {
            tracing::error!(%request_id, %error, "Failed to roll back scheduler booking");
        }
        false
    }

    fn dispatch_non_max_overlap_selection(
        &self,
        request_id: String,
        selection: NonMaxOverlapSelection,
    ) {
        let Some(observer) = self.non_max_overlap_selection_observer.get() else {
            return;
        };
        let observer = Arc::clone(observer);
        let _observer_task = tokio::task::spawn_blocking(move || {
            observer(&request_id, selection);
        });
    }

    fn prefill_load_hint_for(
        &self,
        isl_tokens: usize,
        cached_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let effective_isl = effective_prefill_tokens(isl_tokens, cached_tokens);
        if effective_isl == 0 {
            return None;
        }
        let prefix = isl_tokens - effective_isl;

        let expected_prefill_duration = match &self.prefill_load_estimator {
            Some(estimator) => match estimator.predict_prefill_duration(1, effective_isl, prefix) {
                Ok(expected_prefill_duration) => Some(expected_prefill_duration),
                Err(error) => {
                    tracing::warn!(
                        effective_isl,
                        prefix,
                        "failed to predict prefill duration for active load tracking: {error}"
                    );
                    None
                }
            },
            None => None,
        };

        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: effective_isl,
            expected_prefill_duration,
        })
    }

    /// Check if all eligible workers are prefill-busy based on threshold.
    /// When `pinned_worker` is `Some`, only that exact worker/rank is considered.
    /// Otherwise when `allowed` is `Some`, only those worker IDs are considered;
    /// otherwise all registered workers are checked.
    /// Returns false when no eligible workers exist so the request falls
    /// through to `admit_one`, which returns a proper `NoEndpoints` error.
    fn all_workers_prefill_busy(
        &self,
        class: &PolicyClassConfig,
        eligibility: RoutingEligibility<'_>,
        decay_now: Instant,
    ) -> bool {
        let active_tokens = self.slots.active_tokens(decay_now);
        let configs = self.workers_with_configs.borrow();
        Self::all_workers_prefill_busy_with(&active_tokens, &configs, class, eligibility)
    }

    fn all_workers_prefill_busy_with(
        active_tokens: &HashMap<crate::protocols::WorkerWithDpRank, usize>,
        configs: &HashMap<WorkerId, C>,
        class: &PolicyClassConfig,
        eligibility: RoutingEligibility<'_>,
    ) -> bool {
        if let Some(worker) = eligibility.pinned_worker() {
            let Ok(config) = eligibility.validate_worker_rank(configs, worker) else {
                return false;
            };

            let max_batched = config
                .max_num_batched_tokens()
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
            let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
            return class.worker_is_busy(tokens, max_batched);
        }

        let mut checked_any = false;
        let has_available = eligibility.any_eligible_worker_rank(configs, |worker, config| {
            checked_any = true;
            let max_batched = config
                .max_num_batched_tokens()
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
            let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
            !class.worker_is_busy(tokens, max_batched)
        });

        checked_any && !has_available
    }

    fn add_class_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        let counters = &self.class_counters[class_index];
        counters.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
        counters
            .pending_isl_tokens
            .fetch_add(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        counters
            .pending_cached_tokens
            .fetch_add(snapshot.cached_tokens, AtomicOrdering::Relaxed);
    }

    fn subtract_class_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        let counters = &self.class_counters[class_index];
        counters.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
        counters
            .pending_isl_tokens
            .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        counters
            .pending_cached_tokens
            .fetch_sub(snapshot.cached_tokens, AtomicOrdering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex as StdMutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use rustc_hash::FxHashMap;
    use tokio::sync::{Barrier, watch};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::protocols::{
        ActiveLoad, ActiveSequenceEvent, ExternalSequenceBlockHash, WorkerSelectionResult,
        WorkerWithDpRank,
    };
    use crate::router_hint::RouterHintRootCandidates;
    use crate::scheduling::types::{KvSchedulerError, ScheduleMode};
    use crate::scheduling::{LocalScheduler, OverlapSignals, ScheduleRequest};
    use crate::scheduling::{
        QueueAdmissionDecision, QueueAdmissionPolicy, QueueAdmissionRequest, RefreshedOverlap,
        RouterPolicyConfig,
    };
    use crate::sequences::{ActiveSequencesMultiWorker, SequencePublisher};
    use crate::test_utils::{NoopSequencePublisher, SimpleWorkerConfig};
    use crate::{DefaultWorkerSelector, WorkerSelector};

    fn decay_now() -> Instant {
        Instant::now()
    }

    struct FixedPrefillLoadEstimator {
        duration: Duration,
    }

    impl PrefillLoadEstimator for FixedPrefillLoadEstimator {
        fn predict_prefill_duration(
            &self,
            _batch_size: usize,
            _effective_isl: usize,
            _prefix: usize,
        ) -> anyhow::Result<Duration> {
            Ok(self.duration)
        }
    }

    type SchedulingResponseReceiver =
        tokio::sync::oneshot::Receiver<Result<SchedulingResponse, KvSchedulerError>>;

    struct DropResponseOnLoadPublisher {
        response_rx: Arc<StdMutex<Option<SchedulingResponseReceiver>>>,
    }

    impl SequencePublisher for DropResponseOnLoadPublisher {
        fn enqueue_event(&self, _event: ActiveSequenceEvent) -> anyhow::Result<()> {
            Ok(())
        }

        fn publish_load(&self, _load: ActiveLoad) {
            self.response_rx.lock().unwrap().take();
        }

        fn observe_load(&self, _: &WorkerWithDpRank, _: &str, _: usize, _: usize) {}
    }

    #[derive(Default)]
    struct SelectorRendezvous {
        arrivals: StdMutex<usize>,
        cv: Condvar,
    }

    impl SelectorRendezvous {
        fn wait_for_peer(&self) {
            let mut arrivals = self.arrivals.lock().unwrap();
            *arrivals += 1;

            if *arrivals == 1 {
                let _ = self
                    .cv
                    .wait_timeout(arrivals, Duration::from_millis(100))
                    .unwrap();
                return;
            }

            self.cv.notify_all();
        }
    }

    #[derive(Clone)]
    struct MinDecodeSelector {
        rendezvous: Option<Arc<SelectorRendezvous>>,
    }

    impl WorkerSelector<SimpleWorkerConfig> for MinDecodeSelector {
        fn select_worker(
            &self,
            workers: &HashMap<WorkerId, SimpleWorkerConfig>,
            request: &SchedulingRequest,
            eligibility: RoutingEligibility<'_>,
            block_size: u32,
        ) -> Result<WorkerSelectionResult, KvSchedulerError> {
            if let Some(rendezvous) = &self.rendezvous {
                rendezvous.wait_for_peer();
            }

            let mut best_worker = None;
            eligibility.for_each_eligible_worker_rank(workers, |worker, _| {
                let load = request.worker_load_for(worker);
                let potential_prefill_tokens = if request.track_prefill_tokens {
                    load.active_prefill_tokens
                        .saturating_add(effective_prefill_tokens(
                            request.isl_tokens,
                            request.effective_cached_tokens_for(worker),
                        ))
                } else {
                    0
                };
                let potential_decode_blocks = load.potential_decode_blocks();
                let key = (
                    potential_prefill_tokens,
                    potential_decode_blocks,
                    worker.worker_id,
                    worker.dp_rank,
                );
                if best_worker.is_none_or(|(_, best_key)| key < best_key) {
                    best_worker = Some((worker, key));
                }
            });

            let Some((worker, _)) = best_worker else {
                return Err(KvSchedulerError::NoEndpoints);
            };

            Ok(WorkerSelectionResult {
                worker,
                required_blocks: request.request_blocks(block_size),
                effective_overlap_blocks: request.effective_overlap_blocks_for(worker),
                cached_tokens: request.effective_cached_tokens_for(worker),
                potential_decode_blocks: request
                    .potential_decode_blocks_after_admission(worker, block_size),
            })
        }
    }

    #[derive(Default)]
    struct AdmissionPolicyState {
        deferred: StdMutex<Option<QueueAdmissionId>>,
        dispatched: AtomicUsize,
        reconciled: AtomicUsize,
        completed: AtomicUsize,
        completed_context_tokens: AtomicUsize,
    }

    struct DeferOncePolicy {
        state: Arc<AdmissionPolicyState>,
    }

    impl QueueAdmissionPolicy for DeferOncePolicy {
        fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
            assert_eq!(request.request_id(), "policy-request");
            *self.state.deferred.lock().unwrap() = Some(request.id());
            QueueAdmissionDecision::Defer
        }

        fn on_event(&mut self, event: QueueAdmissionEvent<'_>, ready: &mut Vec<QueueAdmissionId>) {
            match event {
                QueueAdmissionEvent::Reconcile { .. } => {
                    self.state.reconciled.fetch_add(1, Ordering::Relaxed);
                    if let Some(id) = self.state.deferred.lock().unwrap().take() {
                        ready.push(id);
                    }
                }
                QueueAdmissionEvent::Dispatched { .. } => {
                    self.state.dispatched.fetch_add(1, Ordering::Relaxed);
                }
                QueueAdmissionEvent::Completed {
                    request_id,
                    context_tokens,
                } => {
                    assert_eq!(request_id, "policy-request");
                    self.state.completed.fetch_add(1, Ordering::Relaxed);
                    self.state
                        .completed_context_tokens
                        .store(context_tokens.unwrap_or_default(), Ordering::Relaxed);
                }
                _ => {}
            }
        }

        fn reconcile_interval(&self) -> Option<Duration> {
            Some(Duration::from_millis(5))
        }
    }

    struct ReadyPolicy {
        state: Arc<AdmissionPolicyState>,
    }

    impl QueueAdmissionPolicy for ReadyPolicy {
        fn admit(&mut self, _request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
            QueueAdmissionDecision::Ready
        }

        fn on_event(&mut self, event: QueueAdmissionEvent<'_>, _ready: &mut Vec<QueueAdmissionId>) {
            if matches!(event, QueueAdmissionEvent::Completed { .. }) {
                self.state.completed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    struct EligibilityPolicy {
        observed: Arc<AtomicBool>,
    }

    impl QueueAdmissionPolicy for EligibilityPolicy {
        fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
            assert_eq!(request.workers().len(), 2);
            let mut eligible_workers = Vec::new();
            request.for_each_eligible_worker(|worker| {
                eligible_workers.push(worker.worker());
            });
            assert_eq!(eligible_workers, vec![WorkerWithDpRank::new(1, 0)]);
            self.observed.store(true, Ordering::Relaxed);
            QueueAdmissionDecision::Bypass
        }
    }

    /// Bypasses `bypass-*` requests, keeps `ready-*` requests runnable, defers
    /// everything else, and releases whatever it deferred the first time it sees
    /// an abort. That makes one dispatch-stage rejection wake deferred work.
    #[derive(Default)]
    struct WakeDeferredOnAbort {
        deferred: Vec<QueueAdmissionId>,
    }

    impl QueueAdmissionPolicy for WakeDeferredOnAbort {
        fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
            if request.request_id().starts_with("bypass-") {
                return QueueAdmissionDecision::Bypass;
            }
            if request.request_id().starts_with("ready-") {
                return QueueAdmissionDecision::Ready;
            }
            self.deferred.push(request.id());
            QueueAdmissionDecision::Defer
        }

        fn on_event(&mut self, event: QueueAdmissionEvent<'_>, ready: &mut Vec<QueueAdmissionId>) {
            if matches!(event, QueueAdmissionEvent::Aborted { .. }) {
                ready.append(&mut self.deferred);
            }
        }
    }

    /// Defers `defer-*` requests and, on an abort, burns wall-clock inside the
    /// callback before releasing what it holds. A clock read before the callback
    /// would let the released work through a class deadline it has since missed.
    struct SlowWakeOnAbort {
        deferred: Vec<QueueAdmissionId>,
        stall: Duration,
    }

    impl QueueAdmissionPolicy for SlowWakeOnAbort {
        fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
            if request.request_id().starts_with("defer-") {
                self.deferred.push(request.id());
                return QueueAdmissionDecision::Defer;
            }
            QueueAdmissionDecision::Ready
        }

        fn on_event(&mut self, event: QueueAdmissionEvent<'_>, ready: &mut Vec<QueueAdmissionId>) {
            if matches!(event, QueueAdmissionEvent::Aborted { .. }) && !self.deferred.is_empty() {
                std::thread::sleep(self.stall);
                ready.append(&mut self.deferred);
            }
        }
    }

    /// Returns the decision named by the request-ID prefix and counts both
    /// `admit` calls and terminal aborts, so a test can prove where queue
    /// admission sits relative to policy-class selection. It releases whatever
    /// it deferred on the first abort it sees.
    struct CountingDecisionPolicy {
        admits: Arc<AtomicUsize>,
        aborts: Arc<AtomicUsize>,
        deferred: Vec<QueueAdmissionId>,
    }

    impl QueueAdmissionPolicy for CountingDecisionPolicy {
        fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
            self.admits.fetch_add(1, Ordering::Relaxed);
            if request.request_id().starts_with("bypass-") {
                QueueAdmissionDecision::Bypass
            } else if request.request_id().starts_with("defer-") {
                self.deferred.push(request.id());
                QueueAdmissionDecision::Defer
            } else {
                QueueAdmissionDecision::Ready
            }
        }

        fn on_event(&mut self, event: QueueAdmissionEvent<'_>, ready: &mut Vec<QueueAdmissionId>) {
            if matches!(event, QueueAdmissionEvent::Aborted { .. }) {
                self.aborts.fetch_add(1, Ordering::Relaxed);
                ready.append(&mut self.deferred);
            }
        }
    }

    /// Counts `admit` calls and terminal aborts, and burns wall-clock inside
    /// `admit` for request IDs starting with `slow-`, so a request can cross a
    /// short class SLO between the captured arrival and the class-queue
    /// admission check.
    #[derive(Default)]
    struct CountingSlowPolicy {
        admits: Arc<AtomicUsize>,
        aborts: Arc<AtomicUsize>,
        stall: Duration,
    }

    impl QueueAdmissionPolicy for CountingSlowPolicy {
        fn admit(&mut self, request: QueueAdmissionRequest<'_>) -> QueueAdmissionDecision {
            self.admits.fetch_add(1, Ordering::Relaxed);
            if request.request_id().starts_with("slow-") {
                std::thread::sleep(self.stall);
            }
            if request.request_id().starts_with("bypass-") {
                return QueueAdmissionDecision::Bypass;
            }
            QueueAdmissionDecision::Ready
        }

        fn on_event(&mut self, event: QueueAdmissionEvent<'_>, _ready: &mut Vec<QueueAdmissionId>) {
            if matches!(event, QueueAdmissionEvent::Aborted { .. }) {
                self.aborts.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    struct AdmissionPolicySelector {
        selector: MinDecodeSelector,
        policy: Option<Box<dyn QueueAdmissionPolicy>>,
    }

    impl WorkerSelector<SimpleWorkerConfig> for AdmissionPolicySelector {
        fn select_worker(
            &self,
            workers: &HashMap<WorkerId, SimpleWorkerConfig>,
            request: &SchedulingRequest,
            eligibility: RoutingEligibility<'_>,
            block_size: u32,
        ) -> Result<WorkerSelectionResult, KvSchedulerError> {
            self.selector
                .select_worker(workers, request, eligibility, block_size)
        }

        fn take_admission_policy(&mut self) -> Option<Box<dyn QueueAdmissionPolicy>> {
            self.policy.take()
        }
    }

    fn make_queue(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let (queue, slots, _tx) =
            make_queue_with_sender(num_workers, block_size, isl, threshold_frac, None);
        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_profile_and_selector<
        Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static,
    >(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        profile: PolicyProfile,
        selector: Sel,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));
        let configs = (0..num_workers as u64)
            .map(|id| {
                (
                    id,
                    SimpleWorkerConfig {
                        max_num_batched_tokens: Some(isl as u64),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let (_cfg_tx, cfg_rx) = watch::channel(configs);
        let queue = Arc::new(
            SchedulerQueue::new_with_policy_profile(
                Arc::clone(&slots),
                cfg_rx,
                profile,
                block_size,
                selector,
                None,
                None,
                None,
                None,
            )
            .unwrap(),
        );
        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_custom_selector<Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static>(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        selector: Sel,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (_cfg_tx, cfg_rx) = watch::channel(configs);

        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            threshold_frac,
            block_size,
            selector,
            RouterQueuePolicy::Fcfs,
            None,
        ));

        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_local_scheduler_with_custom_selector<
        Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static,
    >(
        num_workers: usize,
        selector: Sel,
        recheck_interval: Duration,
    ) -> (
        LocalScheduler<NoopSequencePublisher, SimpleWorkerConfig, Sel>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        CancellationToken,
    ) {
        let dp_ranges = (0..num_workers as u64)
            .map(|worker_id| (worker_id, (0, 1)))
            .collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            16,
            dp_ranges,
            false,
            0,
            "test",
        ));
        let workers = (0..num_workers as u64)
            .map(|worker_id| {
                (
                    worker_id,
                    SimpleWorkerConfig {
                        total_kv_blocks: Some(1024),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let (_workers_tx, workers_rx) = watch::channel(workers);
        let cancellation = CancellationToken::new();
        let scheduler = LocalScheduler::new_without_overlap_refresh(
            Arc::clone(&slots),
            workers_rx,
            None,
            16,
            selector,
            RouterQueuePolicy::Fcfs,
            None,
            recheck_interval,
            true,
            cancellation.clone(),
            "test",
            false,
        );
        (scheduler, slots, cancellation)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_sender(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (cfg_tx, cfg_rx) = watch::channel(configs);

        let selector = DefaultWorkerSelector::new(None, "test");
        let queue = Arc::new(SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            threshold_frac,
            block_size,
            selector,
            RouterQueuePolicy::Fcfs,
            prefill_load_estimator,
        ));

        (queue, slots, cfg_tx)
    }

    fn policy_profile(yaml: &str) -> PolicyProfile {
        RouterPolicyConfig::from_yaml(yaml)
            .unwrap()
            .resolve_profile(None, None, crate::config::RouterQueuePolicy::Fcfs)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_profile(
        num_workers: usize,
        block_size: u32,
        max_num_batched_tokens: usize,
        profile: PolicyProfile,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let (queue, slots, _cfg_tx) = make_queue_with_profile_and_sender(
            num_workers,
            block_size,
            max_num_batched_tokens,
            profile,
        );
        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_profile_and_sender(
        num_workers: usize,
        block_size: u32,
        max_num_batched_tokens: usize,
        profile: PolicyProfile,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
        watch::Sender<HashMap<u64, SimpleWorkerConfig>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));
        let configs = (0..num_workers as u64)
            .map(|id| {
                (
                    id,
                    SimpleWorkerConfig {
                        max_num_batched_tokens: Some(max_num_batched_tokens as u64),
                        ..Default::default()
                    },
                )
            })
            .collect();
        let (cfg_tx, cfg_rx) = watch::channel(configs);
        let queue = Arc::new(
            SchedulerQueue::new_with_policy_profile(
                Arc::clone(&slots),
                cfg_rx,
                profile,
                block_size,
                DefaultWorkerSelector::new(None, "test"),
                None,
                None,
                None,
                None,
            )
            .unwrap(),
        );
        (queue, slots, cfg_tx)
    }

    fn make_queue_with_providers(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        available_worker_provider: Option<WorkerAvailabilityProvider>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (_cfg_tx, cfg_rx) = watch::channel(configs);

        let selector = DefaultWorkerSelector::new(None, "test");
        let queue = Arc::new(SchedulerQueue::new_with_overlap_refresh(
            Arc::clone(&slots),
            cfg_rx,
            None,
            block_size,
            selector,
            RouterQueuePolicy::Fcfs,
            None,
            None,
            overloaded_worker_provider,
            available_worker_provider,
        ));

        (queue, slots)
    }

    struct CountingRefresher {
        calls: AtomicUsize,
        last_retain_router_hint_chain: AtomicBool,
        response: RefreshedOverlap,
    }

    #[async_trait]
    impl OverlapScoresRefresh for CountingRefresher {
        async fn refresh(
            &self,
            _block_hashes: &[LocalBlockHash],
            retain_router_hint_chain: bool,
        ) -> Option<RefreshedOverlap> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.last_retain_router_hint_chain
                .store(retain_router_hint_chain, Ordering::Relaxed);
            Some(self.response.clone())
        }
    }

    struct BlockingRefresher {
        calls: AtomicUsize,
        started: tokio::sync::Notify,
        release: tokio::sync::Notify,
        response: RefreshedOverlap,
    }

    impl BlockingRefresher {
        fn new(response: RefreshedOverlap) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                started: tokio::sync::Notify::new(),
                release: tokio::sync::Notify::new(),
                response,
            }
        }

        async fn wait_for_calls(&self, target: usize) {
            while self.calls.load(Ordering::Relaxed) < target {
                self.started.notified().await;
            }
        }

        fn release_one(&self) {
            self.release.notify_one();
        }
    }

    #[async_trait]
    impl OverlapScoresRefresh for BlockingRefresher {
        async fn refresh(
            &self,
            _block_hashes: &[LocalBlockHash],
            _retain_router_hint_chain: bool,
        ) -> Option<RefreshedOverlap> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.started.notify_one();
            self.release.notified().await;
            Some(self.response.clone())
        }
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_refresher(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        refresher: Arc<CountingRefresher>,
    ) -> (
        Arc<
            SchedulerQueue<
                NoopSequencePublisher,
                SimpleWorkerConfig,
                DefaultWorkerSelector,
                CountingRefresher,
            >,
        >,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (_cfg_tx, cfg_rx) = watch::channel(configs);

        let queue = Arc::new(SchedulerQueue::new_with_overlap_refresh(
            Arc::clone(&slots),
            cfg_rx,
            threshold_frac,
            block_size,
            DefaultWorkerSelector::new(None, "test"),
            RouterQueuePolicy::Fcfs,
            None,
            Some(refresher),
            None,
            None,
        ));

        (queue, slots)
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_blocking_refresher(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        refresher: Arc<BlockingRefresher>,
        admission_channel_capacity: usize,
    ) -> (
        Arc<
            SchedulerQueue<
                NoopSequencePublisher,
                SimpleWorkerConfig,
                DefaultWorkerSelector,
                BlockingRefresher,
            >,
        >,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let dp_range: HashMap<u64, (u32, u32)> =
            (0..num_workers as u64).map(|id| (id, (0, 1))).collect();
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));

        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        for id in 0..num_workers as u64 {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        let (_cfg_tx, cfg_rx) = watch::channel(configs);

        let queue = Arc::new(
            SchedulerQueue::new_with_policy_profile_and_capacity(
                Arc::clone(&slots),
                cfg_rx,
                PolicyProfile::synthetic(threshold_frac, crate::config::RouterQueuePolicy::Fcfs),
                block_size,
                DefaultWorkerSelector::new(None, "test"),
                None,
                Some(refresher),
                None,
                None,
                admission_channel_capacity,
            )
            .unwrap(),
        );

        (queue, slots)
    }

    fn make_request(
        request_id: &str,
        isl_tokens: usize,
    ) -> (
        SchedulingRequest,
        tokio::sync::oneshot::Receiver<
            Result<SchedulingResponse, crate::scheduling::types::KvSchedulerError>,
        >,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = SchedulingRequest {
            mode: ScheduleMode::Tracked {
                request_id: request_id.to_string(),
            },
            token_seq: None,
            isl_tokens,
            overlap: OverlapSignals::default(),
            router_hint_candidates: None,
            retain_router_hint_chain: false,
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };
        (req, rx)
    }

    #[tokio::test]
    async fn custom_admission_policy_defers_dispatch_and_observes_completion() {
        let state = Arc::new(AdmissionPolicyState::default());
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(DeferOncePolicy {
                state: Arc::clone(&state),
            })),
        };
        let (queue, slots) = make_queue_with_custom_selector(1, 16, 128, None, selector);
        assert_eq!(
            queue.admission_reconcile_interval(),
            Some(Duration::from_millis(5))
        );
        let (mut request, mut response_rx) = make_request("policy-request", 32);
        request.mode = ScheduleMode::TrackedWithLifecycle {
            request_id: "policy-request".to_owned(),
        };
        let lease = queue.new_request_lifecycle_lease(Some("policy-request"));
        let mut lease = queue
            .enqueue_with_block_hashes_and_lease(request, None, lease)
            .await;

        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut response_rx)
                .await
                .is_err()
        );
        queue.update().await;
        let response = tokio::time::timeout(Duration::from_secs(1), &mut response_rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        lease.as_mut().unwrap().disarm();
        assert_eq!(state.dispatched.load(Ordering::Relaxed), 1);
        let reconciled_before_completion = state.reconciled.load(Ordering::Relaxed);

        slots
            .free(&"policy-request".to_owned(), Instant::now())
            .unwrap();
        queue
            .complete_request(
                "policy-request",
                Some(response.best_worker),
                Some(response.best_worker),
                Some(48),
            )
            .await;
        assert_eq!(state.completed.load(Ordering::Relaxed), 1);
        assert_eq!(state.completed_context_tokens.load(Ordering::Relaxed), 48);
        assert_eq!(
            state.reconciled.load(Ordering::Relaxed),
            reconciled_before_completion,
            "completion must not trigger a full policy reconciliation"
        );
    }

    #[tokio::test]
    async fn custom_admission_policy_rejects_duplicate_request_id_while_deferred() {
        let state = Arc::new(AdmissionPolicyState::default());
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(DeferOncePolicy {
                state: Arc::clone(&state),
            })),
        };
        let (queue, slots) = make_queue_with_custom_selector(1, 16, 128, None, selector);

        let (mut first, mut first_response_rx) = make_request("policy-request", 32);
        first.mode = ScheduleMode::TrackedWithLifecycle {
            request_id: "policy-request".to_owned(),
        };
        let first_lease = queue.new_request_lifecycle_lease(Some("policy-request"));
        let mut first_lease = queue
            .enqueue_with_block_hashes_and_lease(first, None, first_lease)
            .await;
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut first_response_rx)
                .await
                .is_err()
        );

        let (mut duplicate, duplicate_response_rx) = make_request("policy-request", 32);
        duplicate.mode = ScheduleMode::TrackedWithLifecycle {
            request_id: "policy-request".to_owned(),
        };
        let duplicate_lease = queue.new_request_lifecycle_lease(Some("policy-request"));
        let _duplicate_lease = queue
            .enqueue_with_block_hashes_and_lease(duplicate, None, duplicate_lease)
            .await;
        let error = duplicate_response_rx.await.unwrap().unwrap_err();
        assert!(matches!(
            error,
            KvSchedulerError::BookingFailed(message)
                if message.contains("already managed by queue admission")
        ));

        queue.update().await;
        let response = first_response_rx.await.unwrap().unwrap();
        first_lease.as_mut().unwrap().disarm();
        slots
            .free(&"policy-request".to_owned(), Instant::now())
            .unwrap();
        assert!(
            queue
                .complete_request(
                    "policy-request",
                    Some(response.best_worker),
                    Some(response.best_worker),
                    Some(48),
                )
                .await
        );
        assert_eq!(state.completed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn custom_admission_policy_observes_request_eligibility() {
        let observed = Arc::new(AtomicBool::new(false));
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(EligibilityPolicy {
                observed: Arc::clone(&observed),
            })),
        };
        let (queue, slots) = make_queue_with_custom_selector(2, 16, 128, None, selector);
        let (mut request, response_rx) = make_request("eligibility-request", 32);
        request.mode = ScheduleMode::TrackedWithLifecycle {
            request_id: "eligibility-request".to_owned(),
        };
        request.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        let lease = queue.new_request_lifecycle_lease(Some("eligibility-request"));
        let mut lease = queue
            .enqueue_with_block_hashes_and_lease(request, None, lease)
            .await;
        let response = response_rx.await.unwrap().unwrap();
        lease.as_mut().unwrap().disarm();

        assert!(observed.load(Ordering::Relaxed));
        assert_eq!(response.best_worker, WorkerWithDpRank::new(1, 0));
        slots
            .free(&"eligibility-request".to_owned(), Instant::now())
            .unwrap();
    }

    #[tokio::test]
    async fn custom_admission_policy_observes_completion_after_worker_removal() {
        let state = Arc::new(AdmissionPolicyState::default());
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(ReadyPolicy {
                state: Arc::clone(&state),
            })),
        };
        let (scheduler, slots, cancellation) =
            make_local_scheduler_with_custom_selector(1, selector, Duration::from_secs(60));
        let worker = WorkerWithDpRank::new(0, 0);
        let response = scheduler
            .schedule_request(ScheduleRequest {
                mode: ScheduleMode::TrackedWithLifecycle {
                    request_id: "removed-worker-request".to_owned(),
                },
                token_seq: None,
                block_hashes: None,
                isl_tokens: 32,
                lora_name: None,
                expected_output_tokens: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: crate::protocols::RoutingConstraints::default(),
                router_config_override: None,
                priority_jump: 0.0,
                strict_priority: 0,
                policy_class: None,
                session_context: None,
                overlap: OverlapSignals::default(),
                router_hint_candidates: None,
                retain_router_hint_chain: false,
                shared_cache_hits: None,
            })
            .await
            .unwrap();
        assert_eq!(response.best_worker, worker);

        slots.reconcile_workers(Vec::new()).unwrap();
        scheduler
            .complete_if_worker("removed-worker-request", worker, 48)
            .await
            .unwrap();

        assert_eq!(state.completed.load(Ordering::Relaxed), 1);
        cancellation.cancel();
    }

    #[tokio::test]
    async fn custom_admission_policy_controls_reconcile_cadence() {
        let state = Arc::new(AdmissionPolicyState::default());
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(DeferOncePolicy {
                state: Arc::clone(&state),
            })),
        };
        let (scheduler, _slots, cancellation) =
            make_local_scheduler_with_custom_selector(1, selector, Duration::from_secs(60));
        tokio::time::sleep(Duration::from_millis(1)).await;

        let response = tokio::time::timeout(
            Duration::from_millis(100),
            scheduler.schedule_request(ScheduleRequest {
                mode: ScheduleMode::TrackedWithLifecycle {
                    request_id: "policy-request".to_owned(),
                },
                token_seq: None,
                block_hashes: None,
                isl_tokens: 32,
                lora_name: None,
                expected_output_tokens: None,
                pinned_worker: None,
                allowed_worker_ids: None,
                routing_constraints: crate::protocols::RoutingConstraints::default(),
                router_config_override: None,
                priority_jump: 0.0,
                strict_priority: 0,
                policy_class: None,
                session_context: None,
                overlap: OverlapSignals::default(),
                router_hint_candidates: None,
                retain_router_hint_chain: false,
                shared_cache_hits: None,
            }),
        )
        .await
        .expect("policy cadence should override the 60 second host interval")
        .unwrap();

        scheduler
            .complete_if_worker("policy-request", response.best_worker, 48)
            .await
            .unwrap();
        assert_eq!(state.dispatched.load(Ordering::Relaxed), 1);
        cancellation.cancel();
    }

    #[test]
    fn non_max_overlap_selection_ignores_ties_pins_and_ineligible_workers() {
        let workers = HashMap::from([
            (0, SimpleWorkerConfig::default()),
            (1, SimpleWorkerConfig::default()),
        ]);
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let (mut request, _rx) = make_request("locality-exclusions", 64);
        request
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 8.0), (worker1, 8.0)]);
        assert!(
            non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 8.0)
                .is_none()
        );

        request
            .overlap
            .effective_overlap_blocks
            .insert(worker1, 2.0);
        request.pinned_worker = Some(worker1);
        assert!(
            non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 2.0)
                .is_none()
        );

        request.pinned_worker = None;
        request.allowed_worker_ids = Some(HashSet::from([worker1.worker_id]));
        assert!(
            non_max_overlap_selection(&workers, &request, request.eligibility(), worker1, 2.0)
                .is_none()
        );
    }

    #[tokio::test]
    async fn scheduler_observer_receives_non_max_overlap_selection() {
        let (queue, _slots) = make_queue_with_custom_selector(
            2,
            16,
            64,
            None,
            MinDecodeSelector { rendezvous: None },
        );
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let (observer_tx, mut observer_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(queue.set_non_max_overlap_selection_observer(Arc::new(
            move |request_id, event| {
                observer_tx
                    .send((request_id.to_string(), event))
                    .expect("observer receiver should remain open");
            }
        )));
        let (mut request, response_rx) = make_request("locality-response", 64);
        request
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 1.0), (worker1, 4.0)]);

        queue.enqueue(request).await;
        let response = response_rx.await.unwrap().unwrap();

        assert_eq!(response.best_worker, worker0);
        let event = tokio::time::timeout(Duration::from_secs(1), observer_rx.recv())
            .await
            .expect("observer did not run")
            .expect("observer channel closed");
        assert_eq!(event.1.overlap_blocks_lost(), 3.0);
        assert_eq!(
            event,
            (
                "locality-response".to_string(),
                NonMaxOverlapSelection {
                    selected_worker: worker0,
                    highest_overlap_worker: worker1,
                    highest_overlap_blocks: 4.0,
                    selected_overlap_blocks: 1.0,
                },
            )
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_observer_does_not_block_actor() {
        let (queue, _slots) = make_queue_with_custom_selector(
            2,
            16,
            64,
            None,
            MinDecodeSelector { rendezvous: None },
        );
        let observer_gate = Arc::new((StdMutex::new(false), Condvar::new()));
        let callback_gate = Arc::clone(&observer_gate);
        let (started_tx, mut started_rx) = tokio::sync::mpsc::unbounded_channel();
        let (finished_tx, mut finished_rx) = tokio::sync::mpsc::unbounded_channel();
        assert!(queue.set_non_max_overlap_selection_observer(Arc::new(
            move |_request_id, _event| {
                started_tx.send(()).unwrap();
                let (released, wake) = &*callback_gate;
                let mut released = released.lock().unwrap();
                while !*released {
                    released = wake.wait(released).unwrap();
                }
                finished_tx.send(()).unwrap();
            }
        )));

        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let (mut first, first_rx) = make_request("blocking-observer", 64);
        first
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 1.0), (worker1, 4.0)]);
        let first_queue = Arc::clone(&queue);
        let first_enqueue = tokio::spawn(async move {
            first_queue.enqueue(first).await;
        });

        tokio::time::timeout(Duration::from_secs(1), started_rx.recv())
            .await
            .expect("observer did not start")
            .expect("observer start channel closed");

        let (second, second_rx) = make_request("actor-remains-responsive", 64);
        tokio::time::timeout(Duration::from_secs(1), queue.enqueue(second))
            .await
            .expect("observer blocked the scheduler actor");

        let (released, wake) = &*observer_gate;
        *released.lock().unwrap() = true;
        wake.notify_one();

        first_enqueue.await.unwrap();
        first_rx.await.unwrap().unwrap();
        second_rx.await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(1), finished_rx.recv())
            .await
            .expect("observer did not finish")
            .expect("observer finish channel closed");
    }

    #[tokio::test]
    async fn scheduler_observer_ignores_advisory_and_abandoned_requests() {
        let (queue, _slots) = make_queue_with_custom_selector(
            2,
            16,
            64,
            None,
            MinDecodeSelector { rendezvous: None },
        );
        let worker0 = WorkerWithDpRank::new(0, 0);
        let worker1 = WorkerWithDpRank::new(1, 0);
        let observed = Arc::new(StdMutex::new(Vec::new()));
        let observer_events = Arc::clone(&observed);
        assert!(queue.set_non_max_overlap_selection_observer(Arc::new(
            move |request_id, event| {
                observer_events
                    .lock()
                    .unwrap()
                    .push((request_id.to_string(), event));
            }
        )));

        let (mut advisory, _advisory_rx) = make_request("locality-advisory", 64);
        advisory
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 1.0), (worker1, 4.0)]);
        let response = queue.select_without_admission(advisory).await.unwrap();
        assert_eq!(response.response.best_worker, worker0);

        let (mut abandoned, abandoned_rx) = make_request("locality-abandoned", 64);
        abandoned
            .overlap
            .effective_overlap_blocks
            .extend([(worker0, 1.0), (worker1, 4.0)]);
        drop(abandoned_rx);
        queue.enqueue(abandoned).await;

        assert!(observed.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn disabled_queueing_has_no_cancellation_lease() {
        let (queue, _slots) = make_queue(1, 16, 64, None);

        assert!(
            queue
                .new_request_lifecycle_lease(Some("default-path"))
                .is_none()
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cancelled_pending_request_is_not_booked() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

        let (first, first_rx) = make_request("first", isl);
        queue.enqueue(first).await;
        first_rx
            .await
            .expect("first response sender dropped")
            .expect("first request should be scheduled");

        let (cancelled, cancelled_rx) = make_request("cancelled", isl);
        queue.enqueue(cancelled).await;
        assert_eq!(queue.pending_count(), 1);
        drop(cancelled_rx);

        slots.free(&"first".to_string(), decay_now()).unwrap();
        queue.update().await;

        assert_eq!(queue.pending_count(), 0);
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unknown_terminal_update_does_not_drain_pending_queue() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

        let (active, active_rx) = make_request("active", isl);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (queued, mut queued_rx) = make_request("queued", isl);
        queue.enqueue(queued).await;
        assert_eq!(queue.pending_count(), 1);

        slots.free(&"active".to_owned(), decay_now()).unwrap();
        assert!(!queue.complete_request("unknown", None, None, None).await);
        assert!(matches!(
            queued_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));

        queue.update().await;
        queued_rx.await.unwrap().unwrap();
        slots.free(&"queued".to_owned(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn plain_tracked_terminal_update_wakes_worker_with_admission_policy() {
        let state = Arc::new(AdmissionPolicyState::default());
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(ReadyPolicy { state })),
        };
        let isl = 512;
        let (queue, slots) = make_queue_with_custom_selector(1, 16, isl, Some(0.0), selector);
        let worker = WorkerWithDpRank::new(0, 0);

        let (active, active_rx) = make_request("active", isl);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (queued, queued_rx) = make_request("queued", isl);
        queue.enqueue(queued).await;
        assert_eq!(queue.pending_count(), 1);

        slots.free(&"active".to_owned(), decay_now()).unwrap();
        assert!(
            !queue
                .complete_request("active", Some(worker), None, None)
                .await,
            "plain tracked request must not report a policy lifecycle event"
        );
        tokio::time::timeout(Duration::from_secs(1), queued_rx)
            .await
            .expect("freed worker did not wake pending request")
            .unwrap()
            .unwrap();

        slots.free(&"queued".to_owned(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test]
    async fn dropped_legacy_lease_retracts_pending_request_immediately() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

        let (first, first_rx) = make_request("legacy-first", isl);
        queue.enqueue(first).await;
        first_rx.await.unwrap().unwrap();

        let (cancelled, cancelled_rx) = make_request("legacy-cancelled", isl);
        let lease = queue
            .new_request_lifecycle_lease(Some("legacy-cancelled"))
            .unwrap();
        let lease = queue
            .enqueue_with_block_hashes_and_lease(cancelled, None, Some(lease))
            .await
            .unwrap();
        assert_eq!(queue.pending_count(), 1);

        drop(cancelled_rx);
        drop(lease);
        tokio::time::timeout(Duration::from_secs(1), async {
            while queue.pending_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("legacy lease did not retract pending request");

        slots.free(&"legacy-first".to_owned(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    /// The no-policy-config fallback profile keeps its pre-existing ordering:
    /// strict-priority tier first, then the `--router-queue-policy` score.
    async fn test_strict_priority_drains_before_policy_score() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

        let (first, first_rx) = make_request("first", isl);
        queue.enqueue(first).await;
        first_rx.await.unwrap().unwrap();

        let (mut low, mut low_rx) = make_request("low", isl);
        low.priority_jump = 10_000.0;
        queue.enqueue(low).await;

        let (mut high, high_rx) = make_request("high", isl);
        high.strict_priority = 1;
        queue.enqueue(high).await;
        assert_eq!(queue.pending_count(), 2);

        slots.free(&"first".to_string(), decay_now()).unwrap();
        queue.update().await;

        let high_response = high_rx.await.unwrap().unwrap();
        assert_eq!(high_response.best_worker, WorkerWithDpRank::new(0, 0));
        assert!(
            low_rx.try_recv().is_err(),
            "lower strict priority should remain queued"
        );

        slots.free(&"high".to_string(), decay_now()).unwrap();
        queue.update().await;
        low_rx.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0);

        slots.free(&"low".to_string(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    /// A configured class orders only by `(deadline, enqueue sequence)`, so the
    /// per-request priority hints that steer the fallback profile do not
    /// reorder it.
    async fn configured_class_order_ignores_per_request_priority_hints() {
        let profile = policy_profile(
            r#"
default_policy_class: fixed
policy_classes:
  - name: fixed
    slo_ms: 600000
    quantum: 1000000
    prefill_busy_threshold: 0
"#,
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 512, profile);

        let (first, first_rx) = make_request("first", 512);
        queue.enqueue(first).await;
        first_rx.await.unwrap().unwrap();

        let (early, early_rx) = make_request("early", 512);
        queue.enqueue(early).await;

        let (mut boosted, mut boosted_rx) = make_request("boosted", 512);
        boosted.priority_jump = 10_000.0;
        boosted.strict_priority = 1;
        queue.enqueue(boosted).await;
        assert_eq!(queue.pending_count(), 2);

        slots
            .mark_prefill_completed(&"first".to_string(), decay_now())
            .unwrap();
        slots.free(&"first".to_string(), decay_now()).unwrap();
        queue.update().await;

        early_rx
            .await
            .unwrap()
            .expect("the earliest deadline dispatches first");
        assert!(
            boosted_rx.try_recv().is_err(),
            "a later arrival cannot jump a deadline-ordered class on priority alone"
        );

        slots
            .mark_prefill_completed(&"early".to_string(), decay_now())
            .unwrap();
        slots.free(&"early".to_string(), decay_now()).unwrap();
        queue.update().await;
        boosted_rx.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_failed_response_delivery_rolls_back_booking() {
        let isl = 512;
        let response_rx = Arc::new(StdMutex::new(None));
        let publisher = DropResponseOnLoadPublisher {
            response_rx: Arc::clone(&response_rx),
        };
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            publisher,
            16,
            HashMap::from([(0, (0, 1))]),
            false,
            0,
            "test",
        ));
        let (_cfg_tx, cfg_rx) = watch::channel(HashMap::from([(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        )]));
        let queue = SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            None,
            16,
            DefaultWorkerSelector::new(None, "test"),
            RouterQueuePolicy::Fcfs,
            None,
        );

        let (request, receiver) = make_request("delivery-race", isl);
        *response_rx.lock().unwrap() = Some(receiver);
        queue.enqueue(request).await;

        assert!(response_rx.lock().unwrap().is_none());
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_flood() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 4;
        let num_tasks = 25;

        let (queue, slots) = make_queue(num_workers, block_size, isl, None);

        let mut handles = Vec::new();
        for i in 0..num_tasks {
            let queue = Arc::clone(&queue);
            let slots = Arc::clone(&slots);
            handles.push(tokio::spawn(async move {
                let req_id = format!("req-{i}");
                let (req, rx) = make_request(&req_id, isl);
                queue.enqueue(req).await;
                let resp = rx.await.expect("oneshot dropped");
                let resp = resp.expect("scheduling failed");
                assert!(resp.best_worker.worker_id < num_workers as u64);

                slots.mark_prefill_completed(&req_id, decay_now()).unwrap();
                slots.free(&req_id, decay_now()).unwrap();
                queue.update().await;
            }));
        }

        for h in handles {
            h.await.expect("task panicked");
        }

        let active = slots.active_tokens(decay_now());
        for (worker, tokens) in &active {
            assert_eq!(
                *tokens, 0,
                "worker {worker:?} still has {tokens} active tokens"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_concurrent_immediate_admissions_see_prior_booking() {
        let selector = MinDecodeSelector {
            rendezvous: Some(Arc::new(SelectorRendezvous::default())),
        };
        let (queue, slots) = make_queue_with_custom_selector(2, 16, 512, None, selector);
        let barrier = Arc::new(Barrier::new(3));

        let (req1, rx1) = make_request("req-1", 512);
        let queue1 = Arc::clone(&queue);
        let barrier1 = Arc::clone(&barrier);
        let handle1 = tokio::spawn(async move {
            barrier1.wait().await;
            queue1.enqueue(req1).await;
        });

        let (req2, rx2) = make_request("req-2", 512);
        let queue2 = Arc::clone(&queue);
        let barrier2 = Arc::clone(&barrier);
        let handle2 = tokio::spawn(async move {
            barrier2.wait().await;
            queue2.enqueue(req2).await;
        });

        barrier.wait().await;
        handle1.await.unwrap();
        handle2.await.unwrap();

        let resp1 = rx1.await.unwrap().unwrap();
        let resp2 = rx2.await.unwrap().unwrap();
        assert_ne!(
            resp1.best_worker, resp2.best_worker,
            "second admission should see the first booking and choose the other idle worker"
        );

        for request_id in ["req-1", "req-2"] {
            slots
                .mark_prefill_completed(&request_id.to_string(), decay_now())
                .unwrap();
            slots.free(&request_id.to_string(), decay_now()).unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_queueing_under_pressure() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 2;
        let num_requests = 10;

        let (queue, slots) = make_queue(num_workers, block_size, isl, Some(0.0));

        let mut receivers = Vec::new();
        let mut req_ids = Vec::new();

        for i in 0..num_requests {
            let req_id = format!("pressure-{i}");
            let (req, rx) = make_request(&req_id, isl);
            queue.enqueue(req).await;
            receivers.push(rx);
            req_ids.push(req_id);
        }

        // Drain pending by cycling mark_prefill_completed + free + update
        // on already-scheduled requests until all receivers have a response.
        for _ in 0..num_requests {
            queue.update().await;
            for rid in &req_ids {
                let _ = slots.mark_prefill_completed(rid, decay_now());
                let _ = slots.free(rid, decay_now());
            }
        }
        queue.update().await;

        let mut ok_count = 0;
        for mut rx in receivers {
            if let Ok(result) = rx.try_recv() {
                result.expect("scheduling returned error");
                ok_count += 1;
            }
        }
        assert_eq!(ok_count, num_requests, "not all requests were scheduled");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pending_requests_receive_shutdown_on_queue_drop() {
        let block_size = 16;
        let isl = 512;
        let (queue, _slots) = make_queue(1, block_size, isl, Some(0.0));

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        rx1.await
            .expect("first response sender dropped")
            .expect("first request should be scheduled");

        let (req2, rx2) = make_request("req-2", isl);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        drop(queue);

        let response = tokio::time::timeout(Duration::from_secs(1), rx2)
            .await
            .expect("shutdown response timed out")
            .expect("pending response sender dropped");
        assert!(matches!(
            response,
            Err(KvSchedulerError::SubscriberShutdown)
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pending_count() {
        let block_size = 16;
        let isl = 512;
        let num_workers = 1;

        // threshold_frac=0.0 means any active tokens trigger queueing
        let (queue, slots) = make_queue(num_workers, block_size, isl, Some(0.0));
        assert_eq!(queue.pending_count(), 0);

        // First request goes through (worker is idle)
        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _resp1 = rx1.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0); // scheduled immediately

        // Second and third requests should be queued (worker is now prefill-busy)
        let (req2, _rx2) = make_request("req-2", isl);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        let (req3, _rx3) = make_request("req-3", isl);
        queue.enqueue(req3).await;
        assert_eq!(queue.pending_count(), 2);

        // Free the first request and update — should drain one from pending
        slots
            .mark_prefill_completed(&"req-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"req-1".to_string(), decay_now()).unwrap();
        queue.update().await;

        // After update, one pending request should have been scheduled
        assert!(
            queue.pending_count() < 2,
            "pending_count should decrease after free+update, got {}",
            queue.pending_count()
        );

        // Free req-2 and update to drain remaining
        let _ = slots.mark_prefill_completed(&"req-2".to_string(), decay_now());
        let _ = slots.free(&"req-2".to_string(), decay_now());
        queue.update().await;
        let _ = slots.mark_prefill_completed(&"req-3".to_string(), decay_now());
        let _ = slots.free(&"req-3".to_string(), decay_now());
        queue.update().await;

        assert_eq!(queue.pending_count(), 0, "all requests should be drained");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn policy_classes_apply_independent_thresholds_and_preserve_backlog_order() {
        let profile = policy_profile(
            r#"
default_policy_class: latency
policy_classes:
  - name: latency
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
  - name: bulk
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 1024
"#,
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

        let (mut active, active_rx) = make_request("active", 64);
        active.policy_class = Some("latency".to_string());
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (mut bulk, bulk_rx) = make_request("bulk", 64);
        bulk.policy_class = Some("bulk".to_string());
        queue.enqueue(bulk).await;
        bulk_rx.await.unwrap().unwrap();

        let (mut queued_first, mut queued_first_rx) = make_request("queued-first", 64);
        queued_first.policy_class = Some("latency".to_string());
        queue.enqueue(queued_first).await;
        assert_eq!(queue.pending_count(), 1);

        for request_id in ["active", "bulk"] {
            slots
                .mark_prefill_completed(&request_id.to_string(), decay_now())
                .unwrap();
            slots.free(&request_id.to_string(), decay_now()).unwrap();
        }

        let (mut queued_second, mut queued_second_rx) = make_request("queued-second", 64);
        queued_second.policy_class = Some("latency".to_string());
        queue.enqueue(queued_second).await;
        // The class still holds a runnable backlog, so the new arrival joins it
        // instead of taking the capacity that just freed.
        assert_eq!(
            queue.pending_count(),
            2,
            "new arrivals must not bypass backlog"
        );
        assert!(queued_first_rx.try_recv().is_err());
        assert!(queued_second_rx.try_recv().is_err());

        queue.update().await;
        queued_first_rx
            .try_recv()
            .expect("first queued request should be admitted")
            .expect("first queued request failed");
        assert!(
            queued_second_rx.try_recv().is_err(),
            "second request should remain behind the admitted head"
        );

        slots
            .mark_prefill_completed(&"queued-first".to_string(), decay_now())
            .unwrap();
        slots
            .free(&"queued-first".to_string(), decay_now())
            .unwrap();
        queue.update().await;
        queued_second_rx.await.unwrap().unwrap();
    }

    fn flat_class_profile() -> PolicyProfile {
        policy_profile(
            r#"
default_policy_class: standard
policy_classes:
  - name: standard
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
  - name: latency
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
"#,
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flat_policy_classes_select_queues_by_exact_name_or_the_default() {
        let (queue, _slots) = make_queue_with_profile(1, 16, 64, flat_class_profile());

        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (implicit, _implicit_rx) = make_request("implicit-default", 64);
        queue.enqueue(implicit).await;

        let (mut blank, _blank_rx) = make_request("blank-class", 64);
        blank.policy_class = Some(String::new());
        queue.enqueue(blank).await;

        let (mut named_default, _named_default_rx) = make_request("named-default", 64);
        named_default.policy_class = Some("standard".to_string());
        queue.enqueue(named_default).await;

        let (mut named_latency, _named_latency_rx) = make_request("named-latency", 64);
        named_latency.policy_class = Some("latency".to_string());
        queue.enqueue(named_latency).await;

        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 3,
                pending_isl_tokens: 192,
                pending_cached_tokens: 0,
            }),
            "absent, empty, and exactly named requests all land in the default class"
        );
        assert_eq!(
            queue.class_queue_stats(1),
            Some(ClassQueueStats {
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 0,
            })
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn unknown_policy_class_is_rejected_before_any_queueing_decision() {
        let (queue, _slots) = make_queue_with_profile(1, 16, 64, flat_class_profile());

        // Workers are idle, so this request would otherwise be dispatched by the
        // poll that follows class selection: an unknown class fails at selection,
        // before any queue storage or dispatch decision.
        let (mut unknown, unknown_rx) = make_request("unknown-class", 64);
        unknown.policy_class = Some("latencee".to_string());
        queue.enqueue(unknown).await;

        let error = unknown_rx.await.unwrap().unwrap_err();
        let KvSchedulerError::UnknownPolicyClass { policy_class } = &error else {
            panic!("expected unknown policy class, got {error:?}");
        };
        assert_eq!(policy_class, "latencee");
        assert!(!error.is_overload());
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 0,
                pending_isl_tokens: 0,
                pending_cached_tokens: 0,
            }),
            "a rejected class name must not touch any class's accounting"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_padded_policy_class_name_is_unknown_rather_than_normalized() {
        let (queue, _slots) = make_queue_with_profile(1, 16, 64, flat_class_profile());

        // Whitespace is part of the name, so a padded or whitespace-only value
        // is a name no class carries rather than a request for the default.
        for requested in [
            " latency",
            "latency ",
            " latency ",
            "\tlatency\n",
            " ",
            " \t\n ",
        ] {
            let (mut padded, padded_rx) = make_request("padded-class", 64);
            padded.policy_class = Some(requested.to_string());
            queue.enqueue(padded).await;

            let error = padded_rx.await.unwrap().unwrap_err();
            let KvSchedulerError::UnknownPolicyClass { policy_class } = &error else {
                panic!("expected {requested:?} to be unknown, got {error:?}");
            };
            assert_eq!(
                policy_class, requested,
                "the error must echo the name the client actually sent"
            );
        }
        assert_eq!(queue.pending_count(), 0);

        // Only the empty name carries no value at all, so it means "no
        // preference" and selects the default class.
        let (mut empty, empty_rx) = make_request("empty-class", 64);
        empty.policy_class = Some(String::new());
        queue.enqueue(empty).await;
        empty_rx
            .await
            .unwrap()
            .expect("an empty class name must select default_policy_class");
    }

    #[tokio::test(start_paused = true)]
    async fn dispatch_rejects_every_queued_request_past_its_class_deadline() {
        let profile = policy_profile(
            r#"
default_policy_class: tight
policy_classes:
  - name: tight
    slo_ms: 1000
    quantum: 1000000
    prefill_busy_threshold: 0
"#,
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (first, first_rx) = make_request("queued-first", 64);
        queue.enqueue(first).await;
        let (second, second_rx) = make_request("queued-second", 64);
        queue.enqueue(second).await;
        assert_eq!(queue.pending_count(), 2);
        assert_eq!(queue.pending_isl_tokens(), 128);

        tokio::time::advance(Duration::from_millis(1_500)).await;
        slots
            .mark_prefill_completed(&"active".to_string(), Instant::now())
            .unwrap();
        slots.free(&"active".to_string(), Instant::now()).unwrap();
        queue.update().await;

        for (label, receiver) in [("first", first_rx), ("second", second_rx)] {
            let error = receiver.await.unwrap().unwrap_err();
            let KvSchedulerError::QueueDeadlineExceeded(expiry) = &error else {
                panic!("expected {label} to miss its deadline, got {error:?}");
            };
            assert_eq!(expiry.policy_class, "tight");
            assert_eq!(expiry.stage, DeadlineStage::Dispatch);
            assert_eq!(expiry.slo_ms, 1_000);
            assert!(expiry.overdue_ms >= 500, "overdue={}", expiry.overdue_ms);
        }

        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.pending_isl_tokens(), 0);
        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 0,
                pending_isl_tokens: 0,
                pending_cached_tokens: 0,
            }),
            "dispatch expiry reverses class accounting exactly once"
        );
    }

    /// Send one lifecycle request through `queue`, returning its response
    /// receiver. The lease is returned so the caller keeps it alive; dropping it
    /// would fire the cleanup path and abort the request.
    #[allow(clippy::type_complexity)]
    async fn enqueue_lifecycle<Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static>(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel>,
        request_id: &str,
        isl_tokens: usize,
        policy_class: Option<&str>,
    ) -> (
        Option<Box<RequestLifecycleLease>>,
        tokio::sync::oneshot::Receiver<Result<SchedulingResponse, KvSchedulerError>>,
    ) {
        let (mut request, response_rx) = make_request(request_id, isl_tokens);
        request.mode = ScheduleMode::TrackedWithLifecycle {
            request_id: request_id.to_owned(),
        };
        request.policy_class = policy_class.map(str::to_owned);
        let lease = queue.new_request_lifecycle_lease(Some(request_id));
        let lease = queue
            .enqueue_with_block_hashes_and_lease(request, None, lease)
            .await;
        (lease, response_rx)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn queue_admission_runs_above_policy_class_selection() {
        // Queue admission sits entirely above policy classes, so every arrival
        // reaches the policy first, whatever class name it carries. The class
        // layer then rejects an unknown name at the one downstream handoff.
        let admits = Arc::new(AtomicUsize::new(0));
        let aborts = Arc::new(AtomicUsize::new(0));
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(CountingDecisionPolicy {
                admits: Arc::clone(&admits),
                aborts: Arc::clone(&aborts),
                deferred: Vec::new(),
            })),
        };
        let (queue, slots) =
            make_queue_with_profile_and_selector(1, 16, 64, flat_class_profile(), selector);

        // Bypass: unmanaged by the policy, so no terminal event, but the policy
        // still saw it before the class layer refused the name.
        let (_bypass_lease, bypass_rx) =
            enqueue_lifecycle(&queue, "bypass-unknown", 64, Some("latencee")).await;
        let error = bypass_rx.await.unwrap().unwrap_err();
        let KvSchedulerError::UnknownPolicyClass { policy_class } = &error else {
            panic!("expected unknown policy class, got {error:?}");
        };
        assert_eq!(policy_class, "latencee");
        assert_eq!(
            admits.load(Ordering::Relaxed),
            1,
            "queue admission must run before policy-class selection"
        );
        assert_eq!(
            aborts.load(Ordering::Relaxed),
            0,
            "a bypassed request is unmanaged, so it reports no terminal event"
        );

        // Ready: managed, so the same downstream rejection reports exactly one
        // terminal abort.
        let (_ready_lease, ready_rx) =
            enqueue_lifecycle(&queue, "ready-unknown", 64, Some("latencee")).await;
        assert!(matches!(
            ready_rx.await.unwrap().unwrap_err(),
            KvSchedulerError::UnknownPolicyClass { .. }
        ));
        assert_eq!(admits.load(Ordering::Relaxed), 2);
        assert_eq!(
            aborts.load(Ordering::Relaxed),
            1,
            "managed work rejected downstream reports exactly one abort"
        );

        // Defer: the policy holds it despite the unknown class name, so nothing
        // downstream has seen it yet.
        let (_defer_lease, defer_rx) =
            enqueue_lifecycle(&queue, "defer-unknown", 64, Some("latencee")).await;
        assert_eq!(admits.load(Ordering::Relaxed), 3);
        assert_eq!(queue.pending_count(), 0);
        for class_index in 0..2 {
            assert_eq!(
                queue.class_queue_stats(class_index).unwrap().pending_count,
                0,
                "deferred work has not selected a class, unknown name or not"
            );
        }
        assert_eq!(aborts.load(Ordering::Relaxed), 1);

        // Waking it puts it through the same downstream handoff, where the class
        // layer refuses the name and the abort is reported exactly once.
        let (_trigger_lease, trigger_rx) =
            enqueue_lifecycle(&queue, "ready-trigger", 64, None).await;
        let trigger_worker = trigger_rx
            .await
            .unwrap()
            .expect("the trigger dispatches through Policy Class and Queue")
            .best_worker;
        assert!(
            queue
                .abort_request("ready-trigger", None, Some(trigger_worker))
                .await
        );
        assert!(matches!(
            defer_rx.await.unwrap().unwrap_err(),
            KvSchedulerError::UnknownPolicyClass { .. }
        ));
        assert_eq!(
            aborts.load(Ordering::Relaxed),
            3,
            "the trigger and the woken request each report exactly one abort"
        );
        assert_eq!(queue.pending_count(), 0);

        // A configured name takes the identical path and dispatches.
        slots
            .free(&"ready-trigger".to_string(), decay_now())
            .unwrap();
        let (_ok_lease, ok_rx) =
            enqueue_lifecycle(&queue, "ready-known", 64, Some("latency")).await;
        ok_rx.await.unwrap().expect("a configured class dispatches");
        assert_eq!(admits.load(Ordering::Relaxed), 5);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn class_queue_admission_rejects_a_request_that_aged_past_its_slo() {
        // The 1ms SLO is crossed by the policy's 40ms stall, which runs after
        // the arrival timestamp is captured and before the admission check.
        let profile = policy_profile(
            r#"
default_policy_class: tight
policy_classes:
  - name: tight
    slo_ms: 1
    quantum: 1000000
    prefill_busy_threshold: 0
"#,
        );
        let aborts = Arc::new(AtomicUsize::new(0));
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(CountingSlowPolicy {
                admits: Arc::new(AtomicUsize::new(0)),
                aborts: Arc::clone(&aborts),
                stall: Duration::from_millis(40),
            })),
        };
        let (queue, slots) = make_queue_with_profile_and_selector(1, 16, 64, profile, selector);

        // Occupy the worker so the next request needs queue storage.
        let (_active_lease, active_rx) = enqueue_lifecycle(&queue, "bypass-active", 64, None).await;
        active_rx.await.unwrap().expect("first request dispatches");

        let (_slow_lease, slow_rx) = enqueue_lifecycle(&queue, "slow-queued", 64, None).await;
        let error = slow_rx.await.unwrap().unwrap_err();
        let KvSchedulerError::QueueDeadlineExceeded(expiry) = &error else {
            panic!("expected an admission-stage deadline rejection, got {error:?}");
        };
        assert_eq!(expiry.stage, DeadlineStage::Admission);
        assert_eq!(expiry.policy_class, "tight");
        assert_eq!(expiry.slo_ms, 1);
        assert!(!error.is_overload());

        assert_eq!(
            queue.pending_count(),
            0,
            "a request rejected at admission never entered queue storage"
        );
        assert_eq!(queue.pending_isl_tokens(), 0);
        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 0,
                pending_isl_tokens: 0,
                pending_cached_tokens: 0,
            }),
            "admission rejection must leave class accounting untouched"
        );
        assert_eq!(
            aborts.load(Ordering::Relaxed),
            1,
            "rejected managed work reports exactly one terminal abort"
        );

        // Free the worker: storage is no longer required, so an equally stale
        // request is admitted directly and never reaches this gate.
        slots
            .mark_prefill_completed(&"bypass-active".to_string(), Instant::now())
            .unwrap();
        slots
            .free(&"bypass-active".to_string(), Instant::now())
            .unwrap();
        let (_immediate_lease, immediate_rx) =
            enqueue_lifecycle(&queue, "slow-immediate", 64, None).await;
        immediate_rx
            .await
            .unwrap()
            .expect("direct admission does not pass the class-queue admission check");
        assert_eq!(
            aborts.load(Ordering::Relaxed),
            1,
            "a directly admitted request must not be aborted"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_idle_arrival_is_admitted_directly_and_a_busy_one_takes_the_class_limit() {
        // The class refuses queue storage outright. That limit belongs to the
        // queue, so it binds only once the router decides the request has to
        // wait: an arrival that finds an idle worker and no backlog is admitted
        // directly and never meets it.
        let profile = policy_profile(
            r#"
default_policy_class: closed
policy_classes:
  - name: closed
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 0
"#,
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

        let (idle, idle_rx) = make_request("idle", 64);
        queue.enqueue(idle).await;
        idle_rx
            .await
            .unwrap()
            .expect("an idle arrival is admitted directly, bypassing the class limit");
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 0,
                pending_isl_tokens: 0,
                pending_cached_tokens: 0,
            }),
            "direct admission touches no class accounting"
        );

        // The worker is busy now, so the next arrival needs storage and the same
        // limit rejects it.
        let (busy, busy_rx) = make_request("busy", 64);
        queue.enqueue(busy).await;
        let error = busy_rx.await.unwrap().unwrap_err();
        let KvSchedulerError::QueueRejected(rejection) = &error else {
            panic!("expected a class-local queue rejection, got {error:?}");
        };
        assert_eq!(rejection.policy_class, "closed");
        assert_eq!(rejection.limit_kind, super::super::QueueLimitKind::Requests);
        assert_eq!(rejection.limit, 0);
        assert_eq!(queue.pending_count(), 0);

        // Free the worker and the direct route is available again.
        slots
            .mark_prefill_completed(&"idle".to_string(), decay_now())
            .unwrap();
        slots.free(&"idle".to_string(), decay_now()).unwrap();
        let (idle_again, idle_again_rx) = make_request("idle-again", 64);
        queue.enqueue(idle_again).await;
        idle_again_rx
            .await
            .unwrap()
            .expect("the direct route returns with the freed worker");
        slots.free(&"idle-again".to_string(), decay_now()).unwrap();
    }

    /// Enqueue one lifecycle-tracked request per ID and return the leases and
    /// response receivers, keyed by request ID.
    #[allow(clippy::type_complexity)]
    async fn enqueue_lifecycle_batch<Sel: WorkerSelector<SimpleWorkerConfig> + Send + 'static>(
        queue: &SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, Sel>,
        request_ids: impl IntoIterator<Item = &'static str>,
        isl_tokens: usize,
    ) -> (
        Vec<Option<Box<RequestLifecycleLease>>>,
        HashMap<
            &'static str,
            tokio::sync::oneshot::Receiver<Result<SchedulingResponse, KvSchedulerError>>,
        >,
    ) {
        let mut leases = Vec::new();
        let mut receivers = HashMap::new();
        for request_id in request_ids {
            let (lease, response_rx) = enqueue_lifecycle(queue, request_id, isl_tokens, None).await;
            leases.push(lease);
            receivers.insert(request_id, response_rx);
        }
        (leases, receivers)
    }

    #[tokio::test(start_paused = true)]
    async fn deferred_work_released_by_an_abort_repeats_the_router_scheduling_decision() {
        let profile = policy_profile(
            r#"
default_policy_class: tight
policy_classes:
  - name: tight
    slo_ms: 1000
    quantum: 1000000
    prefill_busy_threshold: 0
"#,
        );
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(WakeDeferredOnAbort::default())),
        };
        let (queue, slots) = make_queue_with_profile_and_selector(1, 16, 64, profile, selector);

        let (leases, mut receivers) = enqueue_lifecycle_batch(
            &queue,
            ["bypass-active", "ready-runnable", "defer-parked"],
            64,
        )
        .await;

        // `bypass-active` found an idle worker and was admitted directly, and it
        // now makes the worker busy. `ready-runnable` is queued behind it under
        // the same 1s SLO; `defer-parked` never reached a class at all.
        receivers
            .remove("bypass-active")
            .unwrap()
            .await
            .unwrap()
            .expect("the bypassed request reaches the router's scheduling decision");
        assert_eq!(
            queue.pending_count(),
            1,
            "deferred work is absent from pending accounting until it is woken"
        );
        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 0,
            }),
            "deferred work is absent from class statistics until it is woken"
        );

        tokio::time::advance(Duration::from_millis(1_500)).await;
        slots
            .mark_prefill_completed(&"bypass-active".to_string(), Instant::now())
            .unwrap();
        slots
            .free(&"bypass-active".to_string(), Instant::now())
            .unwrap();
        queue.update().await;

        let runnable = receivers
            .remove("ready-runnable")
            .unwrap()
            .await
            .unwrap()
            .unwrap_err();
        let KvSchedulerError::QueueDeadlineExceeded(runnable) = &runnable else {
            panic!("expected a deadline rejection, got {runnable:?}");
        };
        assert_eq!(
            runnable.stage,
            DeadlineStage::Dispatch,
            "the runnable head expired at a dispatch poll"
        );

        // Shedding that head emptied the class and the worker was freed, so the
        // router repeats its ordinary decision for the request the abort
        // released and finds the direct route open. It is dispatched on the same
        // terms as an equally stale fresh arrival would be: queue gates bind
        // only work the router decides has to wait.
        receivers
            .remove("defer-parked")
            .unwrap()
            .await
            .unwrap()
            .expect("released work takes the direct route when the router is idle");
        assert_eq!(queue.pending_count(), 0);
        drop(leases);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bypass_ready_and_woken_defer_share_one_router_scheduling_decision() {
        // Two classes with equal quanta. `prefill_busy_threshold: 0` makes any
        // active work busy, so the worker's state alone decides whether the
        // router admits directly or stores the request.
        let profile = policy_profile(
            r#"
default_policy_class: first
policy_classes:
  - name: first
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
  - name: second
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
"#,
        );
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(WakeDeferredOnAbort::default())),
        };
        let (queue, slots) = make_queue_with_profile_and_selector(1, 16, 64, profile, selector);

        // Ready, idle worker: the router admits it directly, so it never enters
        // class `first`'s queue. It then occupies the only worker.
        let (_blocker_lease, blocker_rx) =
            enqueue_lifecycle(&queue, "ready-blocker", 64, Some("first")).await;
        let blocker_worker = blocker_rx
            .await
            .unwrap()
            .expect("a ready request reaches the router's scheduling decision")
            .best_worker;
        assert_eq!(queue.pending_count(), 0, "direct admission uses no storage");

        // Bypass and Ready with the worker busy: the same decision now stores
        // both, each in its own class.
        let (_bypass_lease, bypass_rx) =
            enqueue_lifecycle(&queue, "bypass-second", 64, Some("second")).await;
        let (_ready_lease, ready_rx) =
            enqueue_lifecycle(&queue, "ready-first", 64, Some("first")).await;
        let (_deferred_lease, deferred_rx) =
            enqueue_lifecycle(&queue, "defer-first", 64, Some("first")).await;

        assert_eq!(
            queue.class_queue_stats(0).unwrap().pending_count,
            1,
            "only the ready request reached class `first`; the deferred one has \
             not selected a class"
        );
        assert_eq!(
            queue.class_queue_stats(1).unwrap().pending_count,
            1,
            "the bypassed request reached class `second` like any other arrival"
        );
        assert_eq!(queue.pending_count(), 2);

        // Aborting the running request releases the deferred one. The worker is
        // still occupied in the slot tracker and class `first` already holds a
        // backlog, so the same decision stores the woken request too.
        assert!(
            queue
                .abort_request("ready-blocker", None, Some(blocker_worker))
                .await
        );
        assert_eq!(
            queue.class_queue_stats(0).unwrap().pending_count,
            2,
            "the released request selected class `first` and joined its queue"
        );
        assert_eq!(queue.class_queue_stats(1).unwrap().pending_count, 1);
        assert_eq!(queue.pending_count(), 3);

        // Release whatever is booked and re-poll, until the backlog drains. All
        // three go through the same ring, so the bypassed, ready, and woken
        // requests are served the same way.
        for _ in 0..4 {
            for request_id in [
                "ready-blocker",
                "bypass-second",
                "ready-first",
                "defer-first",
            ] {
                let _ = slots.mark_prefill_completed(&request_id.to_string(), decay_now());
                let _ = slots.free(&request_id.to_string(), decay_now());
            }
            queue.update().await;
        }
        bypass_rx
            .await
            .unwrap()
            .expect("the bypassed request dispatches from its class queue");
        ready_rx
            .await
            .unwrap()
            .expect("the ready request dispatches from its class queue");
        deferred_rx
            .await
            .unwrap()
            .expect("the woken request dispatches from its class queue");
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn bypass_ready_and_woken_defer_all_admit_directly_when_the_router_is_idle() {
        // No class here ever stores work for capacity, so every arrival that
        // reaches the router's decision takes the direct route — whichever of
        // the three admission outcomes brought it there.
        let profile = policy_profile(
            r#"
default_policy_class: only
policy_classes:
  - name: only
    slo_ms: 600000
    quantum: 1
"#,
        );
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(WakeDeferredOnAbort::default())),
        };
        let (queue, slots) = make_queue_with_profile_and_selector(2, 16, 64, profile, selector);

        let (_bypass_lease, bypass_rx) = enqueue_lifecycle(&queue, "bypass-one", 64, None).await;
        bypass_rx
            .await
            .unwrap()
            .expect("a bypassed request admits directly");

        let (_ready_lease, ready_rx) = enqueue_lifecycle(&queue, "ready-one", 64, None).await;
        let ready_worker = ready_rx
            .await
            .unwrap()
            .expect("a ready request admits directly")
            .best_worker;
        assert_eq!(queue.pending_count(), 0);

        // Park one, then release it. The router repeats the same decision at
        // wake time and finds the direct route open, so the woken request is
        // dispatched without ever entering a class queue.
        let (_deferred_lease, deferred_rx) = enqueue_lifecycle(&queue, "defer-one", 64, None).await;
        assert_eq!(queue.pending_count(), 0);

        assert!(
            queue
                .abort_request("ready-one", None, Some(ready_worker))
                .await
        );
        deferred_rx
            .await
            .unwrap()
            .expect("a woken request admits directly when the router is idle");
        assert_eq!(
            queue.pending_count(),
            0,
            "no arrival here ever needed queue storage"
        );

        for request_id in ["bypass-one", "ready-one", "defer-one"] {
            let _ = slots.free(&request_id.to_string(), decay_now());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn deferred_work_is_absent_from_class_limits_until_it_is_woken() {
        // One class that accepts at most two queued requests per worker.
        let profile = policy_profile(
            r#"
default_policy_class: capped
policy_classes:
  - name: capped
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 2
"#,
        );
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(WakeDeferredOnAbort::default())),
        };
        let (queue, _slots) = make_queue_with_profile_and_selector(1, 16, 64, profile, selector);

        let (_active_lease, active_rx) = enqueue_lifecycle(&queue, "ready-active", 64, None).await;
        let active_worker = active_rx.await.unwrap().unwrap().best_worker;

        let (_parked_lease, parked_rx) = enqueue_lifecycle(&queue, "defer-parked", 64, None).await;
        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 0,
                pending_isl_tokens: 0,
                pending_cached_tokens: 0,
            }),
            "deferred work is absent from class statistics"
        );
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.pending_isl_tokens(), 0);

        // Both queue slots remain available, so the deferred request consumed
        // none of the class's limit.
        let (_first_lease, _first_rx) = enqueue_lifecycle(&queue, "ready-first", 64, None).await;
        let (_second_lease, _second_rx) = enqueue_lifecycle(&queue, "ready-second", 64, None).await;
        assert_eq!(queue.class_queue_stats(0).unwrap().pending_count, 2);

        let (_third_lease, third_rx) = enqueue_lifecycle(&queue, "ready-third", 64, None).await;
        let error = third_rx.await.unwrap().unwrap_err();
        assert!(
            matches!(&error, KvSchedulerError::QueueRejected(rejection)
                if rejection.policy_class == "capped" && rejection.current == 2),
            "the class limit still counts the two queued requests, got {error:?}"
        );

        // Waking the deferred request puts it through the same class-local
        // limit, which the queued pair has already reached.
        assert!(
            queue
                .abort_request("ready-active", None, Some(active_worker))
                .await
        );
        let parked = parked_rx.await.unwrap().unwrap_err();
        assert!(
            matches!(&parked, KvSchedulerError::QueueRejected(rejection)
                if rejection.policy_class == "capped" && rejection.current == 2),
            "a woken request takes the same class limit as any other arrival, got {parked:?}"
        );
        assert_eq!(queue.pending_count(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn cancelling_a_queued_request_frees_class_room_for_the_work_it_wakes() {
        // One class that accepts a single queued request per worker, so the
        // cancelled request and the deferred one cannot both be counted.
        let profile = policy_profile(
            r#"
default_policy_class: capped
policy_classes:
  - name: capped
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 1
"#,
        );
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(WakeDeferredOnAbort::default())),
        };
        let (queue, slots) = make_queue_with_profile_and_selector(1, 16, 64, profile, selector);

        let (_active_lease, active_rx) = enqueue_lifecycle(&queue, "ready-active", 64, None).await;
        active_rx.await.unwrap().expect("the worker is idle");

        // The class's one queue slot is taken, and the deferred request is held
        // above the classes where it consumes nothing.
        let (cancelled_lease, cancelled_rx) =
            enqueue_lifecycle(&queue, "ready-cancelled", 64, None).await;
        let (_parked_lease, mut parked_rx) =
            enqueue_lifecycle(&queue, "defer-parked", 64, None).await;
        assert_eq!(queue.class_queue_stats(0).unwrap().pending_count, 1);

        // Cancelling the queued request retracts it and its class accounting
        // before its abort reaches the policy, so the request that abort
        // releases is measured against the room the cancellation just made.
        drop(cancelled_rx);
        drop(cancelled_lease);

        tokio::time::timeout(Duration::from_secs(1), async {
            while queue.class_queue_stats(0).unwrap().pending_count == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the woken request must take the retracted request's class slot");
        assert!(
            parked_rx.try_recv().is_err(),
            "the woken request was admitted, not rejected against stale accounting"
        );
        assert_eq!(queue.pending_count(), 1);

        // It then dispatches from that class queue like any other admitted work.
        slots
            .mark_prefill_completed(&"ready-active".to_string(), decay_now())
            .unwrap();
        slots
            .free(&"ready-active".to_string(), decay_now())
            .unwrap();
        queue.update().await;
        parked_rx
            .await
            .unwrap()
            .expect("the woken request dispatches from its class queue");
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_direct_dispatch_leaves_no_lease_armed_for_a_retry_to_inherit() {
        // The worker is idle and the class has no backlog, so this request is
        // admitted directly and never enters a class queue; nothing here drains
        // one. Its response receiver is closed before it arrives, so that direct
        // dispatch books, fails delivery, rolls the booking back, and releases
        // the request's lifecycle inside the same command. The lease handed back
        // must not be armed: a same-ID retry would otherwise be killed by the
        // cleanup that stale lease eventually fires.
        let profile = policy_profile(
            r#"
default_policy_class: only
policy_classes:
  - name: only
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
"#,
        );
        let aborts = Arc::new(AtomicUsize::new(0));
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(CountingSlowPolicy {
                admits: Arc::new(AtomicUsize::new(0)),
                aborts: Arc::clone(&aborts),
                stall: Duration::ZERO,
            })),
        };
        let (queue, slots) = make_queue_with_profile_and_selector(1, 16, 64, profile, selector);

        let (mut abandoned, abandoned_rx) = make_request("retried", 64);
        abandoned.mode = ScheduleMode::TrackedWithLifecycle {
            request_id: "retried".to_owned(),
        };
        let abandoned_lease = queue.new_request_lifecycle_lease(Some("retried"));
        drop(abandoned_rx);
        let abandoned_lease = queue
            .enqueue_with_block_hashes_and_lease(abandoned, None, abandoned_lease)
            .await
            .expect("a lifecycle request receives a lease");

        assert!(
            abandoned_lease.request_id.is_none(),
            "the actor released this request while scheduling it, so the lease \
             it hands back must not be armed"
        );
        assert_eq!(aborts.load(Ordering::Relaxed), 1);
        assert_eq!(slots.request_worker(&"retried".to_string()), None);

        // The same ID is accepted again, which is only sound because the actor
        // released the first attempt.
        let (mut retry_lease, retry_rx) = enqueue_lifecycle(&queue, "retried", 64, None).await;
        let retry = retry_rx
            .await
            .unwrap()
            .expect("the retry is admitted once the first attempt was released");

        // Dropping the stale lease must not retract the retry.
        drop(abandoned_lease);
        queue.update().await;
        assert_eq!(
            slots.request_worker(&"retried".to_string()),
            Some(retry.best_worker),
            "a stale lease must not free the retry's booking"
        );
        assert_eq!(
            aborts.load(Ordering::Relaxed),
            1,
            "a stale lease must not abort the retry"
        );
        retry_lease.as_mut().unwrap().disarm();
        slots.free(&"retried".to_string(), decay_now()).unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_slow_wake_callback_expires_the_work_it_releases_at_the_admission_gate() {
        // `tight` is still inside its deadline when the abort callback starts
        // and past it when the callback returns, so the Admission gate can only
        // catch it if it reads the clock after the callback rather than before.
        let profile = policy_profile(
            r#"
default_policy_class: roomy
policy_classes:
  - name: roomy
    slo_ms: 600000
    quantum: 1000000
    prefill_busy_threshold: 0
  - name: tight
    slo_ms: 100
    quantum: 1000000
    prefill_busy_threshold: 0
"#,
        );
        let selector = AdmissionPolicySelector {
            selector: MinDecodeSelector { rendezvous: None },
            policy: Some(Box::new(SlowWakeOnAbort {
                deferred: Vec::new(),
                stall: Duration::from_millis(400),
            })),
        };
        let (queue, _slots) = make_queue_with_profile_and_selector(1, 16, 64, profile, selector);

        let (_active_lease, active_rx) =
            enqueue_lifecycle(&queue, "ready-active", 64, Some("roomy")).await;
        let active_worker = active_rx
            .await
            .unwrap()
            .expect("the worker is idle")
            .best_worker;

        let (_parked_lease, parked_rx) =
            enqueue_lifecycle(&queue, "defer-parked", 64, Some("tight")).await;
        assert_eq!(queue.pending_count(), 0);

        assert!(
            queue
                .abort_request("ready-active", None, Some(active_worker))
                .await
        );
        let error = parked_rx.await.unwrap().unwrap_err();
        let KvSchedulerError::QueueDeadlineExceeded(expiry) = &error else {
            panic!("expected a deadline rejection, got {error:?}");
        };
        assert_eq!(
            expiry.stage,
            DeadlineStage::Admission,
            "the gate must read the clock after the policy callback returned"
        );
        assert_eq!(expiry.policy_class, "tight");
        assert!(expiry.overdue_ms > 0);
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn dispatch_serves_the_first_live_head_after_shedding_expired_ones() {
        let profile = policy_profile(
            r#"
default_policy_class: tight
policy_classes:
  - name: tight
    slo_ms: 1000
    quantum: 1000000
    prefill_busy_threshold: 0
"#,
        );
        let (queue, slots) = make_queue_with_profile(1, 16, 64, profile);

        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (doomed, doomed_rx) = make_request("doomed", 64);
        queue.enqueue(doomed).await;

        // `live` arrives while `doomed` is still inside its deadline, so both
        // sit in the class queue and only the later poll sheds the first one.
        tokio::time::advance(Duration::from_millis(500)).await;
        let (live, live_rx) = make_request("live", 64);
        queue.enqueue(live).await;
        assert_eq!(queue.pending_count(), 2);

        tokio::time::advance(Duration::from_millis(600)).await;

        slots
            .mark_prefill_completed(&"active".to_string(), Instant::now())
            .unwrap();
        slots.free(&"active".to_string(), Instant::now()).unwrap();
        queue.update().await;

        let error = doomed_rx.await.unwrap().unwrap_err();
        assert!(
            matches!(&error, KvSchedulerError::QueueDeadlineExceeded(expiry)
                if expiry.stage == DeadlineStage::Dispatch),
            "expected a dispatch-stage deadline rejection, got {error:?}"
        );
        live_rx
            .await
            .unwrap()
            .expect("the first live head must dispatch in the same drain");
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn class_local_limit_rejection_is_typed_and_not_overload() {
        let profile = policy_profile(
            r#"
default_policy_class: capped
policy_classes:
  - name: capped
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 1
"#,
        );
        let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);

        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (queued, _queued_rx) = make_request("queued", 64);
        queue.enqueue(queued).await;

        let (rejected, rejected_rx) = make_request("rejected", 64);
        queue.enqueue(rejected).await;
        let error = rejected_rx.await.unwrap().unwrap_err();
        let KvSchedulerError::QueueRejected(rejection) = &error else {
            panic!("expected queue rejection, got {error:?}");
        };
        assert_eq!(rejection.policy_class, "capped");
        assert_eq!(rejection.limit_kind, super::super::QueueLimitKind::Requests);
        assert_eq!(rejection.current, 1);
        assert_eq!(rejection.limit, 1);
        assert!(!error.is_overload());

        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 0,
            })
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn per_worker_limit_tracks_discovered_worker_count_without_evicting() {
        let profile = policy_profile(
            r#"
default_policy_class: capped
policy_classes:
  - name: capped
    slo_ms: 600000
    quantum: 1
    prefill_busy_threshold: 0
    request_queue_limit_per_worker: 1
"#,
        );
        let (queue, _slots, cfg_tx) = make_queue_with_profile_and_sender(1, 16, 64, profile);

        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (first, _first_rx) = make_request("first", 64);
        queue.enqueue(first).await;

        cfg_tx.send_modify(|configs| {
            configs.insert(
                1,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(64),
                    ..Default::default()
                },
            );
        });
        let (second, _second_rx) = make_request("second", 64);
        queue.enqueue(second).await;
        assert_eq!(queue.pending_count(), 2);

        cfg_tx.send_modify(|configs| {
            configs.remove(&1);
        });
        let (rejected, rejected_rx) = make_request("rejected", 64);
        queue.enqueue(rejected).await;
        let error = rejected_rx.await.unwrap().unwrap_err();
        let KvSchedulerError::QueueRejected(rejection) = error else {
            panic!("expected queue rejection, got {error:?}");
        };
        assert_eq!(rejection.current, 2);
        assert_eq!(rejection.limit, 1);
        assert_eq!(queue.pending_count(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn test_queue_update_uses_decayed_oldest_prefill_load() {
        let estimator: Arc<dyn PrefillLoadEstimator> = Arc::new(FixedPrefillLoadEstimator {
            duration: Duration::from_secs(10),
        });
        let (queue, _slots, _cfg_tx) =
            make_queue_with_sender(1, 16, 100, Some(0.5), Some(estimator));

        let (req1, rx1) = make_request("req-1", 100);
        queue.enqueue(req1).await;
        let _ = rx1.await.unwrap().unwrap();

        let (req2, mut rx2) = make_request("req-2", 100);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);

        tokio::time::advance(Duration::from_secs(6)).await;
        queue.update().await;

        let scheduled = rx2
            .try_recv()
            .expect("queued request should have been scheduled");
        let response = scheduled.expect("scheduling returned error");
        assert_eq!(response.best_worker.worker_id, 0);
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_overloaded_provider_filters_at_admission() {
        let overloaded_worker_provider: OverloadedWorkerProvider =
            Arc::new(|| Some(Arc::new(HashSet::from([0]))));
        let (queue, _slots) =
            make_queue_with_providers(1, 16, 256, Some(overloaded_worker_provider), None);

        let (req, rx) = make_request("overloaded", 256);
        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(
            resp,
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn hard_availability_provider_filters_unpinned_and_pinned_selection() {
        let available_worker_provider: WorkerAvailabilityProvider =
            Arc::new(|| Some(Arc::new(HashSet::from([1]))));
        let (queue, slots) =
            make_queue_with_providers(2, 16, 256, None, Some(available_worker_provider));

        let request_id = "available".to_string();
        let (request, response_rx) = make_request(&request_id, 256);
        queue.enqueue(request).await;
        let response = response_rx.await.unwrap().unwrap();
        assert_eq!(response.best_worker.worker_id, 1);
        slots
            .mark_prefill_completed(&request_id, decay_now())
            .unwrap();
        slots.free(&request_id, decay_now()).unwrap();

        let (mut pinned, pinned_rx) = make_request("unavailable-pin", 256);
        pinned.pinned_worker = Some(WorkerWithDpRank::from_worker_id(0));
        queue.enqueue(pinned).await;
        assert!(matches!(
            pinned_rx.await.unwrap(),
            Err(KvSchedulerError::NoEndpoints)
        ));
    }

    /// Simulates the EPP path: router starts with zero workers (skip_initial_worker_wait),
    /// then register_workers lazily injects workers before routing.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_register_workers_lazy_epp_path() {
        let block_size = 16;
        let isl = 512;

        // Start with zero workers (mimics skip_initial_worker_wait=true)
        let (queue, slots, cfg_tx) = make_queue_with_sender(0, block_size, isl, None, None);

        // Routing with no workers must fail
        let (req_fail, rx_fail) = make_request("before-register", isl);
        queue.enqueue(req_fail).await;
        let resp = rx_fail.await.expect("oneshot dropped");
        assert!(
            matches!(
                resp,
                Err(crate::scheduling::types::KvSchedulerError::NoEndpoints)
            ),
            "expected NoEndpoints before register_workers, got {resp:?}"
        );

        // Lazily register two workers in the slot tracker (EPP supplies pod list)
        slots.upsert_worker(WorkerDpRange::new(100, 0, 1)).unwrap();
        slots.upsert_worker(WorkerDpRange::new(200, 0, 1)).unwrap();

        // Also update the config watch so the selector can see these workers
        let mut configs = HashMap::new();
        for &id in &[100_u64, 200_u64] {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        cfg_tx.send(configs).unwrap();

        // Routing after registration must succeed and pick one of the registered workers
        let (req_ok, rx_ok) = make_request("after-register", isl);
        queue.enqueue(req_ok).await;
        let resp = rx_ok
            .await
            .expect("oneshot dropped")
            .expect("scheduling failed");
        assert!(
            resp.best_worker.worker_id == 100 || resp.best_worker.worker_id == 200,
            "expected worker 100 or 200, got {}",
            resp.best_worker.worker_id
        );

        // Clean up
        slots
            .mark_prefill_completed(&"after-register".to_string(), decay_now())
            .unwrap();
        slots
            .free(&"after-register".to_string(), decay_now())
            .unwrap();
    }

    /// Register_workers is additive: calling with a new set does NOT remove old workers.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_register_workers_additive() {
        let block_size = 16;
        let isl = 256;

        let (queue, slots, cfg_tx) = make_queue_with_sender(0, block_size, isl, None, None);

        // Register worker 10 in slots and config
        slots.upsert_worker(WorkerDpRange::new(10, 0, 1)).unwrap();

        let mut configs = HashMap::new();
        configs.insert(
            10_u64,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
        cfg_tx.send(configs.clone()).unwrap();

        // Register worker 20 (worker 10 must NOT be evicted)
        slots.upsert_worker(WorkerDpRange::new(20, 0, 1)).unwrap();

        configs.insert(
            20_u64,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
        cfg_tx.send(configs).unwrap();

        // Send enough requests to statistically prove both workers are available
        let mut seen = std::collections::HashSet::new();
        for i in 0..20 {
            let req_id = format!("add-{i}");
            let (req, rx) = make_request(&req_id, isl);
            queue.enqueue(req).await;
            let resp = rx
                .await
                .expect("oneshot dropped")
                .expect("scheduling failed");
            seen.insert(resp.best_worker.worker_id);
            slots.mark_prefill_completed(&req_id, decay_now()).unwrap();
            slots.free(&req_id, decay_now()).unwrap();
        }

        assert!(
            seen.contains(&10) && seen.contains(&20),
            "both workers should be reachable after additive registration, saw: {seen:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn allowed_worker_request_joins_backlog_and_dispatches_within_allow_list() {
        let block_size = 16;
        let isl = 256;
        let (queue, slots) = make_queue(2, block_size, isl, Some(0.0));

        let (active_a, active_a_rx) = make_request("active-a", isl);
        queue.enqueue(active_a).await;
        let active_a_worker = active_a_rx.await.unwrap().unwrap().best_worker.worker_id;

        let (active_b, active_b_rx) = make_request("active-b", isl);
        queue.enqueue(active_b).await;
        active_b_rx.await.unwrap().unwrap();

        let (backlog_head, backlog_head_rx) = make_request("backlog-head", isl);
        queue.enqueue(backlog_head).await;
        assert_eq!(queue.pending_count(), 1);

        slots
            .mark_prefill_completed(&"active-a".to_string(), decay_now())
            .unwrap();
        slots.free(&"active-a".to_string(), decay_now()).unwrap();

        let (mut allowed, mut allowed_rx) = make_request("allowed", isl);
        allowed.allowed_worker_ids = Some(HashSet::from([active_a_worker]));
        queue.enqueue(allowed).await;
        // The class already holds a runnable backlog, so this request is queued
        // behind it rather than taking the worker that just freed.
        assert_eq!(
            queue.pending_count(),
            2,
            "allow-list request must not bypass the existing class backlog"
        );
        assert!(allowed_rx.try_recv().is_err());

        queue.update().await;
        let backlog_head_worker = backlog_head_rx
            .await
            .unwrap()
            .unwrap()
            .best_worker
            .worker_id;
        assert!(allowed_rx.try_recv().is_err());

        slots
            .mark_prefill_completed(&"backlog-head".to_string(), decay_now())
            .unwrap();
        slots
            .free(&"backlog-head".to_string(), decay_now())
            .unwrap();
        queue.update().await;

        let allowed_worker = allowed_rx.await.unwrap().unwrap().best_worker.worker_id;
        assert_eq!(allowed_worker, active_a_worker);

        for request_id in ["active-b", "allowed"] {
            slots
                .mark_prefill_completed(&request_id.to_string(), decay_now())
                .unwrap();
            slots.free(&request_id.to_string(), decay_now()).unwrap();
        }
        assert_eq!(backlog_head_worker, active_a_worker);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_pinned_worker_conflict_with_allowed_ids_fails_early() {
        let (queue, _slots) = make_queue(1, 16, 256, Some(0.0));
        let (mut req, rx) = make_request("conflict", 256);
        req.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
        req.allowed_worker_ids = Some(HashSet::from([1]));

        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(
            resp,
            Err(KvSchedulerError::PinnedWorkerNotAllowed { worker_id: 0 })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_disallowed_worker_ids_fail_without_queueing() {
        let (queue, _slots) = make_queue(1, 16, 256, Some(0.0));
        let (mut req, rx) = make_request("disallowed", 256);
        req.allowed_worker_ids = Some(HashSet::from([999]));

        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(resp, Err(KvSchedulerError::NoEndpoints)));
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_incompatible_required_taints_fail_without_queueing() {
        let (queue, _slots, cfg_tx) = make_queue_with_sender(1, 16, 256, Some(0.0), None);
        let mut configs = HashMap::new();
        configs.insert(
            0_u64,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(256),
                taints: HashSet::from(["mdc-a".to_string()]),
                ..Default::default()
            },
        );
        cfg_tx.send(configs).unwrap();

        let (mut req, rx) = make_request("tainted", 256);
        req.routing_constraints = crate::protocols::RoutingConstraints {
            required_taints: HashSet::from(["mdc-b".to_string()]),
            preferred_taints: HashMap::new(),
        };

        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(resp, Err(KvSchedulerError::NoEndpoints)));
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn blocked_pinned_head_holds_its_class_until_it_can_run() {
        let (queue, slots) = make_queue(2, 16, 256, Some(0.0));

        let (mut first, first_rx) = make_request("pinned-1", 256);
        first.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        queue.enqueue(first).await;
        let first_resp = first_rx.await.unwrap().unwrap();
        assert_eq!(first_resp.best_worker, WorkerWithDpRank::new(1, 0));

        let (mut second, mut second_rx) = make_request("pinned-2", 256);
        second.pinned_worker = Some(WorkerWithDpRank::new(1, 0));
        queue.enqueue(second).await;
        assert_eq!(queue.pending_count(), 1);
        assert!(
            second_rx.try_recv().is_err(),
            "request should remain queued"
        );

        // Worker 0 is idle, but the class already has a backlog and its head is
        // pinned to the busy worker. One queue per class means this request
        // waits behind that head.
        let (mut other_worker, mut other_worker_rx) = make_request("pinned-0", 256);
        other_worker.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
        queue.enqueue(other_worker).await;
        assert_eq!(queue.pending_count(), 2);

        queue.update().await;

        assert_eq!(
            queue.pending_count(),
            2,
            "an undispatchable head blocks the rest of its class"
        );
        assert!(other_worker_rx.try_recv().is_err());
        assert!(second_rx.try_recv().is_err());

        slots
            .mark_prefill_completed(&"pinned-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"pinned-1".to_string(), decay_now()).unwrap();
        queue.update().await;

        let second_resp = second_rx
            .try_recv()
            .expect("the head should dispatch once its worker frees up")
            .expect("scheduling returned error");
        assert_eq!(second_resp.best_worker, WorkerWithDpRank::new(1, 0));
        let other_worker_resp = other_worker_rx
            .try_recv()
            .expect("the request behind the head should follow in arrival order")
            .expect("scheduling returned error");
        assert_eq!(other_worker_resp.best_worker, WorkerWithDpRank::new(0, 0));
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_queue_prefill_busy_check_ignores_untracked_prefill_tokens() {
        let (queue, slots) = make_queue(1, 16, 256, Some(0.0));

        let (mut req1, rx1) = make_request("req-1", 256);
        req1.track_prefill_tokens = false;
        queue.enqueue(req1).await;
        let _resp1 = rx1.await.unwrap().unwrap();
        assert_eq!(
            slots
                .active_tokens(decay_now())
                .get(&WorkerWithDpRank::new(0, 0))
                .copied(),
            Some(0)
        );

        let (req2, rx2) = make_request("req-2", 256);
        queue.enqueue(req2).await;
        let _resp2 = rx2.await.unwrap().unwrap();
        assert_eq!(queue.pending_count(), 0);

        let _ = slots.mark_prefill_completed(&"req-1".to_string(), decay_now());
        let _ = slots.free(&"req-1".to_string(), decay_now());
        let _ = slots.mark_prefill_completed(&"req-2".to_string(), decay_now());
        let _ = slots.free(&"req-2".to_string(), decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn update_refresh_can_change_selected_worker_after_queue_wait() {
        let block_size = 16u32;
        let isl = 64usize;
        let refresher = Arc::new(CountingRefresher {
            calls: AtomicUsize::new(0),
            last_retain_router_hint_chain: AtomicBool::new(false),
            response: RefreshedOverlap {
                router_hint_candidates: Some(RouterHintRootCandidates {
                    block_hashes: vec![
                        ExternalSequenceBlockHash(101),
                        ExternalSequenceBlockHash(102),
                    ],
                    owner_prefix_blocks: vec![(WorkerWithDpRank::new(1, 0), 2)],
                }),
                overlap: OverlapSignals {
                    tier_overlap_blocks: Default::default(),
                    effective_overlap_blocks: HashMap::from([
                        (WorkerWithDpRank::new(0, 0), 1.0),
                        (WorkerWithDpRank::new(1, 0), 9.0),
                    ]),
                    effective_cached_tokens: HashMap::from([
                        (WorkerWithDpRank::new(0, 0), 16),
                        (WorkerWithDpRank::new(1, 0), 144),
                    ]),
                },
            },
        });
        let (queue, slots) =
            make_queue_with_refresher(2, block_size, isl, Some(0.0), refresher.clone());

        let (mut req1, rx1) = make_request("req-1", isl);
        req1.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 3.0);
        req1.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 48);
        queue.enqueue(req1).await;
        let resp1 = rx1.await.expect("rx1 dropped").expect("req-1 failed");
        assert_eq!(resp1.best_worker, WorkerWithDpRank::new(0, 0));

        let (mut req2, rx2) = make_request("req-2", isl);
        req2.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(1, 0), 3.0);
        req2.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(1, 0), 48);
        queue.enqueue(req2).await;
        let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
        assert_eq!(resp2.best_worker, WorkerWithDpRank::new(1, 0));

        let (mut req3, rx3) = make_request("req-3", isl);
        req3.retain_router_hint_chain = true;
        req3.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 8.0);
        req3.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(1, 0), 2.0);
        req3.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 128);
        req3.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(1, 0), 32);
        queue
            .enqueue_with_block_hashes(req3, Some(vec![LocalBlockHash(42)]))
            .await;
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 0);

        tokio::time::advance(Duration::from_secs(11)).await;

        slots.free(&"req-1".to_string(), decay_now()).unwrap();
        slots.free(&"req-2".to_string(), decay_now()).unwrap();
        queue.update().await;

        let resp3 = rx3.await.expect("rx3 dropped").expect("req-3 failed");
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert!(
            refresher
                .last_retain_router_hint_chain
                .load(Ordering::Relaxed)
        );
        assert_eq!(resp3.best_worker, WorkerWithDpRank::new(1, 0));
        assert_eq!(resp3.effective_overlap_blocks, 9.0);
        assert_eq!(resp3.cached_tokens, 144);
        assert_eq!(
            resp3
                .router_hint_candidates
                .as_ref()
                .map(|candidates| candidates.owner_prefix_blocks.as_slice()),
            Some(&[(WorkerWithDpRank::new(1, 0), 2)][..])
        );
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn update_refresh_drops_router_hint_candidates_when_retention_disabled() {
        let block_size = 16u32;
        let isl = 64usize;
        let worker = WorkerWithDpRank::new(0, 0);
        let refresher = Arc::new(CountingRefresher {
            calls: AtomicUsize::new(0),
            last_retain_router_hint_chain: AtomicBool::new(true),
            response: RefreshedOverlap {
                router_hint_candidates: Some(RouterHintRootCandidates {
                    block_hashes: vec![ExternalSequenceBlockHash(101)],
                    owner_prefix_blocks: vec![(worker, 1)],
                }),
                overlap: OverlapSignals {
                    tier_overlap_blocks: Default::default(),
                    effective_overlap_blocks: HashMap::from([(worker, 5.0)]),
                    effective_cached_tokens: HashMap::from([(worker, 80)]),
                },
            },
        });
        let (queue, slots) =
            make_queue_with_refresher(1, block_size, isl, Some(0.0), refresher.clone());

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _ = rx1.await.expect("rx1 dropped").expect("req-1 failed");

        let (req2, rx2) = make_request("req-2", isl);
        queue
            .enqueue_with_block_hashes(req2, Some(vec![LocalBlockHash(42)]))
            .await;
        assert_eq!(queue.pending_count(), 1);

        tokio::time::advance(Duration::from_secs(11)).await;
        slots.free(&"req-1".to_string(), decay_now()).unwrap();
        queue.update().await;

        let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert!(
            !refresher
                .last_retain_router_hint_chain
                .load(Ordering::Relaxed)
        );
        assert_eq!(resp2.best_worker, worker);
        assert_eq!(resp2.effective_overlap_blocks, 5.0);
        assert_eq!(resp2.cached_tokens, 80);
        assert!(resp2.router_hint_candidates.is_none());
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn selected_request_dispatches_after_refresh_if_worker_becomes_busy() {
        let block_size = 16u32;
        let isl = 64usize;
        let worker = WorkerWithDpRank::new(0, 0);
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::from_overlap(
            OverlapSignals {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::from([(worker, 7.0)]),
                effective_cached_tokens: HashMap::from([(worker, 56)]),
            },
        )));
        let (queue, slots) = make_queue_with_blocking_refresher(
            1,
            block_size,
            isl,
            Some(0.0),
            refresher.clone(),
            ADMISSION_CHANNEL_CAPACITY,
        );

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _ = rx1.await.expect("rx1 dropped").expect("req-1 failed");

        let (mut req2, rx2) = make_request("req-2", isl);
        req2.overlap
            .effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 4.0);
        req2.overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 64);
        queue
            .enqueue_with_block_hashes(req2, Some(vec![LocalBlockHash(42)]))
            .await;
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(
            queue.class_queue_stats(0).unwrap().pending_cached_tokens,
            64
        );

        slots
            .mark_prefill_completed(&"req-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"req-1".to_string(), decay_now()).unwrap();

        tokio::time::advance(Duration::from_secs(11)).await;

        let update = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move {
                queue.update().await;
            })
        };
        refresher.wait_for_calls(1).await;
        assert_eq!(
            queue.pending_count(),
            0,
            "DRR-selected request must be removed before refresh"
        );
        assert_eq!(
            queue.class_queue_stats(0).unwrap().pending_cached_tokens,
            0,
            "queue counters must reflect the irrevocable dequeue"
        );

        slots
            .add_request(
                SequenceRequest {
                    request_id: "occupy-during-refresh".to_string(),
                    token_sequence: None,
                    track_prefill_tokens: true,
                    expected_output_tokens: None,
                    prefill_load_hint: Some(PrefillLoadHint {
                        initial_effective_prefill_tokens: isl,
                        expected_prefill_duration: None,
                    }),
                    worker,
                    lora_name: None,
                },
                decay_now(),
            )
            .unwrap();

        refresher.release_one();
        update.await.unwrap();

        let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 1);
        assert_eq!(resp2.best_worker, worker);
        assert_eq!(resp2.effective_overlap_blocks, 7.0);
        assert_eq!(resp2.cached_tokens, 56);
        assert_eq!(queue.pending_count(), 0);

        for request_id in ["occupy-during-refresh", "req-2"] {
            slots
                .mark_prefill_completed(&request_id.to_string(), decay_now())
                .unwrap();
            slots.free(&request_id.to_string(), decay_now()).unwrap();
        }
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn cancelled_enqueue_wait_keeps_cleanup_behind_command() {
        let block_size = 16u32;
        let isl = 64usize;
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) =
            make_queue_with_blocking_refresher(1, block_size, isl, Some(0.0), refresher.clone(), 1);

        let (active, active_rx) = make_request("active", isl);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (queued, queued_rx) = make_request("queued", isl);
        queue
            .enqueue_with_block_hashes(queued, Some(vec![LocalBlockHash(42)]))
            .await;
        slots.free(&"active".to_owned(), decay_now()).unwrap();
        tokio::time::advance(Duration::from_secs(11)).await;

        let update = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move { queue.update().await })
        };
        refresher.wait_for_calls(1).await;

        let (cancelled, cancelled_rx) = make_request("cancelled", isl);
        let lease = queue
            .new_request_lifecycle_lease(Some("cancelled"))
            .unwrap();
        let enqueue = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move {
                queue
                    .enqueue_with_block_hashes_and_lease(cancelled, None, Some(lease))
                    .await
            })
        };
        tokio::task::yield_now().await;
        assert_eq!(queue.admission_tx.capacity(), 0);
        drop(cancelled_rx);
        enqueue.abort();
        assert!(enqueue.await.unwrap_err().is_cancelled());

        // Cancellation drops the acknowledgement receiver, but the lease remains
        // inside the accepted command until the actor establishes request ownership.
        refresher.release_one();
        update.await.unwrap();
        queue.update().await;
        assert_eq!(queue.pending_count(), 0);

        queued_rx.await.unwrap().unwrap();
        slots.free(&"queued".to_owned(), decay_now()).unwrap();
        slots.assert_completely_drained(decay_now());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn continuation_drain_does_not_self_send_into_saturated_actor_channel() {
        let block_size = 16u32;
        let isl = 64usize;
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) =
            make_queue_with_blocking_refresher(1, block_size, isl, Some(0.0), refresher.clone(), 1);

        let (active, active_rx) = make_request("active", isl);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (queued, queued_rx) = make_request("queued", isl);
        queue
            .enqueue_with_block_hashes(queued, Some(vec![LocalBlockHash(42)]))
            .await;
        slots
            .mark_prefill_completed(&"active".to_string(), decay_now())
            .unwrap();
        slots.free(&"active".to_string(), decay_now()).unwrap();
        tokio::time::advance(Duration::from_secs(11)).await;

        let update = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move { queue.update().await })
        };
        refresher.wait_for_calls(1).await;

        let (following, following_rx) = make_request("following", isl);
        let enqueue = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move { queue.enqueue(following).await })
        };
        tokio::task::yield_now().await;
        assert_eq!(
            queue.admission_tx.capacity(),
            0,
            "test must saturate the actor command channel"
        );

        refresher.release_one();
        tokio::time::timeout(Duration::from_secs(1), update)
            .await
            .expect("update deadlocked with a full actor command channel")
            .unwrap();
        queued_rx.await.unwrap().unwrap();

        slots
            .mark_prefill_completed(&"queued".to_string(), decay_now())
            .unwrap();
        slots.free(&"queued".to_string(), decay_now()).unwrap();
        queue.update().await;
        following_rx.await.unwrap().unwrap();
        enqueue.await.unwrap();
    }
}
