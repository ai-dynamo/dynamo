// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rustc_hash::FxHashSet;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::config::RouterQueuePolicy;
use super::filter::RoutingEligibility;
use super::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, read_overlap_refresh_after, refresh_overlap,
};
use super::policy_config::{PolicyClassConfig, PolicyProfile};
use super::policy_queue::{PolicyQueue, QueueSnapshot};
use super::prefill_load::{PrefillLoadEstimator, effective_prefill_tokens};
use super::queue_admission::{
    AdmissionAction, AdmissionDecision, AdmissionTicket, ClassAdmissionAction,
    PolicyClassAdmissionController, PolicyClassAdmissionStrategies, RequestProgressUpdater,
    WorkerEligibility, WorkerEligibilitySnapshot, WorkerPlacement,
};
use super::selector::{DefaultWorkerSelector, WorkerSelector};
use super::types::{
    KvSchedulerError, OverloadedWorkerProvider, RequestOutcome, SchedulingContext,
    SchedulingRequest, SchedulingResponse,
};
use crate::protocols::{
    LocalBlockHash, PrefillLoadHint, WorkerConfigLike, WorkerId, WorkerWithDpRank,
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

struct QueuedRequest {
    request: SchedulingRequest,
    enqueue_at: Instant,
    block_hashes: Option<Vec<LocalBlockHash>>,
    admission: Option<RequestAdmission>,
}

struct RequestAdmission {
    ticket: AdmissionTicket,
    progress: RequestProgressUpdater,
}

#[allow(clippy::large_enum_variant)]
enum AdmissionCommand {
    Enqueue {
        request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        ack_tx: oneshot::Sender<()>,
    },
    Update {
        worker: Option<WorkerWithDpRank>,
        ack_tx: oneshot::Sender<()>,
    },
    Reconcile {
        force: bool,
        ack_tx: oneshot::Sender<()>,
    },
    Dispatched {
        request_id: String,
    },
    Cleanup {
        request_id: String,
        outcome: RequestOutcome,
    },
    Finished {
        request_id: String,
        outcome: RequestOutcome,
    },
}

#[derive(Debug, Clone, Copy)]
struct TrackedAdmission {
    ticket: AdmissionTicket,
    worker: Option<WorkerWithDpRank>,
    dispatched: bool,
}

#[derive(Default)]
struct AdmissionCleanup {
    dirty: Mutex<HashMap<String, RequestOutcome>>,
    pending: AtomicBool,
}

impl AdmissionCleanup {
    fn enqueue(&self, request_id: String, outcome: RequestOutcome) {
        self.dirty.lock().unwrap().insert(request_id, outcome);
        self.pending.store(true, AtomicOrdering::Release);
    }

    fn drain(&self) -> HashMap<String, RequestOutcome> {
        if !self.pending.load(AtomicOrdering::Acquire)
            || !self.pending.swap(false, AtomicOrdering::AcqRel)
        {
            return HashMap::new();
        }
        std::mem::take(&mut *self.dirty.lock().unwrap())
    }
}

/// Single-owner cleanup lease for one scheduler-tracked request.
///
/// Ownership moves from worker selection into the response stream. Dropping
/// either phase reports one terminal outcome to the actor, falling back to the
/// coalesced dirty set only when the actor channel is full.
#[must_use = "dropping the lease reports the request outcome to the scheduler actor"]
pub struct AdmissionLease {
    cleanup: Arc<AdmissionCleanup>,
    actor_tx: mpsc::Sender<AdmissionCommand>,
    request_id: Option<String>,
    outcome: RequestOutcome,
}

impl std::fmt::Debug for AdmissionLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AdmissionLease")
            .field("request_id", &self.request_id)
            .field("outcome", &self.outcome)
            .finish_non_exhaustive()
    }
}

impl AdmissionLease {
    pub fn mark_completed(&mut self, context_tokens: usize) {
        self.outcome = RequestOutcome::Completed { context_tokens };
    }

    pub fn disarm(&mut self) {
        self.request_id = None;
    }
}

impl Drop for AdmissionLease {
    fn drop(&mut self) {
        let Some(request_id) = self.request_id.take() else {
            return;
        };
        let command = AdmissionCommand::Cleanup {
            request_id,
            outcome: self.outcome,
        };
        match self.actor_tx.try_send(command) {
            Ok(()) | Err(mpsc::error::TrySendError::Closed(_)) => {}
            Err(mpsc::error::TrySendError::Full(command)) => {
                let AdmissionCommand::Cleanup {
                    request_id,
                    outcome,
                } = command
                else {
                    unreachable!("lease cleanup only sends terminal outcomes")
                };
                self.cleanup.enqueue(request_id, outcome);
            }
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
    admission: PolicyClassAdmissionController,
    tracked_admissions: HashMap<String, TrackedAdmission>,
    cleanup: Arc<AdmissionCleanup>,
    profile: PolicyProfile,
    queue_recheck_interval: Duration,
    next_queue_recheck: Instant,
    pending_count: Arc<AtomicUsize>,
    pending_isl_tokens: Arc<AtomicUsize>,
    class_counters: Arc<Vec<ClassQueueCounters>>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    start_time: Instant,
    block_size: u32,
    selector: Sel,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    overlap_scores_refresh: Option<Arc<RF>>,
    overlap_refresh_after: Option<Duration>,
    overloaded_worker_provider: Option<OverloadedWorkerProvider>,
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
    admission_enabled: bool,
    supports_overlap_refresh: bool,
    _marker: PhantomData<(Sel, RF)>,
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
        )
        .expect("synthetic policy profile does not require admission strategies")
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
    ) -> Result<Self, KvSchedulerError> {
        Self::new_with_policy_profile_and_admission_strategies(
            slots,
            workers_with_configs,
            profile,
            block_size,
            selector,
            prefill_load_estimator,
            overlap_scores_refresh,
            overloaded_worker_provider,
            Duration::from_secs(60),
            PolicyClassAdmissionStrategies::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_policy_profile_and_admission_strategies(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        profile: PolicyProfile,
        block_size: u32,
        selector: Sel,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        queue_recheck_interval: Duration,
        admission_strategies: PolicyClassAdmissionStrategies,
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
            queue_recheck_interval,
            admission_strategies,
            ADMISSION_CHANNEL_CAPACITY,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_policy_profile_and_capacity(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        profile: PolicyProfile,
        block_size: u32,
        selector: Sel,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
        queue_recheck_interval: Duration,
        admission_strategies: PolicyClassAdmissionStrategies,
        admission_channel_capacity: usize,
    ) -> Result<Self, KvSchedulerError> {
        let admission_enabled = !admission_strategies.is_empty();
        let admission = PolicyClassAdmissionController::new(
            &profile,
            queue_recheck_interval,
            admission_strategies,
        )?;
        let queueing_enabled = profile
            .classes()
            .iter()
            .any(PolicyClassConfig::queueing_enabled)
            || admission_enabled;
        for class in profile.classes() {
            tracing::info!(
                policy_class = class.name,
                queue_policy = %class.queue_policy,
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
        let now = Instant::now();
        let actor = SchedulerQueueActor {
            pending: PolicyQueue::new(profile.clone()),
            admission,
            tracked_admissions: HashMap::new(),
            cleanup: Arc::clone(&cleanup),
            profile,
            queue_recheck_interval,
            next_queue_recheck: now + queue_recheck_interval,
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
            admission_enabled,
            supports_overlap_refresh: overlap_refresh_after.is_some(),
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

    /// Enqueue a new request.
    /// If queueing is disabled or workers have capacity, schedule immediately.
    /// Otherwise park in the pending heap.
    pub async fn enqueue(&self, request: SchedulingRequest) {
        self.enqueue_with_block_hashes(request, None).await;
    }

    pub async fn enqueue_with_block_hashes(
        &self,
        mut request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) {
        let eligibility = request.eligibility();

        if let Err(error) = eligibility.validate_pinned_worker_allowed() {
            request.respond(Err(error));
            return;
        }

        let (ack_tx, ack_rx) = oneshot::channel();
        let command = AdmissionCommand::Enqueue {
            request,
            block_hashes: self.prepare_block_hashes_for_refresh(block_hashes),
            ack_tx,
        };

        if let Err(error) = self.admission_tx.send(command).await {
            let AdmissionCommand::Enqueue { mut request, .. } = error.0 else {
                return;
            };
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
            return;
        }

        if ack_rx.await.is_err() {
            tracing::warn!("scheduler queue actor dropped enqueue acknowledgement");
        }
    }

    /// Called on prefill_complete/free. Drains pending requests while workers have capacity.
    /// Each scheduled request updates active_tokens via add_request, so the prefill-busy check
    /// sees fresh state on the next iteration.
    pub async fn update(&self) {
        self.update_after(None).await;
    }

    pub(crate) async fn update_worker(&self, worker: WorkerWithDpRank) {
        self.update_after(Some(worker)).await;
    }

    async fn update_after(&self, worker: Option<WorkerWithDpRank>) {
        if !self.queueing_enabled {
            return;
        }

        let (ack_tx, ack_rx) = oneshot::channel();
        if self
            .admission_tx
            .send(AdmissionCommand::Update { worker, ack_tx })
            .await
            .is_ok()
        {
            let _ = ack_rx.await;
        }
    }

    pub(crate) async fn dispatched(&self, request_id: &str) {
        if !self.admission_enabled {
            return;
        }
        let _ = self
            .admission_tx
            .send(AdmissionCommand::Dispatched {
                request_id: request_id.to_owned(),
            })
            .await;
    }

    pub(crate) fn cancellation_guard(&self, request_id: Option<&str>) -> Option<AdmissionLease> {
        let request_id = request_id?.to_owned();
        Some(AdmissionLease {
            cleanup: Arc::clone(&self.cleanup),
            actor_tx: self.admission_tx.clone(),
            request_id: Some(request_id),
            outcome: RequestOutcome::Aborted,
        })
    }

    pub(crate) async fn finish(&self, request_id: &str, outcome: RequestOutcome) {
        if !self.admission_enabled {
            return;
        }
        let _ = self
            .admission_tx
            .send(AdmissionCommand::Finished {
                request_id: request_id.to_owned(),
                outcome,
            })
            .await;
    }

    #[cfg(test)]
    pub(crate) async fn reconcile(&self) {
        self.send_reconcile(true).await;
    }

    pub(crate) async fn periodic_reconcile(&self) {
        self.send_reconcile(false).await;
    }

    async fn send_reconcile(&self, force: bool) {
        if !self.admission_enabled {
            self.update().await;
            return;
        }

        let (ack_tx, ack_rx) = oneshot::channel();
        if self
            .admission_tx
            .send(AdmissionCommand::Reconcile { force, ack_tx })
            .await
            .is_ok()
        {
            let _ = ack_rx.await;
        }
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
        while let Some(command) = rx.recv().await {
            match command {
                AdmissionCommand::Enqueue {
                    request,
                    block_hashes,
                    ack_tx,
                } => {
                    let made_ready =
                        self.handle_enqueue(request, block_hashes) | self.drain_cleanup();
                    if made_ready {
                        self.handle_update(None).await;
                    }
                    let _ = ack_tx.send(());
                }
                AdmissionCommand::Update { worker, ack_tx } => {
                    self.handle_update(worker).await;
                    if self.drain_cleanup() {
                        self.handle_update(None).await;
                    }
                    let _ = ack_tx.send(());
                }
                AdmissionCommand::Reconcile { force, ack_tx } => {
                    self.handle_reconcile(force).await;
                    if self.drain_cleanup() {
                        self.handle_update(None).await;
                    }
                    let _ = ack_tx.send(());
                }
                AdmissionCommand::Dispatched { request_id } => {
                    if self.handle_dispatched(&request_id) | self.drain_cleanup() {
                        self.handle_update(None).await;
                    }
                }
                AdmissionCommand::Cleanup {
                    request_id,
                    outcome,
                } => {
                    if self.handle_cleanup(&request_id, outcome) | self.drain_cleanup() {
                        self.handle_update(None).await;
                    }
                }
                AdmissionCommand::Finished {
                    request_id,
                    outcome,
                } => {
                    if self.handle_finished(&request_id, outcome) | self.drain_cleanup() {
                        self.handle_update(None).await;
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
    }

    fn handle_enqueue(
        &mut self,
        mut request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) -> bool {
        let decay_now = Instant::now();
        // Synthetic and explicit selections avoid cache work. Family classification
        // samples overlap once and reuses it if the request enters queue storage.
        let (class_index, mut snapshot) = if let Some(class_index) = self
            .profile
            .direct_class_index(request.policy_class.as_deref())
        {
            (class_index, None)
        } else {
            let workers = self.workers_with_configs.borrow();
            let snapshot = Self::snapshot_for_with(&request, &workers);
            let class_index = self
                .profile
                .resolve_class_index(request.policy_class.as_deref(), snapshot.uncached_tokens);
            (class_index, Some(snapshot))
        };

        if let Some(request_id) = request.mode.admission_request_id()
            && self.tracked_admissions.contains_key(request_id)
        {
            request.respond(Err(KvSchedulerError::BookingFailed(format!(
                "request {request_id} already has an active admission"
            ))));
            return false;
        }

        let mut admission = if request.mode.admission_request_id().is_some()
            && self.admission.has_strategy(class_index)
        {
            let allowed_worker_ids = request.allowed_worker_ids.clone();
            let pinned_worker = request.pinned_worker;
            let routing_constraints = request.routing_constraints.clone();
            let workers = self.workers_with_configs.clone();
            let overloaded_worker_provider = self.overloaded_worker_provider.clone();
            let worker_eligibility = WorkerEligibility::new(move || {
                let workers = workers.borrow();
                let overloaded_worker_ids = overloaded_worker_provider
                    .as_ref()
                    .and_then(|provider| provider());
                let structural_eligibility = RoutingEligibility::new(
                    allowed_worker_ids.as_ref(),
                    None,
                    pinned_worker,
                    &routing_constraints,
                );
                let mut structural_workers = FxHashSet::default();
                structural_eligibility.for_each_eligible_worker_rank(&workers, |worker, _| {
                    structural_workers.insert(worker);
                });
                let Some(overloaded_worker_ids) = overloaded_worker_ids.as_ref() else {
                    return WorkerEligibilitySnapshot::new(structural_workers);
                };
                let mut available_workers = structural_workers.clone();
                available_workers
                    .retain(|worker| !overloaded_worker_ids.contains(&worker.worker_id));
                WorkerEligibilitySnapshot::with_availability(structural_workers, available_workers)
            });
            self.admission
                .admit(
                    class_index,
                    request.session_id.as_deref(),
                    request.isl_tokens,
                    worker_eligibility,
                )
                .map(|(ticket, progress, decision)| {
                    (RequestAdmission { ticket, progress }, decision)
                })
        } else {
            None
        };
        let mut deferred = false;
        if let Some((request_admission, decision)) = admission.as_ref() {
            match decision {
                AdmissionDecision::Bypass => admission = None,
                AdmissionDecision::Ready(placement) => {
                    if let Err(error) = apply_admission_placement(&mut request, *placement) {
                        request.respond(Err(error));
                        return self.abort_admission(request_admission.ticket);
                    }
                    if matches!(placement, WorkerPlacement::Exact(_)) {
                        // Classification stays in the family selected above; queue cost and
                        // accounting follow the strategy's exact worker.
                        snapshot = None;
                    }
                }
                AdmissionDecision::Defer => deferred = true,
            }
        }

        let class = self.profile.class(class_index);
        let should_queue = deferred
            || self.should_queue(class_index, class, || {
                self.all_workers_prefill_busy(class, request.eligibility(), decay_now)
            });
        if !should_queue {
            return self.admit_one(
                request,
                decay_now,
                admission.map(|(admission, _)| admission),
            );
        }

        let snapshot = snapshot.unwrap_or_else(|| self.snapshot_for(&request));
        tracing::debug!(policy_class = class.name, deferred, "queueing request");
        let arrival_offset = self.start_time.elapsed().as_secs_f64();
        let priority_jump = request.priority_jump;
        let strict_priority = request.strict_priority;
        let placement = request
            .pinned_worker
            .map_or(WorkerPlacement::Any, WorkerPlacement::Exact);
        let deferred_id = deferred
            .then(|| admission.as_ref().map(|(admission, _)| admission.ticket.id))
            .flatten();
        let tracked_admission = admission.as_ref().map(|(admission, _)| {
            (
                request
                    .mode
                    .tracked_request_id()
                    .expect("admitted request is tracked")
                    .to_owned(),
                admission.ticket,
            )
        });
        let queued = QueuedRequest {
            request,
            enqueue_at: decay_now,
            block_hashes,
            admission: admission.map(|(admission, _)| admission),
        };
        let worker_count = self.workers_with_configs.borrow().len();
        let enqueue = match deferred_id {
            Some(admission_id) => self.pending.enqueue_deferred(
                class_index,
                worker_count,
                snapshot,
                arrival_offset,
                priority_jump,
                strict_priority,
                admission_id,
                queued,
            ),
            None => self.pending.enqueue(
                class_index,
                worker_count,
                snapshot,
                arrival_offset,
                priority_jump,
                strict_priority,
                placement,
                queued,
            ),
        };
        if let Err((rejection, queued)) = enqueue {
            let made_ready = queued
                .admission
                .as_ref()
                .is_some_and(|admission| self.abort_admission(admission.ticket));
            let mut request = queued.request;
            request.respond(Err(KvSchedulerError::QueueRejected(rejection)));
            return made_ready;
        }
        if let Some((request_id, ticket)) = tracked_admission {
            self.tracked_admissions.insert(
                request_id,
                TrackedAdmission {
                    ticket,
                    worker: None,
                    dispatched: false,
                },
            );
        }
        self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
        self.pending_isl_tokens
            .fetch_add(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        self.add_class_counters(class_index, snapshot);
        false
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

    fn snapshot_for(&self, request: &SchedulingRequest) -> QueueSnapshot {
        let workers = self.workers_with_configs.borrow();
        Self::snapshot_for_with(request, &workers)
    }

    fn snapshot_for_with(
        request: &SchedulingRequest,
        workers: &HashMap<WorkerId, C>,
    ) -> QueueSnapshot {
        // Cache overlap is sampled once and reused for classification, queue
        // limits, ordering, DRR cost, and counters.
        let context = SchedulingContext::new(request, workers);
        QueueSnapshot::new(request.isl_tokens, context.best_cached_tokens())
    }

    fn handle_dispatched(&mut self, request_id: &str) -> bool {
        let Some(tracked) = self.tracked_admissions.get_mut(request_id) else {
            return false;
        };
        if tracked.dispatched {
            return false;
        }
        let Some(worker) = tracked.worker else {
            tracing::debug!(%request_id, "Ignoring dispatch before queue admission");
            return false;
        };
        tracked.dispatched = true;
        let actions = self.admission.dispatched(tracked.ticket, worker);
        self.apply_admission_actions(actions)
    }

    fn handle_cleanup(&mut self, request_id: &str, outcome: RequestOutcome) -> bool {
        let Some(tracked) = self.tracked_admissions.get(request_id).copied() else {
            let owned_request_id = request_id.to_owned();
            let rollback_booking = self.slots.request_worker(&owned_request_id).is_some();
            if rollback_booking
                && let Err(error) = self.slots.free(&owned_request_id, Instant::now())
            {
                tracing::error!(%request_id, %error, "Failed to release dropped scheduler booking");
            }

            let mut removed_pending = false;
            for class_index in 0..self.profile.classes().len() {
                let removed = self.pending.take_if_in_class(class_index, |queued| {
                    queued.request.mode.tracked_request_id() == Some(request_id)
                });
                removed_pending |= !removed.is_empty();
                for entry in removed {
                    self.subtract_pending_counters(class_index, entry.snapshot());
                }
            }
            return rollback_booking || removed_pending;
        };

        let owned_request_id = request_id.to_owned();
        let rollback_booking =
            tracked.worker.is_some() && self.slots.request_worker(&owned_request_id).is_some();
        if rollback_booking && let Err(error) = self.slots.free(&owned_request_id, Instant::now()) {
            tracing::error!(%request_id, %error, "Failed to release dropped scheduler booking");
        }
        rollback_booking | self.handle_finished(request_id, outcome)
    }

    fn drain_cleanup(&mut self) -> bool {
        let dirty = self.cleanup.drain();
        if dirty.is_empty() {
            return false;
        }

        let mut made_ready = false;
        let mut ready_by_class: HashMap<usize, FxHashSet<_>> = HashMap::new();
        let mut unmanaged_request_ids = FxHashSet::default();
        for (request_id, outcome) in dirty {
            let tracked = self.tracked_admissions.get(&request_id).copied();
            if tracked.is_none_or(|tracked| tracked.worker.is_some()) {
                let owned_request_id = request_id.clone();
                if self.slots.request_worker(&owned_request_id).is_some() {
                    if let Err(error) = self.slots.free(&owned_request_id, Instant::now()) {
                        tracing::error!(%request_id, %error, "Failed to release dropped scheduler booking");
                    }
                    made_ready = true;
                }
            }

            let Some(tracked) = tracked else {
                unmanaged_request_ids.insert(request_id);
                continue;
            };
            if tracked.worker.is_some() {
                made_ready |= self.handle_finished(&request_id, outcome);
                continue;
            }

            self.tracked_admissions.remove(&request_id);
            if let Some(entry) = self
                .pending
                .remove_deferred(tracked.ticket.class_index, tracked.ticket.id)
            {
                self.subtract_pending_counters(entry.class_index(), entry.snapshot());
            } else {
                ready_by_class
                    .entry(tracked.ticket.class_index)
                    .or_default()
                    .insert(tracked.ticket.id);
            }
            made_ready |= self.finish_admission(tracked.ticket, outcome);
        }

        // One heap rebuild per affected class keeps cancellation storms O(classes * queue),
        // while counters are released in the same actor turn as the lifecycle event.
        for (class_index, tickets) in ready_by_class {
            let removed = self.pending.take_if_in_class(class_index, |queued| {
                queued
                    .admission
                    .as_ref()
                    .is_some_and(|admission| tickets.contains(&admission.ticket.id))
            });
            debug_assert_eq!(removed.len(), tickets.len());
            for entry in removed {
                self.subtract_pending_counters(class_index, entry.snapshot());
            }
        }
        if !unmanaged_request_ids.is_empty() {
            for class_index in 0..self.profile.classes().len() {
                let removed = self.pending.take_if_in_class(class_index, |queued| {
                    queued
                        .request
                        .mode
                        .tracked_request_id()
                        .is_some_and(|request_id| unmanaged_request_ids.contains(request_id))
                });
                made_ready |= !removed.is_empty();
                for entry in removed {
                    self.subtract_pending_counters(class_index, entry.snapshot());
                }
            }
        }
        made_ready
    }

    fn handle_finished(&mut self, request_id: &str, outcome: RequestOutcome) -> bool {
        let Some(tracked) = self.tracked_admissions.remove(request_id) else {
            return false;
        };
        if tracked.worker.is_none() {
            if let Some(entry) = self
                .pending
                .remove_deferred(tracked.ticket.class_index, tracked.ticket.id)
            {
                self.subtract_pending_counters(entry.class_index(), entry.snapshot());
            } else {
                let removed = self
                    .pending
                    .take_if_in_class(tracked.ticket.class_index, |queued| {
                        queued.request.mode.tracked_request_id() == Some(request_id)
                    });
                debug_assert_eq!(
                    removed.len(),
                    1,
                    "pending admission must have one queue entry"
                );
                for entry in removed {
                    self.subtract_pending_counters(entry.class_index(), entry.snapshot());
                }
            }
        }
        self.finish_admission(tracked.ticket, outcome)
    }

    fn finish_admission(&mut self, ticket: AdmissionTicket, outcome: RequestOutcome) -> bool {
        match outcome {
            RequestOutcome::Completed { context_tokens } => {
                self.complete_admission(ticket, context_tokens)
            }
            RequestOutcome::Aborted => self.abort_admission(ticket),
        }
    }

    fn complete_admission(&mut self, ticket: AdmissionTicket, context_tokens: usize) -> bool {
        let actions = self.admission.completed(ticket, context_tokens);
        self.apply_admission_actions(actions)
    }

    fn abort_admission(&mut self, ticket: AdmissionTicket) -> bool {
        let actions = self.admission.aborted(ticket);
        self.apply_admission_actions(actions)
    }

    fn apply_admission_actions(
        &mut self,
        actions: impl IntoIterator<Item = ClassAdmissionAction>,
    ) -> bool {
        let mut made_ready = false;
        let mut actions: VecDeque<_> = actions.into_iter().collect();
        while let Some(class_action) = actions.pop_front() {
            let AdmissionAction::MakeReady { id, placement } = class_action.action;

            let prepared = {
                let Some(queued) = self
                    .pending
                    .deferred_payload_mut(class_action.class_index, id)
                else {
                    tracing::debug!(
                        admission_id = id.get(),
                        "Ignoring unknown make-ready action"
                    );
                    continue;
                };
                match apply_admission_placement(&mut queued.request, placement) {
                    Err(error) => Err(error),
                    Ok(()) => {
                        let effective_placement = queued
                            .request
                            .pinned_worker
                            .map_or(placement, WorkerPlacement::Exact);
                        let replacement =
                            if matches!(effective_placement, WorkerPlacement::Exact(_)) {
                                let workers = self.workers_with_configs.borrow();
                                Some((
                                    Self::snapshot_for_with(&queued.request, &workers),
                                    queued.request.priority_jump,
                                ))
                            } else {
                                None
                            };
                        Ok((effective_placement, replacement))
                    }
                }
            };

            let (effective_placement, replacement) = match prepared {
                Ok(prepared) => prepared,
                Err(error) => {
                    let Some(entry) = self.pending.remove_deferred(class_action.class_index, id)
                    else {
                        continue;
                    };
                    let snapshot = entry.snapshot();
                    self.subtract_pending_counters(class_action.class_index, snapshot);
                    let mut queued = entry.into_payload();
                    if let Some(admission) = queued.admission {
                        if let Some(request_id) = queued.request.mode.tracked_request_id() {
                            self.tracked_admissions.remove(request_id);
                        }
                        actions.extend(self.admission.aborted(admission.ticket));
                    }
                    queued.request.respond(Err(error));
                    continue;
                }
            };

            let new_snapshot = replacement.map(|(snapshot, _)| snapshot);
            if let Some(old_snapshot) = self.pending.make_ready(
                class_action.class_index,
                id,
                effective_placement,
                replacement,
            ) {
                if let Some(new_snapshot) = new_snapshot {
                    self.replace_pending_snapshot_counters(
                        class_action.class_index,
                        old_snapshot,
                        new_snapshot,
                    );
                }
                made_ready = true;
            } else {
                tracing::debug!(
                    admission_id = id.get(),
                    "Ignoring duplicate make-ready action"
                );
            }
        }
        made_ready
    }

    fn subtract_pending_counters(&self, class_index: usize, snapshot: QueueSnapshot) {
        self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
        self.pending_isl_tokens
            .fetch_sub(snapshot.raw_isl_tokens, AtomicOrdering::Relaxed);
        self.subtract_class_counters(class_index, snapshot);
    }

    fn replace_pending_snapshot_counters(
        &self,
        class_index: usize,
        old: QueueSnapshot,
        new: QueueSnapshot,
    ) {
        debug_assert_eq!(old.raw_isl_tokens, new.raw_isl_tokens);
        let counter = &self.class_counters[class_index].pending_cached_tokens;
        if new.cached_tokens >= old.cached_tokens {
            counter.fetch_add(
                new.cached_tokens - old.cached_tokens,
                AtomicOrdering::Relaxed,
            );
        } else {
            counter.fetch_sub(
                old.cached_tokens - new.cached_tokens,
                AtomicOrdering::Relaxed,
            );
        }
    }

    async fn handle_reconcile(&mut self, force: bool) {
        let now = Instant::now();
        let queue_due = force || now >= self.next_queue_recheck;
        if !force && queue_due {
            self.next_queue_recheck = now + self.queue_recheck_interval;
        }
        let actions = self.admission.reconcile(now, force);
        let made_ready = self.apply_admission_actions(actions);
        if queue_due || made_ready {
            self.handle_update(None).await;
        }
    }

    async fn handle_update(&mut self, worker: Option<WorkerWithDpRank>) {
        if self.pending.pending_count() == 0 {
            return;
        }

        if let Some(worker) = worker {
            self.pending.recheck_worker(worker);
        } else {
            // ponytail: periodic/topology updates use the safe full fallback; thread worker IDs
            // through replica updates if this scan becomes measurable.
            self.pending.recheck_all_workers();
        }

        // Continuation draining stays actor-local; never self-send through the
        // bounded command channel while processing an update.
        loop {
            let decay_now = Instant::now();
            let active_tokens = self.slots.active_tokens(decay_now);
            let popped = {
                let configs = self.workers_with_configs.borrow();
                self.pending.pop_next(|_, class, queued| {
                    // TODO: This preserves head-of-line blocking within each policy
                    // class. A blocked constrained head can stall later entries in
                    // that class until a bounded non-HOL strategy is introduced.
                    !Self::all_workers_prefill_busy_with(
                        &active_tokens,
                        &configs,
                        class,
                        queued.request.eligibility(),
                    )
                })
            };
            let Some(mut popped) = popped else {
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
                queued.enqueue_at,
                decay_now,
            )
            .await;
            let wait_ms = queued.enqueue_at.elapsed().as_millis() as u64;
            if let Some(overlap) = refreshed {
                tracing::info!(
                    request_id = queued.request.mode.request_id().unwrap_or("unknown"),
                    wait_ms,
                    "refreshed overlap scores after long queue wait"
                );
                queued.request.overlap = overlap;
            }
            let admit_now = Instant::now();
            let class_index = popped.class_index();
            let class = self.profile.class(class_index);
            let queued = popped.into_payload();
            let admission = queued.admission;
            let request = queued.request;
            tracing::debug!(
                policy_class = class.name,
                "scheduling request from pending queue"
            );
            self.admit_one(request, admit_now, admission);
        }
    }

    /// Run the full scheduling pipeline for a single request:
    /// compute projected load -> select worker -> book tracked state -> respond.
    fn admit_one(
        &mut self,
        mut request: SchedulingRequest,
        decay_now: Instant,
        admission: Option<RequestAdmission>,
    ) -> bool {
        let admission_key = admission.as_ref().map(|_| {
            request
                .mode
                .tracked_request_id()
                .expect("admitted request is tracked")
                .to_owned()
        });
        request.worker_loads = self
            .slots
            .project_worker_loads(request.token_seq.as_deref(), decay_now);

        let selection = {
            let workers = self.workers_with_configs.borrow();
            let overloaded_worker_ids = self
                .overloaded_worker_provider
                .as_ref()
                .and_then(|provider| provider());
            let eligibility = request.eligibility_with_overloaded(overloaded_worker_ids.as_ref());
            self.selector
                .select_worker(&workers, &request, eligibility, self.block_size)
                .map(|selection| {
                    let config = workers
                        .get(&selection.worker.worker_id)
                        .expect("selected worker config must exist");
                    let selected_worker_tiers = request
                        .overlap
                        .selected_worker_tiers(selection.worker, config);
                    (selection, selected_worker_tiers)
                })
        };

        let (selection, selected_worker_tiers) = match selection {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                request.respond(Err(e));
                if let Some(request_id) = admission_key.as_deref() {
                    self.tracked_admissions.remove(request_id);
                }
                return admission
                    .as_ref()
                    .is_some_and(|admission| self.abort_admission(admission.ticket));
            }
        };

        let response = SchedulingResponse {
            best_worker: selection.worker,
            effective_overlap_blocks: selection.effective_overlap_blocks,
            cached_tokens: selection.cached_tokens,
            selected_worker_tiers,
            request_progress: admission
                .as_ref()
                .map(|admission| admission.progress.clone()),
            admission_lease: None,
        };

        if !request.mode.is_tracked() {
            request.respond(Ok(response));
            debug_assert!(
                admission.is_none(),
                "query-only selection bypasses admission"
            );
            return false;
        }

        let request_id = request
            .mode
            .tracked_request_id()
            .expect("tracked mode always has a request ID")
            .to_string();

        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            selection.cached_tokens,
            request.track_prefill_tokens,
        );

        let sequence_request = SequenceRequest {
            request_id,
            token_sequence: request.token_seq.take(),
            track_prefill_tokens: request.track_prefill_tokens,
            expected_output_tokens: request.expected_output_tokens,
            prefill_load_hint,
            worker: selection.worker,
            lora_name: request.lora_name.take(),
        };
        let delivered = self.book_and_respond(request, sequence_request, response);
        if let Some(admission) = admission {
            if delivered {
                let request_id = admission_key.expect("admitted request has a lifecycle key");
                self.tracked_admissions
                    .entry(request_id)
                    .and_modify(|tracked| {
                        debug_assert_eq!(tracked.ticket, admission.ticket);
                        tracked.worker = Some(selection.worker);
                    })
                    .or_insert(TrackedAdmission {
                        ticket: admission.ticket,
                        worker: Some(selection.worker),
                        dispatched: false,
                    });
            } else {
                if let Some(request_id) = admission_key.as_deref() {
                    self.tracked_admissions.remove(request_id);
                }
                return self.abort_admission(admission.ticket);
            }
        }
        false
    }

    /// Completes the tracked-admission ownership handoff.
    ///
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
            return true;
        }

        tracing::debug!(%request_id, "Rolling back undelivered scheduler booking");
        if let Err(error) = self.slots.free(&request_id, Instant::now()) {
            tracing::error!(%request_id, %error, "Failed to roll back scheduler booking");
        }
        false
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
    /// through to `schedule`, which returns a proper `NoEndpoints` error.
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

fn apply_admission_placement(
    request: &mut SchedulingRequest,
    placement: WorkerPlacement,
) -> Result<(), KvSchedulerError> {
    let WorkerPlacement::Exact(worker) = placement else {
        return Ok(());
    };
    if request.pinned_worker.is_some_and(|pinned| pinned != worker) {
        return Err(KvSchedulerError::BookingFailed(format!(
            "admission placement {worker:?} conflicts with the request's pinned worker"
        )));
    }
    request.pinned_worker = Some(worker);
    request.eligibility().validate_pinned_worker_allowed()
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex as StdMutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use rustc_hash::FxHashMap;
    use tokio::sync::{Barrier, watch};

    use super::*;
    use crate::protocols::{
        ActiveLoad, ActiveSequenceEvent, WorkerSelectionResult, WorkerWithDpRank,
    };
    use crate::scheduling::OverlapSignals;
    use crate::scheduling::types::{KvSchedulerError, ScheduleMode};
    use crate::scheduling::{
        AdmissionEvent, AdmissionId, AdmissionRequest, PolicyClassAdmissionStrategy,
        RefreshedOverlap, RequestProgress, RouterPolicyConfig,
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
        fn publish_event(
            &self,
            _event: &ActiveSequenceEvent,
        ) -> impl std::future::Future<Output = anyhow::Result<()>> + Send {
            std::future::ready(Ok(()))
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
            })
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
            )
            .unwrap(),
        );
        (queue, slots, cfg_tx)
    }

    fn make_queue_with_overload_provider(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        overloaded_worker_provider: OverloadedWorkerProvider,
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
        let queue = Arc::new(SchedulerQueue::new_with_overload_provider(
            Arc::clone(&slots),
            cfg_rx,
            None,
            block_size,
            selector,
            RouterQueuePolicy::Fcfs,
            None,
            Some(overloaded_worker_provider),
        ));

        (queue, slots)
    }

    struct CountingRefresher {
        calls: AtomicUsize,
        response: RefreshedOverlap,
    }

    #[async_trait]
    impl OverlapScoresRefresh for CountingRefresher {
        async fn refresh(&self, _block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap> {
            self.calls.fetch_add(1, Ordering::Relaxed);
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
        async fn refresh(&self, _block_hashes: &[LocalBlockHash]) -> Option<RefreshedOverlap> {
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
                Duration::from_secs(60),
                PolicyClassAdmissionStrategies::new(),
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
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_id: None,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };
        (req, rx)
    }

    fn make_admission_request(
        request_id: &str,
        isl_tokens: usize,
    ) -> (
        SchedulingRequest,
        tokio::sync::oneshot::Receiver<
            Result<SchedulingResponse, crate::scheduling::types::KvSchedulerError>,
        >,
    ) {
        let (mut request, response) = make_request(request_id, isl_tokens);
        request.mode = ScheduleMode::TrackedWithAdmission {
            request_id: request_id.to_owned(),
        };
        (request, response)
    }

    #[derive(Default)]
    struct GateState {
        deferred: Option<AdmissionId>,
        session_id: Option<String>,
        context_tokens: usize,
        progress: Option<RequestProgress>,
        dispatched: Vec<WorkerWithDpRank>,
        completed_context_tokens: Vec<usize>,
        aborted: Vec<AdmissionId>,
    }

    struct ReconcileGate {
        state: Arc<StdMutex<GateState>>,
    }

    impl PolicyClassAdmissionStrategy for ReconcileGate {
        fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision {
            let mut state = self.state.lock().unwrap();
            state.deferred = Some(request.id());
            state.session_id = request.session_id().map(str::to_owned);
            state.context_tokens = request.context_tokens();
            state.progress = Some(request.progress().clone());
            AdmissionDecision::Defer
        }

        fn on_event(&mut self, event: AdmissionEvent) -> Vec<AdmissionAction> {
            let mut state = self.state.lock().unwrap();
            match event {
                AdmissionEvent::Dispatched { worker, .. } => {
                    state.dispatched.push(worker);
                    Vec::new()
                }
                AdmissionEvent::Completed { id, context_tokens } => {
                    if state.deferred == Some(id) {
                        state.deferred = None;
                    }
                    state.completed_context_tokens.push(context_tokens);
                    Vec::new()
                }
                AdmissionEvent::Aborted { id } => {
                    if state.deferred == Some(id) {
                        state.deferred = None;
                    }
                    state.aborted.push(id);
                    Vec::new()
                }
                AdmissionEvent::Reconcile => state
                    .deferred
                    .take()
                    .map(|id| {
                        vec![AdmissionAction::MakeReady {
                            id,
                            placement: WorkerPlacement::Exact(WorkerWithDpRank::new(0, 0)),
                        }]
                    })
                    .unwrap_or_default(),
            }
        }
    }

    fn make_queue_with_admission_strategy(
        strategy: Box<dyn PolicyClassAdmissionStrategy>,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        make_queue_with_admission_strategy_and_workers(strategy, 1)
    }

    fn make_queue_with_admission_strategy_and_workers(
        strategy: Box<dyn PolicyClassAdmissionStrategy>,
        worker_count: u64,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let profile = policy_profile(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: standard
    policy_family: standard
    cache_bucket: all
    quantum: 1
  - name: agents
    queue_admission:
      type: test
    prefill_busy_threshold: 0
    quantum: 1
"#,
        );
        make_queue_with_profile_and_admission_strategy(profile, "agents", strategy, worker_count)
    }

    fn make_queue_with_profile_and_admission_strategy(
        profile: PolicyProfile,
        class_name: &str,
        strategy: Box<dyn PolicyClassAdmissionStrategy>,
        worker_count: u64,
    ) -> (
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>>,
        Arc<ActiveSequencesMultiWorker<NoopSequencePublisher>>,
    ) {
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            16,
            (0..worker_count).map(|worker| (worker, (0, 1))).collect(),
            false,
            0,
            "test",
        ));
        let (_cfg_tx, cfg_rx) = watch::channel(
            (0..worker_count)
                .map(|worker| {
                    (
                        worker,
                        SimpleWorkerConfig {
                            max_num_batched_tokens: Some(1_000),
                            ..Default::default()
                        },
                    )
                })
                .collect(),
        );
        let mut strategies = PolicyClassAdmissionStrategies::new();
        strategies.insert(class_name.to_owned(), strategy);
        let queue = Arc::new(
            SchedulerQueue::new_with_policy_profile_and_admission_strategies(
                Arc::clone(&slots),
                cfg_rx,
                profile,
                16,
                DefaultWorkerSelector::new(None, "test"),
                None,
                None,
                None,
                Duration::from_secs(60),
                strategies,
            )
            .unwrap(),
        );
        (queue, slots)
    }

    #[tokio::test]
    async fn admission_strategy_defers_releases_and_observes_lifecycle() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(ReconcileGate {
            state: Arc::clone(&state),
        }));
        let (mut request, response) = make_admission_request("deferred", 64);
        request.policy_class = Some("agents".to_owned());
        request.session_id = Some("session-a".to_owned());

        queue.enqueue(request).await;
        assert_eq!(queue.pending_count(), 1);
        {
            let state = state.lock().unwrap();
            assert_eq!(state.session_id.as_deref(), Some("session-a"));
            assert_eq!(state.context_tokens, 64);
            assert!(state.dispatched.is_empty());
        }

        queue.reconcile().await;
        let selected = response.await.unwrap().unwrap();
        assert_eq!(selected.best_worker, WorkerWithDpRank::new(0, 0));
        let progress = selected.request_progress.unwrap();
        progress.update_context_tokens(80);
        assert_eq!(
            state
                .lock()
                .unwrap()
                .progress
                .as_ref()
                .unwrap()
                .context_tokens(),
            80
        );
        assert_eq!(queue.pending_count(), 0);
        queue.dispatched("deferred").await;
        queue.dispatched("deferred").await;
        queue
            .finish("deferred", RequestOutcome::Completed { context_tokens: 81 })
            .await;
        queue.update().await;
        {
            let state = state.lock().unwrap();
            assert_eq!(state.dispatched, vec![WorkerWithDpRank::new(0, 0)]);
            assert_eq!(state.completed_context_tokens, vec![81]);
        }

        slots.free(&"deferred".to_owned(), decay_now()).unwrap();
    }

    #[tokio::test]
    async fn cancelled_deferred_request_receives_one_terminal_event() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, _slots) = make_queue_with_admission_strategy(Box::new(ReconcileGate {
            state: Arc::clone(&state),
        }));
        let (mut request, response) = make_admission_request("cancelled", 64);
        request.policy_class = Some("agents".to_owned());
        request.session_id = Some("session-a".to_owned());

        let cancellation = queue.cancellation_guard(Some("cancelled")).unwrap();
        queue.enqueue(request).await;
        drop(response);
        drop(cancellation);
        queue.update().await;

        assert_eq!(queue.pending_count(), 0);
        let state = state.lock().unwrap();
        assert!(state.dispatched.is_empty());
        assert_eq!(state.aborted, vec![AdmissionId::new(0)]);
    }

    struct ReadyGate {
        state: Arc<StdMutex<GateState>>,
    }

    impl PolicyClassAdmissionStrategy for ReadyGate {
        fn admit(&mut self, _request: AdmissionRequest<'_>) -> AdmissionDecision {
            AdmissionDecision::Ready(WorkerPlacement::Any)
        }

        fn on_event(&mut self, event: AdmissionEvent) -> Vec<AdmissionAction> {
            let mut state = self.state.lock().unwrap();
            match event {
                AdmissionEvent::Dispatched { worker, .. } => state.dispatched.push(worker),
                AdmissionEvent::Completed { context_tokens, .. } => {
                    state.completed_context_tokens.push(context_tokens);
                }
                AdmissionEvent::Aborted { id } => state.aborted.push(id),
                AdmissionEvent::Reconcile => {}
            }
            Vec::new()
        }
    }

    struct ExactReadyGate(WorkerWithDpRank);

    impl PolicyClassAdmissionStrategy for ExactReadyGate {
        fn admit(&mut self, _request: AdmissionRequest<'_>) -> AdmissionDecision {
            AdmissionDecision::Ready(WorkerPlacement::Exact(self.0))
        }
    }

    #[tokio::test]
    async fn cancellation_after_response_send_rolls_back_booking() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(ReadyGate {
            state: Arc::clone(&state),
        }));
        let (mut request, response) = make_admission_request("cancelled-after-handoff", 64);
        request.policy_class = Some("agents".to_owned());
        let cancellation = queue
            .cancellation_guard(Some("cancelled-after-handoff"))
            .unwrap();

        queue.enqueue(request).await;
        assert_eq!(
            slots
                .active_request_counts()
                .get(&WorkerWithDpRank::new(0, 0))
                .copied(),
            Some(1)
        );

        drop(cancellation);
        tokio::time::timeout(Duration::from_secs(1), async {
            while state.lock().unwrap().aborted.len() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancellation did not abort admission");

        slots.assert_completely_drained(decay_now());
        assert_eq!(state.lock().unwrap().aborted, vec![AdmissionId::new(0)]);
        drop(response);
    }

    #[tokio::test]
    async fn dropped_completed_lease_commits_authoritative_context() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(ReadyGate {
            state: Arc::clone(&state),
        }));
        let (mut request, response) = make_admission_request("completed-after-terminal", 64);
        request.policy_class = Some("agents".to_owned());
        let mut lease = queue
            .cancellation_guard(Some("completed-after-terminal"))
            .unwrap();

        queue.enqueue(request).await;
        response.await.unwrap().unwrap();
        drop(queue);
        lease.mark_completed(96);
        drop(lease);

        tokio::time::timeout(Duration::from_secs(1), async {
            while state.lock().unwrap().completed_context_tokens != [96] {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completed lease did not commit admission context");
        slots.assert_completely_drained(decay_now());
        assert!(state.lock().unwrap().aborted.is_empty());
    }

    #[test]
    fn lease_drop_uses_actor_channel_then_overflow_cleanup() {
        let cleanup = Arc::new(AdmissionCleanup::default());
        let (actor_tx, mut actor_rx) = mpsc::channel(1);
        let lease = |request_id: &str| AdmissionLease {
            cleanup: Arc::clone(&cleanup),
            actor_tx: actor_tx.clone(),
            request_id: Some(request_id.to_owned()),
            outcome: RequestOutcome::Aborted,
        };

        drop(lease("fast-path"));
        assert!(matches!(
            actor_rx.try_recv(),
            Ok(AdmissionCommand::Cleanup { request_id, .. }) if request_id == "fast-path"
        ));
        assert!(cleanup.drain().is_empty());

        actor_tx
            .try_send(AdmissionCommand::Dispatched {
                request_id: "occupied".to_owned(),
            })
            .unwrap();
        drop(lease("overflow"));
        assert!(matches!(
            cleanup.drain().get("overflow"),
            Some(RequestOutcome::Aborted)
        ));
    }

    #[tokio::test]
    async fn duplicate_request_id_does_not_replace_active_admission() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(ReadyGate {
            state: Arc::clone(&state),
        }));
        let (mut first, first_response) = make_admission_request("duplicate", 64);
        first.policy_class = Some("agents".to_owned());
        queue.enqueue(first).await;
        first_response.await.unwrap().unwrap();

        let (mut duplicate, duplicate_response) = make_admission_request("duplicate", 64);
        duplicate.policy_class = Some("agents".to_owned());
        queue.enqueue(duplicate).await;
        assert!(matches!(
            duplicate_response.await.unwrap(),
            Err(KvSchedulerError::BookingFailed(message))
                if message == "request duplicate already has an active admission"
        ));
        assert_eq!(
            slots
                .active_request_counts()
                .get(&WorkerWithDpRank::new(0, 0))
                .copied(),
            Some(1)
        );
        assert!(state.lock().unwrap().aborted.is_empty());

        queue.finish("duplicate", RequestOutcome::Aborted).await;
        queue.update().await;
        assert_eq!(state.lock().unwrap().aborted, vec![AdmissionId::new(0)]);
        slots.free(&"duplicate".to_owned(), decay_now()).unwrap();
    }

    struct BypassGate {
        events: Arc<AtomicUsize>,
    }

    impl PolicyClassAdmissionStrategy for BypassGate {
        fn admit(&mut self, _request: AdmissionRequest<'_>) -> AdmissionDecision {
            AdmissionDecision::Bypass
        }

        fn on_event(&mut self, _event: AdmissionEvent) -> Vec<AdmissionAction> {
            self.events.fetch_add(1, Ordering::Relaxed);
            Vec::new()
        }
    }

    #[tokio::test]
    async fn legacy_tracked_request_bypasses_admission() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(ReadyGate { state }));
        let (mut request, response) = make_request("legacy", 64);
        request.policy_class = Some("agents".to_owned());

        queue.enqueue(request).await;
        let selected = response.await.unwrap().unwrap();

        assert!(selected.request_progress.is_none());
        slots.free(&"legacy".to_owned(), decay_now()).unwrap();
    }

    #[tokio::test]
    async fn bypassed_request_has_no_admission_lifecycle() {
        let events = Arc::new(AtomicUsize::new(0));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(BypassGate {
            events: Arc::clone(&events),
        }));
        let (mut request, response) = make_admission_request("bypassed", 64);
        request.policy_class = Some("agents".to_owned());
        {
            queue.enqueue(request).await;
            let selected = response.await.unwrap().unwrap();
            assert!(selected.request_progress.is_none());
        }

        queue.dispatched("bypassed").await;
        queue
            .finish("bypassed", RequestOutcome::Completed { context_tokens: 64 })
            .await;
        queue.update().await;
        assert_eq!(events.load(Ordering::Relaxed), 0);
        slots.free(&"bypassed".to_owned(), decay_now()).unwrap();

        let (request, response) = make_request("unmanaged", 64);
        queue.enqueue(request).await;
        let selected = response.await.unwrap().unwrap();
        assert!(selected.request_progress.is_none());
        slots.free(&"unmanaged".to_owned(), decay_now()).unwrap();
    }

    #[tokio::test]
    async fn cancellation_after_bypassed_handoff_rolls_back_booking() {
        let events = Arc::new(AtomicUsize::new(0));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(BypassGate {
            events: Arc::clone(&events),
        }));
        let (mut request, response) = make_admission_request("bypassed-handoff", 64);
        request.policy_class = Some("agents".to_owned());
        let cancellation = queue.cancellation_guard(Some("bypassed-handoff")).unwrap();

        queue.enqueue(request).await;
        assert_eq!(
            slots
                .active_request_counts()
                .get(&WorkerWithDpRank::new(0, 0))
                .copied(),
            Some(1)
        );

        drop(cancellation);
        queue.update().await;

        slots.assert_completely_drained(decay_now());
        assert_eq!(events.load(Ordering::Relaxed), 0);
        drop(response);
    }

    #[derive(Default)]
    struct FinishReleaseGate {
        first: Option<AdmissionId>,
        deferred: Option<AdmissionId>,
    }

    impl PolicyClassAdmissionStrategy for FinishReleaseGate {
        fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision {
            if self.first.is_none() {
                self.first = Some(request.id());
                AdmissionDecision::Ready(WorkerPlacement::Any)
            } else {
                self.deferred = Some(request.id());
                AdmissionDecision::Defer
            }
        }

        fn on_event(&mut self, event: AdmissionEvent) -> Vec<AdmissionAction> {
            let id = match event {
                AdmissionEvent::Completed { id, .. } | AdmissionEvent::Aborted { id } => id,
                _ => return Vec::new(),
            };
            if self.first != Some(id) {
                return Vec::new();
            }
            self.deferred
                .take()
                .map(|id| {
                    vec![AdmissionAction::MakeReady {
                        id,
                        placement: WorkerPlacement::Any,
                    }]
                })
                .unwrap_or_default()
        }
    }

    #[tokio::test]
    async fn lifecycle_action_drains_without_an_unrelated_update() {
        let (queue, slots) =
            make_queue_with_admission_strategy(Box::<FinishReleaseGate>::default());
        let (mut first, first_response) = make_admission_request("first-admitted", 64);
        first.policy_class = Some("agents".to_owned());
        queue.enqueue(first).await;
        first_response.await.unwrap().unwrap();

        let (mut second, second_response) = make_admission_request("second-deferred", 64);
        second.policy_class = Some("agents".to_owned());
        queue.enqueue(second).await;
        assert_eq!(queue.pending_count(), 1);

        slots
            .free(&"first-admitted".to_owned(), decay_now())
            .unwrap();
        queue
            .finish(
                "first-admitted",
                RequestOutcome::Completed { context_tokens: 64 },
            )
            .await;
        tokio::time::timeout(Duration::from_secs(1), second_response)
            .await
            .expect("finish action did not drain the queue")
            .unwrap()
            .unwrap();
        assert_eq!(queue.pending_count(), 0);
        slots
            .free(&"second-deferred".to_owned(), decay_now())
            .unwrap();
    }

    #[derive(Default)]
    struct PreservePinGate {
        deferred: Option<AdmissionId>,
    }

    impl PolicyClassAdmissionStrategy for PreservePinGate {
        fn admit(&mut self, request: AdmissionRequest<'_>) -> AdmissionDecision {
            if self.deferred.is_none() {
                self.deferred = Some(request.id());
                AdmissionDecision::Defer
            } else {
                AdmissionDecision::Ready(WorkerPlacement::Any)
            }
        }

        fn on_event(&mut self, event: AdmissionEvent) -> Vec<AdmissionAction> {
            if !matches!(event, AdmissionEvent::Reconcile) {
                return Vec::new();
            }
            self.deferred
                .take()
                .map(|id| {
                    vec![AdmissionAction::MakeReady {
                        id,
                        placement: WorkerPlacement::Any,
                    }]
                })
                .unwrap_or_default()
        }
    }

    #[tokio::test]
    async fn make_ready_any_preserves_existing_exact_worker_lane() {
        let (queue, slots) =
            make_queue_with_admission_strategy_and_workers(Box::<PreservePinGate>::default(), 2);
        for worker_id in 0..2 {
            let (mut blocker, response) = make_request(&format!("blocker-{worker_id}"), 64);
            blocker.pinned_worker = Some(WorkerWithDpRank::new(worker_id, 0));
            queue.enqueue(blocker).await;
            assert_eq!(
                response.await.unwrap().unwrap().best_worker,
                WorkerWithDpRank::new(worker_id, 0)
            );
        }

        let (mut pinned, pinned_response) = make_admission_request("pinned-deferred", 64);
        pinned.policy_class = Some("agents".to_owned());
        pinned.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
        queue.enqueue(pinned).await;

        let (mut runnable, runnable_response) = make_admission_request("runnable-shared", 64);
        runnable.policy_class = Some("agents".to_owned());
        queue.enqueue(runnable).await;
        assert_eq!(queue.pending_count(), 2);

        slots.free(&"blocker-1".to_owned(), decay_now()).unwrap();
        queue.reconcile().await;
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(1), runnable_response)
                .await
                .expect("pinned lane blocked shared work")
                .unwrap()
                .unwrap()
                .best_worker,
            WorkerWithDpRank::new(1, 0)
        );

        slots.free(&"blocker-0".to_owned(), decay_now()).unwrap();
        queue.update().await;
        assert_eq!(
            pinned_response.await.unwrap().unwrap().best_worker,
            WorkerWithDpRank::new(0, 0)
        );
        slots
            .free(&"runnable-shared".to_owned(), decay_now())
            .unwrap();
        slots
            .free(&"pinned-deferred".to_owned(), decay_now())
            .unwrap();
    }

    #[tokio::test]
    async fn family_ready_exact_uses_pinned_worker_queue_cost() {
        let profile = policy_profile(
            r#"
default_policy_family: agents
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: agents
    policy_family: agents
    cache_bucket: all
    queue_admission:
      type: test
    prefill_busy_threshold: 0
    quantum: 1
"#,
        );
        let worker = WorkerWithDpRank::new(0, 0);
        let (queue, slots) = make_queue_with_profile_and_admission_strategy(
            profile,
            "agents",
            Box::new(ExactReadyGate(worker)),
            2,
        );
        let (mut blocker, blocker_response) = make_request("family-exact-blocker", 64);
        blocker.pinned_worker = Some(worker);
        queue.enqueue(blocker).await;
        blocker_response.await.unwrap().unwrap();

        let (mut request, response) = make_admission_request("family-exact", 64);
        request
            .overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(1, 0), 64);
        queue.enqueue(request).await;

        assert_eq!(queue.class_queue_stats(0).unwrap().pending_cached_tokens, 0);
        slots
            .free(&"family-exact-blocker".to_owned(), decay_now())
            .unwrap();
        queue.update().await;
        assert_eq!(response.await.unwrap().unwrap().best_worker, worker);
        queue.finish("family-exact", RequestOutcome::Aborted).await;
        queue.update().await;
        slots.free(&"family-exact".to_owned(), decay_now()).unwrap();
    }

    #[tokio::test]
    async fn make_ready_exact_recomputes_queue_cost_for_pinned_worker() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy_and_workers(
            Box::new(ReconcileGate {
                state: Arc::clone(&state),
            }),
            2,
        );
        let worker = WorkerWithDpRank::new(0, 0);
        let (mut blocker, blocker_response) = make_request("exact-cost-blocker", 64);
        blocker.pinned_worker = Some(worker);
        queue.enqueue(blocker).await;
        blocker_response.await.unwrap().unwrap();

        let (mut request, response) = make_admission_request("exact-cost", 64);
        request.policy_class = Some("agents".to_owned());
        request
            .overlap
            .effective_cached_tokens
            .insert(WorkerWithDpRank::new(1, 0), 64);
        queue.enqueue(request).await;
        assert_eq!(
            queue.class_queue_stats(1).unwrap().pending_cached_tokens,
            64
        );

        queue.reconcile().await;
        assert_eq!(queue.class_queue_stats(1).unwrap().pending_cached_tokens, 0);

        slots
            .free(&"exact-cost-blocker".to_owned(), decay_now())
            .unwrap();
        queue.update().await;
        assert_eq!(response.await.unwrap().unwrap().best_worker, worker);
        queue.finish("exact-cost", RequestOutcome::Aborted).await;
        queue.update().await;
        slots.free(&"exact-cost".to_owned(), decay_now()).unwrap();
    }

    #[tokio::test]
    async fn cancelled_ready_requests_release_accounting_immediately() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(ReadyGate {
            state: Arc::clone(&state),
        }));

        let (blocker, blocker_response) = make_request("blocker", 64);
        queue.enqueue(blocker).await;
        blocker_response.await.unwrap().unwrap();

        for request_id in [
            "cancelled-ready-0",
            "cancelled-ready-1",
            "cancelled-ready-2",
        ] {
            let (mut cancelled, cancelled_response) = make_admission_request(request_id, 64);
            cancelled.policy_class = Some("agents".to_owned());
            let cancellation = queue.cancellation_guard(Some(request_id)).unwrap();
            queue.enqueue(cancelled).await;
            drop(cancelled_response);
            drop(cancellation);
        }
        queue.update().await;
        assert_eq!(state.lock().unwrap().aborted.len(), 3);
        assert_eq!(queue.pending_count(), 0);
        assert_eq!(queue.pending_isl_tokens(), 0);
        assert_eq!(
            queue.class_queue_stats(1),
            Some(ClassQueueStats {
                pending_count: 0,
                pending_isl_tokens: 0,
                pending_cached_tokens: 0,
            })
        );
        let state = state.lock().unwrap();
        assert!(state.dispatched.is_empty());
        assert_eq!(state.aborted.len(), 3);
        assert_eq!(
            state.aborted.iter().copied().collect::<HashSet<_>>(),
            [
                AdmissionId::new(0),
                AdmissionId::new(1),
                AdmissionId::new(2)
            ]
            .into_iter()
            .collect()
        );
        slots.free(&"blocker".to_owned(), decay_now()).unwrap();
    }

    #[tokio::test]
    async fn cancelled_ready_cleanup_does_not_remove_reused_request_id() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(ReadyGate {
            state: Arc::clone(&state),
        }));

        let (blocker, blocker_response) = make_request("blocker", 64);
        queue.enqueue(blocker).await;
        blocker_response.await.unwrap().unwrap();

        let (mut cancelled, cancelled_response) = make_admission_request("reused", 64);
        cancelled.policy_class = Some("agents".to_owned());
        let cancellation = queue.cancellation_guard(Some("reused")).unwrap();
        queue.enqueue(cancelled).await;
        drop(cancelled_response);
        drop(cancellation);
        queue.update().await;
        assert_eq!(state.lock().unwrap().aborted, vec![AdmissionId::new(0)]);

        let (mut replacement, replacement_response) = make_admission_request("reused", 64);
        replacement.policy_class = Some("agents".to_owned());
        queue.enqueue(replacement).await;
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(state.lock().unwrap().aborted, vec![AdmissionId::new(0)]);

        slots.free(&"blocker".to_owned(), decay_now()).unwrap();
        queue.update().await;
        let selected = replacement_response.await.unwrap().unwrap();
        assert!(selected.request_progress.is_some());

        queue
            .finish("reused", RequestOutcome::Completed { context_tokens: 64 })
            .await;
        queue.update().await;
        assert_eq!(state.lock().unwrap().completed_context_tokens, vec![64]);
        slots.free(&"reused".to_owned(), decay_now()).unwrap();
    }

    #[tokio::test]
    async fn backend_abort_finishes_without_dispatching_admission() {
        let state = Arc::new(StdMutex::new(GateState::default()));
        let (queue, slots) = make_queue_with_admission_strategy(Box::new(ReconcileGate {
            state: Arc::clone(&state),
        }));
        let (mut request, response) = make_admission_request("backend-abort", 64);
        request.policy_class = Some("agents".to_owned());

        queue.enqueue(request).await;
        queue.reconcile().await;
        response.await.unwrap().unwrap();
        queue.finish("backend-abort", RequestOutcome::Aborted).await;
        queue.update().await;

        let state = state.lock().unwrap();
        assert!(state.dispatched.is_empty());
        assert_eq!(state.aborted, vec![AdmissionId::new(0)]);
        slots
            .free(&"backend-abort".to_owned(), decay_now())
            .unwrap();
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

    #[tokio::test]
    async fn dropped_legacy_lease_retracts_pending_request_immediately() {
        let isl = 512;
        let (queue, slots) = make_queue(1, 16, isl, Some(0.0));

        let (first, first_rx) = make_request("legacy-first", isl);
        queue.enqueue(first).await;
        first_rx.await.unwrap().unwrap();

        let lease = queue.cancellation_guard(Some("legacy-cancelled")).unwrap();
        let (cancelled, cancelled_rx) = make_request("legacy-cancelled", isl);
        queue.enqueue(cancelled).await;
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
default_policy_family: latency
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: latency
    policy_family: latency
    cache_bucket: all
    quantum: 1
    prefill_busy_threshold: 0
  - name: bulk
    policy_family: bulk
    cache_bucket: all
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

    #[tokio::test(flavor = "multi_thread")]
    async fn policy_families_and_cache_buckets_select_physical_queues() {
        let profile = policy_profile(
            r#"
default_policy_family: standard
uncached_isl_buckets:
  - min_tokens: 0
    bucket: cached
  - min_tokens: 32
    bucket: uncached
policy_classes:
  - name: cached
    policy_family: standard
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 0
  - name: uncached
    policy_family: standard
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 0
  - name: latency_cached
    policy_family: latency
    cache_bucket: cached
    quantum: 1
    prefill_busy_threshold: 0
  - name: latency_uncached
    policy_family: latency
    cache_bucket: uncached
    quantum: 1
    prefill_busy_threshold: 0
  - name: custom_priority
    quantum: 1
    prefill_busy_threshold: 0
"#,
        );
        let (queue, _slots) = make_queue_with_profile(1, 16, 64, profile);
        let worker = WorkerWithDpRank::new(0, 0);

        let (active, active_rx) = make_request("active", 64);
        queue.enqueue(active).await;
        active_rx.await.unwrap().unwrap();

        let (mut latency_cached, _latency_cached_rx) = make_request("latency-cached", 64);
        latency_cached.policy_class = Some("latency".to_string());
        latency_cached
            .overlap
            .effective_cached_tokens
            .insert(worker, 64);
        queue.enqueue(latency_cached).await;

        let (mut latency_uncached, _latency_uncached_rx) = make_request("latency-uncached", 64);
        latency_uncached.policy_class = Some("latency".to_string());
        queue.enqueue(latency_uncached).await;

        let (mut unknown_cached, _unknown_cached_rx) = make_request("unknown-cached", 64);
        unknown_cached.policy_class = Some("unknown".to_string());
        unknown_cached
            .overlap
            .effective_cached_tokens
            .insert(worker, 64);
        queue.enqueue(unknown_cached).await;

        let (mut ordinary_class_name, _ordinary_class_name_rx) =
            make_request("ordinary-class-name", 64);
        ordinary_class_name.policy_class = Some("latency_cached".to_string());
        queue.enqueue(ordinary_class_name).await;

        let (mut custom, _custom_rx) = make_request("custom", 64);
        custom.policy_class = Some("custom_priority".to_string());
        queue.enqueue(custom).await;

        assert_eq!(
            queue.class_queue_stats(0),
            Some(ClassQueueStats {
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 64,
            })
        );
        assert_eq!(
            queue.class_queue_stats(1),
            Some(ClassQueueStats {
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 0,
            })
        );
        assert_eq!(
            queue.class_queue_stats(2),
            Some(ClassQueueStats {
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 64,
            })
        );
        assert_eq!(
            queue.class_queue_stats(3),
            Some(ClassQueueStats {
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 0,
            })
        );
        assert_eq!(
            queue.class_queue_stats(4),
            Some(ClassQueueStats {
                pending_count: 1,
                pending_isl_tokens: 64,
                pending_cached_tokens: 0,
            })
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn class_local_limit_rejection_is_typed_and_not_overload() {
        let profile = policy_profile(
            r#"
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
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
default_policy_family: capped
uncached_isl_buckets:
  - min_tokens: 0
    bucket: all
policy_classes:
  - name: capped
    policy_family: capped
    cache_bucket: all
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
            Arc::new(|| Some(HashSet::from([0])));
        let (queue, _slots) =
            make_queue_with_overload_provider(1, 16, 256, overloaded_worker_provider);

        let (req, rx) = make_request("overloaded", 256);
        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(matches!(
            resp,
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
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
    async fn test_blocked_pinned_lane_does_not_block_other_worker() {
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

        let (mut other_worker, mut other_worker_rx) = make_request("pinned-0", 256);
        other_worker.pinned_worker = Some(WorkerWithDpRank::new(0, 0));
        queue.enqueue(other_worker).await;
        assert_eq!(queue.pending_count(), 2);

        queue.update().await;

        assert_eq!(queue.pending_count(), 1);
        let other_worker_resp = other_worker_rx
            .try_recv()
            .expect("other worker request should have been scheduled")
            .expect("scheduling returned error");
        assert_eq!(other_worker_resp.best_worker, WorkerWithDpRank::new(0, 0));
        assert!(
            second_rx.try_recv().is_err(),
            "pinned request should still be queued"
        );

        slots
            .mark_prefill_completed(&"pinned-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"pinned-1".to_string(), decay_now()).unwrap();
        queue.update_worker(WorkerWithDpRank::new(1, 0)).await;

        let second_resp = second_rx
            .try_recv()
            .expect("pinned request should have been scheduled");
        let second_resp = second_resp.expect("scheduling returned error");
        assert_eq!(second_resp.best_worker, WorkerWithDpRank::new(1, 0));
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
            response: RefreshedOverlap {
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
        assert_eq!(resp3.best_worker, WorkerWithDpRank::new(1, 0));
        assert_eq!(resp3.effective_overlap_blocks, 9.0);
        assert_eq!(resp3.cached_tokens, 144);
        assert_eq!(queue.pending_count(), 0);
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn selected_request_dispatches_after_refresh_if_worker_becomes_busy() {
        let block_size = 16u32;
        let isl = 64usize;
        let worker = WorkerWithDpRank::new(0, 0);
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap {
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::from([(worker, 7.0)]),
            effective_cached_tokens: HashMap::from([(worker, 56)]),
        }));
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
