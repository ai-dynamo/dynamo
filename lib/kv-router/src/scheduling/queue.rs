// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::config::RouterQueueDepthTiers;
use super::filter::RoutingEligibility;
use super::overlap_refresh::{
    NoopOverlapScoresRefresh, OverlapScoresRefresh, RefreshedOverlap, read_overlap_refresh_after,
    refresh_overlap,
};
use super::policy::{FcfsPolicy, SchedulingPolicy};
use super::prefill_load::{PrefillLoadEstimator, effective_prefill_tokens};
use super::selector::{DefaultWorkerSelector, WorkerSelector};
use super::types::{
    KvSchedulerError, OverloadedWorkerProvider, SchedulingContext, SchedulingRequest,
    SchedulingResponse,
};
use crate::protocols::{
    LocalBlockHash, PrefillLoadHint, RouterBackpressureReason, WorkerConfigLike, WorkerId,
};
use crate::sequences::topology::WorkerDpRange;
use crate::sequences::{ActiveSequencesMultiWorker, SequencePublisher, SequenceRequest};

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

const ADMISSION_CHANNEL_CAPACITY: usize = 65_536;

/// Entry in the priority queue, ordered by key (higher key = higher priority).
struct QueueEntry<K: Ord + Eq> {
    key: K,
    request: SchedulingRequest,
    enqueue_at: Instant,
    block_hashes: Option<Vec<LocalBlockHash>>,
}

impl<K: Ord + Eq> Eq for QueueEntry<K> {}

impl<K: Ord + Eq> PartialEq for QueueEntry<K> {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl<K: Ord + Eq> Ord for QueueEntry<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

impl<K: Ord + Eq> PartialOrd for QueueEntry<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[allow(clippy::large_enum_variant)]
enum AdmissionCommand {
    Enqueue {
        request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
        ack_tx: oneshot::Sender<()>,
    },
    Update {
        ack_tx: oneshot::Sender<()>,
    },
}

struct SchedulerQueueActor<
    P: SequencePublisher,
    C: WorkerConfigLike,
    S: SchedulingPolicy,
    Sel: WorkerSelector<C>,
    RF: OverlapScoresRefresh,
> {
    pending: BinaryHeap<QueueEntry<S::Key>>,
    pending_count: Arc<AtomicUsize>,
    pending_isl_tokens: Arc<AtomicUsize>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    threshold_frac: Option<f64>,
    queue_depth_tiers: RouterQueueDepthTiers,
    start_time: Instant,
    block_size: u32,
    selector: Sel,
    policy: S,
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
    S: SchedulingPolicy = FcfsPolicy,
    Sel: WorkerSelector<C> = DefaultWorkerSelector,
    RF: OverlapScoresRefresh = NoopOverlapScoresRefresh,
> {
    admission_tx: mpsc::Sender<AdmissionCommand>,
    /// Number of requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_count: Arc<AtomicUsize>,
    /// Sum of `isl_tokens` for requests currently parked in the pending queue.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    pending_isl_tokens: Arc<AtomicUsize>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    /// Cached threshold fraction; None means queueing is disabled.
    threshold_frac: Option<f64>,
    supports_overlap_refresh: bool,
    _marker: PhantomData<(S, Sel, RF)>,
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, S, Sel, RF>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_overlap_refresh(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        queue_depth_tiers: RouterQueueDepthTiers,
        block_size: u32,
        selector: Sel,
        policy: S,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<RF>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    ) -> Self {
        if let Some(frac) = threshold_frac {
            tracing::info!("Router queue enabled with threshold fraction {frac}");
        }
        if !queue_depth_tiers.is_unbounded() {
            tracing::info!("Router queue tiered by cache-miss: pending ISL token caps configured");
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
        let (admission_tx, admission_rx) = mpsc::channel(ADMISSION_CHANNEL_CAPACITY);
        let actor = SchedulerQueueActor {
            pending: BinaryHeap::new(),
            pending_count: Arc::clone(&pending_count),
            pending_isl_tokens: Arc::clone(&pending_isl_tokens),
            slots: Arc::clone(&slots),
            workers_with_configs: workers_with_configs.clone(),
            threshold_frac,
            queue_depth_tiers,
            start_time: Instant::now(),
            block_size,
            selector,
            policy,
            prefill_load_estimator,
            overlap_scores_refresh,
            overlap_refresh_after,
            overloaded_worker_provider,
        };
        tokio::spawn(actor.run(admission_rx));
        Self {
            admission_tx,
            pending_count,
            pending_isl_tokens,
            slots,
            workers_with_configs,
            threshold_frac,
            supports_overlap_refresh: overlap_refresh_after.is_some(),
            _marker: PhantomData,
        }
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
> SchedulerQueue<P, C, S, Sel, NoopOverlapScoresRefresh>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        queue_depth_tiers: RouterQueueDepthTiers,
        block_size: u32,
        selector: Sel,
        policy: S,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    ) -> Self {
        Self::new_with_overlap_refresh(
            slots,
            workers_with_configs,
            threshold_frac,
            queue_depth_tiers,
            block_size,
            selector,
            policy,
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
        queue_depth_tiers: RouterQueueDepthTiers,
        block_size: u32,
        selector: Sel,
        policy: S,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    ) -> Self {
        Self::new_with_overlap_refresh(
            slots,
            workers_with_configs,
            threshold_frac,
            queue_depth_tiers,
            block_size,
            selector,
            policy,
            prefill_load_estimator,
            None,
            overloaded_worker_provider,
        )
    }
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueue<P, C, S, Sel, RF>
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
    ///
    /// When `allowed_worker_ids` is set on the request without an exact pin
    /// (external routing), the capacity check is skipped.
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
        if self.threshold_frac.is_none() {
            return;
        }

        let (ack_tx, ack_rx) = oneshot::channel();
        if self
            .admission_tx
            .send(AdmissionCommand::Update { ack_tx })
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
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
    RF: OverlapScoresRefresh + Send + Sync + 'static,
> SchedulerQueueActor<P, C, S, Sel, RF>
{
    async fn run(mut self, mut rx: mpsc::Receiver<AdmissionCommand>) {
        while let Some(command) = rx.recv().await {
            match command {
                AdmissionCommand::Enqueue {
                    request,
                    block_hashes,
                    ack_tx,
                } => {
                    self.handle_enqueue(request, block_hashes);
                    let _ = ack_tx.send(());
                }
                AdmissionCommand::Update { ack_tx } => {
                    self.handle_update().await;
                    let _ = ack_tx.send(());
                }
            }
        }

        while let Some(entry) = self.pending.pop() {
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_sub(entry.request.isl_tokens, AtomicOrdering::Relaxed);

            let mut request = entry.request;
            request.respond(Err(KvSchedulerError::SubscriberShutdown));
        }
    }

    fn handle_enqueue(
        &mut self,
        mut request: SchedulingRequest,
        block_hashes: Option<Vec<LocalBlockHash>>,
    ) {
        let eligibility = request.eligibility();
        let decay_now = Instant::now();

        let Some(threshold) = self.threshold_frac else {
            self.admit_one(request, decay_now);
            return;
        };

        if eligibility.bypasses_capacity_check() {
            self.admit_one(request, decay_now);
            return;
        }

        if self.all_workers_prefill_busy(threshold, request.eligibility(), decay_now) {
            if !self.queue_depth_tiers.is_unbounded() {
                let pending_isl_tokens = self.pending_isl_tokens.load(AtomicOrdering::Relaxed);
                // This is a rejection threshold on current queued ISL, not a hard
                // post-admission bound on `pending + incoming`.
                if let Some(max_isl_tokens) = self.tier_cap_for_request(&request)
                    && pending_isl_tokens >= max_isl_tokens
                {
                    request.respond(Err(KvSchedulerError::Backpressure {
                        reason: RouterBackpressureReason::MaxQueuedIslTokensExceeded,
                        queued_isl_tokens: pending_isl_tokens,
                        max_queued_isl_tokens: Some(max_isl_tokens),
                    }));
                    return;
                }
            }
            tracing::debug!("all workers prefill-busy, queueing request");
            let arrival_offset = self.start_time.elapsed();
            let key = {
                let workers = self.workers_with_configs.borrow();
                self.policy
                    .enqueue_key(arrival_offset, SchedulingContext::new(&request, &workers))
            };
            let isl_tokens = request.isl_tokens;
            self.pending.push(QueueEntry {
                key,
                request,
                enqueue_at: decay_now,
                block_hashes,
            });
            self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_add(isl_tokens, AtomicOrdering::Relaxed);
            return;
        }

        // Strict priority only orders requests parked in `pending`. Preserve
        // direct admission for eligible arrivals to avoid global head-of-line
        // blocking across requests with different worker eligibility.
        self.admit_one(request, decay_now);
    }

    async fn handle_update(&mut self) {
        let Some(threshold) = self.threshold_frac else {
            return;
        };

        if S::DYNAMIC {
            let now = self.start_time.elapsed();
            let workers = self.workers_with_configs.borrow();
            let rekeyed: Vec<_> = std::mem::take(&mut self.pending)
                .into_vec()
                .into_iter()
                .map(|e| QueueEntry {
                    key: self.policy.rekey(
                        now,
                        &e.key,
                        SchedulingContext::new(&e.request, &workers),
                    ),
                    request: e.request,
                    enqueue_at: e.enqueue_at,
                    block_hashes: e.block_hashes,
                })
                .collect();
            self.pending = BinaryHeap::from(rekeyed);
        }

        loop {
            let decay_now = Instant::now();
            let Some(front) = self.pending.peek() else {
                break;
            };
            // TODO: This preserves head-of-line blocking for now to keep queue
            // drain overhead bounded to the heap front. A blocked pinned or
            // otherwise constrained request can temporarily stall later
            // schedulable entries until we adopt a cheaper non-HOL strategy.
            if self.all_workers_prefill_busy(threshold, front.request.eligibility(), decay_now) {
                break;
            }
            let entry = self.pending.pop().expect("heap front vanished before pop");
            let current_pending_count = self.pending_count.load(AtomicOrdering::Relaxed);
            debug_assert!(
                current_pending_count > 0,
                "pending_count underflow on queue drain"
            );
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            let current_pending_isl_tokens = self.pending_isl_tokens.load(AtomicOrdering::Relaxed);
            debug_assert!(
                current_pending_isl_tokens >= entry.request.isl_tokens,
                "pending_isl_tokens underflow: pending={} request_isl_tokens={}",
                current_pending_isl_tokens,
                entry.request.isl_tokens
            );
            self.pending_isl_tokens
                .fetch_sub(entry.request.isl_tokens, AtomicOrdering::Relaxed);
            let mut request = entry.request;
            let refreshed = refresh_overlap(
                self.overlap_scores_refresh.as_deref(),
                self.overlap_refresh_after,
                entry.block_hashes.as_deref(),
                entry.enqueue_at,
                decay_now,
            )
            .await;
            let wait_ms = entry.enqueue_at.elapsed().as_millis() as u64;
            if let Some(RefreshedOverlap {
                tier_overlap_blocks,
                effective_overlap_blocks,
                effective_cached_tokens,
            }) = refreshed
            {
                tracing::info!(
                    request_id = request.maybe_request_id.as_deref().unwrap_or("unknown"),
                    wait_ms,
                    "refreshed overlap scores after long queue wait"
                );
                request.tier_overlap_blocks = tier_overlap_blocks;
                request.effective_overlap_blocks = effective_overlap_blocks;
                request.effective_cached_tokens = effective_cached_tokens;
            }
            let admit_now = Instant::now();
            if self.all_workers_prefill_busy(threshold, request.eligibility(), admit_now) {
                let isl_tokens = request.isl_tokens;
                self.pending.push(QueueEntry {
                    key: entry.key,
                    request,
                    enqueue_at: entry.enqueue_at,
                    block_hashes: entry.block_hashes,
                });
                self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
                self.pending_isl_tokens
                    .fetch_add(isl_tokens, AtomicOrdering::Relaxed);
                break;
            }
            tracing::debug!("scheduling request from pending queue");
            self.admit_one(request, admit_now);
        }
    }

    /// Run the full scheduling pipeline for a single request:
    /// compute projected load -> select worker -> book tracked state -> respond.
    fn admit_one(&self, mut request: SchedulingRequest, decay_now: Instant) {
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
        };

        let selection = match selection {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("scheduling failed: {e}");
                request.respond(Err(e));
                return;
            }
        };

        let response = SchedulingResponse {
            best_worker: selection.worker,
            effective_overlap_blocks: selection.effective_overlap_blocks,
            cached_tokens: selection.cached_tokens,
        };

        if !request.update_states {
            request.respond(Ok(response));
            return;
        }

        let Some(request_id) = request.maybe_request_id.clone() else {
            tracing::error!("No request_id provided to add_request to the slot tracker");
            request.respond(Err(KvSchedulerError::BookingFailed(
                "tracked scheduling request did not include a request_id".to_string(),
            )));
            return;
        };

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
        self.book_and_respond(request, sequence_request, response);
    }

    fn book_and_respond(
        &self,
        mut request: SchedulingRequest,
        sequence_request: SequenceRequest,
        response: SchedulingResponse,
    ) {
        if request.response_is_closed() {
            tracing::debug!(
                request_id = %sequence_request.request_id,
                "Skipping scheduler booking for cancelled request"
            );
            return;
        }

        let request_id = sequence_request.request_id.clone();
        if let Err(error) = self.slots.add_request(sequence_request, Instant::now()) {
            tracing::warn!(%request_id, %error, "Failed to book scheduler state");
            request.respond(Err(KvSchedulerError::BookingFailed(error.to_string())));
            return;
        }

        if request.respond(Ok(response)) {
            return;
        }

        tracing::debug!(%request_id, "Rolling back undelivered scheduler booking");
        if let Err(error) = self.slots.free(&request_id, Instant::now()) {
            tracing::error!(%request_id, %error, "Failed to roll back scheduler booking");
        }
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

    /// Resolve the admission cap for `request` from the cache-miss tier table.
    ///
    /// Returns `None` when capping is disabled.
    fn tier_cap_for_request(&self, request: &SchedulingRequest) -> Option<usize> {
        let workers = self.workers_with_configs.borrow();
        let ctx = SchedulingContext::new(request, &workers);
        let cache_miss_tokens = ctx.best_effective_prefill_tokens();
        // Scale against the total registered worker count instead of projecting
        // the existing global queue onto only this request's eligible workers.
        // For narrowed requests (for example pinned or allow-listed routing),
        // shrinking the cap to the eligible subset would make the global queue
        // look like it must all be drained by that subset, which overstates the
        // backlog pressure on those workers.
        let worker_count = workers.len();
        self.queue_depth_tiers
            .cap_for(cache_miss_tokens, worker_count)
    }

    /// Check if all eligible workers are prefill-busy based on threshold.
    /// When `pinned_worker` is `Some`, only that exact worker/rank is considered.
    /// Otherwise when `allowed` is `Some`, only those worker IDs are considered;
    /// otherwise all registered workers are checked.
    /// Returns false when no eligible workers exist so the request falls
    /// through to `schedule`, which returns a proper `NoEndpoints` error.
    fn all_workers_prefill_busy(
        &self,
        threshold: f64,
        eligibility: RoutingEligibility<'_>,
        decay_now: Instant,
    ) -> bool {
        let active_tokens = self.slots.active_tokens(decay_now);
        let configs = self.workers_with_configs.borrow();

        if let Some(worker) = eligibility.pinned_worker() {
            let Ok(config) = eligibility.validate_worker_rank(&configs, worker) else {
                return false;
            };

            let max_batched = config
                .max_num_batched_tokens()
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
            let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
            return (tokens as f64) > threshold * (max_batched as f64);
        }

        let mut checked_any = false;
        let has_available = eligibility.any_eligible_worker_rank(&configs, |worker, config| {
            checked_any = true;
            let max_batched = config
                .max_num_batched_tokens()
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
            let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
            (tokens as f64) <= threshold * (max_batched as f64)
        });

        checked_any && !has_available
    }
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
    use crate::config::RouterQueueDepthByMissingIslTier;
    use crate::protocols::{
        ActiveLoad, ActiveSequenceEvent, RouterBackpressureReason, WorkerSelectionResult,
        WorkerWithDpRank,
    };
    use crate::scheduling::types::KvSchedulerError;
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
        let (queue, slots, _tx) = make_queue_with_sender_with_tiers(
            num_workers,
            block_size,
            isl,
            threshold_frac,
            RouterQueueDepthTiers::unbounded_cap(),
            None,
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
        Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig, FcfsPolicy, Sel>>,
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
            RouterQueueDepthTiers::unbounded_cap(),
            block_size,
            selector,
            FcfsPolicy,
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
        make_queue_with_sender_with_tiers(
            num_workers,
            block_size,
            isl,
            threshold_frac,
            RouterQueueDepthTiers::unbounded_cap(),
            prefill_load_estimator,
        )
    }

    #[allow(clippy::type_complexity)]
    fn make_queue_with_sender_with_tiers(
        num_workers: usize,
        block_size: u32,
        isl: usize,
        threshold_frac: Option<f64>,
        queue_depth_tiers: RouterQueueDepthTiers,
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
            queue_depth_tiers,
            block_size,
            selector,
            FcfsPolicy,
            prefill_load_estimator,
        ));

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
            RouterQueueDepthTiers::unbounded_cap(),
            block_size,
            selector,
            FcfsPolicy,
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
                FcfsPolicy,
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
            RouterQueueDepthTiers::unbounded_cap(),
            block_size,
            DefaultWorkerSelector::new(None, "test"),
            FcfsPolicy,
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
    ) -> (
        Arc<
            SchedulerQueue<
                NoopSequencePublisher,
                SimpleWorkerConfig,
                FcfsPolicy,
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

        let queue = Arc::new(SchedulerQueue::new_with_overlap_refresh(
            Arc::clone(&slots),
            cfg_rx,
            threshold_frac,
            RouterQueueDepthTiers::unbounded_cap(),
            block_size,
            DefaultWorkerSelector::new(None, "test"),
            FcfsPolicy,
            None,
            Some(refresher),
            None,
        ));

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
            maybe_request_id: Some(request_id.to_string()),
            token_seq: None,
            isl_tokens,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::new(),
            effective_cached_tokens: HashMap::new(),
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: None,
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };
        (req, rx)
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
            RouterQueueDepthTiers::unbounded_cap(),
            16,
            DefaultWorkerSelector::new(None, "test"),
            FcfsPolicy,
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
    async fn test_backpressure_when_missing_isl_tier_cap_reached() {
        let block_size = 16;
        let isl = 512;

        // Single tier at floor=0 with cap=512 ISL tokens: every request matches,
        // queue capped at 512 ISL tokens (1 request worth).
        let tiers = RouterQueueDepthTiers::try_from(vec![RouterQueueDepthByMissingIslTier {
            missing_cache_tokens_floor: 0,
            max_queue_depth: isl,
        }])
        .unwrap();
        let (queue, _slots, _cfg_tx) =
            make_queue_with_sender_with_tiers(1, block_size, isl, Some(0.0), tiers, None);

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _resp1 = rx1.await.unwrap().unwrap();

        let (req2, _rx2) = make_request("req-2", isl);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.pending_isl_tokens(), isl);

        let (req3, rx3) = make_request("req-3", isl);
        queue.enqueue(req3).await;

        let resp3 = rx3.await.expect("oneshot dropped");
        assert!(
            matches!(
                resp3,
                Err(KvSchedulerError::Backpressure {
                    reason: RouterBackpressureReason::MaxQueuedIslTokensExceeded,
                    queued_isl_tokens: 512,
                    max_queued_isl_tokens: Some(512),
                })
            ),
            "expected backpressure when queue is full, got {resp3:?}"
        );
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.pending_isl_tokens(), isl);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_missing_isl_tiers_shed_expensive_first() {
        let block_size = 16;
        let isl = 512;

        // All test requests have 512 cache-miss tokens, so they match the
        // expensive tier and should backpressure once one request is pending.
        let tiers = RouterQueueDepthTiers::try_from(vec![
            RouterQueueDepthByMissingIslTier {
                missing_cache_tokens_floor: 0,
                max_queue_depth: 4 * isl,
            },
            RouterQueueDepthByMissingIslTier {
                missing_cache_tokens_floor: 256,
                max_queue_depth: isl,
            },
        ])
        .unwrap();
        let (queue, _slots, _cfg_tx) =
            make_queue_with_sender_with_tiers(1, block_size, isl, Some(0.0), tiers, None);

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _ = rx1.await.unwrap().unwrap();

        let (req2, _rx2) = make_request("req-2", isl);
        queue.enqueue(req2).await;
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(queue.pending_isl_tokens(), isl);

        let (req3, rx3) = make_request("req-3", isl);
        queue.enqueue(req3).await;

        let resp3 = rx3.await.expect("oneshot dropped");
        assert!(
            matches!(
                resp3,
                Err(KvSchedulerError::Backpressure {
                    reason: RouterBackpressureReason::MaxQueuedIslTokensExceeded,
                    queued_isl_tokens: 512,
                    max_queued_isl_tokens: Some(512),
                })
            ),
            "expensive-tier request should backpressure at depth 1, got {resp3:?}"
        );
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

    #[tokio::test]
    async fn test_no_workers_returns_error() {
        let (queue, _slots) = make_queue(0, 16, 512, None);

        let (req, rx) = make_request("lonely-req", 512);
        queue.enqueue(req).await;

        let resp = rx.await.expect("oneshot dropped");
        assert!(
            matches!(
                resp,
                Err(crate::scheduling::types::KvSchedulerError::NoEndpoints)
            ),
            "expected NoEndpoints, got {resp:?}"
        );
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

    /// Requests with allowed_worker_ids should only route to the specified subset.
    #[tokio::test(flavor = "multi_thread")]
    async fn test_allowed_worker_ids_filter() {
        let block_size = 16;
        let isl = 256;

        let (queue, slots, cfg_tx) = make_queue_with_sender(0, block_size, isl, None, None);

        // Register three workers
        for worker_id in 1..=3 {
            slots
                .upsert_worker(WorkerDpRange::new(worker_id, 0, 1))
                .unwrap();
        }

        let mut configs = HashMap::new();
        for &id in &[1_u64, 2_u64, 3_u64] {
            configs.insert(
                id,
                SimpleWorkerConfig {
                    max_num_batched_tokens: Some(isl as u64),
                    ..Default::default()
                },
            );
        }
        cfg_tx.send(configs).unwrap();

        // Send a request with allowed_worker_ids = {2} only
        let mut allowed = std::collections::HashSet::new();
        allowed.insert(2_u64);

        let (tx, rx) = tokio::sync::oneshot::channel();
        let req = SchedulingRequest {
            maybe_request_id: Some("filter-0".to_string()),
            token_seq: None,
            isl_tokens: isl,
            tier_overlap_blocks: Default::default(),
            effective_overlap_blocks: HashMap::new(),
            effective_cached_tokens: HashMap::new(),
            worker_loads: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
            strict_priority: 0,
            expected_output_tokens: None,
            pinned_worker: None,
            allowed_worker_ids: Some(allowed),
            routing_constraints: crate::protocols::RoutingConstraints::default(),
            shared_cache_hits: None,
            resp_tx: Some(tx),
        };
        queue.enqueue(req).await;
        let resp = rx
            .await
            .expect("oneshot dropped")
            .expect("scheduling failed");
        assert_eq!(
            resp.best_worker.worker_id, 2,
            "request must be routed to allowed worker 2, got {}",
            resp.best_worker.worker_id
        );
        slots
            .mark_prefill_completed(&"filter-0".to_string(), decay_now())
            .unwrap();
        slots.free(&"filter-0".to_string(), decay_now()).unwrap();
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
    async fn test_pinned_request_head_of_line_blocks_other_worker_capacity() {
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

        let (occupy_other, occupy_other_rx) = make_request("worker-0", 256);
        queue.enqueue(occupy_other).await;
        let occupy_other_resp = occupy_other_rx.await.unwrap().unwrap();
        assert_eq!(occupy_other_resp.best_worker, WorkerWithDpRank::new(0, 0));

        let (unpinned, mut unpinned_rx) = make_request("unpinned", 256);
        queue.enqueue(unpinned).await;
        assert_eq!(queue.pending_count(), 2);

        slots
            .mark_prefill_completed(&"worker-0".to_string(), decay_now())
            .unwrap();
        slots.free(&"worker-0".to_string(), decay_now()).unwrap();
        queue.update().await;

        assert_eq!(queue.pending_count(), 2);
        assert!(
            unpinned_rx.try_recv().is_err(),
            "unpinned request should remain queued behind the pinned head"
        );
        assert!(
            second_rx.try_recv().is_err(),
            "pinned request should still be queued"
        );

        slots
            .mark_prefill_completed(&"pinned-1".to_string(), decay_now())
            .unwrap();
        slots.free(&"pinned-1".to_string(), decay_now()).unwrap();
        queue.update().await;

        let second_resp = second_rx
            .try_recv()
            .expect("pinned request should have been scheduled");
        let second_resp = second_resp.expect("scheduling returned error");
        assert_eq!(second_resp.best_worker, WorkerWithDpRank::new(1, 0));

        let unpinned_resp = unpinned_rx
            .try_recv()
            .expect("unpinned request should have been scheduled");
        let unpinned_resp = unpinned_resp.expect("scheduling returned error");
        assert_eq!(unpinned_resp.best_worker, WorkerWithDpRank::new(0, 0));
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
        req1.effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 3.0);
        req1.effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 48);
        queue.enqueue(req1).await;
        let resp1 = rx1.await.expect("rx1 dropped").expect("req-1 failed");
        assert_eq!(resp1.best_worker, WorkerWithDpRank::new(0, 0));

        let (mut req2, rx2) = make_request("req-2", isl);
        req2.effective_overlap_blocks
            .insert(WorkerWithDpRank::new(1, 0), 3.0);
        req2.effective_cached_tokens
            .insert(WorkerWithDpRank::new(1, 0), 48);
        queue.enqueue(req2).await;
        let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
        assert_eq!(resp2.best_worker, WorkerWithDpRank::new(1, 0));

        let (mut req3, rx3) = make_request("req-3", isl);
        req3.effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 8.0);
        req3.effective_overlap_blocks
            .insert(WorkerWithDpRank::new(1, 0), 2.0);
        req3.effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 128);
        req3.effective_cached_tokens
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
    async fn update_requeues_head_if_workers_become_busy_during_refresh() {
        let block_size = 16u32;
        let isl = 64usize;
        let refresher = Arc::new(BlockingRefresher::new(RefreshedOverlap::default()));
        let (queue, slots) =
            make_queue_with_blocking_refresher(1, block_size, isl, Some(0.0), refresher.clone());

        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _ = rx1.await.expect("rx1 dropped").expect("req-1 failed");

        let (mut req2, mut rx2) = make_request("req-2", isl);
        req2.effective_overlap_blocks
            .insert(WorkerWithDpRank::new(0, 0), 4.0);
        req2.effective_cached_tokens
            .insert(WorkerWithDpRank::new(0, 0), 64);
        queue
            .enqueue_with_block_hashes(req2, Some(vec![LocalBlockHash(42)]))
            .await;
        assert_eq!(queue.pending_count(), 1);

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
                    worker: WorkerWithDpRank::new(0, 0),
                    lora_name: None,
                },
                decay_now(),
            )
            .unwrap();

        refresher.release_one();
        update.await.unwrap();

        assert_eq!(queue.pending_count(), 1, "request should be requeued");
        assert!(
            rx2.try_recv().is_err(),
            "request must remain queued while worker became busy again"
        );

        slots
            .mark_prefill_completed(&"occupy-during-refresh".to_string(), decay_now())
            .unwrap();
        slots
            .free(&"occupy-during-refresh".to_string(), decay_now())
            .unwrap();

        let update = {
            let queue = Arc::clone(&queue);
            tokio::spawn(async move {
                queue.update().await;
            })
        };
        refresher.wait_for_calls(2).await;
        refresher.release_one();
        update.await.unwrap();

        let resp2 = rx2.await.expect("rx2 dropped").expect("req-2 failed");
        assert_eq!(refresher.calls.load(Ordering::Relaxed), 2);
        assert_eq!(resp2.best_worker, WorkerWithDpRank::new(0, 0));
        assert_eq!(queue.pending_count(), 0);
    }
}
