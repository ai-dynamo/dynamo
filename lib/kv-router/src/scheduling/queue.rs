// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

use tokio::sync::{mpsc, oneshot, watch};
use tokio::time::Instant;

use super::policy::{FcfsPolicy, SchedulingPolicy};
use super::prefill_load::PrefillLoadEstimator;
use super::selector::{DefaultWorkerSelector, WorkerSelector};
use super::types::{
    KvSchedulerError, OverloadedWorkerProvider, RoutingEligibility, SchedulingContext,
    SchedulingRequest, SchedulingResponse, pinned_worker_config,
};
use crate::protocols::{PrefillLoadHint, WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::sequences::{ActiveSequencesMultiWorker, SequencePublisher, SequenceRequest};

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

const ADMISSION_CHANNEL_CAPACITY: usize = 65_536;

/// Entry in the priority queue, ordered by key (higher key = higher priority).
struct QueueEntry<K: Ord + Eq> {
    key: K,
    request: SchedulingRequest,
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
> {
    pending: BinaryHeap<QueueEntry<S::Key>>,
    pending_count: Arc<AtomicUsize>,
    pending_isl_tokens: Arc<AtomicUsize>,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    threshold_frac: Option<f64>,
    start_time: Instant,
    block_size: u32,
    selector: Sel,
    policy: S,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
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
    _marker: PhantomData<fn() -> (S, Sel)>,
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
> SchedulerQueue<P, C, S, Sel>
{
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        block_size: u32,
        selector: Sel,
        policy: S,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    ) -> Self {
        Self::new_with_overload_provider(
            slots,
            workers_with_configs,
            threshold_frac,
            block_size,
            selector,
            policy,
            prefill_load_estimator,
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
        policy: S,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    ) -> Self {
        if let Some(frac) = threshold_frac {
            tracing::info!("Router queue enabled with threshold fraction {frac}");
        }
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
            start_time: Instant::now(),
            block_size,
            selector,
            policy,
            prefill_load_estimator,
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
            _marker: PhantomData,
        }
    }

    /// Register externally-provided workers in the slot tracker.
    ///
    /// Looks up DP rank/size from the discovery watch channel; defaults to
    /// `(0, 1)` for workers not yet known to discovery.
    pub fn register_workers(&self, worker_ids: &std::collections::HashSet<u64>) {
        let discovery_workers = self.workers_with_configs.borrow();
        let dp_range: std::collections::HashMap<u64, (u32, u32)> = worker_ids
            .iter()
            .map(|&id| {
                let (dp_start, dp_size) = discovery_workers
                    .get(&id)
                    .map(|runtime_config| {
                        (
                            runtime_config.data_parallel_start_rank(),
                            runtime_config.data_parallel_size(),
                        )
                    })
                    .unwrap_or((0, 1));
                (id, (dp_start, dp_size))
            })
            .collect();
        self.slots.register_external_workers(&dp_range);
    }

    /// Enqueue a new request.
    /// If queueing is disabled or workers have capacity, schedule immediately.
    /// Otherwise park in the pending heap.
    ///
    /// When `allowed_worker_ids` is set on the request without an exact pin
    /// (external routing), the capacity check is skipped.
    pub async fn enqueue(&self, mut request: SchedulingRequest) {
        let eligibility = request.eligibility();

        if let Err(error) = eligibility.validate_pinned_worker_allowed() {
            request.respond(Err(error));
            return;
        }

        let (ack_tx, ack_rx) = oneshot::channel();
        let command = AdmissionCommand::Enqueue { request, ack_tx };

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
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike + Send + Sync + 'static,
    S: SchedulingPolicy + 'static,
    Sel: WorkerSelector<C> + Send + 'static,
> SchedulerQueueActor<P, C, S, Sel>
{
    async fn run(mut self, mut rx: mpsc::Receiver<AdmissionCommand>) {
        while let Some(command) = rx.recv().await {
            match command {
                AdmissionCommand::Enqueue { request, ack_tx } => {
                    self.handle_enqueue(request);
                    let _ = ack_tx.send(());
                }
                AdmissionCommand::Update { ack_tx } => {
                    self.handle_update();
                    let _ = ack_tx.send(());
                }
            }
        }
    }

    fn handle_enqueue(&mut self, request: SchedulingRequest) {
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
            tracing::debug!("all workers prefill-busy, queueing request");
            let arrival_offset = self.start_time.elapsed();
            let key = {
                let workers = self.workers_with_configs.borrow();
                self.policy
                    .enqueue_key(arrival_offset, SchedulingContext::new(&request, &workers))
            };
            let isl_tokens = request.isl_tokens;
            self.pending.push(QueueEntry { key, request });
            self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_add(isl_tokens, AtomicOrdering::Relaxed);
            return;
        }

        self.admit_one(request, decay_now);
    }

    fn handle_update(&mut self) {
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
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_sub(entry.request.isl_tokens, AtomicOrdering::Relaxed);
            tracing::debug!("scheduling request from pending queue");
            self.admit_one(entry.request, decay_now);
        }
    }

    /// Run the full scheduling pipeline for a single request:
    /// compute potential load -> select worker -> respond -> book via add_request.
    fn admit_one(&self, mut request: SchedulingRequest, decay_now: Instant) {
        let (decode_blocks, prefill_tokens) = self.slots.potential_blocks_and_tokens_at(
            request.token_seq.as_deref(),
            &request.prefill_token_deltas(),
            decay_now,
        );
        request.decode_blocks = decode_blocks;
        request.prefill_tokens = prefill_tokens;

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

        request.respond(Ok(SchedulingResponse {
            best_worker: selection.worker,
            effective_overlap_blocks: selection.effective_overlap_blocks,
            cached_tokens: selection.cached_tokens,
        }));

        if !request.update_states {
            return;
        }

        let Some(request_id) = request.maybe_request_id else {
            tracing::error!("No request_id provided to add_request to the slot tracker");
            return;
        };

        let prefill_load_hint = self.prefill_load_hint_for(
            request.isl_tokens,
            selection.cached_tokens,
            request.track_prefill_tokens,
        );

        if let Err(e) = self.slots.add_request(
            SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: request.token_seq,
                track_prefill_tokens: request.track_prefill_tokens,
                expected_output_tokens: request.expected_output_tokens,
                prefill_load_hint,
                worker: selection.worker,
                lora_name: request.lora_name.clone(),
            },
            Instant::now(),
        ) {
            tracing::warn!("Failed to add request {request_id}: {e}");
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

        let prefix = cached_tokens.min(isl_tokens);
        let effective_isl = isl_tokens.saturating_sub(prefix);
        if effective_isl == 0 {
            return None;
        }

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
        threshold: f64,
        eligibility: RoutingEligibility<'_>,
        decay_now: Instant,
    ) -> bool {
        let active_tokens = self.slots.active_tokens(decay_now);
        let configs = self.workers_with_configs.borrow();

        if let Some(worker) = eligibility.pinned_worker() {
            let Ok(config) = pinned_worker_config::<C>(&*configs, worker) else {
                return false;
            };
            if !eligibility.allows_worker(worker.worker_id, config) {
                return false;
            }

            let max_batched = config
                .max_num_batched_tokens()
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);
            let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
            return (tokens as f64) > threshold * (max_batched as f64);
        }

        let mut checked_any = false;
        for (&worker_id, config) in configs.iter() {
            if !eligibility.allows_worker(worker_id, config) {
                continue;
            }
            let dp_size = config.data_parallel_size();
            let dp_start_rank = config.data_parallel_start_rank();
            let max_batched = config
                .max_num_batched_tokens()
                .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS);

            for dp_rank in dp_start_rank..dp_start_rank + dp_size {
                checked_any = true;
                let worker = WorkerWithDpRank::new(worker_id, dp_rank);
                let tokens = active_tokens.get(&worker).copied().unwrap_or(0);
                if (tokens as f64) <= threshold * (max_batched as f64) {
                    return false;
                }
            }
        }
        checked_any
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::{Arc, Condvar, Mutex as StdMutex};
    use std::time::Duration;

    use rustc_hash::FxHashMap;
    use tokio::sync::{Barrier, watch};

    use super::*;
    use crate::protocols::{WorkerSelectionResult, WorkerWithDpRank};
    use crate::scheduling::types::KvSchedulerError;
    use crate::sequences::ActiveSequencesMultiWorker;
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

            let Some(worker) = workers
                .iter()
                .filter(|(worker_id, config)| eligibility.allows_worker(**worker_id, *config))
                .flat_map(|(worker_id, config)| {
                    let dp_start = config.data_parallel_start_rank();
                    let dp_end = dp_start + config.data_parallel_size();
                    (dp_start..dp_end)
                        .map(move |dp_rank| WorkerWithDpRank::new(*worker_id, dp_rank))
                })
                .min_by_key(|worker| {
                    (
                        request.prefill_tokens_for(*worker),
                        request.decode_blocks.get(worker).copied().unwrap_or(0),
                        worker.worker_id,
                        worker.dp_rank,
                    )
                })
            else {
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
            block_size,
            selector,
            FcfsPolicy,
            None,
            Some(overloaded_worker_provider),
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
            decode_blocks: FxHashMap::default(),
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
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
        let mut dp_range = std::collections::HashMap::new();
        dp_range.insert(100_u64, (0_u32, 1_u32));
        dp_range.insert(200_u64, (0_u32, 1_u32));
        slots.register_external_workers(&dp_range);

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
        let mut dp1 = std::collections::HashMap::new();
        dp1.insert(10_u64, (0_u32, 1_u32));
        slots.register_external_workers(&dp1);

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
        let mut dp2 = std::collections::HashMap::new();
        dp2.insert(20_u64, (0_u32, 1_u32));
        slots.register_external_workers(&dp2);

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
        let mut dp = std::collections::HashMap::new();
        dp.insert(1_u64, (0_u32, 1_u32));
        dp.insert(2_u64, (0_u32, 1_u32));
        dp.insert(3_u64, (0_u32, 1_u32));
        slots.register_external_workers(&dp);

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
            decode_blocks: FxHashMap::default(),
            prefill_tokens: FxHashMap::default(),
            track_prefill_tokens: true,
            router_config_override: None,
            update_states: true,
            lora_name: None,
            priority_jump: 0.0,
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
}
