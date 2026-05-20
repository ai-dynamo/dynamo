// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::time::Duration;

use tokio::sync::Mutex;
use tokio::sync::watch;
use tokio::time::Instant;

use super::overlap_refresh::OverlapScoresRefresh;
use super::policy::{FcfsPolicy, SchedulingPolicy};
use super::prefill_load::PrefillLoadEstimator;
use super::selector::{DefaultWorkerSelector, WorkerSelector};
use super::types::{
    OverloadedWorkerProvider, RoutingEligibility, SchedulingContext, SchedulingRequest,
    SchedulingResponse, pinned_worker_config,
};
use crate::protocols::{
    LocalBlockHash, PrefillLoadHint, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use crate::sequences::{ActiveSequencesMultiWorker, SequencePublisher, SequenceRequest};

/// Default wait threshold after which a dequeued request gets a fresh overlap-score lookup.
/// Override with `DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS`. A value of `0` disables refresh.
const DEFAULT_OVERLAP_REFRESH_AFTER_SECS: f64 = 10.0;

fn default_overlap_refresh_after() -> Duration {
    // DEFAULT_OVERLAP_REFRESH_AFTER_SECS is a small finite literal; try_from_secs_f64
    // cannot fail here but we still avoid the panicking constructor for consistency.
    Duration::try_from_secs_f64(DEFAULT_OVERLAP_REFRESH_AFTER_SECS)
        .unwrap_or(Duration::from_secs(10))
}

fn read_overlap_refresh_after() -> Option<Duration> {
    let raw = match std::env::var("DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS") {
        Ok(v) => v,
        Err(_) => return Some(default_overlap_refresh_after()),
    };
    let parsed: f64 = match raw.parse() {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(
                value = %raw,
                "invalid DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS, falling back to default {DEFAULT_OVERLAP_REFRESH_AFTER_SECS}s"
            );
            return Some(default_overlap_refresh_after());
        }
    };
    if !parsed.is_finite() {
        tracing::warn!(
            value = %raw,
            "non-finite DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS, disabling overlap refresh"
        );
        return None;
    }
    if parsed <= 0.0 {
        // Explicit disable (e.g. `0` or a negative value).
        return None;
    }
    // try_from_secs_f64 also rejects values that don't fit in u64 seconds, so an env value
    // like 1e100 disables the refresh instead of panicking in Duration::from_secs_f64.
    match Duration::try_from_secs_f64(parsed) {
        Ok(d) => Some(d),
        Err(err) => {
            tracing::warn!(
                value = %raw,
                error = %err,
                "DYN_ROUTER_OVERLAP_REFRESH_AFTER_SECS is out of range; disabling overlap refresh"
            );
            None
        }
    }
}

/// Large default for max_num_batched_tokens when not configured (effectively disables queueing for that worker)
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Entry in the priority queue, ordered by key (higher key = higher priority).
struct QueueEntry<K: Ord + Eq> {
    key: K,
    request: SchedulingRequest,
    /// Instant at which the entry was parked. Used to gate overlap-score refresh on dequeue.
    enqueue_at: Instant,
    /// Block hashes that produced the request's overlap scores. Present when the caller
    /// wired in [`OverlapScoresRefresh`]; required to re-query the indexer at dequeue.
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
    pending: Mutex<BinaryHeap<QueueEntry<S::Key>>>,
    /// Serializes admission so worker selection always sees prior bookings.
    admission_gate: Mutex<()>,
    /// Number of requests currently parked in the pending heap.
    /// Incremented after push, decremented after pop. Lock-free reads via `Relaxed` load.
    ///
    /// Reflects heap occupancy only: during the overlap-refresh window in `update()` a
    /// popped request is "in flight" between `pop` and `admit_one` and is counted as
    /// neither pending nor admitted. The undercount is bounded by the refresh duration
    /// (single indexer lookup, typically µs); if you need an "outstanding work" gauge,
    /// observe pending + active-token deltas together rather than this counter alone.
    pending_count: AtomicUsize,
    /// Sum of `isl_tokens` for requests currently parked in the pending heap.
    /// Same in-flight caveat as `pending_count`.
    pending_isl_tokens: AtomicUsize,
    slots: Arc<ActiveSequencesMultiWorker<P>>,
    workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
    /// Cached threshold fraction; None means queueing is disabled.
    threshold_frac: Option<f64>,
    /// Reference instant for computing arrival offsets.
    start_time: Instant,
    block_size: u32,
    selector: Sel,
    policy: S,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    /// Re-query overlap scores at dequeue time when a request has been waiting longer than
    /// `overlap_refresh_after`. `None` disables the refresh entirely.
    overlap_scores_refresh: Option<Arc<dyn OverlapScoresRefresh>>,
    overlap_refresh_after: Option<Duration>,
    overloaded_worker_provider: Option<OverloadedWorkerProvider>,
}

impl<
    P: SequencePublisher + 'static,
    C: WorkerConfigLike,
    S: SchedulingPolicy,
    Sel: WorkerSelector<C>,
> SchedulerQueue<P, C, S, Sel>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        slots: Arc<ActiveSequencesMultiWorker<P>>,
        workers_with_configs: watch::Receiver<HashMap<WorkerId, C>>,
        threshold_frac: Option<f64>,
        block_size: u32,
        selector: Sel,
        policy: S,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        overlap_scores_refresh: Option<Arc<dyn OverlapScoresRefresh>>,
    ) -> Self {
        Self::new_with_overload_provider(
            slots,
            workers_with_configs,
            threshold_frac,
            block_size,
            selector,
            policy,
            prefill_load_estimator,
            overlap_scores_refresh,
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
        overlap_scores_refresh: Option<Arc<dyn OverlapScoresRefresh>>,
        overloaded_worker_provider: Option<OverloadedWorkerProvider>,
    ) -> Self {
        if let Some(frac) = threshold_frac {
            tracing::info!("Router queue enabled with threshold fraction {frac}");
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
        Self {
            pending: Mutex::new(BinaryHeap::new()),
            admission_gate: Mutex::new(()),
            pending_count: AtomicUsize::new(0),
            pending_isl_tokens: AtomicUsize::new(0),
            slots,
            workers_with_configs,
            threshold_frac,
            start_time: Instant::now(),
            block_size,
            selector,
            policy,
            prefill_load_estimator,
            overlap_scores_refresh,
            overlap_refresh_after,
            overloaded_worker_provider,
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
    pub async fn enqueue(&self, request: SchedulingRequest) {
        self.enqueue_with_block_hashes(request, None).await;
    }

    /// Like [`enqueue`](Self::enqueue) but also stashes the block hashes used to compute
    /// the request's overlap scores. When the queue is configured with an
    /// [`OverlapScoresRefresh`], these hashes let the queue re-query the indexer for fresh
    /// scores if the request waits past `overlap_refresh_after` before dispatch.
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

        let _admission = self.admission_gate.lock().await;
        let decay_now = Instant::now();

        let Some(threshold) = self.threshold_frac else {
            // Queueing disabled — the request dispatches immediately and never sits in the
            // pending heap, so `block_hashes` would never be consulted. Drop them on the floor.
            self.admit_one(request, decay_now).await;
            return;
        };

        if eligibility.bypasses_capacity_check() {
            // Bypasses the capacity check (external routing) — admitted without ever queueing,
            // so `block_hashes` are intentionally discarded.
            self.admit_one(request, decay_now).await;
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
            self.pending.lock().await.push(QueueEntry {
                key,
                request,
                enqueue_at: decay_now,
                block_hashes,
            });
            self.pending_count.fetch_add(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_add(isl_tokens, AtomicOrdering::Relaxed);
        } else {
            // Workers have capacity — admit without queueing. `block_hashes` are intentionally
            // discarded; refresh is only meaningful for requests that actually wait.
            self.admit_one(request, decay_now).await;
        }
    }

    /// Called on prefill_complete/free. Drains pending requests while workers have capacity.
    /// Each scheduled request updates active_tokens via add_request, so the prefill-busy check
    /// sees fresh state on the next iteration.
    pub async fn update(&self) {
        let Some(threshold) = self.threshold_frac else {
            return;
        };

        if S::DYNAMIC {
            let now = self.start_time.elapsed();
            let mut heap = self.pending.lock().await;
            let workers = self.workers_with_configs.borrow();
            let rekeyed: Vec<_> = std::mem::take(&mut *heap)
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
            *heap = BinaryHeap::from(rekeyed);
        }

        loop {
            let admission = self.admission_gate.lock().await;
            let decay_now = Instant::now();
            let mut heap = self.pending.lock().await;
            let Some(front) = heap.peek() else {
                break;
            };
            // TODO: This preserves head-of-line blocking for now to keep queue
            // drain overhead bounded to the heap front. A blocked pinned or
            // otherwise constrained request can temporarily stall later
            // schedulable entries until we adopt a cheaper non-HOL strategy.
            if self.all_workers_prefill_busy(threshold, front.request.eligibility(), decay_now) {
                break;
            }
            let entry = heap.pop().expect("heap front vanished before pop");
            drop(heap);
            self.pending_count.fetch_sub(1, AtomicOrdering::Relaxed);
            self.pending_isl_tokens
                .fetch_sub(entry.request.isl_tokens, AtomicOrdering::Relaxed);
            let mut request = entry.request;
            let block_hashes = entry.block_hashes;
            let enqueue_at = entry.enqueue_at;

            if self.should_refresh_overlap(block_hashes.as_deref(), enqueue_at, decay_now) {
                // The indexer lookup may take milliseconds; release the admission gate so
                // concurrent `enqueue` calls aren't stalled behind this request's refresh.
                drop(admission);
                self.maybe_refresh_overlap(
                    &mut request,
                    block_hashes.as_deref(),
                    enqueue_at,
                    decay_now,
                )
                .await;
                // Reacquire the gate before admit_one so worker-state mutations stay serialized.
                // Use a fresh decay_now and recheck capacity for visibility — we already
                // committed to admitting this entry, but a concurrent enqueue may have filled
                // the slot while we awaited the refresh.
                let _admission = self.admission_gate.lock().await;
                let admit_now = Instant::now();
                if self.all_workers_prefill_busy(threshold, request.eligibility(), admit_now) {
                    tracing::debug!(
                        "all workers prefill-busy after overlap refresh; admitting refreshed request anyway"
                    );
                }
                tracing::debug!("scheduling refreshed request from pending queue");
                self.admit_one(request, admit_now).await;
            } else {
                tracing::debug!("scheduling request from pending queue");
                self.admit_one(request, decay_now).await;
            }
        }
    }

    /// Cheap predicate that mirrors the conditions inside [`Self::maybe_refresh_overlap`].
    /// Used by [`Self::update`] to decide whether to drop the admission gate around the
    /// (potentially slow) indexer lookup.
    fn should_refresh_overlap(
        &self,
        block_hashes: Option<&[LocalBlockHash]>,
        enqueue_at: Instant,
        decay_now: Instant,
    ) -> bool {
        let Some(refresh_after) = self.overlap_refresh_after else {
            return false;
        };
        if self.overlap_scores_refresh.is_none() {
            return false;
        }
        let Some(block_hashes) = block_hashes else {
            return false;
        };
        if block_hashes.is_empty() {
            return false;
        }
        decay_now.saturating_duration_since(enqueue_at) >= refresh_after
    }

    /// If the queue is configured for overlap-score refresh and the dequeued request waited
    /// long enough, re-query the indexer and overwrite the request's overlap fields.
    /// Refresh failures are non-fatal: the request still dispatches, just with stale scores.
    ///
    /// Self-contained: re-checks every precondition (refresher present, threshold met,
    /// non-empty block hashes) so callers can invoke it unconditionally. `should_refresh_overlap`
    /// is a cheap mirror of these checks used by `update()` to avoid a pointless gate cycle.
    ///
    /// Note: `request.shared_cache_hits` is intentionally **not** refreshed. Shared-cache hits
    /// are computed by a separate `shared_cache.check_blocks(tokens, ...)` query that needs the
    /// original tokens, not just block hashes; re-running it would double the per-request
    /// router latency. Refreshing only the indexer-derived fields is correct as long as
    /// shared-cache state doesn't change as rapidly as the radix tree (the common case).
    async fn maybe_refresh_overlap(
        &self,
        request: &mut SchedulingRequest,
        block_hashes: Option<&[LocalBlockHash]>,
        enqueue_at: Instant,
        decay_now: Instant,
    ) {
        let Some(refresh_after) = self.overlap_refresh_after else {
            return;
        };
        let Some(refresher) = &self.overlap_scores_refresh else {
            return;
        };
        let Some(block_hashes) = block_hashes else {
            return;
        };
        // Defensive: an empty slice would overwrite valid overlap data with empty maps.
        // `should_refresh_overlap` already filters this out for update(), but external callers
        // (and the unit tests) reach this entry point directly.
        if block_hashes.is_empty() {
            return;
        }
        let wait = decay_now.saturating_duration_since(enqueue_at);
        if wait < refresh_after {
            return;
        }

        let Some(refreshed) = refresher.refresh(block_hashes).await else {
            // wait_ms here reflects total time from enqueue to now (after the refresh await),
            // so the log captures admission-gate wait + refresh duration.
            let wait_ms = enqueue_at.elapsed().as_millis() as u64;
            tracing::debug!(
                request_id = request.maybe_request_id.as_deref().unwrap_or("unknown"),
                wait_ms = wait_ms,
                "overlap refresh returned None; dispatching with stale scores"
            );
            return;
        };

        let wait_ms = enqueue_at.elapsed().as_millis() as u64;
        tracing::info!(
            request_id = request.maybe_request_id.as_deref().unwrap_or("unknown"),
            wait_ms = wait_ms,
            "refreshed overlap scores after long queue wait"
        );
        request.tier_overlap_blocks = refreshed.tier_overlap_blocks;
        request.effective_overlap_blocks = refreshed.effective_overlap_blocks;
        request.effective_cached_tokens = refreshed.effective_cached_tokens;
        // shared_cache_hits intentionally left alone — see function-level doc.
    }

    /// Run the full scheduling pipeline for a single request:
    /// compute potential load -> select worker -> respond -> book via add_request.
    async fn admit_one(&self, mut request: SchedulingRequest, decay_now: Instant) {
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

    /// Number of requests currently parked in the pending queue (lock-free).
    pub fn pending_count(&self) -> usize {
        self.pending_count.load(AtomicOrdering::Relaxed)
    }

    /// Sum of `isl_tokens` for requests currently parked in the pending queue (lock-free).
    pub fn pending_isl_tokens(&self) -> usize {
        self.pending_isl_tokens.load(AtomicOrdering::Relaxed)
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
    fn make_queue_with_custom_selector<Sel: WorkerSelector<SimpleWorkerConfig>>(
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
            None,
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

    // --- Overlap refresh on dequeue ---------------------------------------------------

    struct CountingRefresher {
        calls: std::sync::atomic::AtomicUsize,
        response: super::super::overlap_refresh::RefreshedOverlap,
    }

    #[async_trait::async_trait]
    impl super::super::overlap_refresh::OverlapScoresRefresh for CountingRefresher {
        async fn refresh(
            &self,
            _block_hashes: &[crate::protocols::LocalBlockHash],
        ) -> Option<super::super::overlap_refresh::RefreshedOverlap> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(self.response.clone())
        }
    }

    fn refresh_test_queue(
        refresh: Option<Arc<dyn super::super::overlap_refresh::OverlapScoresRefresh>>,
        refresh_after: Option<Duration>,
    ) -> Arc<SchedulerQueue<NoopSequencePublisher, SimpleWorkerConfig>> {
        let block_size = 16u32;
        let dp_range: HashMap<u64, (u32, u32)> = HashMap::from([(0, (0, 1))]);
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));
        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        configs.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(64),
                ..Default::default()
            },
        );
        let (_cfg_tx, cfg_rx) = watch::channel(configs);
        let mut queue = SchedulerQueue::new(
            slots,
            cfg_rx,
            None,
            block_size,
            DefaultWorkerSelector::new(None, "test"),
            FcfsPolicy,
            None,
            refresh,
        );
        // The constructor reads the env var; override here so tests are deterministic and
        // don't depend on global env state.
        queue.overlap_refresh_after = refresh_after;
        Arc::new(queue)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maybe_refresh_overwrites_overlap_fields_after_threshold() {
        let mut tier = crate::scheduling::types::TierOverlapBlocks::default();
        tier.device.insert(WorkerWithDpRank::new(0, 0), 7);
        let refresher = Arc::new(CountingRefresher {
            calls: std::sync::atomic::AtomicUsize::new(0),
            response: super::super::overlap_refresh::RefreshedOverlap {
                tier_overlap_blocks: tier.clone(),
                effective_overlap_blocks: HashMap::from([(WorkerWithDpRank::new(0, 0), 7.0)]),
                effective_cached_tokens: HashMap::from([(WorkerWithDpRank::new(0, 0), 112)]),
            },
        });
        let queue = refresh_test_queue(Some(refresher.clone()), Some(Duration::from_millis(10)));

        let (mut request, _rx) = make_request("req", 256);
        let hashes = [LocalBlockHash(42)];
        let enqueue_at = Instant::now();
        let decay_now = enqueue_at + Duration::from_millis(20);
        queue
            .maybe_refresh_overlap(&mut request, Some(&hashes), enqueue_at, decay_now)
            .await;

        assert_eq!(
            refresher.calls.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(request.tier_overlap_blocks.device.len(), 1);
        assert_eq!(
            *request
                .effective_overlap_blocks
                .get(&WorkerWithDpRank::new(0, 0))
                .unwrap(),
            7.0
        );
        assert_eq!(
            *request
                .effective_cached_tokens
                .get(&WorkerWithDpRank::new(0, 0))
                .unwrap(),
            112
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maybe_refresh_skips_when_under_threshold() {
        let refresher = Arc::new(CountingRefresher {
            calls: std::sync::atomic::AtomicUsize::new(0),
            response: super::super::overlap_refresh::RefreshedOverlap::default(),
        });
        let queue = refresh_test_queue(Some(refresher.clone()), Some(Duration::from_secs(10)));

        let (mut request, _rx) = make_request("req", 256);
        let original_overlap = request.effective_overlap_blocks.clone();
        let enqueue_at = Instant::now();
        let decay_now = enqueue_at + Duration::from_millis(5);
        queue
            .maybe_refresh_overlap(&mut request, Some(&[]), enqueue_at, decay_now)
            .await;

        assert_eq!(
            refresher.calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert_eq!(request.effective_overlap_blocks, original_overlap);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maybe_refresh_skips_when_no_block_hashes() {
        let refresher = Arc::new(CountingRefresher {
            calls: std::sync::atomic::AtomicUsize::new(0),
            response: super::super::overlap_refresh::RefreshedOverlap::default(),
        });
        let queue = refresh_test_queue(Some(refresher.clone()), Some(Duration::from_millis(10)));

        let (mut request, _rx) = make_request("req", 256);
        let enqueue_at = Instant::now();
        let decay_now = enqueue_at + Duration::from_millis(20);
        queue
            .maybe_refresh_overlap(&mut request, None, enqueue_at, decay_now)
            .await;

        assert_eq!(
            refresher.calls.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maybe_refresh_is_noop_without_refresher() {
        let queue = refresh_test_queue(None, Some(Duration::from_millis(10)));
        let (mut request, _rx) = make_request("req", 256);
        let original_overlap = request.effective_overlap_blocks.clone();
        let enqueue_at = Instant::now();
        let decay_now = enqueue_at + Duration::from_millis(20);
        queue
            .maybe_refresh_overlap(&mut request, Some(&[]), enqueue_at, decay_now)
            .await;
        assert_eq!(request.effective_overlap_blocks, original_overlap);
    }

    struct FailingRefresher;
    #[async_trait::async_trait]
    impl super::super::overlap_refresh::OverlapScoresRefresh for FailingRefresher {
        async fn refresh(
            &self,
            _block_hashes: &[crate::protocols::LocalBlockHash],
        ) -> Option<super::super::overlap_refresh::RefreshedOverlap> {
            None
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maybe_refresh_returns_none_keeps_original_scores() {
        let queue = refresh_test_queue(
            Some(Arc::new(FailingRefresher)
                as Arc<
                    dyn super::super::overlap_refresh::OverlapScoresRefresh,
                >),
            Some(Duration::from_millis(10)),
        );
        let (mut request, _rx) = make_request("req", 256);
        let original_overlap = request.effective_overlap_blocks.clone();
        let original_tier = request.tier_overlap_blocks.clone();
        let hashes = [LocalBlockHash(42)];
        let enqueue_at = Instant::now();
        let decay_now = enqueue_at + Duration::from_millis(20);
        queue
            .maybe_refresh_overlap(&mut request, Some(&hashes), enqueue_at, decay_now)
            .await;
        assert_eq!(request.effective_overlap_blocks, original_overlap);
        assert_eq!(request.tier_overlap_blocks.device, original_tier.device);
    }

    /// End-to-end coverage of the production refresh path: enqueue with `block_hashes`,
    /// wait past the threshold, trigger `update()`, and verify the refresher was invoked
    /// via the queue rather than via `maybe_refresh_overlap` directly. This is the path
    /// that actually matters in production (admission gate drop+reacquire, capacity
    /// recheck, fresh `Instant` for admit_one); the other refresh tests exercise the
    /// inner function in isolation.
    #[tokio::test(flavor = "multi_thread")]
    async fn update_drains_via_full_refresh_path() {
        use std::sync::atomic::Ordering as AtomicOrd;

        let block_size = 16u32;
        let isl = 64usize;
        let refresher = Arc::new(CountingRefresher {
            calls: std::sync::atomic::AtomicUsize::new(0),
            response: super::super::overlap_refresh::RefreshedOverlap {
                tier_overlap_blocks: Default::default(),
                effective_overlap_blocks: HashMap::from([(WorkerWithDpRank::new(0, 0), 9.0)]),
                effective_cached_tokens: HashMap::from([(WorkerWithDpRank::new(0, 0), 80)]),
            },
        });

        let dp_range: HashMap<u64, (u32, u32)> = HashMap::from([(0, (0, 1))]);
        let slots = Arc::new(ActiveSequencesMultiWorker::new(
            NoopSequencePublisher,
            block_size as usize,
            dp_range,
            false,
            0,
            "test",
        ));
        let mut configs: HashMap<u64, SimpleWorkerConfig> = HashMap::new();
        configs.insert(
            0,
            SimpleWorkerConfig {
                max_num_batched_tokens: Some(isl as u64),
                ..Default::default()
            },
        );
        let (_cfg_tx, cfg_rx) = watch::channel(configs);
        let mut queue = SchedulerQueue::new(
            Arc::clone(&slots),
            cfg_rx,
            Some(0.0), // threshold=0 → busy as soon as any tokens are active
            block_size,
            DefaultWorkerSelector::new(None, "test"),
            FcfsPolicy,
            None,
            Some(refresher.clone() as Arc<dyn super::super::overlap_refresh::OverlapScoresRefresh>),
        );
        // Set the threshold deterministically; the constructor reads env state otherwise.
        queue.overlap_refresh_after = Some(Duration::from_millis(20));
        let queue = Arc::new(queue);

        // First request admits immediately (no active tokens yet) and saturates the worker.
        let (req1, rx1) = make_request("req-1", isl);
        queue.enqueue(req1).await;
        let _ = rx1
            .await
            .expect("rx1 dropped")
            .expect("req-1 scheduling failed");

        // Second request: with threshold=0 and req-1 holding tokens, this parks in the heap.
        let (req2, rx2) = make_request("req-2", isl);
        let hashes = vec![LocalBlockHash(42)];
        queue.enqueue_with_block_hashes(req2, Some(hashes)).await;
        assert_eq!(queue.pending_count(), 1, "req-2 must queue, not admit");
        assert_eq!(
            refresher.calls.load(AtomicOrd::Relaxed),
            0,
            "refresh must not fire at enqueue time"
        );

        // Cross the refresh threshold while req-2 is parked.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Free req-1 so the worker is no longer busy, then drive update().
        slots.free(&"req-1".to_string(), decay_now()).unwrap();
        queue.update().await;

        let _ = rx2
            .await
            .expect("rx2 dropped")
            .expect("req-2 scheduling failed");
        assert_eq!(
            refresher.calls.load(AtomicOrd::Relaxed),
            1,
            "refresh must fire exactly once via the update() path"
        );
        assert_eq!(queue.pending_count(), 0, "heap must drain");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn maybe_refresh_skips_when_block_hashes_empty() {
        // Defensive guard: an empty slice would overwrite valid overlap data with empty maps.
        let refresher = Arc::new(CountingRefresher {
            calls: std::sync::atomic::AtomicUsize::new(0),
            response: super::super::overlap_refresh::RefreshedOverlap::default(),
        });
        let queue = refresh_test_queue(Some(refresher.clone()), Some(Duration::from_millis(10)));
        let (mut request, _rx) = make_request("req", 256);
        let original_overlap = request.effective_overlap_blocks.clone();
        let enqueue_at = Instant::now();
        let decay_now = enqueue_at + Duration::from_millis(20);
        queue
            .maybe_refresh_overlap(&mut request, Some(&[]), enqueue_at, decay_now)
            .await;
        assert_eq!(
            refresher.calls.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "refresher must not be called for empty block_hashes"
        );
        assert_eq!(request.effective_overlap_blocks, original_overlap);
    }
}
