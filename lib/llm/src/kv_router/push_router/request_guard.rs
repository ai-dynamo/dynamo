// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashMap,
    sync::{
        Arc, Weak,
        atomic::{AtomicU8, Ordering},
    },
    time::Duration,
};

use dynamo_kv_router::{
    indexer::{ApproximateAcquireMode, ApproximateLruBlock, RoutingDecisionHashes},
    multi_worker_sequence::active_request_expiry_duration,
    protocols::{
        BlockExtraInfo, BlockHashOptions, WorkerWithDpRank, compute_block_hash_for_seq,
        compute_next_seq_hash,
    },
    scheduling::AttemptId,
    selector::WorkerSelector,
};
use dynamo_runtime::{
    error::DynamoError,
    metrics::frontend_perf::{STAGE_DISPATCH, StageGuard},
    protocols::annotated::Annotated,
};
use tokio::sync::{Notify, watch};
use tokio::time::Instant;

use crate::{
    kv_router::{
        KvRouter,
        indexer::ApproximateRequestLease,
        metrics::RouterRequestMetrics,
        scheduler::{DefaultWorkerSelector, SchedulerBookingCleanup, SchedulerBookingDescriptor},
    },
    local_model::runtime_config::ModelRuntimeConfig,
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        preprocessor::MigrationState,
        timing::{RequestPhase, RequestTracker},
    },
};

#[derive(Clone)]
struct OutputHashBranch {
    tail: Vec<u32>,
    parent_hash: Option<u64>,
    next_position: usize,
    first_mm_info: Option<BlockExtraInfo>,
}

struct MaterializedOutputBlocks {
    parent_hash: Option<u64>,
    blocks: Vec<ApproximateLruBlock>,
    start_position: usize,
    private_blocks: usize,
}

/// Incrementally extends the same canonical hash chain used for prompt routing.
struct CanonicalOutputTracker {
    template: OutputHashBranch,
    branches: HashMap<u32, OutputHashBranch>,
    block_size: u32,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
    is_eagle: bool,
    reported_private_blocks: usize,
}

impl CanonicalOutputTracker {
    fn new(request: &PreprocessedRequest, block_size: u32, is_eagle: bool) -> Self {
        let (tokens, mm_infos) = request.block_mm_routing_info();
        Self::from_parts(
            tokens,
            mm_infos,
            block_size,
            is_eagle,
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.lora_name.clone()),
            request
                .routing
                .as_ref()
                .and_then(|routing| routing.cache_namespace.clone()),
        )
    }

    fn from_parts(
        tokens: &[u32],
        mm_infos: Option<&[Option<BlockExtraInfo>]>,
        block_size: u32,
        is_eagle: bool,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
    ) -> Self {
        let stride = block_size as usize;
        let complete_blocks = if stride == 0 {
            0
        } else if is_eagle {
            tokens.len().saturating_sub(1) / stride
        } else {
            tokens.len() / stride
        };
        let tail_start = complete_blocks.saturating_mul(stride).min(tokens.len());
        let template = OutputHashBranch {
            tail: tokens[tail_start..].to_vec(),
            parent_hash: None,
            next_position: complete_blocks,
            first_mm_info: mm_infos
                .and_then(|infos| infos.get(complete_blocks))
                .cloned()
                .flatten(),
        };
        let reported_private_blocks = usize::from(!template.tail.is_empty());
        Self {
            template,
            branches: HashMap::new(),
            block_size,
            lora_name,
            cache_namespace,
            is_eagle,
            reported_private_blocks,
        }
    }

    fn initial_private_blocks(&self) -> usize {
        usize::from(!self.template.tail.is_empty())
    }

    fn set_prompt_parent(&mut self, parent_hash: Option<u64>) {
        self.template.parent_hash = parent_hash;
    }

    fn observe(&mut self, index: u32, token_ids: &[u32]) -> Option<MaterializedOutputBlocks> {
        if token_ids.is_empty() || self.block_size == 0 {
            return None;
        }

        let stride = self.block_size as usize;
        let window_size = if self.is_eagle { stride + 1 } else { stride };
        let branch = self
            .branches
            .entry(index)
            .or_insert_with(|| self.template.clone());
        branch.tail.extend_from_slice(token_ids);

        let parent_hash = branch.parent_hash;
        let start_position = branch.next_position;
        let mut blocks = Vec::new();
        let mut consumed = 0;
        while branch.tail.len().saturating_sub(consumed) >= window_size {
            let mm_info = branch.first_mm_info.clone().map(Some);
            let mm_infos = mm_info.as_ref().map(std::slice::from_ref);
            let local_hash = compute_block_hash_for_seq(
                &branch.tail[consumed..consumed + window_size],
                self.block_size,
                BlockHashOptions {
                    block_mm_infos: mm_infos.as_deref(),
                    lora_name: self.lora_name.as_deref(),
                    cache_namespace: self.cache_namespace.as_deref(),
                    is_eagle: Some(self.is_eagle),
                },
            )
            .into_iter()
            .next()
            .expect("a complete canonical block must produce one hash");
            let sequence_hash = branch.parent_hash.map_or(local_hash.0, |parent| {
                compute_next_seq_hash(parent, local_hash)
            });
            blocks.push(ApproximateLruBlock {
                local_hash,
                sequence_hash,
            });
            branch.parent_hash = Some(sequence_hash);
            branch.next_position += 1;
            consumed += stride;
            branch.first_mm_info = None;
        }
        if consumed > 0 {
            branch.tail.drain(..consumed);
        }

        let private_blocks = self
            .branches
            .values()
            .filter(|branch| !branch.tail.is_empty())
            .count();
        if blocks.is_empty() && private_blocks == self.reported_private_blocks {
            return None;
        }
        self.reported_private_blocks = private_blocks;
        Some(MaterializedOutputBlocks {
            parent_hash,
            blocks,
            start_position,
            private_blocks,
        })
    }
}

const ATTEMPT_ACTIVE: u8 = 0;
const ATTEMPT_COMPLETING: u8 = 1;
const ATTEMPT_COMPLETE: u8 = 2;

struct RequestAttemptLeaseInner {
    state: AtomicU8,
    scheduler: SchedulerBookingCleanup,
    booking: SchedulerBookingDescriptor,
    approximate_lru: Option<ApproximateRequestLease>,
    completion: Notify,
}

struct RequestAttemptCompletion<'a> {
    attempt: &'a RequestAttemptLeaseInner,
}

impl Drop for RequestAttemptCompletion<'_> {
    fn drop(&mut self) {
        self.attempt
            .state
            .store(ATTEMPT_COMPLETE, Ordering::Release);
        self.attempt.completion.notify_waiters();
    }
}

impl RequestAttemptLeaseInner {
    fn enqueue_cleanup(&self) {
        if self
            .state
            .compare_exchange(
                ATTEMPT_ACTIVE,
                ATTEMPT_COMPLETING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }

        self.scheduler.enqueue(self.booking.clone());
        if let Some(lease) = &self.approximate_lru {
            lease.release_now();
        }
        self.state.store(ATTEMPT_COMPLETE, Ordering::Release);
        self.completion.notify_waiters();
    }

    async fn complete(&self) {
        if self
            .state
            .compare_exchange(
                ATTEMPT_ACTIVE,
                ATTEMPT_COMPLETING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            loop {
                let notified = self.completion.notified();
                if self.state.load(Ordering::Acquire) != ATTEMPT_COMPLETING {
                    break;
                }
                notified.await;
            }
            return;
        }
        let _completion = RequestAttemptCompletion { attempt: self };

        // Both commands are synchronously enqueued before the first await. If the
        // finishing future is cancelled, cleanup still converges in each lane.
        let scheduler_ack = self.scheduler.enqueue_acknowledged(self.booking.clone());
        let lru_ack = self
            .approximate_lru
            .as_ref()
            .map(ApproximateRequestLease::begin_finish)
            .transpose();

        let scheduler_result = scheduler_ack.wait().await;
        if let Err(error) = scheduler_result {
            tracing::warn!(
                request_id = %self.booking.request_id,
                worker = ?self.booking.worker,
                attempt_id = %self.booking.attempt_id,
                %error,
                "Failed to release scheduler booking"
            );
        }
        match lru_ack {
            Ok(Some(Some(ack))) => {
                if let Err(error) = ack.wait().await {
                    tracing::warn!(
                        request_id = %self.booking.request_id,
                        worker = ?self.booking.worker,
                        %error,
                        "Failed to release approximate LRU request lease"
                    );
                }
            }
            Ok(Some(None)) | Ok(None) => {}
            Err(error) => tracing::warn!(
                request_id = %self.booking.request_id,
                worker = ?self.booking.worker,
                %error,
                "Failed to enqueue approximate LRU request release"
            ),
        }
    }

    fn is_active(&self) -> bool {
        self.state.load(Ordering::Acquire) == ATTEMPT_ACTIVE
    }
}

struct RequestAttemptLease {
    inner: Arc<RequestAttemptLeaseInner>,
    progress: watch::Sender<Instant>,
    idle_timeout: Duration,
}

impl RequestAttemptLease {
    fn new(
        scheduler: SchedulerBookingCleanup,
        booking: SchedulerBookingDescriptor,
        approximate_lru: Option<ApproximateRequestLease>,
    ) -> Self {
        let idle_timeout = active_request_expiry_duration();
        let deadline = Instant::now() + idle_timeout;
        let (progress, receiver) = watch::channel(deadline);
        let inner = Arc::new(RequestAttemptLeaseInner {
            state: AtomicU8::new(ATTEMPT_ACTIVE),
            scheduler,
            booking,
            approximate_lru,
            completion: Notify::new(),
        });
        tokio::spawn(expire_request_attempt(Arc::downgrade(&inner), receiver));
        Self {
            inner,
            progress,
            idle_timeout,
        }
    }

    fn refresh(&self) {
        if self.inner.is_active() {
            self.progress
                .send_replace(Instant::now() + self.idle_timeout);
        }
    }

    fn is_active(&self) -> bool {
        self.inner.is_active()
    }

    async fn finish(&self) {
        self.inner.complete().await;
    }
}

impl Drop for RequestAttemptLease {
    fn drop(&mut self) {
        self.inner.enqueue_cleanup();
    }
}

async fn expire_request_attempt(
    attempt: Weak<RequestAttemptLeaseInner>,
    mut progress: watch::Receiver<Instant>,
) {
    loop {
        let deadline = *progress.borrow_and_update();
        tokio::select! {
            _ = tokio::time::sleep_until(deadline) => {
                let Some(attempt) = attempt.upgrade() else {
                    return;
                };
                attempt.complete().await;
                return;
            }
            changed = progress.changed() => {
                if changed.is_err() {
                    return;
                }
            }
        }
    }
}

/// Owns request-scoped timing and metrics state.
struct RequestObservability {
    tracker: Option<Arc<RequestTracker>>,
    request_metrics: Arc<RouterRequestMetrics>,
    cumulative_osl: usize,
    metrics_recorded: bool,
    first_token_recorded: bool,
    dispatch_guard: Option<StageGuard>,
    dispatched: bool,
}

impl RequestObservability {
    fn new(
        tracker: Option<Arc<RequestTracker>>,
        request_metrics: Arc<RouterRequestMetrics>,
    ) -> Self {
        Self {
            tracker,
            request_metrics,
            cumulative_osl: 0,
            metrics_recorded: false,
            first_token_recorded: false,
            dispatch_guard: None,
            dispatched: false,
        }
    }

    fn request_metrics(&self) -> &RouterRequestMetrics {
        &self.request_metrics
    }

    fn start_dispatch(&mut self, phase_label: &str) {
        self.dispatch_guard = Some(StageGuard::new(STAGE_DISPATCH, phase_label));
    }

    fn record_prefill_start(&self) {
        if let Some(tracker) = &self.tracker {
            tracker.record_prefill_start();
        }
    }

    fn mark_dispatched(&mut self) {
        self.dispatched = true;
    }

    fn observe_response(&mut self) {
        // Taking the guard ends dispatch latency exactly once; later responses see None.
        self.dispatch_guard.take();
    }

    fn observe_tokens(&mut self, new_tokens: usize) {
        if !self.first_token_recorded && new_tokens > 0 {
            if let Some(tracker) = &self.tracker {
                tracker.record_first_token();
                if tracker.phase() == RequestPhase::Decode {
                    tracker.record_decode_first_token();
                }
                if let Some(ttft) = tracker.ttft_ms() {
                    self.request_metrics
                        .time_to_first_token_seconds
                        .observe(ttft / 1000.0);
                }
            }
            self.first_token_recorded = true;
        }

        self.cumulative_osl += new_tokens;
    }

    fn cumulative_osl(&self) -> usize {
        self.cumulative_osl
    }

    fn observe_output_block_boundary(&self) {
        let Some(tracker) = &self.tracker else {
            return;
        };

        // Refresh finish time at block boundaries so the streaming ITL sample stays current.
        tracker.record_osl(self.cumulative_osl);
        tracker.record_finish();
        if let Some(avg_itl) = tracker.avg_itl_ms() {
            self.request_metrics
                .inter_token_latency_seconds
                .observe(avg_itl / 1000.0);
        }
    }

    fn record_metrics(&mut self) {
        // A failed dispatch never reached the backend and must not count as a request.
        if self.metrics_recorded || !self.dispatched {
            return;
        }
        self.metrics_recorded = true;

        if let Some(tracker) = &self.tracker {
            tracker.record_finish();
            tracker.record_osl(self.cumulative_osl);
            if let Some(latency) = tracker.kv_transfer_estimated_latency_secs() {
                self.request_metrics
                    .kv_transfer_estimated_latency_seconds
                    .observe(latency);
            }
        }
        if self.cumulative_osl > 0 {
            self.request_metrics
                .output_sequence_tokens
                .observe(self.cumulative_osl as f64);
        }
        self.request_metrics.requests_total.inc();
    }
}

struct OutputBlockUpdate {
    decay_fraction: Option<f64>,
}

/// Tracks when streamed output grows into a new scheduler accounting block.
struct OutputBlockTracker {
    track_output_blocks: bool,
    current_total_blocks: usize,
    isl_tokens: usize,
    block_size: usize,
    expected_output_tokens: Option<u32>,
}

impl OutputBlockTracker {
    fn new(
        track_output_blocks: bool,
        isl_tokens: usize,
        block_size: usize,
        expected_output_tokens: Option<u32>,
    ) -> Self {
        Self {
            track_output_blocks,
            current_total_blocks: isl_tokens.div_ceil(block_size),
            isl_tokens,
            block_size,
            expected_output_tokens,
        }
    }

    fn observe(&mut self, cumulative_osl: usize) -> Option<OutputBlockUpdate> {
        if !self.track_output_blocks {
            return None;
        }

        let new_total_blocks = (self.isl_tokens + cumulative_osl).div_ceil(self.block_size);
        if new_total_blocks <= self.current_total_blocks {
            return None;
        }

        // Advance before returning so a failed scheduler update preserves existing no-retry behavior.
        self.current_total_blocks = new_total_blocks;
        let decay_fraction = self
            .expected_output_tokens
            .map(|expected| (1.0 - cumulative_osl as f64 / expected.max(1) as f64).max(0.0));
        Some(OutputBlockUpdate { decay_fraction })
    }
}

/// Coordinates scheduler cleanup, observability, and streamed load tracking.
///
/// Session-affinity lifetime is separate: `AffinityAcquire` and
/// `AffinityLease` own binding commit, release, and invalidation.
pub(super) struct RequestGuard<Sel = DefaultWorkerSelector>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    chooser: Arc<KvRouter<Sel>>,
    context_id: String,
    worker: WorkerWithDpRank,
    lifecycle: Option<RequestAttemptLease>,
    observability: RequestObservability,
    output_blocks: OutputBlockTracker,
    approximate_lru: Option<ApproximateRequestLease>,
    approximate_lru_capacity: Option<usize>,
    output_hashes: Option<CanonicalOutputTracker>,
    prefill_marked: bool,
    migration_state: Option<MigrationState>,
}

impl<Sel> RequestGuard<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub(super) fn new(
        chooser: Arc<KvRouter<Sel>>,
        request_metrics: Arc<RouterRequestMetrics>,
        context_id: String,
        worker: WorkerWithDpRank,
        attempt_id: Option<AttemptId>,
        request: &PreprocessedRequest,
        scheduler_tracked: bool,
    ) -> Self {
        // Snapshot request-scoped inputs now so the guard can outlive the
        // PreprocessedRequest after it is moved into backend dispatch.
        let block_size = chooser.block_size() as usize;
        let isl_tokens = request.token_ids.len();
        let expected_output_tokens = request
            .routing
            .as_ref()
            .and_then(|routing| routing.expected_output_tokens);
        let track_output_blocks =
            scheduler_tracked && chooser.kv_router_config().router_track_output_blocks;
        if scheduler_tracked {
            request_metrics.requests_started_total().inc();
        }
        let attempt_id = scheduler_tracked
            .then(|| attempt_id.expect("scheduler-tracked selection must carry an attempt ID"));
        let lru_registration = scheduler_tracked
            .then(|| chooser.approximate_lru_rank_registration(worker))
            .flatten();
        let approximate_lru = lru_registration.and_then(|registration| {
            chooser.indexer().begin_approximate_lru_request(
                worker,
                registration.incarnation,
                attempt_id.expect("LRU registration requires an admitted attempt"),
            )
        });
        let output_hashes = approximate_lru
            .as_ref()
            .map(|_| CanonicalOutputTracker::new(request, block_size as u32, chooser.is_eagle()));
        let lifecycle = scheduler_tracked.then(|| {
            let attempt_id = attempt_id.expect("tracked request requires an admitted attempt");
            RequestAttemptLease::new(
                chooser.scheduler_booking_cleanup(),
                SchedulerBookingDescriptor {
                    request_id: context_id.clone(),
                    worker,
                    attempt_id,
                },
                approximate_lru.clone(),
            )
        });

        Self {
            chooser,
            context_id,
            worker,
            lifecycle,
            observability: RequestObservability::new(request.tracker.clone(), request_metrics),
            output_blocks: OutputBlockTracker::new(
                track_output_blocks,
                isl_tokens,
                block_size,
                expected_output_tokens,
            ),
            approximate_lru,
            approximate_lru_capacity: lru_registration
                .and_then(|registration| registration.capacity),
            output_hashes,
            prefill_marked: false,
            migration_state: request.migration_state.clone(),
        }
    }

    pub(super) fn record_migration_failure(&self, error: Option<DynamoError>) {
        if let Some(state) = self.migration_state.as_ref() {
            state.record_failure(self.worker.worker_id, error);
        }
    }

    pub(super) fn request_metrics(&self) -> &RouterRequestMetrics {
        self.observability.request_metrics()
    }

    pub(super) fn start_dispatch(&mut self, phase_label: &str) {
        self.observability.start_dispatch(phase_label);
    }

    pub(super) fn record_prefill_start(&self) {
        self.observability.record_prefill_start();
    }

    pub(super) fn mark_dispatched(&mut self) {
        self.observability.mark_dispatched();
    }

    pub(super) fn has_approximate_lru(&self) -> bool {
        self.approximate_lru.is_some()
    }

    pub(super) async fn acquire_approximate_lru(
        &mut self,
        hashes: RoutingDecisionHashes,
    ) -> Result<(), dynamo_kv_router::indexer::KvRouterError> {
        let parent_hash = hashes.sequence_hashes.last().copied();
        let private_blocks = self
            .output_hashes
            .as_ref()
            .map_or(0, CanonicalOutputTracker::initial_private_blocks);
        let Some(lease) = self.approximate_lru.as_mut() else {
            return Ok(());
        };
        let mode = lease
            .acquire(hashes, private_blocks, self.approximate_lru_capacity)
            .await?;
        if mode == ApproximateAcquireMode::TtlFallback {
            self.output_hashes = None;
            return Ok(());
        }
        if let Some(output_hashes) = self.output_hashes.as_mut() {
            output_hashes.set_prompt_parent(parent_hash);
        }
        Ok(())
    }

    pub(super) async fn on_item(&mut self, item: &Annotated<LLMEngineOutput>) {
        self.observability.observe_response();

        let new_tokens = item.data.as_ref().map_or(0, |data| data.token_ids.len());
        if new_tokens > 0
            && let Some(lifecycle) = &self.lifecycle
        {
            lifecycle.refresh();
        }

        if !self.prefill_marked {
            let has_tokens = item
                .data
                .as_ref()
                .is_some_and(|data| !data.token_ids.is_empty());
            if has_tokens {
                if let Some(lifecycle) = &self.lifecycle
                    && lifecycle.is_active()
                    && let Err(error) = self
                        .chooser
                        .mark_prefill_completed_if_booking(&lifecycle.inner.booking)
                        .await
                {
                    tracing::warn!(
                        request_id = %self.context_id,
                        %error,
                        "Failed to mark prefill completed"
                    );
                }
                self.prefill_marked = true;
            }
        }

        if self
            .lifecycle
            .as_ref()
            .is_some_and(RequestAttemptLease::is_active)
            && let (Some(data), Some(output_hashes), Some(lease)) = (
                item.data.as_ref(),
                self.output_hashes.as_mut(),
                self.approximate_lru.as_ref(),
            )
            && let Some(materialized) =
                output_hashes.observe(data.index.unwrap_or(0), &data.token_ids)
            && let Err(error) = lease.materialize(
                materialized.parent_hash,
                materialized.blocks,
                materialized.start_position,
                materialized.private_blocks,
            )
        {
            tracing::warn!(
                request_id = %self.context_id,
                %error,
                "Failed to materialize approximate LRU output blocks"
            );
        }
        self.observability.observe_tokens(new_tokens);
        let cumulative_osl = self.observability.cumulative_osl();
        let Some(update) = self.output_blocks.observe(cumulative_osl) else {
            return;
        };

        let Some(lifecycle) = &self.lifecycle else {
            return;
        };
        if !lifecycle.is_active() {
            return;
        }
        if let Err(error) = self
            .chooser
            .add_output_block_if_booking(&lifecycle.inner.booking, update.decay_fraction)
            .await
        {
            tracing::warn!(
                request_id = %self.context_id,
                %error,
                "Failed to add output block"
            );
        }

        self.observability.observe_output_block_boundary();
    }

    pub(super) async fn finish(&mut self) {
        // Metrics must observe the completed request before cleanup releases its state.
        self.observability.record_metrics();
        self.finish_lifecycle().await;
    }

    pub(super) async fn abort(&mut self) {
        self.finish_lifecycle().await;
    }

    async fn finish_lifecycle(&self) {
        let Some(lifecycle) = &self.lifecycle else {
            return;
        };
        lifecycle.finish().await;
    }
}

impl<Sel> Drop for RequestGuard<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn drop(&mut self) {
        self.observability.record_metrics();
    }
}

#[cfg(test)]
mod output_hash_tests {
    use super::*;
    use dynamo_kv_router::protocols::{BlockMmObjectInfo, compute_seq_hash_for_block};

    fn direct_blocks(
        tokens: &[u32],
        block_size: u32,
        mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<&str>,
        cache_namespace: Option<&str>,
        is_eagle: bool,
    ) -> Vec<ApproximateLruBlock> {
        let local_hashes = compute_block_hash_for_seq(
            tokens,
            block_size,
            BlockHashOptions {
                block_mm_infos: mm_infos,
                lora_name,
                cache_namespace,
                is_eagle: Some(is_eagle),
            },
        );
        let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
        local_hashes
            .into_iter()
            .zip(sequence_hashes)
            .map(|(local_hash, sequence_hash)| ApproximateLruBlock {
                local_hash,
                sequence_hash,
            })
            .collect()
    }

    #[test]
    fn streamed_chunks_complete_prompt_tail_and_extend_canonical_chain() {
        let prompt = vec![1, 2, 3];
        let mut tracker = CanonicalOutputTracker::from_parts(&prompt, None, 4, false, None, None);
        tracker.set_prompt_parent(None);

        let first = tracker.observe(0, &[4, 5]).unwrap();
        assert_eq!(first.start_position, 0);
        assert_eq!(first.private_blocks, 1);
        let second = tracker.observe(0, &[6, 7, 8]).unwrap();
        assert_eq!(second.start_position, 1);
        assert_eq!(second.private_blocks, 0);

        let expected = direct_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None, None, None, false);
        assert_eq!(
            first
                .blocks
                .into_iter()
                .chain(second.blocks)
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn incomplete_output_tail_is_not_materialized() {
        let mut tracker = CanonicalOutputTracker::from_parts(&[1], None, 4, false, None, None);
        assert!(tracker.observe(0, &[2, 3]).is_none());
        assert_eq!(tracker.initial_private_blocks(), 1);
    }

    #[test]
    fn aligned_prompt_reports_partial_output_as_private_occupancy() {
        let prompt = [1, 2, 3, 4];
        let prompt_block = direct_blocks(&prompt, 4, None, None, None, false);
        let mut tracker = CanonicalOutputTracker::from_parts(&prompt, None, 4, false, None, None);
        tracker.set_prompt_parent(Some(prompt_block[0].sequence_hash));

        let partial = tracker.observe(0, &[5]).unwrap();
        assert!(partial.blocks.is_empty());
        assert_eq!(partial.private_blocks, 1);

        let completed = tracker.observe(0, &[6, 7, 8]).unwrap();
        assert_eq!(completed.blocks.len(), 1);
        assert_eq!(completed.private_blocks, 0);
    }

    #[test]
    fn multiple_choice_streams_keep_independent_hash_tails() {
        let prompt = [1, 2, 3, 4];
        let prompt_block = direct_blocks(&prompt, 4, None, None, None, false);
        let mut tracker = CanonicalOutputTracker::from_parts(&prompt, None, 4, false, None, None);
        tracker.set_prompt_parent(Some(prompt_block[0].sequence_hash));

        let choice_zero = tracker.observe(0, &[5, 6, 7, 8]).unwrap();
        let choice_one = tracker.observe(1, &[9, 10, 11, 12]).unwrap();
        assert_eq!(choice_zero.start_position, 1);
        assert_eq!(choice_one.start_position, 1);
        assert_eq!(
            choice_zero.blocks[0],
            direct_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None, None, None, false)[1]
        );
        assert_eq!(
            choice_one.blocks[0],
            direct_blocks(&[1, 2, 3, 4, 9, 10, 11, 12], 4, None, None, None, false)[1]
        );
    }

    #[test]
    fn eagle_lora_namespace_and_multimodal_hashing_matches_canonical_path() {
        let prompt = vec![10, 11, 12];
        let mm_infos = vec![Some(BlockExtraInfo {
            mm_objects: vec![BlockMmObjectInfo {
                mm_hash: 42,
                offsets: vec![(0, 2)],
            }],
        })];
        let mut tracker = CanonicalOutputTracker::from_parts(
            &prompt,
            Some(&mm_infos),
            4,
            true,
            Some("adapter-a".to_string()),
            Some("tenant-a".to_string()),
        );

        let first = tracker.observe(0, &[13, 14]).unwrap();
        let second = tracker.observe(0, &[15, 16, 17, 18]).unwrap();
        let expected = direct_blocks(
            &[10, 11, 12, 13, 14, 15, 16, 17, 18],
            4,
            Some(&[mm_infos[0].clone(), None]),
            Some("adapter-a"),
            Some("tenant-a"),
            true,
        );
        assert_eq!(
            first
                .blocks
                .into_iter()
                .chain(second.blocks)
                .collect::<Vec<_>>(),
            expected
        );
    }
}
