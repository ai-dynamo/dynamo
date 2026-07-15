// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::scheduling::{
    AdmissionLease, AdmissionToolSummary, RequestOutcome, RequestProgressUpdater,
};
use dynamo_runtime::{
    metrics::frontend_perf::{STAGE_DISPATCH, StageGuard},
    protocols::annotated::Annotated,
};

use crate::{
    kv_router::{
        KvRouter, admission_completion::SharedAdmissionToolSummary, metrics::RouterRequestMetrics,
    },
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RequestTracker},
    },
};

/// Owns scheduler cleanup after a worker is selected.
///
/// `KvPushRouter` installs this through [`RequestGuard`] immediately after
/// selection and before backend dispatch. The admission lease moves directly
/// from the scheduling response into this guard and remains responsible for
/// cleanup until the request ends.
struct RequestCleanup {
    chooser: Arc<KvRouter>,
    context_id: String,
    scheduler_tracked: bool,
    request_progress: Option<RequestProgressUpdater>,
    admission_lease: Option<AdmissionLease>,
    freed: bool,
}

impl RequestCleanup {
    fn new(
        chooser: Arc<KvRouter>,
        context_id: String,
        scheduler_tracked: bool,
        request_progress: Option<RequestProgressUpdater>,
        admission_lease: Option<AdmissionLease>,
    ) -> Self {
        debug_assert!(request_progress.is_none() || scheduler_tracked);
        debug_assert_eq!(request_progress.is_some(), admission_lease.is_some());
        Self {
            chooser,
            context_id,
            scheduler_tracked,
            request_progress,
            admission_lease,
            freed: false,
        }
    }

    async fn finish(&mut self, outcome: RequestOutcome) {
        let result = if let Some(mut lease) = self.admission_lease.take() {
            match outcome {
                RequestOutcome::Completed {
                    context_tokens,
                    tool_summary,
                } => lease.mark_completed_with_tool_summary(context_tokens, tool_summary),
                RequestOutcome::Aborted => lease.mark_aborted(),
            }
            // AdmissionLease reports the terminal outcome to the scheduler actor.
            // The scheduler actor remains the sole owner of booking and queue cleanup.
            drop(lease);
            Ok(())
        } else if !self.scheduler_tracked {
            Ok(())
        } else {
            self.chooser.free(&self.context_id).await
        };
        if let Err(error) = result {
            tracing::warn!(
                request_id = %self.context_id,
                %error,
                "Failed to free request"
            );
        }
        self.freed = true;
    }

    fn mark_completed(&mut self, context_tokens: usize) {
        if let Some(progress) = &self.request_progress {
            progress.update_context_tokens(context_tokens);
        }
        if let Some(lease) = self.admission_lease.as_mut() {
            lease.mark_completed(context_tokens);
        }
    }

    fn mark_dispatched(&mut self) {
        if let Some(lease) = self.admission_lease.as_mut() {
            lease.mark_dispatched();
        }
    }
}

impl Drop for RequestCleanup {
    fn drop(&mut self) {
        let needs_free = !self.freed && self.scheduler_tracked;
        if !needs_free {
            return;
        }

        // AdmissionLease drops after this method and performs actor-owned cleanup.
        if self.admission_lease.is_some() {
            return;
        }

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                request_id = %self.context_id,
                "No tokio runtime for request cleanup"
            );
            return;
        };

        let chooser = self.chooser.clone();
        let context_id = self.context_id.clone();
        handle.spawn(async move {
            let result = chooser.free(&context_id).await;
            if let Err(error) = result {
                tracing::warn!(
                    request_id = %context_id,
                    %error,
                    "Failed to free request from drop guard"
                );
            }
        });
    }
}

/// Owns request-scoped timing and metrics state.
struct RequestObservability {
    tracker: Option<Arc<RequestTracker>>,
    request_metrics: Arc<RouterRequestMetrics>,
    cumulative_osl: usize,
    authoritative_context_tokens: Option<usize>,
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
            authoritative_context_tokens: None,
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

    fn observe_context_tokens(&mut self, context_tokens: usize) {
        self.authoritative_context_tokens = Some(
            self.authoritative_context_tokens
                .map_or(context_tokens, |observed| observed.max(context_tokens)),
        );
    }

    fn context_tokens(&self, initial_context_tokens: usize) -> usize {
        self.authoritative_context_tokens
            .unwrap_or_else(|| initial_context_tokens.saturating_add(self.cumulative_osl))
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
    update_scheduler: bool,
}

/// Tracks when streamed output grows into a new scheduler accounting block.
struct OutputBlockTracker {
    track_output_blocks: bool,
    track_request_progress: bool,
    current_total_blocks: usize,
    isl_tokens: usize,
    block_size: usize,
    expected_output_tokens: Option<u32>,
}

impl OutputBlockTracker {
    fn new(
        track_output_blocks: bool,
        track_request_progress: bool,
        isl_tokens: usize,
        block_size: usize,
        expected_output_tokens: Option<u32>,
    ) -> Self {
        Self {
            track_output_blocks,
            track_request_progress,
            current_total_blocks: isl_tokens.div_ceil(block_size),
            isl_tokens,
            block_size,
            expected_output_tokens,
        }
    }

    fn observe(&mut self, cumulative_osl: usize) -> Option<OutputBlockUpdate> {
        if !self.track_output_blocks && !self.track_request_progress {
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
        Some(OutputBlockUpdate {
            decay_fraction,
            update_scheduler: self.track_output_blocks,
        })
    }
}

/// Coordinates scheduler cleanup, observability, and streamed load tracking.
///
/// Session-affinity lifetime is separate: `AffinityAcquire` and
/// `AffinityLease` own binding commit, release, and invalidation.
pub(super) struct RequestGuard {
    cleanup: RequestCleanup,
    observability: RequestObservability,
    output_blocks: OutputBlockTracker,
    admission_tool_summary: Option<SharedAdmissionToolSummary>,
    initial_context_tokens: usize,
    prefill_marked: bool,
}

impl RequestGuard {
    pub(super) fn new(
        chooser: Arc<KvRouter>,
        request_metrics: Arc<RouterRequestMetrics>,
        context_id: String,
        request: &PreprocessedRequest,
        scheduler_tracked: bool,
        request_progress: Option<RequestProgressUpdater>,
        admission_lease: Option<AdmissionLease>,
        admission_tool_summary: Option<SharedAdmissionToolSummary>,
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
        let track_request_progress = request_progress.is_some();
        if scheduler_tracked {
            request_metrics.requests_started_total().inc();
        }

        Self {
            cleanup: RequestCleanup::new(
                chooser,
                context_id,
                scheduler_tracked,
                request_progress,
                admission_lease,
            ),
            observability: RequestObservability::new(request.tracker.clone(), request_metrics),
            output_blocks: OutputBlockTracker::new(
                track_output_blocks,
                track_request_progress,
                isl_tokens,
                block_size,
                expected_output_tokens,
            ),
            admission_tool_summary,
            initial_context_tokens: isl_tokens,
            prefill_marked: false,
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

    pub(super) async fn mark_dispatched(&mut self) {
        if self.cleanup.request_progress.is_some() {
            // Backend dispatch already succeeded. Record that fact synchronously so
            // lease cleanup still reports Dispatched before the terminal event if it
            // overtakes the actor command.
            self.cleanup.mark_dispatched();
            self.cleanup
                .chooser
                .mark_dispatched(&self.cleanup.context_id)
                .await;
        }
        self.observability.mark_dispatched();
    }

    pub(super) async fn on_item(&mut self, item: &Annotated<LLMEngineOutput>) {
        self.observability.observe_response();

        if let Some(usage) = item
            .data
            .as_ref()
            .and_then(|data| data.completion_usage.as_ref())
        {
            self.observability
                .observe_context_tokens(usage.total_tokens as usize);
        }

        if !self.prefill_marked {
            let has_tokens = item
                .data
                .as_ref()
                .is_some_and(|data| !data.token_ids.is_empty());
            if has_tokens {
                if self.cleanup.scheduler_tracked
                    && let Err(error) = self
                        .cleanup
                        .chooser
                        .mark_prefill_completed(&self.cleanup.context_id)
                        .await
                {
                    tracing::warn!(
                        request_id = %self.cleanup.context_id,
                        %error,
                        "Failed to mark prefill completed"
                    );
                }
                self.prefill_marked = true;
            }
        }

        let new_tokens = item.data.as_ref().map_or(0, |data| data.token_ids.len());
        self.observability.observe_tokens(new_tokens);
        let cumulative_osl = self.observability.cumulative_osl();
        let Some(update) = self.output_blocks.observe(cumulative_osl) else {
            return;
        };

        if let Some(progress) = &self.cleanup.request_progress {
            progress
                .update_context_tokens(self.initial_context_tokens.saturating_add(cumulative_osl));
        }
        if !update.update_scheduler {
            return;
        }

        if let Err(error) = self
            .cleanup
            .chooser
            .add_output_block(&self.cleanup.context_id, update.decay_fraction)
        {
            tracing::warn!(
                request_id = %self.cleanup.context_id,
                %error,
                "Failed to add output block"
            );
        }

        self.observability.observe_output_block_boundary();
    }

    pub(super) async fn finish(&mut self) {
        // Metrics must observe the completed request before cleanup releases its state.
        self.observability.record_metrics();
        self.mark_completed_terminal();
        let context_tokens = self
            .observability
            .context_tokens(self.initial_context_tokens);
        let tool_summary = self.admission_tool_summary.as_ref().map_or_else(
            AdmissionToolSummary::default,
            SharedAdmissionToolSummary::snapshot,
        );
        self.cleanup
            .finish(RequestOutcome::Completed {
                context_tokens,
                tool_summary,
            })
            .await;
    }

    pub(super) fn mark_completed_terminal(&mut self) {
        let context_tokens = self
            .observability
            .context_tokens(self.initial_context_tokens);
        self.cleanup.mark_completed(context_tokens);
    }

    pub(super) async fn abort(&mut self) {
        self.cleanup.finish(RequestOutcome::Aborted).await;
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        // RequestCleanup drops immediately afterward and performs resource cleanup.
        self.observability.record_metrics();
    }
}
