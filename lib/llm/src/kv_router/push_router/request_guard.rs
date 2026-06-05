// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::{
    metrics::frontend_perf::{STAGE_DISPATCH, StageGuard},
    protocols::annotated::Annotated,
};

use crate::{
    kv_router::{KvRouter, metrics::RouterRequestMetrics, sticky::lifecycle::SessionCloseAction},
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RequestTracker},
    },
};

/// Owns resources that must be released even when routing setup or streaming is cancelled.
struct RequestCleanup {
    chooser: Arc<KvRouter>,
    context_id: String,
    scheduler_tracked: bool,
    freed: bool,
    deferred_close: Option<SessionCloseAction>,
}

impl RequestCleanup {
    fn new(chooser: Arc<KvRouter>, context_id: String, scheduler_tracked: bool) -> Self {
        Self {
            chooser,
            context_id,
            scheduler_tracked,
            freed: false,
            deferred_close: None,
        }
    }

    fn set_deferred_close(&mut self, deferred_close: Option<SessionCloseAction>) {
        self.deferred_close = deferred_close;
    }

    async fn finish(&mut self) {
        // Free scheduler state before closing the session so both explicit and
        // drop cleanup preserve the same lifecycle ordering.
        if self.scheduler_tracked
            && let Err(error) = self.chooser.free(&self.context_id).await
        {
            tracing::warn!(
                request_id = %self.context_id,
                %error,
                "Failed to free request"
            );
        }
        self.freed = true;

        if let Some(close) = self.deferred_close.take() {
            close.execute(&self.context_id);
        }
    }
}

impl Drop for RequestCleanup {
    fn drop(&mut self) {
        // Drop cannot await, so transfer any unfinished cleanup into one task.
        let deferred_close = self.deferred_close.take();
        let needs_free = !self.freed && self.scheduler_tracked;
        if deferred_close.is_none() && !needs_free {
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
            // Match explicit finish ordering so session KV closes after scheduler cleanup.
            if needs_free && let Err(error) = chooser.free(&context_id).await {
                tracing::warn!(
                    request_id = %context_id,
                    %error,
                    "Failed to free request from drop guard"
                );
            }
            if let Some(close) = deferred_close {
                close.execute(&context_id);
            }
        });
    }
}

/// Tracks routed request metrics and owns cleanup for scheduler and session state.
pub(super) struct RequestGuard {
    cleanup: RequestCleanup,
    tracker: Option<Arc<RequestTracker>>,
    request_metrics: Arc<RouterRequestMetrics>,
    cumulative_osl: usize,
    metrics_recorded: bool,
    prefill_marked: bool,
    first_token_recorded: bool,
    first_response_received: bool,
    dispatch_guard: Option<StageGuard>,
    track_output_blocks: bool,
    current_total_blocks: usize,
    isl_tokens: usize,
    block_size: usize,
    expected_output_tokens: Option<u32>,
    dispatched: bool,
}

impl RequestGuard {
    pub(super) fn new(
        chooser: Arc<KvRouter>,
        context_id: String,
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
        let request_metrics =
            RouterRequestMetrics::from_component(chooser.client().endpoint.component());

        Self {
            cleanup: RequestCleanup::new(chooser, context_id, scheduler_tracked),
            tracker: request.tracker.clone(),
            request_metrics,
            cumulative_osl: 0,
            metrics_recorded: false,
            prefill_marked: false,
            first_token_recorded: false,
            first_response_received: false,
            dispatch_guard: None,
            track_output_blocks,
            current_total_blocks: isl_tokens.div_ceil(block_size),
            isl_tokens,
            block_size,
            expected_output_tokens,
            dispatched: false,
        }
    }

    pub(super) fn request_metrics(&self) -> &RouterRequestMetrics {
        &self.request_metrics
    }

    pub(super) fn start_dispatch(&mut self, phase_label: &str) {
        self.dispatch_guard = Some(StageGuard::new(STAGE_DISPATCH, phase_label));
    }

    pub(super) fn set_deferred_close(&mut self, deferred_close: Option<SessionCloseAction>) {
        self.cleanup.set_deferred_close(deferred_close);
    }

    pub(super) fn record_prefill_start(&self) {
        if let Some(tracker) = &self.tracker {
            tracker.record_prefill_start();
        }
    }

    pub(super) fn mark_dispatched(&mut self) {
        self.dispatched = true;
    }

    pub(super) async fn on_item(&mut self, item: &Annotated<LLMEngineOutput>) {
        // The first backend response ends dispatch latency, even if it carries no tokens.
        if !self.first_response_received {
            self.first_response_received = true;
            self.dispatch_guard.take();
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
        if !self.track_output_blocks {
            return;
        }

        let new_total_blocks = (self.isl_tokens + self.cumulative_osl).div_ceil(self.block_size);
        if new_total_blocks <= self.current_total_blocks {
            return;
        }

        // Update scheduler load only when generation crosses a block boundary.
        let decay_fraction = self
            .expected_output_tokens
            .map(|expected| (1.0 - self.cumulative_osl as f64 / expected.max(1) as f64).max(0.0));
        if let Err(error) = self
            .cleanup
            .chooser
            .add_output_block(&self.cleanup.context_id, decay_fraction)
        {
            tracing::warn!(
                request_id = %self.cleanup.context_id,
                %error,
                "Failed to add output block"
            );
        }

        if let Some(tracker) = &self.tracker {
            tracker.record_osl(self.cumulative_osl);
            tracker.record_finish();
            if let Some(avg_itl) = tracker.avg_itl_ms() {
                self.request_metrics
                    .inter_token_latency_seconds
                    .observe(avg_itl / 1000.0);
            }
        }

        self.current_total_blocks = new_total_blocks;
    }

    pub(super) async fn finish(&mut self) {
        // Metrics must observe the completed request before cleanup releases its state.
        self.record_metrics();
        self.cleanup.finish().await;
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

impl Drop for RequestGuard {
    fn drop(&mut self) {
        // RequestCleanup drops immediately afterward and performs resource cleanup.
        self.record_metrics();
    }
}
