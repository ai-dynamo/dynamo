// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_kv_router::{protocols::WorkerWithDpRank, selector::WorkerSelector};
use dynamo_runtime::{
    error::DynamoError,
    metrics::frontend_perf::{STAGE_DISPATCH, StageGuard},
    protocols::annotated::Annotated,
};

use crate::{
    kv_router::{KvRouter, metrics::RouterRequestMetrics, scheduler::DefaultWorkerSelector},
    local_model::runtime_config::ModelRuntimeConfig,
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        preprocessor::MigrationState,
        timing::{RequestPhase, RequestTracker},
    },
};

/// Owns scheduler cleanup after a worker is selected.
///
/// `worker` is captured at construction so cleanup targets the booking this
/// guard acquired, even if cleanup is delayed.
struct RequestCleanup<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    chooser: Arc<KvRouter<Sel>>,
    context_id: String,
    worker: WorkerWithDpRank,
    scheduler_tracked: bool,
    freed: bool,
}

impl<Sel> RequestCleanup<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn new(
        chooser: Arc<KvRouter<Sel>>,
        context_id: String,
        worker: WorkerWithDpRank,
        scheduler_tracked: bool,
    ) -> Self {
        Self {
            chooser,
            context_id,
            worker,
            scheduler_tracked,
            freed: false,
        }
    }

    async fn release(&mut self, completed_context_tokens: Option<usize>) {
        if self.freed {
            return;
        }
        if self.scheduler_tracked {
            let result = if let Some(context_tokens) = completed_context_tokens {
                self.chooser
                    .complete_if_worker(&self.context_id, self.worker, context_tokens)
                    .await
            } else {
                self.chooser
                    .abort_if_worker(&self.context_id, self.worker)
                    .await
            };
            if let Err(error) = result {
                tracing::warn!(
                    request_id = %self.context_id,
                    worker = ?self.worker,
                    %error,
                    completed = completed_context_tokens.is_some(),
                    "Failed to release request"
                );
            }
        }
        self.freed = true;
    }

    async fn complete(&mut self, context_tokens: usize) {
        self.release(Some(context_tokens)).await;
    }

    async fn abort(&mut self) {
        self.release(None).await;
    }
}

impl<Sel> Drop for RequestCleanup<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn drop(&mut self) {
        if self.freed || !self.scheduler_tracked {
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
        let worker = self.worker;
        handle.spawn(async move {
            let result = chooser.abort_if_worker(&context_id, worker).await;
            if let Err(error) = result {
                tracing::warn!(
                    request_id = %context_id,
                    ?worker,
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

fn routing_isl_tokens(request: &PreprocessedRequest) -> usize {
    request.block_mm_routing_info().0.len()
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

    fn context_tokens(&self, cumulative_osl: usize) -> usize {
        self.isl_tokens.saturating_add(cumulative_osl)
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
    cleanup: RequestCleanup<Sel>,
    observability: RequestObservability,
    output_blocks: OutputBlockTracker,
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
        request: &PreprocessedRequest,
        scheduler_tracked: bool,
    ) -> Self {
        // Snapshot request-scoped inputs now so the guard can outlive the
        // PreprocessedRequest after it is moved into backend dispatch.
        let block_size = chooser.block_size() as usize;
        let isl_tokens = routing_isl_tokens(request);
        let expected_output_tokens = request
            .routing
            .as_ref()
            .and_then(|routing| routing.expected_output_tokens);
        let track_output_blocks =
            scheduler_tracked && chooser.kv_router_config().router_track_output_blocks;
        if scheduler_tracked {
            request_metrics.requests_started_total().inc();
        }

        Self {
            cleanup: RequestCleanup::new(chooser, context_id, worker, scheduler_tracked),
            observability: RequestObservability::new(request.tracker.clone(), request_metrics),
            output_blocks: OutputBlockTracker::new(
                track_output_blocks,
                isl_tokens,
                block_size,
                expected_output_tokens,
            ),
            prefill_marked: false,
            migration_state: request.migration_state.clone(),
        }
    }

    pub(super) fn record_migration_failure(&self, error: Option<DynamoError>) {
        if let Some(state) = self.migration_state.as_ref() {
            state.record_failure(self.cleanup.worker.worker_id, error);
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

    pub(super) async fn on_item(&mut self, item: &Annotated<LLMEngineOutput>) {
        self.observability.observe_response();

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
        let context_tokens = self
            .output_blocks
            .context_tokens(self.observability.cumulative_osl());
        self.cleanup.complete(context_tokens).await;
    }

    pub(super) async fn abort(&mut self) {
        self.cleanup.abort().await;
    }
}

impl<Sel> Drop for RequestGuard<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    fn drop(&mut self) {
        // RequestCleanup drops immediately afterward and performs resource cleanup.
        self.observability.record_metrics();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::preprocessor::MmRoutingInfo;

    #[test]
    fn terminal_context_uses_multimodal_routing_length() {
        let request = PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1])
            .mm_routing_info(Some(MmRoutingInfo {
                routing_token_ids: vec![1, 2, 3, 4],
                block_mm_infos: Vec::new(),
                expanded_prompt_len: 4,
            }))
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap();

        assert_eq!(routing_isl_tokens(&request), 4);
    }
}
