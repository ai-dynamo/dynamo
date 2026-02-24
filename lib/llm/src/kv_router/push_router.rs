// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_runtime::{
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
};
use futures::stream::{self, StreamExt};
use serde_json::json;
use tracing::Instrument;

use crate::{
    kv_router::{
        KvRouter,
        metrics::RouterRequestMetrics,
        protocols::{TokensWithHashes, WorkerWithDpRank},
    },
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        timing::{RequestPhase, RequestTracker},
    },
};

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
}

/// Result of worker selection containing instance ID, dp_rank, and overlap amount.
struct WorkerSelection {
    instance_id: u64,
    dp_rank: u32,
    overlap_amount: u32,
}

/// Drop guard that manages the full lifecycle of a routed request:
/// per-item tracking (prefill, first token, output blocks) and final cleanup (free + metrics).
///
/// In the happy path, `finish().await` runs cleanup inline in the async context.
/// If the stream is dropped early (e.g., client disconnect, consumer drop), the
/// `Drop` impl fires and spawns a task to call `free()`.
struct RequestGuard {
    chooser: Arc<KvRouter>,
    context_id: String,
    tracker: Option<Arc<RequestTracker>>,
    request_metrics: Arc<RouterRequestMetrics>,
    cumulative_osl: usize,
    metrics_recorded: bool,
    freed: bool,
    prefill_marked: bool,
    first_token_recorded: bool,
    track_output_blocks: bool,
    current_total_blocks: usize,
    isl_tokens: usize,
    block_size: usize,
    expected_output_tokens: Option<u32>,
}

impl RequestGuard {
    async fn on_item(&mut self, item: &Annotated<LLMEngineOutput>) {
        if !self.prefill_marked {
            let has_tokens = item
                .data
                .as_ref()
                .map(|d| !d.token_ids.is_empty())
                .unwrap_or(false);
            if has_tokens {
                if let Err(e) = self.chooser.mark_prefill_completed(&self.context_id).await {
                    tracing::warn!(
                        "Failed to mark prefill completed for request {}: {e}",
                        self.context_id
                    );
                }
                self.prefill_marked = true;
            }
        }

        let new_tokens = item.data.as_ref().map(|d| d.token_ids.len()).unwrap_or(0);

        if !self.first_token_recorded && new_tokens > 0 {
            if let Some(ref tracker) = self.tracker {
                tracker.record_first_token();
                if let Some(ttft) = tracker.ttft_ms() {
                    self.request_metrics
                        .time_to_first_token_seconds
                        .observe(ttft / 1000.0);
                }
            }
            self.first_token_recorded = true;
        }

        self.cumulative_osl += new_tokens;

        if self.track_output_blocks {
            let new_total_blocks =
                (self.isl_tokens + self.cumulative_osl).div_ceil(self.block_size);
            if new_total_blocks > self.current_total_blocks {
                let decay_fraction = self
                    .expected_output_tokens
                    .map(|eot| (1.0 - (self.cumulative_osl as f64 / eot.max(1) as f64)).max(0.0));
                if let Err(e) = self
                    .chooser
                    .add_output_block(&self.context_id, decay_fraction)
                    .await
                {
                    tracing::warn!(
                        "Failed to add output block for request {}: {e}",
                        self.context_id
                    );
                }

                if let Some(ref tracker) = self.tracker {
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
        }
    }

    async fn finish(&mut self) {
        self.record_metrics();
        if let Err(e) = self.chooser.free(&self.context_id).await {
            tracing::warn!("Failed to free request {}: {e}", self.context_id);
        }
        self.freed = true;
    }

    fn record_metrics(&mut self) {
        if self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        if let Some(ref tracker) = self.tracker {
            tracker.record_finish();
            tracker.record_osl(self.cumulative_osl);
            self.request_metrics
                .output_sequence_tokens
                .observe(self.cumulative_osl as f64);
        }
        self.request_metrics.requests_total.inc();
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.record_metrics();
        if !self.freed {
            let chooser = self.chooser.clone();
            let context_id = self.context_id.clone();
            let Ok(handle) = tokio::runtime::Handle::try_current() else {
                tracing::warn!("No tokio runtime for drop guard free of request {context_id}");
                return;
            };
            handle.spawn(async move {
                if let Err(e) = chooser.free(&context_id).await {
                    tracing::warn!("Failed to free request {context_id} (drop guard): {e}");
                }
            });
        }
    }
}

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
    ) -> Self {
        KvPushRouter { inner, chooser }
    }

    /// Select a worker for the request, either using a preselected worker or finding the best match.
    ///
    /// When `is_query_only` is false, this also registers the request with the scheduler via `add_request`.
    async fn select_worker(
        &self,
        context_id: &str,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        is_query_only: bool,
    ) -> Result<WorkerSelection, Error> {
        let routing = request.routing.as_ref();
        let lora_name = routing.and_then(|r| r.lora_name.clone());
        let priority_jump = routing.and_then(|r| r.priority_jump).unwrap_or(0.0);
        let dp_rank = routing.and_then(|r| r.dp_rank).unwrap_or(0);
        let expected_output_tokens = routing.and_then(|r| r.expected_output_tokens);
        let (routing_token_ids, block_mm_infos) = request.block_mm_routing_info();

        // Get pre-selected worker based on phase, with backend_instance_id as fallback
        let preselected_id = match phase {
            RequestPhase::Prefill => {
                routing.and_then(|r| r.prefill_worker_id.or(r.backend_instance_id))
            }
            RequestPhase::Decode => {
                routing.and_then(|r| r.decode_worker_id.or(r.backend_instance_id))
            }
            RequestPhase::Aggregated => routing.and_then(|r| r.backend_instance_id),
        };

        let Some(id) = preselected_id else {
            let (best_worker, overlap_amount) = self
                .chooser
                .find_best_match(
                    Some(context_id),
                    routing_token_ids,
                    block_mm_infos,
                    request.router_config_override.as_ref(),
                    !is_query_only,
                    lora_name,
                    priority_jump,
                )
                .await?;

            if !is_query_only {
                let total_blocks = routing_token_ids
                    .len()
                    .div_ceil(self.chooser.block_size() as usize);
                // NOTE: tests/mm_router/test_vllm_mm_router_e2e.py parses this log line.
                // Keep the "[ROUTING] ... with X/Y blocks overlap" shape stable unless
                // router tests are updated together.
                tracing::debug!(
                    request_id = %context_id,
                    worker_id = best_worker.worker_id,
                    dp_rank = best_worker.dp_rank,
                    overlap_blocks = overlap_amount,
                    total_blocks = total_blocks,
                    "[ROUTING] Best: worker_{} dp_rank={} with {}/{} blocks overlap",
                    best_worker.worker_id,
                    best_worker.dp_rank,
                    overlap_amount,
                    total_blocks,
                );
            }

            return Ok(WorkerSelection {
                instance_id: best_worker.worker_id,
                dp_rank: best_worker.dp_rank,
                overlap_amount,
            });
        };

        tracing::debug!(
            worker_id = id,
            dp_rank = dp_rank,
            ?phase,
            "Routing to specified worker"
        );

        let worker = WorkerWithDpRank::new(id, dp_rank);
        let overlap_blocks = self
            .chooser
            .get_overlap_blocks(routing_token_ids, worker)
            .await?;

        if !is_query_only {
            self.chooser
                .add_request(
                    context_id.to_string(),
                    routing_token_ids,
                    overlap_blocks,
                    expected_output_tokens,
                    worker,
                    lora_name,
                    request.router_config_override.as_ref(),
                )
                .await;
        } else {
            tracing::debug!(
                request_id = %context_id,
                worker_id = id,
                dp_rank = dp_rank,
                "Skipping add_request - query or handled externally"
            );
        }

        Ok(WorkerSelection {
            instance_id: id,
            dp_rank,
            overlap_amount: overlap_blocks,
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for KvPushRouter
{
    /// Generate method that handles KV-aware routing with three distinct behaviors:
    ///
    /// 1. **If `query_instance_id` annotation is set**:
    ///    - Returns the best matching worker ID without routing the request
    ///    - Does NOT update any router local states
    ///    - Response includes worker_instance_id and token_data annotations
    ///
    /// 2. **If `backend_instance_id` is set in the request**:
    ///    - Routes directly to the specified backend instance
    ///    - DOES update router states to track this request (unless query_instance_id is also set)
    ///    - Bypasses the normal KV matching logic
    ///
    /// 3. **If neither are set (default behavior)**:
    ///    - Finds the best worker based on KV cache overlap
    ///    - Updates router states to track the request
    ///    - Routes to the selected worker
    ///
    /// The router state updates include tracking active sequences and managing
    /// prefill/completion lifecycle for proper KV cache management.
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        // Extract context ID for request tracking
        let context_id = request.context().id().to_string();

        // Simple query-only detection: presence of query_instance_id annotation means query-only mode
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();

        // Get phase from tracker (defaults to Aggregated if no tracker or phase not set)
        let phase = request
            .tracker
            .as_ref()
            .map(|t| t.phase())
            .unwrap_or(RequestPhase::Aggregated);

        let block_size = self.chooser.block_size() as usize;
        let selection = self
            .select_worker(&context_id, &request, phase, is_query_only)
            .instrument(tracing::info_span!("kv_router.select_worker"))
            .await?;
        let WorkerSelection {
            instance_id,
            dp_rank,
            overlap_amount,
        } = selection;

        // In approximate mode (use_kv_events=false), record the routing decision
        // so the indexer can track cache state based on routing decisions.
        // This covers both pre-selected workers and find_best_match selections.
        if !is_query_only && !self.chooser.kv_router_config().use_kv_events {
            let worker = WorkerWithDpRank::new(instance_id, dp_rank);
            let mut tokens_with_hashes =
                TokensWithHashes::new(request.token_ids.clone(), self.chooser.block_size());
            if let Err(e) = self
                .chooser
                .indexer()
                .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
                .await
            {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    dp_rank = dp_rank,
                    error = %e,
                    "Failed to record routing decision in approximate mode"
                );
            }
        }

        // Record routing metrics on tracker and observe ISL + prefill start.
        let request_metrics =
            RouterRequestMetrics::from_component(self.chooser.client().endpoint.component());
        if let Some(ref tracker) = request.tracker {
            let (routing_token_ids, _) = request.block_mm_routing_info();
            let isl_blocks = routing_token_ids.len().div_ceil(block_size);
            tracker.record_kv_hit(overlap_amount, isl_blocks);
            tracker.record_isl(
                routing_token_ids.len(),
                overlap_amount as usize * block_size,
            );
            tracker.record_worker_full(instance_id, dp_rank, self.chooser.worker_type());
            if let Some(hit_rate) = tracker.kv_hit_rate() {
                request_metrics.kv_hit_rate.observe(hit_rate);
            }
        }
        request_metrics
            .input_sequence_tokens
            .observe(request.token_ids.len() as f64);

        // Handle query-only requests: early return with worker info
        if is_query_only {
            let stream_context = request.context().clone();
            let worker_id_info = request.tracker.as_ref().and_then(|t| t.get_worker_info());

            tracing::trace!(
                ?phase,
                worker_id = instance_id,
                ?worker_id_info,
                "Returning worker selection (query-only mode)"
            );

            let output = LLMEngineOutput {
                disaggregated_params: Some(json!({
                    "worker_id": worker_id_info,
                    "token_ids": request.token_ids
                })),
                ..Default::default()
            };
            let response = Annotated::from_data(output);
            let stream = stream::iter(vec![response]);
            return Ok(ResponseStream::new(Box::pin(stream), stream_context));
        }

        // Route to worker
        let isl_tokens = request.token_ids.len();
        let expected_output_tokens = request
            .routing
            .as_ref()
            .and_then(|r| r.expected_output_tokens);
        let track_output_blocks = self.chooser.kv_router_config().router_track_output_blocks;
        let tracker = request.tracker.clone();

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = Some(dp_rank);
        let updated_request = context.map(|_| backend_input);

        // Record prefill start right before pushing to backend (OnceLock: first call wins).
        if let Some(ref tracker) = tracker {
            tracker.record_prefill_start();
        }

        let chooser = self.chooser.clone();
        let mut response_stream = self
            .inner
            .direct(updated_request, instance_id)
            .instrument(tracing::info_span!(
                "kv_router.route_request",
                request_id = %context_id,
                worker_id = instance_id,
                dp_rank = dp_rank,
                overlap_blocks = overlap_amount,
                phase = ?phase,
            ))
            .await?;
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();

        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut guard = RequestGuard {
                chooser: chooser.clone(),
                context_id: context_id.clone(),
                tracker: tracker.clone(),
                request_metrics: request_metrics.clone(),
                cumulative_osl: 0,
                metrics_recorded: false,
                freed: false,
                prefill_marked: false,
                first_token_recorded: false,
                track_output_blocks,
                current_total_blocks: isl_tokens.div_ceil(block_size),
                isl_tokens,
                block_size,
                expected_output_tokens,
            };

            loop {
                tokio::select! {
                    biased;

                    _ = context_for_monitoring.stopped() => {
                        tracing::debug!("Request {context_id} cancelled, ending stream");
                        break;
                    }

                    item = response_stream.next() => {
                        let Some(item) = item else {
                            break;
                        };
                        guard.on_item(&item).await;
                        yield item;
                    }
                }
            }

            guard.finish().await;
        });
        Ok(ResponseStream::new(wrapped_stream, stream_context))
    }
}

/// A direct routing wrapper for `RouterMode::Direct`.
///
/// This wraps a `PushRouter` and reads worker IDs from each request's routing hints,
/// then routes directly to the specified worker. Used when an external router
/// (e.g., EPP) handles worker selection.
pub struct DirectRoutingRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
}

impl DirectRoutingRouter {
    pub fn new(inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>) -> Self {
        DirectRoutingRouter { inner }
    }

    /// Extract worker ID from request routing hints.
    /// Returns an error if no worker ID is found (required in direct routing mode).
    fn get_worker_id(request: &PreprocessedRequest) -> Result<u64, Error> {
        let routing = request.routing.as_ref();
        let worker_id = routing.and_then(|r| r.decode_worker_id.or(r.backend_instance_id));

        worker_id.ok_or_else(|| {
            anyhow::anyhow!(
                "Worker ID required (--direct-route) but none found in request. \
                 Expected decode_worker_id or backend_instance_id to be set by external router (e.g., EPP)."
            )
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for DirectRoutingRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        let worker_id = Self::get_worker_id(&request)?;

        tracing::debug!(worker_id = worker_id, "Direct routing to specified worker");

        self.inner.direct(request, worker_id).await
    }
}
