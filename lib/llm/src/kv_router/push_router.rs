// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::protocols::{TokensWithHashes, WorkerWithDpRank};
use dynamo_runtime::{
    dynamo_nvtx_range,
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
        agent_controller::{AgentController, SessionCloseAction},
        metrics::RouterRequestMetrics,
        sticky_sessions::{InMemoryAffinityStore, StickySessionRouter},
    },
    preprocessor::PreprocessedRequest,
    protocols::common::{
        llm_backend::LLMEngineOutput,
        preprocessor::RoutingHints,
        timing::{RequestPhase, RequestTracker},
    },
};

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
    /// Sticky session routing. Lazily activated when requests carry session_control.
    sticky_sessions: Arc<StickySessionRouter>,
    /// Session lifecycle RPCs (open/close). Client is lazy (OnceCell).
    agent_controller: Arc<AgentController>,
}

/// Result of worker selection containing instance ID, dp_rank, and overlap amount.
struct WorkerSelection {
    instance_id: u64,
    backend_dp_rank: Option<u32>,
    bookkeeping_dp_rank: Option<u32>,
    overlap_amount: Option<u32>,
}

fn pinned_worker_hint(
    phase: RequestPhase,
    routing: Option<&RoutingHints>,
) -> Option<(u64, Option<u32>)> {
    let routing = routing?;
    let worker_id = match phase {
        RequestPhase::Prefill => routing.prefill_worker_id.or(routing.backend_instance_id),
        RequestPhase::Decode => routing.decode_worker_id.or(routing.backend_instance_id),
        RequestPhase::Aggregated => routing.backend_instance_id,
    }?;
    let dp_rank = match phase {
        RequestPhase::Prefill => routing.prefill_dp_rank.or(routing.dp_rank),
        RequestPhase::Decode | RequestPhase::Aggregated => routing.dp_rank,
    };
    Some((worker_id, dp_rank))
}

/// Drop guard that manages the full lifecycle of a routed request:
/// per-item tracking (prefill, first token, output blocks) and final cleanup (free + metrics).
///
/// In the happy path, `finish().await` runs cleanup inline in the async context.
/// If the stream is dropped early (e.g., client disconnect, consumer drop), the
/// `Drop` impl fires and spawns a task to call `free()`.
struct RequestGuard {
    chooser: Arc<KvRouter>,
    scheduler_tracked: bool,
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
    /// Deferred session close action (fires after generation completes)
    deferred_close: Option<SessionCloseAction>,
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
                if self.scheduler_tracked
                    && let Err(e) = self.chooser.mark_prefill_completed(&self.context_id).await
                {
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
                // Record decode-phase first token for KV transfer latency metric.
                // In disaggregated serving, first_token_time is locked by the prefill phase,
                // so we need a separate timestamp for the decode worker's first token.
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
        if self.scheduler_tracked
            && let Err(e) = self.chooser.free(&self.context_id).await
        {
            tracing::warn!("Failed to free request {}: {e}", self.context_id);
        }
        self.freed = true;

        // Take to prevent double-fire from Drop
        if let Some(close) = self.deferred_close.take() {
            close.execute(&self.context_id);
        }
    }

    fn record_metrics(&mut self) {
        if self.metrics_recorded {
            return;
        }
        self.metrics_recorded = true;
        if let Some(ref tracker) = self.tracker {
            tracker.record_finish();
            tracker.record_osl(self.cumulative_osl);
            // Observe KV transfer estimated latency (disaggregated paths)
            if let Some(latency) = tracker.kv_transfer_estimated_latency_secs() {
                self.request_metrics
                    .kv_transfer_estimated_latency_seconds
                    .observe(latency);
            }
        }
        self.request_metrics
            .output_sequence_tokens
            .observe(self.cumulative_osl as f64);
        self.request_metrics.requests_total.inc();
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.record_metrics();

        let deferred_close = self.deferred_close.take();
        let needs_free = !self.freed && self.scheduler_tracked;

        if deferred_close.is_none() && !needs_free {
            return;
        }

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                "No tokio runtime for drop guard cleanup of request {}",
                self.context_id
            );
            return;
        };

        // Mirror finish(): free the scheduler slot first, then fire the
        // deferred session close so the worker's KV isn't released while
        // generation teardown is still in progress.
        let chooser = self.chooser.clone();
        let context_id = self.context_id.clone();
        handle.spawn(async move {
            if needs_free && let Err(e) = chooser.free(&context_id).await {
                tracing::warn!("Failed to free request {context_id} (drop guard): {e}");
            }
            if let Some(close) = deferred_close {
                close.execute(&context_id);
            }
        });
    }
}

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
    ) -> Self {
        // Eagerly register router request metrics (as zeros) so they are
        // scrapeable before any requests arrive. Both the frontend pipeline
        // and the standalone router create KvPushRouter, so this covers both.
        RouterRequestMetrics::from_component(chooser.client().endpoint.component());

        // Agent controller manages session lifecycle RPCs (open/close).
        // Always created; the event-plane client inside is lazy (OnceCell)
        // so there is zero cost until a request actually carries session_control.
        let component = chooser.client().endpoint.component().clone();
        let agent_controller = Arc::new(AgentController::new(component));

        // Sticky sessions share expiry handling with the agent controller so
        // router-side reap also closes the worker session.
        let on_expire = {
            let controller = agent_controller.clone();
            Arc::new(move |session_id: String, worker_id: u64| {
                controller
                    .clone()
                    .close_expired_session(session_id, worker_id);
            }) as Arc<dyn Fn(String, u64) + Send + Sync>
        };
        let sticky_sessions = Arc::new(StickySessionRouter::new(
            InMemoryAffinityStore::new_with_on_expire(Some(on_expire)),
        ));

        KvPushRouter {
            inner,
            chooser,
            sticky_sessions,
            agent_controller,
        }
    }

    /// Select a worker for the request, either using an exact phase-specific pin
    /// or by finding the best KV overlap match.
    async fn select_worker(
        &self,
        context_id: &str,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        is_query_only: bool,
    ) -> Result<WorkerSelection, Error> {
        let _nvtx_select = dynamo_nvtx_range!("route.select_worker");
        let routing = request.routing.as_ref();
        let lora_name = routing.and_then(|r| r.lora_name.clone());
        let cache_salt = routing.and_then(|r| r.cache_salt.clone());
        let priority_jump = routing.and_then(|r| r.priority_jump).unwrap_or(0.0);
        let expected_output_tokens = routing.and_then(|r| r.expected_output_tokens);
        let allowed_worker_ids = routing.and_then(|r| r.allowed_worker_ids.clone());
        let (routing_token_ids, block_mm_infos) = request.block_mm_routing_info();
        let Some((pinned_worker_id, requested_dp_rank)) = pinned_worker_hint(phase, routing) else {
            let _nvtx_kv = dynamo_nvtx_range!("route.kv_match");
            let (best_worker, overlap_amount) = self
                .chooser
                .find_best_match(
                    Some(context_id),
                    routing_token_ids,
                    block_mm_infos,
                    request.router_config_override.as_ref(),
                    !is_query_only,
                    lora_name,
                    cache_salt.clone(),
                    priority_jump,
                    expected_output_tokens,
                    None,
                    allowed_worker_ids,
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
                backend_dp_rank: Some(best_worker.dp_rank),
                bookkeeping_dp_rank: Some(best_worker.dp_rank),
                overlap_amount: Some(overlap_amount),
            });
        };

        let resolved_pinned_worker = requested_dp_rank
            .or_else(|| self.chooser.unique_dp_rank_for_worker(pinned_worker_id))
            .map(|dp_rank| WorkerWithDpRank::new(pinned_worker_id, dp_rank));

        if !is_query_only && let Some(pinned_worker) = resolved_pinned_worker {
            let (best_worker, overlap_amount) = self
                .chooser
                .find_best_match(
                    Some(context_id),
                    routing_token_ids,
                    block_mm_infos,
                    request.router_config_override.as_ref(),
                    true,
                    lora_name.clone(),
                    cache_salt.clone(),
                    priority_jump,
                    expected_output_tokens,
                    Some(pinned_worker),
                    allowed_worker_ids,
                )
                .await?;

            return Ok(WorkerSelection {
                instance_id: best_worker.worker_id,
                backend_dp_rank: Some(best_worker.dp_rank),
                bookkeeping_dp_rank: Some(best_worker.dp_rank),
                overlap_amount: Some(overlap_amount),
            });
        }

        let backend_dp_rank = resolved_pinned_worker.map(|worker| worker.dp_rank);

        tracing::debug!(
            worker_id = pinned_worker_id,
            dp_rank = ?backend_dp_rank,
            ?phase,
            "Routing to specified worker"
        );

        let (bookkeeping_dp_rank, overlap_amount) = if let Some(dp_rank) = backend_dp_rank {
            let worker = WorkerWithDpRank::new(pinned_worker_id, dp_rank);
            let overlap_blocks = self
                .chooser
                .get_overlap_blocks(
                    routing_token_ids,
                    block_mm_infos,
                    worker,
                    lora_name.as_deref(),
                )
                .await?;

            if !is_query_only {
                self.chooser
                    .add_request(
                        context_id.to_string(),
                        routing_token_ids,
                        block_mm_infos,
                        overlap_blocks,
                        expected_output_tokens,
                        worker,
                        lora_name,
                        cache_salt,
                        request.router_config_override.as_ref(),
                    )
                    .await;
            } else {
                tracing::debug!(
                    request_id = %context_id,
                    worker_id = pinned_worker_id,
                    dp_rank = dp_rank,
                    "Skipping add_request - query-only request"
                );
            }

            (Some(dp_rank), Some(overlap_blocks))
        } else {
            tracing::debug!(
                request_id = %context_id,
                worker_id = pinned_worker_id,
                ?phase,
                "Routing to specified worker without resolved dp_rank; skipping scheduler bookkeeping"
            );
            (None, None)
        };

        Ok(WorkerSelection {
            instance_id: pinned_worker_id,
            backend_dp_rank,
            bookkeeping_dp_rank,
            overlap_amount,
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
    /// 2. **If a phase-specific worker or `backend_instance_id` is set in the request**:
    ///    - Query-only requests return that worker selection without state updates
    ///    - Execution requests route through the scheduler as an exact pin when dp_rank is resolved
    ///    - If dp_rank cannot be resolved, falls back to direct routing without scheduler bookkeeping
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
        mut request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        // Extract context ID for request tracking
        let context_id = request.context().id().to_string();

        // Simple query-only detection: presence of query_instance_id annotation means query-only mode
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();

        // Resolve session affinity: if the request has a session_id, inject the
        // pinned worker_id into backend_instance_id before worker selection.
        // Skip entirely for non-session requests to keep them off the sticky path.
        if request
            .routing
            .as_ref()
            .and_then(|r| r.session_control.as_ref())
            .is_some()
            && request
                .routing
                .as_ref()
                .and_then(|r| r.backend_instance_id)
                .is_none()
            && let Some(worker_id) = self.sticky_sessions.resolve(&request)
        {
            request.routing_mut().backend_instance_id = Some(worker_id);
        }

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
            backend_dp_rank,
            bookkeeping_dp_rank,
            overlap_amount,
        } = selection;
        let scheduler_tracked = !is_query_only && bookkeeping_dp_rank.is_some();

        // In approximate mode (use_kv_events=false), record the routing decision
        // so the indexer can track cache state based on routing decisions.
        // This covers both pre-selected workers and find_best_match selections.
        if !is_query_only && !self.chooser.kv_router_config().use_kv_events {
            if let Some(dp_rank) = bookkeeping_dp_rank {
                let lora_name = request.routing.as_ref().and_then(|r| r.lora_name.clone());
                let cache_salt = request.routing.as_ref().and_then(|r| r.cache_salt.clone());
                let (routing_token_ids, block_mm_infos) = request.block_mm_routing_info();
                let worker = WorkerWithDpRank::new(instance_id, dp_rank);
                let mut tokens_with_hashes =
                    TokensWithHashes::new(routing_token_ids.to_vec(), self.chooser.block_size())
                        .with_is_eagle(self.chooser.is_eagle());
                if let Some(infos) = block_mm_infos {
                    tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.to_vec());
                }
                if let Some(lora_name) = lora_name {
                    tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name);
                }
                if let Some(cache_salt) = cache_salt {
                    tokens_with_hashes = tokens_with_hashes.with_cache_salt(cache_salt);
                }
                if let Err(e) = self
                    .chooser
                    .record_routing_decision(tokens_with_hashes, worker)
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
            } else {
                tracing::debug!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    "Skipping approximate-mode routing decision for unresolved dp_rank"
                );
            }
        }

        // Record routing metrics on tracker and observe ISL + prefill start.
        let request_metrics =
            RouterRequestMetrics::from_component(self.chooser.client().endpoint.component());
        if let Some(ref tracker) = request.tracker {
            let (routing_token_ids, _) = request.block_mm_routing_info();
            let isl_blocks = routing_token_ids.len().div_ceil(block_size);
            if let Some(overlap_amount) = overlap_amount {
                tracker.record_kv_hit(overlap_amount, isl_blocks);
            }
            tracker.record_isl(
                routing_token_ids.len(),
                overlap_amount.map(|overlap| overlap as usize * block_size),
            );
            tracker.record_worker(instance_id, backend_dp_rank, self.chooser.worker_type());
            tracker.record_router_queue_depth(self.chooser.pending_count());
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

        // Session lifecycle RPCs via agent controller.
        // Fails fast if session_control.open is requested but the client can't be created.
        let deferred_close = self
            .agent_controller
            .on_routed(
                &request,
                instance_id,
                &context_id,
                Some(&*self.sticky_sessions),
            )
            .await?;

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = backend_dp_rank;
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
                dp_rank = ?backend_dp_rank,
                overlap_blocks = ?overlap_amount,
                phase = ?phase,
            ))
            .await?;
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();

        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut guard = RequestGuard {
                chooser: chooser.clone(),
                scheduler_tracked,
                context_id: context_id.clone(),
                tracker: tracker.clone(),
                request_metrics: request_metrics.clone(),
                cumulative_osl: 0,
                metrics_recorded: false,
                freed: false,
                prefill_marked: false,
                first_token_recorded: false,
                track_output_blocks: scheduler_tracked && track_output_blocks,
                current_total_blocks: isl_tokens.div_ceil(block_size),
                isl_tokens,
                block_size,
                expected_output_tokens,
                deferred_close,
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

#[cfg(test)]
mod tests {
    use super::pinned_worker_hint;
    use crate::protocols::common::{preprocessor::RoutingHints, timing::RequestPhase};

    #[test]
    fn pinned_worker_hint_prefill_uses_prefill_worker_before_backend() {
        let routing = RoutingHints {
            backend_instance_id: Some(1),
            prefill_worker_id: Some(2),
            dp_rank: Some(3),
            prefill_dp_rank: Some(4),
            ..Default::default()
        };

        let hint = pinned_worker_hint(RequestPhase::Prefill, Some(&routing));
        assert_eq!(hint, Some((2, Some(4))));
    }

    #[test]
    fn pinned_worker_hint_decode_uses_decode_worker_before_backend() {
        let routing = RoutingHints {
            backend_instance_id: Some(1),
            decode_worker_id: Some(5),
            dp_rank: Some(6),
            ..Default::default()
        };

        let hint = pinned_worker_hint(RequestPhase::Decode, Some(&routing));
        assert_eq!(hint, Some((5, Some(6))));
    }

    #[test]
    fn pinned_worker_hint_aggregated_uses_backend_worker() {
        let routing = RoutingHints {
            backend_instance_id: Some(9),
            dp_rank: Some(7),
            ..Default::default()
        };

        let hint = pinned_worker_hint(RequestPhase::Aggregated, Some(&routing));
        assert_eq!(hint, Some((9, Some(7))));
    }
}
