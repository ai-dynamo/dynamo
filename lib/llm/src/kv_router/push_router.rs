// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use anyhow::Result;
use dynamo_kv_router::{
    RouterConfigOverride,
    indexer::RoutingDecisionHashes,
    protocols::{BlockExtraInfo, RoutingConstraints, TokensWithHashes, WorkerId, WorkerWithDpRank},
    scheduling::{RoutingEligibility, WorkerEligibilityError},
};
use dynamo_runtime::{
    dynamo_nvtx_range,
    error::{DynamoError, ErrorType as DynamoErrorType},
    metrics::frontend_perf::{STAGE_DISPATCH, STAGE_ROUTE, StageGuard},
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
        FindBestMatchOutcome, KvRouter,
        metrics::RouterRequestMetrics,
        sticky::{
            coordinator::{StickySessionCoordinator, sticky_allowed_for_phase},
            lifecycle::SessionCloseAction,
        },
    },
    preprocessor::PreprocessedRequest,
    protocols::{
        TokenIdType,
        common::{
            llm_backend::LLMEngineOutput,
            preprocessor::RoutingHints,
            timing::{RequestPhase, RequestTracker},
        },
    },
};

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
    /// Sticky session routing. Lazily activated when requests carry session_control.
    pub(super) sticky: Arc<StickySessionCoordinator>,
}

/// Result of worker selection containing instance ID, dp_rank, and overlap amount.
struct WorkerSelection {
    instance_id: u64,
    dp_rank: u32,
    overlap_amount: u32,
    effective_overlap_blocks: f64,
    cached_tokens: usize,
    routing_hashes: Option<RoutingDecisionHashes>,
    /// Whether the scheduler is tracking this request (add_request or
    /// find_best_match_details with update_states=true was called).
    scheduler_tracked: bool,
}

// NOTE: In KV router mode, worker selection is DP-rank precise. A pinned
// worker without a concrete dp_rank is invalid unless the worker owns exactly
// one rank and can be resolved unambiguously. Rank 0 is a real rank, not an
// unset sentinel. Do not coerce unresolved ranks to 0.
fn resolve_pinned_worker_rank(
    worker_id: WorkerId,
    requested_dp_rank: Option<u32>,
    unique_dp_rank: Option<u32>,
) -> Result<WorkerWithDpRank, Error> {
    let Some(dp_rank) = requested_dp_rank.or(unique_dp_rank) else {
        return Err(anyhow::anyhow!(
            "Pinned worker {worker_id} requires an explicit dp_rank because it has multiple or unknown DP ranks"
        ));
    };

    Ok(WorkerWithDpRank::new(worker_id, dp_rank))
}

#[derive(Clone, Copy)]
struct RoutingRequestParts<'a> {
    token_ids: &'a [TokenIdType],
    block_mm_infos: Option<&'a [Option<BlockExtraInfo>]>,
}

impl<'a> RoutingRequestParts<'a> {
    fn new(request: &'a PreprocessedRequest) -> Self {
        let (token_ids, block_mm_infos) = request.block_mm_routing_info();
        Self {
            token_ids,
            block_mm_infos,
        }
    }
}

struct BestMatchArgs<'a> {
    context_id: &'a str,
    routing_parts: RoutingRequestParts<'a>,
    router_config_override: Option<&'a RouterConfigOverride>,
    update_states: bool,
    return_routing_hashes: bool,
    lora_name: Option<String>,
    priority_jump: f64,
    expected_output_tokens: Option<u32>,
    pinned_worker: Option<WorkerWithDpRank>,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    routing_constraints: RoutingConstraints,
    scheduler_tracked: bool,
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
    first_response_received: bool,
    dispatch_guard: Option<StageGuard>,
    track_output_blocks: bool,
    current_total_blocks: usize,
    isl_tokens: usize,
    block_size: usize,
    expected_output_tokens: Option<u32>,
    /// Deferred session close action (fires after generation completes)
    deferred_close: Option<SessionCloseAction>,
    /// True once inner.direct() has returned Ok — guards record_metrics() so
    /// that a dispatch failure does not emit metrics for a request that never
    /// reached the backend (spurious requests_total increment, OSL histogram
    /// zeros, premature tracker.record_finish()).
    dispatched: bool,
}

impl RequestGuard {
    async fn on_item(&mut self, item: &Annotated<LLMEngineOutput>) {
        // End dispatch stage on first response from backend (any item, not just tokens).
        if !self.first_response_received {
            self.first_response_received = true;
            self.dispatch_guard.take();
        }

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
        // Skip metrics for requests that never reached the backend (dispatch
        // failure before direct() returned Ok). Recording here would emit
        // spurious requests_total increments and OSL-histogram zeros.
        if self.metrics_recorded || !self.dispatched {
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
        // Only record output sequence length for requests that actually
        // produced output tokens. Recording zero for failed/cancelled requests
        // would corrupt histogram averages (sum/count) and percentiles.
        // Failures are already tracked by requests_total.
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

        let component = chooser.client().endpoint.component().clone();
        let sticky = Arc::new(StickySessionCoordinator::new(component));

        KvPushRouter {
            inner,
            chooser,
            sticky,
        }
    }

    async fn select_best_match(&self, args: BestMatchArgs<'_>) -> Result<WorkerSelection, Error> {
        let outcome = self
            .chooser
            .find_best_match_details(
                Some(args.context_id),
                args.routing_parts.token_ids,
                args.routing_parts.block_mm_infos,
                args.router_config_override,
                args.update_states,
                args.return_routing_hashes,
                args.lora_name,
                args.priority_jump,
                args.expected_output_tokens,
                args.pinned_worker,
                args.allowed_worker_ids,
                args.routing_constraints,
            )
            .await?;

        match outcome {
            FindBestMatchOutcome::Routed {
                worker,
                overlap_blocks,
                effective_overlap_blocks,
                cached_tokens,
                routing_hashes,
            } => Ok(WorkerSelection {
                instance_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                overlap_amount: overlap_blocks,
                effective_overlap_blocks,
                cached_tokens,
                routing_hashes,
                scheduler_tracked: args.scheduler_tracked,
            }),
            FindBestMatchOutcome::Backpressure {
                reason,
                queued_isl_tokens,
                max_queued_isl_tokens,
            } => Err(DynamoError::builder()
                .error_type(DynamoErrorType::ResourceExhausted)
                .message(format!(
                    "router backpressure: {reason:?} (queued_isl_tokens={queued_isl_tokens}, max_queued_isl_tokens={max_queued_isl_tokens:?})"
                ))
                .build()
                .into()),
        }
    }

    /// Select a worker for the request, either using an exact phase-specific pin
    /// or by finding the best KV overlap match.
    async fn select_worker(
        &self,
        context_id: &str,
        request: &PreprocessedRequest,
        routing_parts: RoutingRequestParts<'_>,
        phase: RequestPhase,
        is_query_only: bool,
        sticky_worker: Option<WorkerWithDpRank>,
    ) -> Result<WorkerSelection, Error> {
        let _nvtx_select = dynamo_nvtx_range!("route.select_worker");
        let routing = request.routing.as_ref();
        let lora_name = routing.and_then(|r| r.lora_name.clone());
        let priority_jump = routing.and_then(|r| r.priority_jump).unwrap_or(0.0);
        let expected_output_tokens = routing.and_then(|r| r.expected_output_tokens);
        let allowed_worker_ids = routing.and_then(|r| r.allowed_worker_ids.clone());
        let return_routing_hashes =
            !is_query_only && self.chooser.indexer().records_routing_decisions();
        let routing_constraints = routing
            .and_then(|r| r.routing_constraints.clone())
            .unwrap_or_default();
        let sticky_pin = sticky_worker.map(|worker| (worker.worker_id, Some(worker.dp_rank)));
        let Some((pinned_worker_id, requested_dp_rank)) =
            pinned_worker_hint(phase, routing).or(sticky_pin)
        else {
            let _nvtx_kv = dynamo_nvtx_range!("route.kv_match");
            let selection = self
                .select_best_match(BestMatchArgs {
                    context_id,
                    routing_parts,
                    router_config_override: request.router_config_override.as_ref(),
                    update_states: !is_query_only,
                    return_routing_hashes,
                    lora_name,
                    priority_jump,
                    expected_output_tokens,
                    pinned_worker: None,
                    allowed_worker_ids,
                    routing_constraints: routing_constraints.clone(),
                    scheduler_tracked: !is_query_only,
                })
                .await?;

            if !is_query_only {
                let total_blocks = routing_parts
                    .token_ids
                    .len()
                    .div_ceil(self.chooser.block_size() as usize);
                // tests/utils/router_logs.py parses the structured fields on this event.
                tracing::debug!(
                    request_id = %context_id,
                    worker_id = selection.instance_id,
                    dp_rank = selection.dp_rank,
                    overlap_blocks = selection.overlap_amount,
                    total_blocks = total_blocks,
                    "[ROUTING] Best: worker_{} dp_rank={} with {}/{} blocks overlap",
                    selection.instance_id,
                    selection.dp_rank,
                    selection.overlap_amount,
                    total_blocks,
                );
            }

            return Ok(selection);
        };

        let pinned_worker = resolve_pinned_worker_rank(
            pinned_worker_id,
            requested_dp_rank,
            self.chooser.unique_dp_rank_for_worker(pinned_worker_id),
        )?;
        {
            let configs = self.chooser.workers_with_configs.borrow();
            let eligibility = RoutingEligibility::new(
                allowed_worker_ids.as_ref(),
                None,
                Some(pinned_worker),
                &routing_constraints,
            );
            if let Err(error) = eligibility.validate_worker_rank(&configs, pinned_worker) {
                return Err(anyhow::anyhow!(
                    "Pinned worker {} dp_rank {} is not eligible: {error}",
                    pinned_worker.worker_id,
                    pinned_worker.dp_rank
                ));
            }
        }

        tracing::debug!(
            worker_id = pinned_worker.worker_id,
            dp_rank = pinned_worker.dp_rank,
            ?phase,
            "Routing to specified worker"
        );

        self.select_best_match(BestMatchArgs {
            context_id,
            routing_parts,
            router_config_override: request.router_config_override.as_ref(),
            update_states: !is_query_only,
            return_routing_hashes,
            lora_name,
            priority_jump,
            expected_output_tokens,
            pinned_worker: Some(pinned_worker),
            allowed_worker_ids,
            routing_constraints,
            scheduler_tracked: !is_query_only,
        })
        .await
    }

    fn sticky_worker_ineligibility_for_phase(
        &self,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        worker: WorkerWithDpRank,
    ) -> Option<WorkerEligibilityError> {
        let routing = request.routing.as_ref()?;
        if !sticky_allowed_for_phase(phase, Some(routing)) {
            return None;
        }

        let default_constraints = RoutingConstraints::default();
        let routing_constraints = routing
            .routing_constraints
            .as_ref()
            .unwrap_or(&default_constraints);
        let configs = self.chooser.workers_with_configs.borrow();
        let eligibility = RoutingEligibility::new(
            routing.allowed_worker_ids.as_ref(),
            None,
            Some(worker),
            routing_constraints,
        );
        eligibility.validate_worker_rank(&configs, worker).err()
    }

    pub(crate) fn unbind_ineligible_sticky_worker_for_phase(
        &self,
        context_id: &str,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        worker: WorkerWithDpRank,
    ) -> bool {
        let Some(reason) = self.sticky_worker_ineligibility_for_phase(request, phase, worker)
        else {
            return false;
        };

        let Some((session_id, _binding)) = self.sticky.unbind_for_phase(request, phase) else {
            return false;
        };
        tracing::warn!(
            request_id = %context_id,
            %session_id,
            worker_id = worker.worker_id,
            dp_rank = worker.dp_rank,
            reason = %reason,
            "Sticky worker is no longer eligible; removing session affinity"
        );
        true
    }

    pub(crate) async fn validate_sticky_worker_for_phase(
        &self,
        context_id: &str,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        worker: WorkerWithDpRank,
    ) -> Result<WorkerWithDpRank, Error> {
        let routing_parts = RoutingRequestParts::new(request);
        let selection = self
            .select_worker(
                context_id,
                request,
                routing_parts,
                phase,
                true,
                Some(worker),
            )
            .await?;
        Ok(WorkerWithDpRank::new(
            selection.instance_id,
            selection.dp_rank,
        ))
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
    ///    - Requests route through the scheduler as an exact pin when dp_rank is resolved
    ///    - If dp_rank cannot be resolved, the request is rejected instead of treating rank 0 as a sentinel
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
        let phase_label = phase.to_string();
        let route_guard = StageGuard::new(STAGE_ROUTE, &phase_label);

        let should_record = !is_query_only && self.chooser.indexer().records_routing_decisions();
        let block_size = self.chooser.block_size() as usize;
        let routing_parts = RoutingRequestParts::new(&request);
        let sticky_worker = match self.sticky.worker_for_phase(&request, phase) {
            Some(worker)
                if self.unbind_ineligible_sticky_worker_for_phase(
                    &context_id,
                    &request,
                    phase,
                    worker,
                ) =>
            {
                None
            }
            worker => worker,
        };
        let selection = match self
            .select_worker(
                &context_id,
                &request,
                routing_parts,
                phase,
                is_query_only,
                sticky_worker,
            )
            .instrument(tracing::info_span!("kv_router.select_worker"))
            .await
        {
            Ok(selection) => {
                if sticky_worker.is_some() && !is_query_only {
                    self.sticky.refresh_worker_for_phase(&request, phase);
                }
                selection
            }
            Err(error) if sticky_worker.is_some() => {
                if let Some(worker) = sticky_worker {
                    let unbound = self.unbind_ineligible_sticky_worker_for_phase(
                        &context_id,
                        &request,
                        phase,
                        worker,
                    );
                    tracing::warn!(
                        request_id = %context_id,
                        worker_id = worker.worker_id,
                        dp_rank = worker.dp_rank,
                        error = %error,
                        unbound_due_to_ineligibility = unbound,
                        "Sticky worker routing failed; falling back to normal routing"
                    );
                }
                self.select_worker(
                    &context_id,
                    &request,
                    routing_parts,
                    phase,
                    is_query_only,
                    None,
                )
                .instrument(tracing::info_span!("kv_router.select_worker_fallback"))
                .await?
            }
            Err(error) => return Err(error),
        };
        let WorkerSelection {
            instance_id,
            dp_rank,
            overlap_amount,
            effective_overlap_blocks,
            cached_tokens,
            routing_hashes,
            scheduler_tracked,
        } = selection;

        if should_record {
            let worker = WorkerWithDpRank::new(instance_id, dp_rank);
            let record_result = if let Some(hashes) = routing_hashes {
                self.chooser
                    .record_routing_decision_hashes(hashes, worker)
                    .await
            } else {
                let lora_name = request.routing.as_ref().and_then(|r| r.lora_name.clone());
                let mut tokens_with_hashes = TokensWithHashes::new(
                    routing_parts.token_ids.to_vec(),
                    self.chooser.block_size(),
                )
                .with_is_eagle(self.chooser.is_eagle());
                if let Some(infos) = routing_parts.block_mm_infos {
                    tokens_with_hashes = tokens_with_hashes.with_mm_infos(infos.to_vec());
                }
                if let Some(lora_name) = lora_name {
                    tokens_with_hashes = tokens_with_hashes.with_lora_name(lora_name);
                }
                self.chooser
                    .record_routing_decision(tokens_with_hashes, worker)
                    .await
            };
            if let Err(e) = record_result {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    dp_rank = dp_rank,
                    error = %e,
                    "Failed to record routing decision"
                );
            }
        }

        // Record routing metrics on tracker and observe ISL + prefill start.
        let request_metrics =
            RouterRequestMetrics::from_component(self.chooser.client().endpoint.component());
        if let Some(ref tracker) = request.tracker {
            let isl_blocks = routing_parts.token_ids.len().div_ceil(block_size);
            tracker.record_kv_hit(effective_overlap_blocks, isl_blocks);
            tracker.record_isl(routing_parts.token_ids.len(), Some(cached_tokens));
            tracker.record_worker(instance_id, Some(dp_rank), self.chooser.worker_type());
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

        // End route stage — worker has been selected and routing metrics recorded.
        // Dispatch stage starts immediately so there is no gap between stages.
        drop(route_guard);
        let stage_dispatch_guard = StageGuard::new(STAGE_DISPATCH, &phase_label);

        // Dispatch to worker
        let isl_tokens = request.token_ids.len();
        let expected_output_tokens = request
            .routing
            .as_ref()
            .and_then(|r| r.expected_output_tokens);
        let track_output_blocks = self.chooser.kv_router_config().router_track_output_blocks;
        let tracker = request.tracker.clone();

        // Session lifecycle RPCs.
        // Fails fast if session_control.open is requested but the client can't be created.
        let worker = WorkerWithDpRank::new(instance_id, dp_rank);
        let route_outcome = self.sticky.on_routed(&request, worker, &context_id).await?;
        let deferred_close = route_outcome.deferred_close;

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = Some(dp_rank);
        let updated_request = context.map(|_| backend_input);

        // Record prefill start right before pushing to backend (OnceLock: first call wins).
        if let Some(ref tracker) = tracker {
            tracker.record_prefill_start();
        }

        let chooser = self.chooser.clone();

        // Build the guard BEFORE calling direct() so that its Drop covers the
        // error path as well as the drop-before-first-poll path.
        //
        // Without this, if direct().await? below returns Err, both the
        // scheduler slot (booked by find_best_match with update_states=true)
        // and the SessionCloseAction (obtained above via on_routed) are leaked:
        // SessionCloseAction has no Drop impl, so dropping it never sends the
        // close_session RPC; chooser.free() is only called via RequestGuard::Drop.
        //
        // All guard fields are available here (deferred_close was just obtained;
        // isl_tokens/block_size/tracker were set before request.into_parts()).
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
            first_response_received: false,
            dispatch_guard: Some(stage_dispatch_guard),
            track_output_blocks: scheduler_tracked && track_output_blocks,
            current_total_blocks: isl_tokens.div_ceil(block_size),
            isl_tokens,
            block_size,
            expected_output_tokens,
            deferred_close,
            dispatched: false,
        };

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
        // direct() succeeded — mark dispatched so record_metrics() fires.
        // If direct() returned Err above, guard drops here with dispatched=false
        // → RequestGuard::Drop fires → chooser.free() + deferred_close.execute()
        //   but record_metrics() is suppressed (no backend work was done).
        guard.dispatched = true;
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();

        let wrapped_stream = Box::pin(async_stream::stream! {
            // Move guard into the stream closure. Drop fires here if the stream
            // is polled to completion, or via the outer Drop if never polled.
            let mut guard = guard;

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

/// Extract a phase-specific (worker_id, dp_rank) pin from routing hints.
///
/// Returns `Some((worker_id, optional_dp_rank))` when the request should be
/// pinned to a particular worker, or `None` when the normal KV-overlap
/// selection path should be used.
fn pinned_worker_hint(
    phase: RequestPhase,
    routing: Option<&RoutingHints>,
) -> Option<(u64, Option<u32>)> {
    let routing = routing?;
    match phase {
        RequestPhase::Prefill => {
            let worker_id = routing.prefill_worker_id.or(routing.backend_instance_id)?;
            let dp_rank = routing.prefill_dp_rank.or(routing.dp_rank);
            Some((worker_id, dp_rank))
        }
        RequestPhase::Decode => {
            let worker_id = routing.decode_worker_id.or(routing.backend_instance_id)?;
            let dp_rank = routing.dp_rank;
            Some((worker_id, dp_rank))
        }
        RequestPhase::Aggregated => {
            let worker_id = routing.backend_instance_id?;
            let dp_rank = routing.dp_rank;
            Some((worker_id, dp_rank))
        }
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
    use std::collections::{HashMap, HashSet};

    use dynamo_kv_router::{
        protocols::{RoutingConstraints, WorkerWithDpRank},
        scheduling::{RoutingEligibility, WorkerEligibilityError},
    };

    use super::{pinned_worker_hint, resolve_pinned_worker_rank};
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use crate::protocols::common::{preprocessor::RoutingHints, timing::RequestPhase};

    #[test]
    fn resolve_pinned_worker_rank_uses_explicit_rank_including_zero() {
        let worker = resolve_pinned_worker_rank(7, Some(0), Some(3)).unwrap();
        assert_eq!(worker.worker_id, 7);
        assert_eq!(worker.dp_rank, 0);
    }

    #[test]
    fn resolve_pinned_worker_rank_uses_unique_rank_when_unset() {
        let worker = resolve_pinned_worker_rank(7, None, Some(3)).unwrap();
        assert_eq!(worker.worker_id, 7);
        assert_eq!(worker.dp_rank, 3);
    }

    #[test]
    fn resolve_pinned_worker_rank_rejects_unresolved_rank() {
        let error = resolve_pinned_worker_rank(7, None, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires an explicit dp_rank"));
    }

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

    #[test]
    fn sticky_validation_style_ignores_transient_overload() {
        let worker = WorkerWithDpRank::new(7, 0);
        let configs = HashMap::from([(7, ModelRuntimeConfig::default())]);
        let constraints = RoutingConstraints::default();
        let overloaded = HashSet::from([7]);
        let scheduling_eligibility =
            RoutingEligibility::new(None, Some(&overloaded), Some(worker), &constraints);
        let sticky_eligibility = RoutingEligibility::new(None, None, Some(worker), &constraints);

        assert_eq!(
            scheduling_eligibility
                .validate_worker_rank(&configs, worker)
                .err(),
            Some(WorkerEligibilityError::WorkerOverloaded { worker_id: 7 })
        );
        assert!(
            sticky_eligibility
                .validate_worker_rank(&configs, worker)
                .is_ok()
        );
    }
}
