// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, future::Future, sync::Arc};

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
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
    pipeline::{
        AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, Error, ManyOut, PushRouter,
        ResponseStream, SingleIn, async_trait,
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
        sticky::coordinator::{StickySessionCoordinator, sticky_allowed_for_phase},
    },
    preprocessor::PreprocessedRequest,
    protocols::{
        TokenIdType,
        common::{llm_backend::LLMEngineOutput, preprocessor::RoutingHints, timing::RequestPhase},
    },
};

mod request_guard;

use request_guard::RequestGuard;

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

fn cancelled_error(context_id: &str) -> Error {
    DynamoError::builder()
        .error_type(DynamoErrorType::Cancelled)
        .message(format!("Request {context_id} was cancelled"))
        .build()
        .into()
}

async fn cancel_on_stop<T>(
    context: &dyn AsyncEngineContext,
    context_id: &str,
    operation: impl Future<Output = T>,
) -> Result<T, Error> {
    tokio::pin!(operation);
    tokio::select! {
        biased;

        // Keep completed ownership-bearing results so their normal cleanup can run.
        result = &mut operation => Ok(result),
        _ = context.stopped() => Err(cancelled_error(context_id)),
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
        let request_context = request.context().clone();
        let mut selection_future = Box::pin(async {
            match self
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
                    Ok(selection)
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
                    .await
                }
                Err(error) => Err(error),
            }
        });
        let selection_result = tokio::select! {
            biased;

            _ = request_context.stopped() => None,
            result = &mut selection_future => Some(result),
        };
        drop(selection_future);

        let selection = match selection_result {
            Some(result) => result?,
            None => {
                if !is_query_only && let Err(error) = self.chooser.free(&context_id).await {
                    tracing::warn!(
                        request_id = %context_id,
                        %error,
                        "Failed to free scheduler state after cancellation during worker selection"
                    );
                }
                return Err(cancelled_error(&context_id));
            }
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

        // Tracked selection books scheduler state, so own its cleanup before any later await.
        let mut guard = RequestGuard::new(
            self.chooser.clone(),
            context_id.clone(),
            &request,
            scheduler_tracked,
        );

        if should_record {
            let worker = WorkerWithDpRank::new(instance_id, dp_rank);
            let record_result = if let Some(hashes) = routing_hashes {
                cancel_on_stop(
                    request_context.as_ref(),
                    &context_id,
                    self.chooser.record_routing_decision_hashes(hashes, worker),
                )
                .await?
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
                cancel_on_stop(
                    request_context.as_ref(),
                    &context_id,
                    self.chooser
                        .record_routing_decision(tokens_with_hashes, worker),
                )
                .await?
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
        if let Some(ref tracker) = request.tracker {
            let isl_blocks = routing_parts.token_ids.len().div_ceil(block_size);
            tracker.record_kv_hit(effective_overlap_blocks, isl_blocks);
            tracker.record_isl(routing_parts.token_ids.len(), Some(cached_tokens));
            tracker.record_worker(instance_id, Some(dp_rank), self.chooser.worker_type());
            tracker.record_router_queue_depth(self.chooser.pending_count());
            if let Some(hit_rate) = tracker.kv_hit_rate() {
                guard.request_metrics().kv_hit_rate.observe(hit_rate);
            }
        }
        guard
            .request_metrics()
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
        guard.start_dispatch(&phase_label);

        // Session lifecycle RPCs.
        // Fails fast if session_control.open is requested but the client can't be created.
        let worker = WorkerWithDpRank::new(instance_id, dp_rank);
        let route_outcome = cancel_on_stop(
            request_context.as_ref(),
            &context_id,
            self.sticky.on_routed(&request, worker, &context_id),
        )
        .await??;
        guard.set_deferred_close(route_outcome.deferred_close);

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = Some(dp_rank);
        let updated_request = context.map(|_| backend_input);

        // Record prefill start right before pushing to backend (OnceLock: first call wins).
        guard.record_prefill_start();

        let mut response_stream = cancel_on_stop(
            request_context.as_ref(),
            &context_id,
            self.inner
                .direct(updated_request, instance_id)
                .instrument(tracing::info_span!(
                    "kv_router.route_request",
                    request_id = %context_id,
                    worker_id = instance_id,
                    dp_rank = dp_rank,
                    overlap_blocks = overlap_amount,
                    phase = ?phase,
                )),
        )
        .await??;
        // direct() succeeded — mark dispatched so record_metrics() fires.
        // If direct() returned Err above, guard drops here with dispatched=false
        // → RequestGuard::Drop fires → chooser.free() + deferred_close.execute()
        //   but record_metrics() is suppressed (no backend work was done).
        guard.mark_dispatched();
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
    use std::{
        collections::{HashMap, HashSet},
        future::Future,
        pin::Pin,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        task::{Context, Poll},
    };

    use dynamo_kv_router::{
        protocols::{RoutingConstraints, WorkerWithDpRank},
        scheduling::{RoutingEligibility, WorkerEligibilityError},
    };
    use dynamo_runtime::{
        error::{DynamoError, ErrorType},
        pipeline::{AsyncEngineContext, context::Controller},
    };

    use super::{cancel_on_stop, pinned_worker_hint, resolve_pinned_worker_rank};
    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use crate::protocols::common::{preprocessor::RoutingHints, timing::RequestPhase};

    struct PendingUntilDropped(Arc<AtomicBool>);

    impl Future for PendingUntilDropped {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingUntilDropped {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn cancel_on_stop_drops_pending_operation() {
        let context = Controller::new("cancelled-request".to_string());
        context.stop();
        let dropped = Arc::new(AtomicBool::new(false));

        let error = cancel_on_stop(&context, context.id(), PendingUntilDropped(dropped.clone()))
            .await
            .unwrap_err();

        let error = error
            .downcast_ref::<DynamoError>()
            .expect("cancellation should return DynamoError");
        assert_eq!(error.error_type(), ErrorType::Cancelled);
        assert!(dropped.load(Ordering::SeqCst));
    }

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
