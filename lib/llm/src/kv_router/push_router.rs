// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::protocols::{TokensWithHashes, WorkerWithDpRank};
use dynamo_runtime::{
    metrics::frontend_perf::{STAGE_ROUTE, StageGuard},
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
        KvRouter, metrics::RouterRequestMetrics, sticky::coordinator::StickySessionCoordinator,
    },
    preprocessor::PreprocessedRequest,
    protocols::common::{llm_backend::LLMEngineOutput, timing::RequestPhase},
};

mod cancellation;
mod request_guard;
mod selection;

use cancellation::{cancel_on_stop, cancelled_error};
use request_guard::RequestGuard;
use selection::{RoutingRequestParts, WorkerSelection};

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
    /// Sticky session routing. Lazily activated when requests carry session_control.
    pub(super) sticky: Arc<StickySessionCoordinator>,
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
        // and suppresses metrics; its nested RequestCleanup then frees scheduler
        // state and executes the deferred close action.
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
