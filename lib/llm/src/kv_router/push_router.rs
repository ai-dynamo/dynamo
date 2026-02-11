// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_runtime::{
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        RouterMode, SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
    protocols::maybe_error::MaybeError,
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{
    kv_router::{
        KvRouter,
        metrics::RouterRequestMetrics,
        protocols::{TokensWithHashes, WorkerWithDpRank},
    },
    preprocessor::PreprocessedRequest,
    protocols::common::{llm_backend::LLMEngineOutput, timing::RequestPhase},
};

/// Response from the worker's cache_control service mesh endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheControlResponse {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pinned_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unpinned_count: Option<u32>,
}

impl MaybeError for CacheControlResponse {
    fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        CacheControlResponse {
            status: "error".to_string(),
            message: Some(err.to_string()),
            pinned_count: None,
            unpinned_count: None,
        }
    }

    fn err(&self) -> Option<anyhow::Error> {
        if self.status == "error" {
            Some(anyhow::anyhow!(
                "cache_control error: {}",
                self.message.as_deref().unwrap_or("unknown")
            ))
        } else {
            None
        }
    }
}

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
    cache_control_client: Option<PushRouter<serde_json::Value, Annotated<CacheControlResponse>>>,
}

/// Result of worker selection containing instance ID, dp_rank, and overlap amount.
struct WorkerSelection {
    instance_id: u64,
    dp_rank: u32,
    overlap_amount: u32,
}

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
        cache_control_client: Option<
            PushRouter<serde_json::Value, Annotated<CacheControlResponse>>,
        >,
    ) -> Self {
        KvPushRouter {
            inner,
            chooser,
            cache_control_client,
        }
    }

    /// Select a worker for the request, either using a preselected worker or finding the best match.
    ///
    /// When `is_query_only` is false and `handle_local_updates` is true, this also registers
    /// the request with the scheduler via `add_request`.
    async fn select_worker(
        &self,
        context_id: &str,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        is_query_only: bool,
        handle_local_updates: bool,
    ) -> Result<WorkerSelection, Error> {
        let routing = request.routing.as_ref();
        let lora_name = routing.and_then(|r| r.lora_name.clone());
        let priority_jump = routing.and_then(|r| r.priority_jump).unwrap_or(0.0);
        let dp_rank = routing.and_then(|r| r.dp_rank).unwrap_or(0);
        let expected_output_tokens = routing.and_then(|r| r.expected_output_tokens);

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
                    &request.token_ids,
                    request.router_config_override.as_ref(),
                    !is_query_only,
                    lora_name,
                    priority_jump,
                )
                .await?;

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
            .get_overlap_blocks(&request.token_ids, worker)
            .await?;

        if !is_query_only && handle_local_updates {
            self.chooser
                .add_request(
                    context_id.to_string(),
                    &request.token_ids,
                    overlap_blocks,
                    expected_output_tokens,
                    worker,
                    lora_name,
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

        // Determine if this router should handle local state updates (add_request, free, etc.)
        // Default is true (router handles bookkeeping). Set to false for GAIE Stage 2 where
        // an external orchestrator (e.g., EPP sidecar) handles bookkeeping via C FFI.
        let handle_local_updates = request
            .routing
            .as_ref()
            .and_then(|r| r.enable_local_updates)
            .unwrap_or(true);

        // Get phase from tracker (defaults to Aggregated if no tracker or phase not set)
        let phase = request
            .tracker
            .as_ref()
            .map(|t| t.phase())
            .unwrap_or(RequestPhase::Aggregated);

        let block_size = self.chooser.block_size() as usize;
        let selection = self
            .select_worker(
                &context_id,
                &request,
                phase,
                is_query_only,
                handle_local_updates,
            )
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
            let isl_blocks = request.token_ids.len().div_ceil(block_size);
            tracker.record_kv_hit(overlap_amount, isl_blocks);
            tracker.record_isl(
                request.token_ids.len(),
                overlap_amount as usize * block_size,
            );
            tracker.record_worker_full(instance_id, dp_rank, self.chooser.worker_type());
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
        let track_output_blocks =
            self.chooser.kv_router_config().router_track_output_blocks && handle_local_updates;
        let tracker = request.tracker.clone();

        // Extract PIN hint before consuming the request
        let pin_prefix = request
            .routing
            .as_ref()
            .and_then(|r| r.pin)
            .unwrap_or(false);
        let token_ids_for_pin = if pin_prefix {
            Some(request.token_ids.clone())
        } else {
            None
        };
        let cc_client = if pin_prefix {
            self.cache_control_client.clone()
        } else {
            None
        };

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = Some(dp_rank);
        let updated_request = context.map(|_| backend_input);

        // Record prefill start right before pushing to backend (OnceLock: first call wins).
        if let Some(ref tracker) = tracker {
            tracker.record_prefill_start();
        }

        let chooser = self.chooser.clone();
        let mut response_stream = self.inner.direct(updated_request, instance_id).await?;
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();

        // Wrap stream with lifecycle management (mark_prefill_completed, free)
        // Only perform these operations if handle_local_updates is true.
        // When false, an external caller (e.g., GAIE sidecar) handles bookkeeping via C FFI.
        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut prefill_marked = false;
            let mut first_token_recorded = false;

            // Output block tracking state
            let mut cumulative_osl: usize = 0;
            let mut current_total_blocks = isl_tokens.div_ceil(block_size);

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

                        if handle_local_updates && !prefill_marked {
                            // Only mark prefill completed when we receive actual tokens,
                            // not empty bootstrap info (token_ids: []) from disaggregated prefill
                            let has_tokens = item.data.as_ref()
                                .map(|d| !d.token_ids.is_empty())
                                .unwrap_or(false);
                            if has_tokens {
                                if let Err(e) = chooser.mark_prefill_completed(&context_id).await {
                                    tracing::warn!("Failed to mark prefill completed for request {context_id}: {e}");
                                }
                                prefill_marked = true;
                            }
                        }

                        let new_tokens = item.data.as_ref()
                            .map(|d| d.token_ids.len())
                            .unwrap_or(0);

                        // Record first token time on tracker when actual tokens arrive
                        if !first_token_recorded && new_tokens > 0 {
                            if let Some(ref tracker) = tracker {
                                tracker.record_first_token();
                                if let Some(ttft) = tracker.ttft_ms() {
                                    request_metrics
                                        .time_to_first_token_seconds
                                        .observe(ttft / 1000.0);
                                }
                            }
                            first_token_recorded = true;
                        }

                        cumulative_osl += new_tokens;

                        // Track output blocks if enabled
                        if track_output_blocks {
                            let new_total_blocks = (isl_tokens + cumulative_osl).div_ceil(block_size);
                            if new_total_blocks > current_total_blocks {
                                // New block boundary crossed - add output block with decay
                                // Clamp eot to min 1 to avoid division by zero, and result to min 0.0
                                let decay_fraction = expected_output_tokens.map(|eot| {
                                    (1.0 - (cumulative_osl as f64 / eot.max(1) as f64)).max(0.0)
                                });
                                if let Err(e) = chooser.add_output_block(&context_id, decay_fraction).await {
                                    tracing::warn!(
                                        "Failed to add output block for request {context_id}: {e}"
                                    );
                                }

                                // Update tracker and observe avg ITL at each block boundary
                                if let Some(ref tracker) = tracker {
                                    tracker.record_osl(cumulative_osl);
                                    tracker.record_finish();
                                    if let Some(avg_itl) = tracker.avg_itl_ms() {
                                        request_metrics
                                            .inter_token_latency_seconds
                                            .observe(avg_itl / 1000.0);
                                    }
                                }

                                current_total_blocks = new_total_blocks;
                            }
                        }

                        yield item;
                    }
                }
            }

            // Record final aggregate metrics (histograms sampled once per request)
            if let Some(ref tracker) = tracker {
                tracker.record_finish();
                tracker.record_osl(cumulative_osl);

                request_metrics
                    .output_sequence_tokens
                    .observe(cumulative_osl as f64);
            }
            request_metrics.requests_total.inc();

            // Only call free() if we handle local updates.
            // When handle_local_updates=false, external caller handles cleanup via C FFI.
            if handle_local_updates
                && let Err(e) = chooser.free(&context_id).await
            {
                tracing::warn!("Failed to free request {context_id}: {e}");
            }

            // After stream completion, if pin hint was set, fire-and-forget pin_prefix
            // to the worker. This runs after all response tokens have been yielded.
            if pin_prefix {
                if let (Some(cc), Some(token_ids)) = (&cc_client, &token_ids_for_pin) {
                    let pin_request = serde_json::json!({
                        "action": "pin_prefix",
                        "token_ids": token_ids,
                    });
                    let cc_clone = cc.clone();
                    let pin_context_id = context_id.clone();
                    tokio::spawn(async move {
                        match cc_clone.direct(SingleIn::new(pin_request), instance_id).await {
                            Ok(mut stream) => {
                                if let Some(resp) = stream.next().await {
                                    tracing::debug!(
                                        request_id = %pin_context_id,
                                        worker_id = instance_id,
                                        ?resp,
                                        "pin_prefix response"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    request_id = %pin_context_id,
                                    worker_id = instance_id,
                                    "Failed to pin prefix: {e}"
                                );
                            }
                        }
                    });
                } else if cc_client.is_none() {
                    tracing::warn!(
                        request_id = %context_id,
                        "pin=true but no cache_control_client configured"
                    );
                }
            }
        });
        Ok(ResponseStream::new(wrapped_stream, stream_context))
    }
}
