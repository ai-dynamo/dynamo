// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use futures::{StreamExt, stream};
use serde_json::json;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Endpoint,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Context, ManyOut, Operator, PushRouter,
        ResponseStream, RouterMode, ServerStreamingEngine, SingleIn, async_trait,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use crate::{
    discovery::ModelManager,
    kv_router::{KvPushRouter, KvRouterConfig, RouterConfigOverride},
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
    protocols::common::preprocessor::PrefillResult,
};

/// Errors that can occur during prefill routing
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    /// Prefill router has not been activated yet
    #[error("Prefill router not yet activated")]
    NotActivated,

    /// Error during prefill execution
    /// TODO: Separate prefill worker error from prefill router error
    #[error("Prefill execution failed: {0}")]
    PrefillError(String),

    /// Disaggregated params not found in prefill response
    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

/// Result from calling the prefill router
/// Either just a worker ID (query only) or full prefill result with worker ID
enum PrefillCallResult {
    /// Query only mode: just the selected worker ID.
    /// This is used for GAIE to communicate the choices to the gateway during scheduling stage.
    WorkerIdOnly(u64),
    /// This is Dynamo Standard mode of operation. Used to pass the info from the prefill to decode worker.
    Full(PrefillResult),
}

/// The inner router used by PrefillRouter
enum InnerPrefillRouter {
    /// KV-aware routing using KvPushRouter
    KvRouter(Arc<KvPushRouter>),
    /// Simple routing (RoundRobin, Random, Direct)
    SimpleRouter(Arc<PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>>),
}

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
pub struct PrefillRouter {
    prefill_router: OnceLock<InnerPrefillRouter>,
    cancel_token: CancellationToken,
    router_mode: RouterMode,
    enforce_disagg: bool,
}

impl PrefillRouter {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(router_mode: RouterMode, enforce_disagg: bool) -> Arc<Self> {
        Arc::new(Self {
            prefill_router: OnceLock::new(),
            cancel_token: CancellationToken::new(),
            router_mode,
            enforce_disagg,
        })
    }

    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        enforce_disagg: bool,
    ) -> Arc<Self> {
        let prefill_router = OnceLock::new();
        let cancel_token = CancellationToken::new();

        let router = Arc::new(Self {
            prefill_router,
            cancel_token: cancel_token.clone(),
            router_mode,
            enforce_disagg,
        });

        // Spawn background task to wait for activation
        let router_clone = router.clone();
        tokio::spawn(async move {
            tokio::select! {
                result = activation_rx => {
                    let Ok(endpoint) = result else {
                        tracing::debug!("Prefill router activation channel closed without receiving endpoint");
                        return;
                    };

                    if let Err(e) = router_clone.activate(
                        endpoint,
                        model_manager,
                        kv_cache_block_size,
                        kv_router_config,
                    ).await {
                        tracing::error!(error = %e, "Failed to activate prefill router");
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Prefill router activation cancelled");
                }
            }
        });

        router
    }

    /// Activate the prefill router with the provided endpoint
    async fn activate(
        &self,
        endpoint: Endpoint,
        model_manager: Arc<ModelManager>,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
    ) -> Result<()> {
        tracing::info!(
            router_mode = ?self.router_mode,
            "Activating prefill router"
        );

        let inner_router = if self.router_mode.is_kv_routing() {
            // Create KV chooser using the endpoint
            let kv_chooser = model_manager
                .kv_chooser_for(&endpoint, kv_cache_block_size, kv_router_config)
                .await?;

            // Extract client from kv_chooser to ensure shared state
            let client = kv_chooser.client().clone();

            // Build the PushRouter for prefill with KV mode using the shared client
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
                client,
                RouterMode::KV,
                None, // busy_threshold
                None, // worker_monitor
            )
            .await?;

            // Wrap it in KvPushRouter
            InnerPrefillRouter::KvRouter(Arc::new(KvPushRouter::new(push_router, kv_chooser)))
        } else {
            // Create client for simple router
            let client = endpoint.client().await?;

            // Create simple push router with the frontend's router mode
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
                client,
                self.router_mode,
                None, // busy_threshold
                None, // worker_monitor
            )
            .await?;

            InnerPrefillRouter::SimpleRouter(Arc::new(push_router))
        };

        // Set the router (ignore error if already set)
        let _ = self.prefill_router.set(inner_router);

        tracing::info!(
            router_mode = ?self.router_mode,
            "Prefill router activated successfully"
        );

        Ok(())
    }

    /// Call the prefill router with optional query-only mode (GAIE)
    /// Query - only mode: only get the prefill worker id for the
    /// Default Dynamo mode  - extract structured prefill result in addition to getting the prefill worker
    async fn call_prefill(
        &self,
        request: SingleIn<PreprocessedRequest>,
        query_only: bool,
    ) -> Result<PrefillCallResult, PrefillError> {
        // Get the prefill router, error if not activated
        let Some(prefill_router) = self.prefill_router.get() else {
            return Err(PrefillError::NotActivated);
        };

        // GAIE Query-only mode: select worker without executing prefill
        if query_only {
            match prefill_router {
                InnerPrefillRouter::KvRouter(router) => {
                    // Add query_instance_id annotation to get worker ID without routing
                    let (mut req, ctx) = request.into_parts();
                    req.annotations.push("query_instance_id".to_string());
                    let query_request = ctx.map(|_| req);

                    let mut response = router
                        .generate(query_request)
                        .await
                        .map_err(|e| PrefillError::PrefillError(e.to_string()))?;

                    // Parse worker_instance_id from annotation response
                    while let Some(item) = response.next().await {
                        if let Some(event) = item.event.as_ref()
                            && event == "worker_instance_id"
                            && let Some(comments) = item.comment.as_ref()
                            && let Some(first_comment) = comments.first()
                            && let Ok(id_str) = serde_json::from_str::<String>(first_comment)
                            && let Ok(worker_id) = id_str.parse::<u64>()
                        {
                            return Ok(PrefillCallResult::WorkerIdOnly(worker_id));
                        }
                    }
                    return Err(PrefillError::PrefillError(
                        "Failed to get prefill worker ID from query_instance_id response"
                            .to_string(),
                    ));
                }
                InnerPrefillRouter::SimpleRouter(router) => {
                    // SimpleRouter's generate() doesn't support query-only mode,
                    // so get available workers directly and pick first one
                    let instance_ids = router.client.instance_ids();
                    if instance_ids.is_empty() {
                        return Err(PrefillError::PrefillError(
                            "No prefill workers available".to_string(),
                        ));
                    }
                    return Ok(PrefillCallResult::WorkerIdOnly(instance_ids[0]));
                }
            }
        }

        // Standard Dynamo prefill mode: call generate and extract PrefillResult
        let mut prefill_response = match prefill_router {
            InnerPrefillRouter::KvRouter(router) => router
                .generate(request)
                .await
                .map_err(|e| PrefillError::PrefillError(e.to_string()))?,
            InnerPrefillRouter::SimpleRouter(router) => router
                .generate(request)
                .await
                .map_err(|e| PrefillError::PrefillError(e.to_string()))?,
        };

        let Some(first_output) = prefill_response.next().await else {
            return Err(PrefillError::PrefillError(
                "Prefill router returned no output (stream ended)".to_string(),
            ));
        };

        let mut prompt_tokens_details = first_output
            .data
            .as_ref()
            .and_then(|o| o.completion_usage.as_ref())
            .and_then(|u| u.prompt_tokens_details.clone());

        while let Some(next) = prefill_response.next().await {
            if let Some(o) = next.data.as_ref()
                && prompt_tokens_details.is_none()
            {
                prompt_tokens_details = o
                    .completion_usage
                    .as_ref()
                    .and_then(|u| u.prompt_tokens_details.clone());
            }
        }

        if let Some(err) = first_output.err() {
            return Err(PrefillError::PrefillError(format!(
                "Prefill router returned error in output: {err:?}"
            )));
        }

        let Some(output) = &first_output.data else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output has no data field".to_string(),
            ));
        };

        let Some(disaggregated_params) = output.disaggregated_params.clone() else {
            return Err(PrefillError::NoDisaggregatedParams(
                "Prefill router output missing disaggregated_params".to_string(),
            ));
        };

        Ok(PrefillCallResult::Full(PrefillResult {
            disaggregated_params,
            prompt_tokens_details,
        }))
    }
}

impl Drop for PrefillRouter {
    fn drop(&mut self) {
        tracing::debug!("Dropping PrefillRouter, cancelling background activation task");
        self.cancel_token.cancel();
    }
}

#[async_trait]
impl
    Operator<
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
        SingleIn<PreprocessedRequest>,
        ManyOut<Annotated<LLMEngineOutput>>,
    > for PrefillRouter
{
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        // Extract request data while preserving context
        let (req, context) = request.into_parts();
        let request_id = context.id().to_string();
        let engine_ctx = context.context();

        // GAIE integration scheduling stage: figure out the prefill and decode workers only
        // If query_instance_id annotation is present, only return worker IDs without execution
        if req.has_annotation("query_instance_id") {
            tracing::debug!(
                request_id = %request_id,
                "GAIE Stage 1 (query_instance_id): Querying prefill and decode worker selection"
            );
            return self
                .select_prefill_decode_workers(req, context, next, &request_id, &engine_ctx)
                .await;
        }

        // Standard Dynamo Flow, same as GAIE stage 2 flow.
        // The only difference is that for the GAIE the workers IDs were already selected in stage 1.
        let target_prefill_worker = req.target_prefill_worker_id;
        let target_decode_worker = req.target_decode_worker_id;
        let has_target_workers = target_prefill_worker.is_some() && target_decode_worker.is_some();

        if has_target_workers {
            tracing::debug!(
                request_id = %request_id,
                prefill_worker = ?target_prefill_worker,
                decode_worker = ?target_decode_worker,
                "GAIE state 2: Using provided worker IDs for disaggregated serving"
            );
        }

        // Execute prefill and decode
        self.execute_prefill_and_decode(
            req,
            context,
            next,
            &request_id,
            &engine_ctx,
            target_prefill_worker,
            target_decode_worker,
        )
        .await
    }
}

impl PrefillRouter {
    /// Execute prefill and decode with optional target worker IDs
    /// Unified function for both GAIE Stage 2 (with target workers) and normal Dynamo flow
    #[allow(clippy::too_many_arguments)]
    async fn execute_prefill_and_decode(
        &self,
        req: PreprocessedRequest,
        context: Context<()>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        request_id: &str,
        engine_ctx: &Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>,
        target_prefill_worker: Option<u64>,
        target_decode_worker: Option<u64>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let original_max_tokens = req.stop_conditions.max_tokens;
        let has_target_workers = target_prefill_worker.is_some();

        // Prepare prefill request with max_tokens = 1
        let mut prefill_req = req.clone();
        prefill_req.stop_conditions.max_tokens = Some(1);

        // If target prefill worker specified, route directly to it
        if let Some(prefill_worker_id) = target_prefill_worker {
            prefill_req.backend_instance_id = Some(prefill_worker_id);
        }

        // Attempt prefill
        let prefill_context = Context::with_id(prefill_req, request_id.to_string());
        // Link the prefill context as a child so that kill signals propagate
        engine_ctx.link_child(prefill_context.context());
        let prefill_result = self.call_prefill(prefill_context, false).await;

        // Abort if cancelled during prefill
        if engine_ctx.is_stopped() || engine_ctx.is_killed() {
            tracing::debug!("Abort entering decode after context is stopped or killed");
            return Err(anyhow::anyhow!(
                "Context id {} is stopped or killed",
                engine_ctx.id()
            ));
        }

        // Handle prefill result
        match prefill_result {
            Ok(PrefillCallResult::Full(prefill_result)) => {
                tracing::debug!("Prefill succeeded, using disaggregated params for decode");
                Self::route_to_decode(
                    req,
                    context,
                    next,
                    prefill_result,
                    target_prefill_worker,
                    original_max_tokens,
                    target_decode_worker,
                )
                .await
            }
            Ok(PrefillCallResult::WorkerIdOnly(_)) => {
                // This shouldn't happen with query_only=false
                tracing::error!("Unexpected WorkerIdOnly result in prefill flow");
                if has_target_workers {
                    Err(anyhow::anyhow!("Unexpected WorkerIdOnly result"))
                } else {
                    next.generate(context.map(|_| req)).await
                }
            }
            Err(PrefillError::NotActivated) => {
                if self.enforce_disagg || has_target_workers {
                    tracing::error!(
                        "Prefill router not activated, but disaggregated mode is enforced. Failing request."
                    );
                    return Err(anyhow::anyhow!(PrefillError::NotActivated));
                }
                tracing::debug!("Prefill router not activated, falling back to decode-only");
                next.generate(context.map(|_| req)).await
            }
            Err(e) => {
                if self.enforce_disagg || has_target_workers {
                    tracing::error!(
                        error = %e,
                        "Remote prefill failed, but disaggregated mode is enforced. Failing request."
                    );
                    return Err(anyhow::anyhow!(e));
                }
                tracing::warn!(
                    error = %e,
                    "Remote prefill failed, falling back to decode-only. This may impact performance in disaggregated deployments. Verify prefill workers are healthy and accessible."
                );
                next.generate(context.map(|_| req)).await
            }
        }
    }

    /// Route to decode worker after prefill completes
    /// Common logic for both Dynamo normal flow and Stage 2 GAIE flow
    async fn route_to_decode(
        req: PreprocessedRequest,
        mut context: Context<()>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        prefill_result: PrefillResult,
        prefill_worker_id: Option<u64>,
        original_max_tokens: Option<u32>,
        target_decode_worker_id: Option<u64>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        let mut decode_req = req;

        // Update request with prefill result
        decode_req.prefill_result = Some(prefill_result);
        // Restore original max_tokens for decode
        decode_req.stop_conditions.max_tokens = original_max_tokens;

        // If target decode worker is specified, route directly to it
        if let Some(decode_worker_id) = target_decode_worker_id {
            decode_req.backend_instance_id = Some(decode_worker_id);
        }

        // Set router_config_override for decode: overlap_score_weight = 0
        let existing_override = decode_req.router_config_override.take();
        decode_req.router_config_override = Some(RouterConfigOverride {
            overlap_score_weight: Some(0.0),
            ..existing_override.unwrap_or_default()
        });

        // Store prefill worker ID in context if available
        if let Some(worker_id) = prefill_worker_id {
            context.insert("prefill_worker_id", worker_id);
        }

        // Map the modified request through with preserved context
        let decode_request = context.map(|_| decode_req);
        next.generate(decode_request).await
    }

    /// GAIE Stage 1: Select prefill and decode workers using KV-aware routing
    /// Returns worker IDs without executing prefill/decode.
    async fn select_prefill_decode_workers(
        &self,
        req: PreprocessedRequest,
        _context: Context<()>,
        next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        request_id: &str,
        engine_ctx: &Arc<dyn dynamo_runtime::pipeline::AsyncEngineContext>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        // Query prefill worker using KV-aware routing (query_only=true)
        let mut prefill_query_req = req.clone();
        prefill_query_req
            .annotations
            .push("query_instance_id".to_string());
        let query_context = Context::with_id(prefill_query_req, request_id.to_string());
        let prefill_worker_id = match self.call_prefill(query_context, true).await {
            Ok(PrefillCallResult::WorkerIdOnly(id)) => Some(id),
            Ok(PrefillCallResult::Full(_)) => {
                // Shouldn't happen with query_only=true
                tracing::error!("Unexpected Full result in query-only mode");
                None
            }
            Err(PrefillError::NotActivated) => {
                if self.enforce_disagg {
                    tracing::error!("Prefill router not activated for query stage");
                    return Err(anyhow::anyhow!(PrefillError::NotActivated));
                }
                None
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to query prefill worker");
                None
            }
        };

        // Query decode worker using next stage's router (with query_instance_id annotation)
        // Match Standard Dynamo behavior: ignore KV overlap for decode selection
        // since KV cache will be transferred from prefill worker
        let mut query_req = req.clone();
        query_req.annotations.push("query_instance_id".to_string());
        query_req.router_config_override = Some(RouterConfigOverride {
            overlap_score_weight: Some(0.0),
            ..Default::default()
        });
        let query_context = Context::with_id(query_req, request_id.to_string());
        engine_ctx.link_child(query_context.context());

        let mut decode_response = next.generate(query_context).await?;
        let mut decode_worker_id: Option<u64> = None;

        while let Some(item) = decode_response.next().await {
            if let Some(event) = item.event.as_ref()
                && event == "worker_instance_id"
                && let Some(comments) = item.comment.as_ref()
                && let Some(first_comment) = comments.first()
            {
                if let Ok(id_str) = serde_json::from_str::<String>(first_comment) {
                    decode_worker_id = id_str.parse().ok();
                }
                break;
            }
        }

        // Build response with selected worker IDs and tokens for Stage 2
        // Use nested worker_id structure to match standard Dynamo flow
        let query_response = LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: None,
            index: None,
            disaggregated_params: Some(json!({
                "worker_id": {
                    "prefill_worker_id": prefill_worker_id,
                    "decode_worker_id": decode_worker_id,
                },
                "token_data": req.token_ids,  // Include tokens for Stage 2 to skip re-tokenization
            })),
            extra_args: None,
            completion_usage: None,
        };

        tracing::debug!(
            request_id = %request_id,
            prefill_worker_id = ?prefill_worker_id,
            decode_worker_id = ?decode_worker_id,
            "Worker selection complete, returning IDs"
        );

        let response_stream = stream::once(async { Annotated::from_data(query_response) });
        Ok(ResponseStream::new(
            Box::pin(response_stream),
            engine_ctx.clone(),
        ))
    }
}
