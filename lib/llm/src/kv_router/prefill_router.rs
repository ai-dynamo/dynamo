// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use futures::StreamExt;
use rand::Rng;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Endpoint,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Context, ManyOut, Operator, PushRouter,
        RouterMode, ServerStreamingEngine, SingleIn, async_trait,
    },
    protocols::{EndpointId, annotated::Annotated, maybe_error::MaybeError},
};

use crate::{
    discovery::ModelManager,
    kv_router::{KvPushRouter, KvRouterConfig, RouterConfigOverride},
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
    protocols::common::preprocessor::{BootstrapInfo, PrefillResult},
    protocols::common::timing::{RequestPhase, RequestTracker},
    protocols::openai::nvext::WorkerIdInfo,
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

/// The inner router used by PrefillRouter
#[derive(Clone)]
enum InnerPrefillRouter {
    /// KV-aware routing using KvPushRouter
    KvRouter(Arc<KvPushRouter>),
    /// Simple routing (RoundRobin, Random, Direct)
    SimpleRouter(Arc<PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>>),
}

impl InnerPrefillRouter {
    /// Generate with optional direct routing to specific worker.
    /// For KvRouter, target_worker is ignored since prefill_worker_id is already set on the request.
    /// For SimpleRouter, target_worker triggers direct routing via router.direct().
    async fn generate_to_worker(
        &self,
        request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        match (self, target_worker) {
            // KvRouter: prefill_worker_id already set on request, KvPushRouter::select_worker uses it
            (InnerPrefillRouter::KvRouter(router), _) => router.generate(request).await,
            (InnerPrefillRouter::SimpleRouter(router), Some(worker_id)) => {
                router.direct(request, worker_id).await
            }
            (InnerPrefillRouter::SimpleRouter(router), None) => router.generate(request).await,
        }
    }

    /// Select next worker (for non-KV modes only)
    fn select_next_worker(&self) -> Option<u64> {
        match self {
            InnerPrefillRouter::SimpleRouter(router) => router.select_next_worker(),
            InnerPrefillRouter::KvRouter(_) => None,
        }
    }
}

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
///
/// Supports regular Dynamo and GAIE integrated mode via query_instance_id state machine:
/// - GAIE Stage 1: query_instance_id transitions "" -> "prefill" -> "decode", returns only worker IDs
/// - GAIE Stage 2: routing.prefill_worker_id/routing.decode_worker_id are set, full execution with specified workers
/// - Non-GAIE: like GAIE Stage 2 but the worker ids have to be determined.
pub struct PrefillRouter {
    prefill_router: OnceLock<InnerPrefillRouter>,
    model_manager: Arc<ModelManager>,
    endpoint_id: OnceLock<EndpointId>,
    cancel_token: CancellationToken,
    router_mode: RouterMode,
    enforce_disagg: bool,
}

impl PrefillRouter {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        enforce_disagg: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            prefill_router: OnceLock::new(),
            model_manager,
            endpoint_id: OnceLock::new(),
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
            model_manager: model_manager.clone(),
            endpoint_id: OnceLock::new(),
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

        // Store endpoint_id for later use in build_bootstrap_info
        let _ = self.endpoint_id.set(endpoint.id());

        // Start runtime config watcher for this endpoint (needed for get_disaggregated_endpoint)
        // This must be done before creating the router so bootstrap info is available
        model_manager
            .get_or_create_runtime_config_watcher(&endpoint)
            .await?;

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

    /// Generate a unique bootstrap room ID for disaggregated serving
    fn generate_bootstrap_room() -> u64 {
        rand::rng().random()
    }

    /// Build bootstrap_info for disaggregated serving
    /// If preselected_worker is provided (GAIE Stage 2), use it directly.
    /// Otherwise, query for the best worker (KV mode) or select next worker (non-KV modes).
    async fn build_bootstrap_info(
        &self,
        req: &PreprocessedRequest,
        preselected_worker: Option<u64>,
    ) -> Option<(u64, u32, BootstrapInfo)> {
        let endpoint_id = self.endpoint_id.get()?;
        let prefill_router = self.prefill_router.get()?;

        // Worker selection
        let (worker_id, dp_rank) = if let Some(id) = preselected_worker {
            // GAIE Stage 2: use pre-selected worker
            let dp_rank = req.routing.as_ref().and_then(|r| r.dp_rank).unwrap_or(0);
            tracing::debug!(
                worker_id = id,
                dp_rank = dp_rank,
                "Using pre-selected prefill worker for bootstrap"
            );
            (id, dp_rank)
        } else if self.router_mode.is_kv_routing() {
            // KV mode: use find_best_match
            let kv_router = match prefill_router {
                InnerPrefillRouter::KvRouter(r) => r,
                _ => return None,
            };
            match kv_router
                .chooser
                .find_best_match(None, &req.token_ids, None, false)
                .await
            {
                Ok((worker, _overlap)) => (worker.worker_id, worker.dp_rank),
                Err(_) => return None,
            }
        } else {
            // Non-KV mode: use PushRouter's stateful selection
            let worker_id = prefill_router.select_next_worker()?;
            (worker_id, 0)
        };

        // Get bootstrap info from ModelManager (works for ANY mode)
        let endpoint = self
            .model_manager
            .get_disaggregated_endpoint(endpoint_id, worker_id)?;
        let host = endpoint.bootstrap_host?;
        let port = endpoint.bootstrap_port?;

        let bootstrap_room = Self::generate_bootstrap_room();

        tracing::info!(
            worker_id = worker_id,
            dp_rank = dp_rank,
            bootstrap_host = %host,
            bootstrap_port = port,
            bootstrap_room = bootstrap_room,
            router_mode = ?self.router_mode,
            "Built bootstrap_info upfront before prefill"
        );

        Some((
            worker_id,
            dp_rank,
            BootstrapInfo {
                bootstrap_host: host,
                bootstrap_port: port,
                bootstrap_room,
            },
        ))
    }

    /// Execute prefill with the given router and extract structured result
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization)
    async fn execute_prefill(
        router: Option<InnerPrefillRouter>,
        request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
    ) -> Result<(PrefillResult, Option<u64>), PrefillError> {
        let router = router.ok_or(PrefillError::NotActivated)?;
        let mut prefill_response = router
            .generate_to_worker(request, target_worker)
            .await
            .map_err(|e| PrefillError::PrefillError(e.to_string()))?;

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

        // Extract prefill worker ID from disaggregated_params
        let prefill_worker_id = disaggregated_params
            .get("worker_id")
            .and_then(|worker_id_json| {
                worker_id_json
                    .get("prefill_worker_id")
                    .and_then(|v| v.as_u64())
            });
        Ok((
            PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
            },
            prefill_worker_id,
        ))
    }

    /// Spawn prefill as a background task
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization)
    fn spawn_prefill_task(
        &self,
        prefill_request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
    ) {
        let router = self.prefill_router.get().cloned();

        tokio::spawn(async move {
            match Self::execute_prefill(router, prefill_request, target_worker).await {
                Ok(_) => {
                    tracing::debug!("Prefill background task completed");
                }
                Err(e) => {
                    tracing::warn!("Prefill background task error: {e:?}");
                }
            }
        });
    }

    /// Call the prefill router and extract structured prefill result and worker ID
    async fn call_prefill(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<(PrefillResult, Option<u64>), PrefillError> {
        // For call_prefill path, routing is handled by the router itself (no direct routing needed)
        Self::execute_prefill(self.prefill_router.get().cloned(), request, None).await
    }
}

/// GAIE helper functions for preparing prefill requests
impl PrefillRouter {
    /// Prepare prefill request for GAIE flows
    /// - Stage 1: Sets query_instance_id:prefill annotation
    /// - Stage 2: Sets backend_instance_id to target prefill worker
    fn prepare_prefill_for_gaie(prefill_req: &mut PreprocessedRequest, is_gaie_stage1: bool) {
        if is_gaie_stage1 {
            // GAIE Stage 1: Set query_instance_id to "prefill" for prefill worker selection
            prefill_req
                .annotations
                .retain(|a| !a.starts_with("query_instance_id"));
            prefill_req
                .annotations
                .push(format!("query_instance_id:{}", RequestPhase::Prefill));
        } else if let Some(prefill_worker_id) = prefill_req
            .routing
            .as_ref()
            .and_then(|r| r.prefill_worker_id)
        {
            // GAIE Stage 2: Route to pre-selected prefill worker from the stage 1
            tracing::debug!(
                prefill_worker_id = prefill_worker_id,
                "GAIE Stage 2: Routing prefill to pre-selected worker"
            );
            prefill_req.routing_mut().backend_instance_id = Some(prefill_worker_id);
        }
    }

    /// Prepare decode request for GAIE Stage 1
    /// Extracts prefill_worker_id from prefill result and sets decode annotations
    fn prepare_decode_for_gaie_stage1(
        decode_req: &mut PreprocessedRequest,
        prefill_result: &PrefillResult,
    ) {
        let prefill_worker_id = prefill_result
            .disaggregated_params
            .get("worker_id")
            .and_then(|v| serde_json::from_value::<WorkerIdInfo>(v.clone()).ok())
            .and_then(|info| info.prefill_worker_id);

        if let Some(worker_id) = prefill_worker_id {
            decode_req
                .annotations
                .retain(|a| !a.starts_with("query_instance_id"));
            decode_req
                .annotations
                .push(format!("query_instance_id:{}", RequestPhase::Decode));
            decode_req
                .annotations
                .push(format!("prefill_worker_id:{worker_id}"));
        }
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
        let (mut req, context) = request.into_parts();
        let request_id = context.id().to_string();
        let engine_ctx = context.context();

        // GAIE Stage 1: the presence of the empty query_instance_id signals query-only mode
        // State machine: "" -> "prefill" -> "decode" (disagg) OR "" -> aggregated worker (agg fallback)
        let is_gaie_stage1 = req
            .get_annotation_value("query_instance_id")
            .is_some_and(|s| s.is_empty());

        // Save original max_tokens for decode
        let original_max_tokens = req.stop_conditions.max_tokens;

        // GAIE Stage 1: Check if prefill router is activated - if not, skip to decode
        if is_gaie_stage1 && self.prefill_router.get().is_none() {
            tracing::debug!("GAIE Stage 1: Prefill router not activated, skipping to decode");
            if self.enforce_disagg {
                return Err(anyhow::anyhow!(PrefillError::NotActivated));
            }
            // Fall back to decode-only
            return next.generate(context.map(|_| req)).await;
        }

        // Ensure tracker exists for routing decisions in disaggregated mode.
        // Create one if not provided by the upstream DeltaGenerator.
        if req.tracker.is_none() {
            req.tracker = Some(Arc::new(RequestTracker::new()));
        }
        let tracker = req.tracker.as_ref().unwrap();
        tracker.set_phase(RequestPhase::Prefill);
        tracker.record_prefill_start();

        // Prepare prefill request with max_tokens = 1 (clone after tracker is set)
        let mut prefill_req = req.clone();
        prefill_req.stop_conditions.max_tokens = Some(1);

        // Prepare prefill request for GAIE flows (Stage 1 or Stage 2)
        Self::prepare_prefill_for_gaie(&mut prefill_req, is_gaie_stage1);

        // Try build_bootstrap_info optimization (skip for GAIE Stage 1 which needs query-only flow)
        // For GAIE Stage 2, use prefill_worker_id if provided
        let preselected_worker = prefill_req
            .routing
            .as_ref()
            .and_then(|r| r.prefill_worker_id);

        let prefill_result = if !is_gaie_stage1
            && let Some((worker_id, dp_rank, bootstrap_info)) = self
                .build_bootstrap_info(&prefill_req, preselected_worker)
                .await
        {
            // Bootstrap optimization path: spawn prefill in background
            let routing = prefill_req.routing_mut();
            routing.prefill_worker_id = Some(worker_id);
            routing.backend_instance_id = Some(worker_id); // Route prefill to the SAME worker we got bootstrap_info from
            routing.dp_rank = Some(dp_rank);
            prefill_req.bootstrap_info = Some(bootstrap_info.clone());

            let prefill_context = Context::with_id(prefill_req, request_id.clone());
            engine_ctx.link_child(prefill_context.context());

            self.spawn_prefill_task(prefill_context, Some(worker_id));

            Ok((None, Some(worker_id), Some(bootstrap_info)))
        } else {
            // Original prefill path: wait for prefill to complete
            tracing::debug!(
                is_gaie_stage1 = is_gaie_stage1,
                "Using original prefill path"
            );

            let prefill_context = Context::with_id(prefill_req, request_id.clone());
            engine_ctx.link_child(prefill_context.context());

            self.call_prefill(prefill_context)
                .await
                .map(|(result, worker_id)| (Some(result), worker_id, None))
        };

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
            Ok((maybe_prefill_result, _prefill_worker_id, bootstrap_info)) => {
                tracing::debug!("Prefill completed, proceeding to decode");

                // Set phase to Decode for the decode request
                if let Some(ref tracker) = req.tracker {
                    tracker.set_phase(RequestPhase::Decode);
                }

                let mut decode_req = req;

                // Update request with prefill result
                if is_gaie_stage1 {
                    if let Some(ref prefill_result) = maybe_prefill_result {
                        Self::prepare_decode_for_gaie_stage1(&mut decode_req, prefill_result);
                    }
                } else if let Some(prefill_result) = maybe_prefill_result {
                    // Normal or GAIE Stage 2: Set prefill_result for decode
                    decode_req.prefill_result = Some(prefill_result);
                }

                // Restore original max_tokens for decode
                decode_req.stop_conditions.max_tokens = original_max_tokens;

                // Inject bootstrap_info for decode worker
                if let Some(info) = bootstrap_info {
                    decode_req.bootstrap_info = Some(info);
                }

                // Set router_config_override for decode: overlap_score_weight = 0
                let existing_override = decode_req.router_config_override.take();
                decode_req.router_config_override = Some(RouterConfigOverride {
                    overlap_score_weight: Some(0.0),
                    ..existing_override.unwrap_or_default()
                });

                // GAIE Stage 2: Route to pre-selected decode worker if specified
                if let Some(decode_worker_id) =
                    decode_req.routing.as_ref().and_then(|r| r.decode_worker_id)
                {
                    decode_req.routing_mut().backend_instance_id = Some(decode_worker_id);
                    tracing::debug!(
                        decode_worker_id = decode_worker_id,
                        "GAIE Stage 2: Routing decode to pre-selected worker"
                    );
                }

                // Map the modified request through with preserved context
                let decode_request = context.map(|_| decode_req);
                next.generate(decode_request).await
            }
            Err(PrefillError::NotActivated) => {
                if self.enforce_disagg {
                    tracing::error!(
                        "Prefill router not activated, but disaggregated mode is enforced. Failing request."
                    );
                    return Err(anyhow::anyhow!(PrefillError::NotActivated));
                }
                tracing::debug!("Prefill router not activated, falling back to decode-only");
                next.generate(context.map(|_| req)).await
            }
            Err(e) => {
                if self.enforce_disagg {
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
}
