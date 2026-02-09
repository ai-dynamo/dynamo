// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use futures::StreamExt;
use rand::Rng;
use tokio::sync::{OwnedSemaphorePermit, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

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
    protocols::common::timing::{RequestPhase, RequestTracker, WORKER_TYPE_PREFILL},
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
    /// Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
    /// available in KV routing mode where the router has actual bookkeeping.
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

    /// Peek next worker without incrementing state (for non-KV modes only)
    fn peek_next_worker(&self) -> Option<u64> {
        match self {
            InnerPrefillRouter::SimpleRouter(router) => router.peek_next_worker(),
            InnerPrefillRouter::KvRouter(_) => None,
        }
    }
}

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
///
/// Modes:
/// - Query-only: `query_instance_id` annotation present → returns worker IDs without execution
/// - Pre-routed: `prefill_worker_id`/`decode_worker_id` set → routes to specified workers
/// - Normal: Worker IDs determined by router based on KV cache state
pub struct PrefillRouter {
    prefill_router: OnceLock<InnerPrefillRouter>,
    model_manager: Arc<ModelManager>,
    endpoint_id: OnceLock<EndpointId>,
    cancel_token: CancellationToken,
    router_mode: RouterMode,
    enforce_disagg: bool,
    /// Model name used to look up the worker monitor for prefill client registration
    model_name: String,
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
            model_name: String::new(), // Not used for disabled router
        })
    }

    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        enforce_disagg: bool,
        model_name: String,
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
            model_name,
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
            // Create KV chooser using the endpoint (this is a prefill router)
            let kv_chooser = model_manager
                .kv_chooser_for(
                    &endpoint,
                    kv_cache_block_size,
                    kv_router_config,
                    WORKER_TYPE_PREFILL,
                )
                .await?;

            // Extract client from kv_chooser to ensure shared state
            let client = kv_chooser.client().clone();

            // Register prefill client with worker monitor for TTFT metric cleanup in disaggregated mode
            if let Some(monitor) = model_manager.get_worker_monitor(&self.model_name) {
                monitor.set_prefill_client(client.clone());
            }

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

            // Register prefill client with worker monitor for TTFT metric cleanup in disaggregated mode
            if let Some(monitor) = model_manager.get_worker_monitor(&self.model_name) {
                monitor.set_prefill_client(client.clone());
            }

            // Create simple push router with the frontend's router mode
            // Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
            // available in KV routing mode where the router has actual bookkeeping.
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
            // Extract LORA name from routing hints
            let lora_name = req.routing.as_ref().and_then(|r| r.lora_name.clone());
            match async {
                kv_router
                    .chooser
                    .find_best_match(None, &req.token_ids, None, false, lora_name)
                    .await
            }
            .instrument(tracing::info_span!("kv_find_best_match"))
            .await
            {
                Ok((worker, _overlap)) => (worker.worker_id, worker.dp_rank),
                Err(_) => return None,
            }
        } else {
            // Non-KV mode: use PushRouter's stateful selection
            // We use peek_next_worker instead of select_next_worker to avoid double-incrementing the counter
            // if we fall back to the original path.
            let worker_id = prefill_router.peek_next_worker()?;
            (worker_id, 0)
        };

        // Get bootstrap info from ModelManager (works for ANY mode)
        let endpoint = self
            .model_manager
            .get_disaggregated_endpoint(endpoint_id, worker_id)?;
        let host = endpoint.bootstrap_host?;
        let port = endpoint.bootstrap_port?;

        let bootstrap_room: u64 = rand::rng().random();

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

    /// Execute prefill with the given router and extract structured result.
    ///
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization).
    ///
    /// If `phase_permit` is provided, it is dropped after the first output is received,
    /// allowing subsequent `set_phase` calls to proceed. This is used in the bootstrap
    /// optimization path to ensure `record_worker` completes before the phase changes.
    ///
    /// Returns (PrefillResult, Option<(worker_id, dp_rank)>).
    async fn execute_prefill(
        router: Option<InnerPrefillRouter>,
        request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
        phase_permit: Option<OwnedSemaphorePermit>,
    ) -> Result<(PrefillResult, Option<(u64, u32)>), PrefillError> {
        let router = router.ok_or(PrefillError::NotActivated)?;
        let mut prefill_response = router
            .generate_to_worker(request, target_worker)
            .await
            .map_err(|e| PrefillError::PrefillError(e.to_string()))?;

        // Drop phase permit now - routing is complete, record_worker was called in select_worker.
        // This unblocks set_phase(Decode) in the main task without waiting for prefill output.
        drop(phase_permit);

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

        // Extract prefill worker ID and dp_rank from disaggregated_params
        let prefill_worker_info =
            disaggregated_params
                .get("worker_id")
                .and_then(|worker_id_json| {
                    let worker_id = worker_id_json
                        .get("prefill_worker_id")
                        .and_then(|v| v.as_u64())?;
                    let dp_rank = worker_id_json
                        .get("prefill_dp_rank")
                        .and_then(|v| v.as_u64())
                        .map(|r| r as u32)
                        .unwrap_or(0);
                    Some((worker_id, dp_rank))
                });
        Ok((
            PrefillResult {
                disaggregated_params,
                prompt_tokens_details,
            },
            prefill_worker_info,
        ))
    }

    /// Spawn prefill as a background task.
    ///
    /// Uses direct routing to target_worker when specified (for non-KV modes with bootstrap optimization).
    ///
    /// The `phase_permit` is passed to the spawned task and dropped after the first output,
    /// allowing the main task's `set_phase(Decode)` to proceed.
    fn spawn_prefill_task(
        &self,
        prefill_request: SingleIn<PreprocessedRequest>,
        target_worker: Option<u64>,
        phase_permit: OwnedSemaphorePermit,
    ) {
        let router = self.prefill_router.get().cloned();
        // Capture current span to propagate trace context to the spawned task
        let span = tracing::Span::current();

        tokio::spawn(
            async move {
                match Self::execute_prefill(
                    router,
                    prefill_request,
                    target_worker,
                    Some(phase_permit),
                )
                .await
                {
                    Ok(_) => {
                        tracing::debug!("Prefill background task completed");
                    }
                    Err(e) => {
                        tracing::warn!("Prefill background task error: {e:?}");
                    }
                }
            }
            .instrument(span),
        );
    }

    /// Call the prefill router and extract structured prefill result, worker ID, and dp_rank.
    ///
    /// This is the synchronous prefill path - we wait for prefill to complete before proceeding.
    /// No phase permit is needed since `record_worker` completes before we return.
    ///
    /// Returns (PrefillResult, Option<(worker_id, dp_rank)>).
    async fn call_prefill(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<(PrefillResult, Option<(u64, u32)>), PrefillError> {
        // For call_prefill path, routing is handled by the router itself (no direct routing needed)
        // No phase permit needed - we wait for completion before changing phase
        Self::execute_prefill(self.prefill_router.get().cloned(), request, None, None).await
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

        // Save original max_tokens for decode
        let original_max_tokens = req.stop_conditions.max_tokens;

        // If prefill router is not activated, skip directly to decode
        if self.prefill_router.get().is_none() {
            if self.enforce_disagg {
                return Err(anyhow::anyhow!(PrefillError::NotActivated));
            }
            return next.generate(context.map(|_| req)).await;
        }

        // Ensure tracker exists for routing decisions in disaggregated mode.
        // Create one if not provided by the upstream DeltaGenerator.
        if req.tracker.is_none() {
            req.tracker = Some(Arc::new(RequestTracker::new()));
        }
        let tracker = req.tracker.as_ref().unwrap();
        let prefill_phase_permit = tracker.set_phase(RequestPhase::Prefill).await;
        tracker.record_prefill_start();

        // Prepare prefill request with max_tokens = 1 (clone after tracker is set)
        let mut prefill_req = req.clone();
        prefill_req.stop_conditions.max_tokens = Some(1);

        // Try build_bootstrap_info optimization: if we can get bootstrap info upfront,
        // spawn prefill in background and proceed to decode immediately.
        let preselected_worker = prefill_req
            .routing
            .as_ref()
            .and_then(|r| r.prefill_worker_id);

        let prefill_result = async {
            if let Some((worker_id, dp_rank, bootstrap_info)) = self
                .build_bootstrap_info(&prefill_req, preselected_worker)
                .await
            {
                // Bootstrap optimization path: spawn prefill in background
                // We successfully used the peeked worker, so we must now advance the router state
                // to ensure the next request gets a different worker.
                if !self.router_mode.is_kv_routing()
                    && let Some(router) = self.prefill_router.get()
                {
                    router.select_next_worker();
                }

                // Record prefill worker on the main request's tracker for metrics.
                // (The cloned prefill_req has its own tracker, so we need to record here)
                // Worker type is stored at routing time to avoid expensive MDC lookups when
                // updating Prometheus TTFT metrics later in the response stream.
                if let Some(ref tracker) = req.tracker {
                    tracker.record_prefill_worker_full(worker_id, dp_rank, WORKER_TYPE_PREFILL);
                }

                let routing = prefill_req.routing_mut();
                routing.prefill_worker_id = Some(worker_id);
                routing.dp_rank = Some(dp_rank);
                prefill_req.bootstrap_info = Some(bootstrap_info.clone());

                let prefill_context = Context::with_id(prefill_req, request_id.clone());
                engine_ctx.link_child(prefill_context.context());

                // Pass phase permit to spawned task - it drops after first output (record_worker complete)
                // This allows set_phase(Decode) below to proceed only after prefill routing is done
                self.spawn_prefill_task(prefill_context, Some(worker_id), prefill_phase_permit);

                Ok((None, Some(worker_id), Some(bootstrap_info)))
            } else {
                // Original prefill path: wait for prefill to complete
                tracing::debug!("Using original prefill path");

                // Drop the phase permit before calling call_prefill - we wait for completion
                // so there's no race with set_phase(Decode) below
                drop(prefill_phase_permit);

                let prefill_context = Context::with_id(prefill_req, request_id.clone());
                engine_ctx.link_child(prefill_context.context());

                let result = self.call_prefill(prefill_context).await;

                // Record prefill worker on the main request's tracker for metrics.
                // (call_prefill returns the worker_id and dp_rank from the prefill routing)
                // Worker type is stored at routing time to avoid expensive MDC lookups when
                // updating Prometheus TTFT metrics later in the response stream.
                if let Ok((_, Some((worker_id, dp_rank)))) = &result
                    && let Some(ref tracker) = req.tracker
                {
                    tracker.record_prefill_worker_full(*worker_id, *dp_rank, WORKER_TYPE_PREFILL);
                }

                result.map(|(result, worker_info)| {
                    (Some(result), worker_info.map(|(id, _)| id), None)
                })
            }
        }
        .instrument(tracing::info_span!("prefill_routing"))
        .await;

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

                // Set phase to Decode for the decode request.
                // In bootstrap path, this blocks until the spawned prefill task drops its permit
                // (after first output / record_worker completes), ensuring correct phase for routing.
                if let Some(ref tracker) = req.tracker {
                    let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
                    // Permit is dropped immediately - decode proceeds, no need to hold it
                }

                let mut decode_req = req;

                // Update request with prefill result
                if let Some(prefill_result) = maybe_prefill_result {
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
