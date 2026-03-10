// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use arc_swap::ArcSwap;

use anyhow::Result;
use futures::StreamExt;
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
    kv_router::protocols::WorkerId,
    kv_router::{KvPushRouter, KvRouterConfig, RouterConfigOverride, protocols::BlockExtraInfo},
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

    /// TODO: Separate prefill worker error from prefill router error
    /// Error during prefill execution
    #[error("Prefill execution failed: {0}")]
    PrefillError(
        String,
        #[source] Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
    ),

    /// Disaggregated params not found in prefill response
    #[error("No disaggregated params in prefill response: {0}")]
    NoDisaggregatedParams(String),
}

/// Result of the prefill phase in `generate()`.
enum PrefillOutcome {
    /// Bootstrap optimization: prefill spawned in background, bootstrap info ready
    Bootstrap(BootstrapInfo),
    /// Synchronous prefill completed with result
    Completed(PrefillResult),
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
}

/// Lifecycle state of the prefill router in a disaggregated deployment.
///
/// Encoded as a single enum inside an `ArcSwap` so the full state machine is
/// visible at one glance — no need for a separate `AtomicBool` flag.
#[derive(Clone)]
enum PrefillState {
    /// Disagg not configured; the router is a permanent passthrough.
    /// Created by [`PrefillRouter::disabled()`].
    Passthrough,

    /// Decode worker registered, waiting for its prefill partner to appear.
    /// Created by [`PrefillRouter::new()`] before activation.
    AwaitingPrefill,

    /// Prefill workers discovered and healthy; disagg inference is active.
    /// Transitions from `AwaitingPrefill` when [`PrefillRouter::activate()`] succeeds.
    Active(InnerPrefillRouter),

    /// Prefill workers were discovered then removed; disagg is broken.
    /// Transitions from `Active` when [`PrefillRouter::deactivate()`] is called.
    Deactivated,
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
    state: ArcSwap<PrefillState>,
    model_manager: Arc<ModelManager>,
    endpoint_id: ArcSwap<Option<EndpointId>>,
    cancel_token: CancellationToken,
    router_mode: RouterMode,
    enforce_disagg: bool,
    /// Model name used to look up the worker monitor for prefill client registration
    model_name: String,
    /// Namespace used to look up the correct WorkerSet's worker monitor
    namespace: String,
}

impl PrefillRouter {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        enforce_disagg: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            state: ArcSwap::from_pointee(PrefillState::Passthrough),
            model_manager,
            endpoint_id: ArcSwap::from_pointee(None),
            cancel_token: CancellationToken::new(),
            router_mode,
            enforce_disagg,
            model_name: String::new(),
            namespace: String::new(),
        })
    }

    #[expect(clippy::too_many_arguments)]
    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        enforce_disagg: bool,
        model_name: String,
        namespace: String,
    ) -> Arc<Self> {
        let cancel_token = CancellationToken::new();

        let router = Arc::new(Self {
            state: ArcSwap::from_pointee(PrefillState::AwaitingPrefill),
            model_manager: model_manager.clone(),
            endpoint_id: ArcSwap::from_pointee(None),
            cancel_token: cancel_token.clone(),
            router_mode,
            enforce_disagg,
            model_name,
            namespace,
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

        self.endpoint_id.store(Arc::new(Some(endpoint.id())));

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
            if let Some(monitor) =
                model_manager.get_worker_monitor_for_namespace(&self.model_name, &self.namespace)
            {
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
            if let Some(monitor) =
                model_manager.get_worker_monitor_for_namespace(&self.model_name, &self.namespace)
            {
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

        self.state
            .store(Arc::new(PrefillState::Active(inner_router)));

        tracing::info!(
            router_mode = ?self.router_mode,
            "Prefill router activated successfully"
        );

        Ok(())
    }

    /// Deactivate the prefill router. Called when all prefill workers are removed.
    /// After deactivation, requests fall back to aggregated mode (or fail if enforce_disagg).
    pub fn deactivate(&self) {
        self.state.store(Arc::new(PrefillState::Deactivated));
        self.endpoint_id.store(Arc::new(None));
        tracing::info!(
            model_name = %self.model_name,
            namespace = %self.namespace,
            enforce_disagg = self.enforce_disagg,
            "Prefill router deactivated (prefill workers removed)"
        );
    }

    /// Whether this router can serve requests in its current state.
    /// - Passthrough / AwaitingPrefill / Active: true
    /// - Deactivated + enforce_disagg: false (disagg required but prefill is dead)
    /// - Deactivated + !enforce_disagg: true (falls back to aggregated)
    pub fn can_serve_requests(&self) -> bool {
        match &**self.state.load() {
            PrefillState::Passthrough | PrefillState::AwaitingPrefill | PrefillState::Active(_) => {
                true
            }
            PrefillState::Deactivated => !self.enforce_disagg,
        }
    }

    /// Return a snapshot of the current inner prefill router, if in the `Active` state.
    /// Cheap: `InnerPrefillRouter` holds `Arc`s internally, so the clone only bumps
    /// reference counts.
    fn current_router(&self) -> Option<InnerPrefillRouter> {
        match &**self.state.load() {
            PrefillState::Active(router) => Some(router.clone()),
            _ => None,
        }
    }

    /// Return a snapshot of the current endpoint ID, if set.
    fn current_endpoint_id(&self) -> Option<EndpointId> {
        (*self.endpoint_id.load_full()).clone()
    }

    /// Select a prefill worker and resolve its bootstrap connection info.
    /// If preselected_worker is provided (GAIE Stage 2), use it directly.
    /// Otherwise, query for the best worker (KV mode) or select next worker (non-KV modes).
    async fn resolve_prefill_worker(
        &self,
        req: &PreprocessedRequest,
        preselected_worker: Option<u64>,
    ) -> Option<(u64, u32, BootstrapInfo)> {
        let endpoint_id = self.current_endpoint_id()?;
        if !self.is_activated() {
            return None;
        }

        // Worker selection
        let (worker_id, dp_rank) = if let Some(id) = preselected_worker {
            let dp_rank = req.routing.as_ref().and_then(|r| r.dp_rank).unwrap_or(0);
            tracing::debug!(
                worker_id = id,
                dp_rank = dp_rank,
                "Using pre-selected prefill worker for bootstrap"
            );
            (id, dp_rank)
        } else {
            // Use shared worker selection logic (update_states=false for peek behavior)
            // Extract LORA name and priority jump from routing hints
            let lora_name = req.routing.as_ref().and_then(|r| r.lora_name.clone());
            let priority_jump = req
                .routing
                .as_ref()
                .and_then(|r| r.priority_jump)
                .unwrap_or(0.0);
            let allowed_worker_ids = req
                .routing
                .as_ref()
                .and_then(|r| r.allowed_worker_ids.clone());
            let (routing_token_ids, block_mm_infos) = req.block_mm_routing_info();
            match self
                .query_prefill_worker(
                    routing_token_ids,
                    block_mm_infos,
                    false,
                    lora_name,
                    priority_jump,
                    allowed_worker_ids,
                )
                .await
            {
                Ok((worker_id, dp_rank)) => (worker_id, dp_rank),
                Err(_) => return None,
            }
        };

        // Get bootstrap info from ModelManager (works for ANY mode)
        let endpoint = self
            .model_manager
            .get_disaggregated_endpoint(&endpoint_id, worker_id)?;
        let host = endpoint.bootstrap_host?;
        let port = endpoint.bootstrap_port?;

        let bootstrap_room: u64 = rand::random_range(0..=i64::MAX.cast_unsigned());

        tracing::debug!(
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
    /// optimization path to ensure `record_worker_full` completes before the phase changes.
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
            .map_err(|e| {
                PrefillError::PrefillError(
                    "failed to route to prefill worker".to_string(),
                    Some(e.into()),
                )
            })?;

        // Drop phase permit now - routing is complete, record_worker_full was called in select_worker.
        // This unblocks set_phase(Decode) in the main task without waiting for prefill output.
        drop(phase_permit);

        let Some(first_output) = prefill_response.next().await else {
            return Err(PrefillError::PrefillError(
                "Prefill router returned no output (stream ended)".to_string(),
                None,
            ));
        };

        if let Some(err) = first_output.err() {
            return Err(PrefillError::PrefillError(
                "Prefill router returned error in output".to_string(),
                Some(Box::new(err)),
            ));
        }

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
        let router = self.current_router();
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

    /// Query the best prefill worker without executing a request.
    /// Returns (worker_id, dp_rank).
    ///
    /// This is the shared worker selection logic used by both `resolve_prefill_worker`
    /// and `query_route`.
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
    ) -> Result<(u64, u32)> {
        let prefill_router = self
            .current_router()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match &prefill_router {
            InnerPrefillRouter::KvRouter(r) => {
                let (worker, _overlap) = r
                    .chooser
                    .find_best_match(
                        None,
                        token_ids,
                        block_mm_infos,
                        None,
                        update_states,
                        lora_name,
                        priority_jump,
                        None,
                        allowed_worker_ids,
                    )
                    .await?;
                Ok((worker.worker_id, worker.dp_rank))
            }
            InnerPrefillRouter::SimpleRouter(r) => {
                let worker_id = if update_states {
                    r.select_next_worker()
                } else {
                    r.peek_next_worker()
                }
                .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
                Ok((worker_id, 0))
            }
        }
    }

    /// Check if disaggregated mode is currently active (prefill router activated)
    pub fn is_activated(&self) -> bool {
        matches!(&**self.state.load(), PrefillState::Active(_))
    }

    /// Whether this router was ever activated (prefill workers were discovered at some point)
    pub fn was_ever_activated(&self) -> bool {
        matches!(
            &**self.state.load(),
            PrefillState::Active(_) | PrefillState::Deactivated
        )
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

        // If prefill router is not activated (no prefill workers discovered or
        // prefill died), route directly to decode in aggregated mode.
        // With --enforce-disagg, fail instead of falling back.
        if !self.is_activated() {
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

        // Prepare prefill request with max_tokens = 1 (clone after tracker is set)
        let mut prefill_req = req.clone();
        prefill_req.stop_conditions.max_tokens = Some(1);

        // Try to resolve prefill worker upfront: if we can get bootstrap info early,
        // spawn prefill in background and proceed to decode immediately.
        let preselected_worker = prefill_req
            .routing
            .as_ref()
            .and_then(|r| r.prefill_worker_id);

        let prefill_result = async {
            if let Some((worker_id, dp_rank, bootstrap_info)) = self
                .resolve_prefill_worker(&prefill_req, preselected_worker)
                .await
            {
                // Bootstrap optimization path: spawn prefill in background
                // We successfully used the peeked worker, so we must now advance the router state
                // to ensure the next request gets a different worker.
                if !self.router_mode.is_kv_routing()
                    && let Some(ref router) = self.current_router()
                {
                    router.select_next_worker();
                }

                let routing = prefill_req.routing_mut();
                routing.prefill_worker_id = Some(worker_id);
                routing.dp_rank = Some(dp_rank);
                prefill_req.bootstrap_info = Some(bootstrap_info.clone());

                let prefill_context = Context::with_id(prefill_req, request_id.clone());
                engine_ctx.link_child(prefill_context.context());

                // Pass phase permit to spawned task - it drops after first output (record_worker_full complete)
                // This allows set_phase(Decode) below to proceed only after prefill routing is done
                self.spawn_prefill_task(prefill_context, Some(worker_id), prefill_phase_permit);

                Ok(PrefillOutcome::Bootstrap(bootstrap_info))
            } else {
                // Original prefill path: wait for prefill to complete
                tracing::debug!("Using original prefill path");

                // Drop the phase permit - we wait for completion
                // so there's no race with set_phase(Decode) below
                drop(prefill_phase_permit);

                let prefill_context = Context::with_id(prefill_req, request_id.clone());
                engine_ctx.link_child(prefill_context.context());

                // In Direct mode, pass preselected_worker so execute_prefill uses
                // router.direct() instead of router.generate() (which bails in Direct mode).
                let (result, _worker_info) = Self::execute_prefill(
                    self.current_router(),
                    prefill_context,
                    preselected_worker,
                    None,
                )
                .await?;

                Ok(PrefillOutcome::Completed(result))
            }
        }
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
            Ok(outcome) => {
                tracing::debug!("Prefill completed, proceeding to decode");

                // Set phase to Decode for the decode request.
                // In bootstrap path, this blocks until the spawned prefill task drops its permit
                // (after first output / record_worker_full completes), ensuring correct phase for routing.
                if let Some(ref tracker) = req.tracker {
                    let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
                    // Permit is dropped immediately - decode proceeds, no need to hold it
                }

                let mut decode_req = req;

                match outcome {
                    PrefillOutcome::Bootstrap(info) => {
                        decode_req.bootstrap_info = Some(info);
                    }
                    PrefillOutcome::Completed(result) => {
                        decode_req.prefill_result = Some(result);
                    }
                }

                // Restore original max_tokens for decode
                decode_req.stop_conditions.max_tokens = original_max_tokens;

                // Set router_config_override for decode:
                // - overlap_score_weight = 0 (no KV cache overlap scoring for decode)
                // - assume_kv_reuse = false (generate random hashes since decode workers
                //   may already have blocks cached from prefill transfer)
                let existing_override = decode_req.router_config_override.take();
                decode_req.router_config_override = Some(RouterConfigOverride {
                    overlap_score_weight: Some(0.0),
                    assume_kv_reuse: Some(false),
                    ..existing_override.unwrap_or_default()
                });

                // Map the modified request through with preserved context
                let decode_request = context.map(|_| decode_req);
                next.generate(decode_request).await
            }
            Err(PrefillError::NotActivated) => {
                tracing::error!("Prefill router not activated, failing request");
                Err(anyhow::anyhow!(PrefillError::NotActivated))
            }
            Err(e) => {
                tracing::error!(error = %e, "Remote prefill failed, failing request");
                Err(anyhow::anyhow!(e))
            }
        }
    }
}

#[cfg(test)]
impl PrefillRouter {
    /// Test helper: create a router that simulates "was activated (prefill workers appeared)
    /// but is now deactivated (prefill engine died)".  Used by tests in this crate.
    pub fn make_deactivated_for_test(enforce_disagg: bool) -> Arc<Self> {
        Arc::new(Self {
            state: ArcSwap::from_pointee(PrefillState::Deactivated),
            model_manager: Arc::new(crate::discovery::ModelManager::new()),
            endpoint_id: ArcSwap::from_pointee(None),
            cancel_token: CancellationToken::new(),
            router_mode: dynamo_runtime::pipeline::RouterMode::RoundRobin,
            enforce_disagg,
            model_name: "test-model".to_string(),
            namespace: "test-ns".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── can_serve_requests ────────────────────────────────────────────────────

    /// A brand-new router that was never activated should always allow serving
    /// (the deployment may be purely aggregated).
    #[test]
    fn test_can_serve_requests_never_activated() {
        let mm = Arc::new(crate::discovery::ModelManager::new());

        // enforce_disagg=true: even strict mode allows serving when router never activated
        let router = PrefillRouter::disabled(mm.clone(), RouterMode::RoundRobin, true);
        assert!(
            router.can_serve_requests(),
            "never-activated router must allow serving regardless of enforce_disagg"
        );

        // enforce_disagg=false: same expectation
        let router = PrefillRouter::disabled(mm, RouterMode::RoundRobin, false);
        assert!(router.can_serve_requests());
    }

    /// A router that was once active but has since been deactivated must refuse
    /// serving when enforce_disagg is true (no fallback to aggregated mode).
    #[test]
    fn test_can_serve_requests_deactivated_enforce_disagg() {
        let router = PrefillRouter::make_deactivated_for_test(true);
        assert!(
            !router.can_serve_requests(),
            "deactivated router with enforce_disagg must not allow serving"
        );
    }

    /// When enforce_disagg is false the router should still allow serving after
    /// deactivation — requests fall back to aggregated (decode-only) mode.
    #[test]
    fn test_can_serve_requests_deactivated_no_enforce() {
        let router = PrefillRouter::make_deactivated_for_test(false);
        assert!(
            router.can_serve_requests(),
            "deactivated router without enforce_disagg must allow serving (fallback)"
        );
    }

    // ── deactivate ────────────────────────────────────────────────────────────

    /// deactivate() transitions to `PrefillState::Deactivated` so
    /// is_activated() returns false, but was_ever_activated() keeps returning
    /// true so subsequent can_serve_requests() calls honour enforce_disagg.
    #[test]
    fn test_deactivate_clears_inner_router() {
        // Build a router in the Deactivated state and call deactivate() again
        // to verify it is idempotent and produces the right state.
        let router = PrefillRouter::make_deactivated_for_test(true);

        // Already deactivated (state = Deactivated).
        assert!(!router.is_activated(), "should not be activated");
        assert!(
            router.was_ever_activated(),
            "should record prior activation"
        );

        // Calling deactivate() again must not panic and must preserve state.
        router.deactivate();
        assert!(!router.is_activated());
        assert!(router.was_ever_activated());
    }

    // ── was_ever_activated ────────────────────────────────────────────────────

    /// A freshly-created disabled router should report was_ever_activated()=false.
    #[test]
    fn test_was_ever_activated_false_on_fresh_router() {
        let mm = Arc::new(crate::discovery::ModelManager::new());
        let router = PrefillRouter::disabled(mm, RouterMode::RoundRobin, false);
        assert!(
            !router.was_ever_activated(),
            "fresh disabled router must not claim prior activation"
        );
    }
}
