// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use crate::{
    discovery::ModelManager,
    kv_router::{KvPushRouter, KvRouterConfig, RouterConfigOverride},
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
    protocols::common::preprocessor::{BootstrapInfo, PrefillResult},
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
    /// Execute prefill generation through the underlying router
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
        match self {
            InnerPrefillRouter::KvRouter(router) => router.generate(request).await,
            InnerPrefillRouter::SimpleRouter(router) => router.generate(request).await,
        }
    }
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

    /// Generate a unique bootstrap room ID for disaggregated serving
    fn generate_bootstrap_room() -> u64 {
        rand::rng().random()
    }

    /// Query best worker upfront, build bootstrap_info, and spawn prefill in background
    async fn build_bootstrap_info(
        &self,
        req: &PreprocessedRequest,
    ) -> Option<(u64, u32, BootstrapInfo)> {
        let prefill_router = self.prefill_router.get()?;

        // Only works with KvRouter
        let kv_router = match prefill_router {
            InnerPrefillRouter::KvRouter(r) => r,
            InnerPrefillRouter::SimpleRouter(_) => return None,
        };

        // Query best worker without routing
        let (worker_id, dp_rank) = match kv_router
            .chooser
            .find_best_match(None, &req.token_ids, None, false)
            .await
        {
            Ok((worker, _overlap)) => (worker.worker_id, worker.dp_rank),
            Err(_) => return None,
        };

        // Look up bootstrap endpoint from discovery
        let endpoint = kv_router
            .chooser
            .get_disaggregated_endpoint(worker_id)
            .await?;
        let host = endpoint.bootstrap_host?;
        let port = endpoint.bootstrap_port?;

        let bootstrap_room = Self::generate_bootstrap_room();

        tracing::info!(
            worker_id = worker_id,
            dp_rank = dp_rank,
            bootstrap_host = %host,
            bootstrap_port = port,
            bootstrap_room = bootstrap_room,
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
    async fn execute_prefill(
        router: Option<InnerPrefillRouter>,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<(PrefillResult, Option<u64>), PrefillError> {
        let router = router.ok_or(PrefillError::NotActivated)?;
        let mut prefill_response = router
            .generate(request)
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
    fn spawn_prefill_task(&self, prefill_request: SingleIn<PreprocessedRequest>) {
        let router = self.prefill_router.get().cloned();

        tokio::spawn(async move {
            match Self::execute_prefill(router, prefill_request).await {
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
        Self::execute_prefill(self.prefill_router.get().cloned(), request).await
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

        // Save original max_tokens for decode
        let original_max_tokens = req.stop_conditions.max_tokens;

        // Prepare prefill request with max_tokens = 1
        let mut prefill_req = req.clone();
        prefill_req.stop_conditions.max_tokens = Some(1);

        // Try build_bootstrap_info optimization
        let prefill_result = if let Some((worker_id, dp_rank, bootstrap_info)) =
            self.build_bootstrap_info(&prefill_req).await
        {
            let bootstrap_room = bootstrap_info.bootstrap_room;

            // Prepare request with bootstrap_room and force routing to specific worker
            prefill_req.backend_instance_id = Some(worker_id);
            prefill_req.dp_rank = Some(dp_rank);
            let extra_args = prefill_req
                .extra_args
                .get_or_insert_with(|| serde_json::json!({}));
            if let Some(obj) = extra_args.as_object_mut() {
                obj.insert(
                    "bootstrap_room".to_string(),
                    serde_json::json!(bootstrap_room),
                );
            }

            let prefill_context = Context::with_id(prefill_req, request_id.clone());
            engine_ctx.link_child(prefill_context.context());

            self.spawn_prefill_task(prefill_context);

            Ok((None, Some(worker_id), Some(bootstrap_info)))
        } else {
            // Fallback to original: Wait for prefill to complete
            tracing::debug!("Using original prefill path");

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

                let mut decode_req = req;

                // Update request with prefill result if available (only in original path)
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
