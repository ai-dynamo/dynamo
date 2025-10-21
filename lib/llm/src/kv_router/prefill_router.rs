// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, OnceLock};

use anyhow::Result;
use futures::StreamExt;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    component::Endpoint,
    pipeline::{
        AsyncEngine, Context, ManyOut, Operator, PushRouter, RouterMode, ServerStreamingEngine,
        SingleIn, async_trait,
    },
    protocols::{annotated::Annotated, maybe_error::MaybeError},
};

use crate::{
    discovery::ModelManager,
    kv_router::{KvPushRouter, KvRouterConfig, RouterConfigOverride},
    protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
};

/// Error returned when prefill router is not yet activated
#[derive(Debug, thiserror::Error)]
#[error("Prefill router not yet activated")]
struct NoPrefillRouter;

/// PrefillRouter is a forward-only operator that sits between Migration and the decode router.
/// It optionally calls a prefill worker before routing to decode, extracting disaggregated_params
/// from the prefill response and injecting them into the decode request.
pub struct PrefillRouter {
    prefill_router: OnceLock<Arc<KvPushRouter>>,
    cancel_token: CancellationToken,
}

impl PrefillRouter {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            prefill_router: OnceLock::new(),
            cancel_token: CancellationToken::new(),
        })
    }

    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_manager: Arc<ModelManager>,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
    ) -> Arc<Self> {
        let prefill_router = OnceLock::new();
        let cancel_token = CancellationToken::new();

        let router = Arc::new(Self {
            prefill_router,
            cancel_token: cancel_token.clone(),
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
        tracing::info!("Activating prefill router");

        // Create KV chooser using the component from the endpoint
        let kv_chooser = model_manager
            .kv_chooser_for(endpoint.component(), kv_cache_block_size, kv_router_config)
            .await?;

        // Build the PushRouter for prefill
        let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_threshold(
            endpoint.client().await?,
            RouterMode::KV,
            None, // busy_threshold
            None, // worker_monitor
        )
        .await?;

        // Wrap it in KvPushRouter
        let prefill_router = Arc::new(KvPushRouter::new(push_router, kv_chooser));

        // Set the router (ignore error if already set)
        let _ = self.prefill_router.set(prefill_router);

        tracing::info!("Prefill router activated successfully");

        Ok(())
    }

    /// Call the prefill router and extract disaggregated_params
    async fn call_prefill(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<Option<serde_json::Value>> {
        // Get the prefill router, error if not activated
        let prefill_router = self.prefill_router.get().ok_or(NoPrefillRouter)?;

        // Call prefill router
        let mut prefill_response = match prefill_router.generate(request).await {
            Ok(response) => response,
            Err(e) => {
                tracing::warn!(error = %e, "Prefill router generate call failed");
                return Ok(None);
            }
        };

        let Some(first_output) = prefill_response.next().await else {
            tracing::debug!("Prefill router returned no output (stream ended)");
            return Ok(None);
        };

        if let Some(err) = first_output.err() {
            tracing::debug!(error = ?err, "Prefill router returned error in output");
            return Ok(None);
        }

        let Some(output) = &first_output.data else {
            tracing::debug!("Prefill router output has no data field");
            return Ok(None);
        };

        if output.disaggregated_params.is_none() {
            tracing::debug!("Prefill router output missing disaggregated_params");
        }

        Ok(output.disaggregated_params.clone())
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

        // Prepare prefill request (use fresh context for internal routing)
        let prefill_req = req.clone();
        let prefill_request = Context::with_id(prefill_req, request_id);

        // Attempt prefill and handle results
        match self.call_prefill(prefill_request).await {
            Ok(Some(disaggregated_params)) => {
                tracing::debug!("Prefill succeeded, using disaggregated params for decode");

                // Update request with disaggregated_params and router config
                let mut decode_req = req;
                decode_req.disaggregated_params = Some(disaggregated_params);

                // Set router_config_override for decode: overlap_score_weight = 0
                let existing_override = decode_req.router_config_override.take();
                decode_req.router_config_override = Some(RouterConfigOverride {
                    overlap_score_weight: Some(0.0),
                    ..existing_override.unwrap_or_default()
                });

                // Map the modified request through with preserved context
                let decode_request = context.map(|_| decode_req);
                return next.generate(decode_request).await;
            }
            Ok(None) => {
                tracing::debug!("Prefill returned None, falling back to decode-only");
            }
            Err(e) => {
                tracing::debug!(error = %e, "Prefill error, falling back to decode-only");
            }
        }

        // Fall back to decode-only
        next.generate(context.map(|_| req)).await
    }
}
