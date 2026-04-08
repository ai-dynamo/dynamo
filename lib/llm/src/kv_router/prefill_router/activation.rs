// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::watch;

use dynamo_kv_router::{PrefillLoadEstimator, config::KvRouterConfig};
use dynamo_runtime::{
    component::{Client, Endpoint},
    pipeline::{PushRouter, RouterMode},
    protocols::annotated::Annotated,
};

use super::{InnerPrefillRouter, PrefillRouter};
use crate::{
    discovery::ModelManager,
    kv_router::KvPushRouter,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::WORKER_TYPE_PREFILL,
    },
};

impl PrefillRouter {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        enforce_disagg: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            prefill_router: std::sync::OnceLock::new(),
            model_manager,
            endpoint_id: std::sync::OnceLock::new(),
            cancel_token: tokio_util::sync::CancellationToken::new(),
            router_mode,
            enforce_disagg,
            prefill_load_estimator: None,
            model_name: String::new(),     // Not used for disabled router
            worker_set_key: String::new(), // Not used for disabled router
            is_eagle: false,
        })
    }

    #[expect(clippy::too_many_arguments)]
    pub fn new(
        mut activation_rx: watch::Receiver<Option<Endpoint>>,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        enforce_disagg: bool,
        model_name: String,
        worker_set_key: String,
        is_eagle: bool,
    ) -> Arc<Self> {
        let prefill_router = std::sync::OnceLock::new();
        let cancel_token = tokio_util::sync::CancellationToken::new();

        let router = Arc::new(Self {
            prefill_router,
            model_manager: model_manager.clone(),
            endpoint_id: std::sync::OnceLock::new(),
            cancel_token: cancel_token.clone(),
            router_mode,
            enforce_disagg,
            prefill_load_estimator,
            model_name,
            worker_set_key,
            is_eagle,
        });

        // Spawn background task to wait for the shared prefill endpoint.
        let router_clone = router.clone();
        tokio::spawn(async move {
            loop {
                let endpoint = { activation_rx.borrow().clone() };
                if let Some(endpoint) = endpoint {
                    if let Err(e) = router_clone
                        .activate(
                            endpoint,
                            model_manager.clone(),
                            kv_cache_block_size,
                            kv_router_config.clone(),
                            router_clone.prefill_load_estimator.clone(),
                        )
                        .await
                    {
                        tracing::error!(error = %e, "Failed to activate prefill router");
                    }
                    return;
                }

                tokio::select! {
                    result = activation_rx.changed() => {
                        if result.is_err() {
                            tracing::debug!("Prefill router activation channel closed without receiving endpoint");
                            return;
                        }
                    }
                    _ = cancel_token.cancelled() => {
                        tracing::debug!("Prefill router activation cancelled");
                        return;
                    }
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
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    ) -> Result<()> {
        tracing::info!(
            router_mode = ?self.router_mode,
            "Activating prefill router"
        );

        // Store endpoint_id for later use in resolve_prefill_worker
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
                    prefill_load_estimator,
                    WORKER_TYPE_PREFILL,
                    Some(self.model_name.clone()),
                    self.is_eagle,
                )
                .await?;

            // Extract client from kv_chooser to ensure shared state
            let client = kv_chooser.client().clone();
            self.register_prefill_client(model_manager.as_ref(), &client);

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
            self.register_prefill_client(model_manager.as_ref(), &client);

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

    fn register_prefill_client(&self, model_manager: &ModelManager, client: &Client) {
        if let Some(monitor) =
            model_manager.get_worker_monitor_for_worker_set(&self.model_name, &self.worker_set_key)
        {
            monitor.set_prefill_client(client.clone());
        }
    }
}
