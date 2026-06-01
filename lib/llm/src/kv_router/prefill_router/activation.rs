// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;
use tokio::sync::oneshot;

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
            model_name: String::new(), // Not used for disabled router
            namespace: String::new(),  // Not used for disabled router
            is_eagle: false,
            deactivated: std::sync::atomic::AtomicBool::new(false),
            activated: std::sync::atomic::AtomicBool::new(false),
        })
    }

    #[expect(clippy::too_many_arguments)]
    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        enforce_disagg: bool,
        model_name: String,
        namespace: String,
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
            namespace,
            is_eagle,
            deactivated: std::sync::atomic::AtomicBool::new(false),
            activated: std::sync::atomic::AtomicBool::new(false),
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
                        router_clone.prefill_load_estimator.clone(),
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
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                RouterMode::KV,
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
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                self.router_mode,
                None, // worker_monitor
            )
            .await?;

            InnerPrefillRouter::SimpleRouter(Arc::new(push_router))
        };

        // Set the router (ignore error if already set)
        let _ = self.prefill_router.set(inner_router);
        self.activated.store(true, Ordering::Release);

        tracing::info!(
            router_mode = ?self.router_mode,
            "Prefill router activated successfully"
        );

        Ok(())
    }

    fn register_prefill_client(&self, model_manager: &ModelManager, client: &Client) {
        if let Some(monitor) =
            model_manager.get_worker_monitor_for_namespace(&self.model_name, &self.namespace)
        {
            monitor.set_prefill_client(client.clone());
        }
    }

    // -- Prefill death handling --

    /// Deactivate the prefill router. Called when all prefill workers are removed.
    /// After deactivation, requests fall back to aggregated mode (or fail if enforce_disagg).
    /// The inner router is preserved so that when workers rejoin (same endpoint/discovery),
    /// the Client's discovery subscription picks them up automatically.
    pub fn deactivate(&self) {
        self.deactivated.store(true, Ordering::Release);
        tracing::info!(
            model_name = %self.model_name,
            namespace = %self.namespace,
            enforce_disagg = self.enforce_disagg,
            "Prefill router deactivated (all prefill workers removed)"
        );
    }

    /// Reactivate a deactivated router. Called when prefill workers rejoin.
    /// The inner router's Client re-discovers workers via its discovery subscription.
    ///
    /// Note: there is a brief race between flipping `deactivated=false` (making
    /// `can_serve_requests()` return true) and the Client actually rediscovering
    /// workers. Requests arriving in this window may fail at prefill resolution.
    /// This is bounded by discovery propagation time (typically sub-second).
    ///
    /// Also note: reactivation reuses the existing inner router built from the
    /// original endpoint. If prefill rejoins under a different endpoint identity
    /// (e.g., reconfigured deployment), the stale Client would not discover the
    /// new workers. This is acceptable for normal restart scenarios where the
    /// endpoint identity is stable.
    pub fn reactivate(&self) {
        self.deactivated.store(false, Ordering::Release);
        tracing::info!(
            model_name = %self.model_name,
            namespace = %self.namespace,
            "Prefill router reactivated (prefill workers rejoined)"
        );
    }

    /// Whether this router is currently deactivated (prefill workers died).
    pub fn is_deactivated(&self) -> bool {
        self.deactivated.load(Ordering::Acquire)
    }

    /// Whether this router can serve requests in its current state.
    /// - !enforce_disagg (aggregated passthrough): always servable unless deactivated
    /// - enforce_disagg: only servable when prefill has activated AND is not deactivated,
    ///   so a cold-started strict-disagg model isn't listed before prefill rendezvoused.
    pub fn can_serve_requests(&self) -> bool {
        if self.is_deactivated() {
            return !self.enforce_disagg;
        }

        if !self.enforce_disagg {
            return true;
        }

        self.activated.load(Ordering::Acquire)
    }

    /// Mark this router as activated for testing purposes.
    /// In production, `activate()` sets this flag when the inner router is populated.
    #[cfg(test)]
    pub(crate) fn mark_activated_for_test(&self) {
        self.activated.store(true, Ordering::Release);
    }
}
