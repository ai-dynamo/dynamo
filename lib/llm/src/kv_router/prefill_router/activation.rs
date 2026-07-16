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

use super::{InnerPrefillRouter, PrefillLifecycleState, PrefillRouter};
use crate::{
    discovery::ModelManager,
    kv_router::KvPushRouter,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::WORKER_TYPE_PREFILL,
    },
    session_affinity::create_affinity_coordinator,
};

impl PrefillRouter {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        session_affinity_ttl_secs: Option<u64>,
    ) -> Arc<Self> {
        Arc::new(Self {
            prefill_router: std::sync::OnceLock::new(),
            model_manager,
            endpoint_id: std::sync::OnceLock::new(),
            cancel_token: tokio_util::sync::CancellationToken::new(),
            router_mode,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
            prefill_load_estimator: None,
            model_name: String::new(), // Not used for disabled router
            namespace: String::new(),  // Not used for disabled router
            is_eagle: false,
            lifecycle: std::sync::atomic::AtomicU8::new(PrefillLifecycleState::Pending as u8),
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
        session_affinity_ttl_secs: Option<u64>,
        session_affinity_replica_sync: bool,
        model_name: String,
        namespace: String,
        is_eagle: bool,
        worker_monitor: Option<crate::discovery::KvWorkerMonitor>,
    ) -> Arc<Self> {
        let prefill_router = std::sync::OnceLock::new();
        let cancel_token = tokio_util::sync::CancellationToken::new();

        let router = Arc::new(Self {
            prefill_router,
            model_manager: model_manager.clone(),
            endpoint_id: std::sync::OnceLock::new(),
            cancel_token: cancel_token.clone(),
            router_mode,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
            prefill_load_estimator,
            model_name,
            namespace,
            is_eagle,
            lifecycle: std::sync::atomic::AtomicU8::new(PrefillLifecycleState::Pending as u8),
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
                        session_affinity_replica_sync,
                        worker_monitor.as_ref(),
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
        session_affinity_replica_sync: bool,
        worker_monitor: Option<&crate::discovery::KvWorkerMonitor>,
    ) -> Result<()> {
        tracing::info!(
            router_mode = ?self.router_mode,
            "Activating prefill router"
        );

        // Store endpoint metadata for bootstrap and topology preparation.
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
            Self::attach_prefill_client(worker_monitor, &client);
            let affinity = create_affinity_coordinator(
                self.session_affinity_ttl,
                client.clone(),
                session_affinity_replica_sync,
            )
            .await?;

            // Build the PushRouter for prefill with KV mode using the shared client
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                RouterMode::KV,
                None, // worker_monitor
            )
            .await?;

            // Wrap it in KvPushRouter
            InnerPrefillRouter::KvRouter(Arc::new(KvPushRouter::new_with_coordinator(
                push_router,
                kv_chooser,
                affinity,
            )))
        } else {
            // Create client for simple router
            let client = endpoint.client().await?;
            Self::attach_prefill_client(worker_monitor, &client);
            let affinity = create_affinity_coordinator(
                self.session_affinity_ttl,
                client.clone(),
                session_affinity_replica_sync,
            )
            .await?;

            // Create simple push router with the frontend's router mode
            // Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
            // available in KV routing mode where the router has actual bookkeeping.
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                self.router_mode,
                None, // worker_monitor
            )
            .await?;

            InnerPrefillRouter::SimpleRouter(Arc::new(
                crate::session_affinity::SessionAffinityPushRouter::new_with_coordinator(
                    push_router,
                    affinity,
                    self.router_mode.is_direct_routing(),
                ),
            ))
        };

        // Set the router (ignore error if already set).
        let _ = self.prefill_router.set(inner_router);
        match self.complete_activation() {
            PrefillLifecycleState::Active => {
                tracing::info!(
                    router_mode = ?self.router_mode,
                    "Prefill router activated successfully"
                );
            }
            PrefillLifecycleState::Unavailable => {
                tracing::info!(
                    router_mode = ?self.router_mode,
                    "Prefill router initialized after its workers became unavailable"
                );
            }
            PrefillLifecycleState::Pending => unreachable!("activation must leave pending state"),
        }

        Ok(())
    }

    pub(super) fn complete_activation(&self) -> PrefillLifecycleState {
        match self.lifecycle.compare_exchange(
            PrefillLifecycleState::Pending as u8,
            PrefillLifecycleState::Active as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => PrefillLifecycleState::Active,
            Err(current) => PrefillLifecycleState::from_atomic(current),
        }
    }

    /// Attach the freshly-created prefill `Client` to this WorkerSet's monitor (handed in
    /// at construction). The monitor then publishes the overloaded set to the prefill pool
    /// and watches the prefill endpoint for metric cleanup. No-op for a disabled router.
    fn attach_prefill_client(
        worker_monitor: Option<&crate::discovery::KvWorkerMonitor>,
        client: &Client,
    ) {
        if let Some(monitor) = worker_monitor {
            monitor.attach_prefill_client(client.clone());
        }
    }

    // -- Prefill death handling --

    /// Deactivate the prefill router. Called when all prefill workers are removed.
    /// After deactivation, requests fall back to aggregated mode.
    /// The inner router is preserved so that when workers rejoin (same endpoint/discovery),
    /// the Client's discovery subscription picks them up automatically.
    pub fn deactivate(&self) {
        let transition =
            self.lifecycle
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    match PrefillLifecycleState::from_atomic(current) {
                        PrefillLifecycleState::Pending | PrefillLifecycleState::Active => {
                            Some(PrefillLifecycleState::Unavailable as u8)
                        }
                        PrefillLifecycleState::Unavailable => None,
                    }
                });
        if transition.is_err() {
            return;
        }
        tracing::info!(
            model_name = %self.model_name,
            namespace = %self.namespace,
            "Prefill router deactivated (all prefill workers removed)"
        );
    }

    /// Reactivate a deactivated router. Called when prefill workers rejoin.
    /// The inner router's Client re-discovers workers via its discovery subscription.
    ///
    /// Note: there is a brief race between entering `Active` and the Client
    /// actually rediscovering workers. Requests arriving in this window may fail at prefill resolution.
    /// This is bounded by discovery propagation time (typically sub-second).
    ///
    /// Also note: reactivation reuses the existing inner router built from the
    /// original endpoint. If prefill rejoins under a different endpoint identity
    /// (e.g., reconfigured deployment), the stale Client would not discover the
    /// new workers. This is acceptable for normal restart scenarios where the
    /// endpoint identity is stable.
    pub fn reactivate(&self) {
        let initialized = self.prefill_router.get().is_some();
        let target = if initialized {
            PrefillLifecycleState::Active
        } else {
            PrefillLifecycleState::Pending
        };
        let transition = self.lifecycle.compare_exchange(
            PrefillLifecycleState::Unavailable as u8,
            target as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if let Err(current) = transition {
            PrefillLifecycleState::from_atomic(current);
            return;
        }
        let state =
            if target == PrefillLifecycleState::Pending && self.prefill_router.get().is_some() {
                self.complete_activation()
            } else {
                target
            };
        match state {
            PrefillLifecycleState::Active => {
                tracing::info!(
                    model_name = %self.model_name,
                    namespace = %self.namespace,
                    "Prefill router reactivated (prefill workers rejoined)"
                );
            }
            PrefillLifecycleState::Pending => {
                tracing::info!(
                    model_name = %self.model_name,
                    namespace = %self.namespace,
                    "Prefill workers rejoined before router initialization completed"
                );
            }
            PrefillLifecycleState::Unavailable => {}
        }
    }

    /// Whether this router is currently deactivated (prefill workers died).
    pub fn is_deactivated(&self) -> bool {
        self.lifecycle_state() == PrefillLifecycleState::Unavailable
    }

    /// Whether the inner router has initialized, even if workers are unavailable.
    pub fn is_activated(&self) -> bool {
        self.prefill_router.get().is_some()
    }

    pub(super) fn lifecycle_state(&self) -> PrefillLifecycleState {
        PrefillLifecycleState::from_atomic(self.lifecycle.load(Ordering::Acquire))
    }

    /// Mark this router as active for testing purposes.
    #[cfg(test)]
    pub(crate) fn mark_active_for_test(&self) {
        self.lifecycle
            .store(PrefillLifecycleState::Active as u8, Ordering::Release);
    }
}
