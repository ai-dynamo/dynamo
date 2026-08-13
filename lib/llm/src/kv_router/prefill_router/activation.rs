// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;
use tokio::sync::oneshot;

use dynamo_kv_router::{PrefillLoadEstimator, config::KvRouterConfig};
use dynamo_runtime::{
    component::{Client, Endpoint},
    discovery::DiscoveryQuery,
    pipeline::{PushRouter, RouterMode},
    prelude::DistributedRuntimeProvider,
    protocols::{EndpointId, annotated::Annotated},
};

use super::{InnerPrefillRouter, PrefillLifecycleState, PrefillRouter};
use crate::{
    discovery::ModelManager,
    kv_router::KvPushRouter,
    model_card::ModelDeploymentCard,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::WORKER_TYPE_PREFILL,
    },
    session_affinity::create_affinity_coordinator,
};

struct ActivationConfig {
    endpoint: Endpoint,
    model_manager: Arc<ModelManager>,
    router_mode: RouterMode,
    kv_cache_block_size: u32,
    kv_router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    session_affinity_ttl: Option<std::time::Duration>,
    configured_is_eagle: bool,
    model_name: String,
}

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
            task_guard: None,
            lifecycle: std::sync::atomic::AtomicU8::new(PrefillLifecycleState::Pending as u8),
            #[cfg(test)]
            activation_task_state: Arc::new(()),
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
        model_name: String,
        namespace: String,
        is_eagle: bool,
        worker_monitor: Option<crate::discovery::KvWorkerMonitor>,
        task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
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
            task_guard: task_guard.clone(),
            lifecycle: std::sync::atomic::AtomicU8::new(PrefillLifecycleState::Pending as u8),
            #[cfg(test)]
            activation_task_state: Arc::new(()),
        });

        // Spawn background task to wait for activation
        let router_weak = Arc::downgrade(&router);
        #[cfg(test)]
        let activation_task_state = router.activation_task_state.clone();
        tokio::spawn(async move {
            let _activation_task_guard = task_guard;
            #[cfg(test)]
            let _activation_task_state = activation_task_state;
            tokio::select! {
                result = activation_rx => {
                    let Ok(endpoint) = result else {
                        tracing::debug!("Prefill router activation channel closed without receiving endpoint");
                        return;
                    };
                    let Some(router) = router_weak.upgrade() else {
                        return;
                    };
                    let router_mode = router.router_mode;
                    let session_affinity_ttl = router.session_affinity_ttl;
                    let configured_is_eagle = router.is_eagle;
                    let model_name = router.model_name.clone();
                    let prefill_load_estimator = router.prefill_load_estimator.clone();
                    drop(router);

                    let activation_config = ActivationConfig {
                        endpoint,
                        model_manager,
                        router_mode,
                        kv_cache_block_size,
                        kv_router_config,
                        prefill_load_estimator,
                        session_affinity_ttl,
                        configured_is_eagle,
                        model_name,
                    };
                    let activation = Self::build_activation(activation_config);
                    let activation = tokio::select! {
                        result = activation => result,
                        _ = cancel_token.cancelled() => {
                            tracing::debug!("Prefill router activation cancelled");
                            return;
                        }
                    };
                    match activation {
                        Ok((endpoint_id, inner_router, prefill_client)) => {
                            if let Some(router) = router_weak.upgrade() {
                                Self::attach_prefill_client(
                                    worker_monitor.as_ref(),
                                    &prefill_client,
                                );
                                router.finish_activation(endpoint_id, inner_router);
                            }
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Failed to activate prefill router");
                        }
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::debug!("Prefill router activation cancelled");
                }
            }
        });

        router
    }

    async fn build_activation(
        config: ActivationConfig,
    ) -> Result<(EndpointId, InnerPrefillRouter, Client)> {
        let ActivationConfig {
            endpoint,
            model_manager,
            router_mode,
            kv_cache_block_size,
            kv_router_config,
            prefill_load_estimator,
            session_affinity_ttl,
            configured_is_eagle,
            model_name,
        } = config;
        tracing::info!(?router_mode, "Activating prefill router");

        let endpoint_id = endpoint.id();

        // Start runtime config watcher for this endpoint (needed for get_disaggregated_endpoint)
        // This must be done before creating the router so bootstrap info is available
        model_manager
            .get_or_create_runtime_config_watcher(&endpoint)
            .await?;

        let inner_router = if router_mode.is_kv_routing() {
            let endpoint_id = endpoint.id();
            let discovered_cards = endpoint
                .component()
                .drt()
                .discovery()
                .list(DiscoveryQuery::EndpointModels {
                    namespace: endpoint_id.namespace,
                    component: endpoint_id.component,
                    endpoint: endpoint_id.name,
                })
                .await;
            let is_eagle = match discovered_cards {
                Ok(instances) => instances
                    .into_iter()
                    .find_map(|instance| instance.deserialize_model::<ModelDeploymentCard>().ok())
                    .map_or(configured_is_eagle, |card| card.runtime_config.enable_eagle),
                Err(error) => {
                    tracing::warn!(%error, "Failed to read prefill model card; using configured EAGLE mode");
                    configured_is_eagle
                }
            };

            // Create KV chooser using the endpoint (this is a prefill router)
            let kv_chooser = model_manager
                .kv_chooser_for_with_worker_role(
                    &endpoint,
                    kv_cache_block_size,
                    kv_router_config,
                    prefill_load_estimator,
                    Some(crate::worker_type::WorkerType::Prefill),
                    WORKER_TYPE_PREFILL,
                    Some(model_name),
                    is_eagle,
                )
                .await?;

            // Extract client from kv_chooser to ensure shared state
            let client = kv_chooser.client().clone();
            let affinity =
                create_affinity_coordinator(session_affinity_ttl, client.clone()).await?;
            let prefill_client = client.clone();

            // Build the PushRouter for prefill with KV mode using the shared client
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                RouterMode::KV,
                None, // worker_monitor
            )
            .await?;

            // Wrap it in KvPushRouter
            (
                InnerPrefillRouter::KvRouter(Arc::new(KvPushRouter::new_with_coordinator(
                    push_router,
                    kv_chooser,
                    affinity,
                ))),
                prefill_client,
            )
        } else {
            // Create client for simple router
            let client = endpoint.client().await?;
            let affinity =
                create_affinity_coordinator(session_affinity_ttl, client.clone()).await?;
            let prefill_client = client.clone();

            // Create simple push router with the frontend's router mode
            // Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
            // available in KV routing mode where the router has actual bookkeeping.
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                router_mode,
                None, // worker_monitor
            )
            .await?;

            (
                InnerPrefillRouter::SimpleRouter(Arc::new(
                    crate::session_affinity::SessionAffinityPushRouter::new_with_coordinator(
                        push_router,
                        affinity,
                        router_mode.is_direct_routing(),
                    ),
                )),
                prefill_client,
            )
        };

        Ok((endpoint_id, inner_router.0, inner_router.1))
    }

    fn finish_activation(&self, endpoint_id: EndpointId, inner_router: InnerPrefillRouter) {
        let _ = self.endpoint_id.set(endpoint_id);
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
