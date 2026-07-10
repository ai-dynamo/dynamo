// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

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
};

impl PrefillRouter {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        session_affinity_ttl_secs: Option<u64>,
    ) -> Arc<Self> {
        let (activation_updates, _activation_rx) = tokio::sync::mpsc::unbounded_channel();
        Arc::new(Self {
            prefill_router: std::sync::RwLock::new(None),
            model_manager,
            endpoint_id: std::sync::RwLock::new(None),
            activation_updates,
            activation_generation: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            activation_attempts: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            activation_failures_remaining: std::sync::atomic::AtomicUsize::new(0),
            cancel_token: tokio_util::sync::CancellationToken::new(),
            router_mode,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
            prefill_load_estimator: None,
            model_name: String::new(), // Not used for disabled router
            namespace: String::new(),  // Not used for disabled router
            routing_endpoint_name: None,
            passthrough_when_unavailable: true,
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
        model_name: String,
        namespace: String,
        is_eagle: bool,
        worker_monitor: Option<crate::discovery::KvWorkerMonitor>,
    ) -> Arc<Self> {
        Self::new_inner(
            activation_rx,
            model_manager,
            router_mode,
            kv_cache_block_size,
            kv_router_config,
            prefill_load_estimator,
            session_affinity_ttl_secs,
            model_name,
            namespace,
            is_eagle,
            worker_monitor,
            None,
            true,
        )
    }

    /// Create a fail-closed prefill router whose request candidates come from
    /// a versioned alias while runtime metadata comes from the activated
    /// primary endpoint.
    #[expect(clippy::too_many_arguments)]
    pub fn new_for_endpoint_name(
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
        routing_endpoint_name: String,
    ) -> Arc<Self> {
        Self::new_inner(
            activation_rx,
            model_manager,
            router_mode,
            kv_cache_block_size,
            kv_router_config,
            prefill_load_estimator,
            session_affinity_ttl_secs,
            model_name,
            namespace,
            is_eagle,
            worker_monitor,
            Some(routing_endpoint_name),
            false,
        )
    }

    #[expect(clippy::too_many_arguments)]
    fn new_inner(
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
        routing_endpoint_name: Option<String>,
        passthrough_when_unavailable: bool,
    ) -> Arc<Self> {
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let (activation_updates, mut activation_update_rx) = tokio::sync::mpsc::unbounded_channel();

        let router = Arc::new(Self {
            prefill_router: std::sync::RwLock::new(None),
            model_manager: model_manager.clone(),
            endpoint_id: std::sync::RwLock::new(None),
            activation_updates,
            activation_generation: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            activation_attempts: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            activation_failures_remaining: std::sync::atomic::AtomicUsize::new(0),
            cancel_token: cancel_token.clone(),
            router_mode,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
            prefill_load_estimator,
            model_name,
            namespace,
            routing_endpoint_name,
            passthrough_when_unavailable,
            is_eagle,
            lifecycle: std::sync::atomic::AtomicU8::new(PrefillLifecycleState::Pending as u8),
        });

        // Keep activation watch-driven for the router's full lifetime. A
        // failed build is retried, and endpoint identity refreshes rebuild the
        // client instead of leaving a consumed one-shot stuck in Pending.
        let router_weak = Arc::downgrade(&router);
        tokio::spawn(async move {
            let mut endpoint = tokio::select! {
                result = activation_rx => result.ok(),
                update = activation_update_rx.recv() => update,
                _ = cancel_token.cancelled() => None,
            };
            while let Some(candidate) = endpoint {
                let Some(router) = router_weak.upgrade() else {
                    break;
                };
                let prefill_load_estimator = router.prefill_load_estimator.clone();
                let activation_result = router
                    .activate(
                        candidate.clone(),
                        model_manager.clone(),
                        kv_cache_block_size,
                        kv_router_config.clone(),
                        prefill_load_estimator,
                        worker_monitor.as_ref(),
                    )
                    .await;
                // The background task must not own the router while it waits
                // for discovery updates or retry backoff. Otherwise Drop can
                // never cancel either lifecycle watcher.
                drop(router);
                match activation_result {
                    Ok(()) => {
                        endpoint = tokio::select! {
                            update = activation_update_rx.recv() => update,
                            _ = cancel_token.cancelled() => None,
                        };
                    }
                    Err(error) => {
                        tracing::error!(%error, "Failed to activate prefill router; retrying");
                        endpoint = tokio::select! {
                            update = activation_update_rx.recv() => update,
                            _ = tokio::time::sleep(Duration::from_millis(100)) => Some(candidate),
                            _ = cancel_token.cancelled() => None,
                        };
                    }
                }
            }
            tracing::debug!("Prefill router activation watch stopped");
        });

        router
    }

    /// Activate the prefill router with the provided endpoint
    async fn activate(
        self: &Arc<Self>,
        config_endpoint: Endpoint,
        model_manager: Arc<ModelManager>,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_monitor: Option<&crate::discovery::KvWorkerMonitor>,
    ) -> Result<()> {
        #[cfg(test)]
        {
            self.activation_attempts.fetch_add(1, Ordering::AcqRel);
            if self
                .activation_failures_remaining
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                    remaining.checked_sub(1)
                })
                .is_ok()
            {
                anyhow::bail!("injected transient prefill activation failure");
            }
        }
        tracing::info!(
            router_mode = ?self.router_mode,
            "Activating prefill router"
        );

        let endpoint = self
            .routing_endpoint_name
            .as_ref()
            .map(|name| config_endpoint.component().endpoint(name))
            .unwrap_or_else(|| config_endpoint.clone());

        let endpoint_identity = endpoint.id();
        let already_current = self
            .endpoint_id
            .read()
            .expect("prefill endpoint lock poisoned")
            .as_ref()
            == Some(&endpoint_identity)
            && self
                .prefill_router
                .read()
                .expect("prefill router lock poisoned")
                .is_some();
        if already_current {
            return Ok(());
        }
        // An identity change invalidates the old client immediately. Keep the
        // route closed while the replacement watcher/router is constructed.
        self.activation_generation.fetch_add(1, Ordering::AcqRel);
        self.set_alias_availability(false);

        // Start runtime config watcher for this endpoint (needed for get_disaggregated_endpoint)
        // This must be done before creating the router so bootstrap info is available
        model_manager
            .get_or_create_runtime_config_watcher_for(&endpoint, &config_endpoint)
            .await?;

        let (inner_router, client) = if self.router_mode.is_kv_routing() {
            // Create KV chooser using the endpoint (this is a prefill router)
            let kv_chooser = model_manager
                .kv_chooser_for_with_runtime_config_endpoint(
                    &endpoint,
                    &config_endpoint,
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

            // Build the PushRouter for prefill with KV mode using the shared client
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client.clone(),
                RouterMode::KV,
                None, // worker_monitor
            )
            .await?;

            // Wrap it in KvPushRouter
            (
                InnerPrefillRouter::KvRouter(Arc::new(KvPushRouter::new(
                    push_router,
                    kv_chooser,
                    self.session_affinity_ttl,
                )?)),
                client.clone(),
            )
        } else {
            // Create client for simple router
            let client = endpoint.client().await?;
            Self::attach_prefill_client(worker_monitor, &client);

            // Create simple push router with the frontend's router mode
            // Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
            // available in KV routing mode where the router has actual bookkeeping.
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client.clone(),
                self.router_mode,
                None, // worker_monitor
            )
            .await?;

            (
                InnerPrefillRouter::SimpleRouter(Arc::new(
                    crate::session_affinity::SessionAffinityPushRouter::new(
                        push_router,
                        self.session_affinity_ttl,
                        self.router_mode.is_direct_routing(),
                    )?,
                )),
                client,
            )
        };

        *self
            .endpoint_id
            .write()
            .expect("prefill endpoint lock poisoned") = Some(endpoint_identity);
        *self
            .prefill_router
            .write()
            .expect("prefill router lock poisoned") = Some(inner_router);

        let generation = self
            .activation_generation
            .fetch_add(1, Ordering::AcqRel)
            .wrapping_add(1);
        let mut instances = client.instance_avail_watcher();
        self.set_alias_availability(!instances.borrow_and_update().is_empty());
        let router_weak = Arc::downgrade(self);
        let cancel_token = self.cancel_token.clone();
        let refresh_endpoint = config_endpoint;
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    changed = instances.changed() => {
                        let Some(router) = router_weak.upgrade() else {
                            break;
                        };
                        if router.activation_generation.load(Ordering::Acquire) != generation {
                            break;
                        }
                        if changed.is_err() {
                            router.set_alias_availability(false);
                            *router
                                .endpoint_id
                                .write()
                                .expect("prefill endpoint lock poisoned") = None;
                            let _ = router.activation_updates.send(refresh_endpoint.clone());
                            break;
                        }
                        router.set_alias_availability(!instances.borrow_and_update().is_empty());
                    }
                    _ = cancel_token.cancelled() => break,
                }
            }
        });

        tracing::info!(
            router_mode = ?self.router_mode,
            available = self.is_available(),
            "Prefill router initialized with live alias monitoring"
        );

        Ok(())
    }

    pub(super) fn set_alias_availability(&self, available: bool) {
        self.lifecycle.store(
            if available {
                PrefillLifecycleState::Active as u8
            } else {
                PrefillLifecycleState::Unavailable as u8
            },
            Ordering::Release,
        );
    }

    /// Refresh or retry activation with the latest primary endpoint identity.
    pub fn refresh_activation_endpoint(&self, endpoint: Endpoint) -> bool {
        self.activation_updates.send(endpoint).is_ok()
    }

    #[cfg(test)]
    pub(crate) fn fail_next_activation_for_test(&self) {
        self.activation_failures_remaining
            .store(1, Ordering::Release);
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

    /// Whether this router is currently deactivated (prefill workers died).
    pub fn is_deactivated(&self) -> bool {
        self.lifecycle_state() == PrefillLifecycleState::Unavailable
    }

    /// Whether the inner router has initialized, even if workers are unavailable.
    pub fn is_activated(&self) -> bool {
        self.prefill_router
            .read()
            .expect("prefill router lock poisoned")
            .is_some()
    }

    /// Whether this router is initialized and currently usable.
    pub fn is_available(&self) -> bool {
        self.lifecycle_state() == PrefillLifecycleState::Active
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
