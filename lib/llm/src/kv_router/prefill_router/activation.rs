// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;
use tokio::sync::{oneshot, watch};

use dynamo_kv_router::{
    DEFAULT_ROUTING_GROUP, PrefillLoadEstimator, RoutingPartitionRef,
    conditional_disagg::make_conditional_disagg_policy,
    config::KvRouterConfig,
    selector::{DefaultWorkerSelector, WorkerSelector},
};
use dynamo_runtime::{
    component::{Client, Endpoint},
    discovery::DiscoveryQuery,
    pipeline::{PushRouter, RouterMode},
    prelude::DistributedRuntimeProvider,
    protocols::annotated::Annotated,
};

use super::{
    InnerPrefillRouter, PrefillBinding, PrefillBuildContext, PrefillLifecycleState, PrefillRouter,
};
use crate::{
    discovery::ModelManager,
    kv_router::{KvPushRouter, KvRouter, WorkerSelectorFactory},
    local_model::runtime_config::ModelRuntimeConfig,
    model_card::ModelDeploymentCard,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::WORKER_TYPE_PREFILL,
    },
    session_affinity::create_affinity_coordinator,
};

impl PrefillRouter<DefaultWorkerSelector> {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        session_affinity_ttl_secs: Option<u64>,
    ) -> Arc<Self> {
        Self::disabled_with_selector(model_manager, router_mode, session_affinity_ttl_secs)
    }

    #[expect(clippy::too_many_arguments)]
    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        decode_router: Option<Arc<KvRouter>>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        session_affinity_ttl_secs: Option<u64>,
        model_name: String,
        namespace: String,
        is_eagle: bool,
        worker_monitor: Option<crate::discovery::KvWorkerMonitor>,
    ) -> Arc<Self> {
        Self::new_with_selector_factory(
            Some(activation_rx),
            model_manager,
            router_mode,
            kv_cache_block_size,
            kv_router_config,
            decode_router,
            Arc::new(|config, worker_type, _partition| {
                DefaultWorkerSelector::new(
                    Some(config.clone()),
                    worker_type.default_selector_label(),
                )
            }),
            prefill_load_estimator,
            session_affinity_ttl_secs,
            model_name,
            namespace,
            is_eagle,
            worker_monitor,
            None,
            None,
        )
    }
}

impl<Sel> PrefillRouter<Sel>
where
    Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
{
    pub(crate) fn disabled_with_selector(
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        session_affinity_ttl_secs: Option<u64>,
    ) -> Arc<Self> {
        Arc::new(Self {
            binding: arc_swap::ArcSwapOption::empty(),
            target: parking_lot::Mutex::new(None),
            target_tx: None,
            decode_router: None,
            worker_selector_factory: None,
            decode_session_affinity: std::sync::OnceLock::new(),
            model_manager,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            router_mode,
            encoder_cuda_to_cpu_ratio: None,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
            conditional_disagg_policy: make_conditional_disagg_policy(None),
            conditional_disagg_prefill_busy_threshold: None,
            conditional_disagg_decode_busy_threshold: None,
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
    pub(crate) fn new_with_selector_factory(
        activation_rx: Option<oneshot::Receiver<Endpoint>>,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        decode_router: Option<Arc<KvRouter<Sel>>>,
        worker_selector_factory: WorkerSelectorFactory<Sel>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        session_affinity_ttl_secs: Option<u64>,
        model_name: String,
        namespace: String,
        is_eagle: bool,
        worker_monitor: Option<crate::discovery::KvWorkerMonitor>,
        task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
        encoder_cuda_to_cpu_ratio: Option<usize>,
    ) -> Arc<Self> {
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let (target_tx, target_rx) = watch::channel(None);
        let conditional_disagg_policy = make_conditional_disagg_policy(kv_router_config.as_ref());
        let conditional_disagg_prefill_busy_threshold = kv_router_config.as_ref().and_then(|c| {
            c.conditional_disagg_prefill_busy_threshold
                .or(c.router_queue_threshold)
        });
        let conditional_disagg_decode_busy_threshold = kv_router_config
            .as_ref()
            .and_then(|c| c.conditional_disagg_decode_busy_threshold);

        let router = Arc::new(Self {
            binding: arc_swap::ArcSwapOption::empty(),
            target: parking_lot::Mutex::new(None),
            target_tx: Some(target_tx),
            decode_router,
            worker_selector_factory: Some(worker_selector_factory),
            decode_session_affinity: std::sync::OnceLock::new(),
            model_manager: model_manager.clone(),
            cancel_token: cancel_token.clone(),
            router_mode,
            encoder_cuda_to_cpu_ratio,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
            conditional_disagg_policy,
            conditional_disagg_prefill_busy_threshold,
            conditional_disagg_decode_busy_threshold,
            prefill_load_estimator,
            model_name,
            namespace,
            is_eagle,
            task_guard: task_guard.clone(),
            lifecycle: std::sync::atomic::AtomicU8::new(PrefillLifecycleState::Pending as u8),
            #[cfg(test)]
            activation_task_state: Arc::new(()),
        });

        let router_weak = Arc::downgrade(&router);
        let drive_cancel_token = cancel_token.clone();
        let drive_task_guard = task_guard.clone();
        #[cfg(test)]
        let drive_task_state = router.activation_task_state.clone();
        tokio::spawn(async move {
            let _drive_task_guard = drive_task_guard;
            #[cfg(test)]
            let _drive_task_state = drive_task_state;
            Self::drive_target(
                router_weak,
                target_rx,
                drive_cancel_token,
                kv_cache_block_size,
                kv_router_config,
                worker_monitor,
            )
            .await;
        });
        if let Some(activation_rx) = activation_rx {
            let router = Arc::downgrade(&router);
            let activation_task_guard = task_guard;
            #[cfg(test)]
            let activation_task_state = router
                .upgrade()
                .expect("prefill router exists during construction")
                .activation_task_state
                .clone();
            tokio::spawn(async move {
                let _activation_task_guard = activation_task_guard;
                #[cfg(test)]
                let _activation_task_state = activation_task_state;
                tokio::select! {
                    result = activation_rx => {
                        if let (Ok(endpoint), Some(router)) = (result, router.upgrade()) {
                            router.set_target(Some(endpoint));
                        }
                    }
                    _ = cancel_token.cancelled() => {}
                }
            });
        }

        router
    }

    async fn build_binding(
        context: &PrefillBuildContext<Sel>,
        endpoint: Endpoint,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
    ) -> Result<(PrefillBinding<Sel>, Client)> {
        tracing::info!(
            router_mode = ?context.router_mode,
            "Activating prefill router"
        );

        let endpoint_id = endpoint.id();

        // Start runtime config watcher for this endpoint (needed for get_disaggregated_endpoint)
        // This must be done before creating the router so bootstrap info is available
        context
            .model_manager
            .get_or_create_runtime_config_watcher(&endpoint)
            .await?;

        let inner_router = if context.router_mode.is_kv_routing() {
            let discovered_cards = endpoint
                .component()
                .drt()
                .discovery()
                .list(DiscoveryQuery::EndpointModels {
                    namespace: endpoint_id.namespace.clone(),
                    component: endpoint_id.component.clone(),
                    endpoint: endpoint_id.name.clone(),
                })
                .await;
            let is_eagle = match discovered_cards {
                Ok(instances) => instances
                    .into_iter()
                    .find_map(|instance| instance.deserialize_model::<ModelDeploymentCard>().ok())
                    .map_or(context.is_eagle, |card| card.runtime_config.enable_eagle),
                Err(error) => {
                    tracing::warn!(%error, "Failed to read prefill model card; using configured EAGLE mode");
                    context.is_eagle
                }
            };

            // Create KV chooser using the endpoint (this is a prefill router)
            let effective_kv_router_config = kv_router_config.clone().unwrap_or_default();
            let selector = (context.worker_selector_factory)(
                &effective_kv_router_config,
                crate::worker_type::WorkerType::Prefill,
                RoutingPartitionRef::new(&context.model_name, DEFAULT_ROUTING_GROUP),
            );
            let kv_chooser = context
                .model_manager
                .kv_chooser_for_with_selector(
                    &endpoint,
                    kv_cache_block_size,
                    selector,
                    kv_router_config,
                    context.prefill_load_estimator.clone(),
                    Some(crate::worker_type::WorkerType::Prefill),
                    WORKER_TYPE_PREFILL,
                    Some(context.model_name.clone()),
                    is_eagle,
                )
                .await?;

            // Extract client from kv_chooser to ensure shared state
            let client = kv_chooser.client().clone();
            let affinity =
                create_affinity_coordinator(context.session_affinity_ttl, client.clone()).await?;
            let prefill_client = client.clone();

            // Build the PushRouter for prefill with KV mode using the shared client
            let push_router = PushRouter::<
                PreprocessedRequest,
                Annotated<LLMEngineOutput>,
            >::from_client_with_monitor_and_ratio(
                client,
                RouterMode::KV,
                None,
                context.encoder_cuda_to_cpu_ratio,
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
                create_affinity_coordinator(context.session_affinity_ttl, client.clone()).await?;
            let prefill_client = client.clone();

            // Create simple push router with the frontend's router mode
            // Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
            // available in KV routing mode where the router has actual bookkeeping.
            let push_router = PushRouter::<
                PreprocessedRequest,
                Annotated<LLMEngineOutput>,
            >::from_client_with_monitor_and_ratio(
                client,
                context.router_mode,
                None,
                context.encoder_cuda_to_cpu_ratio,
            )
            .await?;

            (
                InnerPrefillRouter::SimpleRouter(Arc::new(
                    crate::session_affinity::SessionAffinityPushRouter::new_with_coordinator(
                        push_router,
                        affinity,
                        context.router_mode.is_direct_routing(),
                    ),
                )),
                prefill_client,
            )
        };

        Ok((
            PrefillBinding {
                endpoint_id,
                router: inner_router.0,
            },
            inner_router.1,
        ))
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

    async fn drive_target(
        router: std::sync::Weak<Self>,
        mut target_rx: watch::Receiver<Option<Endpoint>>,
        cancel_token: tokio_util::sync::CancellationToken,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        worker_monitor: Option<crate::discovery::KvWorkerMonitor>,
    ) {
        loop {
            let target = target_rx.borrow_and_update().clone();
            let Some(endpoint) = target else {
                tokio::select! {
                    biased;
                    _ = cancel_token.cancelled() => return,
                    changed = target_rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                    }
                }
                continue;
            };
            let endpoint_id = endpoint.id();
            let Some(router_ref) = router.upgrade() else {
                return;
            };
            let reuses_binding = router_ref
                .binding
                .load_full()
                .is_some_and(|binding| binding.endpoint_id == endpoint_id)
                && router_ref.lifecycle_state() == PrefillLifecycleState::Active;
            if reuses_binding {
                drop(router_ref);
                tokio::select! {
                    biased;
                    _ = cancel_token.cancelled() => return,
                    changed = target_rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                    }
                }
                continue;
            }
            let build_context = PrefillBuildContext {
                model_manager: router_ref.model_manager.clone(),
                router_mode: router_ref.router_mode,
                encoder_cuda_to_cpu_ratio: router_ref.encoder_cuda_to_cpu_ratio,
                worker_selector_factory: router_ref
                    .worker_selector_factory
                    .clone()
                    .expect("enabled prefill router has a worker selector factory"),
                prefill_load_estimator: router_ref.prefill_load_estimator.clone(),
                session_affinity_ttl: router_ref.session_affinity_ttl,
                model_name: router_ref.model_name.clone(),
                is_eagle: router_ref.is_eagle,
            };
            drop(router_ref);
            let build = Self::build_binding(
                &build_context,
                endpoint,
                kv_cache_block_size,
                kv_router_config.clone(),
            );
            tokio::pin!(build);
            let result = tokio::select! {
                biased;
                _ = cancel_token.cancelled() => return,
                changed = target_rx.changed() => {
                    if changed.is_err() {
                        return;
                    }
                    continue;
                }
                result = &mut build => result,
            };
            let Some(router_ref) = router.upgrade() else {
                return;
            };
            match result {
                Ok((binding, prefill_client)) => {
                    let current_target = router_ref.target.lock();
                    if current_target.as_ref() != Some(&endpoint_id) {
                        continue;
                    }
                    Self::attach_prefill_client(worker_monitor.as_ref(), &prefill_client);
                    router_ref.binding.store(Some(Arc::new(binding)));
                    router_ref
                        .lifecycle
                        .store(PrefillLifecycleState::Active as u8, Ordering::Release);
                    drop(current_target);
                    tracing::info!(
                        model_name = %router_ref.model_name,
                        namespace = %router_ref.namespace,
                        %endpoint_id,
                        "Prefill router target activated"
                    );
                }
                Err(error) => {
                    if router_ref.target.lock().as_ref() != Some(&endpoint_id) {
                        continue;
                    }
                    tracing::error!(
                        %error,
                        model_name = %router_ref.model_name,
                        namespace = %router_ref.namespace,
                        %endpoint_id,
                        "Failed to activate prefill router target"
                    );
                    drop(router_ref);
                    tokio::select! {
                        biased;
                        _ = cancel_token.cancelled() => return,
                        changed = target_rx.changed() => {
                            if changed.is_err() {
                                return;
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
                    }
                }
            }
        }
    }

    /// Update the desired Prefill endpoint. Clearing is synchronous so requests
    /// holding an older catalog snapshot bypass a removed endpoint before the
    /// replacement catalog is published.
    pub(crate) fn set_target(&self, target: Option<Endpoint>) {
        let target_id = target.as_ref().map(Endpoint::id);
        let mut current = self.target.lock();
        if *current == target_id {
            return;
        }
        *current = target_id.clone();
        let reuses_binding = target_id.is_some()
            && self
                .binding
                .load_full()
                .is_some_and(|binding| Some(&binding.endpoint_id) == target_id.as_ref());
        let lifecycle = if target.is_none() {
            PrefillLifecycleState::Unavailable
        } else if reuses_binding {
            PrefillLifecycleState::Active
        } else {
            self.binding.store(None);
            PrefillLifecycleState::Pending
        };
        self.lifecycle.store(lifecycle as u8, Ordering::Release);
        if let Some(target_tx) = &self.target_tx {
            target_tx.send_replace(target);
        }
    }

    /// Whether the inner router has initialized.
    pub fn is_activated(&self) -> bool {
        self.binding.load().is_some()
    }

    pub(super) fn lifecycle_state(&self) -> PrefillLifecycleState {
        PrefillLifecycleState::from_atomic(self.lifecycle.load(Ordering::Acquire))
    }

    #[cfg(test)]
    pub(crate) fn target_endpoint_id(&self) -> Option<dynamo_runtime::protocols::EndpointId> {
        self.target.lock().clone()
    }
}
