// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context as _, Result};
use tokio::sync::{oneshot, watch};

use dynamo_kv_router::{
    DEFAULT_ROUTING_GROUP, PrefillLoadEstimator, RoutingPartitionRef,
    conditional_disagg::make_conditional_disagg_policy,
    config::KvRouterConfig,
    selector::{DefaultWorkerSelector, WorkerSelector},
};
use dynamo_runtime::{
    component::Endpoint,
    discovery::DiscoveryQuery,
    pipeline::{PushRouter, RouterMode},
    prelude::DistributedRuntimeProvider,
    protocols::annotated::Annotated,
};

use super::{PrefillBinding, PrefillBuildContext, PrefillLifecycleState, PrefillRouter};
use crate::{
    discovery::{LoadThresholdHandle, ModelManager, WorkerSetTarget},
    kv_router::{RouterLoadSource, RoutingHost, RoutingLoadContext, WorkerSelectorFactory},
    local_model::runtime_config::ModelRuntimeConfig,
    model_card::ModelDeploymentCard,
    protocols::common::{
        llm_backend::{LLMEngineOutput, PreprocessedRequest},
        timing::WORKER_TYPE_PREFILL,
    },
    session_affinity::{SessionAffinityMode, create_affinity_coordinator},
};

/// How the prefill worker set wants to be routed to, resolved from its cards.
#[derive(Debug)]
struct PrefillAdvertisement {
    router_mode: RouterMode,
    /// `None` when the card declared nothing, so the decode set's tuning applies.
    kv_router_config: Option<KvRouterConfig>,
    /// Taken from an advertised `router_config` whole, `None` included -- an
    /// advertisement replaces the decode set's configuration rather than
    /// merging with it. `None` here when nothing was advertised at all.
    session_affinity_ttl: Option<Option<std::time::Duration>>,
    is_eagle: bool,
    /// The prefill workers' own block size, which is what their KV events are
    /// keyed on. The decode set's value would index this pool at the wrong
    /// granularity if the two ever differ.
    kv_cache_block_size: u32,
}

/// A prefill worker that declares its own `router_config` governs this hop;
/// one that declares nothing inherits `decode_router_mode`. That is what lets a
/// deployment run KV-routed prefill in front of round-robin decode.
fn resolve_advertisement_from_cards(
    cards: &[ModelDeploymentCard],
    decode_router_mode: RouterMode,
) -> Result<PrefillAdvertisement> {
    let mode_of = |card: &ModelDeploymentCard| {
        card.router_config
            .as_ref()
            .map_or(decode_router_mode, |config| config.router_mode)
    };

    let Some(first) = cards.first() else {
        anyhow::bail!("no readable prefill model card; cannot resolve prefill routing");
    };

    let advertisement = PrefillAdvertisement {
        router_mode: mode_of(first),
        kv_router_config: first
            .router_config
            .as_ref()
            .map(|config| config.kv_router_config.clone()),
        session_affinity_ttl: first.router_config.as_ref().map(|config| {
            config
                .session_affinity_ttl_secs
                .map(std::time::Duration::from_secs)
        }),
        is_eagle: first.runtime_config.enable_eagle,
        kv_cache_block_size: first.kv_cache_block_size,
    };

    // Only legacy explicit endpoints can supply multiple cards here; managed
    // topology supplies the selected card and filters incompatible workers.
    let disagreeing = cards
        .iter()
        .skip(1)
        .filter(|card| mode_of(card) != advertisement.router_mode)
        .count();
    if disagreeing > 0 {
        tracing::warn!(
            resolved_mode = ?advertisement.router_mode,
            disagreeing,
            total = cards.len(),
            "Prefill workers advertise conflicting router modes; using the first"
        );
    }

    Ok(advertisement)
}

impl PrefillRouter<DefaultWorkerSelector> {
    /// Create a disabled prefill router that will never activate (passthrough only)
    pub fn disabled(
        model_manager: Arc<ModelManager>,
        decode_router_mode: RouterMode,
        session_affinity_ttl_secs: Option<u64>,
    ) -> Arc<Self> {
        Self::disabled_with_selector(
            model_manager,
            decode_router_mode,
            session_affinity_ttl_secs,
            SessionAffinityMode::Hard,
        )
    }

    /// `decode_router_mode` is the owning decode worker set's mode. It governs
    /// decode-side decisions and is the fallback for the prefill hop; a prefill
    /// worker that advertises its own `router_config` overrides the latter.
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        activation_rx: oneshot::Receiver<Endpoint>,
        model_manager: Arc<ModelManager>,
        decode_router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        session_affinity_ttl_secs: Option<u64>,
        session_affinity_mode: SessionAffinityMode,
        model_name: String,
        namespace: String,
        load_thresholds: LoadThresholdHandle,
        parent_token: tokio_util::sync::CancellationToken,
    ) -> Arc<Self> {
        Self::new_with_selector_factory(
            Some(activation_rx),
            model_manager,
            decode_router_mode,
            kv_cache_block_size,
            kv_router_config,
            Arc::new(|config, worker_type, _partition| {
                DefaultWorkerSelector::new(
                    Some(config.clone()),
                    worker_type.default_selector_label(),
                )
            }),
            prefill_load_estimator,
            session_affinity_ttl_secs,
            session_affinity_mode,
            model_name,
            namespace,
            load_thresholds,
            parent_token,
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
        decode_router_mode: RouterMode,
        session_affinity_ttl_secs: Option<u64>,
        session_affinity_mode: SessionAffinityMode,
    ) -> Arc<Self> {
        Arc::new(Self {
            binding: arc_swap::ArcSwapOption::empty(),
            target: parking_lot::Mutex::new(None),
            target_tx: None,
            decode_routing_host: std::sync::OnceLock::new(),
            worker_selector_factory: None,
            model_manager,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            decode_router_mode,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
            session_affinity_mode,
            conditional_disagg_policy: make_conditional_disagg_policy(None),
            conditional_disagg_prefill_busy_threshold: None,
            conditional_disagg_decode_busy_threshold: None,
            prefill_load_estimator: None,
            model_name: String::new(), // Not used for disabled router
            namespace: String::new(),  // Not used for disabled router
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
        decode_router_mode: RouterMode,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        worker_selector_factory: WorkerSelectorFactory<Sel>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        session_affinity_ttl_secs: Option<u64>,
        session_affinity_mode: SessionAffinityMode,
        model_name: String,
        namespace: String,
        load_thresholds: LoadThresholdHandle,
        parent_token: tokio_util::sync::CancellationToken,
        task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
    ) -> Arc<Self> {
        let cancel_token = parent_token.child_token();
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
            decode_routing_host: std::sync::OnceLock::new(),
            worker_selector_factory: Some(worker_selector_factory),
            model_manager: model_manager.clone(),
            cancel_token: cancel_token.clone(),
            decode_router_mode,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
            session_affinity_mode,
            conditional_disagg_policy,
            conditional_disagg_prefill_busy_threshold,
            conditional_disagg_decode_busy_threshold,
            prefill_load_estimator,
            model_name,
            namespace,
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
                load_thresholds,
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
                            router.set_target(Some(WorkerSetTarget::Legacy(endpoint)));
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
        target: WorkerSetTarget,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
    ) -> Result<PrefillBinding<Sel>> {
        let target_id = target.id();
        let endpoint = target.endpoint();
        let endpoint_id = endpoint.id();
        let client = target.client(context.parent_token.clone()).await?;

        // Start runtime config watcher for this endpoint (needed for get_disaggregated_endpoint)
        // This must be done before creating the router so bootstrap info is available
        context
            .model_manager
            .get_or_create_runtime_config_watcher(endpoint)
            .await?;

        let advertisement = match &target {
            WorkerSetTarget::Committed(target) => resolve_advertisement_from_cards(
                std::slice::from_ref(target.card.as_ref()),
                context.decode_router_mode,
            )?,
            WorkerSetTarget::Legacy(endpoint) => {
                Self::resolve_prefill_advertisement(context, endpoint).await?
            }
        };
        let prefill_router_mode = advertisement.router_mode;

        // Everything the hop uses comes from the prefill card when it says so,
        // falling back to the decode set. A block size of 0 means the card never
        // declared one, so it cannot be trusted over the decode set's.
        let prefill_block_size = match advertisement.kv_cache_block_size {
            0 => kv_cache_block_size,
            advertised => advertised,
        };
        let prefill_session_affinity_ttl = advertisement
            .session_affinity_ttl
            .unwrap_or(context.session_affinity_ttl);
        let load_context = RoutingLoadContext::start(
            client.clone(),
            RouterLoadSource::Prefill,
            context.load_thresholds.clone(),
            &context.parent_token,
            context.task_guard.clone(),
        )
        .await?;

        // A prefill card that declares a mode may declare KV tuning alongside it;
        // honoring only half of its `RouterConfig` would be a trap. Whichever
        // config wins, `router_track_active_blocks` stays off: prefill routing is
        // prompt-side, and crediting decode blocks here would double-count load.
        let advertised_kv_tuning = advertisement.kv_router_config.is_some();
        let prefill_kv_config = match advertisement.kv_router_config {
            Some(mut advertised) => {
                advertised.router_track_active_blocks = false;
                Some(advertised)
            }
            None => kv_router_config,
        };

        // Logged once per activation, and deliberately the *resolved* values
        // rather than what the card asked for. An advertisement replaces the
        // decode set's configuration rather than merging with it, so a worker
        // that names a mode without restating tuning silently gets defaults --
        // this line is what makes that answerable without a rebuild.
        tracing::info!(
            ?prefill_router_mode,
            decode_router_mode = ?context.decode_router_mode,
            advertised = advertised_kv_tuning,
            is_eagle = advertisement.is_eagle,
            block_size = prefill_block_size,
            session_affinity_ttl = ?prefill_session_affinity_ttl,
            kv_tuning = ?prefill_kv_config,
            "Activating prefill router"
        );

        let router = if prefill_router_mode.is_kv_routing() {
            // Create KV chooser using the endpoint (this is a prefill router)
            let effective_kv_router_config = prefill_kv_config.clone().unwrap_or_default();
            let selector = (context.worker_selector_factory)(
                &effective_kv_router_config,
                crate::worker_type::WorkerType::Prefill,
                RoutingPartitionRef::new(&context.model_name, DEFAULT_ROUTING_GROUP),
            );
            let kv_chooser = context
                .model_manager
                .kv_chooser_for_with_selector_and_client(
                    client.clone(),
                    prefill_block_size,
                    selector,
                    prefill_kv_config,
                    context.prefill_load_estimator.clone(),
                    Some(crate::worker_type::WorkerType::Prefill),
                    WORKER_TYPE_PREFILL,
                    Some(context.model_name.clone()),
                    advertisement.is_eagle,
                    load_context.scheduler_load_sender(),
                    load_context.cancellation_token(),
                )
                .await?;

            let affinity =
                create_affinity_coordinator(prefill_session_affinity_ttl, client.clone()).await?;

            // Build the PushRouter for prefill with KV mode using the shared client
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                RouterMode::KV,
                None, // worker_monitor
            )
            .await?;

            Arc::new(RoutingHost::new_with_load_context_and_coordinator(
                push_router,
                kv_chooser,
                load_context.clone(),
                affinity,
                context.session_affinity_mode,
            ))
        } else {
            let affinity =
                create_affinity_coordinator(prefill_session_affinity_ttl, client.clone()).await?;

            // Create the transport and discovery layer for the builtin policy.
            // Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
            // available in KV routing mode where the router has actual bookkeeping.
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                prefill_router_mode,
                None, // worker_monitor
            )
            .await?;

            Arc::new(RoutingHost::<Sel>::new_builtin_with_coordinator(
                push_router,
                load_context.clone(),
                affinity,
                context.session_affinity_mode,
            )?)
        };

        Ok(PrefillBinding {
            target_id,
            endpoint_id,
            router,
            prefill_router_mode,
        })
    }

    /// Fetch the prefill worker set's own cards and resolve how the prefill hop
    /// should be routed.
    ///
    /// Returns `Err` rather than silently inheriting when the cards cannot be
    /// read. `drive_target` retries activation with backoff, and a delayed
    /// activation is far cheaper than routing an entire deployment with the
    /// wrong strategy because of one transient discovery miss — a downgrade that
    /// would persist until the binding was rebuilt.
    async fn resolve_prefill_advertisement(
        context: &PrefillBuildContext<Sel>,
        endpoint: &Endpoint,
    ) -> Result<PrefillAdvertisement> {
        let endpoint_id = endpoint.id();
        let instances = endpoint
            .component()
            .drt()
            .discovery()
            .list(DiscoveryQuery::EndpointModels {
                namespace: endpoint_id.namespace.clone(),
                component: endpoint_id.component.clone(),
                endpoint: endpoint_id.name.clone(),
            })
            .await
            .with_context(|| format!("listing prefill model cards for {endpoint_id}"))?;

        // An unparseable card is not just a missing EAGLE hint any more: it
        // drops a worker out of the vote that decides how this hop is routed.
        let cards: Vec<ModelDeploymentCard> = instances
            .into_iter()
            .filter_map(|instance| {
                instance
                    .deserialize_model::<ModelDeploymentCard>()
                    .inspect_err(|error| {
                        tracing::debug!(%error, %endpoint_id, "Skipping unreadable prefill card")
                    })
                    .ok()
            })
            .collect();

        resolve_advertisement_from_cards(&cards, context.decode_router_mode)
            .with_context(|| format!("prefill endpoint {endpoint_id}"))
    }

    async fn drive_target(
        router: std::sync::Weak<Self>,
        mut target_rx: watch::Receiver<Option<WorkerSetTarget>>,
        cancel_token: tokio_util::sync::CancellationToken,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        load_thresholds: LoadThresholdHandle,
    ) {
        loop {
            let target = target_rx.borrow_and_update().clone();
            let Some(target) = target else {
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
            let target_id = target.id();
            let endpoint_id = target.endpoint().id();
            let Some(router_ref) = router.upgrade() else {
                return;
            };
            let reuses_binding = router_ref
                .binding
                .load_full()
                .is_some_and(|binding| binding.target_id == target_id)
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
                decode_router_mode: router_ref.decode_router_mode,
                worker_selector_factory: router_ref
                    .worker_selector_factory
                    .clone()
                    .expect("enabled prefill router has a worker selector factory"),
                prefill_load_estimator: router_ref.prefill_load_estimator.clone(),
                session_affinity_ttl: router_ref.session_affinity_ttl,
                session_affinity_mode: router_ref.session_affinity_mode,
                model_name: router_ref.model_name.clone(),
                load_thresholds: load_thresholds.clone(),
                parent_token: cancel_token.child_token(),
                task_guard: router_ref.task_guard.clone(),
            };
            drop(router_ref);
            let build = Self::build_binding(
                &build_context,
                target,
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
                Ok(binding) => {
                    let current_target = router_ref.target.lock();
                    if current_target.as_ref() != Some(&target_id) {
                        continue;
                    }
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
                    if router_ref.target.lock().as_ref() != Some(&target_id) {
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
    pub(crate) fn set_target(&self, target: Option<WorkerSetTarget>) {
        let target_id = target.as_ref().map(WorkerSetTarget::id);
        let mut current = self.target.lock();
        if *current == target_id {
            return;
        }
        *current = target_id.clone();
        let reuses_binding = target_id.is_some()
            && self
                .binding
                .load_full()
                .is_some_and(|binding| Some(&binding.target_id) == target_id.as_ref());
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
        self.target_tx
            .as_ref()
            .and_then(|tx| tx.borrow().as_ref().map(|target| target.endpoint().id()))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::discovery::CommittedWorkerSetTarget;
    use crate::entrypoint::RouterConfig;
    use dynamo_kv_router::config::KvRouterConfig;
    use dynamo_kv_router::protocols::RoutingConstraints;
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::{DiscoverySpec, EventTransportKind},
        distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
        pipeline::{
            AsyncEngine, AsyncEngineContextProvider, ManyOut, Operator, ResponseStream, SingleIn,
            network::Ingress,
        },
        storage::kv,
    };
    use futures::StreamExt;

    fn card(router_config: Option<RouterConfig>) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only("test-model");
        card.router_config = router_config;
        card
    }

    #[test]
    fn inherits_decode_mode_when_card_advertises_nothing() {
        // The pre-override behavior: a prefill worker that says nothing is
        // routed exactly as the decode set is. Every deployment predating this
        // feature lands here, so it must not shift.
        let cards = vec![card(None)];
        let resolved =
            resolve_advertisement_from_cards(&cards, RouterMode::RoundRobin).expect("resolves");
        assert_eq!(resolved.router_mode, RouterMode::RoundRobin);
        assert!(resolved.kv_router_config.is_none());
    }

    #[test]
    fn session_affinity_ttl_is_taken_whole_or_not_at_all() {
        // An advertisement replaces the decode set's configuration rather than
        // merging, so a card that advertises without a TTL means "no affinity",
        // not "inherit the frontend's". A card advertising nothing means inherit.
        let advertised = vec![card(Some(RouterConfig::new(
            RouterMode::KV,
            KvRouterConfig::default(),
        )))];
        assert_eq!(
            resolve_advertisement_from_cards(&advertised, RouterMode::KV)
                .expect("resolves")
                .session_affinity_ttl,
            Some(None),
        );

        let silent = vec![card(None)];
        assert_eq!(
            resolve_advertisement_from_cards(&silent, RouterMode::KV)
                .expect("resolves")
                .session_affinity_ttl,
            None,
        );
    }

    #[test]
    fn no_readable_card_is_an_error_not_a_silent_inherit() {
        // Activation must fail and be retried rather than quietly routing the
        // whole deployment with the decode set's mode.
        let error = resolve_advertisement_from_cards(&[], RouterMode::KV)
            .expect_err("empty card list must not resolve");
        assert!(
            error.to_string().contains("no readable prefill model card"),
            "unexpected error: {error}"
        );
    }

    struct PrefillWorker(u32);

    #[async_trait::async_trait]
    impl
        AsyncEngine<
            SingleIn<PreprocessedRequest>,
            ManyOut<Annotated<LLMEngineOutput>>,
            anyhow::Error,
        > for PrefillWorker
    {
        async fn generate(
            &self,
            request: SingleIn<PreprocessedRequest>,
        ) -> Result<ManyOut<Annotated<LLMEngineOutput>>> {
            let mut output = LLMEngineOutput::stop();
            output.token_ids = vec![self.0];
            Ok(ResponseStream::new(
                Box::pin(futures::stream::iter([Annotated::from_data(output)])),
                request.context(),
            ))
        }
    }

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".into())
            .token_ids(vec![1; 128])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    async fn prefilled_worker(router: &PrefillRouter) -> u32 {
        let mut response = router
            .generate(SingleIn::new(request()), Arc::new(PrefillWorker(99)))
            .await
            .unwrap();
        let worker = response.next().await.unwrap().data.unwrap().token_ids[0];
        while response.next().await.is_some() {}
        worker
    }

    #[tokio::test]
    async fn committed_prefill_uses_selected_card_and_admission_across_endpoint_reuse() {
        // The request-plane server is process-global, but other Tokio tests
        // destroy the runtime that owns its accept loop.
        const TEST: &str = concat!(
            module_path!(),
            "::committed_prefill_uses_selected_card_and_admission_across_endpoint_reuse"
        );
        let test_name = TEST.split_once("::").unwrap().1;
        if std::env::var("DYNAMO_ROUTING_HOP_TEST").as_deref() != Ok(test_name) {
            let mut child = tokio::process::Command::new(std::env::current_exe().unwrap());
            child
                .args(["--exact", test_name, "--nocapture"])
                .env("DYNAMO_ROUTING_HOP_TEST", test_name)
                .env("DYN_TCP_RPC_HOST", "127.0.0.1")
                .env("DYN_TCP_RPC_PORT", "0")
                .env("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1")
                .env("DYN_TCP_RESPONSE_STREAM_PORT", "0")
                .kill_on_drop(true);
            let output = tokio::time::timeout(Duration::from_secs(30), child.output())
                .await
                .expect("prefill subprocess must finish within its deadline")
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        let runtime = Runtime::from_current().unwrap();
        let store = tempfile::tempdir().unwrap();
        let config = || DistributedConfig {
            discovery_backend: DiscoveryBackend::KvStore(kv::Selector::File(store.path().into())),
            nats_config: None,
            request_plane: RequestPlaneMode::Tcp,
            response_plane: None,
            event_transport_kind: EventTransportKind::Zmq,
        };
        let namespace = format!("prefill-admission-{}", uuid::Uuid::new_v4());
        let mut workers = Vec::new();
        let mut runtimes = Vec::new();
        for marker in 0..3 {
            let drt = DistributedRuntime::new(runtime.clone(), config())
                .await
                .unwrap();
            let endpoint = drt
                .namespace(namespace.clone())
                .unwrap()
                .component("workers")
                .unwrap()
                .endpoint("prefill");
            workers.push(
                endpoint
                    .endpoint_builder()
                    .handler(Ingress::for_engine(Arc::new(PrefillWorker(marker))).unwrap())
                    .start_with_registration()
                    .await
                    .unwrap(),
            );
            // Raw cards provide per-worker runtime data, but the hop's effective
            // routing configuration must come from its committed target.
            let mut advertised = card(Some(RouterConfig::new(
                RouterMode::RoundRobin,
                KvRouterConfig::default(),
            )));
            advertised.kv_cache_block_size = if marker == 2 { 32 } else { 64 };
            drt.discovery()
                .register(
                    DiscoverySpec::from_model(
                        namespace.clone(),
                        "workers".into(),
                        "prefill".into(),
                        &advertised,
                    )
                    .unwrap(),
                )
                .await
                .unwrap();
            runtimes.push(drt);
        }
        let drt = DistributedRuntime::new(runtime.clone(), config())
            .await
            .unwrap();
        let endpoint = drt
            .namespace(namespace.clone())
            .unwrap()
            .component("workers")
            .unwrap()
            .endpoint("prefill");
        let raw_client = endpoint.client().await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut instances = raw_client.instance_source.as_ref().clone();
            while instances.borrow_and_update().len() != 3 {
                instances.changed().await.unwrap();
            }
        })
        .await
        .expect("raw endpoint discovery must contain every worker");
        let ids: Vec<_> = workers
            .iter()
            .map(|worker| worker.instance().id())
            .collect();
        let kv_config = KvRouterConfig {
            use_kv_events: false,
            router_track_active_blocks: false,
            ..Default::default()
        };
        let target = |generation, block_size, admitted_ids| {
            let mut selected = card(Some(RouterConfig::new(RouterMode::KV, kv_config.clone())));
            selected.kv_cache_block_size = block_size;
            WorkerSetTarget::Committed(CommittedWorkerSetTarget {
                endpoint: endpoint.clone(),
                group: endpoint.id().to_string(),
                generation,
                card: Arc::new(selected),
                admitted_ids,
            })
        };
        let router = PrefillRouter::new_with_selector_factory(
            None,
            Arc::new(ModelManager::new()),
            RouterMode::RoundRobin,
            16,
            None,
            Arc::new(|config, _, _| DefaultWorkerSelector::new(Some(config.clone()), "prefill")),
            None,
            None,
            SessionAffinityMode::Hard,
            "test-model".into(),
            namespace,
            LoadThresholdHandle::new(Default::default()),
            drt.child_token(),
            None,
        );
        let (admissions, admitted_ids) = watch::channel(vec![ids[0]]);
        router.set_target(Some(target(1, 64, admitted_ids)));
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let worker = prefilled_worker(&router).await;
                if worker != 99 {
                    assert_eq!(worker, 0);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("committed prefill target must activate");
        let retired = router.binding.load_full().unwrap();
        let chooser = retired
            .router
            .kv_router_if_enabled()
            .expect("selected KV mode must override raw RR cards");
        assert_eq!(chooser.block_size(), 64);
        for _ in 0..6 {
            assert_eq!(prefilled_worker(&router).await, 0);
        }
        let reservation = router
            .reserve_prefill_worker(
                "committed-prefill",
                &[1; 128],
                None,
                None,
                None,
                0.0,
                0,
                None,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();
        assert_eq!(reservation.worker_id(), ids[0]);
        assert_eq!(reservation.dp_rank(), Some(0));
        reservation.release().await.unwrap();

        admissions.send_replace(vec![ids[0], ids[1]]);
        admissions.send_replace(vec![ids[1]]);
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let outcome = router
                    .query_prefill_worker(
                        &[1; 128],
                        None,
                        None,
                        None,
                        0.0,
                        0,
                        None,
                        RoutingConstraints::default(),
                    )
                    .await
                    .unwrap();
                let crate::kv_router::prefill_router::PrefillQueryOutcome::Routed {
                    worker_id, ..
                } = outcome
                else {
                    panic!("prefill query must remain routable");
                };
                assert!(
                    ids[..2].contains(&worker_id),
                    "rejected prefill worker became selectable"
                );
                if worker_id == ids[1] {
                    break;
                }
            }
        })
        .await
        .expect("compatible replacement must route through the existing channel");
        assert_eq!(prefilled_worker(&router).await, 1);
        let mut retained_response = retired
            .router
            .generate(SingleIn::new(request()))
            .await
            .unwrap();
        assert_eq!(
            retained_response
                .next()
                .await
                .unwrap()
                .data
                .unwrap()
                .token_ids,
            vec![1]
        );
        while retained_response.next().await.is_some() {}

        admissions.send_replace(Vec::new());
        drop(admissions);
        router.set_target(None);
        let (_successor_admissions, successor_ids) = watch::channel(vec![ids[2]]);
        router.set_target(Some(target(2, 32, successor_ids)));
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let worker = prefilled_worker(&router).await;
                if worker != 99 {
                    assert_eq!(worker, 2);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("same-endpoint successor must activate");
        assert_eq!(
            router
                .binding
                .load_full()
                .unwrap()
                .router
                .kv_router_if_enabled()
                .unwrap()
                .block_size(),
            32
        );
        for _ in 0..6 {
            assert_eq!(prefilled_worker(&router).await, 2);
        }
        assert!(
            tokio::time::timeout(
                Duration::from_secs(5),
                retired.router.generate(SingleIn::new(request()),)
            )
            .await
            .expect("retired prefill admission must fail promptly")
            .is_err()
        );

        drop(router);
        for worker in workers {
            worker.shutdown().await.unwrap();
        }
        runtime.shutdown();
    }
}
