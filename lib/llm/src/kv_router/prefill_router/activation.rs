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

    // A fleet mid-rolling-update can disagree. First card wins, but a silent
    // split means half the fleet is routed on the other half's terms.
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
        Self::disabled_with_selector(model_manager, decode_router_mode, session_affinity_ttl_secs)
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
        decode_router: Option<Arc<KvRouter>>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        session_affinity_ttl_secs: Option<u64>,
        model_name: String,
        namespace: String,
        worker_monitor: Option<crate::discovery::KvWorkerMonitor>,
    ) -> Arc<Self> {
        Self::new_with_selector_factory(
            Some(activation_rx),
            model_manager,
            decode_router_mode,
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
            worker_monitor,
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
            decode_router_mode,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
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
        decode_router: Option<Arc<KvRouter<Sel>>>,
        worker_selector_factory: WorkerSelectorFactory<Sel>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        session_affinity_ttl_secs: Option<u64>,
        model_name: String,
        namespace: String,
        worker_monitor: Option<crate::discovery::KvWorkerMonitor>,
        task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
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
            decode_router_mode,
            session_affinity_ttl: session_affinity_ttl_secs.map(std::time::Duration::from_secs),
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
        let endpoint_id = endpoint.id();

        // Start runtime config watcher for this endpoint (needed for get_disaggregated_endpoint)
        // This must be done before creating the router so bootstrap info is available
        context
            .model_manager
            .get_or_create_runtime_config_watcher(&endpoint)
            .await?;

        let advertisement = Self::resolve_prefill_advertisement(context, &endpoint).await?;
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

        let inner_router = if prefill_router_mode.is_kv_routing() {
            // Create KV chooser using the endpoint (this is a prefill router)
            let effective_kv_router_config = prefill_kv_config.clone().unwrap_or_default();
            let selector = (context.worker_selector_factory)(
                &effective_kv_router_config,
                crate::worker_type::WorkerType::Prefill,
                RoutingPartitionRef::new(&context.model_name, DEFAULT_ROUTING_GROUP),
            );
            let kv_chooser = context
                .model_manager
                .kv_chooser_for_with_selector(
                    &endpoint,
                    prefill_block_size,
                    selector,
                    prefill_kv_config,
                    context.prefill_load_estimator.clone(),
                    Some(crate::worker_type::WorkerType::Prefill),
                    WORKER_TYPE_PREFILL,
                    Some(context.model_name.clone()),
                    advertisement.is_eagle,
                )
                .await?;

            // Extract client from kv_chooser to ensure shared state
            let client = kv_chooser.client().clone();
            let affinity =
                create_affinity_coordinator(prefill_session_affinity_ttl, client.clone()).await?;
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
                create_affinity_coordinator(prefill_session_affinity_ttl, client.clone()).await?;
            let prefill_client = client.clone();

            // Create simple push router with the resolved prefill router mode
            // Note: Per-worker metrics (active_prefill_tokens, active_decode_blocks) are only
            // available in KV routing mode where the router has actual bookkeeping.
            let push_router = PushRouter::<PreprocessedRequest, Annotated<LLMEngineOutput>>::from_client_with_monitor(
                client,
                prefill_router_mode,
                None, // worker_monitor
            )
            .await?;

            (
                InnerPrefillRouter::SimpleRouter(Arc::new(
                    crate::session_affinity::SessionAffinityPushRouter::new_with_coordinator(
                        push_router,
                        affinity,
                        prefill_router_mode.is_direct_routing(),
                    ),
                )),
                prefill_client,
            )
        };

        Ok((
            PrefillBinding {
                endpoint_id,
                router: inner_router.0,
                prefill_router_mode,
            },
            inner_router.1,
        ))
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
                decode_router_mode: router_ref.decode_router_mode,
                worker_selector_factory: router_ref
                    .worker_selector_factory
                    .clone()
                    .expect("enabled prefill router has a worker selector factory"),
                prefill_load_estimator: router_ref.prefill_load_estimator.clone(),
                session_affinity_ttl: router_ref.session_affinity_ttl,
                model_name: router_ref.model_name.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entrypoint::RouterConfig;
    use dynamo_kv_router::config::KvRouterConfig;

    fn card(router_config: Option<RouterConfig>) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only("test-model");
        card.router_config = router_config;
        card
    }

    fn card_with_block_size(block_size: u32) -> ModelDeploymentCard {
        let mut card = card(None);
        card.kv_cache_block_size = block_size;
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
    fn card_router_config_overrides_decode_mode() {
        // The headline case: KV-routed prefill in front of round-robin decode.
        let cards = vec![card(Some(RouterConfig::new(
            RouterMode::KV,
            KvRouterConfig::default(),
        )))];
        let resolved =
            resolve_advertisement_from_cards(&cards, RouterMode::RoundRobin).expect("resolves");
        assert_eq!(resolved.router_mode, RouterMode::KV);
        assert!(resolved.kv_router_config.is_some());
    }

    #[test]
    fn first_card_wins_when_the_fleet_disagrees() {
        // A rolling update can leave old (inheriting) and new (advertising)
        // prefill workers side by side. Resolution must stay deterministic
        // rather than depending on which card happened to be read.
        let cards = vec![
            card(Some(RouterConfig::new(
                RouterMode::KV,
                KvRouterConfig::default(),
            ))),
            card(None),
        ];
        let resolved =
            resolve_advertisement_from_cards(&cards, RouterMode::RoundRobin).expect("resolves");
        assert_eq!(resolved.router_mode, RouterMode::KV);
    }

    #[test]
    fn block_size_comes_from_the_prefill_card() {
        // The prefill pool's KV events are keyed on its own block size. Taking
        // the decode set's would index this pool at the wrong granularity.
        let cards = vec![card_with_block_size(64)];
        let resolved = resolve_advertisement_from_cards(&cards, RouterMode::KV).expect("resolves");
        assert_eq!(resolved.kv_cache_block_size, 64);
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
}
