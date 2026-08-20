// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use arc_swap::ArcSwap;
use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_kv_router::{
    PrefillLoadEstimator,
    config::KvRouterConfig,
    protocols::{KvTransferEnforcement, RoutingConstraints, WorkerId, WorkerWithDpRank},
    selector::{WorkerInputs, WorkerSelector},
};
use tokio_util::sync::CancellationToken;

use super::worker_monitor::LoadThresholdConfig;
use super::{
    KvSourceMembershipWatch, Model, RuntimeConfigWatch, WorkerSet,
    kv_source_watch::KvSourceMembershipCoordinator, runtime_config_watch,
};

use dynamo_runtime::{
    component::{Client, Endpoint, build_transport_type},
    discovery::{Discovery, DiscoverySpec, ModelCardInstanceId},
    pipeline::network::RequestPlanePayloadCodec,
    prelude::DistributedRuntimeProvider,
    protocols::EndpointId,
};

use crate::{
    kv_router::{
        KvEventSourceRequirement, KvRouter, router_endpoint_id, scheduler::DefaultWorkerSelector,
        shared_cache::HicacheSharedKvCache,
    },
    local_model::runtime_config::{
        DisaggregatedEndpoint, ModelRuntimeConfig, VLLM_INFERENCE_V1_GENERATE_CAPABILITY,
        topology_taint,
    },
    lora::state_tracker::LoraWorkerProjection,
    lora::{LoraFilter, LoraRoutingTable, LoraStateTracker, load_estimator::LoadEstimator},
    model_card::{LoraInfo, ModelDeploymentCard},
    types::{
        RealtimeBidirectionalEngine,
        generic::tensor::TensorStreamingEngine,
        openai::{
            audios::OpenAIAudiosStreamingEngine,
            chat_completions::OpenAIChatCompletionsStreamingEngine,
            classify::OpenAIClassifyStreamingEngine, completions::OpenAICompletionsStreamingEngine,
            embeddings::OpenAIEmbeddingsStreamingEngine, generate::GenerateStreamingEngine,
            images::OpenAIImagesStreamingEngine, pooling::OpenAIPoolingStreamingEngine,
            videos::OpenAIVideosStreamingEngine,
        },
    },
    worker_type::WorkerType,
};

struct LoraEndpointDomain {
    routing_table: LoraRoutingTable,
    state_tracker: LoraStateTracker,
    load_estimator: Arc<LoadEstimator>,
    filter: Arc<LoraFilter>,
    controller_started: AtomicBool,
    controller_cancel: parking_lot::Mutex<Option<tokio_util::sync::CancellationToken>>,
}

impl LoraEndpointDomain {
    fn new() -> Self {
        let routing_table = LoraRoutingTable::new();
        let state_tracker = LoraStateTracker::new();
        let filter = Arc::new(LoraFilter::new(
            routing_table.clone(),
            state_tracker.clone(),
        ));
        Self {
            routing_table,
            state_tracker,
            load_estimator: Arc::new(LoadEstimator::new()),
            filter,
            controller_started: AtomicBool::new(false),
            controller_cancel: parking_lot::Mutex::new(None),
        }
    }

    fn shutdown(&self) {
        if let Some(cancel) = self.controller_cancel.lock().take() {
            cancel.cancel();
        }
        self.controller_started.store(false, Ordering::Release);
        self.routing_table.clear();
        self.load_estimator.reset();
        self.state_tracker
            .replace_endpoint_projection(HashMap::new());
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelManagerError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model unavailable: {0}")]
    ModelUnavailable(String),

    #[error("Model already exists: {0}")]
    ModelAlreadyExists(String),
}

/// Sentinel label value used in frontend Prometheus metrics for requests
/// that target an unregistered model. Bounds label cardinality so arbitrary
/// client-supplied model strings cannot create unbounded Prometheus series.
/// The `_model` suffix makes accidental collision with a real model name
/// implausible while keeping the value readable in Grafana dropdowns.
pub const UNKNOWN_METRIC_MODEL: &str = "unknown_model";

#[derive(Default)]
struct CommittedCatalog {
    models: Arc<HashMap<String, Arc<Model>>>,
    cards: Arc<HashMap<String, Arc<ModelDeploymentCard>>>,
    aliases: Arc<HashMap<String, String>>,
}

struct CommittedDiscoveryGroup {
    primary: String,
    namespace: String,
    worker_set_key: String,
    aliases: Vec<String>,
    cards: HashMap<String, ModelDeploymentCard>,
    adapters: HashMap<String, ModelDeploymentCard>,
    representative: ModelDeploymentCard,
    worker_set: Arc<WorkerSet>,
}

#[derive(Default)]
struct PendingLoraProjection {
    base_capacities: Vec<u32>,
    adapters: HashMap<String, LoraInfo>,
}

type EndpointLoraProjection = HashMap<EndpointId, HashMap<WorkerWithDpRank, LoraWorkerProjection>>;

pub(crate) struct RemovedDiscoveryGroup {
    pub(crate) representative: ModelDeploymentCard,
    pub(crate) cards: Vec<ModelDeploymentCard>,
}

/// Central manager for model engines, routing, and configuration.
///
/// Models are stored hierarchically: ModelManager → Model → WorkerSet.
/// Each WorkerSet owns a complete pipeline built from its specific configuration.
///
/// Note: Don't implement Clone for this, put it in an Arc instead.
pub struct ModelManager {
    /// Model name → Model (which contains WorkerSets with engines)
    models: DashMap<String, Arc<Model>>,

    /// Atomically published request-plane view. Discovery lifecycle changes are assembled in the
    /// mutable maps above and become visible with one pointer swap.
    catalog: ArcSwap<CommittedCatalog>,

    /// Per-instance model cards, keyed by instance path. Used for cleanup on worker removal.
    cards: DashMap<String, Arc<ModelDeploymentCard>>,

    /// Controller-owned committed groups, keyed by the controller's stable GroupKey encoding.
    discovery_groups: DashMap<String, CommittedDiscoveryGroup>,

    /// Per-endpoint runtime config watchers. Keyed by EndpointId (includes namespace).
    ///
    /// NOTE: These shared receivers currently live for the manager lifetime. Rebinding to a new
    /// endpoint therefore leaves the previous watcher cached; safe eviction requires shared
    /// ownership tracking because multiple routers may consume the same endpoint watch.
    runtime_configs: DashMap<EndpointId, RuntimeConfigWatch>,

    /// Per-endpoint HiCache state and its one Mooncake event subscriber.
    hicache_caches: DashMap<EndpointId, HicacheSharedKvCache>,

    /// Shared KV-source membership coordinators, scoped by exact serving endpoint.
    /// Weak ownership lets the discovery loop stop when its last consumer goes away.
    kv_source_memberships: DashMap<EndpointId, Weak<KvSourceMembershipCoordinator>>,

    /// Exact endpoint → independent LoRA allocation and load domain.
    lora_domains: DashMap<EndpointId, Arc<LoraEndpointDomain>>,
    committed_lora_endpoints: parking_lot::Mutex<HashSet<EndpointId>>,
    lora_enabled: bool,
    /// Per-decode-endpoint LoRA load-feed subscription handles, so we start exactly one feed
    /// per endpoint and can restart it if the previous one exited (avoids double counting on
    /// rebuilds while keeping the feed durable).
    lora_load_feeds: DashMap<String, tokio::task::JoinHandle<()>>,
    lora_controller_cancel: parking_lot::Mutex<Option<tokio_util::sync::CancellationToken>>,

    /// Alias → primary model name mapping. Used to normalize metrics labels.
    alias_to_primary: DashMap<String, String>,

    /// Serializes name-reservation transitions — the primary claim in
    /// [`Self::add_worker_set`] and the alias claim in [`Self::register_alias`] —
    /// so a name cannot be concurrently claimed as both a primary and an alias.
    /// A cold-path lock (worker registration, not request serving), uncontended
    /// in steady state; held only across in-memory map reads/writes, never across
    /// an `.await`.
    reservation_lock: parking_lot::Mutex<()>,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: DashMap::new(),
            catalog: ArcSwap::from_pointee(CommittedCatalog::default()),
            cards: DashMap::new(),
            discovery_groups: DashMap::new(),
            runtime_configs: DashMap::new(),
            hicache_caches: DashMap::new(),
            kv_source_memberships: DashMap::new(),
            lora_domains: DashMap::new(),
            committed_lora_endpoints: parking_lot::Mutex::new(HashSet::new()),
            lora_enabled: crate::lora::lora_serving_enabled(),
            lora_load_feeds: DashMap::new(),
            lora_controller_cancel: parking_lot::Mutex::new(None),
            alias_to_primary: DashMap::new(),
            reservation_lock: parking_lot::Mutex::new(()),
        }
    }

    fn publish_catalog_locked(&self) {
        let models = self
            .models
            .iter()
            .map(|entry| (entry.key().clone(), Arc::new(entry.value().snapshot())))
            .collect();
        let cards = self
            .cards
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        let aliases = self
            .alias_to_primary
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        self.catalog.store(Arc::new(CommittedCatalog {
            models: Arc::new(models),
            cards: Arc::new(cards),
            aliases: Arc::new(aliases),
        }));
    }

    fn lora_projection_locked(&self) -> EndpointLoraProjection {
        let mut pending: HashMap<EndpointId, HashMap<WorkerWithDpRank, PendingLoraProjection>> =
            HashMap::new();
        for group in self.discovery_groups.iter() {
            for (key, card) in &group.cards {
                let Ok(mcid) = ModelCardInstanceId::from_path(key) else {
                    continue;
                };
                let endpoint_id = EndpointId {
                    namespace: mcid.namespace.clone(),
                    component: mcid.component.clone(),
                    name: mcid.endpoint.clone(),
                };
                let worker = WorkerWithDpRank::new(mcid.instance_id, 0);
                let worker_projection = pending
                    .entry(endpoint_id)
                    .or_default()
                    .entry(worker)
                    .or_default();
                if let Some(capacity) = card.runtime_config.max_gpu_lora_count {
                    worker_projection.base_capacities.push(capacity);
                }

                for (adapter_key, adapter_card) in &group.adapters {
                    let Ok(adapter_mcid) = ModelCardInstanceId::from_path(adapter_key) else {
                        continue;
                    };
                    if adapter_mcid.namespace != mcid.namespace
                        || adapter_mcid.component != mcid.component
                        || adapter_mcid.endpoint != mcid.endpoint
                        || adapter_mcid.instance_id != mcid.instance_id
                    {
                        continue;
                    }
                    if let Some(lora) = &adapter_card.lora {
                        worker_projection
                            .adapters
                            .insert(lora.name.clone(), lora.clone());
                    }
                }
            }
        }

        pending
            .into_iter()
            .map(|(endpoint_id, workers)| {
                let workers = workers
                    .into_iter()
                    .filter_map(|(worker, mut projection)| {
                        projection.base_capacities.sort_unstable();
                        projection.base_capacities.dedup();
                        if projection.base_capacities.len() > 1 {
                            tracing::warn!(
                                endpoint = %endpoint_id,
                                worker_id = worker.worker_id,
                                capacities = ?projection.base_capacities,
                                "Base MDCs disagree on LoRA capacity; using the conservative minimum"
                            );
                        }
                        let mut loras = projection.adapters.into_values().collect::<Vec<_>>();
                        loras.sort_by(|left, right| left.name.cmp(&right.name));
                        let mut adapter_capacities = loras
                            .iter()
                            .filter_map(|lora| lora.max_gpu_lora_count)
                            .collect::<Vec<_>>();
                        adapter_capacities.sort_unstable();
                        adapter_capacities.dedup();
                        if adapter_capacities.len() > 1 {
                            tracing::warn!(
                                endpoint = %endpoint_id,
                                worker_id = worker.worker_id,
                                capacities = ?adapter_capacities,
                                "Adapter MDCs disagree on LoRA capacity; using the conservative minimum"
                            );
                        }
                        let capacity = projection
                            .base_capacities
                            .first()
                            .copied()
                            .or_else(|| adapter_capacities.first().copied())
                            .or_else(|| (!loras.is_empty()).then_some(4))?;
                        Some((worker, LoraWorkerProjection { capacity, loras }))
                    })
                    .collect();
                (endpoint_id, workers)
            })
            .collect()
    }

    fn union_lora_projection(
        before: &EndpointLoraProjection,
        after: &EndpointLoraProjection,
    ) -> EndpointLoraProjection {
        let mut union = before.clone();
        for (endpoint, workers) in after {
            let endpoint_union = union.entry(endpoint.clone()).or_default();
            for (worker, projection) in workers {
                let Some(existing) = endpoint_union.get_mut(worker) else {
                    endpoint_union.insert(*worker, projection.clone());
                    continue;
                };
                existing.capacity = existing.capacity.min(projection.capacity);
                let mut loras = existing
                    .loras
                    .iter()
                    .chain(&projection.loras)
                    .map(|lora| (lora.name.clone(), lora.clone()))
                    .collect::<HashMap<_, _>>()
                    .into_values()
                    .collect::<Vec<_>>();
                loras.sort_by(|left, right| left.name.cmp(&right.name));
                existing.loras = loras;
            }
        }
        union
    }

    fn publish_lora_projection_locked(&self, projection: EndpointLoraProjection) {
        let mut endpoints = self.committed_lora_endpoints.lock();
        let next_endpoints = projection.keys().cloned().collect::<HashSet<_>>();
        for endpoint_id in endpoints.union(&next_endpoints) {
            let workers = projection.get(endpoint_id).cloned().unwrap_or_default();
            if let Some(domain) = self.lora_domains.get(endpoint_id) {
                domain.state_tracker.replace_endpoint_projection(workers);
            } else if !workers.is_empty() {
                self.lora_domain(endpoint_id)
                    .state_tracker
                    .replace_endpoint_projection(workers);
            }
        }
        *endpoints = next_endpoints;
        drop(endpoints);
        self.prune_idle_lora_domains();
    }

    fn prune_idle_lora_domains(&self) {
        let committed = self.committed_lora_endpoints.lock().clone();
        let removable = self
            .lora_domains
            .iter()
            .filter_map(|entry| {
                (!committed.contains(entry.key())
                    && entry.state_tracker.is_empty()
                    && Arc::strong_count(&entry.filter) == 1)
                    .then(|| entry.key().clone())
            })
            .collect::<Vec<_>>();
        for endpoint_id in removable {
            let Some((_, domain)) = self.lora_domains.remove(&endpoint_id) else {
                continue;
            };
            domain.shutdown();
            if let Some((_, feed)) = self.lora_load_feeds.remove(&endpoint_id.to_string()) {
                feed.abort();
            }
        }
    }

    // -- Model access --

    /// Get or create a Model for the given name.
    pub fn get_or_create_model(&self, model_name: &str) -> Arc<Model> {
        self.models
            .entry(model_name.to_string())
            .or_insert_with(|| Arc::new(Model::new(model_name.to_string())))
            .clone()
    }

    /// Get an existing Model, if it exists.
    pub fn get_model(&self, model_name: &str) -> Option<Arc<Model>> {
        self.models
            .get(model_name)
            .map(|entry| entry.value().clone())
    }

    pub(crate) fn get_committed_model(&self, model_name: &str) -> Option<Arc<Model>> {
        self.catalog.load().models.get(model_name).cloned()
    }

    fn get_model_internal(&self, model_name: &str) -> Option<Arc<Model>> {
        self.models
            .get(model_name)
            .map(|entry| entry.value().clone())
    }

    /// Remove a Model if it has no remaining WorkerSets.
    ///
    /// The caller holds `reservation_lock` and publishes the resulting catalog update.
    /// Uses atomic remove_if to avoid TOCTOU race between checking is_empty and removing.
    fn remove_model_if_empty(&self, model_name: &str) {
        if self
            .models
            .remove_if(model_name, |_, model| model.is_empty())
            .is_some()
        {
            tracing::info!(model_name, "Removed empty model from manager");
        }
    }

    /// Add a WorkerSet to a Model under its primary name. Creates the Model if it
    /// doesn't exist. Returns `false` (registering nothing) when `model_name` is
    /// already reserved as another deployment's alias.
    ///
    /// The names a live deployment holds — its primary plus every alias — are
    /// globally reserved until it is removed, so a later deployment cannot claim
    /// any of them, as either a primary or an alias. This is the primary-side
    /// mirror of [`Self::register_alias`] (which rejects an alias colliding with a
    /// live primary or another primary's alias); together they make name
    /// reservation first-come and symmetric across namespaces. A later deployment
    /// re-using a name fails loudly rather than silently displacing the owner.
    ///
    /// Holds `Self::reservation_lock` across the reserved-name check and the
    /// insert so the claim is atomic against a concurrent `register_alias` for
    /// the same name (a name can never end up both a live primary and an alias).
    /// The lock is always taken before any map access, so it never inverts with a
    /// DashMap shard lock.
    pub fn add_worker_set(&self, model_name: &str, namespace: &str, worker_set: WorkerSet) -> bool {
        let _reservation = self.reservation_lock.lock();
        if let Some(reserved_by) = self.alias_to_primary.get(model_name) {
            tracing::warn!(
                model_name,
                reserved_by = reserved_by.value().as_str(),
                "Model name is already reserved as an alias of another deployment — refusing to \
                 register. Choose a different name or remove the conflicting deployment."
            );
            return false;
        }
        let model = self.get_or_create_model(model_name);
        let topology_namespace = worker_set.namespace().to_string();
        model.add_worker_set(namespace.to_string(), Arc::new(worker_set));
        self.reconcile_discovery_topology(model_name, &topology_namespace);
        self.publish_catalog_locked();
        true
    }

    /// Add an already-Arc-wrapped WorkerSet to a Model. Creates the Model if it doesn't exist.
    /// Used to register the same WorkerSet under multiple model names (aliases).
    ///
    /// Logs a warning and skips if a *different* primary already owns this name —
    /// this guards against operator misconfiguration where two unrelated models
    /// declare a colliding alias. The first claim wins; the second is rejected.
    pub fn add_worker_set_arc(
        &self,
        model_name: &str,
        namespace: &str,
        worker_set: Arc<WorkerSet>,
    ) -> bool {
        let _reservation = self.reservation_lock.lock();
        // Collision check: if `model_name` already exists as a primary (i.e.
        // already has worker sets AND is not currently an alias), refuse to
        // clobber it. The two facts are read one map at a time — the `models`
        // guard is dropped before touching `alias_to_primary` — so this never
        // holds one shard lock while acquiring the other (register_alias probes
        // them in the opposite order; holding across would risk a deadlock).
        let is_live_primary = self
            .models
            .get(model_name)
            .is_some_and(|existing| !existing.is_empty());
        if is_live_primary && !self.alias_to_primary.contains_key(model_name) {
            tracing::warn!(
                alias = model_name,
                namespace,
                "Alias collides with a registered primary model — skipping. \
                 Choose a different alias or rename the conflicting model."
            );
            return false;
        }

        let model = self.get_or_create_model(model_name);
        model.add_worker_set(namespace.to_string(), worker_set);
        self.publish_catalog_locked();
        true
    }

    /// Record that `alias` is an alternate name for `primary`. Used to normalize metrics labels.
    ///
    /// The claim is taken atomically through the map entry so two concurrent
    /// registrations of the same alias cannot both succeed. First-write-wins:
    /// re-registering the same alias→primary is idempotent, but a conflicting
    /// primary (or a name already owned by a registered primary model) is
    /// refused and logged so operators find the collision in the logs rather
    /// than through silent metric re-attribution.
    ///
    /// Holds `Self::reservation_lock` across the live-primary probe and the
    /// entry insert so the claim is atomic against a concurrent `add_worker_set`
    /// for the same name. Within that section the `models` guard is dropped before
    /// touching `alias_to_primary` (via `is_some_and`), and the lock is taken
    /// before any map access, so no DashMap shard lock is ever held across another.
    pub fn register_alias(&self, alias: &str, primary: &str) -> bool {
        let _reservation = self.reservation_lock.lock();
        if self
            .models
            .get(alias)
            .is_some_and(|model| !model.is_empty())
            && !self.alias_to_primary.contains_key(alias)
        {
            tracing::warn!(
                alias,
                primary,
                "Alias collides with a registered primary model — refusing to register. \
                 Choose a different alias or rename the conflicting model."
            );
            return false;
        }

        match self.alias_to_primary.entry(alias.to_string()) {
            Entry::Occupied(existing) => {
                if existing.get() != primary {
                    tracing::warn!(
                        alias,
                        new_primary = primary,
                        existing_primary = existing.get().as_str(),
                        "Alias is already claimed by a different primary — refusing to overwrite. \
                         Existing claim wins."
                    );
                    return false;
                }
                // Same alias→same primary — idempotent, no-op.
                true
            }
            Entry::Vacant(slot) => {
                slot.insert(primary.to_string());
                self.publish_catalog_locked();
                true
            }
        }
    }

    /// Remove a previously registered alias mapping once the alias has no WorkerSets.
    pub fn unregister_alias_if_empty(&self, alias: &str, primary: &str) {
        let _reservation = self.reservation_lock.lock();
        if self
            .models
            .get(alias)
            .is_some_and(|model| !model.is_empty())
        {
            return;
        }

        self.alias_to_primary
            .remove_if(alias, |_, existing| existing == primary);
        self.publish_catalog_locked();
    }

    /// Return the primary (canonical) model name for `model`, resolving aliases.
    /// Returns `model` unchanged if it is not an alias.
    pub fn resolve_canonical_name(&self, model: &str) -> String {
        self.catalog
            .load()
            .aliases
            .get(model)
            .cloned()
            .unwrap_or_else(|| model.to_string())
    }

    /// Whether `alias` is currently reserved as an alias of `primary`. Teardown
    /// uses this to clean up only the alias names a deployment actually owns.
    pub fn alias_belongs_to(&self, alias: &str, primary: &str) -> bool {
        self.alias_to_primary
            .get(alias)
            .is_some_and(|owner| owner.value() == primary)
    }

    /// Remove a WorkerSet from a Model. Removes the Model if it becomes empty.
    pub fn remove_worker_set(&self, model_name: &str, namespace: &str) -> Option<Arc<WorkerSet>> {
        let _reservation = self.reservation_lock.lock();
        let model = self.models.get(model_name)?;
        let removed = model.remove_worker_set(namespace);
        let topology_namespace = removed
            .as_ref()
            .map(|worker_set| worker_set.namespace().to_string());
        if let Some(worker_set) = &removed {
            Self::clear_worker_set_targets(worker_set);
        }
        drop(model);
        if let Some(topology_namespace) = topology_namespace {
            self.reconcile_discovery_topology(model_name, &topology_namespace);
        }
        self.remove_model_if_empty(model_name);
        self.publish_catalog_locked();
        removed
    }

    /// Commit a complete discovery group atomically.
    ///
    pub(crate) fn commit_discovery_group(
        &self,
        group_id: &str,
        worker_set_key: &str,
        worker_set: WorkerSet,
        members: Vec<(String, ModelDeploymentCard)>,
        adapters: Vec<(String, ModelDeploymentCard)>,
    ) -> anyhow::Result<()> {
        let representative = members
            .first()
            .map(|(_, card)| card)
            .ok_or_else(|| anyhow::anyhow!("cannot commit an empty discovery group"))?;
        let primary = representative.name().to_string();
        let aliases = representative
            .aliases
            .iter()
            .filter(|alias| alias.as_str() != primary)
            .cloned()
            .collect::<Vec<_>>();
        let namespace = worker_set.namespace().to_string();
        let representative = representative.clone();

        let _reservation = self.reservation_lock.lock();
        let lora_before = self.lora_projection_locked();
        anyhow::ensure!(
            !self.discovery_groups.contains_key(group_id),
            "discovery group {group_id:?} is already committed"
        );
        if let Some(owner) = self.alias_to_primary.get(&primary) {
            anyhow::bail!(
                "model name {primary:?} is reserved as an alias of {:?}",
                owner.value()
            );
        }
        anyhow::ensure!(
            !self.discovery_groups.iter().any(|entry| entry
                .adapters
                .values()
                .any(|adapter| adapter.name() == primary)),
            "model name {primary:?} is reserved by a LoRA adapter"
        );

        for alias in &aliases {
            if let Some(owner) = self.alias_to_primary.get(alias)
                && owner.value() != &primary
            {
                anyhow::bail!("alias {alias:?} is already reserved by {:?}", owner.value());
            }
            let is_other_primary = self
                .models
                .get(alias)
                .is_some_and(|model| !model.is_empty())
                && self
                    .alias_to_primary
                    .get(alias)
                    .is_none_or(|owner| owner.value() != &primary);
            anyhow::ensure!(
                !is_other_primary,
                "alias {alias:?} collides with a registered primary model"
            );
        }
        self.validate_adapter_claims(&primary, adapters.iter().map(|(_, card)| card))?;

        let worker_set = Arc::new(worker_set);
        self.get_or_create_model(&primary)
            .add_worker_set(worker_set_key.to_string(), worker_set.clone());
        for alias in &aliases {
            self.alias_to_primary.insert(alias.clone(), primary.clone());
            self.get_or_create_model(alias)
                .add_worker_set(worker_set_key.to_string(), worker_set.clone());
        }
        for (_, adapter) in &adapters {
            let adapter_view = Arc::new(worker_set.adapter_view(adapter.clone()));
            self.get_or_create_model(adapter.name())
                .add_worker_set(worker_set_key.to_string(), adapter_view);
        }

        let cards = members.into_iter().collect::<HashMap<_, _>>();
        for (key, card) in &cards {
            self.cards.insert(key.clone(), Arc::new(card.clone()));
        }
        let adapters = adapters.into_iter().collect::<HashMap<_, _>>();
        for (key, card) in &adapters {
            self.cards.insert(key.clone(), Arc::new(card.clone()));
        }
        self.discovery_groups.insert(
            group_id.to_string(),
            CommittedDiscoveryGroup {
                primary: primary.clone(),
                namespace: namespace.clone(),
                worker_set_key: worker_set_key.to_string(),
                aliases,
                cards,
                adapters,
                representative,
                worker_set,
            },
        );
        let lora_after = self.lora_projection_locked();
        self.publish_lora_projection_locked(Self::union_lora_projection(&lora_before, &lora_after));
        self.reconcile_discovery_topology(&primary, &namespace);
        self.publish_catalog_locked();
        self.publish_lora_projection_locked(lora_after);
        Ok(())
    }

    fn validate_adapter_claims<'a>(
        &self,
        primary: &str,
        adapters: impl Iterator<Item = &'a ModelDeploymentCard>,
    ) -> anyhow::Result<()> {
        for adapter in adapters {
            let lora = adapter
                .lora
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("adapter card is missing LoRA metadata"))?;
            anyhow::ensure!(!lora.name.is_empty(), "LoRA adapter name cannot be empty");
            let name = adapter.name();
            anyhow::ensure!(
                name != primary,
                "LoRA adapter name {name:?} collides with its base model"
            );
            anyhow::ensure!(
                !self.alias_to_primary.contains_key(name),
                "LoRA adapter name {name:?} collides with a registered alias"
            );
            let owned_by_same_base = self.discovery_groups.iter().any(|entry| {
                entry.primary == primary
                    && entry
                        .adapters
                        .values()
                        .any(|existing| existing.name() == name)
            });
            let collides_with_primary =
                self.models.get(name).is_some_and(|model| !model.is_empty()) && !owned_by_same_base;
            anyhow::ensure!(
                !collides_with_primary,
                "LoRA adapter name {name:?} collides with a registered model"
            );
        }
        Ok(())
    }

    pub(crate) fn replace_discovery_group(
        &self,
        group_id: &str,
        members: Vec<(String, ModelDeploymentCard)>,
        adapters: Vec<(String, ModelDeploymentCard)>,
    ) -> anyhow::Result<()> {
        let _reservation = self.reservation_lock.lock();
        let group = self
            .discovery_groups
            .get(group_id)
            .ok_or_else(|| anyhow::anyhow!("committed discovery group {group_id:?} not found"))?;
        let primary = group.primary.clone();
        let worker_set_key = group.worker_set_key.clone();
        let worker_set = group.worker_set.clone();
        let previous_member_keys = group.cards.keys().cloned().collect::<HashSet<_>>();
        let previous_adapter_keys = group.adapters.keys().cloned().collect::<HashSet<_>>();
        let previous_adapter_names = group
            .adapters
            .values()
            .map(|card| card.name().to_string())
            .collect::<HashSet<_>>();
        drop(group);

        anyhow::ensure!(
            !members.is_empty(),
            "cannot replace with an empty discovery group"
        );
        self.validate_adapter_claims(&primary, adapters.iter().map(|(_, card)| card))?;
        let members = members.into_iter().collect::<HashMap<_, _>>();
        let adapters = adapters.into_iter().collect::<HashMap<_, _>>();
        let desired_member_keys = members.keys().cloned().collect::<HashSet<_>>();
        let desired_adapter_keys = adapters.keys().cloned().collect::<HashSet<_>>();
        let desired_adapter_names = adapters
            .values()
            .map(|card| card.name().to_string())
            .collect::<HashSet<_>>();
        let adapter_views = adapters
            .values()
            .map(|card| {
                (
                    card.name().to_string(),
                    Arc::new(worker_set.adapter_view(card.clone())),
                )
            })
            .collect::<HashMap<_, _>>();
        let lora_before = self.lora_projection_locked();

        for key in previous_member_keys.difference(&desired_member_keys) {
            self.cards.remove(key);
        }
        for key in previous_adapter_keys.difference(&desired_adapter_keys) {
            self.cards.remove(key);
        }
        for (key, card) in members.iter().chain(adapters.iter()) {
            self.cards.insert(key.clone(), Arc::new(card.clone()));
        }

        let mut group = self
            .discovery_groups
            .get_mut(group_id)
            .ok_or_else(|| anyhow::anyhow!("committed discovery group {group_id:?} not found"))?;
        group.representative = members
            .values()
            .next()
            .cloned()
            .expect("non-empty members checked above");
        group.cards = members;
        group.adapters = adapters;
        drop(group);

        for (name, adapter_view) in adapter_views {
            self.get_or_create_model(&name)
                .add_worker_set(worker_set_key.clone(), adapter_view);
        }
        for name in previous_adapter_names.difference(&desired_adapter_names) {
            if let Some(model) = self.models.get(name) {
                model.remove_worker_set(&worker_set_key);
            }
            self.remove_model_if_empty(name);
        }
        let lora_after = self.lora_projection_locked();
        self.publish_lora_projection_locked(Self::union_lora_projection(&lora_before, &lora_after));
        self.publish_catalog_locked();
        self.publish_lora_projection_locked(lora_after);
        Ok(())
    }

    pub(crate) fn discovery_group_adapter_cards(&self, group_id: &str) -> Vec<ModelDeploymentCard> {
        self.discovery_groups
            .get(group_id)
            .map(|group| group.adapters.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Remove a complete discovery group atomically.
    ///
    pub(crate) fn remove_discovery_group(&self, group_id: &str) -> Option<RemovedDiscoveryGroup> {
        let _reservation = self.reservation_lock.lock();
        let lora_before = self.lora_projection_locked();
        let (_, group) = self.discovery_groups.remove(group_id)?;
        let representative = group.representative.clone();
        let topology_namespace = group.namespace.clone();

        for key in group.cards.keys() {
            self.cards.remove(key);
        }
        for key in group.adapters.keys() {
            self.cards.remove(key);
        }
        if let Some(model) = self.models.get(&group.primary)
            && let Some(worker_set) = model.remove_worker_set(&group.worker_set_key)
        {
            Self::clear_worker_set_targets(&worker_set);
        }
        self.remove_model_if_empty(&group.primary);
        for alias in &group.aliases {
            if let Some(model) = self.models.get(alias) {
                model.remove_worker_set(&group.worker_set_key);
            }
            self.remove_model_if_empty(alias);
            if self.models.get(alias).is_none_or(|model| model.is_empty()) {
                self.alias_to_primary
                    .remove_if(alias, |_, owner| owner == &group.primary);
            }
        }
        for adapter in group.adapters.values() {
            if let Some(model) = self.models.get(adapter.name()) {
                model.remove_worker_set(&group.worker_set_key);
            }
            self.remove_model_if_empty(adapter.name());
        }
        let cards = group
            .cards
            .into_values()
            .chain(group.adapters.into_values())
            .collect();
        let removed = RemovedDiscoveryGroup {
            representative,
            cards,
        };
        let lora_after = self.lora_projection_locked();
        self.publish_lora_projection_locked(Self::union_lora_projection(&lora_before, &lora_after));
        self.reconcile_discovery_topology(&group.primary, &topology_namespace);
        self.publish_catalog_locked();
        self.publish_lora_projection_locked(lora_after);
        Some(removed)
    }

    // -- Model cards --

    pub fn get_model_cards(&self) -> Vec<ModelDeploymentCard> {
        self.catalog
            .load()
            .cards
            .values()
            .map(|card| (**card).clone())
            .collect()
    }

    /// Return owned keys for cards in the published catalog.
    pub fn get_model_card_keys(&self) -> Vec<String> {
        self.catalog.load().cards.keys().cloned().collect()
    }

    /// Compatibility path for explicit in-process callers. Discovery uses the atomic group
    /// lifecycle methods above and never publishes a card independently.
    pub fn save_model_card(&self, key: &str, card: ModelDeploymentCard) -> anyhow::Result<()> {
        let _reservation = self.reservation_lock.lock();
        self.cards.insert(key.to_string(), Arc::new(card));
        self.publish_catalog_locked();
        Ok(())
    }

    /// Remove and return model card for this instance's key. We do this when the instance stops.
    pub fn get_model_card(&self, key: &str) -> Option<ModelDeploymentCard> {
        self.catalog
            .load()
            .cards
            .get(key)
            .map(|card| (**card).clone())
    }

    pub fn remove_model_card(&self, key: &str) -> Option<ModelDeploymentCard> {
        let _reservation = self.reservation_lock.lock();
        let removed = self.cards.remove(key).map(|(_, value)| (*value).clone());
        if removed.is_some() {
            self.publish_catalog_locked();
        }
        removed
    }

    // -- Engine accessors (delegate through Model → WorkerSet) --

    /// Check if a decode model (chat or completions) is registered
    pub fn has_decode_model(&self, model: &str) -> bool {
        self.catalog
            .load()
            .models
            .get(model)
            .is_some_and(|m| m.has_decode_engine())
    }

    /// Check if a prefill model is registered
    pub fn has_prefill_model(&self, model: &str) -> bool {
        self.catalog
            .load()
            .models
            .get(model)
            .is_some_and(|m| m.has_prefill())
    }

    /// Check if any model (decode or prefill) is registered.
    pub fn has_model_any(&self, model: &str) -> bool {
        self.has_decode_model(model) || self.has_prefill_model(model)
    }

    /// Check if any engine (chat, completions, embeddings, images, etc.) is
    /// registered under this exact model name. Case-sensitive. Distinct from
    /// [`has_model_any`](Self::has_model_any), which checks specifically for a
    /// decode or prefill engine.
    pub fn has_registered_model(&self, model: &str) -> bool {
        self.catalog.load().models.contains_key(model)
    }

    /// Resolve the model name to use in frontend Prometheus metrics.
    ///
    /// Returns the user-supplied name if a model is registered under it
    /// (preserving original casing), otherwise returns the bounded sentinel
    /// [`UNKNOWN_METRIC_MODEL`]. Callers should use this resolved name
    /// for every metric child created before engine lookup so unknown-model
    /// requests do not pollute Prometheus label cardinality.
    pub fn metric_model_for<'a>(&self, model: &'a str) -> &'a str {
        if self.has_registered_model(model) {
            model
        } else {
            UNKNOWN_METRIC_MODEL
        }
    }

    /// Whether `model` has at least one WorkerSet that can serve an inference
    /// request right now. See [`Model::is_ready_to_serve`].
    pub fn is_model_ready_to_serve(&self, model: &str) -> bool {
        self.catalog
            .load()
            .models
            .get(model)
            .is_some_and(|m| m.is_ready_to_serve())
    }

    /// Whether any registered model can serve at least one inference request
    /// right now. See [`Model::is_ready_to_serve`].
    pub fn has_any_ready_model(&self) -> bool {
        self.catalog
            .load()
            .models
            .values()
            .any(|model| model.is_ready_to_serve())
    }

    pub fn model_display_names(&self) -> HashSet<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.is_displayable())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Display names filtered to models that can actually serve a request right
    /// now — displayable AND with a complete worker set in at least one
    /// namespace ([`Model::has_ready_workers`]). This is the gate the HTTP
    /// listing/default-model paths should apply so a registered-but-incomplete
    /// deployment (e.g. decode-only with no prefill peer) is neither advertised
    /// nor chosen as an implicit default.
    pub fn serving_ready_display_names(&self) -> HashSet<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.is_displayable() && model.has_ready_workers())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_chat_completions_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_chat_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_completions_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_completions_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_embeddings_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_embeddings_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_classify_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_classify_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_pooling_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_pooling_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_tensor_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_tensor_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_images_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_images_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_audios_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_audios_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_videos_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_videos_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_realtime_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_realtime_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_generate_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_generate_engine())
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// List Generate models with an engine that advertises `capability`.
    pub fn list_generate_models_for_capability(&self, capability: &str) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_generate_engine_for_capability(capability))
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn list_prefill_models(&self) -> Vec<String> {
        self.catalog
            .load()
            .models
            .iter()
            .filter(|(_, model)| model.has_prefill())
            .map(|(name, _)| name.clone())
            .collect()
    }

    pub fn get_embeddings_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIEmbeddingsStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_embeddings_engine()
    }

    pub fn get_classify_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIClassifyStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_classify_engine()
    }

    pub fn get_pooling_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIPoolingStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_pooling_engine()
    }

    pub fn get_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAICompletionsStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_completions_engine()
    }

    pub fn get_chat_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIChatCompletionsStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_chat_engine()
    }

    pub fn get_tensor_engine(
        &self,
        model: &str,
    ) -> Result<TensorStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_tensor_engine()
    }

    pub fn get_images_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIImagesStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_images_engine()
    }

    pub fn get_videos_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIVideosStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_videos_engine()
    }

    pub fn get_audios_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIAudiosStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_audios_engine()
    }

    pub fn get_realtime_engine(
        &self,
        model: &str,
    ) -> Result<RealtimeBidirectionalEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_realtime_engine()
    }

    pub fn get_generate_engine(
        &self,
        model: &str,
    ) -> Result<GenerateStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_generate_engine()
    }
    /// Get a Generate engine for `model` from a worker advertising `capability`.
    pub fn get_generate_engine_for_capability(
        &self,
        model: &str,
        capability: &str,
    ) -> Result<GenerateStreamingEngine, ModelManagerError> {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_generate_engine_for_capability(capability)
    }

    // -- Combined engine + parsing options (atomically from one WorkerSet) --

    pub fn get_chat_completions_engine_with_parsing(
        &self,
        model: &str,
    ) -> Result<
        (
            OpenAIChatCompletionsStreamingEngine,
            crate::protocols::openai::ParsingOptions,
        ),
        ModelManagerError,
    > {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_chat_engine_with_parsing()
    }

    pub fn get_completions_engine_with_parsing(
        &self,
        model: &str,
    ) -> Result<
        (
            OpenAICompletionsStreamingEngine,
            crate::protocols::openai::ParsingOptions,
        ),
        ModelManagerError,
    > {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_completions_engine_with_parsing()
    }

    pub fn get_generate_engine_with_parsing(
        &self,
        model: &str,
    ) -> Result<
        (
            GenerateStreamingEngine,
            crate::protocols::openai::ParsingOptions,
        ),
        ModelManagerError,
    > {
        self.catalog
            .load()
            .models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_generate_engine_with_parsing()
    }

    // -- Convenience methods for in-process models (http.rs, grpc.rs) --
    // These create a WorkerSet with a default namespace for local models.
    // Synthetic in-process worker sets are always `Aggregated` (they own
    // their engine inline and don't depend on a peer worker), so we stamp
    // that role onto the card here. The `Prefill` helper, in contrast,
    // tags itself with `WorkerType::Prefill` so the serving-readiness
    // gate sees it correctly.
    // TODO: These methods use ModelDeploymentCard::default() for the WorkerSet, which means
    // parsing_options() returns defaults (no tool_call_parser/reasoning_parser). Pass the real
    // MDC from callers so ParsingOptions reflect the model's actual configuration.

    fn aggregated_local_card() -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Aggregated);
        card.needs = Vec::new();
        card
    }

    pub fn add_chat_completions_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIChatCompletionsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_chat_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_chat_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.chat_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_completions_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAICompletionsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_completions_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_completions_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.completions_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_embeddings_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIEmbeddingsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_embeddings_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_embeddings_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.embeddings_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_classify_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIClassifyStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_classify_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_classify_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.classify_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_pooling_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIPoolingStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_pooling_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_pooling_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.pooling_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_tensor_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: TensorStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_tensor_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_tensor_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.tensor_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_images_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIImagesStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_images_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_images_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.images_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_videos_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIVideosStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_videos_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_videos_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.videos_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_audios_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIAudiosStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_audios_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_audios_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.audios_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_realtime_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: RealtimeBidirectionalEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_realtime_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_realtime_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.realtime_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_generate_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: GenerateStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_generate_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_generate_{}", model);
        let mut card = Self::aggregated_local_card();
        card.runtime_config.runtime_data.insert(
            VLLM_INFERENCE_V1_GENERATE_CAPABILITY.to_string(),
            serde_json::Value::Bool(true),
        );
        let mut ws = WorkerSet::new(namespace.clone(), card_checksum.to_string(), card);
        ws.generate_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    pub fn add_prefill_model(
        &self,
        model: &str,
        card_checksum: &str,
    ) -> Result<(), ModelManagerError> {
        let _reservation = self.reservation_lock.lock();
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_prefill() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_prefill_{}", model);
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Prefill);
        card.needs = vec![vec![crate::worker_type::WorkerType::Decode]];
        let ws = WorkerSet::new(namespace.clone(), card_checksum.to_string(), card);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        self.publish_catalog_locked();
        Ok(())
    }

    // -- Model removal --

    /// Remove a model entirely (all its WorkerSets).
    /// Returns the removed Model, or None if not found.
    pub fn remove_model(&self, model: &str) -> Option<Arc<Model>> {
        let _reservation = self.reservation_lock.lock();
        let removed = self.models.remove(model).map(|(_, value)| value);
        if removed.is_some() {
            self.publish_catalog_locked();
        }
        removed
    }

    // Per-type remove methods for in-process models (used by Python bindings).
    // These remove the specific synthetic WorkerSet created by the corresponding add_*_model method.

    pub fn remove_chat_completions_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_chat_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_completions_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_completions_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_tensor_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_tensor_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_embeddings_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_embeddings_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_classify_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_classify_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_pooling_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_pooling_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_images_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_images_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_videos_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_videos_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_realtime_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_realtime_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn remove_generate_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let namespace = format!("__local_generate_{}", model);
        self.remove_worker_set(model, &namespace)
            .map(|_| ())
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))
    }

    // -- KV Router creation --

    /// Whether to start the LoRA load-estimator feed for a KV router's metric worker type.
    ///
    /// The feed must run for the worker mode that carries the routable request load. In dynamo's
    /// KV path that is `WORKER_TYPE_DECODE`, which the binding assigns to BOTH aggregated and
    /// disaggregated-decode endpoints (any non-prefill endpoint that tracks active blocks; see
    /// `create_kv_router_from_endpoint`). Only disaggregated PREFILL is excluded, so its transient
    /// load does not double-count the decode component's active sequences. Returns false when LoRA
    /// serving is disabled.
    fn should_start_lora_load_feed(lora_enabled: bool, worker_type: &str) -> bool {
        lora_enabled && worker_type == crate::protocols::common::timing::WORKER_TYPE_DECODE
    }

    fn hicache_cache_for(
        &self,
        endpoint: &Endpoint,
        runtime_configs: RuntimeConfigWatch,
    ) -> HicacheSharedKvCache {
        self.hicache_caches
            .entry(endpoint.id())
            .or_insert_with(|| {
                let frontend_kv_events_endpoint = std::env::var("DYN_MOONCAKE_KV_EVENTS_ENDPOINT")
                    .ok()
                    .filter(|endpoint| !endpoint.is_empty());
                let cache = HicacheSharedKvCache::new_with_cancellation_and_endpoint(
                    runtime_configs,
                    endpoint.component().drt().child_token(),
                    frontend_kv_events_endpoint,
                );
                cache.start_subscriber();
                cache
            })
            .clone()
    }

    pub fn remove_hicache_caches(&self, namespace: &str, component: &str) {
        let endpoint_ids = self
            .hicache_caches
            .iter()
            .filter(|entry| {
                entry.key().namespace == namespace && entry.key().component == component
            })
            .map(|entry| entry.key().clone())
            .collect::<Vec<_>>();

        for endpoint_id in endpoint_ids {
            if let Some((_, cache)) = self.hicache_caches.remove(&endpoint_id) {
                cache.shutdown();
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn kv_chooser_for(
        &self,
        endpoint: &Endpoint,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
    ) -> anyhow::Result<crate::kv_router::KvRoutingGraph> {
        self.kv_chooser_for_with_worker_role(
            endpoint,
            kv_cache_block_size,
            kv_router_config,
            prefill_load_estimator,
            None,
            metric_worker_type,
            model_name,
            is_eagle,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn kv_chooser_for_with_worker_role(
        &self,
        endpoint: &Endpoint,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
    ) -> anyhow::Result<crate::kv_router::KvRoutingGraph> {
        let selector = DefaultWorkerSelector::new(kv_router_config.clone(), metric_worker_type);
        self.kv_chooser_for_with_selector(
            endpoint,
            kv_cache_block_size,
            selector,
            kv_router_config,
            prefill_load_estimator,
            worker_role,
            metric_worker_type,
            model_name,
            is_eagle,
        )
        .await
    }

    /// Construct a KV chooser with a selector resolved by the router host at startup.
    #[allow(clippy::too_many_arguments)]
    pub async fn kv_chooser_for_with_selector<Sel>(
        &self,
        endpoint: &Endpoint,
        kv_cache_block_size: u32,
        selector: Sel,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
    ) -> anyhow::Result<crate::kv_router::KvRoutingGraph<Sel>>
    where
        Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
    {
        let client = endpoint.client().await?;
        let source = match worker_role {
            Some(WorkerType::Prefill) => crate::kv_router::RouterLoadSource::Prefill,
            Some(WorkerType::Aggregated) => crate::kv_router::RouterLoadSource::Aggregated,
            Some(WorkerType::Decode) => crate::kv_router::RouterLoadSource::Decode,
            Some(WorkerType::Encode) => anyhow::bail!("encode endpoints do not support KV routing"),
            None if metric_worker_type == crate::protocols::common::timing::WORKER_TYPE_PREFILL => {
                crate::kv_router::RouterLoadSource::Prefill
            }
            None => crate::kv_router::RouterLoadSource::Aggregated,
        };
        let owner = crate::kv_router::TypedRoutingGraph::start(
            client.clone(),
            source,
            crate::discovery::LoadThresholdHandle::new(Default::default()),
            &endpoint.component().drt().child_token(),
            None,
        )
        .await?;
        let router = self
            .kv_chooser_for_with_selector_and_client(
                client,
                kv_cache_block_size,
                selector,
                kv_router_config,
                prefill_load_estimator,
                worker_role,
                metric_worker_type,
                model_name,
                is_eagle,
                owner.scheduler_load_sender(),
                owner.cancellation_token(),
            )
            .await?;
        Ok(crate::kv_router::KvRoutingGraph::new(owner, router))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn kv_chooser_for_with_selector_and_client<Sel>(
        &self,
        client: Client,
        kv_cache_block_size: u32,
        selector: Sel,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        scheduler_load: crate::kv_router::SchedulerLoadSender,
        cancellation_token: CancellationToken,
    ) -> anyhow::Result<Arc<KvRouter<Sel>>>
    where
        Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
    {
        let endpoint = client.endpoint.clone();
        let lora_domain = self.lora_domain(&endpoint.id());

        // Register router via discovery mechanism.
        let drt = endpoint.component().drt();
        let instance_id = drt.discovery().instance_id();

        // Build transport for router endpoint based on request plane mode
        // Use the worker's component name so each target pool gets its own router discovery group
        let router_endpoint_id =
            router_endpoint_id(endpoint.id().namespace, endpoint.id().component);
        let transport = build_transport_type(&endpoint, &router_endpoint_id, instance_id).await?;

        let discovery_spec = DiscoverySpec::Endpoint {
            namespace: router_endpoint_id.namespace.clone(),
            component: router_endpoint_id.component.clone(),
            endpoint: router_endpoint_id.name.clone(),
            transport,
            device_type: None,
            request_plane_codec: Some(RequestPlanePayloadCodec::configured()),
        };

        let registration = drt.register_endpoint_lease(discovery_spec).await?;

        // Get of create runtime config watcher for this endpoint
        let workers_with_configs = self.get_or_create_runtime_config_watcher(&endpoint).await?;

        // A selector that does not consume cache input must not create a shared-cache client or
        // subscribe to its updates.
        let shared_cache: Option<Box<dyn dynamo_kv_router::SharedKvCache>> = if selector
            .required_worker_inputs()
            .contains(WorkerInputs::CACHE)
        {
            match kv_router_config
                .as_ref()
                .map(|c| c.shared_cache_type)
                .unwrap_or_default()
            {
                dynamo_kv_router::SharedCacheType::None => None,
                dynamo_kv_router::SharedCacheType::Hicache => {
                    let worker_component_name = &endpoint.id().component;
                    tracing::info!(
                        worker_component = worker_component_name,
                        "Using HiCache shared KV cache"
                    );
                    Some(Box::new(
                        self.hicache_cache_for(&endpoint, workers_with_configs.clone()),
                    ))
                }
            }
        } else {
            None
        };

        let effective_kv_router_config = kv_router_config.clone().unwrap_or_default();
        let kv_event_source_requirement =
            KvEventSourceRequirement::derive(worker_role, &effective_kv_router_config);
        let cache_required = selector
            .required_worker_inputs()
            .contains(WorkerInputs::CACHE)
            || effective_kv_router_config.serve_indexer
            || matches!(
                kv_event_source_requirement,
                KvEventSourceRequirement::ConditionalDisaggDecodeCache
                    | KvEventSourceRequirement::Unknown
            );
        let kv_source_membership = if cache_required
            && kv_event_source_requirement.should_subscribe(&effective_kv_router_config)
        {
            Some(
                self.get_or_create_kv_source_membership_watch(&endpoint)
                    .await?,
            )
        } else {
            None
        };

        let mut chooser = KvRouter::new_with_worker_role(
            endpoint.clone(),
            client,
            workers_with_configs,
            kv_source_membership,
            kv_cache_block_size,
            selector,
            kv_router_config,
            prefill_load_estimator,
            worker_role,
            metric_worker_type,
            model_name,
            is_eagle,
            shared_cache,
            self.lora_enabled.then(|| lora_domain.filter.clone()),
            scheduler_load,
            cancellation_token,
        )
        .await?;
        chooser.set_endpoint_registration(registration);

        // F2: feed the LoRA LoadEstimator in KV mode. Start exactly one active-sequence
        // subscription per decode endpoint. WORKER_TYPE_DECODE is the routing path for BOTH
        // aggregated and disaggregated-decode deployments: the binding maps any non-prefill,
        // active-block-tracking endpoint to WORKER_TYPE_DECODE (see create_kv_router_from_endpoint
        // in bindings/python/rust/llm/kv.rs), so aggregated KV feeds load here too. Only
        // disaggregated PREFILL is excluded — its load is transient and would double-count the
        // decode component's active sequences. Without this feed the estimator is never fed in KV
        // mode and every LoRA stays "inactive" forever. (Edge case specific to the Python KV path:
        // create_kv_router_from_endpoint infers WORKER_TYPE_PREFILL for a non-prefill endpoint when
        // router_track_active_blocks=false, so that aggregated worker would skip this feed and KV
        // routing is not load-aware — dynamic LoRA allocation then degrades to cold-start pins while
        // the filter still routes by loaded worker. Constructors that pass WORKER_TYPE_DECODE
        // directly, e.g. the watcher / C bindings, are unaffected.)
        if Self::should_start_lora_load_feed(self.lora_enabled, metric_worker_type) {
            let feed_key = endpoint.id().to_string();
            // Start a feed if none runs for this endpoint yet, or restart it if the previous
            // one exited (so a dead subscription does not permanently disable load tracking).
            //
            // Use the DashMap entry API so the check-and-insert is atomic per key: two
            // concurrent `kv_chooser_for` calls for the same component otherwise both observe
            // "no feed" and each spawn a subscription, double-counting active sequences.
            // Holding the entry lock across the spawn serializes them — the loser sees the
            // winner's live handle and skips.
            let started = match self.lora_load_feeds.entry(feed_key) {
                Entry::Occupied(mut entry) => {
                    if entry.get().is_finished() {
                        // Previous feed exited; replace it (aborting the dead handle is a no-op).
                        let handle = self
                            .lora_domain(&endpoint.id())
                            .load_estimator
                            .clone()
                            .start_event_subscription(endpoint.clone());
                        entry.insert(handle);
                        true
                    } else {
                        false
                    }
                }
                Entry::Vacant(entry) => {
                    let handle = self
                        .lora_domain(&endpoint.id())
                        .load_estimator
                        .clone()
                        .start_event_subscription(endpoint.clone());
                    entry.insert(handle);
                    true
                }
            };
            if started {
                tracing::info!(
                    namespace = %endpoint.id().namespace,
                    component = %endpoint.id().component,
                    endpoint = %endpoint.id().name,
                    "Started decode-side LoRA load feed (KV active-sequence subscription)"
                );
            }
        }

        Ok(Arc::new(chooser))
    }

    // ── LoRA allocation accessors ───────────────────────────────────────

    fn lora_domain(&self, endpoint_id: &EndpointId) -> Arc<LoraEndpointDomain> {
        let domain = self
            .lora_domains
            .entry(endpoint_id.clone())
            .or_insert_with(|| Arc::new(LoraEndpointDomain::new()))
            .clone();
        self.ensure_lora_controller(endpoint_id, &domain);
        domain
    }

    fn ensure_lora_controller(&self, endpoint_id: &EndpointId, domain: &Arc<LoraEndpointDomain>) {
        let Some(cancel_token) = self.lora_controller_cancel.lock().clone() else {
            return;
        };
        if domain.controller_started.swap(true, Ordering::AcqRel) {
            return;
        }

        let config = crate::lora::LoraAllocationConfig::from_env();
        if !config.enabled {
            return;
        }
        domain
            .load_estimator
            .set_config(crate::lora::LoadEstimatorConfig {
                rate_window: std::time::Duration::from_secs(config.effective_rate_window_secs()),
                buckets_per_second: config.buckets_per_second,
                predictor_type: config.predictor_type,
                ema_alpha: config.ema_alpha,
                ..Default::default()
            });
        let domain_cancel = cancel_token.child_token();
        *domain.controller_cancel.lock() = Some(domain_cancel.clone());
        let _handle = crate::lora::LoraController::start_for_endpoint(
            endpoint_id.clone(),
            config,
            domain.routing_table.clone(),
            domain.state_tracker.clone(),
            domain.load_estimator.clone(),
            domain_cancel,
        );
    }

    #[cfg(test)]
    pub(crate) fn lora_state_tracker_for(&self, endpoint_id: &EndpointId) -> LoraStateTracker {
        self.lora_domain(endpoint_id).state_tracker.clone()
    }

    pub fn lora_load_estimator_for(&self, endpoint_id: &EndpointId) -> Arc<LoadEstimator> {
        self.lora_domain(endpoint_id).load_estimator.clone()
    }

    pub fn lora_filter_for(&self, endpoint_id: &EndpointId) -> Option<Arc<LoraFilter>> {
        self.lora_enabled
            .then(|| self.lora_domain(endpoint_id).filter.clone())
    }

    pub fn lora_enabled(&self) -> bool {
        self.lora_enabled
    }

    /// Start the LoRA allocation controller background loop.
    pub fn start_lora_controller(
        &self,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> tokio::task::JoinHandle<()> {
        *self.lora_controller_cancel.lock() = Some(cancel_token.clone());
        for entry in self.lora_domains.iter() {
            self.ensure_lora_controller(entry.key(), entry.value());
        }
        tokio::spawn(async move {
            cancel_token.cancelled().await;
        })
    }

    fn supports_encoder_result_handoff(card: &ModelDeploymentCard) -> bool {
        card.runtime_config
            .runtime_data
            .get("encoder_result_handoff")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
    }

    fn clear_worker_set_targets(worker_set: &WorkerSet) {
        if let Some(router) = &worker_set.prefill_router {
            router.set_target(None);
        }
        if let Some(router) = &worker_set.encoder_router {
            router.set_target(None);
        }
    }

    /// Derive routing targets exclusively from committed WorkerSets. The
    /// caller holds `reservation_lock`; target clearing is synchronous and
    /// happens before the catalog pointer is published.
    fn reconcile_discovery_topology(&self, model_name: &str, namespace: &str) {
        let Some(model) = self.get_model_internal(model_name) else {
            return;
        };
        let worker_sets = model
            .worker_sets()
            .into_iter()
            .filter(|worker_set| worker_set.namespace() == namespace)
            .collect::<Vec<_>>();

        let prefill_providers = worker_sets
            .iter()
            .filter(|worker_set| worker_set.card().worker_type == Some(WorkerType::Prefill))
            .filter_map(|worker_set| worker_set.topology_endpoint().cloned())
            .collect::<Vec<_>>();
        let decode_consumers = worker_sets
            .iter()
            .filter(|worker_set| worker_set.card().worker_type == Some(WorkerType::Decode))
            .collect::<Vec<_>>();
        let prefill_target = (prefill_providers.len() == 1 && decode_consumers.len() == 1)
            .then(|| prefill_providers[0].clone());
        for worker_set in &decode_consumers {
            if let Some(router) = &worker_set.prefill_router {
                router.set_target(prefill_target.clone());
            }
        }

        let encode_providers = worker_sets
            .iter()
            .filter(|worker_set| {
                worker_set.card().worker_type == Some(WorkerType::Encode)
                    && worker_set.card().model_type.is_empty()
            })
            .filter_map(|worker_set| worker_set.topology_endpoint().cloned())
            .collect::<Vec<_>>();
        let unique_encode = (encode_providers.len() == 1).then(|| encode_providers[0].clone());
        let capable_prefill = (prefill_providers.len() == 1)
            && worker_sets.iter().any(|worker_set| {
                worker_set.card().worker_type == Some(WorkerType::Prefill)
                    && Self::supports_encoder_result_handoff(worker_set.card())
            });

        for worker_set in &worker_sets {
            let Some(router) = &worker_set.encoder_router else {
                continue;
            };
            let routing_enabled = match worker_set.card().worker_type {
                Some(WorkerType::Decode) => capable_prefill && decode_consumers.len() == 1,
                Some(WorkerType::Prefill) => false,
                Some(WorkerType::Aggregated | WorkerType::Encode) | None => {
                    Self::supports_encoder_result_handoff(worker_set.card())
                }
            };
            router.set_target(routing_enabled.then(|| unique_encode.clone()).flatten());
        }
    }

    // -- Worker monitoring --

    /// Gets or sets the load threshold config for a model's worker monitor.
    /// Checks across all WorkerSets for the model.
    pub fn load_threshold_config(
        &self,
        model: &str,
        config: Option<&LoadThresholdConfig>,
    ) -> Option<LoadThresholdConfig> {
        let model_entry = self.models.get(model)?;
        model_entry.load_threshold_config(config)
    }

    /// Lists all models with worker monitors configured.
    pub fn list_busy_thresholds(&self) -> Vec<(String, LoadThresholdConfig)> {
        let mut result = Vec::new();
        for entry in self.models.iter() {
            if let Some(config) = entry.value().load_threshold_config(None) {
                result.push((entry.key().clone(), config));
            }
        }
        result
    }

    // -- Runtime configs --

    /// Get or create a runtime config watcher for an endpoint.
    /// Spawns a background task that joins instance availability and config discovery.
    /// Returns a `watch::Receiver` with the latest `HashMap<WorkerId, ModelRuntimeConfig>`.
    pub async fn get_or_create_runtime_config_watcher(
        &self,
        endpoint: &Endpoint,
    ) -> anyhow::Result<RuntimeConfigWatch> {
        let endpoint_id = endpoint.id();

        if let Some(existing) = self.runtime_configs.get(&endpoint_id) {
            return Ok(existing.clone());
        }

        // Slow path: create the watch (spawns a background task).
        // If another caller raced us, the entry() below picks up the winner;
        // the loser's background task stops once its receivers are dropped.
        // This registry is keyed by endpoint and outlives any one WorkerSet, so
        // the watch is scoped to the process, not to a caller's own lifecycle.
        let rx = runtime_config_watch(endpoint, endpoint.drt().primary_token()).await?;
        let result = match self.runtime_configs.entry(endpoint_id) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => {
                e.insert(rx.clone());
                rx
            }
        };

        Ok(result)
    }

    /// Get or create the reusable KV-source membership watch for one exact serving endpoint.
    ///
    /// The coordinator reuses this manager's runtime-config watch, dynamically follows its
    /// effective KV-state endpoint, and joins exact KV source advertisements only to serving
    /// worker/rank membership. KV-source health never changes ordinary serving membership.
    pub async fn get_or_create_kv_source_membership_watch(
        &self,
        endpoint: &Endpoint,
    ) -> anyhow::Result<KvSourceMembershipWatch> {
        let runtime_configs = self.get_or_create_runtime_config_watcher(endpoint).await?;
        Ok(self.get_or_create_kv_source_membership_watch_with(
            endpoint.id(),
            runtime_configs,
            endpoint.drt().discovery(),
        ))
    }

    fn get_or_create_kv_source_membership_watch_with(
        &self,
        serving_endpoint: EndpointId,
        runtime_configs: RuntimeConfigWatch,
        discovery: Arc<dyn Discovery>,
    ) -> KvSourceMembershipWatch {
        if let Some(existing) = self
            .kv_source_memberships
            .get(&serving_endpoint)
            .and_then(|entry| entry.value().upgrade())
        {
            return existing.subscribe();
        }

        let candidate = KvSourceMembershipCoordinator::start(
            serving_endpoint.clone(),
            runtime_configs,
            discovery,
        );
        let coordinator = match self.kv_source_memberships.entry(serving_endpoint) {
            Entry::Occupied(mut entry) => match entry.get().upgrade() {
                Some(existing) => existing,
                None => {
                    entry.insert(Arc::downgrade(&candidate));
                    candidate
                }
            },
            Entry::Vacant(entry) => {
                entry.insert(Arc::downgrade(&candidate));
                candidate
            }
        };
        coordinator.subscribe()
    }

    /// Get disaggregated endpoint for a specific worker.
    pub fn get_disaggregated_endpoint(
        &self,
        endpoint_id: &EndpointId,
        worker_id: WorkerId,
    ) -> Option<DisaggregatedEndpoint> {
        let rx = self.runtime_configs.get(endpoint_id)?;
        let configs = rx.borrow();
        configs.get(&worker_id)?.disaggregated_endpoint.clone()
    }

    /// Get the registered `data_parallel_size` for a specific worker.
    /// Used by PD prefill routing so the chosen prefill DP rank can be
    /// encoded into `bootstrap_room` (`bootstrap_room % dp_size == dp_rank`)
    /// and recovered modulo-style on the decode side.
    pub fn get_data_parallel_size(
        &self,
        endpoint_id: &EndpointId,
        worker_id: WorkerId,
    ) -> Option<u32> {
        let rx = self.runtime_configs.get(endpoint_id)?;
        let configs = rx.borrow();
        Some(configs.get(&worker_id)?.data_parallel_size)
    }

    /// Whether any worker on this endpoint advertises a required KV-transfer topology policy.
    pub fn has_kv_transfer_required_routing_policy(&self, endpoint_id: &EndpointId) -> bool {
        let Some(rx) = self.runtime_configs.get(endpoint_id) else {
            return false;
        };
        let configs = rx.borrow();
        has_required_kv_transfer_policy(&configs)
    }

    /// Build topology routing constraints from a selected prefill worker's metadata.
    pub fn get_kv_transfer_routing_constraints(
        &self,
        endpoint_id: &EndpointId,
        worker_id: WorkerId,
    ) -> anyhow::Result<Option<RoutingConstraints>> {
        let Some(rx) = self.runtime_configs.get(endpoint_id) else {
            tracing::debug!(%endpoint_id, worker_id, "no runtime configs for topology routing");
            return Ok(None);
        };
        let configs = rx.borrow();
        let Some(config) = configs.get(&worker_id) else {
            tracing::debug!(
                %endpoint_id,
                worker_id,
                num_workers = configs.len(),
                worker_ids = ?configs.keys().collect::<Vec<_>>(),
                "selected prefill worker missing from runtime configs for topology routing"
            );
            if has_required_kv_transfer_policy(&configs) {
                anyhow::bail!(
                    "selected prefill worker {worker_id} missing from runtime configs for endpoint {endpoint_id}; \
                     cannot derive KV transfer topology constraints for required policy"
                );
            }
            return Ok(None);
        };
        let Some(domain) = config.kv_transfer_domain.as_deref() else {
            tracing::debug!(
                %endpoint_id,
                worker_id,
                topology_domains = ?config.topology_domains,
                "selected prefill worker has no kv_transfer_domain"
            );
            return Ok(None);
        };
        let Some(value) = config.topology_domains.get(domain) else {
            anyhow::bail!(
                "selected prefill worker {worker_id} configured kv_transfer_domain={domain:?}, \
                 but topology_domains does not contain that domain"
            );
        };

        let taint = topology_taint(domain, value);
        let mut constraints = RoutingConstraints::default();
        match config.kv_transfer_enforcement {
            Some(KvTransferEnforcement::Required) => {
                constraints.required_taints.insert(taint);
            }
            Some(KvTransferEnforcement::Preferred) => {
                let Some(weight) = config.kv_transfer_preferred_weight else {
                    anyhow::bail!(
                        "selected prefill worker {worker_id} configured preferred KV transfer \
                         enforcement for domain {domain:?}, but kv_transfer_preferred_weight is missing"
                    );
                };
                constraints.preferred_taints.insert(taint, weight);
            }
            None => {
                anyhow::bail!(
                    "selected prefill worker {worker_id} configured kv_transfer_domain={domain:?}, \
                     but kv_transfer_enforcement is missing"
                );
            }
        };

        Ok(Some(constraints))
    }
}

fn has_required_kv_transfer_policy(configs: &HashMap<WorkerId, ModelRuntimeConfig>) -> bool {
    configs.values().any(|config| {
        matches!(
            config.kv_transfer_enforcement,
            Some(KvTransferEnforcement::Required)
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use dynamo_kv_router::protocols::{KV_EVENT_SUBJECT, WorkerWithDpRank};
    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        discovery::{Discovery, MockDiscovery, SharedMockRegistry},
        distributed::DistributedConfig,
        pipeline::RouterMode,
        transports::event_plane::EventScope,
    };

    use crate::model_card::ModelDeploymentCard;
    use crate::{
        discovery::{KvEventSource, KvSourceStatus},
        local_model::runtime_config::ModelRuntimeConfig,
    };

    fn make_worker_set(namespace: &str, mdcsum: &str) -> WorkerSet {
        WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        )
    }

    fn insert_runtime_configs(
        mm: &ModelManager,
        endpoint_id: &EndpointId,
        configs: HashMap<WorkerId, ModelRuntimeConfig>,
    ) {
        let (_tx, rx) = tokio::sync::watch::channel(configs);
        mm.runtime_configs.insert(endpoint_id.clone(), rx);
    }

    #[tokio::test]
    async fn kv_source_membership_watch_is_shared_by_exact_serving_endpoint() {
        let manager = ModelManager::new();
        let serving_endpoint = EndpointId::from("ns.worker.generate");
        let kv_endpoint = EndpointId::from("ns.worker.kv");
        let worker = WorkerWithDpRank::new(42, 4);
        let (_configs_tx, configs_rx) = tokio::sync::watch::channel(HashMap::from([(
            42,
            ModelRuntimeConfig {
                data_parallel_start_rank: 4,
                data_parallel_size: 1,
                kv_state_endpoint: Some(kv_endpoint.clone()),
                ..Default::default()
            },
        )]));
        let discovery: Arc<dyn Discovery> =
            Arc::new(MockDiscovery::new(Some(1), SharedMockRegistry::new()));

        let mut first = manager.get_or_create_kv_source_membership_watch_with(
            serving_endpoint.clone(),
            configs_rx.clone(),
            discovery.clone(),
        );
        let mut second = manager.get_or_create_kv_source_membership_watch_with(
            serving_endpoint.clone(),
            configs_rx.clone(),
            discovery.clone(),
        );
        assert!(first.shares_coordinator_with(&second));
        let other_endpoint = EndpointId::from("ns.worker.generate-b");
        let other = manager.get_or_create_kv_source_membership_watch_with(
            other_endpoint,
            configs_rx,
            discovery.clone(),
        );
        assert!(!first.shares_coordinator_with(&other));

        let source = KvEventSource {
            kv_state_endpoint: kv_endpoint.clone(),
            worker,
            publisher_id: 100,
            recovery_target: None,
        };
        discovery
            .register(DiscoverySpec::EventSource {
                scope: EventScope::Endpoint {
                    endpoint: kv_endpoint,
                },
                topic: KV_EVENT_SUBJECT.to_string(),
                publisher_id: source.publisher_id,
                metadata: serde_json::to_value(&source).unwrap(),
            })
            .await
            .unwrap();

        for membership in [&mut first, &mut second] {
            tokio::time::timeout(std::time::Duration::from_secs(2), async {
                loop {
                    if membership.borrow().status(&worker)
                        == Some(&KvSourceStatus::ActiveLiveOnly(source.clone()))
                    {
                        break;
                    }
                    membership.changed().await.unwrap();
                }
            })
            .await
            .unwrap();
        }
    }

    fn topology_runtime_config(
        enforcement: KvTransferEnforcement,
        preferred_weight: Option<f32>,
    ) -> ModelRuntimeConfig {
        let mut config = ModelRuntimeConfig {
            kv_transfer_domain: Some("zone".to_string()),
            kv_transfer_enforcement: Some(enforcement),
            kv_transfer_preferred_weight: preferred_weight,
            ..Default::default()
        };
        config
            .topology_domains
            .insert("zone".to_string(), "us-east-1a".to_string());
        config
    }

    #[test]
    fn lora_load_feed_starts_for_aggregated_and_decode_not_prefill() {
        use crate::protocols::common::timing::{WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL};
        // Aggregated and disaggregated-decode deployments both route via WORKER_TYPE_DECODE
        // (create_kv_router_from_endpoint maps any non-prefill, active-block-tracking endpoint to
        // it), so the LoRA load feed must start for that worker type — otherwise the controller
        // would treat every adapter as inactive and never run dynamic allocation.
        assert!(
            ModelManager::should_start_lora_load_feed(true, WORKER_TYPE_DECODE),
            "decode/aggregated KV must start the LoRA load feed"
        );
        // Disaggregated prefill load is fed via the decode component, so prefill must NOT start its
        // own feed (avoids double-counting active sequences).
        assert!(
            !ModelManager::should_start_lora_load_feed(true, WORKER_TYPE_PREFILL),
            "prefill KV must not start its own LoRA load feed"
        );
        // LoRA serving disabled: never start the feed.
        assert!(
            !ModelManager::should_start_lora_load_feed(false, WORKER_TYPE_DECODE),
            "no feed when LoRA serving is disabled"
        );
    }

    #[test]
    fn lora_state_and_load_are_isolated_by_endpoint() {
        use crate::kv_router::protocols::WorkerWithDpRank;
        use crate::model_card::LoraInfo;

        let manager = ModelManager::new();
        let endpoint_a = EndpointId::from("test.worker-a.generate");
        let endpoint_b = EndpointId::from("test.worker-b.generate");
        let worker = WorkerWithDpRank::new(7, 0);
        let adapter = LoraInfo {
            name: "shared-adapter".to_string(),
            max_gpu_lora_count: Some(4),
        };

        let tracker_a = manager.lora_state_tracker_for(&endpoint_a);
        let tracker_b = manager.lora_state_tracker_for(&endpoint_b);
        tracker_a.handle_mdc_addition(worker, &adapter);

        assert!(tracker_a.is_loaded(&adapter.name, &worker));
        assert!(!tracker_b.is_loaded(&adapter.name, &worker));

        let estimator_a = manager.lora_load_estimator_for(&endpoint_a);
        let estimator_b = manager.lora_load_estimator_for(&endpoint_b);
        estimator_a.increment_load(&adapter.name);

        assert_eq!(estimator_a.get_current_load().get(&adapter.name), Some(&1));
        assert!(!estimator_b.get_current_load().contains_key(&adapter.name));
    }

    #[test]
    fn kv_transfer_constraints_build_required_and_preferred_constraints() {
        let mm = ModelManager::new();
        let endpoint_id = EndpointId::from("test.prefill.generate");

        for (worker_id, config) in [
            (
                7,
                topology_runtime_config(KvTransferEnforcement::Required, None),
            ),
            (
                8,
                topology_runtime_config(KvTransferEnforcement::Preferred, Some(0.85)),
            ),
        ] {
            insert_runtime_configs(&mm, &endpoint_id, HashMap::from([(worker_id, config)]));

            let constraints = mm
                .get_kv_transfer_routing_constraints(&endpoint_id, worker_id)
                .unwrap()
                .unwrap();

            match worker_id {
                7 => {
                    assert!(
                        constraints
                            .required_taints
                            .contains("dynamo.topology/zone=us-east-1a")
                    );
                    assert!(constraints.preferred_taints.is_empty());
                }
                8 => {
                    assert!(constraints.required_taints.is_empty());
                    assert_eq!(
                        constraints.preferred_taints["dynamo.topology/zone=us-east-1a"],
                        0.85
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn kv_transfer_required_policy_presence_ignores_preferred_policy() {
        let mm = ModelManager::new();
        let endpoint_id = EndpointId::from("test.prefill.generate");

        insert_runtime_configs(
            &mm,
            &endpoint_id,
            HashMap::from([(
                7,
                topology_runtime_config(KvTransferEnforcement::Preferred, Some(0.85)),
            )]),
        );
        assert!(!mm.has_kv_transfer_required_routing_policy(&endpoint_id));

        insert_runtime_configs(
            &mm,
            &endpoint_id,
            HashMap::from([(
                7,
                topology_runtime_config(KvTransferEnforcement::Required, None),
            )]),
        );
        assert!(mm.has_kv_transfer_required_routing_policy(&endpoint_id));
    }

    #[test]
    fn kv_transfer_constraints_missing_selected_worker_fails_closed_for_required_policy() {
        let mm = ModelManager::new();
        let endpoint_id = EndpointId::from("test.prefill.generate");
        let missing_worker_id = 99;

        insert_runtime_configs(
            &mm,
            &endpoint_id,
            HashMap::from([(
                7,
                topology_runtime_config(KvTransferEnforcement::Preferred, Some(0.85)),
            )]),
        );
        assert!(
            mm.get_kv_transfer_routing_constraints(&endpoint_id, missing_worker_id)
                .unwrap()
                .is_none()
        );

        insert_runtime_configs(
            &mm,
            &endpoint_id,
            HashMap::from([(
                7,
                topology_runtime_config(KvTransferEnforcement::Required, None),
            )]),
        );
        let err = mm
            .get_kv_transfer_routing_constraints(&endpoint_id, missing_worker_id)
            .unwrap_err()
            .to_string();
        assert!(err.contains("selected prefill worker 99 missing from runtime configs"));
        assert!(err.contains("required policy"));
    }

    // -- CRUD delegation tests --

    #[test]
    fn test_add_and_get_worker_set() {
        let mm = ModelManager::new();
        let ws = make_worker_set("ns1", "abc");
        mm.add_worker_set("llama", "ns1", ws);

        let model = mm.get_model("llama");
        assert!(model.is_some());
        let model = model.unwrap();
        assert!(model.has_worker_set("ns1"));
        assert_eq!(model.worker_set_count(), 1);
    }

    #[test]
    fn test_add_worker_set_creates_model() {
        let mm = ModelManager::new();
        assert!(mm.get_model("llama").is_none());

        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        assert!(mm.get_model("llama").is_some());
    }

    #[test]
    fn test_remove_worker_set_removes_empty_model() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        assert!(mm.get_model("llama").is_some());

        let removed = mm.remove_worker_set("llama", "ns1");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().namespace(), "ns1");

        // Model should be auto-removed since it's now empty
        assert!(mm.get_model("llama").is_none());
    }

    #[test]
    fn test_remove_worker_set_keeps_model_with_remaining() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        mm.add_worker_set("llama", "ns2", make_worker_set("ns2", "abc"));

        mm.remove_worker_set("llama", "ns1");

        // Model should still exist with ns2
        let model = mm.get_model("llama").unwrap();
        assert!(!model.has_worker_set("ns1"));
        assert!(model.has_worker_set("ns2"));
        assert_eq!(model.worker_set_count(), 1);
    }

    #[test]
    fn test_remove_worker_set_nonexistent_model() {
        let mm = ModelManager::new();
        assert!(mm.remove_worker_set("llama", "ns1").is_none());
    }

    #[test]
    fn test_remove_worker_set_nonexistent_namespace() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        assert!(mm.remove_worker_set("llama", "ns2").is_none());

        // Model should still exist (ns1 still there)
        assert!(mm.get_model("llama").is_some());
    }

    #[test]
    fn remove_hicache_caches_cancels_only_the_removed_component() {
        let manager = ModelManager::new();
        let (_tx, runtime_configs) =
            tokio::sync::watch::channel(HashMap::<WorkerId, ModelRuntimeConfig>::new());
        let cancelled = tokio_util::sync::CancellationToken::new();
        let retained = tokio_util::sync::CancellationToken::new();
        manager.hicache_caches.insert(
            EndpointId::from("ns.worker.generate"),
            HicacheSharedKvCache::new_with_cancellation(runtime_configs.clone(), cancelled.clone()),
        );
        manager.hicache_caches.insert(
            EndpointId::from("ns.other.generate"),
            HicacheSharedKvCache::new_with_cancellation(runtime_configs, retained.clone()),
        );

        manager.remove_hicache_caches("ns", "worker");

        assert!(cancelled.is_cancelled());
        assert!(!retained.is_cancelled());
        assert!(
            !manager
                .hicache_caches
                .contains_key(&EndpointId::from("ns.worker.generate"))
        );
        assert!(
            manager
                .hicache_caches
                .contains_key(&EndpointId::from("ns.other.generate"))
        );
    }

    #[test]
    fn test_alias_resolution_maps_to_primary() {
        let mm = ModelManager::new();

        assert!(mm.register_alias("llama-alias", "llama"));

        assert_eq!(mm.resolve_canonical_name("llama-alias"), "llama");
        assert_eq!(mm.resolve_canonical_name("llama"), "llama");
    }

    #[test]
    fn test_register_alias_rejects_primary_collision() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama-alias", "ns1", make_worker_set("ns1", "abc"));

        assert!(!mm.register_alias("llama-alias", "llama"));

        assert_eq!(mm.resolve_canonical_name("llama-alias"), "llama-alias");
    }

    #[test]
    fn test_primary_registration_rejected_when_name_reserved_as_alias() {
        let mm = ModelManager::new();

        // Model "a" reserves "shared" as an alias and attaches its worker set.
        assert!(mm.register_alias("shared", "a"));
        assert!(mm.add_worker_set_arc("shared", "ns1", Arc::new(make_worker_set("ns1", "abc"))));
        assert_eq!(mm.resolve_canonical_name("shared"), "a");

        // A later deployment cannot claim "shared" as its own primary — the name
        // stays reserved for "a" (first-come, symmetric with register_alias).
        assert!(!mm.add_worker_set("shared", "ns2", make_worker_set("ns2", "def")));
        assert_eq!(mm.resolve_canonical_name("shared"), "a");

        // "a"'s alias mirror is untouched and no foreign worker set was added.
        let model = mm.get_model("shared").expect("alias model present");
        assert!(model.get_worker_set("ns1").is_some());
        assert!(model.get_worker_set("ns2").is_none());
    }

    #[test]
    fn test_alias_belongs_to_identifies_owner() {
        let mm = ModelManager::new();
        assert!(mm.register_alias("chat", "llama"));

        assert!(mm.alias_belongs_to("chat", "llama"));
        // Not owned by a different primary, and a non-alias name is owned by nobody.
        assert!(!mm.alias_belongs_to("chat", "other"));
        assert!(!mm.alias_belongs_to("not-an-alias", "llama"));

        // A live primary named "chat" (a different deployment) is not an alias of
        // "llama" — so a "llama" teardown must not treat "chat" as its own.
        let mm2 = ModelManager::new();
        mm2.add_worker_set("chat", "ns1", make_worker_set("ns1", "abc"));
        assert!(!mm2.alias_belongs_to("chat", "llama"));
    }

    #[test]
    fn test_primary_registration_succeeds_for_unreserved_name() {
        let mm = ModelManager::new();
        assert!(mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc")));
        assert!(mm.get_model("llama").is_some());
        // A second worker set for the same primary is fine (replicas share a name).
        assert!(mm.add_worker_set("llama", "ns2", make_worker_set("ns2", "abc")));
    }

    #[test]
    fn test_unregister_alias_if_empty_keeps_mapping_with_remaining_worker_sets() {
        let mm = ModelManager::new();
        assert!(mm.register_alias("llama-alias", "llama"));

        assert!(mm.add_worker_set_arc(
            "llama-alias",
            "ns1",
            Arc::new(make_worker_set("ns1", "abc")),
        ));
        assert!(mm.add_worker_set_arc(
            "llama-alias",
            "ns2",
            Arc::new(make_worker_set("ns2", "abc")),
        ));

        mm.remove_worker_set("llama-alias", "ns1");
        mm.unregister_alias_if_empty("llama-alias", "llama");
        assert_eq!(mm.resolve_canonical_name("llama-alias"), "llama");

        mm.remove_worker_set("llama-alias", "ns2");
        mm.unregister_alias_if_empty("llama-alias", "llama");
        assert_eq!(mm.resolve_canonical_name("llama-alias"), "llama-alias");
    }

    #[test]
    fn test_remove_model_if_empty_noop_when_not_empty() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));

        mm.remove_model_if_empty("llama");
        assert!(mm.get_model("llama").is_some()); // Still has ns1
    }

    #[test]
    fn test_remove_model_if_empty_noop_when_missing() {
        let mm = ModelManager::new();
        mm.remove_model_if_empty("nonexistent"); // Should not panic
    }

    #[test]
    fn test_remove_model() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        mm.add_worker_set("llama", "ns2", make_worker_set("ns2", "abc"));

        let removed = mm.remove_model("llama");
        assert!(removed.is_some());
        assert!(mm.get_model("llama").is_none());
    }

    #[test]
    fn test_get_or_create_model_idempotent() {
        let mm = ModelManager::new();
        let m1 = mm.get_or_create_model("llama");
        let m2 = mm.get_or_create_model("llama");
        // Both should point to the same Model (same Arc)
        assert!(Arc::ptr_eq(&m1, &m2));
    }

    #[test]
    fn public_get_model_retains_live_mutation_semantics() {
        let manager = ModelManager::new();
        manager.add_worker_set("llama", "first", make_worker_set("first", "same"));
        let model = manager.get_model("llama").unwrap();

        manager.add_worker_set("llama", "second", make_worker_set("second", "same"));

        assert!(model.has_worker_set("second"));
        assert!(Arc::ptr_eq(&model, &manager.get_model("llama").unwrap()));
    }

    // -- Model listing and filtering tests --

    #[test]
    fn test_has_decode_model() {
        let mm = ModelManager::new();

        // No model → false
        assert!(!mm.has_decode_model("llama"));

        // Prefill-only set (no engines) → false
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        assert!(!mm.has_decode_model("llama"));
    }

    #[test]
    fn test_has_prefill_model() {
        let mm = ModelManager::new();

        // Prefill set = no engines
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        assert!(mm.has_prefill_model("llama"));
    }

    #[test]
    fn test_has_model_any() {
        let mm = ModelManager::new();
        assert!(!mm.has_model_any("llama"));

        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        assert!(mm.has_model_any("llama")); // has prefill
    }

    #[test]
    fn test_metric_model_for_resolves_to_sentinel_for_unknown() {
        let mm = ModelManager::new();
        mm.add_worker_set(
            "Llama-3.1-8B-Instruct",
            "ns1",
            make_worker_set("ns1", "abc"),
        );

        // Registered models preserve their original casing.
        assert_eq!(
            mm.metric_model_for("Llama-3.1-8B-Instruct"),
            "Llama-3.1-8B-Instruct"
        );

        // Case mismatches and unregistered strings collapse to the sentinel so
        // arbitrary client-supplied values cannot create unbounded Prometheus
        // series.
        assert_eq!(
            mm.metric_model_for("llama-3.1-8b-instruct"),
            UNKNOWN_METRIC_MODEL
        );
        assert_eq!(
            mm.metric_model_for("nonexistent-model-1"),
            UNKNOWN_METRIC_MODEL
        );
        assert_eq!(mm.metric_model_for(""), UNKNOWN_METRIC_MODEL);
    }

    #[test]
    fn test_model_display_names_includes_prefill() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));

        let names = mm.model_display_names();
        assert!(names.contains("llama"));
    }

    #[test]
    fn test_model_display_names_empty() {
        let mm = ModelManager::new();
        assert!(mm.model_display_names().is_empty());
    }

    #[test]
    fn test_add_get_remove_realtime_model_round_trip() {
        let mm = ModelManager::new();
        let engine = std::sync::Arc::new(crate::engines::EchoBidirectionalEngine);

        mm.add_realtime_model("rt-echo", "0", engine.clone())
            .unwrap();
        assert!(mm.list_realtime_models().contains(&"rt-echo".to_string()));
        assert!(mm.get_realtime_engine("rt-echo").is_ok());

        mm.remove_realtime_model("rt-echo").unwrap();
        assert!(!mm.list_realtime_models().contains(&"rt-echo".to_string()));
        assert!(matches!(
            mm.get_realtime_engine("rt-echo"),
            Err(ModelManagerError::ModelNotFound(_))
        ));
    }

    #[test]
    fn test_add_realtime_model_duplicate() {
        let mm = ModelManager::new();
        let engine = std::sync::Arc::new(crate::engines::EchoBidirectionalEngine);
        mm.add_realtime_model("rt-echo", "0", engine.clone())
            .unwrap();
        assert!(matches!(
            mm.add_realtime_model("rt-echo", "0", engine),
            Err(ModelManagerError::ModelAlreadyExists(_))
        ));
    }

    #[test]
    fn test_get_realtime_engine_missing() {
        let mm = ModelManager::new();
        assert!(matches!(
            mm.get_realtime_engine("does-not-exist"),
            Err(ModelManagerError::ModelNotFound(_))
        ));
    }

    #[test]
    fn test_list_prefill_models() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        mm.add_worker_set("gpt", "ns1", make_worker_set("ns1", "def"));

        let prefill = mm.list_prefill_models();
        assert_eq!(prefill.len(), 2);
        assert!(prefill.contains(&"llama".to_string()));
        assert!(prefill.contains(&"gpt".to_string()));
    }

    // -- Model card tests --

    #[test]
    fn test_save_and_remove_model_card() {
        let mm = ModelManager::new();
        let card = ModelDeploymentCard::default();
        mm.save_model_card("instance/key/1", card.clone()).unwrap();

        let cards = mm.get_model_cards();
        assert_eq!(cards.len(), 1);

        let removed = mm.remove_model_card("instance/key/1");
        assert!(removed.is_some());
        assert!(mm.get_model_cards().is_empty());
    }

    #[test]
    fn test_remove_model_card_nonexistent() {
        let mm = ModelManager::new();
        assert!(mm.remove_model_card("nonexistent").is_none());
    }

    #[test]
    fn discovery_group_commit_is_all_or_nothing_across_aliases_and_cards() {
        let manager = ModelManager::new();
        manager.add_worker_set("taken", "existing", make_worker_set("existing", "old"));

        let mut blocked_card = ModelDeploymentCard::with_name_only("candidate");
        blocked_card.aliases = vec!["taken".to_string()];
        let blocked_worker_set = WorkerSet::new(
            "deployment".to_string(),
            blocked_card.mdcsum().to_string(),
            blocked_card.clone(),
        );
        let error = manager
            .commit_discovery_group(
                "blocked-group",
                "candidate-workers",
                blocked_worker_set,
                vec![("instance-1".to_string(), blocked_card)],
                Vec::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("collides"));
        assert!(manager.get_model("candidate").is_none());
        assert!(manager.get_model_cards().is_empty());
        assert!(manager.get_model("taken").is_some());

        let mut card = ModelDeploymentCard::with_name_only("committed");
        card.aliases = vec!["alias".to_string()];
        let worker_set = WorkerSet::new(
            "deployment".to_string(),
            card.mdcsum().to_string(),
            card.clone(),
        );
        manager
            .commit_discovery_group(
                "ready-group",
                "committed-workers",
                worker_set,
                vec![("instance-2".to_string(), card)],
                Vec::new(),
            )
            .unwrap();

        assert_eq!(manager.get_model_cards().len(), 1);
        assert!(manager.get_model("committed").is_some());
        assert!(manager.get_model("alias").is_some());
        assert_eq!(manager.resolve_canonical_name("alias"), "committed");

        manager.remove_discovery_group("ready-group").unwrap();
        assert!(manager.get_model_cards().is_empty());
        assert!(manager.get_model("committed").is_none());
        assert!(manager.get_model("alias").is_none());
        assert_eq!(manager.resolve_canonical_name("alias"), "alias");
    }

    #[test]
    fn discovery_group_derives_adapter_model_and_lora_projection() {
        let manager = ModelManager::new();
        let mut base = ModelDeploymentCard::with_name_only("base");
        base.runtime_config.max_gpu_lora_count = Some(6);
        let mut adapter = ModelDeploymentCard::with_name_only("adapter");
        adapter.lora = Some(crate::model_card::LoraInfo {
            name: "adapter".to_string(),
            max_gpu_lora_count: None,
        });
        let worker_set = WorkerSet::new(
            "deployment".to_string(),
            base.mdcsum().to_string(),
            base.clone(),
        );
        let base_mcid = ModelCardInstanceId {
            namespace: "namespace".to_string(),
            component: "worker".to_string(),
            endpoint: "generate".to_string(),
            instance_id: 17,
            model_suffix: None,
        };
        let adapter_mcid = ModelCardInstanceId {
            model_suffix: Some("adapter".to_string()),
            ..base_mcid.clone()
        };

        manager
            .commit_discovery_group(
                "adapter-group",
                "base-workers",
                worker_set,
                vec![(base_mcid.to_path(), base)],
                vec![(adapter_mcid.to_path(), adapter)],
            )
            .unwrap();

        assert_eq!(manager.get_model_cards().len(), 2);
        let base_worker_set = manager
            .get_committed_model("base")
            .and_then(|model| model.get_worker_set("base-workers"))
            .unwrap();
        let adapter_worker_set = manager
            .get_committed_model("adapter")
            .and_then(|model| model.get_worker_set("base-workers"))
            .unwrap();
        assert!(!Arc::ptr_eq(&base_worker_set, &adapter_worker_set));
        assert_eq!(adapter_worker_set.card().name(), "adapter");
        assert_eq!(
            adapter_worker_set
                .card()
                .lora
                .as_ref()
                .map(|lora| lora.name.as_str()),
            Some("adapter")
        );

        let endpoint_id = EndpointId::from("namespace.worker.generate");
        let tracker = manager.lora_state_tracker_for(&endpoint_id);
        let worker = WorkerWithDpRank::new(17, 0);
        assert!(tracker.is_loaded("adapter", &worker));
        assert_eq!(tracker.total_lora_slots(), 6);

        manager.remove_discovery_group("adapter-group").unwrap();
        assert!(manager.get_model_cards().is_empty());
        assert!(manager.get_committed_model("adapter").is_none());
        assert!(tracker.is_empty());
    }

    fn topology_card(role: WorkerType) -> ModelDeploymentCard {
        let mut card = ModelDeploymentCard::with_name_only("topology-model");
        card.worker_type = Some(role);
        card.model_type = match role {
            WorkerType::Decode => crate::model_type::ModelType::Chat,
            _ => crate::model_type::ModelType::empty(),
        };
        card.needs = match role {
            WorkerType::Prefill => vec![vec![WorkerType::Decode, WorkerType::Encode]],
            WorkerType::Decode => vec![vec![WorkerType::Prefill]],
            WorkerType::Encode => vec![vec![WorkerType::Prefill]],
            WorkerType::Aggregated => Vec::new(),
        };
        if role == WorkerType::Prefill {
            card.runtime_config
                .set_engine_specific("encoder_result_handoff", true)
                .unwrap();
        }
        card
    }

    fn topology_worker_set(
        manager: Arc<ModelManager>,
        role: WorkerType,
        endpoint: Endpoint,
    ) -> (
        WorkerSet,
        Option<Arc<crate::kv_router::PrefillRouter>>,
        Option<Arc<crate::kv_router::EncoderRouter>>,
    ) {
        let card = topology_card(role);
        let mut worker_set =
            WorkerSet::new(endpoint.id().namespace, card.mdcsum().to_string(), card);
        worker_set.set_topology_endpoint(endpoint);
        if role != WorkerType::Decode {
            return (worker_set, None, None);
        }

        let (_activation_tx, activation_rx) = tokio::sync::oneshot::channel();
        let prefill = crate::kv_router::PrefillRouter::new(
            activation_rx,
            manager,
            RouterMode::RoundRobin,
            1,
            None,
            None,
            None,
            None,
            "topology-model".to_string(),
            worker_set.namespace().to_string(),
            false,
            crate::discovery::LoadThresholdHandle::new(Default::default()),
            CancellationToken::new(),
        );
        let encoder = crate::kv_router::EncoderRouter::new(
            "topology-model".to_string(),
            worker_set.namespace().to_string(),
        );
        worker_set.prefill_router = Some(prefill.clone());
        worker_set.encoder_router = Some(encoder.clone());
        (worker_set, Some(prefill), Some(encoder))
    }

    #[tokio::test]
    async fn committed_topology_converges_for_every_epd_startup_order() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let orders = [
            [WorkerType::Encode, WorkerType::Prefill, WorkerType::Decode],
            [WorkerType::Encode, WorkerType::Decode, WorkerType::Prefill],
            [WorkerType::Prefill, WorkerType::Encode, WorkerType::Decode],
            [WorkerType::Prefill, WorkerType::Decode, WorkerType::Encode],
            [WorkerType::Decode, WorkerType::Encode, WorkerType::Prefill],
            [WorkerType::Decode, WorkerType::Prefill, WorkerType::Encode],
        ];

        for (index, order) in orders.into_iter().enumerate() {
            let manager = Arc::new(ModelManager::new());
            let namespace = format!("epd-order-{index}");
            let component = distributed
                .namespace(namespace.clone())
                .unwrap()
                .component("workers".to_string())
                .unwrap();
            let prefill_endpoint = component.endpoint("prefill".to_string());
            let encode_endpoint = component.endpoint("encode".to_string());
            let decode_endpoint = component.endpoint("decode".to_string());
            let mut prefill_router = None;
            let mut encoder_router = None;

            for (step, role) in order.into_iter().enumerate() {
                let endpoint = match role {
                    WorkerType::Prefill => prefill_endpoint.clone(),
                    WorkerType::Encode => encode_endpoint.clone(),
                    WorkerType::Decode => decode_endpoint.clone(),
                    WorkerType::Aggregated => unreachable!(),
                };
                let (worker_set, prefill, encoder) =
                    topology_worker_set(manager.clone(), role, endpoint);
                prefill_router = prefill.or(prefill_router);
                encoder_router = encoder.or(encoder_router);
                let card = worker_set.card().clone();
                manager
                    .commit_discovery_group(
                        &format!("{namespace}-{role:?}"),
                        &format!("{role:?}"),
                        worker_set,
                        vec![(format!("{namespace}-{role:?}"), card)],
                        Vec::new(),
                    )
                    .unwrap();
                if step < 2 {
                    assert!(
                        !manager
                            .get_committed_model("topology-model")
                            .unwrap()
                            .namespace_readiness()
                            .ready
                    );
                }
            }

            assert!(
                manager
                    .get_committed_model("topology-model")
                    .unwrap()
                    .namespace_readiness()
                    .ready
            );
            let prefill_router = prefill_router.unwrap();
            let encoder_router = encoder_router.unwrap();
            assert_eq!(
                prefill_router.target_endpoint_id(),
                Some(prefill_endpoint.id())
            );
            assert_eq!(
                encoder_router.target_endpoint_id(),
                Some(encode_endpoint.id())
            );

            if index < 2 {
                let duplicate_endpoint = component.endpoint("prefill-duplicate".to_string());
                let (duplicate, _, _) = topology_worker_set(
                    manager.clone(),
                    WorkerType::Prefill,
                    duplicate_endpoint.clone(),
                );
                let duplicate_card = duplicate.card().clone();
                manager
                    .commit_discovery_group(
                        &format!("{namespace}-prefill-duplicate"),
                        "PrefillDuplicate",
                        duplicate,
                        vec![(format!("{namespace}-prefill-duplicate"), duplicate_card)],
                        Vec::new(),
                    )
                    .unwrap();
                assert!(
                    !manager
                        .get_committed_model("topology-model")
                        .unwrap()
                        .namespace_readiness()
                        .ready
                );
                assert_eq!(prefill_router.target_endpoint_id(), None);

                let removed_group = if index == 0 {
                    format!("{namespace}-prefill-duplicate")
                } else {
                    format!("{namespace}-Prefill")
                };
                manager.remove_discovery_group(&removed_group).unwrap();
                let expected = if index == 0 {
                    prefill_endpoint.id()
                } else {
                    duplicate_endpoint.id()
                };
                assert!(
                    manager
                        .get_committed_model("topology-model")
                        .unwrap()
                        .namespace_readiness()
                        .ready
                );
                assert_eq!(prefill_router.target_endpoint_id(), Some(expected));
            }
        }
        runtime.shutdown();
    }

    // -- is_model_ready_to_serve / has_any_ready_model tests --
    //
    // Regression coverage for the KServe gRPC race where `model_ready` returned
    // true as soon as a ModelDeploymentCard was saved -- before the WorkerSet
    // with engines was attached. These checks must stay false until at least
    // one WorkerSet carries an actual serving engine.

    fn make_chat_engine()
    -> crate::types::openai::chat_completions::OpenAIChatCompletionsStreamingEngine {
        Arc::new(crate::engines::StreamingEngineAdapter::new(
            crate::engines::make_echo_engine(),
        ))
    }

    #[test]
    fn test_is_model_ready_to_serve_false_for_unknown_model() {
        let mm = ModelManager::new();
        assert!(!mm.is_model_ready_to_serve("llama"));
        assert!(!mm.has_any_ready_model());
    }

    #[test]
    fn test_is_model_ready_to_serve_false_for_card_only() {
        // Reproduces the KServe race: a ModelDeploymentCard is saved before the
        // WorkerSet is registered. `is_model_ready_to_serve` must still be false.
        let mm = ModelManager::new();
        let mut card = ModelDeploymentCard::default();
        card.display_name = "llama".to_string();
        mm.save_model_card("instance-1", card).unwrap();

        assert!(!mm.get_model_cards().is_empty(), "card was saved");
        assert!(
            !mm.is_model_ready_to_serve("llama"),
            "card-only registration must not report ready"
        );
        assert!(
            !mm.has_any_ready_model(),
            "card-only registration must not flip server_ready"
        );
    }

    #[test]
    fn test_is_model_ready_to_serve_false_for_prefill_only_worker_set() {
        // Worker set exists but has no engines attached (the lifecycle state
        // between save_model_card and engine wire-up).
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));

        assert!(
            !mm.is_model_ready_to_serve("llama"),
            "WorkerSet without engines must not report ready"
        );
        assert!(!mm.has_any_ready_model());
    }

    #[test]
    fn test_is_model_ready_to_serve_true_after_chat_engine_added() {
        let mm = ModelManager::new();
        mm.add_chat_completions_model("llama", "abc", make_chat_engine())
            .unwrap();

        assert!(mm.is_model_ready_to_serve("llama"));
        assert!(mm.has_any_ready_model());
    }

    #[test]
    fn test_has_any_ready_model_with_mixed_models() {
        // One model is fully wired, another is only card-registered. The
        // server-wide check must report ready as soon as any one model is.
        let mm = ModelManager::new();
        let mut card = ModelDeploymentCard::default();
        card.display_name = "pending-llama".to_string();
        mm.save_model_card("instance-pending", card).unwrap();

        assert!(!mm.has_any_ready_model());

        mm.add_chat_completions_model("ready-llama", "abc", make_chat_engine())
            .unwrap();

        assert!(mm.has_any_ready_model());
        assert!(mm.is_model_ready_to_serve("ready-llama"));
        assert!(!mm.is_model_ready_to_serve("pending-llama"));
    }

    /// A decode-only WorkerSet that needs a prefill peer (absent here), with a
    /// live worker and a chat engine attached: displayable, but its namespace
    /// is not serving-ready.
    fn incomplete_decode_chat_ws(namespace: &str, mdcsum: &str) -> WorkerSet {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Decode);
        card.needs = vec![vec![crate::worker_type::WorkerType::Prefill]];
        // Watch receiver keeps its last value after the sender drops, so
        // worker_count stays 1 without holding the sender.
        let (_tx, rx) = tokio::sync::watch::channel(vec![1u64]);
        let mut ws = WorkerSet::new(namespace.to_string(), mdcsum.to_string(), card);
        ws.set_instance_watcher(rx);
        ws.chat_engine = Some(make_chat_engine());
        ws
    }

    /// Verifies the readiness gate the review (PR #10503) flagged for the
    /// listing, default-model, and error-shape paths. A registered-but-incomplete
    /// deployment (decode-only, no prefill peer) is displayable but must be:
    ///   - excluded from `serving_ready_display_names` (OpenAI/Anthropic listing
    ///     and the audio default-model fallback),
    ///   - reported not-ready by `is_model_ready_to_serve` (KServe), and
    ///   - surfaced as `ModelUnavailable` (503) by the engine getter, not
    ///     `ModelNotFound` (404).
    #[test]
    fn serving_ready_excludes_incomplete_namespace() {
        let mm = ModelManager::new();

        // Complete, serving-ready model (aggregated, live).
        mm.add_chat_completions_model("ready", "mdc-r", make_chat_engine())
            .unwrap();

        // Incomplete model: decode-only, needs a prefill peer that never joins.
        mm.add_worker_set(
            "broken",
            "decode-ns",
            incomplete_decode_chat_ws("decode-ns", "mdc-b"),
        );

        // The incomplete model is still *displayable* (it has a live engine)...
        let displayable = mm.model_display_names();
        assert!(displayable.contains("ready"));
        assert!(
            displayable.contains("broken"),
            "incomplete model is displayable (has a live engine)"
        );

        // ...but only the complete model is *serving-ready* — the gate the
        // listing endpoints and the audio default-model fallback now apply.
        let serving = mm.serving_ready_display_names();
        assert!(serving.contains("ready"));
        assert!(
            !serving.contains("broken"),
            "incomplete model must be excluded from serving_ready_display_names"
        );

        // Point 3: the audio-speech implicit default-model fallback resolves to
        // `serving_ready_display_names().into_iter().next()`. With an incomplete
        // model present, that set excludes it, so the default can only ever
        // resolve to the complete/ready model — never the incomplete one.
        let audio_default = mm.serving_ready_display_names().into_iter().next();
        assert_eq!(
            audio_default.as_deref(),
            Some("ready"),
            "audio default-model fallback must pick the ready model, not the incomplete one"
        );

        // KServe readiness agrees.
        assert!(mm.is_model_ready_to_serve("ready"));
        assert!(!mm.is_model_ready_to_serve("broken"));

        // The engine getter yields ModelUnavailable (mapped to 503 by both the
        // OpenAI and the Anthropic handlers), not ModelNotFound (404), because
        // the engine exists but the namespace is incomplete.
        assert!(
            matches!(
                mm.get_chat_completions_engine("broken"),
                Err(ModelManagerError::ModelUnavailable(_))
            ),
            "incomplete-but-engine-present model must be ModelUnavailable (503), not 404"
        );
    }
}
