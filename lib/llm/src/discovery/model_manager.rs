// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, Ordering},
    },
};

use dashmap::{DashMap, mapref::entry::Entry};
use dynamo_kv_router::{
    PrefillLoadEstimator, RuntimePluginConfig, RuntimePluginRouterRole,
    config::KvRouterConfig,
    protocols::{KvTransferEnforcement, RoutingConstraints, WorkerId},
    selector::{ValidatedWorkerSelector, WorkerSelector},
};
use tokio::sync::oneshot;

use super::worker_monitor::LoadThresholdConfig;
use super::{
    KvSourceMembershipWatch, Model, RuntimeConfigWatch, WorkerSet,
    kv_source_watch::KvSourceMembershipCoordinator, runtime_config_watch,
};

use dynamo_runtime::{
    component::{Endpoint, build_transport_type},
    discovery::{Discovery, DiscoverySpec},
    prelude::DistributedRuntimeProvider,
    protocols::EndpointId,
};

use crate::{
    kv_router::{
        KvRouter, router_endpoint_id, scheduler::DefaultWorkerSelector,
        shared_cache::HicacheSharedKvCache,
    },
    local_model::runtime_config::{DisaggregatedEndpoint, ModelRuntimeConfig, topology_taint},
    lora::{LoraFilter, LoraRoutingTable, LoraStateTracker, load_estimator::LoadEstimator},
    model_card::ModelDeploymentCard,
    types::{
        RealtimeBidirectionalEngine,
        generic::tensor::TensorStreamingEngine,
        openai::{
            audios::OpenAIAudiosStreamingEngine,
            chat_completions::OpenAIChatCompletionsStreamingEngine,
            completions::OpenAICompletionsStreamingEngine,
            embeddings::OpenAIEmbeddingsStreamingEngine, generate::GenerateStreamingEngine,
            images::OpenAIImagesStreamingEngine, videos::OpenAIVideosStreamingEngine,
        },
    },
};

/// State for prefill router activation rendezvous.
///
/// Once a prefill endpoint has been observed for a (model, namespace) pair,
/// `PrefillReady` is left in the activator map indefinitely (until prefill
/// workers all disappear). This survives `register_prefill_router` consuming
/// the entry — that consumer hands out a fresh `oneshot::Receiver` synthesized
/// from the cached endpoint, then re-inserts `PrefillReady` so future decode
/// WorkerSet rebuilds (e.g., decode pod restarts) can find it and activate
/// immediately without waiting for prefill workers to re-register.
enum PrefillActivationState {
    /// Decode model registered, waiting for prefill endpoint
    DecodeWaiting(oneshot::Sender<Endpoint>),
    /// Prefill endpoint observed and cached for this (model, namespace).
    /// Anyone calling `register_prefill_router` synthesizes a fresh
    /// `oneshot::Receiver` from this and re-inserts the cached endpoint.
    ///
    /// Boxed to keep the enum variant sizes balanced (`Endpoint` is much
    /// larger than `oneshot::Sender`). Satisfies `clippy::large_enum_variant`.
    PrefillReady(Box<Endpoint>),
}

/// State for the optional Encode endpoint / token-pipeline rendezvous.
#[derive(Default)]
struct EncoderActivationState {
    consumer: Option<oneshot::Sender<Endpoint>>,
    endpoint: Option<Box<Endpoint>>,
    routing_enabled: bool,
}

struct LoraEndpointDomain {
    routing_table: LoraRoutingTable,
    state_tracker: LoraStateTracker,
    load_estimator: Arc<LoadEstimator>,
    filter: Arc<LoraFilter>,
    controller_started: AtomicBool,
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
        }
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

/// Central manager for model engines, routing, and configuration.
///
/// Models are stored hierarchically: ModelManager → Model → WorkerSet.
/// Each WorkerSet owns a complete pipeline built from its specific configuration.
///
/// Note: Don't implement Clone for this, put it in an Arc instead.
pub struct ModelManager {
    /// Model name → Model (which contains WorkerSets with engines)
    models: DashMap<String, Arc<Model>>,

    /// Per-instance model cards, keyed by instance path. Used for cleanup on worker removal.
    cards: DashMap<String, ModelDeploymentCard>,

    /// Prefill router activation rendezvous, keyed by "model_name:namespace".
    prefill_router_activators: DashMap<String, PrefillActivationState>,

    /// Encode router activation rendezvous, keyed by "model_name:namespace".
    encoder_router_activators: DashMap<String, EncoderActivationState>,

    /// Per-endpoint runtime config watchers. Keyed by EndpointId (includes namespace).
    runtime_configs: DashMap<EndpointId, RuntimeConfigWatch>,

    /// Shared KV-source membership coordinators, scoped by exact serving endpoint.
    /// Weak ownership lets the discovery loop stop when its last consumer goes away.
    kv_source_memberships: DashMap<EndpointId, Weak<KvSourceMembershipCoordinator>>,

    /// Exact endpoint → independent LoRA allocation and load domain.
    lora_domains: DashMap<EndpointId, Arc<LoraEndpointDomain>>,
    lora_enabled: bool,
    /// Per-decode-endpoint LoRA load-feed subscription handles, so we start exactly one feed
    /// per endpoint and can restart it if the previous one exited (avoids double counting on
    /// rebuilds while keeping the feed durable).
    lora_load_feeds: DashMap<String, tokio::task::JoinHandle<()>>,
    lora_controller_cancel: parking_lot::Mutex<Option<tokio_util::sync::CancellationToken>>,
    lora_controller_handles: parking_lot::Mutex<Vec<tokio::task::JoinHandle<()>>>,

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
            cards: DashMap::new(),
            prefill_router_activators: DashMap::new(),
            encoder_router_activators: DashMap::new(),
            runtime_configs: DashMap::new(),
            kv_source_memberships: DashMap::new(),
            lora_domains: DashMap::new(),
            lora_enabled: crate::lora::lora_serving_enabled(),
            lora_load_feeds: DashMap::new(),
            lora_controller_cancel: parking_lot::Mutex::new(None),
            lora_controller_handles: parking_lot::Mutex::new(Vec::new()),
            alias_to_primary: DashMap::new(),
            reservation_lock: parking_lot::Mutex::new(()),
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

    /// Remove a Model if it has no remaining WorkerSets.
    /// Uses atomic remove_if to avoid TOCTOU race between checking is_empty and removing.
    pub fn remove_model_if_empty(&self, model_name: &str) {
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
    /// Holds [`Self::reservation_lock`] across the reserved-name check and the
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
        let topology_namespace = (matches!(
            worker_set.card().worker_type,
            Some(crate::worker_type::WorkerType::Prefill | crate::worker_type::WorkerType::Decode)
        ))
        .then(|| worker_set.namespace().to_string());
        model.add_worker_set(namespace.to_string(), Arc::new(worker_set));
        if let Some(topology_namespace) = topology_namespace {
            self.reconcile_prefill_router_topology(model_name, &topology_namespace);
        }
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
    /// Holds [`Self::reservation_lock`] across the live-primary probe and the
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
                true
            }
        }
    }

    /// Remove a previously registered alias mapping once the alias has no WorkerSets.
    pub fn unregister_alias_if_empty(&self, alias: &str, primary: &str) {
        if self
            .models
            .get(alias)
            .is_some_and(|model| !model.is_empty())
        {
            return;
        }

        self.alias_to_primary
            .remove_if(alias, |_, existing| existing == primary);
    }

    /// Return the primary (canonical) model name for `model`, resolving aliases.
    /// Returns `model` unchanged if it is not an alias.
    pub fn resolve_canonical_name(&self, model: &str) -> String {
        self.alias_to_primary
            .get(model)
            .map(|v| v.value().clone())
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
        let model = self.models.get(model_name)?;
        let removed = model.remove_worker_set(namespace);
        let topology_namespace = removed.as_ref().and_then(|worker_set| {
            matches!(
                worker_set.card().worker_type,
                Some(
                    crate::worker_type::WorkerType::Prefill
                        | crate::worker_type::WorkerType::Decode
                )
            )
            .then(|| worker_set.namespace().to_string())
        });
        drop(model);
        if let Some(topology_namespace) = topology_namespace {
            self.reconcile_prefill_router_topology(model_name, &topology_namespace);
        }
        self.remove_model_if_empty(model_name);
        removed
    }

    // -- Model cards --

    pub fn get_model_cards(&self) -> Vec<ModelDeploymentCard> {
        self.cards.iter().map(|r| r.value().clone()).collect()
    }

    /// Return owned discovery instance keys for the locally recorded cards.
    /// Reconciliation must not hold DashMap guards while it performs
    /// asynchronous cleanup.
    pub fn get_model_card_keys(&self) -> Vec<String> {
        self.cards.iter().map(|r| r.key().clone()).collect()
    }

    /// Save a ModelDeploymentCard from an instance's key so we can fetch it later when the key is
    /// deleted.
    pub fn save_model_card(&self, key: &str, card: ModelDeploymentCard) -> anyhow::Result<()> {
        self.cards.insert(key.to_string(), card);
        Ok(())
    }

    /// Remove and return model card for this instance's key. We do this when the instance stops.
    pub fn get_model_card(&self, key: &str) -> Option<ModelDeploymentCard> {
        self.cards.get(key).map(|r| r.value().clone())
    }

    pub fn remove_model_card(&self, key: &str) -> Option<ModelDeploymentCard> {
        self.cards.remove(key).map(|(_, v)| v)
    }

    // -- Engine accessors (delegate through Model → WorkerSet) --

    /// Check if a decode model (chat or completions) is registered
    pub fn has_decode_model(&self, model: &str) -> bool {
        self.models
            .get(model)
            .is_some_and(|m| m.has_decode_engine())
    }

    /// Check if a prefill model is registered
    pub fn has_prefill_model(&self, model: &str) -> bool {
        self.models.get(model).is_some_and(|m| m.has_prefill())
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
        self.models.contains_key(model)
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
        self.models
            .get(model)
            .is_some_and(|m| m.is_ready_to_serve())
    }

    /// Whether any registered model can serve at least one inference request
    /// right now. See [`Model::is_ready_to_serve`].
    pub fn has_any_ready_model(&self) -> bool {
        self.models
            .iter()
            .any(|entry| entry.value().is_ready_to_serve())
    }

    pub fn model_display_names(&self) -> HashSet<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().is_displayable())
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Display names filtered to models that can actually serve a request right
    /// now — displayable AND with a complete worker set in at least one
    /// namespace ([`Model::has_ready_workers`]). This is the gate the HTTP
    /// listing/default-model paths should apply so a registered-but-incomplete
    /// deployment (e.g. decode-only with no prefill peer) is neither advertised
    /// nor chosen as an implicit default.
    pub fn serving_ready_display_names(&self) -> HashSet<String> {
        self.models
            .iter()
            .filter(|entry| {
                let model = entry.value();
                model.is_displayable() && model.has_ready_workers()
            })
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_chat_completions_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_chat_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_completions_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_completions_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_embeddings_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_embeddings_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_tensor_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_tensor_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_images_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_images_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_audios_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_audios_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_videos_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_videos_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_realtime_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_realtime_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_generate_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_generate_engine())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn list_prefill_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_prefill())
            .map(|entry| entry.key().clone())
            .collect()
    }

    pub fn get_embeddings_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIEmbeddingsStreamingEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_embeddings_engine()
    }

    pub fn get_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAICompletionsStreamingEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_completions_engine()
    }

    pub fn get_chat_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIChatCompletionsStreamingEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_chat_engine()
    }

    pub fn get_tensor_engine(
        &self,
        model: &str,
    ) -> Result<TensorStreamingEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_tensor_engine()
    }

    pub fn get_images_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIImagesStreamingEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_images_engine()
    }

    pub fn get_videos_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIVideosStreamingEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_videos_engine()
    }

    pub fn get_audios_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIAudiosStreamingEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_audios_engine()
    }

    pub fn get_realtime_engine(
        &self,
        model: &str,
    ) -> Result<RealtimeBidirectionalEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_realtime_engine()
    }

    pub fn get_generate_engine(
        &self,
        model: &str,
    ) -> Result<GenerateStreamingEngine, ModelManagerError> {
        self.models
            .get(model)
            .ok_or_else(|| ModelManagerError::ModelNotFound(model.to_string()))?
            .get_generate_engine()
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
        self.models
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
        self.models
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
        self.models
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
        Ok(())
    }

    pub fn add_completions_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAICompletionsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
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
        Ok(())
    }

    pub fn add_embeddings_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIEmbeddingsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
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
        Ok(())
    }

    pub fn add_tensor_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: TensorStreamingEngine,
    ) -> Result<(), ModelManagerError> {
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
        Ok(())
    }

    pub fn add_images_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIImagesStreamingEngine,
    ) -> Result<(), ModelManagerError> {
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
        Ok(())
    }

    pub fn add_videos_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIVideosStreamingEngine,
    ) -> Result<(), ModelManagerError> {
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
        Ok(())
    }

    pub fn add_audios_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIAudiosStreamingEngine,
    ) -> Result<(), ModelManagerError> {
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
        Ok(())
    }

    pub fn add_realtime_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: RealtimeBidirectionalEngine,
    ) -> Result<(), ModelManagerError> {
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
        Ok(())
    }

    pub fn add_generate_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: GenerateStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let model_entry = self.get_or_create_model(model);
        if model_entry.has_generate_engine() {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        let namespace = format!("__local_generate_{}", model);
        let mut ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            Self::aggregated_local_card(),
        );
        ws.generate_engine = Some(engine);
        model_entry.add_worker_set(namespace, Arc::new(ws));
        Ok(())
    }

    pub fn add_prefill_model(
        &self,
        model: &str,
        card_checksum: &str,
    ) -> Result<(), ModelManagerError> {
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
        Ok(())
    }

    // -- Model removal --

    /// Remove a model entirely (all its WorkerSets).
    /// Returns the removed Model, or None if not found.
    pub fn remove_model(&self, model: &str) -> Option<Arc<Model>> {
        self.models.remove(model).map(|(_, m)| m)
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

    /// Whether to start the LoRA load-estimator feed for a KV router being built for `worker_type`.
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

    #[allow(clippy::too_many_arguments)]
    pub async fn kv_chooser_for(
        &self,
        endpoint: &Endpoint,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
    ) -> anyhow::Result<Arc<KvRouter>> {
        if let Some(plugin) = RuntimePluginConfig::from_env()? {
            let router_role = match worker_type {
                crate::protocols::common::timing::WORKER_TYPE_DECODE => {
                    RuntimePluginRouterRole::Decode
                }
                crate::protocols::common::timing::WORKER_TYPE_PREFILL => {
                    RuntimePluginRouterRole::Prefill
                }
                _ => anyhow::bail!(
                    "runtime worker-selector plugin does not support worker type {worker_type:?}"
                ),
            };
            // SAFETY: Configuring the environment variable explicitly opts this process into
            // loading trusted native code that implements the documented plugin ABI.
            let selector =
                unsafe { plugin.load(router_role, kv_router_config.clone().unwrap_or_default()) }?;
            tracing::info!(
                ?selector,
                worker_type,
                "Using runtime worker-selector plugin"
            );
            return self
                .kv_chooser_for_scheduler_selector(
                    endpoint,
                    kv_cache_block_size,
                    kv_router_config,
                    prefill_load_estimator,
                    worker_type,
                    model_name,
                    is_eagle,
                    ValidatedWorkerSelector::new(selector),
                )
                .await;
        }

        let selector = DefaultWorkerSelector::new(kv_router_config.clone(), worker_type);
        self.kv_chooser_for_scheduler_selector(
            endpoint,
            kv_cache_block_size,
            kv_router_config,
            prefill_load_estimator,
            worker_type,
            model_name,
            is_eagle,
            selector,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn kv_chooser_for_scheduler_selector<Sel>(
        &self,
        endpoint: &Endpoint,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        selector: Sel,
    ) -> anyhow::Result<Arc<KvRouter>>
    where
        Sel: WorkerSelector<ModelRuntimeConfig> + Send + 'static,
    {
        let client = endpoint.client().await?;
        let lora_domain = self.lora_domain(&endpoint.id());

        // Register router via discovery mechanism.
        let discovery = endpoint.component().drt().discovery();
        let instance_id = discovery.instance_id();

        // Build transport for router endpoint based on request plane mode
        // Use the worker's component name so each target pool gets its own router discovery group
        let router_endpoint_id =
            router_endpoint_id(endpoint.id().namespace, endpoint.id().component);
        let transport = build_transport_type(endpoint, &router_endpoint_id, instance_id).await?;

        let discovery_spec = DiscoverySpec::Endpoint {
            namespace: router_endpoint_id.namespace.clone(),
            component: router_endpoint_id.component.clone(),
            endpoint: router_endpoint_id.name.clone(),
            transport,
            device_type: None,
        };

        discovery.register(discovery_spec).await?;

        // Get of create runtime config watcher for this endpoint
        let workers_with_configs = self.get_or_create_runtime_config_watcher(endpoint).await?;

        // Build shared cache client based on shared_cache_type.
        let shared_cache: Option<Box<dyn dynamo_kv_router::SharedKvCache>> = match kv_router_config
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
                Some(Box::new(HicacheSharedKvCache::new(
                    workers_with_configs.clone(),
                )))
            }
        };

        let effective_kv_router_config = kv_router_config.clone().unwrap_or_default();
        let kv_source_membership = if !effective_kv_router_config.use_remote_indexer
            && effective_kv_router_config.should_subscribe_to_kv_events()
        {
            Some(
                self.get_or_create_kv_source_membership_watch(endpoint)
                    .await?,
            )
        } else {
            None
        };

        let chooser = KvRouter::new(
            endpoint.clone(),
            client,
            workers_with_configs,
            kv_source_membership,
            kv_cache_block_size,
            selector,
            kv_router_config,
            prefill_load_estimator,
            worker_type,
            model_name,
            is_eagle,
            shared_cache,
            self.lora_enabled.then(|| lora_domain.filter.clone()),
        )
        .await?
        .into_erased();

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
        if Self::should_start_lora_load_feed(self.lora_enabled, worker_type) {
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

    // -- Prefill router coordination --
    // Keyed by "model_name:namespace" so each namespace's decode WorkerSet gets its own
    // prefill router activated by same-namespace prefill workers.

    /// Build a key for a (model, namespace) pair. Used for prefill router activators
    /// and registration guards.
    pub(crate) fn model_namespace_key(model_name: &str, namespace: &str) -> String {
        format!("{}:{}", model_name, namespace)
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
        let handle = crate::lora::LoraController::start_for_endpoint(
            endpoint_id.clone(),
            config,
            domain.routing_table.clone(),
            domain.state_tracker.clone(),
            domain.load_estimator.clone(),
            cancel_token,
        );
        self.lora_controller_handles.lock().push(handle);
    }

    pub fn lora_state_tracker_for(&self, endpoint_id: &EndpointId) -> LoraStateTracker {
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

    /// Register a prefill router for a decode WorkerSet. Returns a receiver that will be
    /// activated when the corresponding prefill model in the same namespace is discovered.
    /// Returns None if a decode WorkerSet in this namespace was already registered.
    pub fn register_prefill_router(
        &self,
        model_name: &str,
        namespace: &str,
        decode_endpoint: &EndpointId,
    ) -> anyhow::Result<Option<oneshot::Receiver<Endpoint>>> {
        if let Some(model) = self.get_model(model_name)
            && let Err(error) =
                model.prefill_router_topology_with_decode_candidate(namespace, decode_endpoint)
        {
            self.deactivate_prefill_router_topology(model_name, namespace);
            return Err(error.into());
        }

        let key = Self::model_namespace_key(model_name, namespace);
        // Use the entry API so the activator state mutation is atomic per-key:
        // a concurrent `remove_prefill_activator` (called by the watcher on
        // prefill-component teardown) can't slip into the gap between a
        // `remove`-then-`insert` pair and miss the cleanup, leaving a stale
        // PrefillReady cached for a prefill that's already gone.
        match self.prefill_router_activators.entry(key) {
            Entry::Occupied(o) => match o.get() {
                PrefillActivationState::PrefillReady(endpoint) => {
                    // Read the cached endpoint without removing the entry — its
                    // shard lock is held for the duration of the OccupiedEntry,
                    // so any concurrent prefill teardown serializes after us
                    // and observes the entry it needs to clear.
                    let endpoint_clone = (**endpoint).clone();
                    let (tx, rx) = oneshot::channel();
                    let _ = tx.send(endpoint_clone);
                    tracing::debug!(
                        model_name = %model_name,
                        namespace = %namespace,
                        "Prefill endpoint cached; returning fresh receiver"
                    );
                    Ok(Some(rx))
                }
                PrefillActivationState::DecodeWaiting(_) => {
                    // Decode already registered — entry stays in place so the
                    // existing live waiter isn't disturbed. Return None to
                    // signal the caller that this shouldn't have happened.
                    tracing::error!(
                        model_name = %model_name,
                        namespace = %namespace,
                        "Decode WorkerSet already registered for this prefill router"
                    );
                    Ok(None)
                }
            },
            Entry::Vacant(v) => {
                // New registration: create tx/rx pair, store sender, return receiver.
                let (tx, rx) = oneshot::channel();
                v.insert(PrefillActivationState::DecodeWaiting(tx));
                tracing::debug!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "No prefill endpoint for namespace yet, storing sender for future activation"
                );
                Ok(Some(rx))
            }
        }
    }

    fn deactivate_prefill_router_topology(&self, model_name: &str, namespace: &str) {
        let Some(model) = self.get_model(model_name) else {
            return;
        };
        for worker_set in model.prefill_routed_decode_worker_sets_in_namespace(namespace) {
            if let Some(ref prefill_router) = worker_set.prefill_router {
                prefill_router.deactivate();
            }
        }
    }

    fn reconcile_prefill_router_topology(&self, model_name: &str, namespace: &str) {
        let Some(model) = self.get_model(model_name) else {
            return;
        };
        let worker_set = match model.unique_prefill_routed_worker_set_in_namespace(namespace) {
            Ok(Some(worker_set)) => worker_set,
            Ok(None) => {
                self.deactivate_prefill_router_topology(model_name, namespace);
                return;
            }
            Err(error) => {
                tracing::error!(
                    model_name,
                    namespace,
                    %error,
                    "P/D routing is disabled until endpoint membership is unique"
                );
                self.deactivate_prefill_router_topology(model_name, namespace);
                return;
            }
        };

        let key = Self::model_namespace_key(model_name, namespace);
        let selected_prefill = model.worker_sets().into_iter().find_map(|worker_set| {
            (worker_set.namespace() == namespace
                && worker_set.card().worker_type == Some(crate::worker_type::WorkerType::Prefill))
            .then(|| worker_set.endpoint_id().cloned())
            .flatten()
        });
        let cached_prefill_matches =
            self.prefill_router_activators
                .get(&key)
                .is_some_and(|state| {
                    matches!(
                        state.value(),
                        PrefillActivationState::PrefillReady(endpoint)
                            if Some(endpoint.id()) == selected_prefill
                    )
                });
        if !cached_prefill_matches {
            tracing::error!(
                model_name,
                namespace,
                ?selected_prefill,
                "P/D routing remains disabled because the sole prefill endpoint does not match the cached activation"
            );
            self.deactivate_prefill_router_topology(model_name, namespace);
            return;
        }

        if let Some(prefill_router) = worker_set.prefill_router.as_ref()
            && prefill_router.is_deactivated()
        {
            prefill_router.reactivate();
        }
    }

    /// Activate a prefill router by sending the endpoint through the oneshot channel.
    /// The namespace must match the decode WorkerSet's namespace.
    pub fn activate_prefill_router(
        &self,
        model_name: &str,
        namespace: &str,
        endpoint: Endpoint,
    ) -> anyhow::Result<()> {
        let key = Self::model_namespace_key(model_name, namespace);

        // Resolve the namespace-level topology before mutating rendezvous
        // state. Endpoint-scoped WorkerSets are leaves, but a model namespace
        // supports only one P/D pairing; discovery order must not choose among
        // multiple prefill-routed decode endpoints.
        let reactivation_target = if let Some(model) = self.get_model(model_name) {
            match model.unique_prefill_routed_worker_set_in_namespace(namespace) {
                Ok(worker_set) => worker_set,
                Err(error) => {
                    self.deactivate_prefill_router_topology(model_name, namespace);
                    return Err(error.into());
                }
            }
        } else {
            None
        };

        // Reactivate any existing deactivated decode-side `PrefillRouter`. Used
        // by the PrefillReady-refresh and Vacant arms — the rebuilding case
        // for prefill workers that previously died and now rejoin.
        let reactivate_if_needed = || {
            if let Some(ws) = reactivation_target.as_ref()
                && let Some(ref pr) = ws.prefill_router
                && pr.is_deactivated()
            {
                pr.reactivate();
                true
            } else {
                false
            }
        };

        // Atomic per-key state transition via the entry API. Replaces the
        // previous `remove → process → insert` pattern, which left a window in
        // which a concurrent `remove_prefill_activator` (prefill teardown via
        // the watcher) could slip in, observe an empty map, and skip the
        // cleanup — letting a stale `PrefillReady` get resurrected here.
        match self.prefill_router_activators.entry(key) {
            Entry::Occupied(mut o) => {
                // Atomically swap the value to a fresh PrefillReady. The old
                // value tells us which transition we just performed.
                let new_value = PrefillActivationState::PrefillReady(Box::new(endpoint.clone()));
                let old = o.insert(new_value);
                // Drop the OccupiedEntry to release the shard lock before any
                // potentially-non-trivial work (e.g. nested DashMap accesses
                // via reactivate_if_needed). The state transition above is
                // already committed.
                drop(o);

                match old {
                    PrefillActivationState::DecodeWaiting(sender) => {
                        // Cold-start (or post-rebuild) handshake: decode
                        // registered first. Wake the waiting receiver.
                        sender.send(endpoint).map_err(|_| {
                            anyhow::anyhow!(
                                "Failed to send endpoint to prefill router activator for {}:{}",
                                model_name,
                                namespace
                            )
                        })?;
                        // Structural reconciliation deactivates an incomplete P/D
                        // topology while this waiter is pending. The now-unique
                        // prefill leaf makes it eligible again.
                        reactivate_if_needed();
                        tracing::info!(
                            model_name = %model_name,
                            namespace = %namespace,
                            "Activated prefill router for decode WorkerSet"
                        );
                    }
                    PrefillActivationState::PrefillReady(_) => {
                        // Stale PrefillReady from a prior handshake. Two cases:
                        //   (a) Duplicate activate_prefill_router call (e.g.,
                        //       the same prefill instance re-publishes its
                        //       endpoint) — just refresh, no router action.
                        //   (b) Prefill rejoin after a transient absence —
                        //       reactivate any deactivated decode-side router.
                        if reactivate_if_needed() {
                            tracing::info!(
                                model_name = %model_name,
                                namespace = %namespace,
                                "Reactivated existing prefill router for decode WorkerSet (prefill rejoin)"
                            );
                        } else {
                            tracing::debug!(
                                model_name = %model_name,
                                namespace = %namespace,
                                "Refreshed cached prefill endpoint for future decode WorkerSet rebuild"
                            );
                        }
                    }
                }
                Ok(())
            }
            Entry::Vacant(v) => {
                // No prior handshake state. Insert a fresh PrefillReady so a
                // future decode rebuild's register_prefill_router finds the
                // cache and activates immediately.
                v.insert(PrefillActivationState::PrefillReady(Box::new(endpoint)));

                // Then handle the prefill-rejoin case: an existing decode-side
                // PrefillRouter that was deactivated when prefill went away.
                if reactivate_if_needed() {
                    tracing::info!(
                        model_name = %model_name,
                        namespace = %namespace,
                        "Reactivated existing prefill router for decode WorkerSet (prefill rejoin)"
                    );
                } else {
                    tracing::info!(
                        model_name = %model_name,
                        namespace = %namespace,
                        "Stored prefill endpoint for future decode WorkerSet registration"
                    );
                }
                Ok(())
            }
        }
    }

    /// Deactivate the prefill router on the decode WorkerSet for the given model/namespace.
    /// Called by the watcher when all prefill workers in a namespace are removed.
    /// After deactivation, requests fall back to aggregated mode.
    pub fn deactivate_prefill_router_for_decode(&self, model_name: &str, namespace: &str) {
        let Some(model) = self.get_model(model_name) else {
            return;
        };

        if let Err(error) = model.unique_prefill_routed_worker_set_in_namespace(namespace) {
            tracing::error!(
                model_name,
                namespace,
                %error,
                "Deactivating every prefill router in ambiguous endpoint-scoped topology"
            );
        }

        for worker_set in model.prefill_routed_decode_worker_sets_in_namespace(namespace) {
            if let Some(ref prefill_router) = worker_set.prefill_router {
                prefill_router.deactivate();
            }
        }
    }

    /// Reconcile the prefill activator after one prefill WorkerSet is removed.
    /// The cached endpoint remains usable when that removal resolves a
    /// duplicate leaf; it is dropped only after the final prefill leaf leaves.
    pub fn remove_prefill_activator(&self, model_name: &str, namespace: &str) {
        if self.get_model(model_name).is_some_and(|model| {
            model.worker_sets().into_iter().any(|worker_set| {
                worker_set.namespace() == namespace
                    && worker_set.card().worker_type
                        == Some(crate::worker_type::WorkerType::Prefill)
            })
        }) {
            self.reconcile_prefill_router_topology(model_name, namespace);
            return;
        }

        let key = Self::model_namespace_key(model_name, namespace);
        if self.prefill_router_activators.remove(&key).is_some() {
            tracing::debug!(
                model_name = %model_name,
                namespace = %namespace,
                "Cleaned up prefill router activator for removed WorkerSet"
            );
        }
    }

    /// Remove a stale `DecodeWaiting(sender)` entry on decode WorkerSet teardown,
    /// while preserving any `PrefillReady(endpoint)` cache.
    ///
    /// When a decode WorkerSet is torn down, the `DecodeWaiting` entry's sender
    /// targets a `oneshot::Receiver` held by the now-dropped `PrefillRouter`. If
    /// we leave it in the map:
    ///   - the next decode rebuild's `register_prefill_router` finds the stale
    ///     `DecodeWaiting`, hits the `Some(DecodeWaiting)` arm, and returns
    ///     `None` — so the rebuilt WorkerSet has no `PrefillRouter` at all;
    ///   - when prefill finally registers, `activate_prefill_router` wakes the
    ///     orphaned receiver and activates a router that's about to be dropped,
    ///     producing log lines that look like success while the rebuilt
    ///     WorkerSet still has nothing.
    ///
    /// `PrefillReady` must be left intact — it's a cache of the prefill endpoint
    /// that survives decode rebuilds (PR 8965's primary contribution).
    pub fn remove_decode_prefill_waiter(&self, model_name: &str, namespace: &str) {
        let key = Self::model_namespace_key(model_name, namespace);
        // Atomic remove-if-stale: only drop the entry if it's `DecodeWaiting`,
        // leaving `PrefillReady` cache entries untouched.
        let removed = self.prefill_router_activators.remove_if(&key, |_, v| {
            matches!(v, PrefillActivationState::DecodeWaiting(_))
        });
        if removed.is_some() {
            tracing::debug!(
                model_name = %model_name,
                namespace = %namespace,
                "Removed stale DecodeWaiting activator on decode WorkerSet teardown"
            );
        }
    }

    /// Register the optional encoder hop for a token-serving WorkerSet.
    /// A cached Encode endpoint activates rebuilt consumers immediately.
    pub fn register_encoder_router(
        &self,
        model_name: &str,
        namespace: &str,
    ) -> Option<oneshot::Receiver<Endpoint>> {
        let key = Self::model_namespace_key(model_name, namespace);
        match self.encoder_router_activators.entry(key) {
            Entry::Occupied(mut o) => {
                if o.get().consumer.is_some() {
                    tracing::error!(
                        model_name = %model_name,
                        namespace = %namespace,
                        "Token WorkerSet already registered for this encoder router"
                    );
                    return None;
                }
                let (tx, rx) = oneshot::channel();
                let state = o.get_mut();
                if state.routing_enabled {
                    if let Some(endpoint) = state.endpoint.as_ref() {
                        let _ = tx.send((**endpoint).clone());
                    } else {
                        state.consumer = Some(tx);
                    }
                } else {
                    state.consumer = Some(tx);
                }
                Some(rx)
            }
            Entry::Vacant(v) => {
                let (tx, rx) = oneshot::channel();
                v.insert(EncoderActivationState {
                    consumer: Some(tx),
                    ..Default::default()
                });
                Some(rx)
            }
        }
    }

    /// Mark the model namespace as explicitly depending on an Encode worker.
    /// Activation waits for both this signal and an Encode endpoint so worker
    /// discovery order does not change routing behavior.
    pub fn enable_encoder_routing(&self, model_name: &str, namespace: &str) {
        let key = Self::model_namespace_key(model_name, namespace);
        // Option::zip would eagerly evaluate consumer.take(), dropping the
        // waiter when the Encode endpoint has not arrived yet.
        #[allow(clippy::manual_option_zip)]
        let sender_and_endpoint = match self.encoder_router_activators.entry(key) {
            Entry::Occupied(mut o) => {
                let state = o.get_mut();
                state.routing_enabled = true;
                state
                    .endpoint
                    .as_ref()
                    .map(|endpoint| (**endpoint).clone())
                    .and_then(|endpoint| state.consumer.take().map(|sender| (sender, endpoint)))
            }
            Entry::Vacant(v) => {
                v.insert(EncoderActivationState {
                    routing_enabled: true,
                    ..Default::default()
                });
                None
            }
        };
        if let Some((sender, endpoint)) = sender_and_endpoint {
            let _ = sender.send(endpoint);
        }
    }

    /// Publish an Encode endpoint and activate any waiting token pipeline.
    pub fn activate_encoder_router(&self, model_name: &str, namespace: &str, endpoint: Endpoint) {
        let key = Self::model_namespace_key(model_name, namespace);
        let sender = match self.encoder_router_activators.entry(key) {
            Entry::Occupied(mut o) => {
                let state = o.get_mut();
                state.endpoint = Some(Box::new(endpoint.clone()));
                state
                    .routing_enabled
                    .then(|| state.consumer.take())
                    .flatten()
            }
            Entry::Vacant(v) => {
                v.insert(EncoderActivationState {
                    endpoint: Some(Box::new(endpoint.clone())),
                    ..Default::default()
                });
                None
            }
        };
        if let Some(sender) = sender {
            if sender.send(endpoint).is_err() {
                tracing::warn!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "Encoder router consumer disappeared before activation; endpoint remains cached"
                );
            }
        } else {
            self.reactivate_encoder_routers(model_name, namespace);
        }
    }

    fn reactivate_encoder_routers(&self, model_name: &str, namespace: &str) {
        if let Some(model) = self.get_model(model_name) {
            for ws in model.worker_sets() {
                if ws.namespace() == namespace
                    && let Some(ref router) = ws.encoder_router
                    && router.is_deactivated()
                {
                    router.reactivate();
                }
            }
        }
    }

    /// Drop a stale Encode endpoint and make existing token pipelines bypass it.
    pub fn remove_encoder_activator(&self, model_name: &str, namespace: &str) {
        let key = Self::model_namespace_key(model_name, namespace);
        if let Entry::Occupied(mut o) = self.encoder_router_activators.entry(key) {
            let should_remove = {
                let state = o.get_mut();
                state.endpoint = None;
                state.consumer.is_none()
            };
            if should_remove {
                o.remove();
            }
        }
    }

    pub fn deactivate_encoder_router_for_consumers(&self, model_name: &str, namespace: &str) {
        if let Some(model) = self.get_model(model_name) {
            for ws in model.worker_sets() {
                if ws.namespace() == namespace
                    && let Some(ref router) = ws.encoder_router
                {
                    router.deactivate();
                }
            }
        }
    }

    /// Remove a waiter owned by a token WorkerSet that failed or was removed,
    /// while preserving a cached live Encode endpoint for rebuilds.
    pub fn remove_consumer_encoder_waiter(&self, model_name: &str, namespace: &str) {
        let key = Self::model_namespace_key(model_name, namespace);
        if let Some(mut state) = self.encoder_router_activators.get_mut(&key) {
            state.consumer = None;
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
        let rx = runtime_config_watch(endpoint).await?;
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
        discovery::{Discovery, DiscoverySpec, MockDiscovery, SharedMockRegistry},
        distributed::DistributedConfig,
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

    // -- Prefill router rendezvous tests --
    // Note: activate_prefill_router requires an Endpoint (needs DistributedRuntime),
    // so we test the registration state machine and cleanup only.

    fn decode_endpoint_id(namespace: &str) -> EndpointId {
        EndpointId {
            namespace: namespace.to_string(),
            component: "decode".to_string(),
            name: "generate".to_string(),
        }
    }

    #[test]
    fn test_prefill_router_register_new() {
        let mm = ModelManager::new();

        // First registration for a (model, namespace) returns Some(rx)
        let rx = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(rx.is_some());
    }

    #[test]
    fn test_prefill_router_double_register_returns_none() {
        let mm = ModelManager::new();

        let rx1 = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(rx1.is_some());

        // Second registration for the same (model, namespace) returns None
        let rx2 = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(rx2.is_none());
    }

    #[test]
    fn test_prefill_router_different_namespaces_independent() {
        let mm = ModelManager::new();

        // Different namespaces should be independent
        let rx1 = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        let rx2 = mm
            .register_prefill_router("llama", "ns2", &decode_endpoint_id("ns2"))
            .unwrap();
        assert!(rx1.is_some());
        assert!(rx2.is_some());
    }

    #[test]
    fn test_prefill_router_different_models_independent() {
        let mm = ModelManager::new();

        // Different models should be independent
        let rx1 = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        let rx2 = mm
            .register_prefill_router("gpt", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(rx1.is_some());
        assert!(rx2.is_some());
    }

    #[test]
    fn test_prefill_router_remove_allows_reregister() {
        let mm = ModelManager::new();

        let rx = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(rx.is_some());

        // Remove the activator
        mm.remove_prefill_activator("llama", "ns1");

        // Should be able to register again
        let rx2 = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(rx2.is_some());
    }

    #[test]
    fn test_prefill_router_remove_nonexistent_noop() {
        let mm = ModelManager::new();
        // Should not panic
        mm.remove_prefill_activator("llama", "ns1");
    }

    #[test]
    fn test_encoder_router_registration_and_waiter_cleanup() {
        let mm = ModelManager::new();
        let rx = mm.register_encoder_router("llama", "ns1");
        assert!(rx.is_some());
        assert!(mm.register_encoder_router("llama", "ns1").is_none());

        drop(rx);
        mm.remove_consumer_encoder_waiter("llama", "ns1");
        assert!(mm.register_encoder_router("llama", "ns1").is_some());
    }

    #[test]
    fn test_encoder_router_different_namespaces_are_independent() {
        let mm = ModelManager::new();
        assert!(mm.register_encoder_router("llama", "ns1").is_some());
        assert!(mm.register_encoder_router("llama", "ns2").is_some());
    }

    #[test]
    fn test_remove_encoder_activator_drops_stale_state_without_waiter() {
        let mm = ModelManager::new();
        let key = ModelManager::model_namespace_key("llama", "ns1");
        mm.encoder_router_activators.insert(
            key.clone(),
            EncoderActivationState {
                routing_enabled: true,
                ..Default::default()
            },
        );

        mm.remove_encoder_activator("llama", "ns1");

        assert!(!mm.encoder_router_activators.contains_key(&key));
        let _receiver = mm
            .register_encoder_router("llama", "ns1")
            .expect("registration after cleanup must succeed");
        let state = mm.encoder_router_activators.get(&key).unwrap();
        assert!(!state.routing_enabled);
        assert!(state.consumer.is_some());
    }

    #[test]
    fn test_remove_encoder_activator_preserves_waiting_consumer() {
        let mm = ModelManager::new();
        let key = ModelManager::model_namespace_key("llama", "ns1");
        let _receiver = mm
            .register_encoder_router("llama", "ns1")
            .expect("consumer registration must succeed");
        mm.enable_encoder_routing("llama", "ns1");

        mm.remove_encoder_activator("llama", "ns1");

        let state = mm.encoder_router_activators.get(&key).unwrap();
        assert!(state.consumer.is_some());
        assert!(state.endpoint.is_none());
        assert!(state.routing_enabled);
    }

    #[derive(Clone, Copy)]
    enum EncoderActivationSignal {
        Register,
        Enable,
        Activate,
    }

    #[tokio::test]
    async fn test_encoder_router_activates_in_every_signal_order() {
        use EncoderActivationSignal::{Activate, Enable, Register};

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("encoder-activation-orders".to_string())
            .unwrap()
            .component("encoder".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let orders = [
            [Register, Enable, Activate],
            [Register, Activate, Enable],
            [Enable, Register, Activate],
            [Enable, Activate, Register],
            [Activate, Register, Enable],
            [Activate, Enable, Register],
        ];

        for order in orders {
            let mm = ModelManager::new();
            let mut receiver = None;
            for signal in order {
                match signal {
                    Register => {
                        receiver = mm.register_encoder_router("llama", "ns1");
                    }
                    Enable => mm.enable_encoder_routing("llama", "ns1"),
                    Activate => {
                        mm.activate_encoder_router("llama", "ns1", endpoint.clone());
                    }
                }
            }

            let activated = receiver
                .expect("every order registers a consumer")
                .await
                .expect("all three signals must activate the consumer");
            assert_eq!(activated, endpoint);
        }

        runtime.shutdown();
    }

    #[tokio::test]
    async fn test_encoder_router_repeated_enable_preserves_waiter() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let endpoint = distributed
            .namespace("encoder-repeated-enable".to_string())
            .unwrap()
            .component("encoder".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        let mm = ModelManager::new();
        let receiver = mm
            .register_encoder_router("llama", "ns1")
            .expect("consumer registration must succeed");

        mm.enable_encoder_routing("llama", "ns1");
        mm.enable_encoder_routing("llama", "ns1");
        mm.activate_encoder_router("llama", "ns1", endpoint.clone());

        assert_eq!(receiver.await.unwrap(), endpoint);
        runtime.shutdown();
    }

    #[test]
    fn test_encoder_router_reactivates_every_matching_worker_set() {
        let router_a = crate::kv_router::EncoderRouter::disabled();
        let router_b = crate::kv_router::EncoderRouter::disabled();
        let mm = ModelManager::new();
        let mut ws_a = make_worker_set("ns1", "checksum-a");
        ws_a.encoder_router = Some(router_a.clone());
        let mut ws_b = make_worker_set("ns1", "checksum-b");
        ws_b.encoder_router = Some(router_b.clone());
        mm.add_worker_set("llama", "ns1:chat:decode", ws_a);
        mm.add_worker_set("llama", "ns1:completions:decode", ws_b);

        mm.deactivate_encoder_router_for_consumers("llama", "ns1");
        assert!(router_a.is_deactivated());
        assert!(router_b.is_deactivated());

        mm.reactivate_encoder_routers("llama", "ns1");

        assert!(!router_a.is_deactivated());
        assert!(!router_b.is_deactivated());
    }

    // -- remove_decode_prefill_waiter tests (stale-DecodeWaiting cleanup) --

    /// Decode WorkerSet teardown while still in DecodeWaiting must drop the
    /// stale waiter so a subsequent decode rebuild can register fresh.
    #[test]
    fn test_remove_decode_prefill_waiter_clears_decodewaiting() {
        let mm = ModelManager::new();

        // Decode registers first → DecodeWaiting in map.
        let rx1 = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(rx1.is_some());

        // Decode WorkerSet is removed before prefill registers. Drop the
        // receiver to mirror PrefillRouter being dropped along with the
        // WorkerSet.
        drop(rx1);

        // Watcher's decode-teardown path calls this:
        mm.remove_decode_prefill_waiter("llama", "ns1");

        // Rebuild path: a new register_prefill_router must succeed.
        let rx2 = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(
            rx2.is_some(),
            "after stale-DecodeWaiting cleanup, decode rebuild must get a fresh rx"
        );
    }

    /// Removing the waiter when the activator is already empty must not panic.
    #[test]
    fn test_remove_decode_prefill_waiter_empty_noop() {
        let mm = ModelManager::new();
        mm.remove_decode_prefill_waiter("llama", "ns1");
        // And the next register must still work.
        let rx = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap();
        assert!(rx.is_some());
    }

    #[test]
    fn test_model_namespace_key_format() {
        assert_eq!(
            ModelManager::model_namespace_key("llama", "ns1"),
            "llama:ns1"
        );
        assert_eq!(
            ModelManager::model_namespace_key("gpt-4", "default-abc"),
            "gpt-4:default-abc"
        );
    }

    // -- deactivate_prefill_router_for_decode tests --

    use crate::kv_router::PrefillRouter;

    /// Helper: make a WorkerSet with an activated PrefillRouter attached.
    fn make_worker_set_with_prefill_router(namespace: &str, mdcsum: &str) -> WorkerSet {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(crate::worker_type::WorkerType::Decode);
        let mut ws = WorkerSet::new(namespace.to_string(), mdcsum.to_string(), card);
        let pr = PrefillRouter::disabled(
            std::sync::Arc::new(ModelManager::new()),
            dynamo_runtime::pipeline::RouterMode::RoundRobin,
            None,
        );
        pr.mark_active_for_test();
        ws.prefill_router = Some(pr);
        ws
    }

    fn make_endpoint_worker_set_with_prefill_router(namespace: &str, component: &str) -> WorkerSet {
        let mut worker_set = make_worker_set_with_prefill_router(namespace, component);
        worker_set.set_endpoint_id(EndpointId {
            namespace: namespace.to_string(),
            component: component.to_string(),
            name: "generate".to_string(),
        });
        worker_set
    }

    fn make_typed_endpoint_worker_set(
        namespace: &str,
        component: &str,
        worker_type: crate::worker_type::WorkerType,
    ) -> WorkerSet {
        let mut card = ModelDeploymentCard::default();
        card.worker_type = Some(worker_type);
        let mut worker_set = WorkerSet::new(namespace.to_string(), component.to_string(), card);
        worker_set.set_endpoint_id(EndpointId {
            namespace: namespace.to_string(),
            component: component.to_string(),
            name: "generate".to_string(),
        });
        worker_set
    }

    /// Calling deactivate on a non-existent model must not panic.
    #[test]
    fn test_deactivate_prefill_router_for_decode_noop_missing_model() {
        let mm = ModelManager::new();
        mm.deactivate_prefill_router_for_decode("nonexistent", "ns1");
    }

    /// Calling deactivate on a WorkerSet without a prefill_router must not panic.
    #[test]
    fn test_deactivate_prefill_router_for_decode_noop_no_router() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"));
        mm.deactivate_prefill_router_for_decode("llama", "ns1");
    }

    /// Deactivation updates routing lifecycle but does not add a second visibility policy on top
    /// of registered worker topology.
    #[test]
    fn test_deactivate_prefill_router_for_decode_keeps_model_visible() {
        let mm = ModelManager::new();
        mm.add_worker_set(
            "llama",
            "ns1",
            make_worker_set_with_prefill_router("ns1", "abc"),
        );

        // Model is visible before deactivation.
        assert!(mm.model_display_names().contains("llama"));

        mm.deactivate_prefill_router_for_decode("llama", "ns1");

        assert!(mm.model_display_names().contains("llama"));
        let router = mm
            .get_model("llama")
            .and_then(|model| model.get_worker_set("ns1"))
            .and_then(|ws| ws.prefill_router.clone())
            .expect("prefill router");
        assert!(router.is_deactivated());

        // Idempotent: calling again must not panic.
        mm.deactivate_prefill_router_for_decode("llama", "ns1");
        assert!(mm.model_display_names().contains("llama"));
    }

    #[tokio::test]
    async fn ambiguous_decode_leaf_disables_only_pd_and_recovers_on_removal() {
        use crate::worker_type::WorkerType;

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let mm = ModelManager::new();
        mm.add_worker_set(
            "llama",
            "decode-a",
            make_endpoint_worker_set_with_prefill_router("ns1", "decode-a"),
        );
        mm.add_worker_set(
            "llama",
            "prefill",
            make_typed_endpoint_worker_set("ns1", "prefill", WorkerType::Prefill),
        );
        mm.add_worker_set("llama", "ordinary", make_worker_set("ns1", "ordinary"));

        let endpoint = distributed
            .namespace("ns1".to_string())
            .unwrap()
            .component("prefill".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        mm.prefill_router_activators.insert(
            ModelManager::model_namespace_key("llama", "ns1"),
            PrefillActivationState::PrefillReady(Box::new(endpoint)),
        );
        let router = mm
            .get_model("llama")
            .and_then(|model| model.get_worker_set("decode-a"))
            .and_then(|worker_set| worker_set.prefill_router.clone())
            .expect("decode router");
        router.mark_active_for_test();

        mm.add_worker_set(
            "llama",
            "decode-b",
            make_typed_endpoint_worker_set("ns1", "decode-b", WorkerType::Decode),
        );

        assert!(router.is_deactivated());
        assert_eq!(mm.get_model("llama").unwrap().worker_set_count(), 4);
        assert!(mm.get_model("llama").unwrap().has_worker_set("ordinary"));

        mm.remove_worker_set("llama", "decode-b");
        assert!(!router.is_deactivated());
        assert_eq!(mm.get_model("llama").unwrap().worker_set_count(), 3);
        assert!(mm.get_model("llama").unwrap().has_worker_set("ordinary"));
        runtime.shutdown();
    }

    #[tokio::test]
    async fn completing_initial_pd_handshake_reactivates_decode_router() {
        use crate::worker_type::WorkerType;

        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let mm = ModelManager::new();
        let activation = mm
            .register_prefill_router("llama", "ns1", &decode_endpoint_id("ns1"))
            .unwrap()
            .expect("decode activation receiver");

        mm.add_worker_set(
            "llama",
            "decode",
            make_endpoint_worker_set_with_prefill_router("ns1", "decode"),
        );
        let router = mm
            .get_model("llama")
            .and_then(|model| model.get_worker_set("decode"))
            .and_then(|worker_set| worker_set.prefill_router.clone())
            .expect("decode router");
        assert!(router.is_deactivated());

        mm.add_worker_set(
            "llama",
            "prefill",
            make_typed_endpoint_worker_set("ns1", "prefill", WorkerType::Prefill),
        );
        let endpoint = distributed
            .namespace("ns1".to_string())
            .unwrap()
            .component("prefill".to_string())
            .unwrap()
            .endpoint("generate".to_string());
        mm.activate_prefill_router("llama", "ns1", endpoint)
            .unwrap();

        assert!(!router.is_deactivated());
        drop(activation);
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
