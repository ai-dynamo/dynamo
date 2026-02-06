// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use dashmap::{DashMap, mapref::entry::Entry};
use tokio::sync::oneshot;

use super::worker_monitor::LoadThresholdConfig;
use super::{KvWorkerMonitor, Model, RuntimeConfigWatch, WorkerSet, runtime_config_watch};

use dynamo_runtime::{
    component::{Client, Endpoint, build_transport_type},
    discovery::DiscoverySpec,
    prelude::DistributedRuntimeProvider,
    protocols::EndpointId,
};

use crate::{
    kv_router::{
        KvRouter, KvRouterConfig, protocols::WorkerId, router_endpoint_id,
        scheduler::DefaultWorkerSelector,
    },
    local_model::runtime_config::DisaggregatedEndpoint,
    model_card::ModelDeploymentCard,
    types::{
        generic::tensor::TensorStreamingEngine,
        openai::{
            chat_completions::OpenAIChatCompletionsStreamingEngine,
            completions::OpenAICompletionsStreamingEngine,
            embeddings::OpenAIEmbeddingsStreamingEngine, images::OpenAIImagesStreamingEngine,
            videos::OpenAIVideosStreamingEngine,
        },
    },
};

/// State for prefill router activation rendezvous
enum PrefillActivationState {
    /// Decode model registered, waiting for prefill endpoint
    DecodeWaiting(oneshot::Sender<Endpoint>),
    /// Prefill endpoint arrived, waiting for decode model to register
    PrefillReady(oneshot::Receiver<Endpoint>),
}

#[derive(Debug, thiserror::Error)]
pub enum ModelManagerError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model already exists: {0}")]
    ModelAlreadyExists(String),
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

    /// Per-instance model cards, keyed by instance path. Used for cleanup on worker removal.
    cards: DashMap<String, ModelDeploymentCard>,

    /// Prefill router activation rendezvous, keyed by model name.
    prefill_router_activators: DashMap<String, PrefillActivationState>,

    /// Per-endpoint runtime config watchers. Keyed by EndpointId (includes namespace).
    runtime_configs: DashMap<EndpointId, RuntimeConfigWatch>,
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
            runtime_configs: DashMap::new(),
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
        self.models.get(model_name).map(|entry| entry.value().clone())
    }

    /// Remove a Model if it has no remaining WorkerSets.
    pub fn remove_model_if_empty(&self, model_name: &str) {
        // Only remove if the model is empty (no worker sets)
        if let Some(model) = self.models.get(model_name) {
            if model.is_empty() {
                drop(model); // Release the read reference before removing
                self.models.remove(model_name);
                tracing::info!(model_name, "Removed empty model from manager");
            }
        }
    }

    /// Add a WorkerSet to a Model. Creates the Model if it doesn't exist.
    pub fn add_worker_set(
        &self,
        model_name: &str,
        namespace: &str,
        worker_set: WorkerSet,
    ) {
        let model = self.get_or_create_model(model_name);
        model.add_worker_set(namespace.to_string(), Arc::new(worker_set));
    }

    /// Remove a WorkerSet from a Model. Removes the Model if it becomes empty.
    pub fn remove_worker_set(
        &self,
        model_name: &str,
        namespace: &str,
    ) -> Option<Arc<WorkerSet>> {
        let model = self.models.get(model_name)?;
        let removed = model.remove_worker_set(namespace);
        drop(model);
        self.remove_model_if_empty(model_name);
        removed
    }

    // -- Checksum validation --

    /// Check if a candidate checksum is valid for a model in a specific namespace.
    /// Returns Some(true) if valid, Some(false) if mismatches existing WorkerSet in same namespace,
    /// None if model doesn't exist or no WorkerSet exists for this namespace (new namespace is OK).
    pub fn is_valid_checksum(
        &self,
        _model_type: crate::model_type::ModelType,
        model_name: &str,
        namespace: &str,
        candidate_checksum: &str,
    ) -> Option<bool> {
        let model = self.models.get(model_name)?;
        let existing_checksum = model.checksum_for_namespace(namespace)?;
        tracing::debug!(
            model_name,
            namespace,
            existing_checksum,
            candidate_checksum,
            "is_valid_checksum: checking against same-namespace WorkerSet"
        );
        Some(existing_checksum == candidate_checksum)
    }

    // -- Model cards --

    pub fn get_model_cards(&self) -> Vec<ModelDeploymentCard> {
        self.cards.iter().map(|r| r.value().clone()).collect()
    }

    /// Save a ModelDeploymentCard from an instance's key so we can fetch it later when the key is
    /// deleted.
    pub fn save_model_card(&self, key: &str, card: ModelDeploymentCard) -> anyhow::Result<()> {
        self.cards.insert(key.to_string(), card);
        Ok(())
    }

    /// Remove and return model card for this instance's key. We do this when the instance stops.
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
        self.models
            .get(model)
            .is_some_and(|m| m.has_prefill())
    }

    /// Check if any model (decode or prefill) is registered.
    pub fn has_model_any(&self, model: &str) -> bool {
        self.has_decode_model(model) || self.has_prefill_model(model)
    }

    pub fn model_display_names(&self) -> HashSet<String> {
        let mut names = HashSet::new();
        for entry in self.models.iter() {
            let model = entry.value();
            if model.has_chat_engine()
                || model.has_completions_engine()
                || model.has_embeddings_engine()
                || model.has_tensor_engine()
                || model.has_images_engine()
                || model.has_videos_engine()
                || model.has_prefill()
            {
                names.insert(entry.key().clone());
            }
        }
        names
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

    pub fn list_videos_models(&self) -> Vec<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().has_videos_engine())
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

    // -- Convenience methods for in-process models (http.rs, grpc.rs) --
    // These create a WorkerSet with a default namespace for local models.

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
            ModelDeploymentCard::default(),
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
            ModelDeploymentCard::default(),
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
            ModelDeploymentCard::default(),
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
            ModelDeploymentCard::default(),
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
            ModelDeploymentCard::default(),
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
            ModelDeploymentCard::default(),
        );
        ws.videos_engine = Some(engine);
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
        let ws = WorkerSet::new(
            namespace.clone(),
            card_checksum.to_string(),
            ModelDeploymentCard::default(),
        );
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

    // -- KV Router creation --

    pub async fn kv_chooser_for(
        &self,
        endpoint: &Endpoint,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        worker_type: &'static str,
    ) -> anyhow::Result<Arc<KvRouter>> {
        let client = endpoint.client().await?;

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
        };

        discovery.register(discovery_spec).await?;

        let workers_with_configs = self.get_or_create_runtime_config_watcher(endpoint).await?;

        let selector = Box::new(DefaultWorkerSelector::new(kv_router_config));
        let chooser = KvRouter::new(
            endpoint.clone(),
            client,
            workers_with_configs,
            kv_cache_block_size,
            Some(selector),
            kv_router_config,
            worker_type,
        )
        .await?;
        Ok(Arc::new(chooser))
    }

    // -- Worker monitor creation --

    pub fn get_or_create_worker_monitor(
        &self,
        model: &str,
        client: Client,
        config: LoadThresholdConfig,
    ) -> KvWorkerMonitor {
        // Check if any existing WorkerSet for this model has a monitor
        if let Some(existing) = self.get_worker_monitor(model) {
            existing.set_load_threshold_config(&config);
            return existing;
        }
        // Create new monitor
        KvWorkerMonitor::new(client, config)
    }

    // -- Prefill router coordination --
    // Keyed by "model_name:namespace" so each namespace's decode WorkerSet gets its own
    // prefill router activated by same-namespace prefill workers.

    fn prefill_key(model_name: &str, namespace: &str) -> String {
        format!("{}:{}", model_name, namespace)
    }

    /// Register a prefill router for a decode WorkerSet. Returns a receiver that will be
    /// activated when the corresponding prefill model in the same namespace is discovered.
    /// Returns None if a decode WorkerSet in this namespace was already registered.
    pub fn register_prefill_router(
        &self,
        model_name: String,
        namespace: &str,
    ) -> Option<oneshot::Receiver<Endpoint>> {
        let key = Self::prefill_key(&model_name, namespace);
        match self.prefill_router_activators.remove(&key) {
            Some((_, PrefillActivationState::PrefillReady(rx))) => {
                tracing::debug!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "Prefill endpoint already available for namespace, returning receiver"
                );
                Some(rx)
            }
            Some((key, PrefillActivationState::DecodeWaiting(tx))) => {
                tracing::error!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "Decode WorkerSet already registered for this prefill router"
                );
                self.prefill_router_activators
                    .insert(key, PrefillActivationState::DecodeWaiting(tx));
                None
            }
            None => {
                let (tx, rx) = oneshot::channel();
                self.prefill_router_activators.insert(
                    key,
                    PrefillActivationState::DecodeWaiting(tx),
                );
                tracing::debug!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "No prefill endpoint for namespace yet, storing sender for future activation"
                );
                Some(rx)
            }
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
        let key = Self::prefill_key(model_name, namespace);
        match self.prefill_router_activators.remove(&key) {
            Some((_, PrefillActivationState::DecodeWaiting(sender))) => {
                sender.send(endpoint).map_err(|_| {
                    anyhow::anyhow!(
                        "Failed to send endpoint to prefill router activator for {}:{}",
                        model_name,
                        namespace
                    )
                })?;
                tracing::info!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "Activated prefill router for decode WorkerSet"
                );
                Ok(())
            }
            Some((_, PrefillActivationState::PrefillReady(_))) => {
                anyhow::bail!(
                    "Prefill router for {}:{} already activated",
                    model_name,
                    namespace
                );
            }
            None => {
                let (tx, rx) = oneshot::channel();
                tx.send(endpoint).map_err(|_| {
                    anyhow::anyhow!(
                        "Failed to send endpoint for prefill model {}:{}",
                        model_name,
                        namespace
                    )
                })?;
                self.prefill_router_activators.insert(
                    key,
                    PrefillActivationState::PrefillReady(rx),
                );
                tracing::info!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "Stored prefill endpoint for future decode WorkerSet registration"
                );
                Ok(())
            }
        }
    }

    /// Remove the prefill router activator for a (model, namespace) pair.
    /// Called when a WorkerSet is removed to prevent stale activators.
    pub fn remove_prefill_activator(&self, model_name: &str, namespace: &str) {
        let key = Self::prefill_key(model_name, namespace);
        if self.prefill_router_activators.remove(&key).is_some() {
            tracing::debug!(
                model_name = %model_name,
                namespace = %namespace,
                "Cleaned up prefill router activator for removed WorkerSet"
            );
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

    /// Gets an existing worker monitor for a model, if one exists.
    pub fn get_worker_monitor(&self, model: &str) -> Option<KvWorkerMonitor> {
        let model_entry = self.models.get(model)?;
        model_entry.get_worker_monitor()
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
}
