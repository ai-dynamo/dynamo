// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc};

use dashmap::{DashMap, mapref::entry::Entry};
use tokio::sync::oneshot;

use super::worker_monitor::LoadThresholdConfig;
use super::{KvWorkerMonitor, Model, RuntimeConfigWatch, WorkerSet, runtime_config_watch};

use dynamo_runtime::{
    component::{Endpoint, build_transport_type},
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

    #[error(
        "Checksum mismatch for model {model}: expected {expected}, got {got}. All WorkerSets of a model must share the same checksum. Drain all old workers before deploying a new version."
    )]
    ChecksumMismatch {
        model: String,
        expected: String,
        got: String,
    },
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

    /// Prefill router activation rendezvous, keyed by "model_name:namespace".
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

    /// Add a WorkerSet to a Model. Creates the Model if it doesn't exist.
    /// Returns `Err` if the WorkerSet's checksum doesn't match the model's canonical checksum.
    pub fn add_worker_set(
        &self,
        model_name: &str,
        namespace: &str,
        worker_set: WorkerSet,
    ) -> Result<(), ModelManagerError> {
        let model = self.get_or_create_model(model_name);
        model.add_worker_set(namespace.to_string(), Arc::new(worker_set))
    }

    /// Remove a WorkerSet from a Model. Removes the Model if it becomes empty.
    pub fn remove_worker_set(&self, model_name: &str, namespace: &str) -> Option<Arc<WorkerSet>> {
        let model = self.models.get(model_name)?;
        let removed = model.remove_worker_set(namespace);
        drop(model);
        self.remove_model_if_empty(model_name);
        removed
    }

    // -- Checksum validation --

    /// Check if a candidate checksum is valid for a model.
    /// Returns `Some(true)` if it matches the model's canonical checksum, `Some(false)` if it
    /// doesn't match, or `None` if the model doesn't exist or has no canonical checksum yet.
    pub fn is_valid_checksum(&self, model_name: &str, candidate_checksum: &str) -> Option<bool> {
        let model = self.models.get(model_name)?;
        model.is_valid_checksum(candidate_checksum)
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
        self.models.get(model).is_some_and(|m| m.has_prefill())
    }

    /// Check if any model (decode or prefill) is registered.
    pub fn has_model_any(&self, model: &str) -> bool {
        self.has_decode_model(model) || self.has_prefill_model(model)
    }

    pub fn model_display_names(&self) -> HashSet<String> {
        self.models
            .iter()
            .filter(|entry| entry.value().is_displayable())
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
    // TODO: These methods use ModelDeploymentCard::default() for the WorkerSet, which means
    // parsing_options() returns defaults (no tool_call_parser/reasoning_parser). Pass the real
    // MDC from callers so ParsingOptions reflect the model's actual configuration.

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
        model_entry.add_worker_set(namespace, Arc::new(ws))?;
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
        model_entry.add_worker_set(namespace, Arc::new(ws))?;
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
        model_entry.add_worker_set(namespace, Arc::new(ws))?;
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
        model_entry.add_worker_set(namespace, Arc::new(ws))?;
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
        model_entry.add_worker_set(namespace, Arc::new(ws))?;
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
        model_entry.add_worker_set(namespace, Arc::new(ws))?;
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
        model_entry.add_worker_set(namespace, Arc::new(ws))?;
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
        };

        discovery.register(discovery_spec).await?;

        // Get of create runtime config watcher for this endpoint
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

    // -- Prefill router coordination --
    // Keyed by "model_name:namespace" so each namespace's decode WorkerSet gets its own
    // prefill router activated by same-namespace prefill workers.

    /// Build a key for a (model, namespace) pair. Used for prefill router activators
    /// and registration guards.
    pub(crate) fn model_namespace_key(model_name: &str, namespace: &str) -> String {
        format!("{}:{}", model_name, namespace)
    }

    /// Register a prefill router for a decode WorkerSet. Returns a receiver that will be
    /// activated when the corresponding prefill model in the same namespace is discovered.
    /// Returns None if a decode WorkerSet in this namespace was already registered.
    pub fn register_prefill_router(
        &self,
        model_name: &str,
        namespace: &str,
    ) -> Option<oneshot::Receiver<Endpoint>> {
        let key = Self::model_namespace_key(model_name, namespace);
        match self.prefill_router_activators.remove(&key) {
            Some((_, PrefillActivationState::PrefillReady(rx))) => {
                // Prefill endpoint already arrived - rx will immediately resolve
                tracing::debug!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "Prefill endpoint already available for namespace, returning receiver"
                );
                Some(rx)
            }
            Some((key, PrefillActivationState::DecodeWaiting(tx))) => {
                // Decode already registered - this shouldn't happen, restore state and return None
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
                // New registration: create tx/rx pair, store sender and return receiver
                let (tx, rx) = oneshot::channel();
                self.prefill_router_activators
                    .insert(key, PrefillActivationState::DecodeWaiting(tx));
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
        let key = Self::model_namespace_key(model_name, namespace);
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
                // Try to reactivate an existing deactivated router first.
                // This handles prefill rejoin after a transient failure: the decode
                // WorkerSet's PrefillRouter already exists but is in Deactivated state.
                if let Some(model) = self.get_model(model_name)
                    && let Some(ws) = model.get_worker_set(namespace)
                    && let Some(ref pr) = ws.prefill_router
                {
                    pr.reactivate(endpoint);
                    tracing::info!(
                        model_name = %model_name,
                        namespace = %namespace,
                        "Reactivated existing prefill router for decode WorkerSet (prefill rejoin)"
                    );
                    return Ok(());
                }

                // No existing router — store endpoint for a future decode registration.
                let (tx, rx) = oneshot::channel();
                tx.send(endpoint).map_err(|_| {
                    anyhow::anyhow!(
                        "Failed to send endpoint for prefill model {}:{}",
                        model_name,
                        namespace
                    )
                })?;
                self.prefill_router_activators
                    .insert(key, PrefillActivationState::PrefillReady(rx));
                tracing::info!(
                    model_name = %model_name,
                    namespace = %namespace,
                    "Stored prefill endpoint for future decode WorkerSet registration"
                );
                Ok(())
            }
        }
    }

    /// Deactivate the live prefill router on the decode WorkerSet for a (model, namespace).
    /// Called when the prefill WorkerSet is removed so the decode side falls back to
    /// aggregated mode (or fails cleanly with enforce_disagg).
    pub fn deactivate_prefill_router_for_decode(&self, model_name: &str, namespace: &str) {
        if let Some(model) = self.get_model(model_name)
            && let Some(ws) = model.get_worker_set(namespace)
            && let Some(ref pr) = ws.prefill_router
        {
            pr.deactivate();
        }
    }

    /// Remove the prefill router activator for a (model, namespace) pair.
    /// Called when a WorkerSet is removed to prevent stale activators.
    pub fn remove_prefill_activator(&self, model_name: &str, namespace: &str) {
        let key = Self::model_namespace_key(model_name, namespace);
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

    /// Gets an existing worker monitor for a specific namespace of a model.
    pub fn get_worker_monitor_for_namespace(
        &self,
        model: &str,
        namespace: &str,
    ) -> Option<KvWorkerMonitor> {
        let model_entry = self.models.get(model)?;
        model_entry.get_worker_monitor_for_namespace(namespace)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_card::ModelDeploymentCard;

    fn make_worker_set(namespace: &str, mdcsum: &str) -> WorkerSet {
        WorkerSet::new(
            namespace.to_string(),
            mdcsum.to_string(),
            ModelDeploymentCard::default(),
        )
    }

    // -- CRUD delegation tests --

    #[test]
    fn test_add_and_get_worker_set() {
        let mm = ModelManager::new();
        let ws = make_worker_set("ns1", "abc");
        mm.add_worker_set("llama", "ns1", ws).unwrap();

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

        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        assert!(mm.get_model("llama").is_some());
    }

    #[test]
    fn test_remove_worker_set_removes_empty_model() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
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
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        mm.add_worker_set("llama", "ns2", make_worker_set("ns2", "abc"))
            .unwrap();

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
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        assert!(mm.remove_worker_set("llama", "ns2").is_none());

        // Model should still exist (ns1 still there)
        assert!(mm.get_model("llama").is_some());
    }

    #[test]
    fn test_remove_model_if_empty_noop_when_not_empty() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();

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
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        mm.add_worker_set("llama", "ns2", make_worker_set("ns2", "abc"))
            .unwrap();

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

    // -- Checksum validation tests --

    #[test]
    fn test_is_valid_checksum_match() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc123"))
            .unwrap();

        assert_eq!(mm.is_valid_checksum("llama", "abc123"), Some(true));
    }

    #[test]
    fn test_is_valid_checksum_mismatch() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc123"))
            .unwrap();

        assert_eq!(mm.is_valid_checksum("llama", "wrong"), Some(false));
    }

    #[test]
    fn test_is_valid_checksum_no_canonical_yet() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc123"))
            .unwrap();

        // Canonical is set, so even for a "new namespace" scenario the checksum is checked
        assert_eq!(mm.is_valid_checksum("llama", "abc123"), Some(true));
        assert_eq!(mm.is_valid_checksum("llama", "xyz"), Some(false));
    }

    #[test]
    fn test_is_valid_checksum_missing_model() {
        let mm = ModelManager::new();
        assert_eq!(mm.is_valid_checksum("nonexistent", "abc"), None);
    }

    #[test]
    fn test_is_valid_checksum_cross_namespace_enforcement() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "checksum_a"))
            .unwrap();

        // A different namespace with a different checksum should be rejected at the model level
        assert_eq!(mm.is_valid_checksum("llama", "checksum_b"), Some(false));

        // Same checksum is accepted
        assert_eq!(mm.is_valid_checksum("llama", "checksum_a"), Some(true));
    }

    // -- Model listing and filtering tests --

    #[test]
    fn test_has_decode_model() {
        let mm = ModelManager::new();

        // No model → false
        assert!(!mm.has_decode_model("llama"));

        // Prefill-only set (no engines) → false
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        assert!(!mm.has_decode_model("llama"));
    }

    #[test]
    fn test_has_prefill_model() {
        let mm = ModelManager::new();

        // Prefill set = no engines
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        assert!(mm.has_prefill_model("llama"));
    }

    #[test]
    fn test_has_model_any() {
        let mm = ModelManager::new();
        assert!(!mm.has_model_any("llama"));

        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        assert!(mm.has_model_any("llama")); // has prefill
    }

    #[test]
    fn test_model_display_names_includes_prefill() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();

        let names = mm.model_display_names();
        assert!(names.contains("llama"));
    }

    #[test]
    fn test_model_display_names_empty() {
        let mm = ModelManager::new();
        assert!(mm.model_display_names().is_empty());
    }

    #[test]
    fn test_list_prefill_models() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        mm.add_worker_set("gpt", "ns1", make_worker_set("ns1", "def"))
            .unwrap();

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

    #[test]
    fn test_prefill_router_register_new() {
        let mm = ModelManager::new();

        // First registration for a (model, namespace) returns Some(rx)
        let rx = mm.register_prefill_router("llama", "ns1");
        assert!(rx.is_some());
    }

    #[test]
    fn test_prefill_router_double_register_returns_none() {
        let mm = ModelManager::new();

        let rx1 = mm.register_prefill_router("llama", "ns1");
        assert!(rx1.is_some());

        // Second registration for the same (model, namespace) returns None
        let rx2 = mm.register_prefill_router("llama", "ns1");
        assert!(rx2.is_none());
    }

    #[test]
    fn test_prefill_router_different_namespaces_independent() {
        let mm = ModelManager::new();

        // Different namespaces should be independent
        let rx1 = mm.register_prefill_router("llama", "ns1");
        let rx2 = mm.register_prefill_router("llama", "ns2");
        assert!(rx1.is_some());
        assert!(rx2.is_some());
    }

    #[test]
    fn test_prefill_router_different_models_independent() {
        let mm = ModelManager::new();

        // Different models should be independent
        let rx1 = mm.register_prefill_router("llama", "ns1");
        let rx2 = mm.register_prefill_router("gpt", "ns1");
        assert!(rx1.is_some());
        assert!(rx2.is_some());
    }

    #[test]
    fn test_prefill_router_remove_allows_reregister() {
        let mm = ModelManager::new();

        let rx = mm.register_prefill_router("llama", "ns1");
        assert!(rx.is_some());

        // Remove the activator
        mm.remove_prefill_activator("llama", "ns1");

        // Should be able to register again
        let rx2 = mm.register_prefill_router("llama", "ns1");
        assert!(rx2.is_some());
    }

    #[test]
    fn test_prefill_router_remove_nonexistent_noop() {
        let mm = ModelManager::new();
        // Should not panic
        mm.remove_prefill_activator("llama", "ns1");
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

    /// Calling deactivate_prefill_router_for_decode on a non-existent model must not panic.
    #[test]
    fn test_deactivate_prefill_router_for_decode_noop_missing_model() {
        let mm = ModelManager::new();
        // No model registered — must be a no-op
        mm.deactivate_prefill_router_for_decode("nonexistent", "ns1");
    }

    /// Calling deactivate_prefill_router_for_decode on a WorkerSet that has no
    /// prefill_router must not panic (pure aggregated deployment).
    #[test]
    fn test_deactivate_prefill_router_for_decode_noop_no_router() {
        let mm = ModelManager::new();
        mm.add_worker_set("llama", "ns1", make_worker_set("ns1", "abc"))
            .unwrap();
        // WorkerSet.prefill_router is None — must be a no-op
        mm.deactivate_prefill_router_for_decode("llama", "ns1");
    }

    /// Full pipeline test: deactivate_prefill_router_for_decode finds the right
    /// WorkerSet, calls deactivate() on its PrefillRouter, and the model is
    /// correctly hidden from model_display_names() when enforce_disagg=true.
    ///
    /// The WorkerSet is seeded with a router already in the "was activated, now
    /// deactivated" state (simulating a prefill engine that has just died).
    /// After calling deactivate_prefill_router_for_decode the model must remain
    /// hidden and the call must be idempotent.
    ///
    /// Guards the same `!ws.can_serve_requests()` check in `is_displayable()` as
    /// `test_is_displayable_false_when_prefill_dies_enforce_disagg`, but exercises
    /// the full ModelManager → Model → WorkerSet → PrefillRouter chain.
    /// Removing the guard causes this test to fail — `model_display_names()`
    /// returns the model even though prefill is dead and enforce_disagg is set.
    #[test]
    fn test_deactivate_prefill_router_for_decode_hides_model() {
        use crate::kv_router::PrefillRouter;

        let mm = ModelManager::new();

        // Build a WorkerSet whose prefill_router simulates "was active, now dead"
        // with enforce_disagg=true → can_serve_requests()=false → model hidden.
        let mut ws = make_worker_set("ns1", "abc");
        ws.prefill_router = Some(PrefillRouter::make_deactivated_for_test(true));
        mm.add_worker_set("llama", "ns1", ws).unwrap();

        // Model must already be absent (enforce_disagg + dead prefill).
        assert!(
            !mm.model_display_names().contains("llama"),
            "model with dead prefill+enforce_disagg must not appear in display names"
        );

        // Calling deactivate again must be idempotent and must not panic.
        mm.deactivate_prefill_router_for_decode("llama", "ns1");

        assert!(
            !mm.model_display_names().contains("llama"),
            "model must remain hidden after re-deactivation"
        );
    }

    // These exercise the full lifecycle that the watcher drives in production:
    //   decode registers → prefill registers → prefill dies → model hidden/visible
    /// Full disagg lifecycle with enforce_disagg=true.
    ///
    /// When the prefill engine dies and enforce_disagg is set, the model must be
    /// hidden from /v1/models after deactivation.
    ///
    /// Guards the `!ws.can_serve_requests()` check in `is_displayable()`.
    #[test]
    fn test_disagg_lifecycle_prefill_death_hides_model() {
        use crate::kv_router::PrefillRouter;

        let mm = ModelManager::new();

        // Step 1: Decode WorkerSet registers with an AwaitingPrefill router.
        // Model should be visible (decode is ready, prefill pending).
        let mut decode_ws = make_worker_set("decode-ns", "abc");
        decode_ws.prefill_router = Some(PrefillRouter::make_awaiting_for_test(true));
        mm.add_worker_set("llama", "decode-ns", decode_ws).unwrap();

        assert!(
            mm.model_display_names().contains("llama"),
            "step 1: model must be visible while awaiting prefill"
        );

        // Step 2: Prefill WorkerSet registers (same model, different namespace key).
        // Model should still be visible.
        let prefill_ws = make_worker_set("prefill-ns", "abc");
        mm.add_worker_set("llama", "prefill-ns", prefill_ws)
            .unwrap();

        assert!(
            mm.model_display_names().contains("llama"),
            "step 2: model must be visible with both decode and prefill"
        );

        // Step 3: Prefill WorkerSet removed (engine dies).
        // Model is still visible because the decode router hasn't been deactivated yet.
        mm.remove_worker_set("llama", "prefill-ns");

        assert!(
            mm.model_display_names().contains("llama"),
            "step 3: model must be visible before deactivation (router still AwaitingPrefill)"
        );

        // Step 4: Deactivate the prefill router on the decode side.
        // (In production, watcher::handle_delete does this.)
        // With enforce_disagg=true, the model must now be hidden.
        mm.deactivate_prefill_router_for_decode("llama", "decode-ns");

        assert!(
            !mm.model_display_names().contains("llama"),
            "step 4: model must be hidden after prefill death with enforce_disagg=true"
        );
    }

    /// Full disagg lifecycle with enforce_disagg=false (fallback allowed).
    ///
    /// When the prefill engine dies but enforce_disagg is not set, the model
    /// should remain visible — requests fall back to aggregated mode.
    #[test]
    fn test_disagg_lifecycle_prefill_death_keeps_model_no_enforce() {
        use crate::kv_router::PrefillRouter;

        let mm = ModelManager::new();

        // Step 1: Decode WorkerSet with enforce_disagg=false.
        let mut decode_ws = make_worker_set("decode-ns", "abc");
        decode_ws.prefill_router = Some(PrefillRouter::make_awaiting_for_test(false));
        mm.add_worker_set("llama", "decode-ns", decode_ws).unwrap();

        assert!(
            mm.model_display_names().contains("llama"),
            "step 1: model must be visible while awaiting prefill"
        );

        // Step 2: Prefill registers.
        let prefill_ws = make_worker_set("prefill-ns", "abc");
        mm.add_worker_set("llama", "prefill-ns", prefill_ws)
            .unwrap();

        assert!(
            mm.model_display_names().contains("llama"),
            "step 2: model must be visible with both decode and prefill"
        );

        // Step 3: Prefill removed.
        mm.remove_worker_set("llama", "prefill-ns");

        assert!(
            mm.model_display_names().contains("llama"),
            "step 3: model must be visible before deactivation"
        );

        // Step 4: Deactivate prefill router.
        // With enforce_disagg=false, model stays visible (fallback to aggregated).
        mm.deactivate_prefill_router_for_decode("llama", "decode-ns");

        assert!(
            mm.model_display_names().contains("llama"),
            "step 4: model must remain visible (enforce_disagg=false, fallback allowed)"
        );
    }

    /// Full disagg lifecycle including prefill rejoin after transient failure.
    ///
    /// Lifecycle:
    ///   decode registers → prefill dies → model hidden → prefill rejoins → model visible
    ///
    /// Verifies that deactivate_prefill_router_for_decode() hides the model and
    /// that a subsequent reactivation (simulated via state transition) restores it.
    /// Full reactivation through activate_prefill_router() requires a real Endpoint
    /// (runtime), so we simulate the synchronous reactivation state change.
    #[test]
    fn test_disagg_lifecycle_prefill_rejoin_restores_model() {
        use crate::kv_router::PrefillRouter;

        let mm = ModelManager::new();

        // Step 1: Decode with AwaitingPrefill router (enforce_disagg=true).
        let mut decode_ws = make_worker_set("decode-ns", "abc");
        decode_ws.prefill_router = Some(PrefillRouter::make_awaiting_for_test(true));
        mm.add_worker_set("llama", "decode-ns", decode_ws).unwrap();

        assert!(
            mm.model_display_names().contains("llama"),
            "step 1: model must be visible while awaiting prefill"
        );

        // Step 2: Prefill dies → deactivate.
        mm.deactivate_prefill_router_for_decode("llama", "decode-ns");

        assert!(
            !mm.model_display_names().contains("llama"),
            "step 2: model must be hidden after prefill death (enforce_disagg=true)"
        );

        // Step 3: Simulate prefill rejoin by transitioning the router back to
        // AwaitingPrefill.  In production, activate_prefill_router() calls
        // pr.reactivate(endpoint) which does this + spawns activation task.
        // Here we use simulate_reactivation() which doesn't require a real Endpoint.
        if let Some(model) = mm.get_model("llama")
            && let Some(ws) = model.get_worker_set("decode-ns")
            && let Some(ref pr) = ws.prefill_router
        {
            assert!(
                !pr.can_serve_requests(),
                "must be non-serving before rejoin"
            );
            pr.simulate_reactivation();
        } else {
            panic!("decode WorkerSet or prefill_router not found");
        }

        assert!(
            mm.model_display_names().contains("llama"),
            "step 3: model must be visible again after prefill rejoin"
        );
    }
}
