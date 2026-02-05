// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dashmap::{DashMap, mapref::entry::Entry};
use parking_lot::RwLock;
use tokio::sync::oneshot;

use super::worker_monitor::LoadThresholdConfig;
use super::{KvWorkerMonitor, RuntimeConfigs};

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
    model_type::ModelType,
    types::{
        generic::tensor::TensorStreamingEngine,
        openai::{
            chat_completions::OpenAIChatCompletionsStreamingEngine,
            completions::OpenAICompletionsStreamingEngine,
            embeddings::OpenAIEmbeddingsStreamingEngine, images::OpenAIImagesStreamingEngine,
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
/// Manages model lifecycle including engines, KV routers, prefill coordination,
/// and per-model busy thresholds for load-based request rejection.
///
/// Note: Don't implement Clone for this, put it in an Arc instead.
pub struct ModelManager {
    // We read a lot and write rarely, so these three are RwLock
    completion_engines: RwLock<ModelEngines<OpenAICompletionsStreamingEngine>>,
    chat_completion_engines: RwLock<ModelEngines<OpenAIChatCompletionsStreamingEngine>>,
    embeddings_engines: RwLock<ModelEngines<OpenAIEmbeddingsStreamingEngine>>,
    images_engines: RwLock<ModelEngines<OpenAIImagesStreamingEngine>>,
    tensor_engines: RwLock<ModelEngines<TensorStreamingEngine>>,
    // Prefill models don't have engines - they're only tracked for discovery/lifecycle
    prefill_engines: RwLock<ModelEngines<()>>,

    cards: DashMap<String, ModelDeploymentCard>,
    kv_choosers: DashMap<EndpointId, Arc<KvRouter>>,
    prefill_router_activators: DashMap<String, PrefillActivationState>,

    // Per-model monitoring: worker_monitors for load-based rejection, runtime_configs for KvScheduler
    worker_monitors: DashMap<String, KvWorkerMonitor>,
    runtime_configs: DashMap<EndpointId, Arc<RuntimeConfigs>>,
}

impl Default for ModelManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            completion_engines: RwLock::new(ModelEngines::default()),
            chat_completion_engines: RwLock::new(ModelEngines::default()),
            embeddings_engines: RwLock::new(ModelEngines::default()),
            images_engines: RwLock::new(ModelEngines::default()),
            tensor_engines: RwLock::new(ModelEngines::default()),
            prefill_engines: RwLock::new(ModelEngines::default()),
            cards: DashMap::new(),
            kv_choosers: DashMap::new(),
            prefill_router_activators: DashMap::new(),
            worker_monitors: DashMap::new(),
            runtime_configs: DashMap::new(),
        }
    }

    pub fn is_valid_checksum(
        &self,
        model_type: ModelType,
        model_name: &str,
        candidate_checksum: &str,
    ) -> Option<bool> {
        let mut results = vec![];
        for unit in model_type.units() {
            let maybe_valid_checksum = match unit {
                ModelType::Chat => self.chat_completion_engines.read().checksum(model_name),
                ModelType::Completions => self.completion_engines.read().checksum(model_name),
                ModelType::Embedding => self.embeddings_engines.read().checksum(model_name),
                ModelType::TensorBased => self.tensor_engines.read().checksum(model_name),
                ModelType::Images => self.images_engines.read().checksum(model_name),
                ModelType::Prefill => self.prefill_engines.read().checksum(model_name),
                _ => {
                    continue;
                }
            };
            if let Some(is_valid) = maybe_valid_checksum.map(|valid_checksum| {
                tracing::debug!(
                    model_name,
                    valid_checksum,
                    candidate_checksum,
                    "is_valid_checksum: check case"
                );
                valid_checksum == candidate_checksum
            }) {
                results.push(is_valid)
            }
        }
        if results.is_empty() {
            None
        } else {
            // The checksum is valid if it is correct for all the ModelType in the bitflag.
            Some(results.into_iter().all(|x| x))
        }
    }

    pub fn get_model_cards(&self) -> Vec<ModelDeploymentCard> {
        self.cards.iter().map(|r| r.value().clone()).collect()
    }

    /// Check if a decode model (chat or completions) is registered
    pub fn has_decode_model(&self, model: &str) -> bool {
        self.chat_completion_engines.read().contains(model)
            || self.completion_engines.read().contains(model)
    }

    /// Check if a prefill model is registered
    pub fn has_prefill_model(&self, model: &str) -> bool {
        self.prefill_engines.read().contains(model)
    }

    /// Check if any model (decode or prefill) is registered.
    /// Note: For registration skip-checks, use has_decode_model() or has_prefill_model() instead.
    pub fn has_model_any(&self, model: &str) -> bool {
        self.has_decode_model(model) || self.has_prefill_model(model)
    }

    pub fn model_display_names(&self) -> HashSet<String> {
        self.list_chat_completions_models()
            .into_iter()
            .chain(self.list_completions_models())
            .chain(self.list_embeddings_models())
            .chain(self.list_tensor_models())
            .chain(self.list_prefill_models())
            .collect()
    }

    pub fn list_chat_completions_models(&self) -> Vec<String> {
        self.chat_completion_engines.read().list()
    }

    pub fn list_completions_models(&self) -> Vec<String> {
        self.completion_engines.read().list()
    }

    pub fn list_embeddings_models(&self) -> Vec<String> {
        self.embeddings_engines.read().list()
    }

    pub fn list_tensor_models(&self) -> Vec<String> {
        self.tensor_engines.read().list()
    }

    pub fn list_prefill_models(&self) -> Vec<String> {
        self.prefill_engines.read().list()
    }

    pub fn add_completions_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAICompletionsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self.completion_engines.write();
        clients.add(model, card_checksum, engine)
    }

    pub fn add_chat_completions_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIChatCompletionsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self.chat_completion_engines.write();
        clients.add(model, card_checksum, engine)
    }

    pub fn add_embeddings_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIEmbeddingsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self.embeddings_engines.write();
        clients.add(model, card_checksum, engine)
    }

    pub fn add_tensor_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: TensorStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self.tensor_engines.write();
        clients.add(model, card_checksum, engine)
    }

    pub fn add_images_model(
        &self,
        model: &str,
        card_checksum: &str,
        engine: OpenAIImagesStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self.images_engines.write();
        clients.add(model, card_checksum, engine)
    }

    pub fn add_prefill_model(
        &self,
        model: &str,
        card_checksum: &str,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self.prefill_engines.write();
        clients.add(model, card_checksum, ())
    }

    pub fn remove_completions_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self.completion_engines.write();
        clients.remove(model)
    }

    pub fn remove_chat_completions_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self.chat_completion_engines.write();
        clients.remove(model)
    }

    pub fn remove_embeddings_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self.embeddings_engines.write();
        clients.remove(model)
    }

    pub fn remove_tensor_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self.tensor_engines.write();
        clients.remove(model)
    }

    pub fn remove_images_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self.images_engines.write();
        clients.remove(model)
    }

    pub fn remove_prefill_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self.prefill_engines.write();
        clients.remove(model)
    }

    pub fn get_embeddings_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIEmbeddingsStreamingEngine, ModelManagerError> {
        self.embeddings_engines
            .read()
            .get(model)
            .cloned()
            .ok_or(ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn get_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAICompletionsStreamingEngine, ModelManagerError> {
        self.completion_engines
            .read()
            .get(model)
            .cloned()
            .ok_or(ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn get_chat_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIChatCompletionsStreamingEngine, ModelManagerError> {
        self.chat_completion_engines
            .read()
            .get(model)
            .cloned()
            .ok_or(ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn get_tensor_engine(
        &self,
        model: &str,
    ) -> Result<TensorStreamingEngine, ModelManagerError> {
        self.tensor_engines
            .read()
            .get(model)
            .cloned()
            .ok_or(ModelManagerError::ModelNotFound(model.to_string()))
    }

    pub fn get_images_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIImagesStreamingEngine, ModelManagerError> {
        self.images_engines
            .read()
            .get(model)
            .cloned()
            .ok_or(ModelManagerError::ModelNotFound(model.to_string()))
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

    pub async fn kv_chooser_for(
        &self,
        endpoint: &Endpoint,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
        worker_type: &'static str,
    ) -> anyhow::Result<Arc<KvRouter>> {
        let endpoint_id = endpoint.id();

        if let Some(kv_chooser) = self.get_kv_chooser(&endpoint_id) {
            // Check if the existing router has a different block size
            if kv_chooser.block_size() != kv_cache_block_size {
                tracing::warn!(
                    endpoint = %endpoint_id,
                    existing_block_size = %kv_chooser.block_size(),
                    requested_block_size = %kv_cache_block_size,
                    "KV Router block size mismatch! Endpoint is requesting a different kv_cache_block_size than the existing router. \
                     This will cause routing to fail silently. Consider using the same block size or restarting the router."
                );
            }
            return Ok(kv_chooser);
        }

        let client = endpoint.client().await?;

        // Register router via discovery mechanism
        let discovery = endpoint.component().drt().discovery();
        let instance_id = discovery.instance_id();

        // Build transport for router endpoint based on request plane mode
        // Use KV_ROUTER_COMPONENT as the component name to distinguish from the generate endpoint's component
        let router_endpoint_id = router_endpoint_id(endpoint.id().namespace);
        let transport = build_transport_type(endpoint, &router_endpoint_id, instance_id).await?;

        let discovery_spec = DiscoverySpec::Endpoint {
            namespace: router_endpoint_id.namespace.clone(),
            component: router_endpoint_id.component.clone(),
            endpoint: router_endpoint_id.name.clone(),
            transport,
        };

        discovery.register(discovery_spec).await?;

        // Get or create runtime config watcher for this endpoint
        let workers_with_configs = self.get_or_create_runtime_config_watcher(endpoint).await?;

        let selector = Box::new(DefaultWorkerSelector::new(kv_router_config));
        let chooser = KvRouter::new(
            endpoint.clone(),
            client,
            workers_with_configs,
            kv_cache_block_size,
            Some(selector),
            kv_router_config,
            instance_id,
            worker_type,
        )
        .await?;
        let new_kv_chooser = Arc::new(chooser);
        self.kv_choosers.insert(endpoint_id, new_kv_chooser.clone());
        Ok(new_kv_chooser)
    }

    fn get_kv_chooser(&self, id: &EndpointId) -> Option<Arc<KvRouter>> {
        self.kv_choosers.get(id).map(|r| r.value().clone())
    }

    /// Register a prefill router for a decode model. Returns a receiver that will be
    /// activated when the corresponding prefill model is discovered.
    /// Returns None if the decode model was already registered.
    pub fn register_prefill_router(
        &self,
        model_name: String,
    ) -> Option<oneshot::Receiver<Endpoint>> {
        match self.prefill_router_activators.remove(&model_name) {
            Some((_, PrefillActivationState::PrefillReady(rx))) => {
                // Prefill endpoint already arrived - rx will immediately resolve
                tracing::debug!(
                    model_name = %model_name,
                    "Prefill endpoint already available, returning receiver with endpoint"
                );
                Some(rx)
            }
            Some((key, PrefillActivationState::DecodeWaiting(tx))) => {
                // Decode already registered - this shouldn't happen, restore state and return None
                tracing::error!(
                    model_name = %model_name,
                    "Decode model already registered for this prefill router"
                );
                self.prefill_router_activators
                    .insert(key, PrefillActivationState::DecodeWaiting(tx));
                None
            }
            None => {
                // New registration: create tx/rx pair, store sender and return receiver
                let (tx, rx) = oneshot::channel();
                self.prefill_router_activators.insert(
                    model_name.clone(),
                    PrefillActivationState::DecodeWaiting(tx),
                );
                tracing::debug!(
                    model_name = %model_name,
                    "No prefill endpoint available yet, storing sender for future activation"
                );
                Some(rx)
            }
        }
    }

    /// Activate a prefill router by sending the endpoint through the oneshot channel.
    /// If no decode model has registered yet, stores the endpoint for future retrieval.
    pub fn activate_prefill_router(
        &self,
        model_name: &str,
        endpoint: Endpoint,
    ) -> anyhow::Result<()> {
        match self.prefill_router_activators.remove(model_name) {
            Some((_, PrefillActivationState::DecodeWaiting(sender))) => {
                // Decode model already registered
                sender.send(endpoint).map_err(|_| {
                    anyhow::anyhow!(
                        "Failed to send endpoint to prefill router activator for model: {}",
                        model_name
                    )
                })?;

                tracing::info!(
                    model_name = %model_name,
                    "Activated prefill router for already-registered decode model"
                );

                Ok(())
            }
            Some((_, PrefillActivationState::PrefillReady(_))) => {
                // Prefill already activated - this shouldn't happen
                anyhow::bail!("Prefill router for model {} already activated", model_name);
            }
            None => {
                // Decode model not registered yet - create pair and immediately send endpoint
                let (tx, rx) = oneshot::channel();

                tx.send(endpoint).map_err(|_| {
                    anyhow::anyhow!("Failed to send endpoint for prefill model: {}", model_name)
                })?;

                // Store the receiver for when decode model registers
                self.prefill_router_activators.insert(
                    model_name.to_string(),
                    PrefillActivationState::PrefillReady(rx),
                );

                tracing::info!(
                    model_name = %model_name,
                    "Stored prefill endpoint for future decode model registration"
                );

                Ok(())
            }
        }
    }

    pub fn get_model_tool_call_parser(&self, model: &str) -> Option<String> {
        self.cards
            .iter()
            .find(|r| r.value().display_name == model)
            .and_then(|r| r.value().runtime_config.tool_call_parser.clone())
    }

    /// Creates parsing options with tool call parser and reasoning parser for the specified model.
    /// Currently reasoning parser is not implemented (returns None).
    pub fn get_parsing_options(&self, model: &str) -> crate::protocols::openai::ParsingOptions {
        let tool_call_parser = self.get_model_tool_call_parser(model);
        let reasoning_parser = None; // TODO: Implement reasoning parser

        crate::protocols::openai::ParsingOptions::new(tool_call_parser, reasoning_parser)
    }

    /// Gets or sets the load threshold config for a model's worker monitor.
    /// Pass `Some(config)` to update, `None` to get. Returns `None` if no monitor exists.
    pub fn load_threshold_config(
        &self,
        model: &str,
        config: Option<&LoadThresholdConfig>,
    ) -> Option<LoadThresholdConfig> {
        let monitor = self.worker_monitors.get(model)?;
        if let Some(cfg) = config {
            monitor.set_load_threshold_config(cfg);
        }
        Some(monitor.load_threshold_config())
    }

    /// Gets an existing worker monitor for a model, if one exists.
    pub fn get_worker_monitor(&self, model: &str) -> Option<KvWorkerMonitor> {
        self.worker_monitors.get(model).map(|m| m.clone())
    }

    /// Gets or creates a worker monitor for a model. Updates thresholds if monitor exists.
    pub fn get_or_create_worker_monitor(
        &self,
        model: &str,
        client: Client,
        config: LoadThresholdConfig,
    ) -> KvWorkerMonitor {
        if let Some(existing) = self.worker_monitors.get(model) {
            existing.set_load_threshold_config(&config);
            return existing.clone();
        }
        let monitor = KvWorkerMonitor::new(client, config);
        self.worker_monitors
            .insert(model.to_string(), monitor.clone());
        monitor
    }

    /// Get or create a runtime config watcher for an endpoint.
    /// Spawns a background task to watch for worker config changes.
    /// Returns a shared RuntimeConfigs that KvScheduler can use directly.
    pub async fn get_or_create_runtime_config_watcher(
        &self,
        endpoint: &Endpoint,
    ) -> anyhow::Result<Arc<RuntimeConfigs>> {
        let endpoint_id = endpoint.id();

        // Fast path: return existing if present
        if let Some(existing) = self.runtime_configs.get(&endpoint_id) {
            return Ok(existing.clone());
        }

        // Atomic get-or-insert to avoid TOCTOU race
        let inner = Arc::new(RuntimeConfigs::new());
        let (result, is_new) = match self.runtime_configs.entry(endpoint_id) {
            Entry::Occupied(e) => (e.get().clone(), false),
            Entry::Vacant(e) => {
                e.insert(inner.clone());
                (inner, true)
            }
        };

        // Only spawn watcher if we were the one who inserted
        if is_new {
            result.start_watcher(endpoint).await?;
        }

        Ok(result)
    }

    /// Get disaggregated endpoint for a specific worker.
    /// Used by PrefillRouter for bootstrap info - works for ANY routing mode.
    pub fn get_disaggregated_endpoint(
        &self,
        endpoint_id: &EndpointId,
        worker_id: WorkerId,
    ) -> Option<DisaggregatedEndpoint> {
        let inner = self.runtime_configs.get(endpoint_id)?;
        let config_ref = inner.configs.get(&worker_id)?;
        config_ref.as_ref()?.disaggregated_endpoint.clone()
    }

    /// Lists all models with worker monitors configured.
    pub fn list_busy_thresholds(&self) -> Vec<(String, LoadThresholdConfig)> {
        self.worker_monitors
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().load_threshold_config()))
            .collect()
    }
}

pub struct ModelEngines<E> {
    /// Optional default model name
    default: Option<String>,
    engines: HashMap<String, E>,
    /// Key: Model name, value: Checksum of the ModelDeploymentCard. New instances must have the
    /// same card.
    checksums: HashMap<String, String>,
}

impl<E> Default for ModelEngines<E> {
    fn default() -> Self {
        Self {
            default: None,
            engines: HashMap::new(),
            checksums: HashMap::new(),
        }
    }
}

impl<E> ModelEngines<E> {
    #[allow(dead_code)]
    fn set_default(&mut self, model: &str) {
        self.default = Some(model.to_string());
    }

    #[allow(dead_code)]
    fn clear_default(&mut self) {
        self.default = None;
    }

    fn add(&mut self, model: &str, checksum: &str, engine: E) -> Result<(), ModelManagerError> {
        if self.engines.contains_key(model) {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        self.engines.insert(model.to_string(), engine);
        self.checksums
            .insert(model.to_string(), checksum.to_string());
        Ok(())
    }

    fn remove(&mut self, model: &str) -> Result<(), ModelManagerError> {
        if self.engines.remove(model).is_none() {
            return Err(ModelManagerError::ModelNotFound(model.to_string()));
        }
        let _ = self.checksums.remove(model);
        Ok(())
    }

    fn get(&self, model: &str) -> Option<&E> {
        self.engines.get(model)
    }

    fn contains(&self, model: &str) -> bool {
        self.engines.contains_key(model)
    }

    pub fn list(&self) -> Vec<String> {
        self.engines.keys().map(|k| k.to_owned()).collect()
    }

    /// Returns a newly allocated String for called convenience. All the places I use
    /// this I need a String.
    pub fn checksum(&self, model: &str) -> Option<String> {
        self.checksums.get(model).map(|s| s.to_string())
    }
}
