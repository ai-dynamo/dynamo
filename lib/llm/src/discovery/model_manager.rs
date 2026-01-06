// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use dashmap::{DashMap, mapref::entry::Entry};
use parking_lot::{Mutex, RwLock};
use tokio::sync::oneshot;

use crate::discovery::KvWorkerMonitor;

use dynamo_runtime::{
    component::{Client, Endpoint, build_transport_type},
    discovery::{DiscoveryQuery, DiscoverySpec, watch_and_extract_field},
    prelude::DistributedRuntimeProvider,
    protocols::EndpointId,
};

use crate::{
    kv_router::{
        KvRouter, KvRouterConfig, protocols::WorkerId, router_endpoint_id,
        scheduler::DefaultWorkerSelector,
    },
    local_model::runtime_config::{DisaggregatedEndpoint, ModelRuntimeConfig},
    model_card::ModelDeploymentCard,
    model_type::ModelType,
    types::{
        generic::tensor::TensorStreamingEngine,
        openai::{
            chat_completions::OpenAIChatCompletionsStreamingEngine,
            completions::OpenAICompletionsStreamingEngine,
            embeddings::OpenAIEmbeddingsStreamingEngine,
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
    tensor_engines: RwLock<ModelEngines<TensorStreamingEngine>>,
    // Prefill models don't have engines - they're only tracked for discovery/lifecycle
    prefill_engines: RwLock<ModelEngines<()>>,

    // These are Mutex because we read and write rarely and equally
    cards: Mutex<HashMap<String, ModelDeploymentCard>>,
    kv_choosers: Mutex<HashMap<EndpointId, Arc<KvRouter>>>,
    prefill_router_activators: Mutex<HashMap<String, PrefillActivationState>>,

    /// Per-model worker monitors for dynamic KV cache load rejection.
    /// Key: model name, Value: cloneable monitor (all fields are Arc).
    /// HTTP endpoint can update thresholds via monitor.set_threshold().
    worker_monitors: RwLock<HashMap<String, KvWorkerMonitor>>,

    /// Runtime configs per endpoint using DashMap for lock-free access.
    /// Outer DashMap: keyed by EndpointId
    /// Inner Arc<DashMap>: keyed by WorkerId, shared with KvScheduler
    runtime_configs: DashMap<EndpointId, Arc<DashMap<WorkerId, Option<ModelRuntimeConfig>>>>,
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
            tensor_engines: RwLock::new(ModelEngines::default()),
            prefill_engines: RwLock::new(ModelEngines::default()),
            cards: Mutex::new(HashMap::new()),
            kv_choosers: Mutex::new(HashMap::new()),
            prefill_router_activators: Mutex::new(HashMap::new()),
            worker_monitors: RwLock::new(HashMap::new()),
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
        self.cards.lock().values().cloned().collect()
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

    /// Save a ModelDeploymentCard from an instance's ModelDeploymentCard key so we can fetch it later when the key is
    /// deleted.
    pub fn save_model_card(&self, key: &str, card: ModelDeploymentCard) -> anyhow::Result<()> {
        self.cards.lock().insert(key.to_string(), card);
        Ok(())
    }

    /// Remove and return model card for this instance's etcd key. We do this when the instance stops.
    pub fn remove_model_card(&self, key: &str) -> Option<ModelDeploymentCard> {
        self.cards.lock().remove(key)
    }

    pub async fn kv_chooser_for(
        &self,
        endpoint: &Endpoint,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
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

        // Use instance_id (hex) as the consumer ID for NATS consumer coordination
        let consumer_id = instance_id.to_string();

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
            consumer_id,
        )
        .await?;
        let new_kv_chooser = Arc::new(chooser);
        self.kv_choosers
            .lock()
            .insert(endpoint_id, new_kv_chooser.clone());
        Ok(new_kv_chooser)
    }

    fn get_kv_chooser(&self, id: &EndpointId) -> Option<Arc<KvRouter>> {
        self.kv_choosers.lock().get(id).cloned()
    }

    /// Register a prefill router for a decode model. Returns a receiver that will be
    /// activated when the corresponding prefill model is discovered.
    /// Returns None if the decode model was already registered.
    pub fn register_prefill_router(
        &self,
        model_name: String,
    ) -> Option<oneshot::Receiver<Endpoint>> {
        let mut activators = self.prefill_router_activators.lock();

        match activators.remove(&model_name) {
            Some(PrefillActivationState::PrefillReady(rx)) => {
                // Prefill endpoint already arrived - rx will immediately resolve
                tracing::debug!(
                    model_name = %model_name,
                    "Prefill endpoint already available, returning receiver with endpoint"
                );
                Some(rx)
            }
            Some(PrefillActivationState::DecodeWaiting(tx)) => {
                // Decode already registered - this shouldn't happen, restore state and return None
                tracing::error!(
                    model_name = %model_name,
                    "Decode model already registered for this prefill router"
                );
                activators.insert(model_name, PrefillActivationState::DecodeWaiting(tx));
                None
            }
            None => {
                // New registration: create tx/rx pair, store sender and return receiver
                let (tx, rx) = oneshot::channel();
                activators.insert(
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
        let mut activators = self.prefill_router_activators.lock();

        match activators.remove(model_name) {
            Some(PrefillActivationState::DecodeWaiting(sender)) => {
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
            Some(PrefillActivationState::PrefillReady(_)) => {
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
                activators.insert(
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
            .lock()
            .values()
            .find(|c| c.display_name == model)
            .and_then(|c| c.runtime_config.tool_call_parser.as_ref())
            .map(|parser| parser.to_string())
    }

    /// Creates parsing options with tool call parser and reasoning parser for the specified model.
    /// Currently reasoning parser is not implemented (returns None).
    pub fn get_parsing_options(&self, model: &str) -> crate::protocols::openai::ParsingOptions {
        let tool_call_parser = self.get_model_tool_call_parser(model);
        let reasoning_parser = None; // TODO: Implement reasoning parser

        crate::protocols::openai::ParsingOptions::new(tool_call_parser, reasoning_parser)
    }

    /// Gets or sets the busy threshold for a model via its worker monitor.
    ///
    /// Get or set the active decode blocks threshold for a model's worker monitor.
    ///
    /// This is the primary API for HTTP endpoints and external callers.
    /// The threshold (0.0 to 1.0) controls when workers are marked as "busy"
    /// based on KV cache block utilization.
    ///
    /// # Arguments
    ///
    /// * `model` - The model name
    /// * `threshold` - `Some(value)` to set, `None` to get existing
    ///
    /// # Returns
    ///
    /// The threshold value as f64, or `None` if no monitor exists for this model.
    pub fn active_decode_blocks_threshold(
        &self,
        model: &str,
        threshold: Option<f64>,
    ) -> Option<f64> {
        let monitors = self.worker_monitors.read();
        let monitor = monitors.get(model)?;

        match threshold {
            Some(value) => {
                monitor.set_active_decode_blocks_threshold(value);
                Some(value)
            }
            None => Some(monitor.active_decode_blocks_threshold()),
        }
    }

    /// Get or set the active prefill tokens threshold for a model's worker monitor.
    ///
    /// The threshold is a literal token count (not a percentage).
    ///
    /// # Arguments
    ///
    /// * `model` - The model name
    /// * `threshold` - `Some(value)` to set, `None` to get existing
    ///
    /// # Returns
    ///
    /// The threshold value as u64, or `None` if no monitor exists for this model.
    pub fn active_prefill_tokens_threshold(
        &self,
        model: &str,
        threshold: Option<u64>,
    ) -> Option<u64> {
        let monitors = self.worker_monitors.read();
        let monitor = monitors.get(model)?;

        match threshold {
            Some(value) => {
                monitor.set_active_prefill_tokens_threshold(value);
                Some(value)
            }
            None => Some(monitor.active_prefill_tokens_threshold()),
        }
    }

    /// Gets or creates a worker monitor for a model.
    ///
    /// If a monitor already exists, updates its thresholds and returns a clone.
    /// If no monitor exists, creates one with the given client and thresholds.
    ///
    /// # Arguments
    ///
    /// * `model` - The model name
    /// * `client` - The client for subscribing to KV metrics (only used if creating new)
    /// * `active_decode_blocks_threshold` - The initial/updated active decode blocks threshold value (0.0-1.0)
    /// * `active_prefill_tokens_threshold` - The initial/updated active prefill tokens threshold value (literal token count)
    ///
    /// # Returns
    ///
    /// A cloneable monitor that shares state with the stored instance.
    pub fn get_or_create_worker_monitor(
        &self,
        model: &str,
        client: Client,
        active_decode_blocks_threshold: f64,
        active_prefill_tokens_threshold: u64,
    ) -> KvWorkerMonitor {
        let mut monitors = self.worker_monitors.write();

        if let Some(existing) = monitors.get(model) {
            existing.set_active_decode_blocks_threshold(active_decode_blocks_threshold);
            existing.set_active_prefill_tokens_threshold(active_prefill_tokens_threshold);
            existing.clone()
        } else {
            let monitor = KvWorkerMonitor::new(
                client,
                active_decode_blocks_threshold,
                active_prefill_tokens_threshold,
            );
            monitors.insert(model.to_string(), monitor.clone());
            monitor
        }
    }

    /// Gets an existing worker monitor for a model, if one exists.
    pub fn get_worker_monitor(&self, model: &str) -> Option<KvWorkerMonitor> {
        self.worker_monitors.read().get(model).cloned()
    }

    /// Get or create a runtime config watcher for an endpoint.
    /// Spawns a background task to watch DiscoveryQuery::EndpointModels.
    /// Returns a shared Arc<DashMap> that KvScheduler can use directly.
    pub async fn get_or_create_runtime_config_watcher(
        &self,
        endpoint: &Endpoint,
    ) -> anyhow::Result<Arc<DashMap<WorkerId, Option<ModelRuntimeConfig>>>> {
        let endpoint_id = endpoint.id();

        // Fast path: return existing if present
        if let Some(existing) = self.runtime_configs.get(&endpoint_id) {
            return Ok(existing.clone());
        }

        // Atomic get-or-insert to avoid TOCTOU race
        let inner_map = Arc::new(DashMap::new());
        let (map, is_new) = match self.runtime_configs.entry(endpoint_id) {
            Entry::Occupied(e) => (e.get().clone(), false),
            Entry::Vacant(e) => {
                e.insert(inner_map.clone());
                (inner_map, true)
            }
        };

        // Only spawn watcher if we were the one who inserted
        if is_new {
            self.spawn_runtime_config_watcher(endpoint, map.clone())
                .await?;
        }

        Ok(map)
    }

    /// Get disaggregated endpoint for a specific worker.
    /// Used by PrefillRouter for bootstrap info - works for ANY routing mode.
    pub fn get_disaggregated_endpoint(
        &self,
        endpoint_id: &EndpointId,
        worker_id: WorkerId,
    ) -> Option<DisaggregatedEndpoint> {
        let inner_map = self.runtime_configs.get(endpoint_id)?;
        let config_ref = inner_map.get(&worker_id)?;
        config_ref.as_ref()?.disaggregated_endpoint.clone()
    }

    /// Spawn background task to watch runtime configs via discovery.
    async fn spawn_runtime_config_watcher(
        &self,
        endpoint: &Endpoint,
        inner_map: Arc<DashMap<WorkerId, Option<ModelRuntimeConfig>>>,
    ) -> anyhow::Result<()> {
        let component = endpoint.component();
        let cancellation_token = component.drt().primary_token();

        // Set up discovery watch for EndpointModels
        let discovery = component.drt().discovery();
        let endpoint_id = endpoint.id();
        let discovery_key = DiscoveryQuery::EndpointModels {
            namespace: endpoint_id.namespace.clone(),
            component: endpoint_id.component.clone(),
            endpoint: endpoint_id.name.clone(),
        };
        let discovery_stream = discovery
            .list_and_watch(discovery_key.clone(), Some(cancellation_token.clone()))
            .await?;

        // Extract runtime_config from ModelDeploymentCard
        let mut runtime_configs_rx =
            watch_and_extract_field(discovery_stream, |card: ModelDeploymentCard| {
                card.runtime_config
            });

        // Also watch instance IDs
        let client = endpoint.client().await?;
        let mut instance_ids_rx = client.instance_avail_watcher();

        // Spawn background task to update inner_map
        let cancel_token = cancellation_token.clone();
        tokio::spawn(async move {
            tracing::trace!("ModelManager runtime config watcher started");
            loop {
                // Wait for either instances or configs to change
                tokio::select! {
                    _ = cancel_token.cancelled() => {
                        tracing::trace!("ModelManager runtime config watcher shutting down");
                        break;
                    }
                    result = instance_ids_rx.changed() => {
                        if result.is_err() {
                            tracing::warn!("instance IDs watch sender shutdown in ModelManager");
                            break;
                        }
                    }
                    result = runtime_configs_rx.changed() => {
                        if result.is_err() {
                            tracing::warn!("runtime configs watch sender shutdown in ModelManager");
                            break;
                        }
                    }
                }

                // Get the latest values from both channels
                let new_instance_ids = instance_ids_rx.borrow_and_update().clone();
                let new_configs = runtime_configs_rx.borrow_and_update().clone();

                // Update the DashMap
                // First, remove workers that no longer exist
                let current_workers: HashSet<WorkerId> =
                    inner_map.iter().map(|r| *r.key()).collect();
                let new_workers: HashSet<WorkerId> = new_instance_ids.iter().copied().collect();
                for removed_worker in current_workers.difference(&new_workers) {
                    inner_map.remove(removed_worker);
                }

                // Then, add/update workers
                for worker_id in &new_instance_ids {
                    let config = new_configs.get(worker_id).cloned();
                    if config.is_some() {
                        let prev_config = inner_map.get(worker_id);
                        if prev_config.as_ref().map(|r| r.value()) != Some(&config) {
                            tracing::info!(
                                "ModelManager: Runtime config found for worker_id: {}",
                                worker_id
                            );
                        }
                    }
                    inner_map.insert(*worker_id, config);
                }

                tracing::trace!(
                    "ModelManager: Updated runtime_configs with {} workers",
                    inner_map.len()
                );
            }
            tracing::trace!("ModelManager runtime config watcher shutting down");
        });

        Ok(())
    }

    /// Lists all models that have worker monitors (and thus busy thresholds) configured.
    ///
    /// Returns a vector of (model_name, active_decode_blocks_threshold, active_prefill_tokens_threshold) tuples.
    pub fn list_busy_thresholds(&self) -> Vec<(String, f64, u64)> {
        self.worker_monitors
            .read()
            .iter()
            .map(|(k, monitor)| {
                (
                    k.clone(),
                    monitor.active_decode_blocks_threshold(),
                    monitor.active_prefill_tokens_threshold(),
                )
            })
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
