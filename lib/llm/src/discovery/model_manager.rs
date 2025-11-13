// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use parking_lot::{Mutex, RwLock};
use tokio::sync::oneshot;

use dynamo_runtime::prelude::DistributedRuntimeProvider;
use dynamo_runtime::{component::Endpoint, storage::key_value_store::Key};

use crate::{
    discovery::KV_ROUTERS_ROOT_PATH,
    kv_router::{KvRouter, KvRouterConfig, scheduler::DefaultWorkerSelector},
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

// Don't implement Clone for this, put it in an Arc instead.
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
    kv_choosers: Mutex<HashMap<String, Arc<KvRouter>>>, // Key: component service_name
    prefill_router_activators: Mutex<HashMap<String, PrefillActivationState>>,
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

    pub fn has_model_any(&self, model: &str) -> bool {
        self.chat_completion_engines.read().contains(model)
            || self.completion_engines.read().contains(model)
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
        let endpoint_path = endpoint.path();

        if let Some(kv_chooser) = self.get_kv_chooser(&endpoint_path) {
            // Check if the existing router has a different block size
            if kv_chooser.block_size() != kv_cache_block_size {
                tracing::warn!(
                    endpoint = %endpoint_path,
                    existing_block_size = %kv_chooser.block_size(),
                    requested_block_size = %kv_cache_block_size,
                    "KV Router block size mismatch! Endpoint is requesting a different kv_cache_block_size than the existing router. \
                     This will cause routing to fail silently. Consider using the same block size or restarting the router."
                );
            }
            return Ok(kv_chooser);
        }

        let client = endpoint.client().await?;
        let store = endpoint.component().drt().store();
        let router_bucket = store
            .get_or_create_bucket(KV_ROUTERS_ROOT_PATH, None)
            .await?;
        let router_uuid = uuid::Uuid::new_v4();
        let router_key = Key::from_raw(format!("{}/{router_uuid}", endpoint.path()));
        let json_router_config = serde_json::to_vec_pretty(&kv_router_config.unwrap_or_default())?;
        router_bucket
            .insert(&router_key, json_router_config.into(), 0)
            .await?;

        let selector = Box::new(DefaultWorkerSelector::new(kv_router_config));
        let chooser = KvRouter::new(
            endpoint.clone(),
            client,
            kv_cache_block_size,
            Some(selector),
            kv_router_config,
            router_uuid.to_string(),
        )
        .await?;
        let new_kv_chooser = Arc::new(chooser);
        self.kv_choosers
            .lock()
            .insert(endpoint_path, new_kv_chooser.clone());
        Ok(new_kv_chooser)
    }

    fn get_kv_chooser(&self, service_name: &str) -> Option<Arc<KvRouter>> {
        self.kv_choosers.lock().get(service_name).cloned()
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
