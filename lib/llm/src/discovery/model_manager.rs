// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, RwLock},
};

use anyhow::Context;
use dynamo_runtime::{component::Component, prelude::DistributedRuntimeProvider, slug::Slug};

use crate::{
    discovery::ModelEntry,
    kv_router::{KvRouter, KvRouterConfig, scheduler::DefaultWorkerSelector},
    types::openai::{
        chat_completions::OpenAIChatCompletionsStreamingEngine,
        completions::OpenAICompletionsStreamingEngine, embeddings::OpenAIEmbeddingsStreamingEngine,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum ModelManagerError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model already exists: {0}")]
    ModelAlreadyExists(String),

    #[error("Lock poisoned: {0}")]
    LockPoisoned(&'static str),
}

// Don't implement Clone for this, put it in an Arc instead.
pub struct ModelManager {
    // We read a lot and write rarely, so these three are RwLock
    completion_engines: RwLock<ModelEngines<OpenAICompletionsStreamingEngine>>,
    chat_completion_engines: RwLock<ModelEngines<OpenAIChatCompletionsStreamingEngine>>,
    embeddings_engines: RwLock<ModelEngines<OpenAIEmbeddingsStreamingEngine>>,

    // These two are Mutex because we read and write rarely and equally
    entries: Mutex<HashMap<String, ModelEntry>>,
    kv_choosers: Mutex<HashMap<String, Arc<KvRouter>>>,
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
            entries: Mutex::new(HashMap::new()),
            kv_choosers: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_model_entries(&self) -> Result<Vec<ModelEntry>, ModelManagerError> {
        let guard = self
            .entries
            .lock()
            .map_err(|_| ModelManagerError::LockPoisoned("entries"))?;
        Ok(guard.values().cloned().collect())
    }

    pub fn has_model_any(&self, model: &str) -> Result<bool, ModelManagerError> {
        let chat = self
            .chat_completion_engines
            .read()
            .map_err(|_| ModelManagerError::LockPoisoned("chat_completion_engines"))?;
        if chat.contains(model) {
            return Ok(true);
        }
        let comp = self
            .completion_engines
            .read()
            .map_err(|_| ModelManagerError::LockPoisoned("completion_engines"))?;
        Ok(comp.contains(model))
    }

    pub fn model_display_names(&self) -> Result<HashSet<String>, ModelManagerError> {
        let chat = self.list_chat_completions_models()?;
        let comp = self.list_completions_models()?;
        let embed = self.list_embeddings_models()?;
        Ok(chat.into_iter().chain(comp).chain(embed).collect())
    }

    pub fn list_chat_completions_models(&self) -> Result<Vec<String>, ModelManagerError> {
        let guard = self
            .chat_completion_engines
            .read()
            .map_err(|_| ModelManagerError::LockPoisoned("chat_completion_engines"))?;
        Ok(guard.list())
    }

    pub fn list_completions_models(&self) -> Result<Vec<String>, ModelManagerError> {
        let guard = self
            .completion_engines
            .read()
            .map_err(|_| ModelManagerError::LockPoisoned("completion_engines"))?;
        Ok(guard.list())
    }

    pub fn list_embeddings_models(&self) -> Result<Vec<String>, ModelManagerError> {
        let guard = self
            .embeddings_engines
            .read()
            .map_err(|_| ModelManagerError::LockPoisoned("embeddings_engines"))?;
        Ok(guard.list())
    }

    pub fn add_completions_model(
        &self,
        model: &str,
        engine: OpenAICompletionsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self
            .completion_engines
            .write()
            .map_err(|_| ModelManagerError::LockPoisoned("completion_engines"))?;
        clients.add(model, engine)
    }

    pub fn add_chat_completions_model(
        &self,
        model: &str,
        engine: OpenAIChatCompletionsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self
            .chat_completion_engines
            .write()
            .map_err(|_| ModelManagerError::LockPoisoned("chat_completion_engines"))?;
        clients.add(model, engine)
    }

    pub fn add_embeddings_model(
        &self,
        model: &str,
        engine: OpenAIEmbeddingsStreamingEngine,
    ) -> Result<(), ModelManagerError> {
        let mut clients = self
            .embeddings_engines
            .write()
            .map_err(|_| ModelManagerError::LockPoisoned("embeddings_engines"))?;
        clients.add(model, engine)
    }

    pub fn remove_completions_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self
            .completion_engines
            .write()
            .map_err(|_| ModelManagerError::LockPoisoned("completion_engines"))?;
        clients.remove(model)
    }

    pub fn remove_chat_completions_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self
            .chat_completion_engines
            .write()
            .map_err(|_| ModelManagerError::LockPoisoned("chat_completion_engines"))?;
        clients.remove(model)
    }

    pub fn remove_embeddings_model(&self, model: &str) -> Result<(), ModelManagerError> {
        let mut clients = self
            .embeddings_engines
            .write()
            .map_err(|_| ModelManagerError::LockPoisoned("embeddings_engines"))?;
        clients.remove(model)
    }

    pub fn get_embeddings_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIEmbeddingsStreamingEngine, ModelManagerError> {
        self.embeddings_engines
            .read()
            .map_err(|_| ModelManagerError::LockPoisoned("embeddings_engines"))?
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
            .map_err(|_| ModelManagerError::LockPoisoned("completion_engines"))?
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
            .map_err(|_| ModelManagerError::LockPoisoned("chat_completion_engines"))?
            .get(model)
            .cloned()
            .ok_or(ModelManagerError::ModelNotFound(model.to_string()))
    }

    /// Save a ModelEntry under an instance's etcd `models/` key so we can fetch it later when the key is
    /// deleted from etcd.
    pub fn save_model_entry(&self, key: &str, entry: ModelEntry) -> Result<(), ModelManagerError> {
        let mut guard = self
            .entries
            .lock()
            .map_err(|_| ModelManagerError::LockPoisoned("entries"))?;
        guard.insert(key.to_string(), entry);
        Ok(())
    }

    /// Remove and return model entry for this instance's etcd key. We do this when the instance stops.
    pub fn remove_model_entry(&self, key: &str) -> Result<Option<ModelEntry>, ModelManagerError> {
        let mut guard = self
            .entries
            .lock()
            .map_err(|_| ModelManagerError::LockPoisoned("entries"))?;
        Ok(guard.remove(key))
    }

    pub async fn kv_chooser_for(
        &self,
        model_name: &str,
        component: &Component,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
    ) -> anyhow::Result<Arc<KvRouter>> {
        if let Some(kv_chooser) = self
            .get_kv_chooser(model_name)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
        {
            // Check if the existing router has a different block size
            if kv_chooser.block_size() != kv_cache_block_size {
                tracing::warn!(
                    model_name = %model_name,
                    existing_block_size = %kv_chooser.block_size(),
                    requested_block_size = %kv_cache_block_size,
                    "KV Router block size mismatch! Model is requesting a different kv_cache_block_size than the existing router. \
                     This will cause routing to fail silently. Consider using the same block size or restarting the router."
                );
            }
            return Ok(kv_chooser);
        }
        self.create_kv_chooser(model_name, component, kv_cache_block_size, kv_router_config)
            .await
    }

    fn get_kv_chooser(&self, model_name: &str) -> Result<Option<Arc<KvRouter>>, ModelManagerError> {
        let guard = self
            .kv_choosers
            .lock()
            .map_err(|_| ModelManagerError::LockPoisoned("kv_choosers"))?;
        Ok(guard.get(model_name).cloned())
    }

    /// Create and return a KV chooser for this component and model
    async fn create_kv_chooser(
        &self,
        model_name: &str,
        component: &Component,
        kv_cache_block_size: u32,
        kv_router_config: Option<KvRouterConfig>,
    ) -> anyhow::Result<Arc<KvRouter>> {
        let etcd_client = component
            .drt()
            .etcd_client()
            .ok_or_else(|| anyhow::anyhow!("KV routing requires etcd (dynamic mode)"))?;
        let router_key = format!(
            "kv_routers/{}/{}",
            Slug::from_string(model_name),
            uuid::Uuid::new_v4()
        );
        etcd_client
            .kv_create(
                &router_key,
                serde_json::to_vec_pretty(&kv_router_config.unwrap_or_default())?,
                None, // use primary lease
            )
            .await?;

        let selector = Box::new(DefaultWorkerSelector::new(kv_router_config));
        let chooser = KvRouter::new(
            component.clone(),
            kv_cache_block_size,
            Some(selector),
            kv_router_config,
        )
        .await?;
        let new_kv_chooser = Arc::new(chooser);
        self.kv_choosers
            .lock()
            .map_err(|_| ModelManagerError::LockPoisoned("kv_choosers"))
            .context("failed to acquire kv_choosers lock for insert")?
            .insert(model_name.to_string(), new_kv_chooser.clone());
        Ok(new_kv_chooser)
    }

    pub fn get_model_tool_call_parser(&self, model: &str) -> Option<String> {
        match self.entries.lock() {
            Ok(entries) => entries
                .values()
                .find(|entry| entry.name == model)
                .and_then(|entry| entry.runtime_config.as_ref())
                .and_then(|config| config.tool_call_parser.clone())
                .map(|parser| parser.to_string()),
            Err(_) => None,
        }
    }
}

pub struct ModelEngines<E> {
    /// Optional default model name
    default: Option<String>,
    engines: HashMap<String, E>,
}

impl<E> Default for ModelEngines<E> {
    fn default() -> Self {
        Self {
            default: None,
            engines: HashMap::new(),
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

    fn add(&mut self, model: &str, engine: E) -> Result<(), ModelManagerError> {
        if self.engines.contains_key(model) {
            return Err(ModelManagerError::ModelAlreadyExists(model.to_string()));
        }
        self.engines.insert(model.to_string(), engine);
        Ok(())
    }

    fn remove(&mut self, model: &str) -> Result<(), ModelManagerError> {
        if self.engines.remove(model).is_none() {
            return Err(ModelManagerError::ModelNotFound(model.to_string()));
        }
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
}
