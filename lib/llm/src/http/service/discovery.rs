// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc::Receiver, Notify};

use dynamo_runtime::{
    component::{self, Component, Instance},
    pipeline::{
        network::egress::push_router::PushRouter, ManyOut, Operator, RouterMode, SegmentSource,
        ServiceBackend, SingleIn, Source,
    },
    protocols::{self, annotated::Annotated},
    slug::Slug,
    transports::etcd::{self, KeyValue, WatchEvent},
    DistributedRuntime,
};

use super::ModelManager;
use crate::protocols::openai::completions::{CompletionRequest, CompletionResponse};
use crate::{
    backend::Backend,
    model_type::ModelType,
    preprocessor::{BackendInput, OpenAIPreprocessor},
    protocols::common::llm_backend::LLMEngineOutput,
};
use crate::{
    key_value_store::{EtcdStorage, KeyValueStore, KeyValueStoreManager},
    model_card::{self, ModelDeploymentCard},
};
use crate::{
    kv_router::{scheduler::DefaultWorkerSelector, KvPushRouter, KvRouter},
    protocols::openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
    protocols::openai::embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
};
use tracing;

/// [ModelEntry] is a struct that contains the information for the HTTP service to discover models
/// from the etcd cluster.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelEntry {
    /// Public name of the model
    /// This will be used to identify the model in the HTTP service and the value used in an
    /// an [OAI ChatRequest][crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest].
    pub name: String,

    /// Component of the endpoint.
    pub endpoint: protocols::Endpoint,

    /// Specifies whether the model is a chat or completion model.s
    pub model_type: ModelType,
}

impl ModelEntry {
    pub fn requires_preprocessing(&self) -> bool {
        matches!(self.model_type, ModelType::Backend)
    }

    /// Fetch the ModelDeploymentCard from NATS.
    /// This does not touch it's fields so you may need to call move_from_nats on it.
    pub async fn load_mdc(
        &self,
        etcd_client: &etcd::Client,
    ) -> anyhow::Result<ModelDeploymentCard> {
        let kvstore: Box<dyn KeyValueStore> = Box::new(EtcdStorage::new(etcd_client.clone()));
        let card_store = Arc::new(KeyValueStoreManager::new(kvstore));
        let card_key = ModelDeploymentCard::service_name_slug(&self.name);
        match card_store
            .load::<ModelDeploymentCard>(model_card::ROOT_PATH, &card_key)
            .await
        {
            Ok(Some(mdc)) => Ok(mdc),
            Ok(None) => {
                anyhow::bail!("Missing ModelDeploymentCard in etcd under key {card_key}");
            }
            Err(err) => {
                anyhow::bail!(
                    "Error fetching ModelDeploymentCard from etcd under key {card_key}. {err}"
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelNetworkName(String);

impl ModelNetworkName {
    /// Key to store this model entry in networked key-value store (etcd).
    ///
    /// It looks like this:
    /// ns.cp.ep-694d967ca5efd804
    fn from_parts(namespace: &str, component: &str, endpoint: &str, lease_id: i64) -> Self {
        let model_root = component::MODEL_ROOT_PATH;
        let slug = Slug::slugify(&format!("{namespace}.{component}.{endpoint}-{lease_id:x}"));
        ModelNetworkName(format!("{model_root}/{slug}"))
    }

    // We can't do From<&component::Endpoint> here because we also need the lease_id
    pub fn from_local(endpoint: &component::Endpoint, lease_id: i64) -> Self {
        Self::from_parts(
            &endpoint.component().namespace().to_string(),
            &endpoint.component().name(),
            endpoint.name(),
            lease_id,
        )
    }

    /// Fetch the ModelEntry from etcd.
    pub async fn load_entry(&self, etcd_client: &etcd::Client) -> anyhow::Result<ModelEntry> {
        let mut model_entries = etcd_client.kv_get(self.to_string(), None).await?;
        if model_entries.is_empty() {
            anyhow::bail!("No ModelEntry in etcd for key {self}");
        }
        let model_entry = model_entries.remove(0);
        serde_json::from_slice(model_entry.value()).with_context(|| {
            format!(
                "Error deserializing JSON. Key={self}. JSON={}",
                model_entry.value_str().unwrap_or("INVALID UTF-8")
            )
        })
    }

    /// Fetch the ModelDeploymentCard from NATS.
    /// This does not touch it's fields so you may need to call move_from_nats on it.
    /// TODO We have potentially two for each endpoint, one Chat and one Completion.
    pub async fn load_mdc(
        &self,
        etcd_client: &etcd::Client,
    ) -> anyhow::Result<ModelDeploymentCard> {
        let entry = self.load_entry(etcd_client).await?;
        entry.load_mdc(etcd_client).await
    }
}

impl From<&Instance> for ModelNetworkName {
    fn from(cei: &Instance) -> Self {
        Self::from_parts(
            &cei.namespace,
            &cei.component,
            &cei.endpoint,
            cei.instance_id,
        )
    }
}

impl std::fmt::Display for ModelNetworkName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct ModelWatcher {
    manager: ModelManager,
    drt: DistributedRuntime,
    router_mode: RouterMode,
    notify_on_model: Notify,
}

impl ModelWatcher {
    pub async fn new(
        runtime: DistributedRuntime,
        model_manager: ModelManager,
        router_mode: RouterMode,
    ) -> anyhow::Result<ModelWatcher> {
        Ok(Self {
            manager: model_manager,
            drt: runtime,
            router_mode,
            notify_on_model: Notify::new(),
        })
    }

    /// Wait until we have at least one chat completions model and return it's name.
    pub async fn wait_for_chat_model(&self) -> String {
        // Loop in case it gets added and immediately deleted
        loop {
            if let Some(model_name) = self.manager.list_chat_completions_models().first() {
                return model_name.to_owned();
            }
            self.notify_on_model.notified().await
        }
    }

    pub async fn watch(self: Arc<Self>, mut events_rx: Receiver<WatchEvent>) {
        tracing::debug!("model watcher started");

        while let Some(event) = events_rx.recv().await {
            match event {
                WatchEvent::Put(kv) => {
                    let model_entry = match serde_json::from_slice::<ModelEntry>(kv.value()) {
                        Ok(model_entry) => model_entry,
                        Err(err) => {
                            match kv.value_str() {
                                Ok(value) => {
                                    tracing::error!(%err, value, "Invalid JSON in model entry")
                                }
                                Err(value_str_err) => {
                                    tracing::error!(original_error = %err, %value_str_err, "Invalid UTF-8 string in model entry, expected JSON")
                                }
                            }
                            continue;
                        }
                    };
                    if self.manager.has_model_any(&model_entry.name) {
                        tracing::trace!(
                            service_name = model_entry.name,
                            "New endpoint for existing model"
                        );
                        self.notify_on_model.notify_waiters();
                        continue;
                    }

                    match self.clone().handle_put(&kv, &model_entry).await {
                        Ok(()) => {
                            tracing::info!(model_name = model_entry.name, "added model");
                            self.notify_on_model.notify_waiters();
                        }
                        Err(e) => {
                            tracing::error!(%e, "error adding model {}", model_entry.name);
                        }
                    }
                }
                WatchEvent::Delete(kv) => match self.clone().handle_delete(&kv).await {
                    Ok(model_name) => {
                        tracing::info!("removed model {}", model_name);
                    }
                    Err(e) => {
                        tracing::error!("error removing model: {}", e);
                    }
                },
            }
        }
    }

    /// Returns the name of the model we just deleted
    async fn handle_delete(self: Arc<ModelWatcher>, kv: &KeyValue) -> anyhow::Result<String> {
        let key = kv.key_str()?;
        let model_entry = match self.manager.state.entries.lock().unwrap().remove(key) {
            Some(entry) => entry,
            None => {
                anyhow::bail!("Missing ModelEntry for {key}");
            }
        };
        let model_name = &model_entry.name;
        tracing::debug!(model_name, "removing model");

        // Ignore the errors because model could be either type
        let _ = self.manager.remove_chat_completions_model(model_name);
        let _ = self.manager.remove_completions_model(model_name);
        let _ = self.manager.remove_embeddings_model(model_name);

        // We own model_entry now so take ownership of the name
        Ok(model_entry.name)
    }

    // Handles a PUT event from etcd, this usually means adding a new model to the list of served
    // models.
    //
    // If this method errors, for the near term, we will delete the offending key.
    async fn handle_put(
        self: Arc<ModelWatcher>,
        kv: &KeyValue,
        model_entry: &ModelEntry,
    ) -> anyhow::Result<()> {
        let key = kv.key_str()?;
        let endpoint_id = model_entry.endpoint.clone();
        let component = self
            .drt
            .namespace(&endpoint_id.namespace)?
            .component(&endpoint_id.component)?;
        let client = component.endpoint(&endpoint_id.name).client().await?;

        let Some(etcd_client) = self.drt.etcd_client() else {
            // Should be impossible because we only get here on an etcd event
            anyhow::bail!("Missing etcd_client");
        };
        let card = match model_entry.load_mdc(&etcd_client).await {
            Ok(card) => {
                tracing::debug!(card.display_name, "adding model");
                Some(card)
            }
            Err(err) => {
                // `dynamo serve` isn't using MDC yet so can't be an error
                tracing::info!(%err, "load_mdc did not complete");
                None
            }
        };
        // We need to save the entry to know what the model is called when we delete it
        self.manager
            .state
            .entries
            .lock()
            .unwrap()
            .insert(key.to_string(), model_entry.clone());

        match model_entry.model_type {
            ModelType::Backend => {
                // A Backend model expects pre-processed requests meaning it's up to us whether we
                // handle Chat or Completions requests, so handle both.

                let Some(mut card) = card else {
                    anyhow::bail!("Missing model deployment card");
                };
                // Download tokenizer.json etc to local disk
                // This cache_dir is a tempfile::TempDir will be deleted on drop. I _think_
                // OpenAIPreprocessor::new loads the files, so we can delete them after this
                // function. Needs checking carefully, possibly we need to store it in state.
                let _cache_dir = Some(card.move_from_nats(self.drt.nats_client()).await?);

                let frontend = SegmentSource::<
                    SingleIn<NvCreateChatCompletionRequest>,
                    ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
                >::new();
                let preprocessor = OpenAIPreprocessor::new(card.clone()).await?.into_operator();
                let backend = Backend::from_mdc(card.clone()).await?.into_operator();
                let router = PushRouter::<BackendInput, Annotated<LLMEngineOutput>>::from_client(
                    client.clone(),
                    self.router_mode,
                )
                .await?;
                let service_backend = match self.router_mode {
                    RouterMode::Random | RouterMode::RoundRobin | RouterMode::Direct(_) => {
                        ServiceBackend::from_engine(Arc::new(router))
                    }
                    RouterMode::KV => {
                        let chooser = self.kv_chooser_for(&model_entry.name, &component).await?;
                        let kv_push_router = KvPushRouter::new(router, chooser);
                        ServiceBackend::from_engine(Arc::new(kv_push_router))
                    }
                };

                let chat_engine = frontend
                    .link(preprocessor.forward_edge())?
                    .link(backend.forward_edge())?
                    .link(service_backend)?
                    .link(backend.backward_edge())?
                    .link(preprocessor.backward_edge())?
                    .link(frontend)?;
                self.manager
                    .add_chat_completions_model(&model_entry.name, chat_engine)?;

                let frontend = SegmentSource::<
                    SingleIn<CompletionRequest>,
                    ManyOut<Annotated<CompletionResponse>>,
                >::new();
                let preprocessor = OpenAIPreprocessor::new(card.clone()).await?.into_operator();
                let backend = Backend::from_mdc(card.clone()).await?.into_operator();
                let router = PushRouter::<BackendInput, Annotated<LLMEngineOutput>>::from_client(
                    client,
                    self.router_mode,
                )
                .await?;
                let service_backend = match self.router_mode {
                    RouterMode::Random | RouterMode::RoundRobin | RouterMode::Direct(_) => {
                        ServiceBackend::from_engine(Arc::new(router))
                    }
                    RouterMode::KV => {
                        let chooser = self.kv_chooser_for(&model_entry.name, &component).await?;
                        let kv_push_router = KvPushRouter::new(router, chooser);
                        ServiceBackend::from_engine(Arc::new(kv_push_router))
                    }
                };

                let completions_engine = frontend
                    .link(preprocessor.forward_edge())?
                    .link(backend.forward_edge())?
                    .link(service_backend)?
                    .link(backend.backward_edge())?
                    .link(preprocessor.backward_edge())?
                    .link(frontend)?;
                self.manager
                    .add_completions_model(&model_entry.name, completions_engine)?;
            }
            ModelType::Chat => {
                let push_router = PushRouter::<
                    NvCreateChatCompletionRequest,
                    Annotated<NvCreateChatCompletionStreamResponse>,
                >::from_client(client, Default::default())
                .await?;
                let engine = Arc::new(push_router);
                self.manager
                    .add_chat_completions_model(&model_entry.name, engine)?;
            }
            ModelType::Completion => {
                let push_router =
                    PushRouter::<CompletionRequest, Annotated<CompletionResponse>>::from_client(
                        client,
                        Default::default(),
                    )
                    .await?;
                let engine = Arc::new(push_router);
                self.manager
                    .add_completions_model(&model_entry.name, engine)?;
            }
            ModelType::Embedding => {
                let push_router = PushRouter::<
                    NvCreateEmbeddingRequest,
                    Annotated<NvCreateEmbeddingResponse>,
                >::from_client(client, Default::default())
                .await?;
                let engine = Arc::new(push_router);
                self.manager
                    .add_embeddings_model(&model_entry.name, engine)?;
            }
        }

        Ok(())
    }

    async fn kv_chooser_for(
        &self,
        model_name: &str,
        component: &Component,
    ) -> anyhow::Result<Arc<KvRouter>> {
        if let Some(kv_chooser) = self
            .manager
            .state
            .kv_choosers
            .lock()
            .unwrap()
            .get(model_name)
        {
            // Return early to avoid holding the lock during the await later
            return Ok(Arc::clone(kv_chooser));
        }
        let selector = Box::new(DefaultWorkerSelector {});
        let chooser = KvRouter::new(
            component.clone(),
            crate::DEFAULT_KV_BLOCK_SIZE,
            Some(selector),
        )
        .await?;
        let new_kv_chooser = Arc::new(chooser);
        self.manager
            .state
            .kv_choosers
            .lock()
            .unwrap()
            .insert(model_name.to_string(), new_kv_chooser.clone());
        Ok(new_kv_chooser)
    }
}
