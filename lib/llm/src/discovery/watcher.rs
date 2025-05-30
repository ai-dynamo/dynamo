// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Context as _;
use tokio::sync::{mpsc::Receiver, Notify};

use dynamo_runtime::{
    pipeline::{
        network::egress::push_router::PushRouter, ManyOut, Operator, RouterMode, SegmentSource,
        ServiceBackend, SingleIn, Source,
    },
    protocols::annotated::Annotated,
    transports::etcd::{KeyValue, WatchEvent},
    DistributedRuntime,
};

use crate::{
    backend::Backend,
    kv_router::KvPushRouter,
    model_type::ModelType,
    preprocessor::{BackendInput, OpenAIPreprocessor},
    protocols::common::llm_backend::LLMEngineOutput,
    protocols::openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
    protocols::openai::completions::{CompletionRequest, CompletionResponse},
    protocols::openai::embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
};

use super::{ModelEntry, ModelManager, MODEL_ROOT_PATH};

pub struct ModelWatcher {
    manager: Arc<ModelManager>,
    drt: DistributedRuntime,
    router_mode: RouterMode,
    notify_on_model: Notify,
}

impl ModelWatcher {
    pub fn new(
        runtime: DistributedRuntime,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
    ) -> ModelWatcher {
        Self {
            manager: model_manager,
            drt: runtime,
            router_mode,
            notify_on_model: Notify::new(),
        }
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

    pub async fn watch(&self, mut events_rx: Receiver<WatchEvent>) {
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
                    let key = match kv.key_str() {
                        Ok(k) => k,
                        Err(err) => {
                            tracing::error!(%err, ?kv, "Invalid UTF-8 string in model entry key, skipping");
                            continue;
                        }
                    };
                    self.manager.save_model_entry(key, model_entry.clone());

                    if self.manager.has_model_any(&model_entry.name) {
                        tracing::trace!(name = model_entry.name, "New endpoint for existing model");
                        self.notify_on_model.notify_waiters();
                        continue;
                    }

                    match self.handle_put(&model_entry).await {
                        Ok(()) => {
                            tracing::info!(model_name = model_entry.name, "added model");
                            self.notify_on_model.notify_waiters();
                        }
                        Err(e) => {
                            tracing::error!(%e, "error adding model {}", model_entry.name);
                        }
                    }
                }
                WatchEvent::Delete(kv) => match self.handle_delete(&kv).await {
                    Ok(Some(model_name)) => {
                        tracing::info!("removed model {}", model_name);
                    }
                    Ok(None) => {
                        // There are other instances running this model, nothing to do
                    }
                    Err(e) => {
                        tracing::error!("error removing model: {}", e);
                    }
                },
            }
        }
    }

    /// If the last instance running this model has gone delete it.
    /// Returns the name of the model we just deleted, if any.
    async fn handle_delete(&self, kv: &KeyValue) -> anyhow::Result<Option<String>> {
        let key = kv.key_str()?;
        let model_entry = match self.manager.remove_model_entry(key) {
            Some(entry) => entry,
            None => {
                anyhow::bail!("Missing ModelEntry for {key}");
            }
        };
        let model_name = model_entry.name;
        let active_instances = self
            .entries_for_model(&model_name)
            .await
            .with_context(|| model_name.clone())?;
        if !active_instances.is_empty() {
            return Ok(None);
        }

        // Ignore the errors because model could be either type
        let _ = self.manager.remove_chat_completions_model(&model_name);
        let _ = self.manager.remove_completions_model(&model_name);
        let _ = self.manager.remove_embeddings_model(&model_name);

        Ok(Some(model_name))
    }

    // Handles a PUT event from etcd, this usually means adding a new model to the list of served
    // models.
    async fn handle_put(&self, model_entry: &ModelEntry) -> anyhow::Result<()> {
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
                        let chooser = self
                            .manager
                            .kv_chooser_for(&model_entry.name, &component, card.kv_cache_block_size)
                            .await?;
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
                        let chooser = self
                            .manager
                            .kv_chooser_for(&model_entry.name, &component, card.kv_cache_block_size)
                            .await?;
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

    /// All the registered ModelEntry, one per instance
    pub async fn all_entries(&self) -> anyhow::Result<Vec<ModelEntry>> {
        let Some(etcd_client) = self.drt.etcd_client() else {
            anyhow::bail!("all_entries: Missing etcd client");
        };
        let kvs = etcd_client.kv_get_prefix(MODEL_ROOT_PATH).await?;
        let mut entries = Vec::with_capacity(kvs.len());
        for kv in kvs {
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
            entries.push(model_entry);
        }
        Ok(entries)
    }

    pub async fn entries_for_model(&self, model_name: &str) -> anyhow::Result<Vec<ModelEntry>> {
        let mut all = self.all_entries().await?;
        all.retain(|entry| entry.name == model_name);
        Ok(all)
    }
}
