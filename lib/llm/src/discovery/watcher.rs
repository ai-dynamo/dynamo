// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use tokio::sync::mpsc::Sender;

use anyhow::Context as _;
use tokio::sync::{Notify, mpsc::Receiver};

use dynamo_runtime::{
    DistributedRuntime,
    pipeline::{
        ManyOut, Operator, RouterMode, SegmentSource, ServiceBackend, SingleIn, Source,
        network::egress::push_router::PushRouter,
    },
    protocols::annotated::Annotated,
    transports::etcd::{KeyValue, WatchEvent},
};

use crate::{
    backend::Backend,
    entrypoint,
    kv_router::KvRouterConfig,
    model_type::ModelType,
    preprocessor::{OpenAIPreprocessor, PreprocessedEmbeddingRequest},
    protocols::{
        common::llm_backend::EmbeddingsEngineOutput,
        openai::{
            chat_completions::{
                NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            },
            completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
            embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
        },
    },
};

use super::{MODEL_ROOT_PATH, ModelEntry, ModelManager};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelUpdate {
    Added(ModelType),
    Removed(ModelType),
}

pub struct ModelWatcher {
    manager: Arc<ModelManager>,
    drt: DistributedRuntime,
    router_mode: RouterMode,
    notify_on_model: Notify,
    model_update_tx: Option<Sender<ModelUpdate>>,
    kv_router_config: Option<KvRouterConfig>,
    busy_threshold: Option<f64>,
}

const ALL_MODEL_TYPES: &[ModelType] =
    &[ModelType::Chat, ModelType::Completion, ModelType::Embedding];

impl ModelWatcher {
    pub fn new(
        runtime: DistributedRuntime,
        model_manager: Arc<ModelManager>,
        router_mode: RouterMode,
        kv_router_config: Option<KvRouterConfig>,
        busy_threshold: Option<f64>,
    ) -> ModelWatcher {
        Self {
            manager: model_manager,
            drt: runtime,
            router_mode,
            notify_on_model: Notify::new(),
            model_update_tx: None,
            kv_router_config,
            busy_threshold,
        }
    }

    pub fn set_notify_on_model_update(&mut self, tx: Sender<ModelUpdate>) {
        self.model_update_tx = Some(tx);
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

                    if let Some(tx) = &self.model_update_tx {
                        tx.send(ModelUpdate::Added(model_entry.model_type))
                            .await
                            .ok();
                    }

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
                        Err(err) => {
                            tracing::error!(
                                error = format!("{err:#}"),
                                "error adding model {}",
                                model_entry.name
                            );
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
        let chat_model_remove_err = self.manager.remove_chat_completions_model(&model_name);
        let completions_model_remove_err = self.manager.remove_completions_model(&model_name);
        let embeddings_model_remove_err = self.manager.remove_embeddings_model(&model_name);

        let mut chat_model_removed = false;
        let mut completions_model_removed = false;
        let mut embeddings_model_removed = false;

        if chat_model_remove_err.is_ok() && self.manager.list_chat_completions_models().is_empty() {
            chat_model_removed = true;
        }
        if completions_model_remove_err.is_ok() && self.manager.list_completions_models().is_empty()
        {
            completions_model_removed = true;
        }
        if embeddings_model_remove_err.is_ok() && self.manager.list_embeddings_models().is_empty() {
            embeddings_model_removed = true;
        }

        if !chat_model_removed && !completions_model_removed && !embeddings_model_removed {
            tracing::debug!(
                "No updates to send for model {}: chat_model_removed: {}, completions_model_removed: {}, embeddings_model_removed: {}",
                model_name,
                chat_model_removed,
                completions_model_removed,
                embeddings_model_removed
            );
        } else {
            for model_type in ALL_MODEL_TYPES {
                if ((chat_model_removed && *model_type == ModelType::Chat)
                    || (completions_model_removed && *model_type == ModelType::Completion)
                    || (embeddings_model_removed && *model_type == ModelType::Embedding))
                    && let Some(tx) = &self.model_update_tx
                {
                    tx.send(ModelUpdate::Removed(*model_type)).await.ok();
                }
            }
        }

        Ok(Some(model_name))
    }

    // Handles a PUT event from etcd, this usually means adding a new model to the list of served
    // models.
    async fn handle_put(&self, model_entry: &ModelEntry) -> anyhow::Result<()> {
        let endpoint_id = &model_entry.endpoint_id;
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

                let kv_chooser = if self.router_mode == RouterMode::KV {
                    Some(
                        self.manager
                            .kv_chooser_for(
                                &model_entry.name,
                                &component,
                                card.kv_cache_block_size,
                                self.kv_router_config,
                            )
                            .await?,
                    )
                } else {
                    None
                };

                let chat_engine = entrypoint::build_routed_pipeline::<
                    NvCreateChatCompletionRequest,
                    NvCreateChatCompletionStreamResponse,
                >(
                    &card,
                    &client,
                    self.router_mode,
                    self.busy_threshold,
                    kv_chooser.clone(),
                )
                .await?;
                self.manager
                    .add_chat_completions_model(&model_entry.name, chat_engine)?;

                let completions_engine = entrypoint::build_routed_pipeline::<
                    NvCreateCompletionRequest,
                    NvCreateCompletionResponse,
                >(
                    &card,
                    &client,
                    self.router_mode,
                    self.busy_threshold,
                    kv_chooser,
                )
                .await?;
                self.manager
                    .add_completions_model(&model_entry.name, completions_engine)?;
            }
            ModelType::Chat => {
                let push_router = PushRouter::<
                    NvCreateChatCompletionRequest,
                    Annotated<NvCreateChatCompletionStreamResponse>,
                >::from_client_with_threshold(
                    client, Default::default(), self.busy_threshold
                )
                .await?;
                let engine = Arc::new(push_router);
                self.manager
                    .add_chat_completions_model(&model_entry.name, engine)?;
            }
            ModelType::Completion => {
                let push_router = PushRouter::<
                    NvCreateCompletionRequest,
                    Annotated<NvCreateCompletionResponse>,
                >::from_client_with_threshold(
                    client, Default::default(), self.busy_threshold
                )
                .await?;
                let engine = Arc::new(push_router);
                self.manager
                    .add_completions_model(&model_entry.name, engine)?;
            }
            ModelType::Embedding => {
                let Some(mut card) = card else {
                    anyhow::bail!("Missing model deployment card for embedding model");
                };

                // Download tokenizer files to local disk
                let _cache_dir = Some(card.move_from_nats(self.drt.nats_client()).await?);

                // Create preprocessing pipeline similar to Backend
                let frontend = SegmentSource::<
                    SingleIn<NvCreateEmbeddingRequest>,
                    ManyOut<Annotated<NvCreateEmbeddingResponse>>,
                >::new();

                let preprocessor = OpenAIPreprocessor::new(card.clone()).await?.into_operator();
                let backend = Backend::from_mdc(card.clone()).await?.into_operator();

                let router = PushRouter::<
                    PreprocessedEmbeddingRequest,
                    Annotated<EmbeddingsEngineOutput>,
                >::from_client_with_threshold(
                    client, self.router_mode, self.busy_threshold
                )
                .await?;

                // Note: Embeddings don't need KV routing complexity
                let service_backend = ServiceBackend::from_engine(Arc::new(router));

                // Link the pipeline: frontend -> preprocessor -> backend -> service_backend -> backend -> preprocessor -> frontend
                let embedding_engine = frontend
                    .link(preprocessor.forward_edge())?
                    .link(backend.forward_edge())?
                    .link(service_backend)?
                    .link(backend.backward_edge())?
                    .link(preprocessor.backward_edge())?
                    .link(frontend)?;

                self.manager
                    .add_embeddings_model(&model_entry.name, embedding_engine)?;
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
