// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use anyhow::Context as _;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Receiver;

use dynamo_runtime::{
    component::{self, ComponentEndpointInfo},
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
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
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
        endpoint_id: protocols::Endpoint,
        etcd_client: etcd::Client,
    ) -> anyhow::Result<ModelDeploymentCard> {
        let kvstore: Box<dyn KeyValueStore> =
            Box::new(EtcdStorage::new(etcd_client.clone(), endpoint_id));
        let card_store = Arc::new(KeyValueStoreManager::new(kvstore));
        let card_key = ModelDeploymentCard::service_name_slug(&self.name);
        match card_store
            .load::<ModelDeploymentCard>(model_card::BUCKET_NAME, &card_key)
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
        ModelNetworkName(
            Slug::slugify(&format!("{namespace}.{component}.{endpoint}-{lease_id:x}")).to_string(),
        )
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
    pub async fn load_entry(&self, etcd_client: etcd::Client) -> anyhow::Result<ModelEntry> {
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
        endpoint_id: protocols::Endpoint,
        etcd_client: etcd::Client,
    ) -> anyhow::Result<ModelDeploymentCard> {
        let entry = self.load_entry(etcd_client.clone()).await?;
        entry.load_mdc(endpoint_id, etcd_client).await
    }
}

impl From<&ComponentEndpointInfo> for ModelNetworkName {
    fn from(cei: &ComponentEndpointInfo) -> Self {
        Self::from_parts(&cei.namespace, &cei.component, &cei.endpoint, cei.lease_id)
    }
}

impl std::fmt::Display for ModelNetworkName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct ModelWatchState {
    pub prefix: String,
    pub manager: ModelManager,
    pub drt: DistributedRuntime,
}

pub async fn model_watcher(state: Arc<ModelWatchState>, mut events_rx: Receiver<WatchEvent>) {
    tracing::debug!("model watcher started");

    while let Some(event) = events_rx.recv().await {
        match event {
            WatchEvent::Put(kv) => {
                let model_entry = match serde_json::from_slice::<ModelEntry>(kv.value()) {
                    Ok(model_entry) => model_entry,
                    Err(err) => {
                        tracing::error!(%err, ?kv, "Invalid JSON in model entry");
                        continue;
                    }
                };
                if state.manager.has_model_any(&model_entry.name) {
                    tracing::trace!(
                        service_name = model_entry.name,
                        "New endpoint for existing model"
                    );
                    continue;
                }

                match handle_put(&model_entry, state.clone()).await {
                    Ok(()) => {
                        tracing::info!(model_name = model_entry.name, "added model");
                    }
                    Err(e) => {
                        tracing::error!(%e, "error adding model {}", model_entry.name);
                    }
                }
            }
            WatchEvent::Delete(kv) => match handle_delete(&kv, state.clone()).await {
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

async fn handle_delete(kv: &KeyValue, state: Arc<ModelWatchState>) -> anyhow::Result<&str> {
    let key = kv.key_str()?;
    tracing::debug!(key, "removing model");

    let model_name = key.trim_start_matches(&state.prefix);

    // Ignore the errors because model could be either type
    let _ = state.manager.remove_chat_completions_model(model_name);
    let _ = state.manager.remove_completions_model(model_name);

    Ok(model_name)
}

// Handles a PUT event from etcd, this usually means adding a new model to the list of served
// models.
//
// If this method errors, for the near term, we will delete the offending key.
async fn handle_put(model_entry: &ModelEntry, state: Arc<ModelWatchState>) -> anyhow::Result<()> {
    let endpoint_id = model_entry.endpoint.clone();
    let client = state
        .drt
        .namespace(&endpoint_id.namespace)?
        .component(&endpoint_id.component)?
        .endpoint(&endpoint_id.name)
        .client()
        .await?;

    let Some(etcd_client) = state.drt.etcd_client() else {
        // Should be impossible because we only get here on an etcd event
        anyhow::bail!("Missing etcd_client");
    };
    let card = match model_entry.load_mdc(endpoint_id, etcd_client).await {
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
            let _cache_dir = Some(card.move_from_nats(state.drt.nats_client()).await?);

            let frontend = SegmentSource::<
                SingleIn<NvCreateChatCompletionRequest>,
                ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            >::new();
            let preprocessor = OpenAIPreprocessor::new(card.clone()).await?.into_operator();
            let backend = Backend::from_mdc(card.clone()).await?.into_operator();
            let router = PushRouter::<BackendInput, Annotated<LLMEngineOutput>>::from_client(
                client.clone(),
                RouterMode::Random, // TODO how do we configure this?
            )
            .await?;

            let chat_engine = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(ServiceBackend::from_engine(Arc::new(router)))?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;
            state
                .manager
                .add_chat_completions_model(&model_entry.name, chat_engine)?;

            let frontend = SegmentSource::<
                SingleIn<CompletionRequest>,
                ManyOut<Annotated<CompletionResponse>>,
            >::new();
            let preprocessor = OpenAIPreprocessor::new(card.clone()).await?.into_operator();
            let backend = Backend::from_mdc(card.clone()).await?.into_operator();
            let router = PushRouter::<BackendInput, Annotated<LLMEngineOutput>>::from_client(
                client,
                RouterMode::Random, // TODO how do we configure this?
            )
            .await?;

            let completions_engine = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(ServiceBackend::from_engine(Arc::new(router)))?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;
            state
                .manager
                .add_completions_model(&model_entry.name, completions_engine)?;
        }
        ModelType::Chat => {
            let push_router = PushRouter::<
                NvCreateChatCompletionRequest,
                Annotated<NvCreateChatCompletionStreamResponse>,
            >::from_client(client, Default::default())
            .await?;
            let engine = Arc::new(push_router);
            state
                .manager
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
            state
                .manager
                .add_completions_model(&model_entry.name, engine)?;
        }
    }

    Ok(())
}
