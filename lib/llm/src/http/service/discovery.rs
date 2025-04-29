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
    pipeline::network::egress::push_router::PushRouter,
    protocols::{self, annotated::Annotated},
    raise,
    slug::Slug,
    transports::etcd::{self, KeyValue, WatchEvent},
    DistributedRuntime,
};

use super::ModelManager;
use crate::model_type::ModelType;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::protocols::openai::completions::{CompletionRequest, CompletionResponse};
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

    pub async fn load_mdc(
        &self,
        endpoint_id: protocols::Endpoint,
        etcd_client: etcd::Client,
    ) -> anyhow::Result<ModelDeploymentCard> {
        let network_name = self;
        let model_entries = etcd_client.kv_get(network_name.to_string(), None).await?;
        if model_entries.is_empty() {
            anyhow::bail!("No ModelEntry in etcd for key {network_name}");
        }
        let entry: ModelEntry =
            serde_json::from_slice(model_entries[0].value()).with_context(|| {
                format!(
                    "Error deserializing JSON. Key={network_name}. JSON={}",
                    model_entries[0].value_str().unwrap_or("INVALID UTF-8")
                )
            })?;
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
    pub model_type: ModelType,
    pub manager: ModelManager,
    pub drt: DistributedRuntime,
}

pub async fn model_watcher(state: Arc<ModelWatchState>, mut events_rx: Receiver<WatchEvent>) {
    tracing::debug!("model watcher started");

    while let Some(event) = events_rx.recv().await {
        match event {
            WatchEvent::Put(kv) => {
                let key = match kv.key_str() {
                    Ok(key) => key,
                    Err(err) => {
                        tracing::error!(%err, ?kv, "Invalid UTF8 in model key");
                        continue;
                    }
                };
                tracing::debug!(key, "adding model");

                // model_entry.name is the service name (e.g. "Llama-3.2-3B-Instruct")
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

                match handle_put(model_entry, state.clone()).await {
                    Ok((model_name, model_type)) => {
                        tracing::info!("added {} model: {}", model_type, model_name);
                    }
                    Err(e) => {
                        tracing::error!("error adding model: {}", e);
                    }
                }
            }
            WatchEvent::Delete(kv) => match handle_delete(&kv, state.clone()).await {
                Ok((model_name, model_type)) => {
                    tracing::info!("removed {} model: {}", model_type, model_name);
                }
                Err(e) => {
                    tracing::error!("error removing model: {}", e);
                }
            },
        }
    }
}

async fn handle_delete(
    kv: &KeyValue,
    state: Arc<ModelWatchState>,
) -> anyhow::Result<(&str, ModelType)> {
    let key = kv.key_str()?;
    tracing::debug!(key, "removing model");

    let model_name = key.trim_start_matches(&state.prefix);

    match state.model_type {
        ModelType::Chat => state.manager.remove_chat_completions_model(model_name)?,
        ModelType::Completion => state.manager.remove_completions_model(model_name)?,
    };

    Ok((model_name, state.model_type))
}

// Handles a PUT event from etcd, this usually means adding a new model to the list of served
// models.
//
// If this method errors, for the near term, we will delete the offending key.
async fn handle_put(
    model_entry: ModelEntry,
    state: Arc<ModelWatchState>,
) -> anyhow::Result<(String, ModelType)> {
    if model_entry.model_type != state.model_type {
        raise!(
            "model type mismatch: {} != {}",
            model_entry.model_type,
            state.model_type
        );
    }

    match state.model_type {
        ModelType::Chat => {
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
            let mdc = match model_entry.load_mdc(endpoint_id, etcd_client).await {
                Ok(mdc) => Some(mdc),
                Err(err) => {
                    // `dynamo serve` isn't using MDC yet so can't be an error
                    tracing::info!(%err, "load_mdc did not complete");
                    None
                }
            };

            if mdc.is_some() && mdc.as_ref().unwrap().requires_preprocessing {
                // Note requires_preprocessing is never true in our code right now
                todo!("Ingress-side pre-processing not supported yet");
            } else {
                let push_router = PushRouter::<
                    NvCreateChatCompletionRequest,
                    Annotated<NvCreateChatCompletionStreamResponse>,
                >::from_client(client, Default::default())
                .await?;
                state
                    .manager
                    .add_chat_completions_model(&model_entry.name, Arc::new(push_router))?;
            }
        }
        ModelType::Completion => {
            let client = state
                .drt
                .namespace(model_entry.endpoint.namespace)?
                .component(model_entry.endpoint.component)?
                .endpoint(model_entry.endpoint.name)
                .client()
                .await?;

            // TODO: Handle pre-processing once it moves ingress-side

            let push_router =
                PushRouter::<CompletionRequest, Annotated<CompletionResponse>>::from_client(
                    client,
                    Default::default(),
                )
                .await?;
            state
                .manager
                .add_completions_model(&model_entry.name, Arc::new(push_router))?;
        }
    }

    Ok((model_entry.name, state.model_type))
}
