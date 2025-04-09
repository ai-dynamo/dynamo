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

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Receiver;

use dynamo_runtime::{
    protocols::{self, annotated::Annotated},
    raise,
    transports::etcd::{KeyValue, WatchEvent},
    DistributedRuntime,
};

use super::ModelManager;
use crate::model_type::ModelType;
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
};
use crate::protocols::openai::completions::{CompletionRequest, CompletionResponse};
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
            let client = state
                .drt
                .namespace(model_entry.endpoint.namespace)?
                .component(model_entry.endpoint.component)?
                .endpoint(model_entry.endpoint.name)
                .client::<NvCreateChatCompletionRequest, Annotated<NvCreateChatCompletionStreamResponse>>()
                .await?;
            state
                .manager
                .add_chat_completions_model(&model_entry.name, Arc::new(client))?;
        }
        ModelType::Completion => {
            let client = state
                .drt
                .namespace(model_entry.endpoint.namespace)?
                .component(model_entry.endpoint.component)?
                .endpoint(model_entry.endpoint.name)
                .client::<CompletionRequest, Annotated<CompletionResponse>>()
                .await?;
            state
                .manager
                .add_completions_model(&model_entry.name, Arc::new(client))?;
        }
    }

    Ok((model_entry.name, state.model_type))
}
