// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_runtime::transports::etcd;
use dynamo_runtime::{
    protocols,
    slug::Slug,
    storage::key_value_store::{EtcdStorage, KeyValueStore, KeyValueStoreManager},
};
use serde::{Deserialize, Serialize};

use crate::{
    local_model::runtime_config::ModelRuntimeConfig,
    model_card::{self, ModelDeploymentCard},
    model_type::ModelType,
};

/// [ModelEntry] contains the information to discover models from the etcd cluster.
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelEntry {
    /// Public name of the model
    /// Used to identify the model in the HTTP service from the value used in an OpenAI ChatRequest.
    pub name: String,

    /// How to address this on the network
    #[serde(rename = "endpoint")]
    pub endpoint_id: protocols::EndpointId,

    /// Specifies whether the model is a chat, completions, etc model.
    pub model_type: ModelType,

    /// Runtime configuration specific to this model instance
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_config: Option<ModelRuntimeConfig>,
}

impl ModelEntry {
    /// Slugified display name for use in etcd and NATS
    pub fn slug(&self) -> Slug {
        Slug::from_string(&self.name)
    }

    pub fn requires_preprocessing(&self) -> bool {
        matches!(self.model_type, ModelType::Backend)
    }

    /// Fetch the ModelDeploymentCard from etcd.
    /// This does not touch its fields so you may need to call move_from_nats on it.
    pub async fn load_mdc(
        &self,
        etcd_client: &etcd::Client,
    ) -> anyhow::Result<ModelDeploymentCard> {
        let kvstore: Box<dyn KeyValueStore> = Box::new(EtcdStorage::new(etcd_client.clone()));
        let card_store = Arc::new(KeyValueStoreManager::new(kvstore));
        let card_key = self.slug();
        tracing::info!("ModelEntry::load_mdc - Loading MDC from etcd with key: {}, model name: {}", card_key, self.name);

        // Debug: Let's fetch the raw JSON first to inspect it
        let etcd_key = format!("{}/{}", model_card::ROOT_PATH, card_key);
        match etcd_client
            .kv_get(etcd_key.as_str(), None)
            .await
        {
            Ok(kv_list) => {
                if let Some(first_kv) = kv_list.first() {
                    let raw_json = String::from_utf8_lossy(first_kv.value());
                    tracing::info!("Raw MDC JSON from etcd (first 500 chars): {}",
                        &raw_json.chars().take(500).collect::<String>());
                    if raw_json.contains("custom_chat_template") {
                        tracing::info!("Raw JSON contains 'custom_chat_template' field");
                    } else {
                        tracing::warn!("Raw JSON does NOT contain 'custom_chat_template' field!");
                    }
                } else {
                    tracing::warn!("No KV entries found in etcd for this key");
                }
            }
            Err(e) => {
                tracing::warn!("Could not fetch raw MDC JSON from etcd: {}", e);
            }
        }

        match card_store
            .load::<ModelDeploymentCard>(model_card::ROOT_PATH, &card_key)
            .await
        {
            Ok(Some(mdc)) => {
                tracing::info!("ModelEntry::load_mdc - Successfully loaded MDC. Has custom_chat_template: {}", mdc.custom_chat_template.is_some());
                Ok(mdc)
            }
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
