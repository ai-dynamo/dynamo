// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use dynamo_runtime::protocols::Endpoint as EndpointId;
use dynamo_runtime::slug::Slug;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::{
    component::{Component, Endpoint},
    storage::key_value_store::{EtcdStorage, KeyValueStore, KeyValueStoreManager},
};

use crate::discovery::ModelEntry;
use crate::entrypoint::RouterConfig;
use crate::model_card::{self, ModelDeploymentCard};
use crate::model_type::ModelType;
use crate::request_template::RequestTemplate;

mod network_name;
pub use network_name::ModelNetworkName;

/// Prefix for Hugging Face model repository
const HF_SCHEME: &str = "hf://";

/// What we call a model if the user didn't provide a name. Usually this means the name
/// is invisible, for example in a text chat.
const DEFAULT_NAME: &str = "dynamo";

/// Engines don't usually provide a default, so we do.
const DEFAULT_KV_CACHE_BLOCK_SIZE: u32 = 16;

/// We can't have it default to 0, so pick something
const DEFAULT_HTTP_PORT: u16 = 8080;

pub struct LocalModelBuilder {
    model_path: Option<PathBuf>,
    model_name: Option<String>,
    model_config: Option<PathBuf>,
    endpoint_id: Option<EndpointId>,
    context_length: Option<u32>,
    template_file: Option<PathBuf>,
    router_config: Option<RouterConfig>,
    kv_cache_block_size: u32,
    http_port: u16,
    migration_limit: u32,
    is_mocker: bool,
    user_data: Option<serde_json::Value>,
}

impl Default for LocalModelBuilder {
    fn default() -> Self {
        LocalModelBuilder {
            kv_cache_block_size: DEFAULT_KV_CACHE_BLOCK_SIZE,
            http_port: DEFAULT_HTTP_PORT,
            model_path: Default::default(),
            model_name: Default::default(),
            model_config: Default::default(),
            endpoint_id: Default::default(),
            context_length: Default::default(),
            template_file: Default::default(),
            router_config: Default::default(),
            migration_limit: Default::default(),
            is_mocker: Default::default(),
            user_data: Default::default(),
        }
    }
}

impl LocalModelBuilder {
    pub fn model_path(&mut self, model_path: Option<PathBuf>) -> &mut Self {
        self.model_path = model_path;
        self
    }

    pub fn model_name(&mut self, model_name: Option<String>) -> &mut Self {
        self.model_name = model_name;
        self
    }

    pub fn model_config(&mut self, model_config: Option<PathBuf>) -> &mut Self {
        self.model_config = model_config;
        self
    }

    pub fn endpoint_id(&mut self, endpoint_id: Option<EndpointId>) -> &mut Self {
        self.endpoint_id = endpoint_id;
        self
    }

    pub fn context_length(&mut self, context_length: Option<u32>) -> &mut Self {
        self.context_length = context_length;
        self
    }

    /// Passing None resets it to default
    pub fn kv_cache_block_size(&mut self, kv_cache_block_size: Option<u32>) -> &mut Self {
        self.kv_cache_block_size = kv_cache_block_size.unwrap_or(DEFAULT_KV_CACHE_BLOCK_SIZE);
        self
    }

    /// Passing None resets it to default
    pub fn http_port(&mut self, port: Option<u16>) -> &mut Self {
        self.http_port = port.unwrap_or(DEFAULT_HTTP_PORT);
        self
    }

    pub fn router_config(&mut self, router_config: Option<RouterConfig>) -> &mut Self {
        self.router_config = router_config;
        self
    }

    pub fn request_template(&mut self, template_file: Option<PathBuf>) -> &mut Self {
        self.template_file = template_file;
        self
    }

    pub fn migration_limit(&mut self, migration_limit: Option<u32>) -> &mut Self {
        self.migration_limit = migration_limit.unwrap_or(0);
        self
    }

    pub fn is_mocker(&mut self, is_mocker: bool) -> &mut Self {
        self.is_mocker = is_mocker;
        self
    }

    pub fn user_data(&mut self, user_data: Option<serde_json::Value>) -> &mut Self {
        self.user_data = user_data;
        self
    }

    /// Make an LLM ready for use:
    /// - Download it from Hugging Face (and NGC in future) if necessary
    /// - Resolve the path
    /// - Load it's ModelDeploymentCard card
    /// - Name it correctly
    ///
    /// The model name will depend on what "model_path" is:
    /// - A folder: The last part of the folder name: "/data/llms/Qwen2.5-3B-Instruct" -> "Qwen2.5-3B-Instruct"
    /// - A file: The GGUF filename: "/data/llms/Qwen2.5-3B-Instruct-Q6_K.gguf" -> "Qwen2.5-3B-Instruct-Q6_K.gguf"
    /// - An HF repo: The HF repo name: "Qwen/Qwen3-0.6B" stays the same
    pub async fn build(&mut self) -> anyhow::Result<LocalModel> {
        // Generate an endpoint ID for this model if the user didn't provide one.
        // The user only provides one if exposing the model.
        let endpoint_id = self
            .endpoint_id
            .take()
            .unwrap_or_else(|| internal_endpoint("local_model"));
        let template = self
            .template_file
            .as_deref()
            .map(RequestTemplate::load)
            .transpose()?;

        // echo_full engine doesn't need a path. It's an edge case, move it out of the way.
        if self.model_path.is_none() {
            let mut card = ModelDeploymentCard::with_name_only(
                self.model_name.as_deref().unwrap_or(DEFAULT_NAME),
            );
            card.migration_limit = self.migration_limit;
            card.user_data = self.user_data.take();
            return Ok(LocalModel {
                card,
                full_path: PathBuf::new(),
                endpoint_id,
                template,
                http_port: self.http_port,
                router_config: self.router_config.take().unwrap_or_default(),
            });
        }

        // Main logic. We are running a model.
        let model_path = self.model_path.take().unwrap();
        let model_path = model_path.to_str().context("Invalid UTF-8 in model path")?;

        // Check for hf:// prefix first, in case we really want an HF repo but it conflicts
        // with a relative path.
        let is_hf_repo =
            model_path.starts_with(HF_SCHEME) || !fs::exists(model_path).unwrap_or(false);
        let relative_path = model_path.trim_start_matches(HF_SCHEME);
        let full_path = if is_hf_repo {
            // HF download if necessary
            super::hub::from_hf(relative_path, self.is_mocker).await?
        } else {
            fs::canonicalize(relative_path)?
        };
        // --model-config takes precedence over --model-path
        let model_config_path = self.model_config.as_ref().unwrap_or(&full_path);

        let mut card = ModelDeploymentCard::load(&model_config_path).await?;

        // Usually we infer from the path, self.model_name is user override
        let model_name = self.model_name.take().unwrap_or_else(|| {
            if is_hf_repo {
                // HF repos use their full name ("org/name") not the folder name
                relative_path.to_string()
            } else {
                full_path
                    .iter()
                    .next_back()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| {
                        // Panic because we can't do anything without a model
                        panic!("Invalid model path, too short: '{}'", full_path.display())
                    })
            }
        });
        card.set_name(&model_name);

        card.kv_cache_block_size = self.kv_cache_block_size;

        // Override max number of tokens in context. We usually only do this to limit kv cache allocation.
        if let Some(context_length) = self.context_length {
            card.context_length = context_length;
        }

        card.migration_limit = self.migration_limit;
        card.user_data = self.user_data.take();

        Ok(LocalModel {
            card,
            full_path,
            endpoint_id,
            template,
            http_port: self.http_port,
            router_config: self.router_config.take().unwrap_or_default(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct LocalModel {
    full_path: PathBuf,
    card: ModelDeploymentCard,
    endpoint_id: EndpointId,
    template: Option<RequestTemplate>,
    http_port: u16, // Only used if input is HTTP server
    router_config: RouterConfig,
}

impl LocalModel {
    pub fn card(&self) -> &ModelDeploymentCard {
        &self.card
    }

    pub fn path(&self) -> &Path {
        &self.full_path
    }

    pub fn display_name(&self) -> &str {
        &self.card.display_name
    }

    pub fn service_name(&self) -> &str {
        &self.card.service_name
    }

    pub fn request_template(&self) -> Option<RequestTemplate> {
        self.template.clone()
    }

    pub fn http_port(&self) -> u16 {
        self.http_port
    }

    pub fn router_config(&self) -> &RouterConfig {
        &self.router_config
    }

    pub fn is_gguf(&self) -> bool {
        // GGUF is the only file (not-folder) we accept, so we don't need to check the extension
        // We will error when we come to parse it
        self.full_path.is_file()
    }

    /// An endpoint to identify this model by.
    pub fn endpoint_id(&self) -> &EndpointId {
        &self.endpoint_id
    }

    /// Drop the LocalModel returning it's ModelDeploymentCard.
    /// For the case where we only need the card and don't want to clone it.
    pub fn into_card(self) -> ModelDeploymentCard {
        self.card
    }

    /// Attach this model the endpoint. This registers it on the network
    /// allowing ingress to discover it.
    pub async fn attach(
        &mut self,
        endpoint: &Endpoint,
        model_type: ModelType,
    ) -> anyhow::Result<()> {
        // A static component doesn't have an etcd_client because it doesn't need to register
        let Some(etcd_client) = endpoint.drt().etcd_client() else {
            anyhow::bail!("Cannot attach to static endpoint");
        };
        self.ensure_unique(endpoint.component(), self.display_name())
            .await?;

        // Store model config files in NATS object store
        let nats_client = endpoint.drt().nats_client();
        self.card.move_to_nats(nats_client.clone()).await?;

        // Publish the Model Deployment Card to etcd
        let kvstore: Box<dyn KeyValueStore> = Box::new(EtcdStorage::new(etcd_client.clone()));
        let card_store = Arc::new(KeyValueStoreManager::new(kvstore));
        let key = self.card.slug().to_string();
        card_store
            .publish(model_card::ROOT_PATH, None, &key, &mut self.card)
            .await?;

        // Publish our ModelEntry to etcd. This allows ingress to find the model card.
        // (Why don't we put the model card directly under this key?)
        let network_name = ModelNetworkName::from_local(endpoint, etcd_client.lease_id());
        tracing::debug!("Registering with etcd as {network_name}");
        let model_registration = ModelEntry {
            name: self.display_name().to_string(),
            endpoint: endpoint.id(),
            model_type,
        };
        etcd_client
            .kv_create(
                network_name.to_string(),
                serde_json::to_vec_pretty(&model_registration)?,
                None, // use primary lease
            )
            .await
    }

    /// Ensure that each component serves only one model.
    /// We can have multiple instances of the same model running using the same component name
    /// (they get load balanced, and are differentiated in etcd by their lease_id).
    /// We cannot have multiple models with the same component name.
    ///
    /// Returns an error if there is already a component by this name serving a different model.
    async fn ensure_unique(&self, component: &Component, model_name: &str) -> anyhow::Result<()> {
        let Some(etcd_client) = component.drt().etcd_client() else {
            // A static component is necessarily unique, it cannot register
            return Ok(());
        };
        for endpoint_info in component.list_instances().await? {
            let network_name: ModelNetworkName = (&endpoint_info).into();

            if let Ok(entry) = network_name.load_entry(&etcd_client).await {
                if entry.name != model_name {
                    anyhow::bail!("Duplicate component. Attempt to register model {model_name} at {component}, which is already used by {network_name} running model {}.", entry.name);
                }
            }
        }
        Ok(())
    }
}

/// A random endpoint to use for internal communication
/// We can't hard code because we may be running several on the same machine (GPUs 0-3 and 4-7)
fn internal_endpoint(engine: &str) -> EndpointId {
    EndpointId {
        namespace: Slug::slugify(&uuid::Uuid::new_v4().to_string()).to_string(),
        component: engine.to_string(),
        name: "generate".to_string(),
    }
}
