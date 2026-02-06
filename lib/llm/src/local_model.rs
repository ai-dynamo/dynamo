// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::DiscoveryInstance;
use dynamo_runtime::discovery::DiscoverySpec;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::slug::Slug;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use dynamo_runtime::utils::get_http_rpc_host_from_env;

use crate::entrypoint::RouterConfig;
use crate::mocker::protocols::{MockEngineArgs, WorkerType};
use crate::model_card::ModelDeploymentCard;
use crate::model_type::{ModelInput, ModelType};
use crate::preprocessor::media::{ImageDecoder, MediaDecoder, MediaFetcher};
use crate::request_template::RequestTemplate;

pub mod runtime_config;

use runtime_config::ModelRuntimeConfig;

/// What we call a model if the user didn't provide a name. Usually this means the name
/// is invisible, for example in a text chat.
const DEFAULT_NAME: &str = "dynamo";

/// Engines don't usually provide a default, so we do.
const DEFAULT_KV_CACHE_BLOCK_SIZE: u32 = 16;

/// We can't have it default to 0, so pick something
/// 'pub' because the bindings use it for consistency.
pub const DEFAULT_HTTP_PORT: u16 = 8080;

pub struct LocalModelBuilder {
    model_path: Option<PathBuf>,
    source_path: Option<PathBuf>,
    model_name: Option<String>,
    endpoint_id: Option<EndpointId>,
    context_length: Option<u32>,
    template_file: Option<PathBuf>,
    router_config: Option<RouterConfig>,
    kv_cache_block_size: u32,
    http_host: Option<String>,
    http_port: u16,
    http_metrics_port: Option<u16>,
    tls_cert_path: Option<PathBuf>,
    tls_key_path: Option<PathBuf>,
    migration_limit: u32,
    is_mocker: bool,
    extra_engine_args: Option<PathBuf>,
    runtime_config: ModelRuntimeConfig,
    user_data: Option<serde_json::Value>,
    custom_template_path: Option<PathBuf>,
    namespace: Option<String>,
    media_decoder: Option<MediaDecoder>,
    media_fetcher: Option<MediaFetcher>,
}

impl Default for LocalModelBuilder {
    fn default() -> Self {
        LocalModelBuilder {
            kv_cache_block_size: DEFAULT_KV_CACHE_BLOCK_SIZE,
            http_host: Default::default(),
            http_port: DEFAULT_HTTP_PORT,
            http_metrics_port: None,
            tls_cert_path: Default::default(),
            tls_key_path: Default::default(),
            model_path: Default::default(),
            source_path: Default::default(),
            model_name: Default::default(),
            endpoint_id: Default::default(),
            context_length: Default::default(),
            template_file: Default::default(),
            router_config: Default::default(),
            migration_limit: Default::default(),
            is_mocker: Default::default(),
            extra_engine_args: Default::default(),
            runtime_config: Default::default(),
            user_data: Default::default(),
            custom_template_path: Default::default(),
            namespace: Default::default(),
            media_decoder: Default::default(),
            media_fetcher: Default::default(),
        }
    }
}

impl LocalModelBuilder {
    /// The path must exist, the model is already downloaded
    pub fn model_path(&mut self, model_path: PathBuf) -> &mut Self {
        self.model_path = Some(model_path);
        self
    }

    /// The HF name of the model before we downloaded it, or a local path if
    /// that was given on the cmd line. We need this because `model_path` is always
    /// a local path.
    pub fn source_path(&mut self, source_path: PathBuf) -> &mut Self {
        self.source_path = Some(source_path);
        self
    }

    pub fn model_name(&mut self, model_name: Option<String>) -> &mut Self {
        self.model_name = model_name;
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

    pub fn http_host(&mut self, host: Option<String>) -> &mut Self {
        self.http_host = host;
        self
    }

    pub fn http_port(&mut self, port: u16) -> &mut Self {
        self.http_port = port;
        self
    }

    pub fn http_metrics_port(&mut self, port: Option<u16>) -> &mut Self {
        self.http_metrics_port = port;
        self
    }

    pub fn tls_cert_path(&mut self, p: Option<PathBuf>) -> &mut Self {
        self.tls_cert_path = p;
        self
    }

    pub fn tls_key_path(&mut self, p: Option<PathBuf>) -> &mut Self {
        self.tls_key_path = p;
        self
    }

    pub fn router_config(&mut self, router_config: Option<RouterConfig>) -> &mut Self {
        self.router_config = router_config;
        self
    }

    pub fn namespace(&mut self, namespace: Option<String>) -> &mut Self {
        self.namespace = namespace;
        self
    }

    pub fn request_template(&mut self, template_file: Option<PathBuf>) -> &mut Self {
        self.template_file = template_file;
        self
    }

    pub fn custom_template_path(&mut self, custom_template_path: Option<PathBuf>) -> &mut Self {
        self.custom_template_path = custom_template_path;
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

    pub fn extra_engine_args(&mut self, extra_engine_args: Option<PathBuf>) -> &mut Self {
        self.extra_engine_args = extra_engine_args;
        self
    }

    pub fn runtime_config(&mut self, runtime_config: ModelRuntimeConfig) -> &mut Self {
        self.runtime_config = runtime_config;
        self
    }

    pub fn user_data(&mut self, user_data: Option<serde_json::Value>) -> &mut Self {
        self.user_data = user_data;
        self
    }

    pub fn media_decoder(&mut self, media_decoder: Option<MediaDecoder>) -> &mut Self {
        self.media_decoder = media_decoder;
        self
    }

    pub fn media_fetcher(&mut self, media_fetcher: Option<MediaFetcher>) -> &mut Self {
        self.media_fetcher = media_fetcher;
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

        // Override runtime configs with mocker engine args (applies to both paths)
        if self.is_mocker
            && let Some(path) = &self.extra_engine_args
        {
            let mocker_engine_args = MockEngineArgs::from_json_file(path)
                .expect("Failed to load mocker engine args for runtime config overriding.");
            self.kv_cache_block_size = mocker_engine_args.block_size as u32;
            self.runtime_config.total_kv_blocks = Some(mocker_engine_args.num_gpu_blocks as u64);
            self.runtime_config.max_num_seqs = mocker_engine_args.max_num_seqs.map(|v| v as u64);
            self.runtime_config.max_num_batched_tokens =
                mocker_engine_args.max_num_batched_tokens.map(|v| v as u64);
            self.runtime_config.enable_local_indexer = mocker_engine_args.enable_local_indexer;
            self.runtime_config.data_parallel_size = mocker_engine_args.dp_size;
            self.media_decoder = Some(MediaDecoder {
                image: Some(ImageDecoder::default()),
                #[cfg(feature = "media-ffmpeg")]
                video: None,
            });
            self.media_fetcher = Some(MediaFetcher::default());

            // Set bootstrap endpoint for prefill workers with bootstrap_port configured
            if mocker_engine_args.worker_type == WorkerType::Prefill
                && let Some(port) = mocker_engine_args.bootstrap_port
            {
                let host = get_http_rpc_host_from_env();
                self.runtime_config.disaggregated_endpoint =
                    Some(runtime_config::DisaggregatedEndpoint {
                        bootstrap_host: Some(host),
                        bootstrap_port: Some(port),
                    });
                tracing::info!(
                    bootstrap_port = port,
                    "Mocker prefill worker: publishing bootstrap endpoint to discovery"
                );
            }
        }

        // frontend and echo engine don't need a path.
        if self.model_path.is_none() {
            let mut card = ModelDeploymentCard::with_name_only(
                self.model_name.as_deref().unwrap_or(DEFAULT_NAME),
            );
            card.kv_cache_block_size = self.kv_cache_block_size;
            card.migration_limit = self.migration_limit;
            card.user_data = self.user_data.take();
            card.runtime_config = self.runtime_config.clone();
            card.media_decoder = self.media_decoder.clone();
            card.media_fetcher = self.media_fetcher.clone();

            return Ok(LocalModel {
                card,
                full_path: PathBuf::new(),
                endpoint_id,
                template,
                http_host: self.http_host.take(),
                http_port: self.http_port,
                http_metrics_port: self.http_metrics_port,
                tls_cert_path: self.tls_cert_path.take(),
                tls_key_path: self.tls_key_path.take(),
                router_config: self.router_config.take().unwrap_or_default(),
                runtime_config: self.runtime_config.clone(),
                namespace: self.namespace.clone(),
            });
        }

        // Main logic. We are running a model.
        let model_path = self.model_path.take().unwrap();
        if !model_path.exists() {
            anyhow::bail!(
                "Path does not exist: '{}'. Use LocalModel::fetch to download it.",
                model_path.display(),
            );
        }
        let model_path = fs::canonicalize(model_path)?;

        let mut card =
            ModelDeploymentCard::load_from_disk(&model_path, self.custom_template_path.as_deref())?;
        // Source path is the `--model-path` the user passed. By now our `model_path` is the local
        // path of the downloaded model.
        if let Some(source_path) = self.source_path.take() {
            card.set_source_path(source_path);
        }
        // The served model name defaults to the full model path.
        // This matches what vllm and sglang do.
        let alt = card.source_path().to_string();
        card.set_name(self.model_name.as_deref().unwrap_or(&alt));

        card.kv_cache_block_size = self.kv_cache_block_size;

        // Override max number of tokens in context. We usually only do this to limit kv cache allocation.
        if let Some(context_length) = self.context_length {
            card.context_length = context_length;
        }

        card.migration_limit = self.migration_limit;
        card.user_data = self.user_data.take();
        card.runtime_config = self.runtime_config.clone();
        card.media_decoder = self.media_decoder.clone();
        card.media_fetcher = self.media_fetcher.clone();

        Ok(LocalModel {
            card,
            full_path: model_path,
            endpoint_id,
            template,
            http_host: self.http_host.take(),
            http_port: self.http_port,
            http_metrics_port: self.http_metrics_port,
            tls_cert_path: self.tls_cert_path.take(),
            tls_key_path: self.tls_key_path.take(),
            router_config: self.router_config.take().unwrap_or_default(),
            runtime_config: self.runtime_config.clone(),
            namespace: self.namespace.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct LocalModel {
    full_path: PathBuf,
    card: ModelDeploymentCard,
    endpoint_id: EndpointId,
    template: Option<RequestTemplate>,
    http_host: Option<String>,
    http_port: u16,
    http_metrics_port: Option<u16>,
    tls_cert_path: Option<PathBuf>,
    tls_key_path: Option<PathBuf>,
    router_config: RouterConfig,
    runtime_config: ModelRuntimeConfig,
    namespace: Option<String>,
}

impl LocalModel {
    /// Ensure a model is accessible locally, returning it's path.
    /// Downloads the model from Hugging Face if necessary.
    /// If ignore_weights is true, model weight files will be skipped and only the model config
    /// will be downloaded.
    /// Returns the path to the model files
    pub async fn fetch(remote_name: &str, ignore_weights: bool) -> anyhow::Result<PathBuf> {
        super::hub::from_hf(remote_name, ignore_weights).await
    }

    pub fn card(&self) -> &ModelDeploymentCard {
        &self.card
    }

    pub fn path(&self) -> &Path {
        &self.full_path
    }

    /// Human friendly model name. This is the correct name.
    pub fn display_name(&self) -> &str {
        &self.card.display_name
    }

    /// The name under which we make this model available over HTTP.
    /// A slugified version of the model's name, for use in NATS, etcd, etc.
    pub fn service_name(&self) -> &str {
        self.card.slug().as_ref()
    }

    pub fn request_template(&self) -> Option<RequestTemplate> {
        self.template.clone()
    }

    pub fn http_host(&self) -> Option<String> {
        self.http_host.clone()
    }

    pub fn http_port(&self) -> u16 {
        self.http_port
    }

    pub fn http_metrics_port(&self) -> Option<u16> {
        self.http_metrics_port
    }

    pub fn tls_cert_path(&self) -> Option<&Path> {
        self.tls_cert_path.as_deref()
    }

    pub fn tls_key_path(&self) -> Option<&Path> {
        self.tls_key_path.as_deref()
    }

    pub fn router_config(&self) -> &RouterConfig {
        &self.router_config
    }

    pub fn runtime_config(&self) -> &ModelRuntimeConfig {
        &self.runtime_config
    }

    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
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

    /// Attach this model to the endpoint. This registers it on the network
    /// allowing ingress to discover it.
    ///
    /// For base models, pass `lora_name = None`.
    /// For LoRA adapters, pass `lora_name = Some("adapter-name")`.
    pub async fn attach(
        &mut self,
        endpoint: &Endpoint,
        model_type: ModelType,
        model_input: ModelInput,
        lora_info: Option<crate::model_card::LoraInfo>,
    ) -> anyhow::Result<()> {
        self.card.model_type = model_type;
        self.card.model_input = model_input;
        self.card.lora = lora_info.clone();

        // Compute model_suffix from lora_name if present
        let model_suffix = lora_info
            .as_ref()
            .map(|info| Slug::slugify(&info.name).to_string());

        let suffix_for_log = model_suffix
            .as_ref()
            .map(|s| format!("/{}", s))
            .unwrap_or_default();
        tracing::debug!(
            "Registering MDC at path: {}/{}/{}/{:x}{}",
            endpoint.component().namespace().name(),
            endpoint.component().name(),
            endpoint.name(),
            endpoint.drt().connection_id(),
            suffix_for_log
        );

        let source_path = PathBuf::from(self.card.source_path());
        if !source_path.exists() {
            // The consumers of MDC (frontend) might not have the same local path as us, so
            // replace disk paths with a custom URL like "hf://Qwen/Qwen3-0.6B/config.json".
            //
            // We can't do this if the model came from disk, as it might not be the same version
            // as on Hugging Face (if it exists there at all).
            //
            // The URL is not used by anything. Frontend will download the repo and edit these
            // paths to be local, so only the filename part matters currently.
            // Possibly we should just use the filenames here. The URL feels nicer to me, it makes
            // each field fully identified and fetchable independently.
            self.card
                .move_to_url(&format!("hf://{}/", self.card.source_path()))
                .context("move_to_url")?;
        }

        // Register the Model Deployment Card via discovery interface
        // The model_suffix (for LoRA) will be appended AFTER the instance_id
        let discovery = endpoint.drt().discovery();
        let spec = DiscoverySpec::from_model_with_suffix(
            endpoint.component().namespace().name().to_string(),
            endpoint.component().name().to_string(),
            endpoint.name().to_string(),
            &self.card,
            model_suffix,
        )?;
        let _instance = discovery.register(spec).await?;

        Ok(())
    }

    /// Helper associated function to detach a model from an endpoint
    ///
    /// For base models, pass `lora_name = None`.
    /// For LoRA adapters, pass `lora_name = Some("adapter-name")`.
    pub async fn detach_from_endpoint(
        endpoint: &Endpoint,
        lora_name: Option<&str>,
    ) -> anyhow::Result<()> {
        let drt = endpoint.drt();
        let instance_id = drt.connection_id();
        let endpoint_id = endpoint.id();

        // Compute model_suffix from lora_name if present
        let model_suffix = lora_name.map(|name| Slug::slugify(name).to_string());

        let instance = DiscoveryInstance::Model {
            namespace: endpoint_id.namespace,
            component: endpoint_id.component,
            endpoint: endpoint_id.name,
            instance_id,
            card_json: serde_json::Value::Null,
            model_suffix,
        };

        let discovery = drt.discovery();
        discovery.unregister(instance).await?;

        if let Some(lora_name) = lora_name {
            tracing::info!(
                "Successfully unregistered LoRA '{}' from discovery",
                lora_name
            );
        } else {
            tracing::info!("Successfully unregistered model from discovery");
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
