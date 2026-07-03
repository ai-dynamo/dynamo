// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::DiscoveryInstance;
use dynamo_runtime::discovery::DiscoverySpec;
use dynamo_runtime::protocols::EndpointId;
use dynamo_runtime::slug::Slug;
use dynamo_runtime::traits::DistributedRuntimeProvider;
use modelexpress_common::providers::{HuggingFaceProvider, ModelProviderTrait as _};

use crate::common::checked_file::CheckedFile;
use crate::entrypoint::RouterConfig;
use crate::frontend_config::{AdmissionGateConfig, FrontendApiConfig, MetricsConfig};
use crate::model_card::{ModelDeploymentCard, is_weight_file};
use crate::model_type::{ModelInput, ModelType};
use crate::preprocessor::media::{MediaDecoder, MediaFetcher};
use crate::request_template::RequestTemplate;

pub mod runtime_config;

use runtime_config::{ModelRuntimeConfig, TokenizerBackend};

/// What we call a model if the user didn't provide a name. Usually this means the name
/// is invisible, for example in a text chat.
const DEFAULT_NAME: &str = "dynamo";

/// Engines don't usually provide a default, so we do.
const DEFAULT_KV_CACHE_BLOCK_SIZE: u32 = 16;

/// We can't have it default to 0, so pick something
/// 'pub' because the bindings use it for consistency.
pub const DEFAULT_HTTP_PORT: u16 = 8080;

/// Default for `LocalModelBuilder::self_host_metadata`. Truthy values opt in.
pub const ENV_SELF_HOST_METADATA: &str = "DYN_SELF_HOST_METADATA";

fn env_self_host_metadata_default() -> bool {
    let value = std::env::var(ENV_SELF_HOST_METADATA).ok();
    self_host_metadata_default(value.as_deref())
}

fn self_host_metadata_default(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub struct LocalModelBuilder {
    model_path: Option<PathBuf>,
    source_path: Option<PathBuf>,
    model_name: Option<String>,
    model_aliases: Vec<String>,
    endpoint_id: Option<EndpointId>,
    template_file: Option<PathBuf>,
    router_config: Option<RouterConfig>,
    kv_cache_block_size: u32,
    http_host: Option<String>,
    http_port: u16,
    http_metrics_port: Option<u16>,
    metrics_config: MetricsConfig,
    frontend_api_config: FrontendApiConfig,
    admission_gate_config: AdmissionGateConfig,
    tls_cert_path: Option<PathBuf>,
    tls_key_path: Option<PathBuf>,
    migration_limit: u32,
    migration_max_seq_len: Option<u32>,
    is_mocker: bool,
    extra_engine_args: Option<PathBuf>,
    runtime_config: ModelRuntimeConfig,
    self_host_metadata: bool,
    user_data: Option<serde_json::Value>,
    custom_template_path: Option<PathBuf>,
    namespace: Option<String>,
    namespace_prefix: Option<String>,
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
            metrics_config: Default::default(),
            frontend_api_config: Default::default(),
            admission_gate_config: Default::default(),
            tls_cert_path: Default::default(),
            tls_key_path: Default::default(),
            model_path: Default::default(),
            source_path: Default::default(),
            model_name: Default::default(),
            model_aliases: Default::default(),
            endpoint_id: Default::default(),
            template_file: Default::default(),
            router_config: Default::default(),
            migration_limit: Default::default(),
            migration_max_seq_len: Default::default(),
            is_mocker: Default::default(),
            extra_engine_args: Default::default(),
            runtime_config: Default::default(),
            self_host_metadata: env_self_host_metadata_default(),
            user_data: Default::default(),
            custom_template_path: Default::default(),
            namespace: Default::default(),
            namespace_prefix: Default::default(),
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

    pub fn model_aliases(&mut self, aliases: Vec<String>) -> &mut Self {
        self.model_aliases = aliases;
        self
    }

    pub fn endpoint_id(&mut self, endpoint_id: Option<EndpointId>) -> &mut Self {
        self.endpoint_id = endpoint_id;
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

    pub fn metrics_prefix(&mut self, prefix: Option<String>) -> &mut Self {
        self.metrics_config.set_prefix(prefix);
        self
    }

    pub fn metrics_config(&mut self, metrics_config: MetricsConfig) -> &mut Self {
        self.metrics_config = metrics_config;
        self
    }

    pub fn frontend_api_config(&mut self, frontend_api_config: FrontendApiConfig) -> &mut Self {
        self.frontend_api_config = frontend_api_config;
        self
    }

    pub fn admission_gate_config(
        &mut self,
        admission_gate_config: AdmissionGateConfig,
    ) -> &mut Self {
        self.admission_gate_config = admission_gate_config;
        self
    }

    pub fn enable_anthropic_api(&mut self, enabled: bool) -> &mut Self {
        self.frontend_api_config
            .anthropic_mut()
            .set_enabled(enabled);
        self
    }

    pub fn strip_anthropic_preamble(&mut self, enabled: bool) -> &mut Self {
        self.frontend_api_config
            .anthropic_mut()
            .set_strip_preamble(enabled);
        self
    }

    pub fn enable_streaming_tool_dispatch(&mut self, enabled: bool) -> &mut Self {
        self.frontend_api_config
            .streaming_dispatch_mut()
            .set_tool_dispatch(enabled);
        self
    }

    pub fn enable_streaming_reasoning_dispatch(&mut self, enabled: bool) -> &mut Self {
        self.frontend_api_config
            .streaming_dispatch_mut()
            .set_reasoning_dispatch(enabled);
        self
    }

    /// Opt in or out of self-hosting MDC artifacts. Default `false`.
    /// Set this at runtime with environment variable DYN_SELF_HOST_METADATA.
    pub fn self_host_metadata(&mut self, enabled: bool) -> &mut Self {
        self.self_host_metadata = enabled;
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

    pub fn namespace_prefix(&mut self, namespace_prefix: Option<String>) -> &mut Self {
        self.namespace_prefix = namespace_prefix;
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

    pub fn migration_max_seq_len(&mut self, max_seq_len: Option<u32>) -> &mut Self {
        self.migration_max_seq_len = max_seq_len;
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

    pub fn tokenizer_backend(&mut self, tokenizer_backend: Option<TokenizerBackend>) -> &mut Self {
        if let Some(tokenizer_backend) = tokenizer_backend {
            self.runtime_config.tokenizer_backend = Some(tokenizer_backend);
        }
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

        // Pick up a stable routing id from `DYN_STABLE_ROUTING_ID`. No-op if the caller
        // already supplied one or the env var is unset. Published in etcd so routing
        // layers can keep cache assignments stable across worker restarts.
        self.runtime_config.populate_stable_routing_id_from_env();
        self.runtime_config
            .validate_config()
            .map_err(anyhow::Error::msg)?;
        self.runtime_config.add_topology_taints();

        let template = self
            .template_file
            .as_deref()
            .map(RequestTemplate::load)
            .transpose()?;

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
            card.router_config = self.router_config.clone();
            if !self.model_aliases.is_empty() {
                card.set_aliases(self.model_aliases.clone());
            }

            return Ok(LocalModel {
                card,
                full_path: PathBuf::new(),
                endpoint_id,
                template,
                http_host: self.http_host.take(),
                http_port: self.http_port,
                http_metrics_port: self.http_metrics_port,
                metrics_config: self.metrics_config.clone(),
                frontend_api_config: self.frontend_api_config.clone(),
                admission_gate_config: self.admission_gate_config.clone(),
                tls_cert_path: self.tls_cert_path.take(),
                tls_key_path: self.tls_key_path.take(),
                router_config: self.router_config.take().unwrap_or_default(),
                runtime_config: self.runtime_config.clone(),
                namespace: self.namespace.clone(),
                namespace_prefix: self.namespace_prefix.clone(),
                migration_limit: self.migration_limit,
                migration_max_seq_len: self.migration_max_seq_len,
                self_host_metadata: self.self_host_metadata,
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

        card.migration_limit = self.migration_limit;
        card.user_data = self.user_data.take();
        card.runtime_config = self.runtime_config.clone();
        card.media_decoder = self.media_decoder.clone();
        card.media_fetcher = self.media_fetcher.clone();
        card.router_config = self.router_config.clone();
        if !self.model_aliases.is_empty() {
            card.set_aliases(self.model_aliases.clone());
        }

        Ok(LocalModel {
            card,
            full_path: model_path,
            endpoint_id,
            template,
            http_host: self.http_host.take(),
            http_port: self.http_port,
            http_metrics_port: self.http_metrics_port,
            metrics_config: self.metrics_config.clone(),
            frontend_api_config: self.frontend_api_config.clone(),
            admission_gate_config: self.admission_gate_config.clone(),
            tls_cert_path: self.tls_cert_path.take(),
            tls_key_path: self.tls_key_path.take(),
            router_config: self.router_config.take().unwrap_or_default(),
            runtime_config: self.runtime_config.clone(),
            namespace: self.namespace.clone(),
            namespace_prefix: self.namespace_prefix.clone(),
            migration_limit: self.migration_limit,
            migration_max_seq_len: self.migration_max_seq_len,
            self_host_metadata: self.self_host_metadata,
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
    metrics_config: MetricsConfig,
    frontend_api_config: FrontendApiConfig,
    admission_gate_config: AdmissionGateConfig,
    tls_cert_path: Option<PathBuf>,
    tls_key_path: Option<PathBuf>,
    router_config: RouterConfig,
    runtime_config: ModelRuntimeConfig,
    namespace: Option<String>,
    namespace_prefix: Option<String>,
    migration_limit: u32,
    migration_max_seq_len: Option<u32>,
    self_host_metadata: bool,
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

    pub fn metrics_prefix(&self) -> Option<String> {
        self.metrics_config.prefix()
    }

    pub fn metrics_config(&self) -> &MetricsConfig {
        &self.metrics_config
    }

    pub fn frontend_api_config(&self) -> &FrontendApiConfig {
        &self.frontend_api_config
    }

    pub fn admission_gate_config(&self) -> &AdmissionGateConfig {
        &self.admission_gate_config
    }

    pub fn enable_anthropic_api(&self) -> bool {
        self.frontend_api_config.anthropic().enabled()
    }

    pub fn strip_anthropic_preamble(&self) -> bool {
        self.frontend_api_config.anthropic().strip_preamble()
    }

    pub fn enable_streaming_tool_dispatch(&self) -> bool {
        self.frontend_api_config
            .streaming_dispatch()
            .tool_dispatch()
    }

    pub fn enable_streaming_reasoning_dispatch(&self) -> bool {
        self.frontend_api_config
            .streaming_dispatch()
            .reasoning_dispatch()
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

    pub fn migration_limit(&self) -> u32 {
        self.migration_limit
    }

    pub fn migration_max_seq_len(&self) -> Option<u32> {
        self.migration_max_seq_len
    }

    pub fn namespace(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    pub fn namespace_prefix(&self) -> Option<&str> {
        self.namespace_prefix.as_deref()
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
    ///
    /// `worker_type` and `needs` carry the model-serving-readiness fields.
    /// Cards without a declared `worker_type` are treated as misconfigured
    /// and do not count toward readiness.
    pub async fn attach(
        &mut self,
        endpoint: &Endpoint,
        model_type: ModelType,
        model_input: ModelInput,
        lora_info: Option<crate::model_card::LoraInfo>,
        worker_type: Option<crate::worker_type::WorkerType>,
        needs: Vec<Vec<crate::worker_type::WorkerType>>,
    ) -> anyhow::Result<()> {
        self.card.model_type = model_type;
        self.card.model_input = model_input;
        self.card.worker_type = worker_type;
        self.card.needs = needs;
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

        if self.self_host_metadata {
            self.move_to_self_host(endpoint, model_suffix.as_deref())
                .context("move_to_self_host")?;
        }

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

    /// Local-path slots register in the registry and get rewritten to
    /// `http://<worker>/v1/metadata/<slug>/<suffix>/<filename>`. URL slots
    /// (`hf://`, etc.) are left alone so existing transports keep working.
    /// `model_suffix` is the LoRA slug, or `None` for the base model
    /// (recorded as `BASE_SUFFIX` in the registry).
    fn move_to_self_host(
        &mut self,
        endpoint: &Endpoint,
        model_suffix: Option<&str>,
    ) -> anyhow::Result<()> {
        let drt = endpoint.drt();
        let namespace = endpoint.component().namespace().name().to_string();
        let component = endpoint.component().name().to_string();
        let endpoint_name = endpoint.name().to_string();
        let Some(base_url) = self_host_base_url(drt)? else {
            tracing::warn!(
                model_slug = %self.card.slug(),
                "self_host_metadata enabled but system_status_server is not \
                 running (DYN_SYSTEM_PORT unset); skipping http rewrites — \
                 set DYN_SYSTEM_PORT to enable",
            );
            return Ok(());
        };
        let model_slug = self.card.slug().to_string();
        let suffix = model_suffix.unwrap_or(dynamo_runtime::metadata_registry::BASE_SUFFIX);
        let registry = drt.metadata_artifacts();
        let instance_id = drt.connection_id();
        let owner = (instance_id, model_suffix.map(str::to_string));

        // Advertise non-typed siblings (preprocessor_config.json,
        // special_tokens_map.json, …) so external preprocessors that load
        // via `from_pretrained(slug_dir)` see a complete model dir.
        let typed_filenames: HashSet<String> = self
            .card
            .iter_metadata_files()
            .iter()
            .filter_map(|(cf, _)| {
                cf.path()
                    .and_then(Path::file_name)
                    .and_then(|n| n.to_str())
                    .map(str::to_string)
            })
            .collect();
        let harvested =
            harvest_extra_files(&self.full_path, &typed_filenames).with_context(|| {
                format!(
                    "harvesting extra metadata files from {}",
                    self.full_path.display()
                )
            })?;
        self.card.extra_files.extend(harvested);

        let mut rewritten = 0usize;
        for (cf, _) in self.card.iter_metadata_files_mut() {
            let Some(local_path) = cf.path().map(Path::to_path_buf) else {
                continue;
            };
            let Some(filename) = local_path
                .file_name()
                .and_then(|f| f.to_str())
                .map(|s| s.to_string())
            else {
                continue;
            };
            let absolute = match std::path::absolute(&local_path) {
                Ok(p) => p,
                Err(err) => {
                    tracing::warn!(
                        path = %local_path.display(),
                        %err,
                        "failed to absolutize self-host metadata path; skipping",
                    );
                    continue;
                }
            };

            let url = url::Url::parse(&format!(
                "{base_url}/v1/metadata/{namespace}/{component}/{endpoint_name}/{model_slug}/{suffix}/{filename}"
            ))?;
            registry
                .register(
                    &owner,
                    &namespace,
                    &component,
                    &endpoint_name,
                    &model_slug,
                    suffix,
                    &filename,
                    absolute,
                )
                .context("registering metadata artifact")?;
            cf.move_to_url(url);
            rewritten += 1;
        }

        tracing::debug!(
            model_slug,
            suffix,
            rewritten,
            base_url,
            "self-hosting model metadata artifacts"
        );
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

        let model_suffix = lora_name.map(|name| Slug::slugify(name).to_string());
        let registry_owner = (instance_id, model_suffix.clone());

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
        drt.metadata_artifacts()
            .unregister_for_owner(&registry_owner);

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

/// `None` when `system_status_server` isn't running (no `DYN_SYSTEM_PORT`)
/// — lets default-on behavior degrade gracefully without erroring.
pub(crate) fn self_host_base_url(
    drt: &dynamo_runtime::DistributedRuntime,
) -> anyhow::Result<Option<String>> {
    let Some(info) = drt.system_status_server_info() else {
        return Ok(None);
    };

    let configured = dynamo_runtime::RuntimeConfig::from_settings()
        .unwrap_or_default()
        .system_host;
    let host = match configured.as_str() {
        "0.0.0.0" | "::" | "[::]" => dynamo_runtime::utils::local_ip_for_advertise(),
        _ => configured,
    };

    Ok(Some(format!("http://{host}:{}", info.port())))
}

/// Scan `model_dir` for files to advertise alongside the typed MDC slots.
/// Skips weights, dotfiles / README (`is_ignored`), already-typed
/// filenames, and anything that isn't a regular file. Non-recursive.
/// Returns an empty vec when `model_dir` doesn't exist (e.g. name-only
/// `LocalModel` placeholders).
fn harvest_extra_files(
    model_dir: &Path,
    typed_filenames: &HashSet<String>,
) -> anyhow::Result<Vec<CheckedFile>> {
    if !model_dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in
        fs::read_dir(model_dir).with_context(|| format!("read_dir {}", model_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        // `Path::is_file` uses `fs::metadata` which follows symlinks.
        // `entry.file_type()` / `entry.metadata()` use lstat on Unix —
        // they would skip the HF blob symlinks we need to harvest.
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if typed_filenames.contains(name)
            || is_weight_file(&path)
            || HuggingFaceProvider::is_ignored(name)
        {
            continue;
        }
        out.push(CheckedFile::from_disk(&path)?);
    }
    Ok(out)
}

#[cfg(test)]
mod env_self_host_metadata_tests {
    use super::*;

    #[test]
    fn env_default_parsing() {
        assert!(!self_host_metadata_default(None), "unset → default OFF");

        for v in [
            "0", "false", "FALSE", "no", "NO", "off", "OFF", "", "garbage",
        ] {
            assert!(
                !self_host_metadata_default(Some(v)),
                "expected OFF for {v:?}"
            );
        }
        for v in ["1", "true", "TRUE", "yes", "Yes", "on", "ON"] {
            assert!(self_host_metadata_default(Some(v)), "expected ON for {v:?}");
        }
    }
}

#[cfg(test)]
mod harvest_extra_files_tests {
    use super::*;

    #[test]
    fn filters_weights_typed_and_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let touch = |name: &str| {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        };
        // typed slot (excluded by name)
        touch("config.json");
        // weights — covers the mx-narrow case (safetensors) and the
        // ecosystem-wide case (.pt) which mx alone wouldn't catch.
        touch("model.safetensors");
        touch("pytorch_lora_weights.pt");
        // dotfile / README (excluded by is_ignored)
        touch(".gitattributes");
        touch("README.md");
        // genuine extras (kept)
        touch("preprocessor_config.json");
        touch("special_tokens_map.json");
        // subdir (skipped — not a file)
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        // symlink to a real file (must be followed — HF snapshot dirs
        // are full of these pointing into blobs/)
        let target = dir.path().join("target_for_symlink");
        std::fs::write(&target, b"x").unwrap();
        std::os::unix::fs::symlink(&target, dir.path().join("added_tokens.json")).unwrap();

        let typed: HashSet<String> = ["config.json".to_string()].into_iter().collect();
        let mut names: Vec<String> = harvest_extra_files(dir.path(), &typed)
            .unwrap()
            .iter()
            .map(|cf| {
                cf.path()
                    .unwrap()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "added_tokens.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "target_for_symlink",
            ]
        );
    }

    #[test]
    fn returns_empty_for_missing_dir() {
        let typed: HashSet<String> = HashSet::new();
        // bogus path: doesn't exist on disk; must not error
        let result = harvest_extra_files(Path::new("/nonexistent/dynamo/test/path"), &typed);
        assert!(result.unwrap().is_empty());
    }
}
