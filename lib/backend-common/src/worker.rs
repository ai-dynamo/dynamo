// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `Worker` — runtime lifecycle driver for an [`LLMEngine`].
//!
//! Creates the `DistributedRuntime`, starts the engine, registers the
//! model, serves the endpoint, and runs cleanup on shutdown. Non-generic
//! over the engine type so a PyO3-wrapped engine (phase 2) can feed in
//! through the same `Arc<dyn LLMEngine>` path.

use std::path::PathBuf;
use std::sync::Arc;

use dynamo_llm::local_model::LocalModel;
use dynamo_llm::local_model::LocalModelBuilder;
use dynamo_llm::local_model::runtime_config::ModelRuntimeConfig;
use dynamo_llm::model_type::{ModelInput, ModelType};
use dynamo_runtime::pipeline::network::Ingress;
use dynamo_runtime::{DistributedRuntime, Runtime};

use crate::adapter::EngineAdapter;
use crate::engine::{EngineConfig, LLMEngine};
use crate::error::{BackendError, DynamoError, ErrorType};

/// Per-worker runtime configuration.
#[derive(Clone, Debug)]
pub struct WorkerConfig {
    /// Dynamo namespace for discovery routing.
    pub namespace: String,
    /// Component name within the namespace.
    pub component: String,
    /// Endpoint name exposed by this worker (e.g. `"generate"`).
    pub endpoint: String,
    /// HF repo name or local model path. Empty means name-only registration
    /// (no tokenizer / chat-template on the card).
    pub model_name: String,
    /// Public-facing model name (operator CLI override). When unset, the
    /// served name falls back to `EngineConfig.served_model_name`, then to
    /// `EngineConfig.model`.
    pub served_model_name: Option<String>,
    /// Whether the engine consumes tokens (`Tokens`) or raw text (`Text`).
    pub model_input: ModelInput,
    /// Comma-separated list, e.g. `"chat,completions"`.
    /// Accepted values: `chat`, `completions`, `embedding`/`embeddings`,
    /// `tensor`, `prefill` (see `parse_endpoint_types`).
    pub endpoint_types: String,
    /// Optional path to a custom Jinja chat template. When `None`, the
    /// template shipped with `model_name` is used.
    pub custom_jinja_template: Option<PathBuf>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            endpoint: "generate".to_string(),
            model_name: String::new(),
            served_model_name: None,
            model_input: ModelInput::Tokens,
            endpoint_types: "chat,completions".to_string(),
            custom_jinja_template: None,
        }
    }
}

/// Runtime host for an [`LLMEngine`].
///
/// `run()` creates the distributed runtime, calls `engine.start()`,
/// registers the model, serves the endpoint, and calls
/// `engine.cleanup()` on shutdown (guaranteed once `start()` succeeded).
pub struct Worker {
    engine: Arc<dyn LLMEngine>,
    config: WorkerConfig,
}

impl Worker {
    pub fn new(engine: Arc<dyn LLMEngine>, config: WorkerConfig) -> Self {
        Self { engine, config }
    }

    /// Lifecycle driver. Takes owned `self` — `Worker` is single-shot and
    /// cannot be reused after `run()` returns.
    ///
    /// Shutdown behaviour: `endpoint_builder().graceful_shutdown(true)`
    /// is passed to `dynamo-runtime`, which drains in-flight requests on
    /// SIGTERM/SIGINT. The drain deadline is controlled by
    /// `dynamo-runtime` (not exposed here today); long-running requests
    /// may be forcibly terminated once that deadline elapses. Engines
    /// that hold external resources per request should ensure their
    /// `generate` stream body releases them via RAII so drop paths
    /// (including forced drops) are safe.
    pub async fn run(self, runtime: Runtime) -> Result<(), DynamoError> {
        let Self { engine, config } = self;

        let drt = DistributedRuntime::from_settings(runtime)
            .await
            .map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::CannotConnect),
                    format!("distributed runtime: {e}"),
                )
            })?;
        tracing::debug!("distributed runtime connected");

        let component = drt
            .namespace(&config.namespace)
            .and_then(|ns| ns.component(&config.component))
            .map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::CannotConnect),
                    format!("component: {e}"),
                )
            })?;
        let endpoint = component.endpoint(&config.endpoint);
        tracing::debug!(
            namespace = %config.namespace,
            component = %config.component,
            endpoint = %config.endpoint,
            "component and endpoint resolved"
        );

        // engine.start() returns a DynamoError directly — propagate as-is.
        let engine_config = engine.start().await?;
        tracing::debug!(model = %engine_config.model, "engine.start() complete");

        // Cleanup must run even when registration or serve fails, so wrap
        // post-start work in a helper whose result is propagated after cleanup.
        let serve_result = serve(&engine, &config, &engine_config, endpoint).await;

        if let Err(cleanup_err) = engine.cleanup().await {
            tracing::error!(error = %cleanup_err, "engine cleanup failed");
        } else {
            tracing::info!("Engine cleanup complete");
        }

        serve_result
    }
}

async fn serve(
    engine: &Arc<dyn LLMEngine>,
    config: &WorkerConfig,
    engine_config: &EngineConfig,
    endpoint: dynamo_runtime::component::Endpoint,
) -> Result<(), DynamoError> {
    let model_type = parse_endpoint_types(&config.endpoint_types)?;

    let mut local_model = build_local_model(config, engine_config).await?;
    tracing::debug!("local model built");
    local_model
        .attach(&endpoint, model_type, config.model_input, None)
        .await
        .map_err(|e| {
            err(
                ErrorType::Backend(BackendError::Unknown),
                format!("model attach: {e}"),
            )
        })?;
    tracing::debug!("model registered with discovery");

    let served =
        resolve_served_name(config, engine_config).unwrap_or_else(|| engine_config.model.clone());
    tracing::info!(
        "Serving {} on {}.{}.{}",
        served,
        config.namespace,
        config.component,
        config.endpoint
    );

    let ingress =
        Ingress::for_engine(Arc::new(EngineAdapter::new(engine.clone()))).map_err(|e| {
            err(
                ErrorType::Backend(BackendError::Unknown),
                format!("ingress: {e}"),
            )
        })?;

    endpoint
        .endpoint_builder()
        .handler(ingress)
        .graceful_shutdown(true)
        .start()
        .await
        .map_err(|e| {
            err(
                ErrorType::Backend(BackendError::Unknown),
                format!("serve: {e}"),
            )
        })
}

/// Convenience shorthand for `DynamoError::builder().error_type(..).message(..).build()`.
fn err(error_type: ErrorType, message: impl Into<String>) -> DynamoError {
    DynamoError::builder()
        .error_type(error_type)
        .message(message)
        .build()
}

/// Resolve the public-facing served-model name.
///
/// Priority: `WorkerConfig.served_model_name` (operator CLI override) →
/// `EngineConfig.served_model_name` (engine's preferred advertise-as name).
/// Returns `None` if neither is set; callers fall back to
/// `EngineConfig.model`.
fn resolve_served_name(config: &WorkerConfig, engine_config: &EngineConfig) -> Option<String> {
    config
        .served_model_name
        .clone()
        .or_else(|| engine_config.served_model_name.clone())
}

fn parse_endpoint_types(s: &str) -> Result<ModelType, DynamoError> {
    let mut out = ModelType::empty();
    let mut any = false;
    for raw in s.split(',') {
        let t = raw.trim().to_ascii_lowercase();
        if t.is_empty() {
            continue;
        }
        let flag = match t.as_str() {
            "chat" => ModelType::Chat,
            "completions" => ModelType::Completions,
            "embedding" | "embeddings" => ModelType::Embedding,
            "tensor" => ModelType::TensorBased,
            "prefill" => ModelType::Prefill,
            other => {
                return Err(err(
                    ErrorType::Backend(BackendError::InvalidArgument),
                    format!("unknown endpoint type '{other}'"),
                ));
            }
        };
        out |= flag;
        any = true;
    }
    if !any {
        return Err(err(
            ErrorType::Backend(BackendError::InvalidArgument),
            "endpoint_types cannot be empty",
        ));
    }
    Ok(out)
}

async fn build_local_model(
    config: &WorkerConfig,
    engine_config: &EngineConfig,
) -> Result<LocalModel, DynamoError> {
    let served_name = resolve_served_name(config, engine_config)
        .or_else(|| Some(engine_config.model.clone()))
        .filter(|s| !s.is_empty());

    let rt_cfg = ModelRuntimeConfig {
        total_kv_blocks: engine_config.total_kv_blocks,
        max_num_seqs: engine_config.max_num_seqs,
        max_num_batched_tokens: engine_config.max_num_batched_tokens,
        ..ModelRuntimeConfig::default()
    };

    let mut builder = LocalModelBuilder::default();
    builder
        .model_name(served_name)
        .context_length(engine_config.context_length)
        .kv_cache_block_size(engine_config.kv_cache_block_size)
        .custom_template_path(config.custom_jinja_template.clone())
        .runtime_config(rt_cfg);

    // Resolve WorkerConfig.model_name into a local path. Empty string means
    // name-only mode (no tokenizer / chat template on the card).
    if !config.model_name.is_empty() {
        let source = config.model_name.clone();
        let local_path = if std::fs::exists(&source).map_err(|e| {
            err(
                ErrorType::Backend(BackendError::InvalidArgument),
                format!("model path: {e}"),
            )
        })? {
            PathBuf::from(&source)
        } else {
            LocalModel::fetch(&source, false).await.map_err(|e| {
                err(
                    ErrorType::Backend(BackendError::CannotConnect),
                    format!("fetch '{source}': {e}"),
                )
            })?
        };
        builder.model_path(local_path);
        builder.source_path(PathBuf::from(source));
    }

    builder.build().await.map_err(|e| {
        err(
            ErrorType::Backend(BackendError::Unknown),
            format!("build local model: {e}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error_type_of(result: Result<ModelType, DynamoError>) -> ErrorType {
        result.unwrap_err().error_type()
    }

    #[test]
    fn parse_endpoint_types_happy_path() {
        let got = parse_endpoint_types("chat,completions").unwrap();
        assert_eq!(got, ModelType::Chat | ModelType::Completions);
    }

    #[test]
    fn parse_endpoint_types_single() {
        assert_eq!(parse_endpoint_types("chat").unwrap(), ModelType::Chat);
        assert_eq!(
            parse_endpoint_types("completions").unwrap(),
            ModelType::Completions
        );
        assert_eq!(
            parse_endpoint_types("embedding").unwrap(),
            ModelType::Embedding
        );
    }

    #[test]
    fn parse_endpoint_types_trims_and_lowercases() {
        let got = parse_endpoint_types("  Chat , COMPLETIONS  ").unwrap();
        assert_eq!(got, ModelType::Chat | ModelType::Completions);
    }

    #[test]
    fn parse_endpoint_types_rejects_empty() {
        assert_eq!(
            error_type_of(parse_endpoint_types("")),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert_eq!(
            error_type_of(parse_endpoint_types("   ,  ")),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn parse_endpoint_types_rejects_unknown() {
        let e = parse_endpoint_types("chat,bogus").unwrap_err();
        assert_eq!(
            e.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(e.to_string().contains("bogus"));
    }
}
