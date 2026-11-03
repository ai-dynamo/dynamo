// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{Context, ensure};
use clap::{CommandFactory, FromArgMatches};
use dynamo_backend_common::{DisaggregationMode, ModelInput, RawEngine, Worker, WorkerConfig};
use dynamo_llm::grpc::service::kserve::inference;
use dynamo_llm::protocols::tensor::TensorModelConfig;
use dynamo_runtime::DistributedRuntime;
use prost::Message;
use sha2::{Digest, Sha256};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, Task};
use crate::client::TritonClient;
use crate::engine::TritonModelEngine;
use crate::pooling::{Classify, Embeddings, TextEngine, TextModel, TextTask};

/// Run the sidecar from the command line until a model worker stops or the process receives SIGTERM or SIGINT.
///
/// Parses and validates the command line, then writes the resolved endpoints to their environment variables for subprocesses.
/// Call this before starting any other thread: it modifies the process environment.
pub fn run() -> anyhow::Result<()> {
    let matches = Args::command().get_matches();
    let mut args = Args::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());
    args.validate().map_err(anyhow::Error::msg)?;
    // Logging starts with the runtime, after parsing, so `run` logs these.
    let parsed_values = args.parsed_values(&matches);
    for (name, value) in args.endpoint_env() {
        // SAFETY: `main` calls `run` first, and this runs before `run_with_drt` starts the Tokio runtime, while the process has one thread.
        unsafe { std::env::set_var(name, value) };
    }
    // The shared runner races `bootstrap` against SIGTERM and SIGINT, so a signal stops the sidecar while it is still connecting to Triton.
    dynamo_sidecar_common::run_with_drt(args.runtime.clone(), bootstrap(args, parsed_values), serve)
}

/// Triton connection and model configurations found during startup.
struct Bootstrap {
    args: Args,
    client: TritonClient,
    configs: Vec<(String, inference::ModelConfig)>,
}

/// Connect to Triton and fetch the configuration of every model to register.
async fn bootstrap(
    args: Args,
    parsed_values: Vec<(&'static str, String)>,
) -> anyhow::Result<Bootstrap> {
    for (option, value) in parsed_values {
        tracing::debug!(option, value, "Parsed");
    }
    tracing::info!(
        grpc_endpoint = args
            .grpc_endpoint
            .as_ref()
            .map_or("disabled", |endpoint| endpoint.as_str()),
        http_endpoint = args
            .http_endpoint
            .as_ref()
            .map_or("disabled", |endpoint| endpoint.as_str()),
        "Registered Triton Server endpoints: "
    );
    let transport = args.grpc.config();
    let client = TritonClient::connect(
        args.grpc_endpoint.as_ref(),
        args.http_endpoint.as_ref(),
        transport,
    )
    .await?;
    tracing::info!(protocols = ?client.protocols(), "Connected to Triton");
    let models = discover_models(&args, &client).await?;
    ensure!(!models.is_empty(), "Triton has no ready models to register");

    let mut configs = Vec::with_capacity(models.len());
    for model in models {
        let config = client.model_config(&model).await?;
        configs.push((model, config));
    }
    Ok(Bootstrap {
        args,
        client,
        configs,
    })
}

/// Serve every model in `bootstrap` until one model worker stops or `shutdown` is cancelled.
async fn serve(
    drt: DistributedRuntime,
    Bootstrap {
        args,
        client,
        configs,
    }: Bootstrap,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let mut workers = JoinSet::new();
    let (mut tensor_models, mut classify_models, mut embeddings_models) = (0usize, 0usize, 0usize);
    for (model, config) in configs {
        let task = args.task(&model);
        *match task {
            Task::Tensor => &mut tensor_models,
            Task::Classify => &mut classify_models,
            Task::Embeddings => &mut embeddings_models,
        } += 1;
        tracing::info!(
            model,
            task = task.as_str(),
            frontend_endpoint = match task {
                Task::Tensor => "KServe tensor inference",
                Task::Classify => Classify::ENDPOINT,
                Task::Embeddings => Embeddings::ENDPOINT,
            },
            "Registering Triton model"
        );
        let (engine, model_input, endpoint_types): (Arc<dyn RawEngine>, _, _) = match task {
            Task::Tensor => {
                let tensor_model_config = TensorModelConfig {
                    name: model.clone(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    triton_model_config: Some(config.encode_to_vec()),
                };
                let engine =
                    TritonModelEngine::new(model.clone(), client.clone(), tensor_model_config);
                (Arc::new(engine), ModelInput::Tensor, "tensor")
            }
            Task::Classify => (
                text_engine(Classify::default(), &model, &config, &client)?,
                ModelInput::Text,
                "classify",
            ),
            Task::Embeddings => (
                text_engine(Embeddings, &model, &config, &client)?,
                ModelInput::Text,
                "embedding",
            ),
        };
        let worker_config = WorkerConfig {
            namespace: args.namespace.clone(),
            component: args.component.clone(),
            endpoint: endpoint_slug(&model),
            model_name: String::new(),
            served_model_name: Some(model.clone()),
            model_input,
            endpoint_types: endpoint_types.to_string(),
            enable_local_indexer: false,
            enable_kv_routing: false,
            disaggregation_mode: DisaggregationMode::Aggregated,
            runtime: args.runtime.clone(),
            metrics_labels: vec![("model".to_string(), model.clone())],
            ..WorkerConfig::default()
        };
        let drt = drt.clone();
        let shutdown = shutdown.clone();
        workers.spawn(async move {
            Worker::new_raw(engine, worker_config)
                .run_with_drt(drt, shutdown)
                .await
                .map_err(anyhow::Error::from)
        });
    }

    tracing::info!(
        model_count = workers.len(),
        tensor_models,
        classify_models,
        embeddings_models,
        "Triton sidecar models registered"
    );
    let first = workers
        .join_next()
        .await
        .context("Triton sidecar started without model workers")?;
    shutdown.cancel();
    let mut result = first.context("Triton model worker task panicked")?;
    while let Some(joined) = workers.join_next().await {
        match joined {
            Ok(Ok(())) => {}
            Ok(Err(error)) if result.is_ok() => result = Err(error),
            Err(error) if result.is_ok() => {
                result = Err(anyhow::Error::new(error).context("Triton model worker task panicked"))
            }
            _ => {}
        }
    }
    result
}

fn text_engine<T: TextTask>(
    task: T,
    model: &str,
    config: &inference::ModelConfig,
    client: &TritonClient,
) -> anyhow::Result<Arc<dyn RawEngine>> {
    let model = TextModel::new::<T>(model, config, client.clone())?;
    Ok(Arc::new(TextEngine::new(task, model)))
}

/// Models to register: the explicit `--export` list plus every `--classify` and `--embeddings` model, or else every ready model in Triton.
async fn discover_models(args: &Args, client: &TritonClient) -> anyhow::Result<Vec<String>> {
    if !args.model_names.is_empty() {
        let models: Vec<String> = args
            .model_names
            .iter()
            .chain(args.text_models())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        tracing::info!(
            ?models,
            "Using Triton models from --export, --classify, and --embeddings"
        );
        return Ok(models);
    }

    let models = client
        .ready_models()
        .await
        .context("query Triton repository index")?;
    tracing::info!(?models, "Discovered ready Triton models");
    for model in args.text_models() {
        ensure!(
            models.contains(model),
            "Triton model '{model}' passed to {} is not ready; ready models: {models:?}",
            match args.task(model) {
                Task::Classify => "--classify",
                _ => "--embeddings",
            }
        );
    }
    Ok(models)
}

fn endpoint_slug(model: &str) -> String {
    let safe: String = model
        .chars()
        .map(|character| {
            let character = character.to_ascii_lowercase();
            if character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || matches!(character, '-' | '_')
            {
                character
            } else {
                '_'
            }
        })
        .collect();
    let digest = Sha256::digest(model.as_bytes());
    format!(
        "{safe}_{:08x}",
        u32::from_be_bytes(digest[..4].try_into().unwrap())
    )
}

#[cfg(test)]
mod tests {
    use super::endpoint_slug;

    #[test]
    fn endpoint_slug_is_safe_and_unique() {
        let short = endpoint_slug("NOP_TYPE_BOOL_-1,-1");
        let long = endpoint_slug("NOP_TYPE_BOOL_-1,-1,-1");
        assert!(
            short
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        );
        assert_ne!(short, long);
        assert!(!long.starts_with(&short));
    }
}
