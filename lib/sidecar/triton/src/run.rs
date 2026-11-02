// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use anyhow::{Context, ensure};
use clap::Parser;
use dynamo_backend_common::{DisaggregationMode, ModelInput, RawEngine, Worker, WorkerConfig};
use dynamo_llm::grpc::service::kserve::inference;
use dynamo_llm::grpc::service::kserve::inference::model_config::SchedulingChoice;
use dynamo_llm::protocols::tensor::TensorModelConfig;
use dynamo_runtime::Runtime;
use dynamo_runtime::system_status_server::SystemProbePolicy;
use prost::Message;
use sha2::{Digest, Sha256};
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::args::{Args, Task};
use crate::client::TritonClient;
use crate::engine::TritonModelEngine;
use crate::pooling::{Classify, Embeddings, TextEngine, TextModel, TextTask};

type RunFuture = Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send>>;

pub fn from_cli() -> anyhow::Result<impl FnOnce(Runtime, CancellationToken) -> RunFuture> {
    let args = Args::parse();
    args.validate().map_err(anyhow::Error::msg)?;
    Ok(move |runtime, shutdown| -> RunFuture { Box::pin(run(args, runtime, shutdown)) })
}

async fn run(args: Args, runtime: Runtime, shutdown: CancellationToken) -> anyhow::Result<()> {
    let distributed = args.runtime.to_distributed_config()?;
    let drt = dynamo_runtime::DistributedRuntime::new_with_probe_policy(
        runtime,
        distributed,
        SystemProbePolicy::RuntimeOnly,
    )
    .await?;
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
    if args.task != Task::Tensor && args.model_names.is_empty() {
        configs = skip_ensemble_dependencies(configs, args.task)?;
    }

    let mut workers = JoinSet::new();
    for (model, config) in configs {
        let (engine, model_input, endpoint_types): (Arc<dyn RawEngine>, _, _) = match args.task {
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
                text_engine(Classify::default(), &model, &config, &args, &client)?,
                ModelInput::Text,
                "classify",
            ),
            Task::Embeddings => (
                text_engine(Embeddings, &model, &config, &args, &client)?,
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
        "Triton sidecar models discovered"
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
    args: &Args,
    client: &TritonClient,
) -> anyhow::Result<Arc<dyn RawEngine>> {
    let (input, output) = args.tensor_name_overrides();
    let model = TextModel::new::<T>(model, config, input, output, client.clone())?;
    Ok(Arc::new(TextEngine::new(task, model)))
}

/// Drop models that only serve as steps of another model's ensemble.
///
/// A classify or embedding repository pairs one user-facing ensemble (STRING in, FP32 out) with tokenizer and encoder models that have neither shape.
/// Those stay loaded in Triton for their ensemble but are not registered, so they do not fail startup.
fn skip_ensemble_dependencies(
    configs: Vec<(String, inference::ModelConfig)>,
    task: Task,
) -> anyhow::Result<Vec<(String, inference::ModelConfig)>> {
    let dependencies: BTreeSet<&str> = configs
        .iter()
        .filter_map(|(_, config)| match &config.scheduling_choice {
            Some(SchedulingChoice::EnsembleScheduling(ensemble)) => Some(ensemble),
            _ => None,
        })
        .flat_map(|ensemble| ensemble.step.iter().map(|step| step.model_name.as_str()))
        .filter(|name| !name.is_empty())
        .collect();
    let (skipped, exposed): (Vec<_>, Vec<_>) = configs
        .iter()
        .map(|(name, _)| name.as_str())
        .partition(|name| dependencies.contains(name));
    let (task_name, endpoint) = match task {
        Task::Classify => ("classify", Classify::ENDPOINT),
        Task::Embeddings => ("embeddings", Embeddings::ENDPOINT),
        Task::Tensor => unreachable!("tensor models are not filtered"),
    };
    ensure!(
        !exposed.is_empty(),
        "No user-facing {task_name} models found: every ready model is a step of another model's ensemble. \
         Add a user-facing ensemble (STRING in, FP32 out) that wires them together, or run with --task tensor."
    );
    if !skipped.is_empty() {
        tracing::info!(
            ?skipped,
            endpoint,
            "Skipping ensemble dependency models; they stay loaded in Triton for their ensembles"
        );
    }
    let skipped: BTreeSet<String> = skipped.into_iter().map(str::to_string).collect();
    Ok(configs
        .into_iter()
        .filter(|(name, _)| !skipped.contains(name))
        .collect())
}

async fn discover_models(args: &Args, client: &TritonClient) -> anyhow::Result<Vec<String>> {
    if !args.model_names.is_empty() {
        return Ok(args
            .model_names
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .map(str::to_string)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect());
    }

    client
        .ready_models()
        .await
        .context("query Triton repository index")
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
    use dynamo_llm::grpc::service::kserve::inference;
    use dynamo_llm::grpc::service::kserve::inference::model_config::SchedulingChoice;

    use super::{endpoint_slug, skip_ensemble_dependencies};
    use crate::args::Task;

    fn ensemble(steps: &[&str]) -> inference::ModelConfig {
        inference::ModelConfig {
            scheduling_choice: Some(SchedulingChoice::EnsembleScheduling(
                inference::ModelEnsembling {
                    step: steps
                        .iter()
                        .map(|name| inference::model_ensembling::Step {
                            model_name: name.to_string(),
                            ..Default::default()
                        })
                        .collect(),
                },
            )),
            ..Default::default()
        }
    }

    #[test]
    fn skips_ensemble_dependencies() {
        let configs = vec![
            ("encoder".to_string(), Default::default()),
            (
                "text_classifier".to_string(),
                ensemble(&["tokenizer", "encoder"]),
            ),
            ("tokenizer".to_string(), Default::default()),
            ("other".to_string(), Default::default()),
        ];
        let exposed: Vec<_> = skip_ensemble_dependencies(configs, Task::Classify)
            .unwrap()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(exposed, ["text_classifier", "other"]);

        let only_dependencies = vec![
            ("a".to_string(), ensemble(&["b"])),
            ("b".to_string(), ensemble(&["a"])),
        ];
        let error = skip_ensemble_dependencies(only_dependencies, Task::Embeddings)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("No user-facing embeddings models"),
            "{error}"
        );
    }

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
