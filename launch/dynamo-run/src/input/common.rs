// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;

use dynamo_llm::{
    backend::{Backend, ExecutionContext},
    discovery::{ModelManager, ModelWatcher, MODEL_ROOT_PATH},
    engines::StreamingEngineAdapter,
    model_card::ModelDeploymentCard,
    preprocessor::OpenAIPreprocessor,
    protocols::common::llm_backend::{BackendInput, BackendOutput},
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            OpenAIChatCompletionsStreamingEngine,
        },
        Annotated,
    },
};
use dynamo_runtime::{
    engine::{AsyncEngineStream, Data},
    pipeline::{Context, ManyOut, Operator, ServiceBackend, ServiceFrontend, SingleIn, Source},
    DistributedRuntime, Runtime,
};
use std::sync::Arc;

use crate::EngineConfig;

pub struct PreparedEngine {
    pub service_name: String,
    pub engine: OpenAIChatCompletionsStreamingEngine,
    pub inspect_template: bool,
}

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
pub async fn prepare_engine(
    runtime: Runtime,
    engine_config: EngineConfig,
) -> anyhow::Result<PreparedEngine> {
    match engine_config {
        EngineConfig::Dynamic => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;

            let Some(etcd_client) = distributed_runtime.etcd_client() else {
                anyhow::bail!("Cannot be both static mode and run with dynamic discovery.");
            };
            let model_manager = Arc::new(ModelManager::new());
            let watch_obj = Arc::new(ModelWatcher::new(
                distributed_runtime,
                model_manager.clone(),
                dynamo_runtime::pipeline::RouterMode::RoundRobin,
            ));
            let models_watcher = etcd_client.kv_get_and_watch_prefix(MODEL_ROOT_PATH).await?;
            let (_prefix, _watcher, receiver) = models_watcher.dissolve();

            let inner_watch_obj = watch_obj.clone();
            let _watcher_task = tokio::spawn(async move {
                inner_watch_obj.watch(receiver).await;
            });
            tracing::info!("Waiting for remote model..");

            // TODO: We use the first model to appear, usually we have only one
            // We should add slash commands to text input `/model <name>` to choose,
            // '/models` to list, and notifications when models are added / removed.

            let model_service_name = watch_obj.wait_for_chat_model().await;
            let engine = model_manager.get_chat_completions_engine(&model_service_name)?;
            Ok(PreparedEngine {
                service_name: model_service_name,
                engine,
                inspect_template: false,
            })
        }
        EngineConfig::StaticFull { engine, model } => {
            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with engine pre-processing");
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            Ok(PreparedEngine {
                service_name,
                engine,
                inspect_template: false,
            })
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            model,
        } => {
            let pipeline = build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine)
            .await?;

            let service_name = model.service_name().to_string();
            tracing::debug!("Model: {service_name} with Dynamo pre-processing");
            Ok(PreparedEngine {
                service_name,
                engine: pipeline,
                inspect_template: true,
            })
        }
    }
}

pub async fn build_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
) -> anyhow::Result<Arc<ServiceFrontend<SingleIn<Req>, ManyOut<Annotated<Resp>>>>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
        Context<Req>,
        Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
        Context<BackendInput>,
        Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>,
    >,
{
    let frontend = ServiceFrontend::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
    let preprocessor = OpenAIPreprocessor::new((*card).clone())
        .await?
        .into_operator();
    let backend = Backend::from_mdc((*card).clone()).await?.into_operator();
    let engine = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(engine)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link(frontend)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::types::openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{CompletionRequest, CompletionResponse},
    };

    const HF_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../lib/llm/tests/data/sample-models/mock-llama-3.1-8b-instruct"
    );

    #[tokio::test]
    async fn test_build_chat_completions_pipeline_core_engine_succeeds() -> anyhow::Result<()> {
        // Create test model card
        let card = ModelDeploymentCard::load(HF_PATH).await?;
        let engine = dynamo_llm::engines::make_engine_core();

        // Build pipeline for chat completions
        let pipeline = build_pipeline::<
            NvCreateChatCompletionRequest,
            NvCreateChatCompletionStreamResponse,
        >(&card, engine)
        .await?;

        // Verify pipeline was created
        assert!(Arc::strong_count(&pipeline) >= 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_build_completions_pipeline_core_engine_succeeds() -> anyhow::Result<()> {
        // Create test model card
        let card = ModelDeploymentCard::load(HF_PATH).await?;
        let engine = dynamo_llm::engines::make_engine_core();

        // Build pipeline for completions
        let pipeline =
            build_pipeline::<CompletionRequest, CompletionResponse>(&card, engine).await?;

        // Verify pipeline was created
        assert!(Arc::strong_count(&pipeline) >= 1);

        Ok(())
    }
}
