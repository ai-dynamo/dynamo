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

use std::pin::Pin;

use dynamo_llm::{
    backend::{Backend, ExecutionContext},
    engines::StreamingEngineAdapter,
    http::service::discovery::ModelNetworkName,
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
    pipeline::{
        Context, ManyOut, Operator, PushRouter, ServiceBackend, ServiceFrontend, SingleIn, Source,
    },
    DistributedRuntime, Runtime,
};
use std::sync::Arc;

use crate::{flags::RouterMode, EngineConfig, Flags};

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
pub async fn prepare_engine(
    runtime: Runtime,
    flags: Flags,
    engine_config: EngineConfig,
) -> anyhow::Result<(String, OpenAIChatCompletionsStreamingEngine, bool)> {
    match engine_config {
        EngineConfig::Dynamic(endpoint_id) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;

            let endpoint = distributed_runtime
                .namespace(endpoint_id.namespace.clone())?
                .component(endpoint_id.component.clone())?
                .endpoint(endpoint_id.name.clone());

            let client = endpoint.client().await?;
            let router = match &flags.router_mode {
                RouterMode::Random | RouterMode::RoundRobin => {
                    tracing::info!("Waiting for remote model..");

                    // We then use the ModelDeploymentCard's `requires_preprocessing`
                    // field to decide what kind of PushRouter to make.
                    let remote_endpoints = client.wait_for_endpoints().await?;
                    debug_assert!(!remote_endpoints.is_empty());
                    tracing::info!(count = remote_endpoints.len(), "Model(s) discovered");

                    let network_name: ModelNetworkName = (&remote_endpoints[0]).into();
                    let Some(etcd_client) = distributed_runtime.etcd_client() else {
                        anyhow::bail!("Cannot run distributed components without etcd");
                    };
                    let mdc = network_name.load_mdc(endpoint_id, etcd_client).await?;
                    if mdc.requires_preprocessing {
                        // Note requires_preprocessing is never true in our code right now
                        todo!("Ingress-side pre-processing not supported yet");
                    } else {
                        PushRouter::<
                            NvCreateChatCompletionRequest,
                            Annotated<NvCreateChatCompletionStreamResponse>,
                        >::from_client(client, flags.router_mode.into())
                        .await?
                    }
                }
                RouterMode::KV => todo!(),
            };

            // The service_name isn't used for text chat outside of logs,
            // so use the path. That avoids having to listen on etcd for model registration.
            let service_name = endpoint.subject();
            Ok((service_name, Arc::new(router), false))
        }
        EngineConfig::StaticFull {
            service_name,
            engine,
            card: _card,
        } => {
            tracing::debug!("Model: {service_name}");
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            Ok((service_name, engine, false))
        }
        EngineConfig::StaticCore {
            service_name,
            engine: inner_engine,
            card,
        } => {
            let pipeline = build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(&card, inner_engine)
            .await?;

            tracing::debug!("Model: {service_name} with pre-processing");
            Ok((service_name, pipeline, true))
        }
        EngineConfig::None => unreachable!(),
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
        let card = ModelDeploymentCard::from_local_path(HF_PATH, None).await?;
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
        let card = ModelDeploymentCard::from_local_path(HF_PATH, None).await?;
        let engine = dynamo_llm::engines::make_engine_core();

        // Build pipeline for completions
        let pipeline =
            build_pipeline::<CompletionRequest, CompletionResponse>(&card, engine).await?;

        // Verify pipeline was created
        assert!(Arc::strong_count(&pipeline) >= 1);

        Ok(())
    }
}
