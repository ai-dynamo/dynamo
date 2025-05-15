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
    kv_router::{scheduler::DefaultWorkerSelector, KvPushRouter, KvRouter},
    model_card::ModelDeploymentCard,
    model_type::ModelType,
    preprocessor::OpenAIPreprocessor,
    protocols::common::llm_backend::{BackendInput, BackendOutput, LLMEngineOutput},
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
        Context, ManyOut, Operator, PushRouter, SegmentSource, ServiceBackend, ServiceFrontend,
        SingleIn, Source,
    },
    DistributedRuntime, Runtime,
};
use std::sync::Arc;

use crate::{flags::RouterMode, EngineConfig, Flags};

pub struct PreparedEngine {
    pub service_name: String,
    pub engine: OpenAIChatCompletionsStreamingEngine,
    pub inspect_template: bool,
    pub _cache_dir: Option<tempfile::TempDir>,
}

/// Turns an EngineConfig into an OpenAI chat-completions and completions supported StreamingEngine.
pub async fn prepare_engine(
    runtime: Runtime,
    flags: Flags,
    engine_config: EngineConfig,
) -> anyhow::Result<PreparedEngine> {
    match engine_config {
        EngineConfig::Dynamic(endpoint_id) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;

            let endpoint = distributed_runtime
                .namespace(endpoint_id.namespace.clone())?
                .component(endpoint_id.component.clone())?
                .endpoint(endpoint_id.name.clone());

            let client = endpoint.client().await?;
            let mut cache_dir = None;

            tracing::info!("Waiting for remote model..");

            let remote_endpoints = client.wait_for_endpoints().await?;
            debug_assert!(!remote_endpoints.is_empty());
            tracing::info!(count = remote_endpoints.len(), "Model(s) discovered");

            let network_name: ModelNetworkName = (&remote_endpoints[0]).into();
            let Some(etcd_client) = distributed_runtime.etcd_client() else {
                anyhow::bail!("Cannot run distributed components without etcd");
            };
            let network_entry = network_name.load_entry(etcd_client.clone()).await?;
            let mut card = network_entry.load_mdc(endpoint_id, etcd_client).await?;

            let engine: OpenAIChatCompletionsStreamingEngine = match network_entry.model_type {
                ModelType::Backend => {
                    // Download tokenizer.json etc to local disk
                    cache_dir = Some(
                        card.move_from_nats(distributed_runtime.nats_client())
                            .await?,
                    );

                    // The backend doesn't mind what we expose to the user (chat or
                    // completions), and this function is only used by text and batch input so
                    // the user doesn't see the HTTP request. So use Chat.
                    let frontend = SegmentSource::<
                        SingleIn<NvCreateChatCompletionRequest>,
                        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
                    >::new();
                    let preprocessor = OpenAIPreprocessor::new(card.clone()).await?.into_operator();
                    let backend = Backend::from_mdc(card.clone()).await?.into_operator();
                    let router =
                        PushRouter::<BackendInput, Annotated<LLMEngineOutput>>::from_client(
                            client,
                            flags.router_mode.as_runtime(),
                        )
                        .await?;
                    let service_backend = match &flags.router_mode {
                        RouterMode::Random | RouterMode::RoundRobin => {
                            ServiceBackend::from_engine(Arc::new(router))
                        }
                        RouterMode::KV => {
                            let selector = Box::new(DefaultWorkerSelector {});
                            let chooser = KvRouter::new(
                                endpoint.component().clone(),
                                dynamo_llm::DEFAULT_KV_BLOCK_SIZE,
                                Some(selector),
                            )
                            .await?;
                            let kv_push_router = KvPushRouter::new(router, Arc::new(chooser));
                            ServiceBackend::from_engine(Arc::new(kv_push_router))
                        }
                    };
                    frontend
                        .link(preprocessor.forward_edge())?
                        .link(backend.forward_edge())?
                        .link(service_backend)?
                        .link(backend.backward_edge())?
                        .link(preprocessor.backward_edge())?
                        .link(frontend)?
                }
                ModelType::Chat => Arc::new(
                    PushRouter::<
                        NvCreateChatCompletionRequest,
                        Annotated<NvCreateChatCompletionStreamResponse>,
                    >::from_client(client, flags.router_mode.as_runtime())
                    .await?,
                ),
                ModelType::Completion => {
                    anyhow::bail!(
                        "text and batch input only accept remote Chat models, not Completion"
                    );
                    /*
                    Arc::new(
                        PushRouter::<
                            CompletionRequest,
                            Annotated<CompletionResponse>,
                        >::from_client(
                            client, flags.router_mode.into()
                        )
                        .await?,
                    )
                    */
                }
            };
            // The service_name isn't used for text chat outside of logs,
            // so use the path. That avoids having to listen on etcd for model registration.
            let service_name = endpoint.subject();
            Ok(PreparedEngine {
                service_name,
                engine,
                inspect_template: false,
                _cache_dir: cache_dir,
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
                _cache_dir: None,
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
                _cache_dir: None,
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
