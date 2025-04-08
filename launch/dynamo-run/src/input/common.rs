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

use crate::{flags::RouterMode, EngineConfig, Flags};
use dynamo_llm::{
    backend::Backend,
    preprocessor::OpenAIPreprocessor,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
            OpenAIChatCompletionsStreamingEngine,
        },
        Annotated,
    },
};
use dynamo_runtime::{
    pipeline::{ManyOut, Operator, ServiceBackend, ServiceFrontend, SingleIn, Source},
    DistributedRuntime, Runtime,
};
use std::sync::Arc;

/// Turns an EngineConfig into an OpenAIChatCompletionsStreamingEngine.
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

            let mut client = endpoint.client::<NvCreateChatCompletionRequest, Annotated<NvCreateChatCompletionStreamResponse>>().await?;

            match &flags.router_mode {
                RouterMode::Random | RouterMode::RoundRobin => {
                    client.set_router_mode(flags.router_mode.into());
                    tracing::info!("Waiting for remote model..");
                    client.wait_for_endpoints().await?;
                    tracing::info!("Model discovered");
                }
                RouterMode::KV => todo!(),
            }

            // The service_name isn't used for text chat outside of logs,
            // so use the path. That avoids having to listen on etcd for model registration.
            let service_name = endpoint.subject();
            Ok((service_name, Arc::new(client), false))
        }
        EngineConfig::StaticFull {
            service_name,
            engine,
        } => {
            tracing::debug!("Model: {service_name}");
            Ok((service_name, engine, false))
        }
        EngineConfig::StaticCore {
            service_name,
            engine: inner_engine,
            card,
        } => {
            let frontend = ServiceFrontend::<
                SingleIn<NvCreateChatCompletionRequest>,
                ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
            >::new();
            let preprocessor = OpenAIPreprocessor::new(*card.clone())
                .await?
                .into_operator();
            let backend = Backend::from_tokenizer(card.tokenizer_hf()?)
                .await?
                .into_operator();
            let engine = ServiceBackend::from_engine(inner_engine);

            let pipeline = frontend
                .link(preprocessor.forward_edge())?
                .link(backend.forward_edge())?
                .link(engine)?
                .link(backend.backward_edge())?
                .link(preprocessor.backward_edge())?
                .link(frontend)?;

            tracing::debug!("Model: {service_name} with pre-processing");
            Ok((service_name, pipeline, true))
        }
        EngineConfig::None => unreachable!(),
    }
}
