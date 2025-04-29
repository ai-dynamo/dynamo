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

use std::sync::Arc;

use crate::input::common;
use crate::{EngineConfig, Flags};
use dynamo_llm::{
    engines::StreamingEngineAdapter,
    http::service::{discovery, service_v2},
    model_type::ModelType,
    request_template::RequestTemplate,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        openai::completions::{CompletionRequest, CompletionResponse},
    },
};
use dynamo_runtime::{DistributedRuntime, Runtime};

/// Build and run an HTTP service
pub async fn run(
    runtime: Runtime,
    flags: Flags,
    engine_config: EngineConfig,
    template: Option<RequestTemplate>,
) -> anyhow::Result<()> {
    let http_service = service_v2::HttpService::builder()
        .port(flags.http_port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .with_request_template(template)
        .build()?;
    match engine_config {
        EngineConfig::Dynamic(endpoint) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            match distributed_runtime.etcd_client() {
                Some(etcd_client) => {
                    // This will attempt to connect to NATS and etcd

                    let component = distributed_runtime
                        .namespace(endpoint.namespace)?
                        .component(endpoint.component)?;
                    let network_prefix = component.service_name();

                    // Listen for models registering themselves in etcd, add them to HTTP service
                    let state = Arc::new(discovery::ModelWatchState {
                        prefix: network_prefix.clone(),
                        model_type: ModelType::Chat,
                        manager: http_service.model_manager().clone(),
                        drt: distributed_runtime.clone(),
                    });
                    tracing::info!("Waiting for remote model at {network_prefix}");
                    let models_watcher =
                        etcd_client.kv_get_and_watch_prefix(network_prefix).await?;
                    let (_prefix, _watcher, receiver) = models_watcher.dissolve();
                    let _watcher_task = tokio::spawn(discovery::model_watcher(state, receiver));
                }
                None => {
                    // Static endpoints don't need discovery
                }
            }
        }
        EngineConfig::StaticFull {
            service_name,
            engine,
            ..
        } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = http_service.model_manager();
            manager.add_completions_model(&service_name, engine.clone())?;
            manager.add_chat_completions_model(&service_name, engine)?;
        }
        EngineConfig::StaticCore {
            service_name,
            engine: inner_engine,
            card,
        } => {
            let manager = http_service.model_manager();

            let chat_pipeline = common::build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(&card, inner_engine.clone())
            .await?;
            manager.add_chat_completions_model(&service_name, chat_pipeline)?;

            let cmpl_pipeline = common::build_pipeline::<CompletionRequest, CompletionResponse>(
                &card,
                inner_engine,
            )
            .await?;
            manager.add_completions_model(&service_name, cmpl_pipeline)?;
        }
        EngineConfig::None => unreachable!(),
    }
    http_service.run(runtime.primary_token()).await
}
