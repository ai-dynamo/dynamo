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
use dynamo_llm::http::service::ModelManager;
use dynamo_llm::{
    engines::StreamingEngineAdapter,
    http::service::{discovery, service_v2},
    request_template::RequestTemplate,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        openai::completions::{CompletionRequest, CompletionResponse},
    },
};
use dynamo_runtime::transports::etcd;
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
                    run_watcher(
                        distributed_runtime.clone(),
                        http_service.model_manager().clone(),
                        etcd_client.clone(),
                        &network_prefix,
                    )
                    .await?;
                }
                None => {
                    // Static endpoints don't need discovery
                }
            }
        }
        EngineConfig::StaticFull { engine, model } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let manager = http_service.model_manager();
            manager.add_completions_model(model.service_name(), engine.clone())?;
            manager.add_chat_completions_model(model.service_name(), engine)?;
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            model,
        } => {
            let manager = http_service.model_manager();

            let chat_pipeline = common::build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse,
            >(model.card(), inner_engine.clone())
            .await?;
            manager.add_chat_completions_model(model.service_name(), chat_pipeline)?;

            let cmpl_pipeline = common::build_pipeline::<CompletionRequest, CompletionResponse>(
                model.card(),
                inner_engine,
            )
            .await?;
            manager.add_completions_model(model.service_name(), cmpl_pipeline)?;
        }
    }
    http_service.run(runtime.primary_token()).await?;
    runtime.shutdown(); // Cancel primary token
    Ok(())
}

/// Spawns a task that watches for new models in etcd at network_prefix,
/// and registers them with the ModelManager so that the HTTP service can use them.
async fn run_watcher(
    distributed_runtime: DistributedRuntime,
    model_manager: ModelManager,
    etcd_client: etcd::Client,
    network_prefix: &str,
) -> anyhow::Result<()> {
    let state = Arc::new(discovery::ModelWatchState {
        prefix: network_prefix.to_string(),
        manager: model_manager,
        drt: distributed_runtime.clone(),
    });
    tracing::info!("Watching for remote model at {network_prefix}");
    let models_watcher = etcd_client.kv_get_and_watch_prefix(network_prefix).await?;
    let (_prefix, _watcher, receiver) = models_watcher.dissolve();
    let _watcher_task = tokio::spawn(discovery::model_watcher(state, receiver));
    Ok(())
}
