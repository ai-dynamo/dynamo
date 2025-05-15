// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use std::{future::Future, pin::Pin, sync::Arc};

use dynamo_llm::{
    backend::Backend,
    engines::StreamingEngineAdapter,
    model_type::ModelType,
    preprocessor::{BackendInput, BackendOutput},
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        Annotated,
    },
};
use dynamo_runtime::engine::AsyncEngineStream;
use dynamo_runtime::pipeline::{
    network::Ingress, Context, ManyOut, Operator, SegmentSource, ServiceBackend, SingleIn, Source,
};
use dynamo_runtime::{protocols::Endpoint as EndpointId, DistributedRuntime};

use crate::EngineConfig;

pub async fn run(
    distributed_runtime: DistributedRuntime,
    path: String,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let cancel_token = distributed_runtime.primary_token().clone();
    let endpoint_id: EndpointId = path.parse()?;

    let component = distributed_runtime
        .namespace(&endpoint_id.namespace)?
        .component(&endpoint_id.component)?;
    let endpoint = component
        .service_builder()
        .create()
        .await?
        .endpoint(&endpoint_id.name);

    let (rt_fut, card): (Pin<Box<dyn Future<Output = _> + Send + 'static>>, _) = match engine_config
    {
        EngineConfig::StaticFull { engine, mut model } => {
            let engine = Arc::new(StreamingEngineAdapter::new(engine));
            let ingress_chat = Ingress::<
                Context<NvCreateChatCompletionRequest>,
                Pin<Box<dyn AsyncEngineStream<Annotated<NvCreateChatCompletionStreamResponse>>>>,
            >::for_engine(engine)?;

            model.attach(&endpoint, ModelType::Chat).await?;
            let fut_chat = endpoint.endpoint_builder().handler(ingress_chat).start();

            (Box::pin(fut_chat), Some(model.card().clone()))
        }
        EngineConfig::StaticCore {
            engine: inner_engine,
            mut model,
        } => {
            // Pre-processing is done ingress-side, so it should be already done.
            let frontend =
                SegmentSource::<SingleIn<BackendInput>, ManyOut<Annotated<BackendOutput>>>::new();
            let backend = Backend::from_mdc(model.card().clone())
                .await?
                .into_operator();
            let engine = ServiceBackend::from_engine(inner_engine);
            let pipeline = frontend
                .link(backend.forward_edge())?
                .link(engine)?
                .link(backend.backward_edge())?
                .link(frontend)?;
            let ingress = Ingress::for_pipeline(pipeline)?;

            model.attach(&endpoint, ModelType::Backend).await?;
            let fut = endpoint.endpoint_builder().handler(ingress).start();

            (Box::pin(fut), Some(model.card().clone()))
        }
        EngineConfig::Dynamic(_) => {
            // We can only get here for in=dyn out=vllm|sglang`, because vllm and sglang are a
            // subprocess that we talk to like a remote endpoint.
            // That means the vllm/sglang subprocess is doing all the work, we are idle.
            (never_ready(), None)
        }
    };

    tokio::select! {
        _ = rt_fut => {
            tracing::debug!("Endpoint ingress ended");
        }
        _ = cancel_token.cancelled() => {
        }
    }

    // Cleanup on shutdown
    if let Some(mut card) = card {
        if let Err(err) = card
            .delete_from_nats(distributed_runtime.nats_client())
            .await
        {
            tracing::error!(%err, "delete_from_nats error on shutdown");
        }
    }

    Ok(())
}

fn never_ready() -> Pin<Box<dyn Future<Output = anyhow::Result<()>> + Send + 'static>> {
    Box::pin(std::future::pending::<anyhow::Result<()>>())
}
