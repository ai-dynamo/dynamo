// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{future::Future, pin::Pin, sync::Arc};

use crate::{
    backend::Backend,
    engines::StreamingEngineAdapter,
    model_type::ModelType,
    preprocessor::{BackendOutput, PreprocessedRequest},
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

use crate::entrypoint::EngineConfig;

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
            let frontend = SegmentSource::<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<BackendOutput>>,
            >::new();
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
            unreachable!("An endpoint input will never have a Dynamic engine");
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
