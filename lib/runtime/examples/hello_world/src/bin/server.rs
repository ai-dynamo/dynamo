// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{
    DistributedRuntime, Result, Runtime, Worker, logging,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait, network::Ingress,
    },
    protocols::annotated::Annotated,
    stream,
};
use hello_world::DEFAULT_NAMESPACE;
use std::sync::Arc;

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    backend(distributed).await
}

struct RequestHandler {}

impl RequestHandler {
    fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let (data, ctx) = input.into_parts();

        let chars = data
            .chars()
            .map(|c| Annotated::from_data(c.to_string()))
            .collect::<Vec<_>>();

        let stream = stream::iter(chars);

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

async fn backend(runtime: DistributedRuntime) -> Result<()> {
    // attach an ingress to an engine
    let ingress = Ingress::for_engine(RequestHandler::new())?;

    // // make the ingress discoverable via a component service
    // // we must first create a service, then we can attach one more more endpoints
    let mut component = runtime.namespace(DEFAULT_NAMESPACE)?.component("backend")?;
    component.add_stats_service().await?;
    component
        .endpoint("generate")
        .endpoint_builder()
        .handler(ingress)
        .start()
        .await
}
