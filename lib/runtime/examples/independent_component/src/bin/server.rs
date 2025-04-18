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

use dynamo_runtime::{
    error, logging,
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
    stream, DistributedRuntime, Result, Runtime, Worker,
};
use independent_component::DEFAULT_NAMESPACE;
use std::sync::Arc;

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    let result = backend(distributed).await;
    tracing::info!("backend shutdown cleanly");
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    tracing::info!("backend shutting down");
    result
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

        // emit 1 character a second
        let stream = async_stream::stream! {
            for char in chars {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                yield char
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

async fn backend(runtime: DistributedRuntime) -> Result<()> {
    // attach an ingress to an engine
    let ingress = Ingress::for_engine(RequestHandler::new())?;

    let custom_lease = runtime
        .etcd_client()
        .ok_or(error!("etcd client not found"))?
        .create_lease(1)
        .await?;
    tracing::info!("created custom lease: {:?}", custom_lease);

    // // make the ingress discoverable via a component service
    // // we must first create a service, then we can attach one more more endpoints
    runtime
        .namespace(DEFAULT_NAMESPACE)?
        .component("backend")?
        .service_builder()
        .create()
        .await?
        .endpoint("generate")
        .endpoint_builder()
        .handler(ingress)
        .lease(Some(custom_lease))
        .start()
        .await
}
