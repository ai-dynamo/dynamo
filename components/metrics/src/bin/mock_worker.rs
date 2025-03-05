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

use rand::Rng;
use std::sync::Arc;
use triton_distributed_llm::kv_router::protocols::ForwardPassMetrics;
use triton_distributed_runtime::{
    logging,
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
    stream, DistributedRuntime, Result, Runtime, Worker,
};

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

    // make the ingress discoverable via a component service
    // we must first create a service, then we can attach one more more endpoints

    runtime
        .namespace("triton-init")?
        .component("backend")?
        .service_builder()
        .create()
        .await?
        .endpoint("generate")
        .endpoint_builder()
        // Dummy stats handler to demonstrate how to attach a custom stats handler
        .stats_handler(|_stats| {
            println!("stats in: {:?}", _stats);
            let request_total_slots = 100;
            let request_active_slots = rand::thread_rng().gen_range(0..request_total_slots);
            let kv_total_blocks = 100;
            let kv_active_blocks = rand::thread_rng().gen_range(0..kv_total_blocks);
            let stats = ForwardPassMetrics {
                request_active_slots,
                request_total_slots,
                kv_active_blocks,
                kv_total_blocks,
            };
            println!("stats out: {:?}", stats);
            serde_json::to_value(stats).unwrap()
        })
        .handler(ingress)
        .start()
        .await
}
