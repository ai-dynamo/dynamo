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

use async_nats::service::endpoint::Stats;
use dynamo_llm::kv_router::{
    protocols::ForwardPassMetrics, scheduler::KVHitRateEvent, KV_HIT_RATE_SUBJECT,
};
use dynamo_runtime::{
    component::Namespace,
    logging,
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
    stream,
    traits::events::EventPublisher,
    DistributedRuntime, Result, Runtime, Worker,
};
use rand::Rng;
use std::sync::Arc;
use tokio::time::{interval, Duration};

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    backend(distributed).await
}

struct MockRequestHandler {}

impl MockRequestHandler {
    fn new() -> Arc<Self> {
        Arc::new(Self {})
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for MockRequestHandler {
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

/// Spawns a background task that periodically publishes mock KV hit rate events
async fn mock_event_publisher(namespace: Namespace) {
    // NOTE: These events are just for testing, and shouldn't be interpreted
    // in correlation with the stats handler's data:
    // 1. The worker ID associated with the events here won't match the
    // worker ID of the endpoint's service stats handler.
    // 2. These events aren't coming through the KV Router, so the metrics won't
    // be reflective of the KV Router's performance.
    // 3. The data in these events aren't in sync with the stats handler's
    // ForwardPassMetrics data, so they may not correlate well.
    let worker_id = rand::thread_rng().gen_range(1..=1000);

    let mut interval = interval(Duration::from_secs(1));
    loop {
        interval.tick().await;

        // Generate random KV hit rate event using a new thread_rng each time
        let isl_blocks = rand::thread_rng().gen_range(0..=100);
        let overlap_blocks = rand::thread_rng().gen_range(0..=isl_blocks);

        let event = KVHitRateEvent {
            worker_id,
            isl_blocks,
            overlap_blocks,
        };

        if let Err(e) = namespace.publish(KV_HIT_RATE_SUBJECT, &event).await {
            tracing::warn!("Failed to publish KV hit rate event: {e}");
        } else {
            tracing::info!(
                "Published KV hit rate event: worker_id={worker_id}, isl_blocks={isl_blocks}, overlap_blocks={overlap_blocks}, hit_rate={:.2}%",
                (overlap_blocks as f64 / isl_blocks as f64) * 100.0
            );
        }
    }
}

/// Generates mock forward pass metrics for stats handler
fn mock_stats_handler(_stats: Stats) -> serde_json::Value {
    println!("stats in: {:?}", _stats);
    let request_total_slots = 100;
    let request_active_slots = rand::thread_rng().gen_range(0..=request_total_slots);
    let kv_total_blocks = 100;
    let kv_active_blocks = rand::thread_rng().gen_range(0..=kv_total_blocks);
    let num_requests_waiting = rand::thread_rng().gen_range(0..=100);
    let gpu_cache_usage_perc = rand::thread_rng().gen_range(0.0..=1.0);
    let gpu_prefix_cache_hit_rate = rand::thread_rng().gen_range(0.0..=1.0);
    let stats = ForwardPassMetrics {
        request_active_slots,
        request_total_slots,
        kv_active_blocks,
        kv_total_blocks,
        num_requests_waiting,
        gpu_cache_usage_perc,
        gpu_prefix_cache_hit_rate,
    };
    println!("stats out: {:?}", stats);
    serde_json::to_value(stats).unwrap()
}

async fn backend(runtime: DistributedRuntime) -> Result<()> {
    let namespace = runtime.namespace("dynamo")?;

    // Spawn background task for publishing KV hit rate events
    let namespace_clone = namespace.clone();
    tokio::spawn(async move {
        mock_event_publisher(namespace_clone).await;
    });

    // attach an ingress to an engine
    let ingress = Ingress::for_engine(MockRequestHandler::new())?;

    // make the ingress discoverable via a component service
    // we must first create a service, then we can attach one more more endpoints
    namespace
        .component("backend")?
        .service_builder()
        .create()
        .await?
        .endpoint("generate")
        .endpoint_builder()
        // Dummy stats handler to demonstrate how to attach a custom stats handler
        .stats_handler(mock_stats_handler)
        .handler(ingress)
        .start()
        .await
}
