// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::kv_router::{
    KV_HIT_RATE_SUBJECT,
    protocols::{ForwardPassMetrics, KvStats, WorkerStats},
    scheduler::KVHitRateEvent,
};
use dynamo_runtime::{
    DistributedRuntime, Result, Runtime, Worker,
    component::{Namespace, service::EndpointStats},
    logging,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait, network::Ingress,
    },
    protocols::annotated::Annotated,
    stream,
    traits::events::EventPublisher,
};
use rand::Rng;
use std::sync::Arc;
use tokio::time::{Duration, interval};

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

// FIXME: These events are just for testing and may not currently be used.
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
    let worker_id = rand::rng().random_range(1..=1000);

    let mut interval = interval(Duration::from_secs(1));
    loop {
        interval.tick().await;

        // Generate random KV hit rate event using a new thread_rng each time
        let isl_blocks = rand::rng().random_range(0..=100);
        let overlap_blocks = rand::rng().random_range(0..=isl_blocks);

        let event = KVHitRateEvent {
            worker_id,
            isl_blocks,
            overlap_blocks: overlap_blocks as u32,
        };

        if let Err(e) = namespace.publish(KV_HIT_RATE_SUBJECT, &event).await {
            tracing::warn!("Failed to publish KV hit rate event: {e}");
        } else {
            tracing::debug!(
                "Published KV hit rate event: worker_id={worker_id}, isl_blocks={isl_blocks}, overlap_blocks={overlap_blocks}, hit_rate={:.2}%",
                (overlap_blocks as f64 / isl_blocks as f64) * 100.0
            );
        }
    }
}

/// Generates mock forward pass metrics for stats handler
fn mock_stats_handler(_stats: EndpointStats) -> serde_json::Value {
    let request_total_slots = 100;
    let request_active_slots = rand::rng().random_range(0..=request_total_slots);
    let kv_total_blocks = 100;
    let kv_active_blocks = rand::rng().random_range(0..=kv_total_blocks);
    let num_requests_waiting = rand::rng().random_range(0..=100);
    let gpu_cache_usage_perc = rand::rng().random_range(0.0..=1.0);
    let gpu_prefix_cache_hit_rate = rand::rng().random_range(0.0..=1.0);

    let worker_stats = WorkerStats {
        data_parallel_rank: None, // Default for backwards compatibility
        request_active_slots,
        request_total_slots,
        num_requests_waiting,
    };

    let kv_stats = KvStats {
        kv_active_blocks,
        kv_total_blocks,
        gpu_cache_usage_perc,
        gpu_prefix_cache_hit_rate,
    };

    let spec_decode_stats = None;

    let stats = ForwardPassMetrics {
        worker_stats,
        kv_stats,
        spec_decode_stats,
    };
    tracing::info!("Stats: {stats:?}");
    serde_json::to_value(stats).unwrap()
}

async fn backend(runtime: DistributedRuntime) -> Result<()> {
    let namespace = runtime.namespace("dynamo")?;
    // we must first create a service, then we can attach one more more endpoints
    let component = namespace
        .component("MyComponent")?
        .service_builder()
        .create()
        .await?;
    let endpoint = component.endpoint("my_endpoint");
    tracing::info!("Starting Mock Worker on Endpoint: {}", endpoint.path());

    // Spawn background task for publishing KV hit rate events
    let namespace_clone = namespace.clone();
    tokio::spawn(async move {
        mock_event_publisher(namespace_clone).await;
    });

    // Attach an ingress to the engine
    let ingress = Ingress::for_engine(MockRequestHandler::new())?;

    // Make the ingress discoverable via a component service
    endpoint
        .endpoint_builder()
        // Dummy stats handler to demonstrate how to attach a custom stats handler
        .stats_handler(mock_stats_handler)
        .handler(ingress)
        .start()
        .await
}
