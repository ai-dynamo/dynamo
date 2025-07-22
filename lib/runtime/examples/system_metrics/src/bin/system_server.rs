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

use system_metrics::{MyStats, DEFAULT_NAMESPACE};

use dynamo_runtime::{
    logging,
    metrics::MetricsRegistry,
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    protocols::annotated::Annotated,
    stream, DistributedRuntime, Result, Runtime, Worker,
};

use prometheus::{Counter, Histogram};
use std::sync::Arc;

/// Service metrics struct using the metric classes from metrics.rs
pub struct MySystemStatsMetrics {
    pub request_counter: Arc<Counter>,
    pub request_duration: Arc<Histogram>,
}

impl MySystemStatsMetrics {
    /// Create a new ServiceMetrics instance using the metric backend
    pub fn new<R: MetricsRegistry>(
        metrics_registry: Arc<R>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let request_counter = metrics_registry.create_counter(
            "service_requests_total",
            "Total number of requests processed",
            &[("service", "backend")],
        )?;
        let request_duration = metrics_registry.create_histogram(
            "service_request_duration_seconds",
            "Time spent processing requests",
            &[("service", "backend")],
            None,
        )?;
        Ok(Self {
            request_counter,
            request_duration,
        })
    }
}

fn main() -> Result<()> {
    logging::init();
    let worker = Worker::from_settings()?;
    worker.execute(app)
}

async fn app(runtime: Runtime) -> Result<()> {
    let distributed = DistributedRuntime::from_settings(runtime.clone()).await?;
    backend(distributed).await
}

struct RequestHandler {
    metrics: Arc<MySystemStatsMetrics>,
}

impl RequestHandler {
    fn new(metrics: Arc<MySystemStatsMetrics>) -> Arc<Self> {
        Arc::new(Self { metrics })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
    async fn generate(&self, input: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let start_time = std::time::Instant::now();

        // Record request start
        self.metrics.request_counter.inc();

        let (data, ctx) = input.into_parts();

        let chars = data
            .chars()
            .map(|c| Annotated::from_data(c.to_string()))
            .collect::<Vec<_>>();

        let stream = stream::iter(chars);

        // Record request duration
        let duration = start_time.elapsed();
        self.metrics
            .request_duration
            .observe(duration.as_secs_f64());

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

async fn backend(drt: DistributedRuntime) -> Result<()> {
    let endpoint = drt
        .namespace(DEFAULT_NAMESPACE)?
        .component("component")?
        .service_builder()
        .create()
        .await?
        .endpoint("endpoint");

    // make the ingress discoverable via a component service
    // we must first create a service, then we can attach one more more endpoints
    // attach an ingress to an engine, with the RequestHandler using the metrics struct
    let endpoint_metrics = Arc::new(
        MySystemStatsMetrics::new(Arc::new(endpoint.clone()))
            .map_err(|e| Error::msg(e.to_string()))?,
    );
    let ingress = Ingress::for_engine(RequestHandler::new(endpoint_metrics.clone()))?;

    endpoint
        .endpoint_builder()
        .stats_handler(|_stats| {
            println!("Stats handler called with stats: {:?}", _stats);
            let stats = MyStats { val: 10 };
            serde_json::to_value(stats).unwrap()
        })
        .handler(ingress)
        .start()
        .await?;

    Ok(())
}
