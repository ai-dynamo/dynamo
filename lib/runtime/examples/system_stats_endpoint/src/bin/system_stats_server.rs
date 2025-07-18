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

use system_stats_endpoint::{MyStats, DEFAULT_NAMESPACE};

use dynamo_runtime::{
    logging,
    pipeline::{
        async_trait, network::Ingress, AsyncEngine, AsyncEngineContextProvider, Error, ManyOut,
        ResponseStream, SingleIn,
    },
    profiling::MetricsRegistry,
    protocols::annotated::Annotated,
    stream, DistributedRuntime, Result, Runtime, Worker,
};

use prometheus::Counter;
use std::sync::Arc;

/// Service metrics struct using the metric classes from profiling.rs
pub struct MySystemStatsMetrics<R: MetricsRegistry> {
    pub registry: Arc<R>,
    pub request_counter: Arc<Counter>,
}

impl<R: MetricsRegistry> MySystemStatsMetrics<R> {
    /// Create a new ServiceMetrics instance using the metric backend
    pub fn new(registry: Arc<R>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let request_counter = registry.create_counter(
            "service_requests_total",
            "Total number of requests processed",
            &[("service", "backend")],
        )?;
        Ok(Self {
            registry,
            request_counter,
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

struct RequestHandler<R: MetricsRegistry> {
    metrics: Arc<MySystemStatsMetrics<R>>,
}

impl<R: MetricsRegistry> RequestHandler<R> {
    fn new(metrics: Arc<MySystemStatsMetrics<R>>) -> Arc<Self> {
        Arc::new(Self { metrics })
    }
}

#[async_trait]
impl<R: MetricsRegistry> AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error>
    for RequestHandler<R>
{
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

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

async fn backend(drt: DistributedRuntime) -> Result<()> {
    let drt_prometheus_output = drt.encode_prometheus_fmt().unwrap();
    println!("Distributed Runtime in Prometheus format:");
    println!("{}", drt_prometheus_output);

    let namespace = drt.namespace(DEFAULT_NAMESPACE)?;

    let metrics = Arc::new(
        MySystemStatsMetrics::new(Arc::new(namespace.clone()))
            .map_err(|e| Error::msg(e.to_string()))?,
    );

    let namespace_prometheus_output = namespace.encode_prometheus_fmt().unwrap();
    println!("Metrics Registry in Prometheus format:");
    println!("{}", namespace_prometheus_output);

    drt.debug_prometheus_registry_keys();

    // make the ingress discoverable via a component service
    // we must first create a service, then we can attach one more more endpoints
    // attach an ingress to an engine, with the RequestHandler using the metrics struct
    let ingress = Ingress::for_engine(RequestHandler::new(metrics.clone()))?;

    namespace
        .component("component")?
        .service_builder()
        .create()
        .await?
        .endpoint("endpoint")
        .endpoint_builder()
        .stats_handler(|stats| {
            println!("stats: {:?}", stats);
            let stats = MyStats { val: 10 };
            serde_json::to_value(stats).unwrap()
        })
        .handler(ingress)
        .start()
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_system_stats_metrics_with_profiling_backend() {
        println!("=== MySystemStatsMetrics with Profiling Backend Test ===");

        // Create a distributed runtime for testing
        let runtime = Runtime::from_settings().unwrap();
        let drt = tokio::runtime::Runtime::new().unwrap().block_on(async {
            DistributedRuntime::from_settings(runtime.clone())
                .await
                .unwrap()
        });

        let drt_metrics = MySystemStatsMetrics::new(Arc::new(drt.clone())).unwrap();
        // Test the metrics functionality
        drt_metrics.request_counter.inc_by(2.0);

        let ns = drt.namespace(DEFAULT_NAMESPACE).unwrap();
        let ns_metrics = MySystemStatsMetrics::new(Arc::new(ns)).unwrap();
        ns_metrics.request_counter.inc();
        ns_metrics.request_counter.inc_by(2.0);

        drt.debug_prometheus_registry_keys();
        println!(
            "drt.encode_prometheus_fmt(): {}",
            drt.encode_prometheus_fmt().unwrap()
        );

        // Get the Prometheus metrics output for drt_metrics
        match drt_metrics.registry.encode_prometheus_fmt() {
            Ok(prometheus_text) => {
                println!("\n=== DRT PROMETHEUS METRICS OUTPUT ===");
                println!("{}", prometheus_text);
                println!("=== END DRT PROMETHEUS METRICS OUTPUT ===\n");

                // Define the expected exact Prometheus text for drt_metrics
                // Now includes both root and namespace metrics with correct prefix
                let expected_prometheus_text = r#"# HELP service_requests_total Total number of requests processed
# TYPE service_requests_total counter
service_requests_total 2
# HELP system_stats__service_requests_total Total number of requests processed
# TYPE system_stats__service_requests_total counter
system_stats__service_requests_total 3
"#;

                // Compare the entire text content for drt_metrics
                assert_eq!(
                    prometheus_text, expected_prometheus_text,
                    "\n=== COMPARISON FAILED ===\n\
                     Expected:\n{}\n\
                     Actual:\n{}\n\
                     =========================",
                    expected_prometheus_text, prometheus_text
                );

                println!("✅ All metric assertions passed for drt_metrics! Exact text match.");
            }
            Err(e) => {
                panic!("Failed to get metrics for drt_metrics: {}", e);
            }
        }

        // Get the Prometheus metrics output for ns_metrics
        match ns_metrics.registry.encode_prometheus_fmt() {
            Ok(prometheus_text) => {
                println!("\n=== PROMETHEUS METRICS OUTPUT ===");
                println!("{}", prometheus_text);
                println!("=== END PROMETHEUS METRICS OUTPUT ===\n");

                // Define the expected exact Prometheus text
                // Now uses correct system_stats prefix without leading underscore
                let expected_prometheus_text = r#"# HELP system_stats__service_requests_total Total number of requests processed
# TYPE system_stats__service_requests_total counter
system_stats__service_requests_total 3
"#;

                // Compare the entire text content
                assert_eq!(
                    prometheus_text, expected_prometheus_text,
                    "\n=== COMPARISON FAILED ===\n\
                     Expected:\n{}\n\
                     Actual:\n{}\n\
                     =========================",
                    expected_prometheus_text, prometheus_text
                );

                println!("✅ All metric assertions passed! Exact text match.");
            }
            Err(e) => {
                panic!("Failed to get metrics: {}", e);
            }
        }

        println!("=== ServiceMetrics Test Complete ===");
    }
}
