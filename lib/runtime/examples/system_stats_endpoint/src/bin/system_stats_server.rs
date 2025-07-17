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

use http_server_metrics::{MyStats, DEFAULT_NAMESPACE};

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

use prometheus::{Counter, Gauge, Histogram};
use std::sync::Arc;

/// Service metrics struct using the metric classes from profiling.rs
pub struct MySystemStatsMetrics {
    drt: Arc<dyn MetricsRegistry>,
    pub request_counter: Arc<Counter>,
    pub active_requests_gauge: Arc<Gauge>,
    pub request_duration_histogram: Arc<Histogram>,
}

impl MetricsRegistry for MySystemStatsMetrics {
    fn metrics_prefix(&self) -> String {
        "example_system_stats".to_string()
    }

    fn metrics_hierarchy(&self) -> Vec<String> {
        vec![self.metrics_prefix()]
    }

    fn drt(&self) -> &crate::DistributedRuntime {
        self.drt.drt()
    }
}

impl MySystemStatsMetrics {
    /// Create a new ServiceMetrics instance using the metric backend
    pub fn new(
        drt: Arc<dyn MetricsRegistry>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        Self::create_with_registry(drt)
    }

    fn create_with_registry(
        registry: Arc<dyn MetricsRegistry>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let request_counter = registry.create_counter(
            "service_requests_total",
            "Total number of requests processed",
            &[("service", "backend")],
        )?;
        let active_requests_gauge = registry.create_gauge(
            "service_active_requests",
            "Number of requests currently being processed",
            &[("service", "backend")],
        )?;
        let request_duration_histogram = registry.create_histogram(
            "service_request_duration_seconds",
            "Request duration in seconds",
            &[("service", "backend")],
        )?;

        Ok(Self {
            drt: registry,
            request_counter,
            active_requests_gauge,
            request_duration_histogram,
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
        self.metrics.active_requests_gauge.inc();

        let (data, ctx) = input.into_parts();

        let chars = data
            .chars()
            .map(|c| Annotated::from_data(c.to_string()))
            .collect::<Vec<_>>();

        let stream = stream::iter(chars);

        // Calculate duration
        let duration = start_time.elapsed().as_secs_f64();

        // Record request end
        self.metrics.active_requests_gauge.dec();
        self.metrics.request_duration_histogram.observe(duration);
        // self.metrics.response_size_histogram.observe(response_size); // This line was removed

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

async fn backend(drt: DistributedRuntime) -> Result<()> {
    /*
    service, <namespace>_<service>__<metric_name>
    component, <namespace>_<component>__<metric_name>
    endpoint, <namespace>_<service>_<component>_<endpoint>__<metric_name>
        */
    let namespace = drt.namespace(DEFAULT_NAMESPACE)?;

    // Get the metrics backend from the runtime
    let metrics_registry = namespace.metrics_registry();
    // Initialize metrics using the profiling-based struct
    let metrics = Arc::new(
        MySystemStatsMetrics::new(metrics_registry.clone()).map_err(|e| Error::msg(e.to_string()))?,
    );

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

        // Create MySystemStatsMetrics using the new struct
        let metrics = MySystemStatsMetrics::new(Arc::new(drt)).unwrap();

        println!("Created MySystemStatsMetrics with profiling backend");

        // Test the metrics functionality
        metrics.request_counter.inc();
        metrics.request_counter.inc_by(2.0);
        metrics.active_requests_gauge.set(5.0);
        metrics.active_requests_gauge.inc();
        metrics.request_duration_histogram.observe(0.1);
        metrics.request_duration_histogram.observe(0.25);

        // Get the Prometheus metrics output
        match metrics.drt.encode_prometheus_str() {
            Ok(prometheus_text) => {
                println!("\n=== PROMETHEUS METRICS OUTPUT ===");
                println!("{}", prometheus_text);
                println!("=== END PROMETHEUS METRICS OUTPUT ===\n");

                // Define the expected exact Prometheus text
                let expected_prometheus_text = r#"# HELP service_active_requests Number of requests currently being processed
# TYPE service_active_requests gauge
service_active_requests 6
# HELP service_request_duration_seconds Request duration in seconds
# TYPE service_request_duration_seconds histogram
service_request_duration_seconds_bucket{le="0.005"} 0
service_request_duration_seconds_bucket{le="0.01"} 0
service_request_duration_seconds_bucket{le="0.025"} 0
service_request_duration_seconds_bucket{le="0.05"} 0
service_request_duration_seconds_bucket{le="0.1"} 1
service_request_duration_seconds_bucket{le="0.25"} 2
service_request_duration_seconds_bucket{le="0.5"} 2
service_request_duration_seconds_bucket{le="1"} 2
service_request_duration_seconds_bucket{le="2.5"} 2
service_request_duration_seconds_bucket{le="5"} 2
service_request_duration_seconds_bucket{le="10"} 2
service_request_duration_seconds_bucket{le="+Inf"} 2
service_request_duration_seconds_sum 0.35
service_request_duration_seconds_count 2
# HELP service_requests_total Total number of requests processed
# TYPE service_requests_total counter
service_requests_total 3
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
