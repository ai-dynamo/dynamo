// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::{
    DistributedRuntime,
    metrics::MetricsHierarchy,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait, network::Ingress,
    },
    protocols::annotated::Annotated,
    stream,
};
use prometheus::IntCounter;
use std::sync::Arc;

pub const DEFAULT_NAMESPACE: &str = "dyn_example_namespace";
pub const DEFAULT_COMPONENT: &str = "dyn_example_component";
pub const DEFAULT_ENDPOINT: &str = "dyn_example_endpoint";

/// Stats structure returned by the endpoint's stats handler
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct MyStats {
    // Example value for demonstration purposes
    pub val: i32,
}

/// Custom metrics for system stats with data bytes tracking
#[derive(Clone, Debug)]
pub struct MySystemStatsMetrics {
    pub data_bytes_processed: IntCounter,
}

impl MySystemStatsMetrics {
    pub fn from_endpoint(endpoint: &dynamo_runtime::component::Endpoint) -> anyhow::Result<Self> {
        let data_bytes_processed = endpoint.metrics().create_intcounter(
            "my_custom_bytes_processed_total",
            "Example of a custom metric. Total number of data bytes processed by system handler",
            &[],
        )?;

        Ok(Self {
            data_bytes_processed,
        })
    }
}

#[derive(Clone)]
pub struct RequestHandler {
    metrics: Option<MySystemStatsMetrics>,
}

impl RequestHandler {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { metrics: None })
    }

    pub fn with_metrics(metrics: MySystemStatsMetrics) -> Arc<Self> {
        Arc::new(Self {
            metrics: Some(metrics),
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for RequestHandler {
    async fn generate(
        &self,
        input: SingleIn<String>,
    ) -> anyhow::Result<ManyOut<Annotated<String>>> {
        let (data, ctx) = input.into_parts();

        // Track data bytes processed if metrics are available
        if let Some(metrics) = &self.metrics {
            metrics.data_bytes_processed.inc_by(data.len() as u64);
        }

        let chars = data
            .chars()
            .map(|c| Annotated::from_data(c.to_string()))
            .collect::<Vec<_>>();

        let stream = stream::iter(chars);

        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

/// Backend function that sets up the system status server with metrics and ingress handler
/// This function can be reused by integration tests to ensure they use the exact same setup
pub async fn backend(drt: DistributedRuntime, endpoint_name: Option<&str>) -> anyhow::Result<()> {
    let endpoint_name = endpoint_name.unwrap_or(DEFAULT_ENDPOINT);

    let mut component = drt
        .namespace(DEFAULT_NAMESPACE)?
        .component(DEFAULT_COMPONENT)?;
    component.add_stats_service().await?;
    let endpoint = component.endpoint(endpoint_name);

    // Create custom metrics for system stats
    let system_metrics =
        MySystemStatsMetrics::from_endpoint(&endpoint).expect("Failed to create system metrics");

    // Use the factory pattern - single line factory call with metrics
    let ingress = Ingress::for_engine(RequestHandler::with_metrics(system_metrics))?;

    endpoint
        .endpoint_builder()
        .stats_handler(|_stats| {
            println!("Stats handler called with stats: {:?}", _stats);
            // TODO(keivenc): return a real stats object
            let stats = MyStats { val: 10 };
            serde_json::to_value(stats).unwrap()
        })
        .handler(ingress)
        .start()
        .await?;

    Ok(())
}
