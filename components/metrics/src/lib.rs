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

//! Library functions for the metrics application.

use axum::{routing::get, Router};
use prometheus::register_gauge_vec;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use triton_distributed_llm::kv_router::protocols::ForwardPassMetrics;
use triton_distributed_llm::kv_router::scheduler::Endpoint;
use triton_distributed_llm::kv_router::scoring::ProcessedEndpoints;

use triton_distributed_runtime::{
    distributed::Component, service::EndpointInfo, utils::Duration, Result,
};

/// Configuration for LLM worker load capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMWorkerLoadCapacityConfig {
    pub component_name: String,
    pub endpoint_name: String,
}

// TODO: This is _really_ close to the async_nats::service::Stats object,
// but it's missing a few fields like "name", so use a temporary struct
// for easy deserialization. Ideally, this type already exists or can
// be exposed in the library somewhere.
/// Stats structure returned from NATS service API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsWithData {
    // Standard NATS Service API fields
    pub average_processing_time: f64,
    pub last_error: String,
    pub num_errors: u64,
    pub num_requests: u64,
    pub processing_time: u64,
    pub queue_group: String,
    // Field containing custom stats handler data
    pub data: serde_json::Value,
}

/// Prometheus metrics server for exposing metrics
pub struct PrometheusMetricsServer {
    metrics: PrometheusMetrics,
}

impl PrometheusMetricsServer {
    /// Initialize the metrics server
    pub fn new() -> Result<Self> {
        Ok(Self {
            metrics: PrometheusMetrics::new()?,
        })
    }

    /// Start the metrics server on the specified port
    pub fn start(&mut self, port: u16) {
        // Create an axum router with a metrics endpoint
        let app = Router::new().route(
            "/metrics",
            get(|| async {
                // Gather and encode metrics
                use prometheus::Encoder;
                let encoder = prometheus::TextEncoder::new();
                let mut buffer = Vec::new();
                encoder.encode(&prometheus::gather(), &mut buffer).unwrap();
                String::from_utf8(buffer).unwrap()
            }),
        );

        // Create a socket address to listen on
        let addr = SocketAddr::from(([0, 0, 0, 0], port));

        // Spawn the server in a background task
        tokio::spawn(async move {
            axum::Server::bind(&addr)
                .serve(app.into_make_service())
                .await
                .unwrap();
        });

        tracing::info!("Prometheus metrics server started at {addr:?}/metrics");
    }

    /// Update metrics with current values
    pub fn update(&mut self, config: &LLMWorkerLoadCapacityConfig, processed: &ProcessedEndpoints) {
        self.metrics.update(config, processed);
    }
}

/// Prometheus metrics collection
pub struct PrometheusMetrics {
    kv_blocks_active: prometheus::GaugeVec,
    kv_blocks_total: prometheus::GaugeVec,
    requests_active: prometheus::GaugeVec,
    requests_total: prometheus::GaugeVec,
    load_avg: prometheus::GaugeVec,
    load_std: prometheus::GaugeVec,
}

impl PrometheusMetrics {
    /// Initialize all metrics
    fn new() -> Result<Self> {
        Ok(Self {
            kv_blocks_active: register_gauge_vec!(
                "llm_kv_blocks_active",
                "Active KV cache blocks",
                &["component", "endpoint", "worker_id"]
            )?,
            kv_blocks_total: register_gauge_vec!(
                "llm_kv_blocks_total",
                "Total KV cache blocks",
                &["component", "endpoint", "worker_id"]
            )?,
            requests_active: register_gauge_vec!(
                "llm_requests_active_slots",
                "Active request slots",
                &["component", "endpoint", "worker_id"]
            )?,
            requests_total: register_gauge_vec!(
                "llm_requests_total_slots",
                "Total request slots",
                &["component", "endpoint", "worker_id"]
            )?,
            load_avg: register_gauge_vec!(
                "llm_load_avg",
                "Average load across workers",
                &["component", "endpoint"]
            )?,
            load_std: register_gauge_vec!(
                "llm_load_std",
                "Load standard deviation across workers",
                &["component", "endpoint"]
            )?,
        })
    }

    /// Helper method to set a gauge with worker-specific labels (3 labels)
    fn set_worker_gauge(
        &self,
        gauge: &prometheus::GaugeVec,
        config: &LLMWorkerLoadCapacityConfig,
        worker_id: &str,
        value: f64,
    ) {
        gauge
            .with_label_values(&[&config.component_name, &config.endpoint_name, worker_id])
            .set(value);
    }

    /// Helper method to set a gauge with component/endpoint labels only (2 labels)
    fn set_endpoint_gauge(
        &self,
        gauge: &prometheus::GaugeVec,
        config: &LLMWorkerLoadCapacityConfig,
        value: f64,
    ) {
        gauge
            .with_label_values(&[&config.component_name, &config.endpoint_name])
            .set(value);
    }

    /// Update metrics with current values
    fn update(&self, config: &LLMWorkerLoadCapacityConfig, processed: &ProcessedEndpoints) {
        // Update per-worker metrics
        for endpoint in processed.endpoints.iter() {
            let worker_id = endpoint.worker_id().to_string();
            let metrics = endpoint.data.clone();

            self.set_worker_gauge(
                &self.kv_blocks_active,
                config,
                &worker_id,
                metrics.kv_active_blocks as f64,
            );
            self.set_worker_gauge(
                &self.kv_blocks_total,
                config,
                &worker_id,
                metrics.kv_total_blocks as f64,
            );
            self.set_worker_gauge(
                &self.requests_active,
                config,
                &worker_id,
                metrics.request_active_slots as f64,
            );
            self.set_worker_gauge(
                &self.requests_total,
                config,
                &worker_id,
                metrics.request_total_slots as f64,
            );
        }

        // Update aggregate metrics
        self.set_endpoint_gauge(&self.load_avg, config, processed.load_avg);
        self.set_endpoint_gauge(&self.load_std, config, processed.load_std);
    }
}

/// Collect endpoints from a component
pub async fn collect_endpoints(
    component: &Component,
    subject: &str,
    timeout: Duration,
) -> Result<Vec<EndpointInfo>> {
    // Collect stats from each backend
    let stream = component.scrape_stats(timeout).await?;

    // Filter the stats by the service subject
    let endpoints = stream
        .into_endpoints()
        .filter(|e| e.subject.starts_with(subject))
        .collect::<Vec<_>>();
    tracing::debug!("Endpoints: {endpoints:?}");

    if endpoints.is_empty() {
        tracing::warn!("No endpoints found matching subject {subject}");
    }

    Ok(endpoints)
}

/// Extract metrics from endpoints
pub fn extract_metrics(endpoints: &[EndpointInfo]) -> Vec<ForwardPassMetrics> {
    let endpoint_data = endpoints.iter().map(|e| e.data.clone()).collect::<Vec<_>>();

    // Extract StatsWithData objects from endpoint services
    let stats: Vec<StatsWithData> = endpoint_data
        .iter()
        .filter_map(|e| {
            let metrics_data = e.as_ref()?;
            metrics_data.clone().decode::<StatsWithData>().ok()
        })
        .collect();
    tracing::debug!("Stats: {stats:?}");

    // Extract ForwardPassMetrics nested within Stats object
    let metrics: Vec<ForwardPassMetrics> = stats
        .iter()
        .filter_map(
            |s| match serde_json::from_value::<ForwardPassMetrics>(s.data.clone()) {
                Ok(metrics) => Some(metrics),
                Err(err) => {
                    tracing::warn!("Error decoding metrics: {err}");
                    None
                }
            },
        )
        .collect();
    tracing::debug!("Metrics: {metrics:?}");

    metrics
}

/// Create ProcessedEndpoints from metrics and endpoints
pub fn postprocess_metrics(
    metrics: &[ForwardPassMetrics],
    endpoints: &[EndpointInfo],
) -> ProcessedEndpoints {
    let processed_endpoints: Vec<Endpoint> = metrics
        .iter()
        .zip(endpoints.iter())
        .filter_map(|(m, e)| {
            e.id().ok().map(|id| Endpoint {
                name: format!("worker-{id}"),
                subject: e.subject.clone(),
                data: m.clone(),
            })
        })
        .collect();

    ProcessedEndpoints::new(processed_endpoints)
}
