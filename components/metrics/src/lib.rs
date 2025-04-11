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
//!
//! This library provides functionality to expose Prometheus metrics either through a local HTTP server
//! or by pushing to a Prometheus PushGateway.
//!
//! # Examples
//!
//! ## Using the metrics pull mode
//! ```no_run
//! use metrics::{PrometheusMetricsCollector, MetricsMode};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut collector = PrometheusMetricsCollector::new()?;
//!
//!     // Start a metrics server with default values
//!     collector.start(MetricsMode::default())?;
//!
//!     // Or explicitly specify values
//!     collector.start(MetricsMode::Pull {
//!         host: "127.0.0.1".to_string(),
//!         port: 9090,
//!     })?;
//!
//!     // Or use the convenience constructor
//!     collector.start(MetricsMode::new_pull())?;
//!
//!     // Your application code here
//!     tokio::signal::ctrl_c().await?;
//!
//!     // Stop the metrics server gracefully
//!     collector.stop();
//!     Ok(())
//! }
//! ```
//!
//! ## Using the Push mode
//! ```no_run
//! use metrics::{PrometheusMetricsCollector, MetricsMode};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut collector = PrometheusMetricsCollector::new()?;
//!
//!     // Start pushing metrics to a Prometheus PushGateway with default values
//!     collector.start(MetricsMode::new_push())?;
//!
//!     // Or explicitly specify values
//!     collector.start(MetricsMode::Push {
//!         host: "127.0.0.1".to_string(),
//!         port: 9091,
//!         job: "custom_job".to_string(),
//!         interval: 5, // Push every 5 seconds
//!     })?;
//!
//!     // Your application code here
//!     tokio::signal::ctrl_c().await?;
//!
//!     // Stop pushing metrics gracefully
//!     collector.stop();
//!     Ok(())
//! }

use axum::{routing::get, Router};
use prometheus::{register_counter_vec, register_gauge_vec, Encoder, TextEncoder};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::Duration as StdDuration;

use dynamo_llm::kv_router::protocols::ForwardPassMetrics;
use dynamo_llm::kv_router::scheduler::Endpoint;
use dynamo_llm::kv_router::scoring::ProcessedEndpoints;

use dynamo_runtime::{
    distributed::Component, error, service::EndpointInfo, utils::Duration, Result,
};

/// Configuration for metrics collection mode
#[derive(Debug, Clone)]
pub enum MetricsMode {
    /// Host a Prometheus metrics server for pull-based collection
    Pull {
        /// Host to listen on (e.g. "0.0.0.0")
        host: String,
        /// Port to listen on (e.g. 9091)
        port: u16,
    },
    /// Push to a Prometheus PushGateway
    Push {
        /// PushGateway host (e.g. "http://localhost")
        host: String,
        /// PushGateway port (e.g. 9091)
        port: u16,
        /// Job name for the metrics
        job: String,
        /// Push interval in seconds
        interval: u64,
    },
}

impl Default for MetricsMode {
    fn default() -> Self {
        Self::new_pull()
    }
}

impl MetricsMode {
    /// Create a new Pull mode with default values
    pub fn new_pull() -> Self {
        Self::Pull {
            host: "0.0.0.0".to_string(),
            port: 9091,
        }
    }

    /// Create a new Push mode with default values
    pub fn new_push() -> Self {
        Self::Push {
            host: "127.0.0.1".to_string(),
            port: 9091,
            job: "dynamo_metrics".to_string(),
            interval: 2,
        }
    }
}

/// Configuration for LLM worker load capacity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMWorkerLoadCapacityConfig {
    pub component_name: String,
    pub endpoint_name: String,
}

/// Metrics collector for exposing metrics to prometheus/grafana
pub struct PrometheusMetricsCollector {
    metrics: PrometheusMetrics,
    mode: Option<MetricsMode>,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl PrometheusMetricsCollector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            metrics: PrometheusMetrics::new()?,
            mode: None,
            shutdown_tx: None,
        })
    }

    /// Start metrics collection with the specified mode
    pub fn start(&mut self, mode: MetricsMode) -> Result<()> {
        // Store the mode
        self.mode = Some(mode.clone());

        match mode {
            MetricsMode::Pull { host, port } => self.start_pull_mode(host, port),
            MetricsMode::Push {
                host,
                port,
                job,
                interval,
            } => self.start_push_mode(host, port, job, interval),
        }
    }

    /// Stop metrics collection
    pub fn stop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }

    /// Start a metrics server for pull-based collection on the specified port
    fn start_pull_mode(&mut self, host: String, port: u16) -> Result<()> {
        // Create an axum router with a metrics endpoint
        let app = Router::new().route(
            "/metrics",
            get(|| async {
                // Gather and encode metrics
                let encoder = TextEncoder::new();
                let mut buffer = Vec::new();
                encoder.encode(&prometheus::gather(), &mut buffer).unwrap();
                String::from_utf8(buffer).unwrap()
            }),
        );

        // Create a socket address to listen on
        let ip_addr = host.parse().map_err(|e| {
            error!("Failed to parse host '{}' as IP address: {}. Use a valid IPv4 or IPv6 address (e.g. '0.0.0.0' or '127.0.0.1')", host, e)
        })?;
        let addr = SocketAddr::new(ip_addr, port);

        // Create shutdown channel
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(tx);

        // Try to bind to the address first to fail early if it's not available
        let server = match axum::Server::try_bind(&addr) {
            Ok(server) => server,
            Err(e) => {
                return Err(error!(
                    "Failed to bind to address {}: {}. The port may be in use.",
                    addr, e
                ));
            }
        };

        // Spawn the server in a background task
        tokio::spawn(async move {
            let server = server.serve(app.into_make_service());

            // Create a future that completes when shutdown signal is received
            let shutdown_future = async {
                rx.await.ok();
            };

            // Run the server with graceful shutdown
            tokio::select! {
                result = server => {
                    if let Err(e) = result {
                        tracing::error!("Metrics server error: {}", e);
                    }
                },
                _ = shutdown_future => {
                    tracing::info!("Metrics server shutting down gracefully");
                },
            }
        });

        tracing::info!("Prometheus metrics server started at {addr}/metrics");
        Ok(())
    }

    /// Start pushing metrics to a Prometheus PushGateway
    fn start_push_mode(
        &mut self,
        host: String,
        port: u16,
        job: String,
        interval: u64,
    ) -> Result<()> {
        // Create shutdown channel
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        self.shutdown_tx = Some(tx);

        // Create HTTP client
        let client = Client::new();
        let url = format!("http://{host}:{port}/metrics/job/{job}");
        let url_clone = url.clone();
        let interval_duration = StdDuration::from_secs(interval);

        // Spawn background task to periodically push metrics
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval_duration);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Gather and encode metrics
                        let encoder = TextEncoder::new();
                        let mut buffer = Vec::new();
                        if let Err(e) = encoder.encode(&prometheus::gather(), &mut buffer) {
                            tracing::error!("Failed to encode metrics: {}", e);
                            continue;
                        }

                        // Push metrics to the gateway
                        match client.post(&url)
                            .header("Content-Type", encoder.format_type())
                            .body(buffer)
                            .send()
                            .await
                        {
                            Ok(response) => {
                                if response.status().is_success() {
                                    tracing::debug!("Successfully pushed metrics to PushGateway");
                                } else {
                                    tracing::error!(
                                        "Failed to push metrics to PushGateway. Status: {}, Error: {:?}",
                                        response.status(),
                                        response.text().await
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::error!("Failed to push metrics to PushGateway: {}", e);
                            }
                        }
                    }
                    _ = &mut rx => {
                        tracing::info!("Stopping metrics push task");
                        break;
                    }
                }
            }
        });

        tracing::info!(
            "Started pushing metrics to PushGateway at '{url_clone}' with job name '{job}'"
        );
        Ok(())
    }

    /// Update metrics with current values
    pub fn update(&mut self, config: &LLMWorkerLoadCapacityConfig, processed: &ProcessedEndpoints) {
        self.metrics.update(config, processed);
    }

    /// Update KV hit rate metrics
    pub fn update_kv_hit_rate(
        &mut self,
        config: &LLMWorkerLoadCapacityConfig,
        worker_id: i64,
        isl_blocks: usize,
        overlap_blocks: usize,
    ) {
        self.metrics
            .update_kv_hit_rate(config, worker_id, isl_blocks, overlap_blocks);
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
    // KV hit rate metrics
    kv_hit_rate_percent: prometheus::GaugeVec,
    // FIXME: These are currently unused outside of mock_worker
    kv_hit_rate_isl_blocks: prometheus::CounterVec,
    kv_hit_rate_overlap_blocks: prometheus::CounterVec,
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
            // KV hit rate (ForwardPassMetrics)
            kv_hit_rate_percent: register_gauge_vec!(
                "llm_kv_hit_rate_percent",
                "KV hit rate percentage per worker",
                &["component", "endpoint", "worker_id"]
            )?,
            // FIXME: Cleanup/remove event based metrics after finalizaing
            //        metrics collection approach with vllm/trtllm workers.
            // Event-based KV hit rate metrics (not currently used outside mock worker)
            kv_hit_rate_isl_blocks: register_counter_vec!(
                "llm_kv_hit_rate_isl_blocks",
                "Cumulative count of ISL blocks in KV hit rate events",
                &["component", "endpoint", "worker_id"]
            )?,
            kv_hit_rate_overlap_blocks: register_counter_vec!(
                "llm_kv_hit_rate_overlap_blocks",
                "Cumulative count of overlapping blocks in KV hit rate events",
                &["component", "endpoint", "worker_id"]
            )?,
        })
    }

    /// Helper method to set a gauge with worker-specific labels (3 labels)
    fn set_worker_gauge(
        &self,
        gauge: &prometheus::GaugeVec,
        config: &LLMWorkerLoadCapacityConfig,
        worker_id: &String,
        value: f64,
    ) {
        gauge
            .with_label_values(&[&config.component_name, &config.endpoint_name, worker_id])
            .set(value);
    }

    /// Helper method to increment a counter with worker-specific labels (3 labels)
    fn increment_worker_counter(
        &self,
        counter: &prometheus::CounterVec,
        config: &LLMWorkerLoadCapacityConfig,
        worker_id: &String,
        value: f64,
    ) {
        counter
            .with_label_values(&[&config.component_name, &config.endpoint_name, worker_id])
            .inc_by(value);
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
        for (worker_id, endpoint) in processed.endpoints.iter() {
            let worker_id = worker_id.to_string();
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
            self.set_worker_gauge(
                &self.kv_hit_rate_percent,
                config,
                &worker_id,
                metrics.gpu_prefix_cache_hit_rate as f64,
            );
        }

        // Update aggregate metrics
        self.set_endpoint_gauge(&self.load_avg, config, processed.load_avg);
        self.set_endpoint_gauge(&self.load_std, config, processed.load_std);
    }

    /// Update KV hit rate metrics
    pub fn update_kv_hit_rate(
        &self,
        config: &LLMWorkerLoadCapacityConfig,
        worker_id: i64,
        isl_blocks: usize,
        overlap_blocks: usize,
    ) {
        let worker_id_str = worker_id.to_string();

        // Increment the ISL blocks and overlap blocks counters
        self.increment_worker_counter(
            &self.kv_hit_rate_isl_blocks,
            config,
            &worker_id_str,
            isl_blocks as f64,
        );

        self.increment_worker_counter(
            &self.kv_hit_rate_overlap_blocks,
            config,
            &worker_id_str,
            overlap_blocks as f64,
        );

        // TODO: The cumulative hit rate percentage can probably be computed by consumers
        // of Prometheus metrics like Grafana instead, but we'll compute it here for now
        // for convenient debugging/logging.
        // Calculate and set the cumulative hit rate percentage
        let cumulative_isl = self
            .kv_hit_rate_isl_blocks
            .with_label_values(&[
                &config.component_name,
                &config.endpoint_name,
                &worker_id_str,
            ])
            .get();

        let cumulative_overlap = self
            .kv_hit_rate_overlap_blocks
            .with_label_values(&[
                &config.component_name,
                &config.endpoint_name,
                &worker_id_str,
            ])
            .get();

        if cumulative_isl > 0.0 {
            let cumulative_hit_rate = (cumulative_overlap / cumulative_isl) * 100.0;
            tracing::debug!(
                "Estimated Cumulative KV hit rate: {cumulative_hit_rate:.2}% (Overlap: {cumulative_overlap} / ISL: {cumulative_isl})"
            );
        }
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
    Ok(endpoints)
}

/// Extract metrics from endpoints
pub fn extract_metrics(endpoints: &[EndpointInfo]) -> Vec<ForwardPassMetrics> {
    let endpoint_data = endpoints.iter().map(|e| e.data.clone()).collect::<Vec<_>>();

    // Extract ForwardPassMetrics objects from endpoint services
    let metrics: Vec<ForwardPassMetrics> = endpoint_data
        .iter()
        .filter_map(|e| {
            let metrics_data = e.as_ref()?;

            match metrics_data.clone().decode::<ForwardPassMetrics>() {
                Ok(stats) => Some(stats),
                Err(err) => {
                    tracing::error!(
                        "Failed to decode ForwardPassMetrics data: {}. Raw data: {:?}",
                        err,
                        metrics_data
                    );
                    None
                }
            }
        })
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
