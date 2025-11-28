// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// DEPRECATED: To be removed after custom backends migrate to Dynamo backend.
//
// Custom backend metrics polling and collection.
//
// This module provides a bridge to poll metrics from custom backends (like NIM) that expose
// their own metrics endpoints, and makes them available through Prometheus.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::Deserialize;

/// Maximum number of custom backend gauges that can be registered to prevent unbounded growth.
pub const MAX_CUSTOM_BACKEND_GAUGES: usize = 100;

/// Registry for custom backend metrics discovered at runtime.
///
/// Metrics from custom backends are exposed as Prometheus gauges since we're setting
/// absolute values received from polling, not incrementing them locally.
///
/// All metrics are automatically prefixed when registered. For example, if the prefix is
/// `dynamo_component` and a backend reports a gauge named `kv_cache_usage_perc`, it will
/// be exposed as `dynamo_component_kv_cache_usage_perc` in Prometheus metrics.
pub struct CustomBackendMetricsRegistry {
    gauges: Mutex<HashMap<String, prometheus::Gauge>>,
    prefix: String,
    prometheus_registry: prometheus::Registry,
}

impl CustomBackendMetricsRegistry {
    pub fn new(prefix: String, prometheus_registry: prometheus::Registry) -> Self {
        Self {
            gauges: Mutex::new(HashMap::new()),
            prefix,
            prometheus_registry,
        }
    }

    /// Get or create a gauge for the given metric name, registering it with Prometheus if new.
    /// Returns None if the maximum number of gauges has been reached.
    fn get_or_create_gauge(&self, name: &str) -> Option<prometheus::Gauge> {
        let mut gauges = self.gauges.lock().unwrap();

        if let Some(gauge) = gauges.get(name) {
            return Some(gauge.clone());
        }

        // Cap the number of gauges to prevent unbounded growth
        if gauges.len() >= MAX_CUSTOM_BACKEND_GAUGES {
            tracing::warn!(
                "Maximum number of custom backend gauges ({}) reached, dropping metric: {}",
                MAX_CUSTOM_BACKEND_GAUGES,
                name
            );
            return None;
        }

        let full_name = format!("{}_{}", self.prefix, name);
        let gauge = prometheus::Gauge::new(full_name.as_str(), name)
            .unwrap_or_else(|e| panic!("Failed to create gauge {}: {}", full_name, e));

        if let Err(e) = self.prometheus_registry.register(Box::new(gauge.clone())) {
            tracing::warn!(
                "Failed to register custom backend gauge {}: {}",
                full_name,
                e
            );
        }

        gauges.insert(name.to_string(), gauge.clone());
        Some(gauge)
    }

    /// Update a gauge metric with a new value.
    pub fn set_gauge(&self, name: &str, value: f64) {
        if let Some(gauge) = self.get_or_create_gauge(name) {
            gauge.set(value);
        }
    }
}

/// Response format from custom backend runtime_stats endpoint
#[derive(Debug, Deserialize)]
struct CustomBackendStatsResponse {
    metrics: CustomBackendMetrics,
}

#[derive(Debug, Deserialize)]
struct CustomBackendMetrics {
    gauges: HashMap<String, f64>,
}

/// Spawn a background task that polls custom backend metrics periodically.
///
/// All metrics collected from the backend will be prefixed according to the registry's prefix
/// (typically `dynamo_component_`). For example, a backend gauge `kv_cache_usage_perc` will
/// appear as `dynamo_component_kv_cache_usage_perc` in Prometheus.
///
/// This task does not use a CancellationToken for graceful shutdown. When the executable exits,
/// the task is abruptly terminated by the tokio runtime shutdown. This is acceptable because
/// metrics polling is non-critical with no risk of data corruption or resource leaks, typical
/// polling intervals are short, and the Worker already has a graceful shutdown timeout mechanism.
pub fn spawn_custom_backend_polling_task(
    drt: dynamo_runtime::DistributedRuntime,
    namespace_component_endpoint: String,
    polling_interval_secs: f64,
    registry: Arc<CustomBackendMetricsRegistry>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        tracing::info!(
            namespace_component_endpoint=%namespace_component_endpoint,
            interval_secs=polling_interval_secs,
            "Starting custom backend metrics polling"
        );

        // Parse namespace.component.endpoint format
        let parts: Vec<&str> = namespace_component_endpoint.split('.').collect();
        if parts.len() != 3 {
            tracing::error!(
                namespace_component_endpoint=%namespace_component_endpoint,
                "Invalid endpoint format, expected 'namespace.component.endpoint'"
            );
            return;
        }
        let (namespace, component_name, endpoint_name) = (parts[0], parts[1], parts[2]);

        // Get namespace, component, and endpoint from DRT
        let Ok(ns) = drt.namespace(namespace.to_string()) else {
            tracing::error!("Namespace not available: {}", namespace);
            return;
        };
        let Ok(component) = ns.component(component_name) else {
            tracing::error!("Component not available: {}", component_name);
            return;
        };
        let endpoint = component.endpoint(endpoint_name);

        // Wait for client to be ready (backend might not be available yet)
        let client = loop {
            match endpoint.client().await {
                Ok(client) => break client,
                Err(e) => {
                    tracing::warn!(
                        error=%e,
                        namespace=%namespace,
                        component=%component_name,
                        endpoint=%endpoint_name,
                        "Failed to create client for custom backend endpoint, retrying in 5s"
                    );
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
        };

        // Create router for sending requests to the backend
        use dynamo_runtime::pipeline::{PushRouter, RouterMode};
        use dynamo_runtime::protocols::annotated::Annotated;
        let Ok(router) =
            PushRouter::<String, Annotated<String>>::from_client(client, RouterMode::Random).await
        else {
            tracing::error!(
                namespace=%namespace,
                component=%component_name,
                endpoint=%endpoint_name,
                "Failed to create router for custom backend endpoint"
            );
            return;
        };

        tracing::info!(
            namespace=%namespace,
            component=%component_name,
            endpoint=%endpoint_name,
            "Custom backend metrics polling started"
        );

        // Poll backend at regular intervals
        let interval = Duration::from_secs_f64(polling_interval_secs);
        loop {
            tokio::time::sleep(interval).await;

            match poll_backend_once(&router, &registry).await {
                Ok(num_metrics) => {
                    tracing::debug!(
                        num_metrics=%num_metrics,
                        "Successfully polled custom backend metrics"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        error=%e,
                        "Failed to poll custom backend metrics"
                    );
                }
            }
        }
    })
}

/// Poll the backend once and update the registry.
async fn poll_backend_once(
    router: &dynamo_runtime::pipeline::PushRouter<
        String,
        dynamo_runtime::protocols::annotated::Annotated<String>,
    >,
    registry: &Arc<CustomBackendMetricsRegistry>,
) -> anyhow::Result<usize> {
    use dynamo_runtime::pipeline::Context;

    let response_stream = router.random(Context::new("".to_string())).await?;

    // Collect responses from the stream
    let mut responses = Vec::new();
    {
        use futures::StreamExt;
        let mut stream = response_stream;
        while let Some(response) = stream.next().await {
            if let Some(data) = response.data {
                responses.push(data);
            }
        }
    }

    if responses.is_empty() {
        anyhow::bail!("No responses received from custom backend");
    }

    // Parse the first response as JSON
    // Expected format from backend (as JSON string):
    // {
    //   "schema_version": 1,
    //   "worker_id": "mock-worker-1",
    //   "backend": "vllm",
    //   "ts": 1759967807,
    //   "metrics": {
    //     "gauges": {
    //       "kv_cache_usage_perc": 0.3,
    //       "gpu_utilization_perc": 75.5,
    //       "active_requests": 5
    //     }
    //   }
    // }
    let stats: CustomBackendStatsResponse = serde_json::from_str(&responses[0])
        .map_err(|e| anyhow::anyhow!("Failed to parse backend stats JSON: {}", e))?;

    // Update gauges in the registry
    for (name, value) in &stats.metrics.gauges {
        registry.set_gauge(name, *value);
    }

    Ok(stats.metrics.gauges.len())
}
