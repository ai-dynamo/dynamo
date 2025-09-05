// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo

#![allow(dead_code)]
#![allow(unused_imports)]

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
    time::{Duration, Instant},
};

pub use anyhow::{
    Context as ErrorContext, Error, Ok as OK, Result, anyhow as error, bail as raise,
};

use async_once_cell::OnceCell;

mod config;
pub use config::RuntimeConfig;

pub mod component;
pub mod discovery;
pub mod engine;
pub mod health_check;
pub mod system_status_server;
pub use system_status_server::SystemStatusServerInfo;
pub mod instances;
pub mod logging;
pub mod metrics;
pub mod pipeline;
pub mod prelude;
pub mod protocols;
pub mod runnable;
pub mod runtime;
pub mod service;
pub mod slug;
pub mod storage;
pub mod traits;
pub mod transports;
pub mod utils;
pub mod worker;

pub mod distributed;
pub use distributed::distributed_test_utils;
pub use futures::stream;
pub use tokio_util::sync::CancellationToken;
pub use worker::Worker;

use crate::metrics::prometheus_names::distributed_runtime;

use component::{Endpoint, InstanceSource};
use utils::GracefulShutdownTracker;

use config::HealthStatus;

/// Types of Tokio runtimes that can be used to construct a Dynamo [Runtime].
#[derive(Clone)]
enum RuntimeType {
    Shared(Arc<tokio::runtime::Runtime>),
    External(tokio::runtime::Handle),
}

/// Local [Runtime] which provides access to shared resources local to the physical node/machine.
#[derive(Debug, Clone)]
pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,
    secondary: RuntimeType,
    cancellation_token: CancellationToken,
    endpoint_shutdown_token: CancellationToken,
    graceful_shutdown_tracker: Arc<GracefulShutdownTracker>,
}

/// Information about an endpoint's health status and response tracking
#[derive(Clone, Debug)]
pub struct EndpointHealthInfo {
    /// Current health status of the endpoint
    pub status: HealthStatus,
    /// Last time the endpoint responded (if any)
    pub last_response_time: Option<Instant>,
}

impl EndpointHealthInfo {
    /// Create a new EndpointHealthInfo with the given status and no response time
    pub fn new(status: HealthStatus) -> Self {
        Self {
            status,
            last_response_time: None,
        }
    }

    /// Check if the endpoint has responded recently (within the given duration)
    pub fn has_responded_recently(&self, within: Duration) -> bool {
        self.last_response_time
            .map(|t| t.elapsed() < within)
            .unwrap_or(false)
    }

    /// Update the last response time to now
    pub fn update_response_time(&mut self) {
        self.last_response_time = Some(Instant::now());
    }
}

/// Current Health Status
/// If use_endpoint_health_status is set then
/// initialize the endpoint_health hashmap to the
/// starting health status
#[derive(Clone)]
pub struct SystemHealth {
    system_health: HealthStatus,
    endpoint_health: Arc<std::sync::RwLock<HashMap<String, EndpointHealthInfo>>>,
    /// Maps endpoint subject to health check payload
    health_check_payloads: Arc<std::sync::RwLock<HashMap<String, serde_json::Value>>>,
    use_endpoint_health_status: Vec<String>,
    health_path: String,
    live_path: String,
    start_time: Instant,
    uptime_gauge: OnceLock<prometheus::Gauge>,
    /// Notifier for health check timer resets
    health_check_notifier: OnceLock<Arc<tokio::sync::Notify>>,
}

impl SystemHealth {
    pub fn new(
        starting_health_status: HealthStatus,
        use_endpoint_health_status: Vec<String>,
        health_path: String,
        live_path: String,
    ) -> Self {
        let mut endpoint_health = HashMap::new();
        for endpoint in &use_endpoint_health_status {
            endpoint_health.insert(
                endpoint.clone(),
                EndpointHealthInfo::new(starting_health_status.clone()),
            );
        }
        SystemHealth {
            system_health: starting_health_status,
            endpoint_health: Arc::new(std::sync::RwLock::new(endpoint_health)),
            health_check_payloads: Arc::new(std::sync::RwLock::new(HashMap::new())),
            use_endpoint_health_status,
            health_path,
            live_path,
            start_time: Instant::now(),
            uptime_gauge: OnceLock::new(),
            health_check_notifier: OnceLock::new(),
        }
    }
    pub fn set_health_status(&mut self, status: HealthStatus) {
        self.system_health = status;
    }

    pub fn set_endpoint_health_status(&self, endpoint: &str, status: HealthStatus) {
        let mut endpoint_health = self.endpoint_health.write().unwrap();
        endpoint_health
            .entry(endpoint.to_string())
            .or_insert_with(|| EndpointHealthInfo::new(status.clone()))
            .status = status.clone();
    }

    /// Update the last response time for an endpoint
    pub fn update_last_response_time(&self, endpoint: &str) {
        let mut endpoint_health = self.endpoint_health.write().unwrap();
        endpoint_health
            .entry(endpoint.to_string())
            .or_insert_with(|| EndpointHealthInfo::new(HealthStatus::Ready))
            .update_response_time();
    }

    /// Get the last response time for an endpoint
    pub fn get_last_response_time(&self, endpoint: &str) -> Option<Instant> {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health
            .get(endpoint)
            .and_then(|info| info.last_response_time)
    }

    /// Check if an endpoint has responded recently (within the given duration)
    pub fn has_responded_recently(&self, endpoint: &str, within: std::time::Duration) -> bool {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health
            .get(endpoint)
            .map(|info| info.has_responded_recently(within))
            .unwrap_or(false)
    }

    /// Returns the overall health status and endpoint health statuses
    pub fn get_health_status(&self) -> (bool, HashMap<String, String>) {
        let endpoint_health = self.endpoint_health.read().unwrap();
        let mut endpoints: HashMap<String, String> = HashMap::new();

        for (endpoint, info) in endpoint_health.iter() {
            endpoints.insert(
                endpoint.clone(),
                if info.status == HealthStatus::Ready {
                    "ready".to_string()
                } else {
                    "notready".to_string()
                },
            );
        }

        let healthy = if !self.use_endpoint_health_status.is_empty() {
            self.use_endpoint_health_status.iter().all(|endpoint| {
                endpoint_health
                    .get(endpoint)
                    .is_some_and(|info| info.status == HealthStatus::Ready)
            })
        } else {
            self.system_health == HealthStatus::Ready
        };

        (healthy, endpoints)
    }

    /// Get detailed health info for an endpoint
    pub fn get_endpoint_health_info(&self, endpoint: &str) -> Option<EndpointHealthInfo> {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health.get(endpoint).cloned()
    }

    /// Register a health check payload for an endpoint
    pub fn register_health_check_payload(&self, endpoint: &str, payload: serde_json::Value) {
        let mut payloads = self.health_check_payloads.write().unwrap();
        payloads.insert(endpoint.to_string(), payload);

        // Also initialize endpoint health info if needed
        let mut endpoint_health = self.endpoint_health.write().unwrap();
        endpoint_health
            .entry(endpoint.to_string())
            .or_insert_with(|| EndpointHealthInfo::new(HealthStatus::Ready));
    }

    /// Get all health check payloads
    pub fn get_health_check_payloads(&self) -> Vec<(String, serde_json::Value)> {
        let payloads = self.health_check_payloads.read().unwrap();
        payloads
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Check if any health check payloads are registered
    pub fn has_health_check_payloads(&self) -> bool {
        let payloads = self.health_check_payloads.read().unwrap();
        !payloads.is_empty()
    }

    /// Get list of endpoints with health check payloads
    pub fn get_health_check_endpoints(&self) -> Vec<String> {
        let payloads = self.health_check_payloads.read().unwrap();
        payloads.keys().cloned().collect()
    }

    /// Get health check payload for a specific endpoint
    pub fn get_health_check_payload(&self, endpoint: &str) -> Option<serde_json::Value> {
        let payloads = self.health_check_payloads.read().unwrap();
        payloads.get(endpoint).cloned()
    }

    /// Remove health check payload for an endpoint
    pub fn remove_health_check_payload(&self, endpoint: &str) {
        let mut payloads = self.health_check_payloads.write().unwrap();
        payloads.remove(endpoint);
    }

    /// Get the endpoint health status (Ready/NotReady)
    pub fn get_endpoint_health_status(&self, endpoint: &str) -> Option<HealthStatus> {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health
            .get(endpoint)
            .map(|info| info.status.clone())
    }

    /// Set the health check notifier
    pub fn set_health_check_notifier(
        &self,
        notifier: Arc<tokio::sync::Notify>,
    ) -> anyhow::Result<()> {
        self.health_check_notifier
            .set(notifier)
            .map_err(|_| anyhow::anyhow!("Health check notifier already set"))?;
        Ok(())
    }

    /// Get the health check notifier
    pub fn get_health_check_notifier(&self) -> Option<Arc<tokio::sync::Notify>> {
        self.health_check_notifier.get().cloned()
    }

    /// Get the response times map (for testing)
    pub fn get_response_times(&self) -> Arc<std::sync::Mutex<HashMap<String, Instant>>> {
        // Create a new map with current response times
        let endpoint_health = self.endpoint_health.read().unwrap();
        let mut times = HashMap::new();
        for (endpoint, info) in endpoint_health.iter() {
            if let Some(time) = info.last_response_time {
                times.insert(endpoint.clone(), time);
            }
        }
        Arc::new(std::sync::Mutex::new(times))
    }

    /// Initialize the uptime gauge using the provided metrics registry
    pub fn initialize_uptime_gauge<T: crate::metrics::MetricsRegistry>(
        &self,
        registry: &T,
    ) -> anyhow::Result<()> {
        let gauge = registry.create_gauge(
            distributed_runtime::UPTIME_SECONDS,
            "Total uptime of the DistributedRuntime in seconds",
            &[],
        )?;
        self.uptime_gauge
            .set(gauge)
            .map_err(|_| anyhow::anyhow!("uptime_gauge already initialized"))?;
        Ok(())
    }

    /// Get the current uptime as a Duration
    pub fn uptime(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Update the uptime gauge with the current uptime value
    pub fn update_uptime_gauge(&self) {
        if let Some(gauge) = self.uptime_gauge.get() {
            gauge.set(self.uptime().as_secs_f64());
        }
    }
}

/// Type alias for runtime callback functions to reduce complexity
///
/// This type represents an Arc-wrapped callback function that can be:
/// - Shared efficiently across multiple threads and contexts
/// - Cloned without duplicating the underlying closure
/// - Used in generic contexts requiring 'static lifetime
///
/// The Arc wrapper is included in the type to make sharing explicit.
type RuntimeCallback = Arc<dyn Fn() -> anyhow::Result<()> + Send + Sync + 'static>;

/// Structure to hold Prometheus registries and associated callbacks for a given hierarchy
pub struct MetricsRegistryEntry {
    /// The Prometheus registry for this prefix
    pub prometheus_registry: prometheus::Registry,
    /// List of function callbacks that receive a reference to any MetricsRegistry
    pub runtime_callbacks: Vec<RuntimeCallback>,
}

impl MetricsRegistryEntry {
    /// Create a new metrics registry entry with an empty registry and no callbacks
    pub fn new() -> Self {
        Self {
            prometheus_registry: prometheus::Registry::new(),
            runtime_callbacks: Vec::new(),
        }
    }

    /// Add a callback function that receives a reference to any MetricsRegistry
    pub fn add_callback(&mut self, callback: RuntimeCallback) {
        self.runtime_callbacks.push(callback);
    }

    /// Execute all runtime callbacks and return their results
    pub fn execute_callbacks(&self) -> Vec<anyhow::Result<()>> {
        self.runtime_callbacks
            .iter()
            .map(|callback| callback())
            .collect()
    }

    /// Returns true if a metric with the given name already exists in the Prometheus registry
    pub fn has_metric_named(&self, metric_name: &str) -> bool {
        self.prometheus_registry
            .gather()
            .iter()
            .any(|mf| mf.name() == metric_name)
    }
}

impl Default for MetricsRegistryEntry {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MetricsRegistryEntry {
    fn clone(&self) -> Self {
        Self {
            prometheus_registry: self.prometheus_registry.clone(),
            runtime_callbacks: Vec::new(), // Callbacks cannot be cloned, so we start with an empty list
        }
    }
}

/// Distributed [Runtime] which provides access to shared resources across the cluster, this includes
/// communication protocols and transports.
#[derive(Clone)]
pub struct DistributedRuntime {
    // local runtime
    runtime: Runtime,

    // we might consider a unifed transport manager here
    etcd_client: Option<transports::etcd::Client>,
    nats_client: transports::nats::Client,
    tcp_server: Arc<OnceCell<Arc<transports::tcp::server::TcpStreamServer>>>,
    system_status_server: Arc<OnceLock<Arc<system_status_server::SystemStatusServerInfo>>>,

    // local registry for components
    // the registry allows us to use share runtime resources across instances of the same component object.
    // take for example two instances of a client to the same remote component. The registry allows us to use
    // a single endpoint watcher for both clients, this keeps the number background tasking watching specific
    // paths in etcd to a minimum.
    component_registry: component::Registry,

    // Will only have static components that are not discoverable via etcd, they must be know at
    // startup. Will not start etcd.
    is_static: bool,

    instance_sources: Arc<tokio::sync::Mutex<HashMap<Endpoint, Weak<InstanceSource>>>>,

    // Health Status
    system_health: Arc<std::sync::Mutex<SystemHealth>>,

    // This map associates metric prefixes with their corresponding Prometheus registries and callbacks.
    // Uses RwLock for better concurrency - multiple threads can read (execute callbacks) simultaneously.
    hierarchy_to_metricsregistry: Arc<std::sync::RwLock<HashMap<String, MetricsRegistryEntry>>>,
}
