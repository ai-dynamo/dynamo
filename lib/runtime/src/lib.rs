// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dynamo

#![allow(dead_code)]
#![allow(unused_imports)]

use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, Weak},
    time::Instant,
};
use tokio::sync::mpsc;

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

/// Health check target containing instance info and payload
#[derive(Clone, Debug)]
pub struct HealthCheckTarget {
    pub instance: component::Instance,
    pub payload: serde_json::Value,
}

/// Current Health Status
/// System health is determined by all endpoints that have registered health checks
#[derive(Clone)]
pub struct SystemHealth {
    system_health: HealthStatus,
    endpoint_health: Arc<std::sync::RwLock<HashMap<String, HealthStatus>>>,
    /// Maps endpoint subject to health check target (instance + payload)
    health_check_targets: Arc<std::sync::RwLock<HashMap<String, HealthCheckTarget>>>,
    /// Maps endpoint subject to its specific health check notifier
    health_check_notifiers: Arc<std::sync::RwLock<HashMap<String, Arc<tokio::sync::Notify>>>>,
    /// Channel for new endpoint registrations
    /// This solves the race condition where HealthCheckManager starts before endpoints are registered
    /// Using a channel ensures no registrations are lost.
    new_endpoint_tx: mpsc::UnboundedSender<String>,
    new_endpoint_rx: Arc<std::sync::Mutex<Option<mpsc::UnboundedReceiver<String>>>>,
    health_path: String,
    live_path: String,
    start_time: Instant,
    uptime_gauge: OnceLock<prometheus::Gauge>,
}

impl SystemHealth {
    pub fn new(
        starting_health_status: HealthStatus,
        health_path: String,
        live_path: String,
    ) -> Self {
        // Create the channel for endpoint registration notifications
        let (tx, rx) = mpsc::unbounded_channel();

        SystemHealth {
            system_health: starting_health_status,
            endpoint_health: Arc::new(std::sync::RwLock::new(HashMap::new())),
            health_check_targets: Arc::new(std::sync::RwLock::new(HashMap::new())),
            health_check_notifiers: Arc::new(std::sync::RwLock::new(HashMap::new())),
            new_endpoint_tx: tx,
            new_endpoint_rx: Arc::new(std::sync::Mutex::new(Some(rx))),
            health_path,
            live_path,
            start_time: Instant::now(),
            uptime_gauge: OnceLock::new(),
        }
    }
    pub fn set_health_status(&mut self, status: HealthStatus) {
        self.system_health = status;
    }

    pub fn set_endpoint_health_status(&self, endpoint: &str, status: HealthStatus) {
        let mut endpoint_health = self.endpoint_health.write().unwrap();
        endpoint_health.insert(endpoint.to_string(), status);
    }

    /// Returns the overall health status and endpoint health statuses
    /// System health is determined by ALL endpoints that have registered health checks
    pub fn get_health_status(&self) -> (bool, HashMap<String, String>) {
        let health_check_targets = self.health_check_targets.read().unwrap();
        let endpoint_health = self.endpoint_health.read().unwrap();
        let mut endpoints: HashMap<String, String> = HashMap::new();

        for (endpoint, status) in endpoint_health.iter() {
            endpoints.insert(
                endpoint.clone(),
                if *status == HealthStatus::Ready {
                    "ready".to_string()
                } else {
                    "notready".to_string()
                },
            );
        }

        // If we have registered health check targets, use them to determine health
        let healthy = if !health_check_targets.is_empty() {
            health_check_targets
                .iter()
                .all(|(endpoint_subject, _target)| {
                    endpoint_health
                        .get(endpoint_subject)
                        .is_some_and(|status| *status == HealthStatus::Ready)
                })
        } else {
            // No health check targets registered, use simple system health
            self.system_health == HealthStatus::Ready
        };

        (healthy, endpoints)
    }

    /// Register a health check target for an endpoint
    pub fn register_health_check_target(
        &self,
        endpoint_subject: &str,
        instance: component::Instance,
        payload: serde_json::Value,
    ) {
        let key = endpoint_subject.to_owned();

        // Atomically check+insert under a single write lock to avoid races.
        let inserted = {
            let mut targets = self.health_check_targets.write().unwrap();
            match targets.entry(key.clone()) {
                std::collections::hash_map::Entry::Occupied(_) => false,
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(HealthCheckTarget { instance, payload });
                    true
                }
            }
        };

        if !inserted {
            tracing::warn!(
                "Attempted to re-register health check for endpoint '{}'; ignoring.",
                key
            );
            return;
        }

        // Create and store a unique notifier for this endpoint (idempotent).
        {
            let mut notifiers = self.health_check_notifiers.write().unwrap();
            notifiers
                .entry(key.clone())
                .or_insert_with(|| Arc::new(tokio::sync::Notify::new()));
        }

        // Initialize endpoint health status conservatively to NotReady.
        {
            let mut endpoint_health = self.endpoint_health.write().unwrap();
            endpoint_health
                .entry(key.clone())
                .or_insert(HealthStatus::NotReady);
        }

        if let Err(e) = self.new_endpoint_tx.send(key.clone()) {
            tracing::error!(
                "Failed to send endpoint '{}' registration to health check manager: {}. \
                 Health checks will not be performed for this endpoint.",
                key,
                e
            );
        }
    }

    /// Get all health check targets
    pub fn get_health_check_targets(&self) -> Vec<(String, HealthCheckTarget)> {
        let targets = self.health_check_targets.read().unwrap();
        targets
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Check if any health check targets are registered
    pub fn has_health_check_targets(&self) -> bool {
        let targets = self.health_check_targets.read().unwrap();
        !targets.is_empty()
    }

    /// Get list of endpoints with health check targets
    pub fn get_health_check_endpoints(&self) -> Vec<String> {
        let targets = self.health_check_targets.read().unwrap();
        targets.keys().cloned().collect()
    }

    /// Get health check target for a specific endpoint
    pub fn get_health_check_target(&self, endpoint: &str) -> Option<HealthCheckTarget> {
        let targets = self.health_check_targets.read().unwrap();
        targets.get(endpoint).cloned()
    }

    /// Get the endpoint health status (Ready/NotReady)
    pub fn get_endpoint_health_status(&self, endpoint: &str) -> Option<HealthStatus> {
        let endpoint_health = self.endpoint_health.read().unwrap();
        endpoint_health.get(endpoint).cloned()
    }

    /// Get the endpoint-specific health check notifier
    pub fn get_endpoint_health_check_notifier(
        &self,
        endpoint_subject: &str,
    ) -> Option<Arc<tokio::sync::Notify>> {
        let notifiers = self.health_check_notifiers.read().unwrap();
        notifiers.get(endpoint_subject).cloned()
    }

    /// Take the receiver for new endpoint registrations (can only be called once)
    /// This is used by HealthCheckManager to receive notifications of new endpoints
    pub fn take_new_endpoint_receiver(&self) -> Option<mpsc::UnboundedReceiver<String>> {
        self.new_endpoint_rx.lock().unwrap().take()
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
