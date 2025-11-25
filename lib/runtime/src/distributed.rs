// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::component::{Component, Instance};
use crate::pipeline::PipelineError;
use crate::pipeline::network::manager::NetworkManager;
use crate::service::{ComponentNatsServerPrometheusMetrics, ServiceClient, ServiceSet};
use crate::storage::key_value_store::{
    EtcdStore, KeyValueStore, KeyValueStoreEnum, KeyValueStoreManager, KeyValueStoreSelect,
    MemoryStore,
};
use crate::transports::nats::DRTNatsClientPrometheusMetrics;
use crate::{
    component::{self, ComponentBuilder, Endpoint, Namespace},
    discovery::Discovery,
    metrics::PrometheusUpdateCallback,
    metrics::{MetricsHierarchy, MetricsRegistry},
    transports::{etcd, nats, tcp},
};
use crate::{discovery, system_status_server, transports};

use super::utils::GracefulShutdownTracker;
use crate::SystemHealth;
use crate::runtime::Runtime;

// Used instead of std::cell::OnceCell because get_or_try_init there is nightly
use async_once_cell::OnceCell;

use std::fmt;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;
use tokio::sync::watch::Receiver;

use anyhow::Result;
use derive_getters::Dissolve;
use figment::error;
use std::collections::HashMap;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

type InstanceMap = HashMap<Endpoint, Weak<Receiver<Vec<Instance>>>>;

/// Distributed [Runtime] which provides access to shared resources across the cluster, this includes
/// communication protocols and transports.
#[derive(Clone)]
pub struct DistributedRuntime {
    // local runtime
    runtime: Runtime,

    nats_client: Option<transports::nats::Client>,
    store: KeyValueStoreManager,
    network_manager: Arc<NetworkManager>,
    tcp_server: Arc<OnceCell<Arc<transports::tcp::server::TcpStreamServer>>>,
    system_status_server: Arc<OnceLock<Arc<system_status_server::SystemStatusServerInfo>>>,
    request_plane: RequestPlaneMode,

    // Service discovery client
    discovery_client: Arc<dyn discovery::Discovery>,

    // Discovery metadata (only used for Kubernetes backend)
    // Shared with system status server to expose via /metadata endpoint
    discovery_metadata: Option<Arc<tokio::sync::RwLock<discovery::DiscoveryMetadata>>>,

    // local registry for components
    // the registry allows us to use share runtime resources across instances of the same component object.
    // take for example two instances of a client to the same remote component. The registry allows us to use
    // a single endpoint watcher for both clients, this keeps the number background tasking watching specific
    // paths in etcd to a minimum.
    component_registry: component::Registry,

    instance_sources: Arc<tokio::sync::Mutex<InstanceMap>>,

    // Health Status
    system_health: Arc<parking_lot::Mutex<SystemHealth>>,

    // This hierarchy's own metrics registry
    metrics_registry: MetricsRegistry,
}

impl MetricsHierarchy for DistributedRuntime {
    fn basename(&self) -> String {
        "".to_string() // drt has no basename. Basename only begins with the Namespace.
    }

    fn parent_hierarchies(&self) -> Vec<&dyn MetricsHierarchy> {
        vec![] // drt is the root, so no parent hierarchies
    }

    fn get_metrics_registry(&self) -> &MetricsRegistry {
        &self.metrics_registry
    }
}

impl std::fmt::Debug for DistributedRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DistributedRuntime")
    }
}

impl DistributedRuntime {
    pub async fn new(runtime: Runtime, config: DistributedConfig) -> Result<Self> {
        let (selected_kv_store, nats_config, request_plane) = config.dissolve();

        let runtime_clone = runtime.clone();

        let store = match selected_kv_store {
            KeyValueStoreSelect::Etcd(etcd_config) => {
                let etcd_client = etcd::Client::new(*etcd_config, runtime_clone).await.inspect_err(|err|
                    // The returned error doesn't show because of a dropped runtime error, so
                    // log it first.
                    tracing::error!(%err, "Could not connect to etcd. Pass `--store-kv ..` to use a different backend or start etcd."))?;
                KeyValueStoreManager::etcd(etcd_client)
            }
            KeyValueStoreSelect::File(root) => KeyValueStoreManager::file(root),
            KeyValueStoreSelect::Memory => KeyValueStoreManager::memory(),
        };

        let nats_client = match nats_config {
            Some(nc) => Some(nc.connect().await?),
            None => None,
        };

        // Start system status server for health and metrics if enabled in configuration
        let config = crate::config::RuntimeConfig::from_settings().unwrap_or_default();
        // IMPORTANT: We must extract cancel_token from runtime BEFORE moving runtime into the struct below.
        // This is because after moving, runtime is no longer accessible in this scope (ownership rules).
        let cancel_token = if config.system_server_enabled() {
            Some(runtime.clone().child_token())
        } else {
            None
        };
        let starting_health_status = config.starting_health_status.clone();
        let use_endpoint_health_status = config.use_endpoint_health_status.clone();
        let health_endpoint_path = config.system_health_path.clone();
        let live_endpoint_path = config.system_live_path.clone();
        let system_health = Arc::new(parking_lot::Mutex::new(SystemHealth::new(
            starting_health_status,
            use_endpoint_health_status,
            health_endpoint_path,
            live_endpoint_path,
        )));

        // Initialize discovery client based on backend configuration
        let discovery_backend =
            std::env::var("DYN_DISCOVERY_BACKEND").unwrap_or_else(|_| "kv_store".to_string());

        let (discovery_client, discovery_metadata) = match discovery_backend.as_str() {
            "kubernetes" => {
                tracing::info!("Initializing Kubernetes discovery backend");
                let metadata = Arc::new(tokio::sync::RwLock::new(
                    crate::discovery::DiscoveryMetadata::new(),
                ));
                let client = crate::discovery::KubeDiscoveryClient::new(
                    metadata.clone(),
                    runtime.primary_token(),
                )
                .await
                .inspect_err(
                    |err| tracing::error!(%err, "Failed to initialize Kubernetes discovery client"),
                )?;
                (Arc::new(client) as Arc<dyn Discovery>, Some(metadata))
            }
            _ => {
                tracing::info!("Initializing KV store discovery backend");
                use crate::discovery::KVStoreDiscovery;
                (
                    Arc::new(KVStoreDiscovery::new(
                        store.clone(),
                        runtime.primary_token(),
                    )) as Arc<dyn Discovery>,
                    None,
                )
            }
        };

        let component_registry = component::Registry::new();
        let nats_client_for_metrics = nats_client.clone();

        // NetworkManager for request plane
        let network_manager = NetworkManager::new(
            runtime.child_token(),
            nats_client.clone().map(|c| c.client().clone()),
            component_registry.clone(),
            request_plane,
        );

        let distributed_runtime = Self {
            runtime,
            store,
            network_manager: Arc::new(network_manager),
            nats_client,
            tcp_server: Arc::new(OnceCell::new()),
            system_status_server: Arc::new(OnceLock::new()),
            discovery_client,
            discovery_metadata,
            component_registry,
            instance_sources: Arc::new(Mutex::new(HashMap::new())),
            metrics_registry: crate::MetricsRegistry::new(),
            system_health,
            request_plane,
        };

        if let Some(nats_client_for_metrics) = nats_client_for_metrics {
            let nats_client_metrics = DRTNatsClientPrometheusMetrics::new(
                &distributed_runtime,
                nats_client_for_metrics.client().clone(),
            )?;
            // Register a callback to update NATS client metrics on the DRT's metrics registry
            let nats_client_callback = Arc::new({
                let nats_client_clone = nats_client_metrics.clone();
                move || {
                    nats_client_clone.set_from_client_stats();
                    Ok(())
                }
            });
            distributed_runtime
                .metrics_registry
                .add_update_callback(nats_client_callback);
        }

        // Initialize the uptime gauge in SystemHealth
        distributed_runtime
            .system_health
            .lock()
            .initialize_uptime_gauge(&distributed_runtime)?;

        // Handle system status server initialization
        if let Some(cancel_token) = cancel_token {
            // System server is enabled - start both the state and HTTP server
            let host = config.system_host.clone();
            let port = config.system_port as u16;

            // Start system status server (it creates SystemStatusState internally)
            match crate::system_status_server::spawn_system_status_server(
                &host,
                port,
                cancel_token,
                Arc::new(distributed_runtime.clone()),
                distributed_runtime.discovery_metadata.clone(),
            )
            .await
            {
                Ok((addr, handle)) => {
                    tracing::info!("System status server started successfully on {}", addr);

                    // Store system status server information
                    let system_status_server_info =
                        crate::system_status_server::SystemStatusServerInfo::new(
                            addr,
                            Some(handle),
                        );

                    // Initialize the system_status_server field
                    distributed_runtime
                        .system_status_server
                        .set(Arc::new(system_status_server_info))
                        .expect("System status server info should only be set once");
                }
                Err(e) => {
                    tracing::error!("System status server startup failed: {}", e);
                }
            }
        } else {
            // System server HTTP is disabled, but uptime metrics are still being tracked via SystemHealth
            tracing::debug!(
                "System status server HTTP endpoints disabled, but uptime metrics are being tracked"
            );
        }

        // Start health check manager if enabled
        if config.health_check_enabled {
            let health_check_config = crate::health_check::HealthCheckConfig {
                canary_wait_time: std::time::Duration::from_secs(config.canary_wait_time_secs),
                request_timeout: std::time::Duration::from_secs(
                    config.health_check_request_timeout_secs,
                ),
            };

            // Start the health check manager (spawns per-endpoint monitoring tasks)
            match crate::health_check::start_health_check_manager(
                distributed_runtime.clone(),
                Some(health_check_config),
            )
            .await
            {
                Ok(()) => tracing::info!(
                    "Health check manager started (canary_wait_time: {}s, request_timeout: {}s)",
                    config.canary_wait_time_secs,
                    config.health_check_request_timeout_secs
                ),
                Err(e) => tracing::error!("Health check manager failed to start: {}", e),
            }
        }

        Ok(distributed_runtime)
    }

    pub async fn from_settings(runtime: Runtime) -> Result<Self> {
        let config = DistributedConfig::from_settings();
        Self::new(runtime, config).await
    }

    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn primary_token(&self) -> CancellationToken {
        self.runtime.primary_token()
    }

    // TODO: Don't hand out pointers, instead have methods to use the registry in friendly ways
    // (without being aware of async locks and so on)
    pub fn component_registry(&self) -> &component::Registry {
        &self.component_registry
    }

    // TODO: Don't hand out pointers, instead provide system health related services.
    pub fn system_health(&self) -> Arc<parking_lot::Mutex<SystemHealth>> {
        self.system_health.clone()
    }

    pub fn connection_id(&self) -> u64 {
        self.discovery_client.instance_id()
    }

    pub fn shutdown(&self) {
        self.runtime.shutdown();
        self.store.shutdown();
    }

    /// Create a [`Namespace`]
    pub fn namespace(&self, name: impl Into<String>) -> Result<Namespace> {
        Namespace::new(self.clone(), name.into())
    }

    /// Returns the discovery interface for service registration and discovery
    pub fn discovery(&self) -> Arc<dyn Discovery> {
        self.discovery_client.clone()
    }

    pub async fn tcp_server(&self) -> Result<Arc<tcp::server::TcpStreamServer>> {
        Ok(self
            .tcp_server
            .get_or_try_init(async move {
                let options = tcp::server::ServerOptions::default();
                let server = tcp::server::TcpStreamServer::new(options).await?;
                Ok::<_, PipelineError>(server)
            })
            .await?
            .clone())
    }

    /// Get the network manager
    ///
    /// The network manager consolidates all network configuration and provides
    /// unified access to request plane servers and clients.
    pub fn network_manager(&self) -> Arc<NetworkManager> {
        self.network_manager.clone()
    }

    /// Get the request plane server (convenience method)
    ///
    /// This is a shortcut for `network_manager().await?.server().await`.
    pub async fn request_plane_server(
        &self,
    ) -> Result<Arc<dyn crate::pipeline::network::ingress::unified_server::RequestPlaneServer>>
    {
        self.network_manager().server().await
    }

    /// Get system status server information if available
    pub fn system_status_server_info(
        &self,
    ) -> Option<Arc<crate::system_status_server::SystemStatusServerInfo>> {
        self.system_status_server.get().cloned()
    }

    /// An interface to store things outside of the process. Usually backed by something like etcd.
    /// Currently does key-value, but will grow to include whatever we need to store.
    pub fn store(&self) -> &KeyValueStoreManager {
        &self.store
    }

    /// How the frontend should talk to the backend.
    pub fn request_plane(&self) -> RequestPlaneMode {
        self.request_plane
    }

    pub fn child_token(&self) -> CancellationToken {
        self.runtime.child_token()
    }

    pub(crate) fn graceful_shutdown_tracker(&self) -> Arc<GracefulShutdownTracker> {
        self.runtime.graceful_shutdown_tracker()
    }

    pub fn instance_sources(&self) -> Arc<Mutex<InstanceMap>> {
        self.instance_sources.clone()
    }

    /// TODO: This is a temporary KV router measure for component/component.rs EventPublisher impl for
    /// Component, to allow it to publish to NATS. KV Router is the only user.
    pub(crate) async fn kv_router_nats_publish(
        &self,
        subject: String,
        payload: bytes::Bytes,
    ) -> anyhow::Result<()> {
        let Some(nats_client) = self.nats_client.as_ref() else {
            anyhow::bail!("KV router's EventPublisher requires NATS");
        };
        Ok(nats_client.client().publish(subject, payload).await?)
    }

    /// TODO: This is a temporary KV router measure for component/component.rs EventSubscriber impl for
    /// Component, to allow it to subscribe to NATS. KV Router is the only user.
    pub(crate) async fn kv_router_nats_subscribe(
        &self,
        subject: String,
    ) -> Result<async_nats::Subscriber> {
        let Some(nats_client) = self.nats_client.as_ref() else {
            anyhow::bail!("KV router's EventSubscriber requires NATS");
        };
        Ok(nats_client.client().subscribe(subject).await?)
    }

    /// Start NATS metrics service in the background to isolate the async,
    /// and because we don't need it yet.
    /// TODO: This and the things it calls should be in a nats module somewhere.
    pub fn start_stats_service(&self, component: Component) {
        let drt = self.clone();
        self.runtime().secondary().spawn(async move {
            let service_name = component.service_name();
            if let Err(err) = drt.add_stats_service(component).await {
                tracing::error!(error = %err, component = service_name, "Failed starting stats service");
            }
        });
    }

    /// Gather NATS metrics
    async fn add_stats_service(&self, component: Component) -> anyhow::Result<()> {
        let service_name = component.service_name();

        // Pre-check to save cost of creating the service, but don't hold the lock
        if self
            .component_registry()
            .inner
            .lock()
            .await
            .services
            .contains_key(&service_name)
        {
            // The NATS service is per component, but it is called from `serve_endpoint`, and there
            // are often multiple endpoints for a component (e.g. `clear_kv_blocks` and `generate`).
            tracing::trace!("Service {service_name} already exists");
            return Ok(());
        }

        let Some(nats_client) = self.nats_client.as_ref() else {
            anyhow::bail!("Cannot create NATS service without NATS.");
        };
        let description = None;
        let (nats_service, stats_reg) =
            crate::component::service::build_nats_service(nats_client, &component, description)
                .await?;

        let mut guard = self.component_registry().inner.lock().await;
        if !guard.services.contains_key(&service_name) {
            // Normal case
            guard.services.insert(service_name.clone(), nats_service);
            guard.stats_handlers.insert(service_name.clone(), stats_reg);

            tracing::info!("Added NATS / stats service {service_name}");

            drop(guard);
        } else {
            drop(guard);
            let _ = nats_service.stop().await;
            // The NATS service is per component, but it is called from `serve_endpoint`, and there
            // are often multiple endpoints for a component (e.g. `clear_kv_blocks` and `generate`).
            // TODO: Is this still true?
            return Ok(());
        }

        let cancel_token = self.primary_token();
        let service_client = self
            .nats_client
            .as_ref()
            .map(|nc| ServiceClient::new(nc.clone()))
            .ok_or_else(|| {
                anyhow::anyhow!("Stats service requires NATS client to collect service metrics.")
            })?;
        // If there is another component with the same service name, this will fail.
        let component_metrics = ComponentNatsServerPrometheusMetrics::new(&component)?;

        self.runtime().secondary().spawn(nats_metrics_worker(
            cancel_token,
            service_client,
            component_metrics,
            component,
        ));
        Ok(())
    }
}

/// Add Prometheus metrics for this component's NATS service stats.
///
/// Starts a background task that periodically requests service statistics from NATS
/// and updates the corresponding Prometheus metrics. The first scrape happens immediately,
/// then subsequent scrapes occur at a fixed interval of 9.8 seconds (MAX_WAIT_MS),
/// which should be near or smaller than typical Prometheus scraping intervals to ensure
/// metrics are fresh when Prometheus collects them.
async fn nats_metrics_worker(
    cancel_token: CancellationToken,
    service_client: ServiceClient,
    component_metrics: ComponentNatsServerPrometheusMetrics,
    component: Component,
) {
    const MAX_WAIT_MS: Duration = Duration::from_millis(9800); // Should be <= Prometheus scrape interval
    let timeout = Duration::from_millis(500);
    let mut interval = tokio::time::interval(MAX_WAIT_MS);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let service_name = component.service_name();
    loop {
        tokio::select! {
            result = service_client.collect_services(&service_name, timeout) => {
                match result {
                    Ok(service_set) => {
                        component_metrics.update_from_service_set(&service_set);
                    }
                    Err(err) => {
                        tracing::error!("Background scrape failed for {service_name}: {err}",);
                        component_metrics.reset_to_zeros();
                    }
                }
            }
            _ = cancel_token.cancelled() => {
                tracing::trace!("nats_metrics_worker stopped");
                break;
            }
        }

        interval.tick().await;
    }
}

#[derive(Dissolve)]
pub struct DistributedConfig {
    pub store_backend: KeyValueStoreSelect,
    pub nats_config: Option<nats::ClientOptions>,
    pub request_plane: RequestPlaneMode,
}

impl DistributedConfig {
    pub fn from_settings() -> DistributedConfig {
        let request_plane = RequestPlaneMode::from_env();
        DistributedConfig {
            store_backend: KeyValueStoreSelect::Etcd(Box::default()),
            nats_config: if request_plane.is_nats() {
                Some(nats::ClientOptions::default())
            } else {
                None
            },
            request_plane,
        }
    }

    pub fn for_cli() -> DistributedConfig {
        let etcd_config = etcd::ClientOptions {
            attach_lease: false,
            ..Default::default()
        };
        let request_plane = RequestPlaneMode::from_env();
        DistributedConfig {
            store_backend: KeyValueStoreSelect::Etcd(Box::new(etcd_config)),
            nats_config: if request_plane.is_nats() {
                Some(nats::ClientOptions::default())
            } else {
                None
            },
            request_plane,
        }
    }

    /// A DistributedConfig that isn't distributed, for when the frontend and backend are in the
    /// same process.
    pub fn process_local() -> DistributedConfig {
        DistributedConfig {
            store_backend: KeyValueStoreSelect::Memory,
            nats_config: None,
            // This won't be used in process local, so we likely need a "none" option to
            // communicate that and avoid opening the ports.
            request_plane: RequestPlaneMode::Tcp,
        }
    }
}

/// Request plane transport mode configuration
///
/// This determines how requests are distributed from routers to workers:
/// - `Nats`: Use NATS for request distribution (default, legacy)
/// - `Http`: Use HTTP/2 for request distribution
/// - `Tcp`: Use raw TCP for request distribution with msgpack support
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPlaneMode {
    /// Use NATS for request plane (default for backward compatibility)
    Nats,
    /// Use HTTP/2 for request plane
    Http,
    /// Use raw TCP for request plane with msgpack support
    Tcp,
}

impl Default for RequestPlaneMode {
    fn default() -> Self {
        Self::Nats
    }
}

impl fmt::Display for RequestPlaneMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nats => write!(f, "nats"),
            Self::Http => write!(f, "http"),
            Self::Tcp => write!(f, "tcp"),
        }
    }
}

impl std::str::FromStr for RequestPlaneMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "nats" => Ok(Self::Nats),
            "http" => Ok(Self::Http),
            "tcp" => Ok(Self::Tcp),
            _ => Err(anyhow::anyhow!(
                "Invalid request plane mode: '{}'. Valid options are: 'nats', 'http', 'tcp'",
                s
            )),
        }
    }
}

impl RequestPlaneMode {
    /// Get the request plane mode from environment variable (uncached)
    /// Reads from `DYN_REQUEST_PLANE` environment variable.
    fn from_env() -> Self {
        std::env::var("DYN_REQUEST_PLANE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_default()
    }

    pub fn is_nats(&self) -> bool {
        matches!(self, RequestPlaneMode::Nats)
    }
}

pub mod distributed_test_utils {
    //! Common test helper functions for DistributedRuntime tests

    /// Helper function to create a DRT instance for integration-only tests.
    /// Uses from_current to leverage existing tokio runtime
    /// Note: Settings are read from environment variables inside DistributedRuntime::from_settings
    #[cfg(feature = "integration")]
    pub async fn create_test_drt_async() -> super::DistributedRuntime {
        use crate::{storage::key_value_store::KeyValueStoreSelect, transports::nats};

        let rt = crate::Runtime::from_current().unwrap();
        let config = super::DistributedConfig {
            store_backend: KeyValueStoreSelect::Memory,
            nats_config: Some(nats::ClientOptions::default()),
            request_plane: crate::distributed::RequestPlaneMode::default(),
        };
        super::DistributedRuntime::new(rt, config).await.unwrap()
    }
}

#[cfg(all(test, feature = "integration"))]
mod tests {
    use super::RequestPlaneMode;
    use super::distributed_test_utils::create_test_drt_async;

    #[tokio::test]
    async fn test_drt_uptime_after_delay_system_disabled() {
        use crate::config::environment_names::runtime::system as env_system;
        // Test uptime with system status server disabled
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, None::<&str>)], async {
            // Start a DRT
            let drt = create_test_drt_async().await;

            // Wait 50ms
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

            // Check that uptime is 50+ ms
            let uptime = drt.system_health.lock().uptime();
            assert!(
                uptime >= std::time::Duration::from_millis(50),
                "Expected uptime to be at least 50ms, but got {:?}",
                uptime
            );

            println!(
                "✓ DRT uptime test passed (system disabled): uptime = {:?}",
                uptime
            );
        })
        .await;
    }

    #[tokio::test]
    async fn test_drt_uptime_after_delay_system_enabled() {
        use crate::config::environment_names::runtime::system as env_system;
        // Test uptime with system status server enabled
        temp_env::async_with_vars([(env_system::DYN_SYSTEM_PORT, Some("8081"))], async {
            // Start a DRT
            let drt = create_test_drt_async().await;

            // Wait 50ms
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

            // Check that uptime is 50+ ms
            let uptime = drt.system_health.lock().uptime();
            assert!(
                uptime >= std::time::Duration::from_millis(50),
                "Expected uptime to be at least 50ms, but got {:?}",
                uptime
            );

            println!(
                "✓ DRT uptime test passed (system enabled): uptime = {:?}",
                uptime
            );
        })
        .await;
    }

    #[test]
    fn test_request_plane_mode_from_str() {
        assert_eq!(
            "nats".parse::<RequestPlaneMode>().unwrap(),
            RequestPlaneMode::Nats
        );
        assert_eq!(
            "http".parse::<RequestPlaneMode>().unwrap(),
            RequestPlaneMode::Http
        );
        assert_eq!(
            "tcp".parse::<RequestPlaneMode>().unwrap(),
            RequestPlaneMode::Tcp
        );
        assert_eq!(
            "NATS".parse::<RequestPlaneMode>().unwrap(),
            RequestPlaneMode::Nats
        );
        assert_eq!(
            "HTTP".parse::<RequestPlaneMode>().unwrap(),
            RequestPlaneMode::Http
        );
        assert_eq!(
            "TCP".parse::<RequestPlaneMode>().unwrap(),
            RequestPlaneMode::Tcp
        );
        assert!("invalid".parse::<RequestPlaneMode>().is_err());
    }

    #[test]
    fn test_request_plane_mode_display() {
        assert_eq!(RequestPlaneMode::Nats.to_string(), "nats");
        assert_eq!(RequestPlaneMode::Http.to_string(), "http");
        assert_eq!(RequestPlaneMode::Tcp.to_string(), "tcp");
    }
}
