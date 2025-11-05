// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Etcd-backed peer discovery with TTL and automatic cleanup.
//!
//! This implementation provides centralized discovery using etcd with:
//! - Automatic TTL-based expiration
//! - Heartbeat keep-alive for registration freshness
//! - Transaction-based collision detection
//! - Graceful cleanup on unregister
//!
//! # Example
//!
//! ```no_run
//! use dynamo_am_discovery::etcd::EtcdConfig;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let discovery = EtcdConfigBuilder::default()
//!     .cluster_id("my-cluster-peers")
//!     .endpoints(vec!["http://localhost:2379".to_string()])
//!     .build()
//!     .await?;
//!
//! // Use the discovery system
//! // let peer_discovery = discovery.peer_discovery().unwrap();
//! // peer_discovery.register_instance(instance_id, address).await?;
//! # Ok(())
//! # }
//! ```

mod client;
mod error;
mod keep_alive;
mod lease;
mod operations;
mod peer;

use keep_alive::KeepAliveTask;
use lease::LeaseState;
use operations::OperationExecutor;
use peer::EtcdPeerDiscovery;

use anyhow::{Context, Result};
use derive_builder::Builder;
use parking_lot::{Mutex, RwLock};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use validator::Validate;

use crate::peer::PeerDiscovery;

use super::DiscoverySystem;

/// Validates that a Duration is within the specified range (in seconds).
fn validate_ttl(ttl: &Duration) -> Result<(), validator::ValidationError> {
    let secs = ttl.as_secs();
    if !(10..=600).contains(&secs) {
        return Err(validator::ValidationError::new("ttl_range"));
    }
    Ok(())
}

/// Configuration for etcd-backed discovery.
///
/// # Example
///
/// ```no_run
/// use dynamo_am_discovery::etcd::EtcdConfig;
/// use std::time::Duration;
///
/// # async fn example() -> anyhow::Result<()> {
/// let system = EtcdConfigBuilder::default()
///     .cluster_id("my-cluster")
///     .endpoints(vec!["http://localhost:2379".to_string()])
///     .ttl(Duration::from_secs(60))
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Builder, Validate)]
#[builder(build_fn(private, name = "build_config"), pattern = "owned")]
pub struct EtcdConfig {
    /// Cluster ID / key prefix for discovery data (required)
    #[builder(setter(into))]
    #[validate(custom(function = "super::validation::validate_cluster_id"))]
    pub cluster_id: String,

    /// Etcd cluster endpoints (e.g., `["http://localhost:2379"]`)
    #[builder(default = "vec![\"http://localhost:2379\".to_string()]")]
    pub endpoints: Vec<String>,

    /// Lease TTL duration (default: 60 seconds, min: 10s, max: 600s)
    #[builder(default = "Duration::from_secs(60)")]
    #[validate(custom(function = "validate_ttl"))]
    pub ttl: Duration,

    /// Timeout for individual operations (default: 30 seconds)
    #[builder(default = "Duration::from_secs(30)")]
    pub operation_timeout: Duration,

    /// Maximum number of retries for operations (default: 3)
    #[builder(default = "3")]
    #[validate(range(min = 0, max = 3))]
    pub max_retries: u32,

    /// Initial backoff duration for reconnection attempts (default: 500ms)
    #[builder(default = "Duration::from_millis(500)")]
    pub initial_backoff: Duration,

    /// Minimum backoff duration for reconnection attempts (default: 50ms)
    #[builder(default = "Duration::from_millis(50)")]
    pub min_backoff: Duration,

    /// Maximum backoff duration for reconnection attempts (default: 5s)
    #[builder(default = "Duration::from_secs(5)")]
    pub max_backoff: Duration,
}

/// Extension for EtcdConfigBuilder to provide async build.
impl EtcdConfigBuilder {
    /// Build and initialize the etcd discovery system.
    ///
    /// This combines configuration validation and async system initialization into
    /// a single call.
    ///
    /// # Returns
    ///
    /// * `Ok(Arc<dyn DiscoverySystem>)` - Successfully connected to etcd
    /// * `Err` - Failed to build config or connect to etcd cluster
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use dynamo_am_discovery::etcd::EtcdConfig;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let system = EtcdConfigBuilder::default()
    ///     .cluster_id("my-cluster-peers")
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn build(self) -> Result<Arc<dyn DiscoverySystem>, anyhow::Error> {
        // Build the config using the private generated method
        let config = self
            .build_config()
            .map_err(|e| anyhow::anyhow!("Failed to build config: {}", e))?;

        // Initialize the system
        let system = EtcdDiscoverySystem::new(config).await?;
        Ok(system)
    }
}

/// Private implementation of etcd-backed discovery system.
///
/// Manages connection, lease, keep-alive, and provides PeerDiscovery instances.
struct EtcdDiscoverySystem {
    client: Arc<client::Client>,
    lease_state: Arc<RwLock<LeaseState>>,
    config: EtcdConfig,
    keep_alive_handle: Mutex<Option<JoinHandle<()>>>,
    shutdown: CancellationToken,
    peer_discovery: OnceLock<Arc<dyn PeerDiscovery>>,
}

impl std::fmt::Debug for EtcdDiscoverySystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EtcdDiscoverySystem")
            .field("cluster_id", &self.config.cluster_id)
            .field("endpoints", &self.config.endpoints)
            .field("ttl", &self.config.ttl)
            .finish()
    }
}

impl EtcdDiscoverySystem {
    /// Create a new etcd discovery system.
    ///
    /// # Steps
    ///
    /// 1. Connect to etcd cluster
    /// 2. Create lease with TTL
    /// 3. Start keep-alive task
    /// 4. Return system ready for use
    #[allow(clippy::await_holding_lock, clippy::new_ret_no_self)]
    async fn new(config: EtcdConfig) -> Result<Arc<dyn DiscoverySystem>> {
        // Connect to etcd with backoff configuration
        let client = Arc::new(
            client::Client::new(
                config.endpoints.clone(),
                None,
                config.initial_backoff,
                config.min_backoff,
                config.max_backoff,
            )
            .await
            .context("Failed to connect to etcd cluster")?,
        );

        tracing::info!(
            "Connected to etcd cluster: {:?}, cluster_id: {}, TTL: {:?}",
            config.endpoints,
            config.cluster_id,
            config.ttl
        );

        // Initialize lease state
        let lease_state = Arc::new(RwLock::new(LeaseState::new(config.ttl)));

        // Create and ensure lease
        // Note: This is initialization code, no concurrency yet
        {
            let mut etcd_client = client.get_client();
            lease_state
                .write()
                .ensure_lease(&mut etcd_client)
                .await
                .context("Failed to create initial lease")?;
        }

        let shutdown = CancellationToken::new();

        let system = Arc::new(Self {
            client: client.clone(),
            lease_state: lease_state.clone(),
            config,
            keep_alive_handle: Mutex::new(None),
            shutdown: shutdown.clone(),
            peer_discovery: OnceLock::new(),
        });

        // Start keep-alive task
        let keep_alive_task = KeepAliveTask::new(client, lease_state, system.config.ttl, shutdown);
        let handle = keep_alive_task.spawn();
        *system.keep_alive_handle.lock() = Some(handle);

        tracing::info!("Etcd discovery system initialized successfully");

        Ok(system)
    }
}

impl DiscoverySystem for EtcdDiscoverySystem {
    fn peer_discovery(&self) -> Option<Arc<dyn PeerDiscovery>> {
        Some(
            self.peer_discovery
                .get_or_init(|| {
                    let executor = OperationExecutor::new(
                        self.client.clone(),
                        self.config.operation_timeout,
                        self.config.max_retries,
                    );

                    let discovery: Arc<dyn PeerDiscovery> = Arc::new(EtcdPeerDiscovery::new(
                        executor,
                        self.lease_state.clone(),
                        self.config.cluster_id.clone(),
                    ));
                    discovery
                })
                .clone(),
        )
    }

    fn shutdown(&self) {
        tracing::info!("Shutting down EtcdDiscoverySystem");

        // Signal shutdown to all tasks
        self.shutdown.cancel();

        // Abort keep-alive task
        if let Some(handle) = self.keep_alive_handle.lock().take() {
            handle.abort();
        }

        tracing::info!("EtcdDiscoverySystem shutdown complete");
    }
}

impl Drop for EtcdDiscoverySystem {
    fn drop(&mut self) {
        // Ensure shutdown is called on drop
        self.shutdown();
    }
}

#[cfg(all(test, feature = "etcd"))]
mod tests {
    use super::*;
    use crate::peer::{InstanceId, WorkerAddress};
    use crate::systems::DiscoverySystem;
    use crate::systems::test_support::{
        checksum_validation, collision_detection, not_found_errors,
        register_and_discover_by_instance_id, register_and_discover_by_worker_id,
    };
    use std::sync::Arc;

    // Note: These tests require a running etcd instance
    //
    // Quick start:
    //   docker run -d -p 2379:2379 --name etcd-test quay.io/coreos/etcd:v3.5.0 \
    //     /usr/local/bin/etcd --advertise-client-urls http://0.0.0.0:2379 \
    //     --listen-client-urls http://0.0.0.0:2379
    //
    // Run tests (enabled by default with 'testing-etcd' feature):
    //   cargo test --package dynamo-discovery --lib --features etcd
    //
    // To skip these tests, disable the feature:
    //   cargo test --package dynamo-discovery --lib --features etcd --no-default-features

    /// Helper function to get etcd endpoint for tests
    fn etcd_endpoint() -> String {
        std::env::var("ETCD_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:2379".to_string())
    }

    fn make_test_address() -> WorkerAddress {
        WorkerAddress::from_bytes(b"127.0.0.1:8080".as_slice())
    }

    fn system_factory(
        cluster_id: String,
    ) -> impl std::future::Future<Output = anyhow::Result<Arc<dyn DiscoverySystem>>> {
        let endpoint = etcd_endpoint();
        async move {
            EtcdConfigBuilder::default()
                .cluster_id(cluster_id)
                .endpoints(vec![endpoint])
                .ttl(Duration::from_secs(30))
                .build()
                .await
        }
    }

    #[cfg_attr(not(feature = "testing-etcd"), ignore)]
    #[tokio::test]
    async fn test_etcd_register_and_discover_by_worker_id() {
        register_and_discover_by_worker_id(system_factory)
            .await
            .expect("worker_id discovery test failed");
    }

    #[cfg_attr(not(feature = "testing-etcd"), ignore)]
    #[tokio::test]
    async fn test_etcd_register_and_discover_by_instance_id() {
        register_and_discover_by_instance_id(system_factory)
            .await
            .expect("instance_id discovery test failed");
    }

    #[cfg_attr(not(feature = "testing-etcd"), ignore)]
    #[tokio::test]
    async fn test_etcd_collision_detection() {
        collision_detection(system_factory)
            .await
            .expect("collision detection test failed");
    }

    #[cfg_attr(not(feature = "testing-etcd"), ignore)]
    #[tokio::test]
    async fn test_etcd_checksum_validation() {
        checksum_validation(system_factory)
            .await
            .expect("checksum validation test failed");
    }

    #[cfg_attr(not(feature = "testing-etcd"), ignore)]
    #[tokio::test]
    async fn test_etcd_not_found_errors() {
        not_found_errors(system_factory)
            .await
            .expect("not found error test failed");
    }

    #[cfg_attr(not(feature = "testing-etcd"), ignore)]
    #[tokio::test]
    async fn test_etcd_unregister_revokes_lease() {
        let system = system_factory("test-revoke".to_string())
            .await
            .expect("Failed to build discovery system");

        let peer_discovery = system
            .peer_discovery()
            .expect("Peer discovery should be available");

        let instance_id = InstanceId::new_v4();
        let address = make_test_address();
        let worker_id = instance_id.worker_id();

        peer_discovery
            .register_instance(instance_id, address.clone())
            .await
            .unwrap();

        // Verify it's registered
        let found = peer_discovery
            .discover_by_worker_id(worker_id)
            .await
            .unwrap();
        assert_eq!(found.instance_id(), instance_id);

        // Unregister should revoke lease immediately
        peer_discovery
            .unregister_instance(instance_id)
            .await
            .unwrap();

        // Should no longer be discoverable (no need to wait for TTL)
        let result = peer_discovery.discover_by_worker_id(worker_id).await;
        assert!(
            result.is_err(),
            "Unregistered peer should not be discoverable"
        );

        system.shutdown();
    }

    #[cfg_attr(not(feature = "testing-etcd"), ignore)]
    #[tokio::test]
    async fn test_etcd_multiple_discovery_instances() {
        // Test that multiple discovery instances can share the same etcd
        let cluster_id = "test-shared".to_string();
        let system1 = system_factory(cluster_id.clone())
            .await
            .expect("Failed to build discovery system 1");

        let system2 = system_factory(cluster_id)
            .await
            .expect("Failed to build discovery system 2");

        let peer_discovery1 = system1
            .peer_discovery()
            .expect("Peer discovery 1 should be available");
        let peer_discovery2 = system2
            .peer_discovery()
            .expect("Peer discovery 2 should be available");

        let instance_id = InstanceId::new_v4();
        let address = make_test_address();
        let worker_id = instance_id.worker_id();

        // Register on discovery1
        peer_discovery1
            .register_instance(instance_id, address.clone())
            .await
            .unwrap();

        // Should be visible from discovery2
        let found = peer_discovery2
            .discover_by_worker_id(worker_id)
            .await
            .unwrap();
        assert_eq!(found.instance_id(), instance_id);

        // Cleanup from either instance should work
        peer_discovery2
            .unregister_instance(instance_id)
            .await
            .unwrap();

        // Should no longer be discoverable
        let result = peer_discovery1.discover_by_worker_id(worker_id).await;
        assert!(result.is_err());

        system1.shutdown();
        system2.shutdown();
    }
}
