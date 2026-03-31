// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration types for the distributed registry.

use std::net::SocketAddr;
use std::time::Duration;

use anyhow::Result;

/// Hub configuration for the registry server.
///
/// # Example
/// ```text
/// let config = RegistryHubConfig {
///     capacity: 1_000_000,
///     addr: "0.0.0.0:5555".parse().unwrap(),
///     lease_timeout: Duration::from_secs(30),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RegistryHubConfig {
    /// Registry capacity (number of entries).
    pub capacity: u64,

    /// TCP address to bind on.
    ///
    /// All workers connect to this single port; queries and registrations
    /// are multiplexed over the same connection using frame type tagging.
    ///
    /// Example: "0.0.0.0:5555", "127.0.0.1:5555"
    pub addr: SocketAddr,

    /// Lease timeout for `can_offload` claims.
    ///
    /// When a worker calls `can_offload`, it gets exclusive leases on the
    /// returned hashes. If the worker doesn't call `register` within this
    /// timeout, the leases expire and other workers can claim them.
    ///
    /// Default: 30 seconds
    pub lease_timeout: Duration,
}

impl Default for RegistryHubConfig {
    fn default() -> Self {
        Self {
            capacity: 1_000_000,
            addr: "0.0.0.0:5555".parse().unwrap(),
            lease_timeout: Duration::from_secs(30),
        }
    }
}

impl RegistryHubConfig {
    /// Create a new config with specified capacity.
    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            capacity,
            ..Default::default()
        }
    }

    /// Create config listening on a specific port on all interfaces.
    pub fn on_port(port: u16) -> Self {
        Self {
            addr: format!("0.0.0.0:{port}").parse().unwrap(),
            ..Default::default()
        }
    }

    /// Create config from environment variables.
    ///
    /// Environment variables:
    /// - `DYN_REGISTRY_HUB_CAPACITY`: Registry capacity (default: 1000000)
    /// - `DYN_REGISTRY_HUB_ADDR`: Hub bind address (default: 0.0.0.0:5555)
    /// - `DYN_REGISTRY_HUB_LEASE_TIMEOUT_SECS`: Lease timeout in seconds (default: 30)
    pub fn from_env() -> Self {
        Self {
            capacity: std::env::var("DYN_REGISTRY_HUB_CAPACITY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000),
            addr: std::env::var("DYN_REGISTRY_HUB_ADDR")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| "0.0.0.0:5555".parse().unwrap()),
            lease_timeout: Duration::from_secs(
                std::env::var("DYN_REGISTRY_HUB_LEASE_TIMEOUT_SECS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(30),
            ),
        }
    }

    /// Set lease timeout.
    pub fn with_lease_timeout(mut self, timeout: Duration) -> Self {
        self.lease_timeout = timeout;
        self
    }
}

/// Client configuration for registry workers.
///
/// # Example
/// ```text
/// let config = RegistryClientConfig::connect_to("leader.local", 5555);
/// ```
#[derive(Debug, Clone)]
pub struct RegistryClientConfig {
    /// Hub address to connect to.
    ///
    /// Example: "192.168.1.100:5555", "leader.local:5555"
    pub hub_addr: SocketAddr,

    /// Worker rank (0-indexed).
    ///
    /// Used to generate the default namespace if not explicitly set.
    pub rank: Option<usize>,

    /// Total number of workers.
    ///
    /// Used to generate the default namespace if not explicitly set.
    pub world_size: Option<usize>,

    /// Namespace for this worker's storage.
    ///
    /// Used as part of the registry key to enable cross-instance deduplication.
    /// Can represent a bucket, directory, or any storage-specific identifier.
    /// If not set, defaults to "worker-{rank}-{world_size}" or "worker-{pid}".
    /// Example: "worker-0-8", "instance-abc123", "/mnt/cache/worker-0"
    pub namespace: String,

    /// Batch size for registrations before auto-flush.
    ///
    /// Registrations are batched for efficiency. When the batch reaches
    /// this size, it's automatically sent to the hub.
    pub batch_size: usize,

    /// Batch timeout before auto-flush.
    ///
    /// If a batch has been pending for this duration without reaching
    /// `batch_size`, it's automatically flushed.
    pub batch_timeout: Duration,

    /// Optional local cache capacity (0 = disabled).
    ///
    /// If > 0, the client maintains a local cache to reduce
    /// network round-trips for frequently accessed hashes.
    pub local_cache_capacity: u64,
}

impl Default for RegistryClientConfig {
    fn default() -> Self {
        Self {
            hub_addr: "127.0.0.1:5555".parse().unwrap(),
            rank: None,
            world_size: None,
            namespace: default_namespace(None, None),
            batch_size: 100,
            batch_timeout: Duration::from_millis(10),
            local_cache_capacity: 0,
        }
    }
}

/// Generate a unique default namespace to prevent accidental cross-worker deduplication.
///
/// Format: `worker-{rank}-{world_size}` (e.g., "worker-0-8")
/// Falls back to `worker-{pid}` if rank/world_size are unavailable.
fn default_namespace(rank: Option<usize>, world_size: Option<usize>) -> String {
    match (rank, world_size) {
        (Some(r), Some(ws)) => format!("worker-{r}-{ws}"),
        (Some(r), None) => format!("worker-{r}"),
        _ => format!("worker-{}", std::process::id()),
    }
}

impl RegistryClientConfig {
    /// Check if distributed registry is enabled via environment.
    ///
    /// Returns true if `DYN_REGISTRY_ENABLE=1` or `DYN_REGISTRY_ENABLE=true`
    pub fn is_enabled() -> bool {
        std::env::var("DYN_REGISTRY_ENABLE")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    }

    /// Create config connecting to a specific hub address.
    pub fn connect_to(hub_host: &str, port: u16) -> Self {
        Self {
            hub_addr: format!("{hub_host}:{port}").parse().unwrap_or_else(|_| {
                // Fall back to 127.0.0.1 if the host is not an IP literal
                format!("127.0.0.1:{port}").parse().unwrap()
            }),
            ..Default::default()
        }
    }

    /// Create config from environment variables.
    ///
    /// Environment variables:
    /// - `DYN_REGISTRY_ENABLE`: Set to "1" or "true" to enable distributed registry
    /// - `DYN_REGISTRY_CLIENT_ADDR`: Hub address to connect to (default: 127.0.0.1:5555)
    /// - `DYN_REGISTRY_CLIENT_NAMESPACE`: Namespace identifier (default: "worker-{pid}")
    /// - `DYN_REGISTRY_CLIENT_BATCH_SIZE`: Batch size (default: 100)
    /// - `DYN_REGISTRY_CLIENT_BATCH_TIMEOUT_MS`: Batch timeout in ms (default: 10)
    /// - `DYN_REGISTRY_CLIENT_LOCAL_CACHE`: Local cache capacity (default: 0)
    ///
    /// Note: For rank/world_size-based namespaces, use `with_rank_and_world_size()`
    /// after calling `from_env()`.
    pub fn from_env() -> Self {
        Self {
            hub_addr: std::env::var("DYN_REGISTRY_CLIENT_ADDR")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| "127.0.0.1:5555".parse().unwrap()),
            rank: None,
            world_size: None,
            namespace: std::env::var("DYN_REGISTRY_CLIENT_NAMESPACE")
                .unwrap_or_else(|_| default_namespace(None, None)),
            batch_size: std::env::var("DYN_REGISTRY_CLIENT_BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100),
            batch_timeout: Duration::from_millis(
                std::env::var("DYN_REGISTRY_CLIENT_BATCH_TIMEOUT_MS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10),
            ),
            local_cache_capacity: std::env::var("DYN_REGISTRY_CLIENT_LOCAL_CACHE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
        }
    }

    /// Enable local caching with specified capacity.
    pub fn with_local_cache(mut self, capacity: u64) -> Self {
        self.local_cache_capacity = capacity;
        self
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set namespace explicitly.
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Set rank and world size, updating the namespace accordingly.
    ///
    /// This generates a namespace of the form "worker-{rank}-{world_size}".
    pub fn with_rank_and_world_size(mut self, rank: usize, world_size: usize) -> Self {
        self.rank = Some(rank);
        self.world_size = Some(world_size);
        self.namespace = default_namespace(self.rank, self.world_size);
        self
    }

    /// Create a velo TCP transport connected to the hub.
    pub async fn create_transport(&self) -> Result<super::core::VeloClientTransport> {
        super::core::VeloClientTransport::connect(self.hub_addr).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_config_default() {
        let config = RegistryHubConfig::default();
        assert_eq!(config.capacity, 1_000_000);
        assert_eq!(config.addr.port(), 5555);
    }

    #[test]
    fn test_hub_config_with_capacity() {
        let config = RegistryHubConfig::with_capacity(500_000);
        assert_eq!(config.capacity, 500_000);
    }

    #[test]
    fn test_hub_config_on_port() {
        let config = RegistryHubConfig::on_port(6000);
        assert_eq!(config.addr.port(), 6000);
    }

    #[test]
    fn test_client_config_default() {
        let config = RegistryClientConfig::default();
        assert_eq!(config.hub_addr.port(), 5555);
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.batch_timeout, Duration::from_millis(10));
        assert_eq!(config.local_cache_capacity, 0);
    }

    #[test]
    fn test_client_config_connect_to() {
        let config = RegistryClientConfig::connect_to("127.0.0.1", 6000);
        assert_eq!(config.hub_addr.port(), 6000);
    }

    #[test]
    fn test_client_config_builder() {
        let config = RegistryClientConfig::default()
            .with_local_cache(10_000)
            .with_batch_size(50);
        assert_eq!(config.local_cache_capacity, 10_000);
        assert_eq!(config.batch_size, 50);
    }

    #[test]
    fn test_client_config_with_rank_and_world_size() {
        let config = RegistryClientConfig::default().with_rank_and_world_size(3, 8);
        assert_eq!(config.rank, Some(3));
        assert_eq!(config.world_size, Some(8));
        assert_eq!(config.namespace, "worker-3-8");
    }

    #[test]
    fn test_default_namespace_fallback() {
        let config = RegistryClientConfig::default();
        assert!(config.namespace.starts_with("worker-"));
        assert!(config.namespace.contains(&std::process::id().to_string()));
    }
}
