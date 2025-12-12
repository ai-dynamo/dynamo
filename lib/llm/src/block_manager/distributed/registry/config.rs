// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration types for the distributed registry.

use std::time::Duration;

/// Hub configuration for the registry server.
///
/// # Example
/// ```ignore
/// let config = RegistryHubConfig {
///     capacity: 1_000_000,
///     query_addr: "tcp://*:5555".to_string(),
///     register_addr: "tcp://*:5556".to_string(),
///     lease_timeout: Duration::from_secs(30),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RegistryHubConfig {
    /// Registry capacity (number of entries).
    ///
    /// Memory impact: ~72 bytes per entry (Moka overhead).
    /// - 100K entries ≈ 7.2 MB
    /// - 1M entries ≈ 72 MB
    /// - 10M entries ≈ 720 MB
    pub capacity: u64,

    /// ZMQ REP socket address for queries (REQ/REP pattern).
    ///
    /// Workers send queries here and wait for responses.
    /// Example: "tcp://*:5555" or "tcp://0.0.0.0:5555"
    pub query_addr: String,

    /// ZMQ SUB socket address for registrations (PUB/SUB pattern).
    ///
    /// Workers publish registrations here (fire-and-forget).
    /// Example: "tcp://*:5556" or "tcp://0.0.0.0:5556"
    pub register_addr: String,

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
            query_addr: "tcp://*:5555".to_string(),
            register_addr: "tcp://*:5556".to_string(),
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

    /// Create config from environment variables.
    ///
    /// Environment variables:
    /// - `DYN_REGISTRY_HUB_CAPACITY`: Registry capacity (default: 1000000)
    /// - `DYN_REGISTRY_HUB_QUERY_ADDR`: Query address (default: tcp://*:5555)
    /// - `DYN_REGISTRY_HUB_REGISTER_ADDR`: Register address (default: tcp://*:5556)
    /// - `DYN_REGISTRY_HUB_LEASE_TIMEOUT_SECS`: Lease timeout in seconds (default: 30)
    pub fn from_env() -> Self {
        Self {
            capacity: std::env::var("DYN_REGISTRY_HUB_CAPACITY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1_000_000),
            query_addr: std::env::var("DYN_REGISTRY_HUB_QUERY_ADDR")
                .unwrap_or_else(|_| "tcp://*:5555".to_string()),
            register_addr: std::env::var("DYN_REGISTRY_HUB_REGISTER_ADDR")
                .unwrap_or_else(|_| "tcp://*:5556".to_string()),
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
/// ```ignore
/// let config = RegistryClientConfig {
///     hub_query_addr: "tcp://leader:5555".to_string(),
///     hub_register_addr: "tcp://leader:5556".to_string(),
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct RegistryClientConfig {
    /// Hub query address to connect to (REQ/REP).
    ///
    /// Example: "tcp://leader:5555" or "tcp://192.168.1.100:5555"
    pub hub_query_addr: String,

    /// Hub register address to connect to (PUB/SUB).
    ///
    /// Example: "tcp://leader:5556" or "tcp://192.168.1.100:5556"
    pub hub_register_addr: String,

    /// Bucket name for this worker's storage.
    ///
    /// Each worker has its own bucket for storing KV blocks. This is used
    /// as part of the registry key to enable cross-instance deduplication.
    /// Example: "worker-0", "instance-abc123"
    pub bucket_name: String,

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

    /// Request timeout for queries.
    ///
    /// How long to wait for a response from the hub before timing out.
    pub request_timeout: Duration,

    /// Optional local cache capacity (0 = disabled).
    ///
    /// If > 0, the client maintains a local Moka cache to reduce
    /// network round-trips for frequently accessed hashes.
    pub local_cache_capacity: u64,
}

impl Default for RegistryClientConfig {
    fn default() -> Self {
        Self {
            hub_query_addr: "tcp://localhost:5555".to_string(),
            hub_register_addr: "tcp://localhost:5556".to_string(),
            bucket_name: "default".to_string(),
            batch_size: 100,
            batch_timeout: Duration::from_millis(10),
            request_timeout: Duration::from_secs(5),
            local_cache_capacity: 0,
        }
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
    pub fn connect_to(hub_host: &str, query_port: u16, register_port: u16) -> Self {
        Self {
            hub_query_addr: format!("tcp://{}:{}", hub_host, query_port),
            hub_register_addr: format!("tcp://{}:{}", hub_host, register_port),
            ..Default::default()
        }
    }

    /// Create config from environment variables.
    ///
    /// Environment variables:
    /// - `DYN_REGISTRY_ENABLE`: Set to "1" or "true" to enable distributed registry
    /// - `DYN_REGISTRY_CLIENT_QUERY_ADDR`: Query address (default: tcp://localhost:5555)
    /// - `DYN_REGISTRY_CLIENT_REGISTER_ADDR`: Register address (default: tcp://localhost:5556)
    /// - `DYN_KVBM_OBJECT_BUCKET`: object bucket name (reused for registry bucket identification)
    /// - `DYN_REGISTRY_CLIENT_BATCH_SIZE`: Batch size (default: 100)
    /// - `DYN_REGISTRY_CLIENT_BATCH_TIMEOUT_MS`: Batch timeout in ms (default: 10)
    /// - `DYN_REGISTRY_CLIENT_REQUEST_TIMEOUT_MS`: Request timeout in ms (default: 5000)
    /// - `DYN_REGISTRY_CLIENT_LOCAL_CACHE`: Local cache capacity (default: 0)
    pub fn from_env() -> Self {
        Self {
            hub_query_addr: std::env::var("DYN_REGISTRY_CLIENT_QUERY_ADDR")
                .unwrap_or_else(|_| "tcp://localhost:5555".to_string()),
            hub_register_addr: std::env::var("DYN_REGISTRY_CLIENT_REGISTER_ADDR")
                .unwrap_or_else(|_| "tcp://localhost:5556".to_string()),
            // Bucket name from DYN_KVBM_OBJECT_BUCKET.
            // Note: {worker_id} template must be substituted by the caller.
            bucket_name: std::env::var("DYN_KVBM_OBJECT_BUCKET")
                .unwrap_or_else(|_| "default".to_string()),
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
            request_timeout: Duration::from_millis(
                std::env::var("DYN_REGISTRY_CLIENT_REQUEST_TIMEOUT_MS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(5000),
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

    /// Set request timeout.
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Set bucket name.
    pub fn with_bucket_name(mut self, name: impl Into<String>) -> Self {
        self.bucket_name = name.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_config_default() {
        let config = RegistryHubConfig::default();
        assert_eq!(config.capacity, 1_000_000);
        assert_eq!(config.query_addr, "tcp://*:5555");
        assert_eq!(config.register_addr, "tcp://*:5556");
    }

    #[test]
    fn test_hub_config_with_capacity() {
        let config = RegistryHubConfig::with_capacity(500_000);
        assert_eq!(config.capacity, 500_000);
    }

    #[test]
    fn test_client_config_default() {
        let config = RegistryClientConfig::default();
        assert_eq!(config.hub_query_addr, "tcp://localhost:5555");
        assert_eq!(config.hub_register_addr, "tcp://localhost:5556");
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.batch_timeout, Duration::from_millis(10));
        assert_eq!(config.local_cache_capacity, 0);
    }

    #[test]
    fn test_client_config_connect_to() {
        let config = RegistryClientConfig::connect_to("leader.local", 6000, 6001);
        assert_eq!(config.hub_query_addr, "tcp://leader.local:6000");
        assert_eq!(config.hub_register_addr, "tcp://leader.local:6001");
    }

    #[test]
    fn test_client_config_builder() {
        let config = RegistryClientConfig::default()
            .with_local_cache(10_000)
            .with_batch_size(50)
            .with_request_timeout(Duration::from_secs(10));

        assert_eq!(config.local_cache_capacity, 10_000);
        assert_eq!(config.batch_size, 50);
        assert_eq!(config.request_timeout, Duration::from_secs(10));
    }
}

