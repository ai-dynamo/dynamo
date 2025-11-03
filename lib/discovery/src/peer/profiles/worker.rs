// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker discovery profiles for different deployment scenarios.
//!
//! This module provides pre-configured discovery setups optimized for
//! worker instances in various environments.

use anyhow::{anyhow, Result};
use std::sync::Arc;

use crate::PeerDiscovery;

#[cfg(feature = "p2p")]
use crate::systems::{peer_discovery_handle, P2pConfig};
#[cfg(feature = "p2p")]
use crate::{InMemoryDiscovery, RankedDiscovery};

/// Worker discovery profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerProfile {
    /// Development profile: Local P2P mesh with mDNS discovery.
    ///
    /// Features:
    /// - Cache-aside pattern (local InMemory + P2P DHT)
    /// - mDNS for automatic peer discovery (no bootstrap needed)
    /// - High replication factor for resilience
    /// - Short TTL for fast churn detection
    Dev,

    /// Production profile: HTTP client to discovery service.
    ///
    /// Features:
    /// - Centralized discovery via HTTP service
    /// - Heartbeat-based TTL management
    /// - Graceful registration/unregistration
    Prod,

    /// Custom profile: User-provided configuration.
    Custom,
}

/// Configuration for the Dev worker profile.
///
/// This profile creates a local P2P mesh where workers discover each other
/// directly via mDNS, without requiring a bootstrap service. Data is cached
/// locally in memory for fast lookups, with P2P DHT as fallback.
///
/// **Publication Strategy:**
/// - Records auto-republish every 20s to keep DHT fresh
/// - Record TTL is 60s (3x republication interval)
/// - No manual heartbeat needed - DHT handles expiration
///
/// # Example
///
/// ```no_run
/// # use dynamo_am_discovery::profiles::DevProfileConfig;
/// # async fn example() -> anyhow::Result<()> {
/// let discovery = DevProfileConfig::builder()
///     .cluster_id("my-dev-cluster")
///     .build()?
///     .build()
///     .await?;
///
/// // Register with discovery
/// // discovery.register(instance_id, worker_address).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DevProfileConfig {
    /// Cluster ID for network isolation
    pub cluster_id: String,
    /// P2P listen port (default: 0 = random)
    pub listen_port: u16,
    /// DHT replication factor (default: 15 for dev resilience)
    pub replica_factor: usize,
    /// Record TTL in seconds (default: 60)
    ///
    /// How long records stay in DHT before expiring.
    /// Should be 2-3x the publication interval.
    pub record_ttl_secs: u64,
    /// Publication interval in seconds (default: 20)
    ///
    /// How often to automatically republish records.
    /// Should be less than record_ttl_secs to prevent expiration.
    pub publication_interval_secs: u64,
    /// Enable mDNS (default: true)
    pub enable_mdns: bool,
}

impl DevProfileConfig {
    /// Create a new builder for DevProfileConfig.
    pub fn builder() -> DevProfileConfigBuilder {
        DevProfileConfigBuilder::default()
    }

    /// Build a discovery instance from this configuration.
    ///
    /// Returns `RankedDiscovery<InMemory[0], P2P[10]>` for cache-aside pattern.
    ///
    /// # Returns
    ///
    /// * `Ok(Arc<dyn PeerDiscovery>)` - Successfully initialized dev profile
    /// * `Err` - Failed to setup P2P swarm
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use dynamo_am_discovery::profiles::DevProfileConfig;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let config = DevProfileConfig::builder()
    ///     .cluster_id("test-cluster")
    ///     .build()?;
    /// let discovery = config.build().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn build(self) -> Result<Arc<dyn PeerDiscovery>, anyhow::Error> {
        #[cfg(feature = "p2p")]
        {
            // Create local InMemory cache (priority 0 = highest)
            let in_memory = Arc::new(InMemoryDiscovery::new());

            // Create P2P DHT backend (priority 10 = lower than cache)
            let system = P2pConfig::builder()
                .cluster_id(&self.cluster_id)
                .listen_port(self.listen_port)
                .enable_mdns(self.enable_mdns)
                .replication_factor(self.replica_factor)
                .record_ttl_secs(self.record_ttl_secs)
                // Enable automatic republication for dev profile
                .publication_interval_secs(Some(self.publication_interval_secs))
                .provider_publication_interval_secs(Some(self.publication_interval_secs))
                .build().await?;

            let p2p = peer_discovery_handle(Arc::clone(&system))
                .ok_or_else(|| anyhow!("P2P system does not provide peer discovery"))?;

            // Combine with RankedDiscovery for cache-aside pattern
            // InMemory serves hot lookups, P2P provides distributed discovery
            Ok(Arc::new(RankedDiscovery::new(vec![
                (0, in_memory), // Local cache (highest priority)
                (10, p2p),      // DHT fallback (lower priority)
            ])))
        }

        #[cfg(not(feature = "p2p"))]
        {
            anyhow::bail!("Dev profile requires the 'p2p' feature to be enabled")
        }
    }
}

/// Builder for DevProfileConfig.
#[derive(Debug, Clone, Default)]
pub struct DevProfileConfigBuilder {
    cluster_id: Option<String>,
    listen_port: Option<u16>,
    replica_factor: Option<usize>,
    record_ttl_secs: Option<u64>,
    publication_interval_secs: Option<u64>,
    enable_mdns: Option<bool>,
}

impl DevProfileConfigBuilder {
    /// Set the cluster ID for network isolation (required).
    ///
    /// All workers must use the same cluster_id to discover each other.
    pub fn cluster_id(mut self, cluster_id: impl Into<String>) -> Self {
        self.cluster_id = Some(cluster_id.into());
        self
    }

    /// Set the P2P listen port (default: 0 = random).
    pub fn listen_port(mut self, port: u16) -> Self {
        self.listen_port = Some(port);
        self
    }

    /// Set the DHT replication factor (default: 15).
    ///
    /// Higher values provide better resilience but more network overhead.
    /// Recommended for dev: 10-20
    pub fn replica_factor(mut self, factor: usize) -> Self {
        self.replica_factor = Some(factor);
        self
    }

    /// Set the record TTL in seconds (default: 60).
    ///
    /// How long records stay in the DHT before expiring.
    /// Should be 2-3x the publication interval.
    pub fn record_ttl_secs(mut self, ttl: u64) -> Self {
        self.record_ttl_secs = Some(ttl);
        self
    }

    /// Set the publication interval in seconds (default: 20).
    ///
    /// How often to automatically republish records to the DHT.
    /// Should be less than record_ttl_secs to prevent expiration.
    pub fn publication_interval_secs(mut self, interval: u64) -> Self {
        self.publication_interval_secs = Some(interval);
        self
    }

    /// Enable mDNS for automatic local peer discovery (default: true).
    pub fn enable_mdns(mut self, enable: bool) -> Self {
        self.enable_mdns = Some(enable);
        self
    }

    /// Build the DevProfileConfig.
    ///
    /// # Errors
    ///
    /// Returns an error if the required `cluster_id` field was not set.
    pub fn build(self) -> Result<DevProfileConfig, anyhow::Error> {
        let cluster_id = self
            .cluster_id
            .ok_or_else(|| anyhow::anyhow!("cluster_id is required for DevProfileConfig"))?;

        Ok(DevProfileConfig {
            cluster_id,
            listen_port: self.listen_port.unwrap_or(0),
            replica_factor: self.replica_factor.unwrap_or(15),
            record_ttl_secs: self.record_ttl_secs.unwrap_or(60),
            publication_interval_secs: self.publication_interval_secs.unwrap_or(20),
            enable_mdns: self.enable_mdns.unwrap_or(true),
        })
    }
}

/// Dev profile constructor for convenience.
///
/// This is a convenience type alias for building dev profile configurations.
pub type DevProfile = DevProfileConfig;
