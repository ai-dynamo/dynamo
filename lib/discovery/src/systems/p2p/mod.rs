// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Libp2p-backed peer discovery system mirroring the etcd system interface.
//!
//! This implementation wraps the legacy `p2p` discovery backend in the shared
//! [`DiscoverySystem`] abstraction so callers can type-erase the runtime and
//! request concrete discovery capabilities on demand.

mod swarm;

use anyhow::Result;
use derive_builder::Builder;
use std::sync::Arc;
use validator::Validate;

use crate::peer::PeerDiscovery;

use super::DiscoverySystem;

/// Configuration for libp2p-based discovery.
///
/// # Example
///
/// ```no_run
/// use dynamo_am_discovery::systems::P2pConfig;
///
/// # async fn example() -> anyhow::Result<()> {
/// let system = P2pConfig::builder()
///     .cluster_id("my-cluster")
///     .enable_mdns(true)
///     .build()
///     .await?;
///
/// let peer_discovery = system
///     .peer_discovery()
///     .expect("p2p system always provides peer discovery");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Builder, Validate)]
#[builder(pattern = "owned", build_fn(private, name = "build_config"))]
pub struct P2pConfig {
    /// Cluster ID / swarm key for private network admission (required)
    #[builder(setter(into))]
    #[validate(custom(function = "super::validation::validate_cluster_id"))]
    pub cluster_id: String,

    /// Port to listen on for incoming connections (default: 0 = random)
    #[builder(default = "0")]
    pub listen_port: u16,

    /// Bootstrap peer addresses (format: "host:port" or Multiaddr strings)
    #[builder(default = "Vec::new()")]
    pub bootstrap_peers: Vec<String>,

    /// DHT replication factor (default: 3)
    #[builder(default = "16")]
    pub replication_factor: usize,

    /// Enable mDNS for local network discovery (default: false)
    #[builder(default = "false")]
    pub enable_mdns: bool,

    /// Record TTL in seconds (default: 600)
    #[builder(default = "600")]
    pub record_ttl_secs: u64,

    /// Publication interval in seconds (default: None = disabled)
    #[builder(default = "None")]
    pub publication_interval_secs: Option<u64>,

    /// Provider publication interval in seconds (default: None = disabled)
    #[builder(default = "None")]
    pub provider_publication_interval_secs: Option<u64>,
}

impl P2pConfigBuilder {
    /// Build and initialize the P2P discovery system.
    pub async fn build(self) -> Result<Arc<dyn DiscoverySystem>, anyhow::Error> {
        let config = self
            .build_config()
            .map_err(|e| anyhow::anyhow!("Failed to build config: {e}"))?;

        P2pDiscoverySystem::new(config).await
    }
}

struct P2pDiscoverySystem {
    config: P2pConfig,
    peer_discovery: Arc<swarm::P2pDiscovery>,
}

impl std::fmt::Debug for P2pDiscoverySystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("P2pDiscoverySystem")
            .field("cluster_id", &self.config.cluster_id)
            .field("listen_port", &self.config.listen_port)
            .field("bootstrap_peers", &self.config.bootstrap_peers)
            .finish()
    }
}

impl P2pDiscoverySystem {
    async fn new(config: P2pConfig) -> Result<Arc<dyn DiscoverySystem>, anyhow::Error> {
        let peer_discovery = Arc::new(
            swarm::P2pDiscovery::new(
                config.cluster_id.clone(),
                config.listen_port,
                config.bootstrap_peers.clone(),
                config.replication_factor,
                config.enable_mdns,
                config.record_ttl_secs,
                config.publication_interval_secs,
                config.provider_publication_interval_secs,
            )
            .await?,
        );

        Ok(Arc::new(Self {
            config,
            peer_discovery,
        }))
    }
}

impl DiscoverySystem for P2pDiscoverySystem {
    fn peer_discovery(&self) -> Option<Arc<dyn PeerDiscovery>> {
        let discovery: Arc<dyn PeerDiscovery> = self.peer_discovery.clone();
        Some(discovery)
    }

    fn shutdown(&self) {
        tracing::info!("Shutting down P2pDiscoverySystem");
        self.peer_discovery.shutdown();
    }
}

impl Drop for P2pDiscoverySystem {
    fn drop(&mut self) {
        self.shutdown();
    }
}
