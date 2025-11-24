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

const DEFAULT_LISTEN_PORT: u16 = 0;
const DEFAULT_REPLICATION_FACTOR: usize = 3;
const DEFAULT_RECORD_TTL_SECS: u64 = 600;

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
    #[builder(default = "DEFAULT_LISTEN_PORT")]
    pub listen_port: u16,

    /// Bootstrap peer addresses (format: "host:port" or Multiaddr strings)
    #[builder(default = "Vec::new()")]
    pub bootstrap_peers: Vec<String>,

    /// DHT replication factor (default: 3)
    #[builder(default = "DEFAULT_REPLICATION_FACTOR")]
    pub replication_factor: usize,

    /// Enable mDNS for local network discovery (default: false)
    #[builder(default = "false")]
    pub enable_mdns: bool,

    /// Record TTL in seconds (default: 600)
    #[builder(default = "DEFAULT_RECORD_TTL_SECS")]
    pub record_ttl_secs: u64,

    /// Publication interval in seconds (default: ttl / 2)
    #[builder(default = "None")]
    pub publication_interval_secs: Option<u64>,

    /// Provider publication interval in seconds (default: ttl / 2)
    #[builder(default = "None")]
    pub provider_publication_interval_secs: Option<u64>,
}

impl P2pConfigBuilder {
    /// Build and initialize the P2P discovery system.
    pub async fn build(self) -> Result<Arc<dyn DiscoverySystem>, anyhow::Error> {
        let mut config = self
            .build_config()
            .map_err(|e| anyhow::anyhow!("Failed to build config: {e}"))?;

        // Default heartbeat intervals to half the TTL to keep records alive.
        let default_interval = (config.record_ttl_secs / 2).max(1);
        config
            .publication_interval_secs
            .get_or_insert(default_interval);
        config
            .provider_publication_interval_secs
            .get_or_insert(default_interval);

        P2pDiscoverySystem::from_config(config).await
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
    async fn from_config(config: P2pConfig) -> Result<Arc<dyn DiscoverySystem>, anyhow::Error> {
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

#[cfg(all(test, feature = "p2p"))]
mod tests {
    use super::*;
    use crate::peer::{InstanceId, WorkerAddress};
    use crate::systems::test_support::{
        checksum_validation, collision_detection, not_found_errors,
        register_and_discover_by_instance_id, register_and_discover_by_worker_id,
    };
    use crate::systems::{DiscoveryQueryError, DiscoverySystem, peer_discovery_handle};
    use std::sync::Arc;

    fn system_factory(
        cluster_id: String,
    ) -> impl std::future::Future<Output = anyhow::Result<Arc<dyn DiscoverySystem>>> {
        async move {
            P2pConfigBuilder::default()
                .cluster_id(cluster_id)
                .listen_port(DEFAULT_LISTEN_PORT)
                .build()
                .await
        }
    }

    #[tokio::test]
    async fn test_p2p_register_and_discover_by_worker_id() {
        register_and_discover_by_worker_id(system_factory)
            .await
            .expect("worker_id discovery test failed");
    }

    #[tokio::test]
    async fn test_p2p_register_and_discover_by_instance_id() {
        register_and_discover_by_instance_id(system_factory)
            .await
            .expect("instance_id discovery test failed");
    }

    #[tokio::test]
    async fn test_p2p_collision_detection() {
        collision_detection(system_factory)
            .await
            .expect("collision detection test failed");
    }

    #[tokio::test]
    async fn test_p2p_checksum_validation() {
        checksum_validation(system_factory)
            .await
            .expect("checksum validation test failed");
    }

    #[tokio::test]
    async fn test_p2p_not_found_errors() {
        not_found_errors(system_factory)
            .await
            .expect("not found error test failed");
    }

    #[tokio::test]
    async fn test_p2p_unregister_marks_tombstone() {
        let system = system_factory("test-unregister".to_string())
            .await
            .expect("Failed to build discovery system");

        let discovery =
            peer_discovery_handle(Arc::clone(&system)).expect("Peer discovery should be available");

        let instance_id = InstanceId::new_v4();
        let address = WorkerAddress::from_bytes(b"127.0.0.1:9000".as_slice());
        let worker_id = instance_id.worker_id();

        discovery
            .register_instance(instance_id, address)
            .await
            .expect("registration should succeed");

        discovery
            .unregister_instance(instance_id)
            .await
            .expect("unregister should publish tombstone");

        let result = discovery.discover_by_worker_id(worker_id).await;
        assert!(matches!(result, Err(DiscoveryQueryError::NotFound)));

        system.shutdown();
    }
}
