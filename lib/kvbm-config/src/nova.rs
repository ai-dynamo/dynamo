// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Nova transport and discovery configuration.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::discovery::DiscoveryConfig;

/// Nova configuration combining backend and discovery settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct NovaConfig {
    #[validate(nested)]
    pub backend: NovaBackendConfig,

    /// Discovery configuration. None = discovery disabled.
    #[serde(default)]
    pub discovery: Option<DiscoveryConfig>,
}

impl NovaConfig {
    /// Build a Nova instance from this configuration.
    ///
    /// This creates:
    /// 1. A TCP transport bound to the configured address
    /// 2. A discovery backend based on the configured type (if any)
    /// 3. A Nova instance combining both
    pub async fn build_nova(&self) -> Result<std::sync::Arc<dynamo_nova::Nova>> {
        use std::sync::Arc;
        use std::time::Duration;

        use dynamo_nova::Nova;
        use dynamo_nova_backend::tcp::TcpTransportBuilder;

        // 1. Build TCP transport
        let bind_addr = self.backend.resolve_bind_addr()?;
        let tcp_transport = TcpTransportBuilder::new()
            .bind_addr(bind_addr)
            .build()
            .context("Failed to build TCP transport")?;
        let tcp_transport = Arc::new(tcp_transport);

        tracing::info!("Built TCP transport bound to {}", bind_addr);

        // 2. Build discovery backend based on configuration
        let mut builder = Nova::builder().add_transport(tcp_transport);

        if let Some(discovery_config) = &self.discovery {
            match discovery_config {
                DiscoveryConfig::Etcd(cfg) => {
                    use dynamo_nova_discovery::systems::EtcdConfigBuilder;

                    let discovery_system = EtcdConfigBuilder::default()
                        .cluster_id(&cfg.cluster_id)
                        .endpoints(cfg.endpoints.clone())
                        .ttl(Duration::from_secs(cfg.ttl_secs))
                        .operation_timeout(Duration::from_secs(cfg.operation_timeout_secs))
                        .max_retries(cfg.max_retries)
                        .build()
                        .await
                        .context("Failed to build etcd discovery system")?;

                    if let Some(peer_discovery) = discovery_system.peer_discovery() {
                        builder = builder.add_discovery_backend(peer_discovery);
                        tracing::info!("Built etcd discovery with cluster_id: {}", cfg.cluster_id);
                    }
                }
                DiscoveryConfig::P2p(cfg) => {
                    use dynamo_nova_discovery::systems::P2pConfigBuilder;

                    let discovery_system = P2pConfigBuilder::default()
                        .cluster_id(&cfg.cluster_id)
                        .listen_port(cfg.listen_port)
                        .bootstrap_peers(cfg.bootstrap_peers.clone())
                        .replication_factor(cfg.replication_factor)
                        .enable_mdns(cfg.enable_mdns)
                        .record_ttl_secs(cfg.record_ttl_secs)
                        .build()
                        .await
                        .context("Failed to build P2P discovery system")?;

                    if let Some(peer_discovery) = discovery_system.peer_discovery() {
                        builder = builder.add_discovery_backend(peer_discovery);
                        tracing::info!("Built P2P discovery with cluster_id: {}", cfg.cluster_id);
                    }
                }
                DiscoveryConfig::Filesystem(cfg) => {
                    use dynamo_nova_discovery::systems::FilesystemPeerDiscovery;

                    let peer_discovery = FilesystemPeerDiscovery::new(&cfg.path)
                        .context("Failed to build filesystem discovery")?;

                    builder = builder.add_discovery_backend(Arc::new(peer_discovery));
                    tracing::info!("Built filesystem discovery from: {:?}", cfg.path);
                }
            }
        }

        // 3. Build Nova
        let nova = builder.build().await.context("Failed to build Nova")?;

        Ok(nova)
    }
}

/// Nova backend (transport) configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct NovaBackendConfig {
    /// IP address to bind (mutually exclusive with tcp_interface).
    /// e.g., "0.0.0.0" or "192.168.1.100"
    pub tcp_addr: Option<String>,

    /// Network interface to bind (mutually exclusive with tcp_addr).
    /// e.g., "eth0", "ens192"
    pub tcp_interface: Option<String>,

    /// TCP port to bind. 0 means OS-assigned (ephemeral port).
    #[serde(default)]
    pub tcp_port: u16,
}

impl NovaBackendConfig {
    /// Resolve the bind address from either interface name or explicit address.
    ///
    /// Returns error if both tcp_addr and tcp_interface are specified.
    pub fn resolve_bind_addr(&self) -> Result<SocketAddr> {
        let ip = match (&self.tcp_addr, &self.tcp_interface) {
            (Some(_), Some(_)) => {
                bail!("tcp_addr and tcp_interface are mutually exclusive")
            }
            (Some(addr), None) => addr
                .parse::<IpAddr>()
                .with_context(|| format!("Invalid IP address: {}", addr))?,
            (None, Some(iface)) => get_interface_ip(iface)
                .with_context(|| format!("Failed to get IP for interface: {}", iface))?,
            (None, None) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        };
        Ok(SocketAddr::new(ip, self.tcp_port))
    }
}

/// Get the IP address for a network interface.
fn get_interface_ip(interface_name: &str) -> Result<IpAddr> {
    use nix::ifaddrs::getifaddrs;

    let addrs = getifaddrs().context("Failed to get interface addresses")?;

    for ifaddr in addrs {
        if ifaddr.interface_name == interface_name
            && let Some(addr) = ifaddr.address
        {
            // Prefer IPv4 addresses
            if let Some(sockaddr) = addr.as_sockaddr_in() {
                return Ok(IpAddr::V4(sockaddr.ip()));
            }
            // Fall back to IPv6 if no IPv4
            if let Some(sockaddr) = addr.as_sockaddr_in6() {
                return Ok(IpAddr::V6(sockaddr.ip()));
            }
        }
    }

    bail!("No IP address found for interface: {}", interface_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_backend_config() {
        let config = NovaBackendConfig::default();
        assert!(config.tcp_addr.is_none());
        assert!(config.tcp_interface.is_none());
        assert_eq!(config.tcp_port, 0);
    }

    #[test]
    fn test_resolve_bind_addr_default() {
        let config = NovaBackendConfig::default();
        let addr = config.resolve_bind_addr().unwrap();
        assert_eq!(addr.ip(), IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        assert_eq!(addr.port(), 0);
    }

    #[test]
    fn test_resolve_bind_addr_explicit() {
        let config = NovaBackendConfig {
            tcp_addr: Some("192.168.1.100".to_string()),
            tcp_interface: None,
            tcp_port: 8080,
        };
        let addr = config.resolve_bind_addr().unwrap();
        assert_eq!(addr.ip(), IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)));
        assert_eq!(addr.port(), 8080);
    }

    #[test]
    fn test_resolve_bind_addr_mutual_exclusivity() {
        let config = NovaBackendConfig {
            tcp_addr: Some("0.0.0.0".to_string()),
            tcp_interface: Some("eth0".to_string()),
            tcp_port: 0,
        };
        let result = config.resolve_bind_addr();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("mutually exclusive")
        );
    }
}
