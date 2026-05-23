// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Messenger transport and discovery configuration.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use validator::Validate;

use crate::discovery::DiscoveryConfig;

fn default_init_timeout_secs() -> u64 {
    1800
}

fn default_uds_enabled() -> bool {
    true
}

/// Messenger configuration combining backend and discovery settings.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MessengerConfig {
    #[validate(nested)]
    pub backend: MessengerBackendConfig,

    /// Discovery configuration. None = discovery disabled.
    #[serde(default)]
    pub discovery: Option<DiscoveryConfig>,

    /// Leader-worker initialization timeout in seconds.
    ///
    /// How long the leader waits for all workers to connect before giving up.
    /// Default: 1800 (30 minutes).
    ///
    /// V1 compat: `DYN_KVBM_LEADER_WORKER_INIT_TIMEOUT_SECS`
    #[serde(default = "default_init_timeout_secs")]
    #[validate(range(min = 1))]
    pub init_timeout_secs: u64,
}

impl Default for MessengerConfig {
    fn default() -> Self {
        Self {
            backend: MessengerBackendConfig::default(),
            discovery: None,
            init_timeout_secs: default_init_timeout_secs(),
        }
    }
}

// Runtime construction (`build_velo` / `build_velo_with_discovery` /
// `build_messenger`) lives in the `kvbm-runtime` crate so this crate stays
// free of the `velo` dependency. See `kvbm_runtime::build_velo*`.

/// Messenger backend (transport) configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MessengerBackendConfig {
    /// IP address to bind (mutually exclusive with tcp_interface).
    /// e.g., "0.0.0.0" or "192.168.1.100"
    pub tcp_addr: Option<String>,

    /// Network interface to bind (mutually exclusive with tcp_addr).
    /// e.g., "eth0", "ens192"
    pub tcp_interface: Option<String>,

    /// TCP port to bind. 0 means OS-assigned (ephemeral port).
    #[serde(default)]
    pub tcp_port: u16,

    /// Enable a side-by-side UDS transport for same-host peer communication.
    ///
    /// When `true`, velo prefers UDS over TCP for any peer whose advertised
    /// socket path is visible on this host's filesystem; cross-host peers
    /// transparently fall back to TCP via velo's host-affinity logic.
    #[serde(default = "default_uds_enabled")]
    pub uds_enabled: bool,

    /// Directory in which to bind the UDS socket. `None` means
    /// [`std::env::temp_dir`] (typically `/tmp`). The filename is generated
    /// per worker and is unique per process.
    #[serde(default)]
    pub uds_dir: Option<PathBuf>,
}

impl Default for MessengerBackendConfig {
    fn default() -> Self {
        Self {
            tcp_addr: None,
            tcp_interface: None,
            tcp_port: 0,
            uds_enabled: default_uds_enabled(),
            uds_dir: None,
        }
    }
}

impl MessengerBackendConfig {
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
        let config = MessengerBackendConfig::default();
        assert!(config.tcp_addr.is_none());
        assert!(config.tcp_interface.is_none());
        assert_eq!(config.tcp_port, 0);
        assert!(config.uds_enabled);
        assert!(config.uds_dir.is_none());
    }

    #[test]
    fn test_uds_disabled_via_serde() {
        // Omitted field defaults to true; explicit false survives the round-trip.
        let cfg: MessengerBackendConfig = serde_json::from_str("{}").unwrap();
        assert!(cfg.uds_enabled);

        let cfg: MessengerBackendConfig =
            serde_json::from_str(r#"{"uds_enabled": false}"#).unwrap();
        assert!(!cfg.uds_enabled);

        let cfg: MessengerBackendConfig =
            serde_json::from_str(r#"{"uds_dir": "/run/kvbm"}"#).unwrap();
        assert_eq!(
            cfg.uds_dir.as_deref(),
            Some(std::path::Path::new("/run/kvbm"))
        );
    }

    #[test]
    fn test_resolve_bind_addr_default() {
        let config = MessengerBackendConfig::default();
        let addr = config.resolve_bind_addr().unwrap();
        assert_eq!(addr.ip(), IpAddr::V4(Ipv4Addr::UNSPECIFIED));
        assert_eq!(addr.port(), 0);
    }

    #[test]
    fn test_resolve_bind_addr_explicit() {
        let config = MessengerBackendConfig {
            tcp_addr: Some("192.168.1.100".to_string()),
            tcp_port: 8080,
            ..Default::default()
        };
        let addr = config.resolve_bind_addr().unwrap();
        assert_eq!(addr.ip(), IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)));
        assert_eq!(addr.port(), 8080);
    }

    #[test]
    fn test_resolve_bind_addr_mutual_exclusivity() {
        let config = MessengerBackendConfig {
            tcp_addr: Some("0.0.0.0".to_string()),
            tcp_interface: Some("eth0".to_string()),
            ..Default::default()
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
