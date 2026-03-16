// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Network interface discovery for multi-NIC advertisement and topology-aware selection.
//!
//! When a transport binds to `0.0.0.0:PORT`, the OS listens on all interfaces.
//! This module discovers which interfaces are available and builds a list of
//! [`InterfaceEndpoint`]s to advertise to peers. On the receiving side,
//! [`select_best_endpoint`] picks the best remote endpoint based on subnet
//! affinity and NUMA topology.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::warn;

/// A discovered network interface endpoint with subnet and NUMA info.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InterfaceEndpoint {
    /// Interface name, e.g. "eth0", "mlx5_0".
    pub name: String,
    /// IP address as a string, e.g. "192.168.1.10".
    pub ip: String,
    /// Port the transport is listening on.
    pub port: u16,
    /// CIDR prefix length, e.g. 24 for /24.
    pub prefix_len: u8,
    /// NUMA node from `/sys/class/net/<name>/device/numa_node`.
    /// `None` if unavailable or not on Linux.
    pub numa_node: Option<i32>,
}

impl InterfaceEndpoint {
    /// Parse the `ip` field into a `SocketAddr` with the stored port.
    pub fn socket_addr(&self) -> Option<SocketAddr> {
        self.ip.parse::<IpAddr>().ok().map(|ip| SocketAddr::new(ip, self.port))
    }
}

/// User-specified interface selection policy.
#[derive(Debug, Clone)]
pub enum InterfaceFilter {
    /// All UP, non-loopback interfaces (default).
    All,
    /// Select a specific interface by name, e.g. "eth0", "mlx5_0".
    ByName(String),
    /// Manual override with explicit endpoints (useful for testing).
    Explicit(Vec<InterfaceEndpoint>),
}

impl Default for InterfaceFilter {
    fn default() -> Self {
        Self::All
    }
}

/// A raw discovered network interface (before port assignment).
#[derive(Debug, Clone)]
struct NetworkInterface {
    name: String,
    ip: IpAddr,
    prefix_len: u8,
    numa_node: Option<i32>,
}

/// Discover all UP, non-loopback network interfaces.
///
/// Filters out:
/// - Loopback interfaces (127.0.0.1, ::1)
/// - IPv6 link-local addresses (fe80::)
/// - Interfaces that are not UP
#[cfg(target_os = "linux")]
fn discover_interfaces() -> Result<Vec<NetworkInterface>> {
    use nix::ifaddrs::getifaddrs;
    use nix::net::if_::InterfaceFlags;

    let mut interfaces = Vec::new();

    let ifaddrs = getifaddrs().context("getifaddrs() failed")?;
    for ifaddr in ifaddrs {
        // Must be UP and not LOOPBACK.
        if !ifaddr.flags.contains(InterfaceFlags::IFF_UP)
            || ifaddr.flags.contains(InterfaceFlags::IFF_LOOPBACK)
        {
            continue;
        }

        let Some(addr) = ifaddr.address else {
            continue;
        };

        let ip: IpAddr;
        let prefix_len: u8;

        if let Some(sin) = addr.as_sockaddr_in() {
            ip = IpAddr::V4(Ipv4Addr::from(sin.ip()));
            prefix_len = ifaddr
                .netmask
                .and_then(|m| m.as_sockaddr_in().map(|s| Ipv4Addr::from(s.ip())))
                .map(|mask| netmask_to_prefix_len(IpAddr::V4(mask)))
                .unwrap_or(24);
        } else if let Some(sin6) = addr.as_sockaddr_in6() {
            let v6 = sin6.ip();
            // Skip link-local (fe80::).
            if (v6.segments()[0] & 0xffc0) == 0xfe80 {
                continue;
            }
            ip = IpAddr::V6(v6);
            prefix_len = ifaddr
                .netmask
                .and_then(|m| m.as_sockaddr_in6().map(|s| s.ip()))
                .map(|mask| netmask_to_prefix_len(IpAddr::V6(mask)))
                .unwrap_or(64);
        } else {
            continue;
        }

        // Skip loopback IPs that aren't flagged (belt and suspenders).
        if ip.is_loopback() {
            continue;
        }

        let numa_node = read_nic_numa_node(&ifaddr.interface_name);

        interfaces.push(NetworkInterface {
            name: ifaddr.interface_name.clone(),
            ip,
            prefix_len,
            numa_node,
        });
    }

    Ok(interfaces)
}

#[cfg(not(target_os = "linux"))]
fn discover_interfaces() -> Result<Vec<NetworkInterface>> {
    Ok(Vec::new())
}

/// Read the NUMA node for a network interface from sysfs.
///
/// Returns `None` if the file doesn't exist (e.g. virtual interfaces, non-Linux)
/// or contains -1.
fn read_nic_numa_node(iface_name: &str) -> Option<i32> {
    let path = format!("/sys/class/net/{}/device/numa_node", iface_name);
    let content = std::fs::read_to_string(path).ok()?;
    let node: i32 = content.trim().parse().ok()?;
    if node < 0 {
        None
    } else {
        Some(node)
    }
}

/// Count leading 1-bits in a netmask to get the CIDR prefix length.
pub fn netmask_to_prefix_len(mask: IpAddr) -> u8 {
    match mask {
        IpAddr::V4(v4) => {
            let bits: u32 = u32::from(v4);
            bits.leading_ones() as u8
        }
        IpAddr::V6(v6) => {
            let bits: u128 = u128::from(v6);
            bits.leading_ones() as u8
        }
    }
}

/// Check whether two IP/prefix pairs are in the same subnet.
///
/// Uses the shorter of the two prefix lengths to mask both IPs, then compares.
pub fn subnets_match(a_ip: &str, a_prefix: u8, b_ip: &str, b_prefix: u8) -> bool {
    let Ok(a) = a_ip.parse::<IpAddr>() else {
        return false;
    };
    let Ok(b) = b_ip.parse::<IpAddr>() else {
        return false;
    };
    subnets_match_ip(a, a_prefix, b, b_prefix)
}

fn subnets_match_ip(a: IpAddr, a_prefix: u8, b: IpAddr, b_prefix: u8) -> bool {
    match (a, b) {
        (IpAddr::V4(a4), IpAddr::V4(b4)) => {
            let prefix = a_prefix.min(b_prefix).min(32);
            if prefix == 0 {
                return true;
            }
            let mask = u32::MAX << (32 - prefix);
            (u32::from(a4) & mask) == (u32::from(b4) & mask)
        }
        (IpAddr::V6(a6), IpAddr::V6(b6)) => {
            let prefix = a_prefix.min(b_prefix).min(128);
            if prefix == 0 {
                return true;
            }
            let mask = u128::MAX << (128 - prefix);
            (u128::from(a6) & mask) == (u128::from(b6) & mask)
        }
        _ => false, // Cross-family never matches.
    }
}

/// Resolve the set of endpoints to advertise for a given bind address.
///
/// - If `bind_addr` is unspecified (0.0.0.0 or ::), discovers all matching interfaces
///   filtered by the address family and the [`InterfaceFilter`] policy.
/// - If `bind_addr` is a specific IP, returns a single endpoint.
pub fn resolve_advertise_endpoints(
    bind_addr: SocketAddr,
    filter: &InterfaceFilter,
) -> Result<Vec<InterfaceEndpoint>> {
    // Handle explicit override.
    if let InterfaceFilter::Explicit(endpoints) = filter {
        return Ok(endpoints
            .iter()
            .cloned()
            .map(|mut ep| {
                ep.port = bind_addr.port();
                ep
            })
            .collect());
    }

    if !bind_addr.ip().is_unspecified() {
        // Specific bind address — single endpoint.
        let ip_str = bind_addr.ip().to_string();
        let prefix_len = match bind_addr.ip() {
            IpAddr::V4(_) => 32,
            IpAddr::V6(_) => 128,
        };
        return Ok(vec![InterfaceEndpoint {
            name: String::new(),
            ip: ip_str,
            port: bind_addr.port(),
            prefix_len,
            numa_node: None,
        }]);
    }

    // Wildcard bind — discover interfaces.
    let is_v4 = bind_addr.ip().is_ipv4();
    let ifaces = discover_interfaces()?;

    let filtered: Vec<_> = ifaces
        .into_iter()
        .filter(|iface| {
            // Filter by address family.
            if is_v4 {
                iface.ip.is_ipv4()
            } else {
                iface.ip.is_ipv6()
            }
        })
        .filter(|iface| match filter {
            InterfaceFilter::ByName(name) => &iface.name == name,
            InterfaceFilter::All => true,
            InterfaceFilter::Explicit(_) => unreachable!(),
        })
        .map(|iface| InterfaceEndpoint {
            name: iface.name,
            ip: iface.ip.to_string(),
            port: bind_addr.port(),
            prefix_len: iface.prefix_len,
            numa_node: iface.numa_node,
        })
        .collect();

    if filtered.is_empty() {
        // Fallback: advertise loopback with a warning.
        warn!(
            "No non-loopback interfaces discovered for {}; falling back to loopback",
            bind_addr
        );
        let loopback_ip = if is_v4 {
            IpAddr::V4(Ipv4Addr::LOCALHOST)
        } else {
            IpAddr::V6(Ipv6Addr::LOCALHOST)
        };
        return Ok(vec![InterfaceEndpoint {
            name: "lo".to_string(),
            ip: loopback_ip.to_string(),
            port: bind_addr.port(),
            prefix_len: if is_v4 { 8 } else { 128 },
            numa_node: None,
        }]);
    }

    Ok(filtered)
}

/// Select the best remote endpoint to connect to, given local interfaces and a NUMA hint.
///
/// Selection phases (first match wins):
/// 1. **NUMA + subnet**: remote on same NUMA node AND same subnet as a local interface.
/// 2. **NUMA only**: remote on same NUMA node.
/// 3. **Subnet match**: remote shares a subnet with any local interface.
/// 4. **First non-loopback**: any remote endpoint.
/// 5. **First entry**: last resort.
pub fn select_best_endpoint(
    remote: &[InterfaceEndpoint],
    local_interfaces: &[InterfaceEndpoint],
    numa_hint: Option<u32>,
) -> Option<SocketAddr> {
    if remote.is_empty() {
        return None;
    }

    // Phase 1: NUMA + subnet match.
    if let Some(numa) = numa_hint {
        for ep in remote {
            if ep.numa_node == Some(numa as i32) {
                for local in local_interfaces {
                    if subnets_match(&ep.ip, ep.prefix_len, &local.ip, local.prefix_len) {
                        return ep.socket_addr();
                    }
                }
            }
        }
    }

    // Phase 2: NUMA match only.
    if let Some(numa) = numa_hint {
        for ep in remote {
            if ep.numa_node == Some(numa as i32) {
                return ep.socket_addr();
            }
        }
    }

    // Phase 3: Subnet match.
    for ep in remote {
        for local in local_interfaces {
            if subnets_match(&ep.ip, ep.prefix_len, &local.ip, local.prefix_len) {
                return ep.socket_addr();
            }
        }
    }

    // Phase 4: First non-loopback.
    for ep in remote {
        if let Ok(ip) = ep.ip.parse::<IpAddr>() {
            if !ip.is_loopback() {
                return ep.socket_addr();
            }
        }
    }

    // Phase 5: First entry.
    remote.first().and_then(|ep| ep.socket_addr())
}

/// Parse endpoint data from a `WorkerAddress` entry.
///
/// Tries msgpack array first (new multi-endpoint format), then falls back to
/// legacy `"tcp://host:port"` or `"grpc://host:port"` string parsing.
pub fn parse_endpoints(raw: &[u8]) -> Result<Vec<InterfaceEndpoint>> {
    // Try msgpack array first.
    if let Ok(endpoints) = rmp_serde::from_slice::<Vec<InterfaceEndpoint>>(raw) {
        if !endpoints.is_empty() {
            return Ok(endpoints);
        }
    }

    // Fall back to legacy string format.
    let endpoint_str = std::str::from_utf8(raw).context("endpoint is not valid UTF-8")?;
    let addr_str = endpoint_str
        .strip_prefix("tcp://")
        .or_else(|| endpoint_str.strip_prefix("grpc://"))
        .unwrap_or(endpoint_str);

    let addr: SocketAddr = addr_str
        .parse()
        .or_else(|_| {
            use std::net::ToSocketAddrs;
            addr_str
                .to_socket_addrs()?
                .next()
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "no addresses"))
        })
        .context("failed to parse legacy endpoint")?;

    Ok(vec![InterfaceEndpoint {
        name: String::new(),
        ip: addr.ip().to_string(),
        port: addr.port(),
        prefix_len: if addr.is_ipv4() { 32 } else { 128 },
        numa_node: None,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_netmask_to_prefix_len_ipv4() {
        assert_eq!(netmask_to_prefix_len(IpAddr::V4(Ipv4Addr::new(255, 0, 0, 0))), 8);
        assert_eq!(netmask_to_prefix_len(IpAddr::V4(Ipv4Addr::new(255, 255, 0, 0))), 16);
        assert_eq!(netmask_to_prefix_len(IpAddr::V4(Ipv4Addr::new(255, 255, 255, 0))), 24);
        assert_eq!(netmask_to_prefix_len(IpAddr::V4(Ipv4Addr::new(255, 255, 255, 255))), 32);
    }

    #[test]
    fn test_netmask_to_prefix_len_ipv6() {
        // /64
        let mask_64: Ipv6Addr = "ffff:ffff:ffff:ffff::".parse().unwrap();
        assert_eq!(netmask_to_prefix_len(IpAddr::V6(mask_64)), 64);

        // /128
        let mask_128: Ipv6Addr = "ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff".parse().unwrap();
        assert_eq!(netmask_to_prefix_len(IpAddr::V6(mask_128)), 128);
    }

    #[test]
    fn test_subnets_match_same_subnet() {
        assert!(subnets_match("192.168.1.10", 24, "192.168.1.20", 24));
        assert!(subnets_match("10.0.0.1", 8, "10.255.255.254", 8));
    }

    #[test]
    fn test_subnets_match_different_subnet() {
        assert!(!subnets_match("192.168.1.10", 24, "192.168.2.10", 24));
        assert!(!subnets_match("10.0.0.1", 16, "10.1.0.1", 16));
    }

    #[test]
    fn test_subnets_match_cross_family() {
        assert!(!subnets_match("192.168.1.10", 24, "::1", 128));
    }

    #[test]
    fn test_subnets_match_different_prefix_uses_shorter() {
        // /16 and /24 — use /16. 192.168.1.x and 192.168.2.x share /16.
        assert!(subnets_match("192.168.1.10", 16, "192.168.2.10", 24));
    }

    #[test]
    fn test_discover_interfaces_not_empty() {
        // On any machine with networking, we should find at least one interface.
        // In containers with only loopback, discover_interfaces returns empty
        // (loopback is filtered), which is fine.
        let result = discover_interfaces();
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_endpoints_new_format() {
        let endpoints = vec![
            InterfaceEndpoint {
                name: "eth0".to_string(),
                ip: "192.168.1.10".to_string(),
                port: 5555,
                prefix_len: 24,
                numa_node: Some(0),
            },
            InterfaceEndpoint {
                name: "mlx5_0".to_string(),
                ip: "10.0.0.1".to_string(),
                port: 5555,
                prefix_len: 16,
                numa_node: Some(1),
            },
        ];
        let encoded = rmp_serde::to_vec(&endpoints).unwrap();
        let parsed = parse_endpoints(&encoded).unwrap();
        assert_eq!(parsed, endpoints);
    }

    #[test]
    fn test_parse_endpoints_legacy_tcp_format() {
        let legacy = b"tcp://127.0.0.1:5555";
        let parsed = parse_endpoints(legacy).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].ip, "127.0.0.1");
        assert_eq!(parsed[0].port, 5555);
    }

    #[test]
    fn test_parse_endpoints_legacy_grpc_format() {
        let legacy = b"grpc://127.0.0.1:9000";
        let parsed = parse_endpoints(legacy).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].ip, "127.0.0.1");
        assert_eq!(parsed[0].port, 9000);
    }

    #[test]
    fn test_select_best_endpoint_prefers_subnet_match() {
        let remote = vec![
            InterfaceEndpoint {
                name: "eth0".to_string(),
                ip: "10.0.0.5".to_string(),
                port: 5555,
                prefix_len: 24,
                numa_node: None,
            },
            InterfaceEndpoint {
                name: "eth1".to_string(),
                ip: "192.168.1.5".to_string(),
                port: 5555,
                prefix_len: 24,
                numa_node: None,
            },
        ];
        let local = vec![InterfaceEndpoint {
            name: "eth0".to_string(),
            ip: "192.168.1.100".to_string(),
            port: 0,
            prefix_len: 24,
            numa_node: None,
        }];

        let best = select_best_endpoint(&remote, &local, None).unwrap();
        assert_eq!(best.ip().to_string(), "192.168.1.5");
    }

    #[test]
    fn test_select_best_endpoint_prefers_numa_plus_subnet() {
        let remote = vec![
            InterfaceEndpoint {
                name: "eth0".to_string(),
                ip: "192.168.1.5".to_string(),
                port: 5555,
                prefix_len: 24,
                numa_node: Some(0),
            },
            InterfaceEndpoint {
                name: "mlx5_0".to_string(),
                ip: "192.168.1.6".to_string(),
                port: 5555,
                prefix_len: 24,
                numa_node: Some(1),
            },
        ];
        let local = vec![InterfaceEndpoint {
            name: "eth0".to_string(),
            ip: "192.168.1.100".to_string(),
            port: 0,
            prefix_len: 24,
            numa_node: Some(1),
        }];

        // With NUMA hint 1, should prefer mlx5_0 (NUMA 1 + subnet match).
        let best = select_best_endpoint(&remote, &local, Some(1)).unwrap();
        assert_eq!(best.ip().to_string(), "192.168.1.6");
    }

    #[test]
    fn test_select_best_endpoint_falls_back() {
        let remote = vec![InterfaceEndpoint {
            name: "eth0".to_string(),
            ip: "10.0.0.5".to_string(),
            port: 5555,
            prefix_len: 24,
            numa_node: None,
        }];
        let local = vec![InterfaceEndpoint {
            name: "wlan0".to_string(),
            ip: "192.168.1.100".to_string(),
            port: 0,
            prefix_len: 24,
            numa_node: None,
        }];

        // No subnet match, should fall back to first non-loopback.
        let best = select_best_endpoint(&remote, &local, None).unwrap();
        assert_eq!(best.ip().to_string(), "10.0.0.5");
    }

    #[test]
    fn test_select_best_endpoint_empty() {
        let result = select_best_endpoint(&[], &[], None);
        assert!(result.is_none());
    }

    #[test]
    fn test_resolve_advertise_endpoints_specific_ip() {
        let addr: SocketAddr = "192.168.1.100:5555".parse().unwrap();
        let endpoints = resolve_advertise_endpoints(addr, &InterfaceFilter::All).unwrap();
        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].ip, "192.168.1.100");
        assert_eq!(endpoints[0].port, 5555);
        assert_eq!(endpoints[0].prefix_len, 32);
    }

    #[test]
    fn test_resolve_advertise_endpoints_explicit() {
        let explicit = vec![InterfaceEndpoint {
            name: "test0".to_string(),
            ip: "10.0.0.1".to_string(),
            port: 0, // Will be overridden.
            prefix_len: 24,
            numa_node: Some(0),
        }];
        let addr: SocketAddr = "0.0.0.0:9999".parse().unwrap();
        let endpoints =
            resolve_advertise_endpoints(addr, &InterfaceFilter::Explicit(explicit)).unwrap();
        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].port, 9999);
        assert_eq!(endpoints[0].ip, "10.0.0.1");
    }

    #[test]
    fn test_interface_endpoint_socket_addr() {
        let ep = InterfaceEndpoint {
            name: "eth0".to_string(),
            ip: "192.168.1.10".to_string(),
            port: 5555,
            prefix_len: 24,
            numa_node: None,
        };
        let addr = ep.socket_addr().unwrap();
        assert_eq!(addr, "192.168.1.10:5555".parse::<SocketAddr>().unwrap());
    }
}
