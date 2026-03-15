// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared utility functions for transport implementations.

use std::net::SocketAddr;

/// Resolve a wildcard bind address to a routable address for advertisement.
///
/// When binding to 0.0.0.0 (IPv4 unspecified) or :: (IPv6 unspecified),
/// we need to advertise a routable address that peers can actually connect to.
///
/// For 0.0.0.0, we use 127.0.0.1 (localhost) which works for same-machine communication.
/// For ::, we use ::1 (IPv6 localhost).
///
/// In a production multi-node deployment, this should be replaced with actual
/// network interface discovery or explicit configuration.
pub fn resolve_advertise_address(bind_addr: SocketAddr) -> SocketAddr {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    match bind_addr.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => {
            // 0.0.0.0 -> 127.0.0.1 for local testing
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), bind_addr.port())
        }
        IpAddr::V6(ip) if ip.is_unspecified() => {
            // :: -> ::1 for local testing
            SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), bind_addr.port())
        }
        _ => {
            // Already a specific address, use as-is
            bind_addr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    #[test]
    fn test_resolve_ipv4_unspecified() {
        let bind_addr: SocketAddr = "0.0.0.0:12345".parse().unwrap();
        let resolved = resolve_advertise_address(bind_addr);
        assert_eq!(resolved.ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
        assert_eq!(resolved.port(), 12345);
    }

    #[test]
    fn test_resolve_ipv4_specific() {
        let specific: SocketAddr = "192.168.1.100:8080".parse().unwrap();
        let resolved = resolve_advertise_address(specific);
        assert_eq!(resolved, specific);
    }

    #[test]
    fn test_resolve_ipv6_unspecified() {
        let bind_addr: SocketAddr = "[::]:12345".parse().unwrap();
        let resolved = resolve_advertise_address(bind_addr);
        assert_eq!(resolved.ip(), IpAddr::V6(Ipv6Addr::LOCALHOST));
        assert_eq!(resolved.port(), 12345);
    }

    #[test]
    fn test_resolve_ipv6_specific() {
        let specific: SocketAddr = "[::1]:8080".parse().unwrap();
        let resolved = resolve_advertise_address(specific);
        assert_eq!(resolved, specific);
    }
}
