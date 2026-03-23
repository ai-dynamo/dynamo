// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal utilities shared across transport implementations.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

/// Resolve an unspecified bind IP to a routable loopback address.
///
/// When binding to `0.0.0.0` or `::`, the OS accepts connections on all
/// interfaces, but the literal address is not routable for peers. This
/// function maps unspecified addresses to their loopback counterpart so
/// that advertised endpoints contain a connectable address.
///
/// For multi-node deployments the caller should supply a specific bind
/// address rather than relying on this fallback.
pub(crate) fn resolve_advertise_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V4(v4) if v4.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(v6) if v6.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipv4_unspecified_becomes_localhost() {
        assert_eq!(
            resolve_advertise_ip(IpAddr::V4(Ipv4Addr::UNSPECIFIED)),
            IpAddr::V4(Ipv4Addr::LOCALHOST),
        );
    }

    #[test]
    fn test_ipv6_unspecified_becomes_localhost() {
        assert_eq!(
            resolve_advertise_ip(IpAddr::V6(Ipv6Addr::UNSPECIFIED)),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
        );
    }

    #[test]
    fn test_specific_ipv4_unchanged() {
        let ip = IpAddr::V4(Ipv4Addr::new(192, 168, 1, 50));
        assert_eq!(resolve_advertise_ip(ip), ip);
    }

    #[test]
    fn test_specific_ipv6_unchanged() {
        let ip = IpAddr::V6(Ipv6Addr::new(0x2001, 0xdb8, 0, 0, 0, 0, 0, 1));
        assert_eq!(resolve_advertise_ip(ip), ip);
    }

    #[test]
    fn test_loopback_unchanged() {
        assert_eq!(
            resolve_advertise_ip(IpAddr::V4(Ipv4Addr::LOCALHOST)),
            IpAddr::V4(Ipv4Addr::LOCALHOST),
        );
        assert_eq!(
            resolve_advertise_ip(IpAddr::V6(Ipv6Addr::LOCALHOST)),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
        );
    }
}
