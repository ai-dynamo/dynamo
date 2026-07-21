// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local IP address resolution for advertising endpoints.

use anyhow::{Result, bail};
use local_ip_address::{Error, list_afinet_netifas, local_ip, local_ipv6};
use std::net::{IpAddr, Ipv4Addr};

const FALLBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// IP address resolution interface used by networking components and tests.
pub trait IpResolver {
    fn local_ip(&self) -> Result<IpAddr, Error>;
    fn local_ipv6(&self) -> Result<IpAddr, Error>;

    fn list_afinet_netifas(&self) -> Result<Vec<(String, IpAddr)>, Error> {
        list_afinet_netifas()
    }
}

/// Default resolver backed by the local network interfaces.
pub struct DefaultIpResolver;

impl IpResolver for DefaultIpResolver {
    fn local_ip(&self) -> Result<IpAddr, Error> {
        local_ip()
    }

    fn local_ipv6(&self) -> Result<IpAddr, Error> {
        local_ipv6()
    }
}

/// Resolve the local IP for advertising endpoints, falling back to 127.0.0.1.
///
/// IPv6 addresses are bracketed (e.g. `[::1]`) so the result is safe to
/// interpolate into a `host:port` URL.
pub fn local_ip_for_advertise() -> String {
    resolve(DefaultIpResolver)
}

/// TCP RPC host: `DYN_TCP_RPC_HOST` if set, otherwise the resolved local IP.
pub fn tcp_rpc_host_from_env() -> String {
    std::env::var("DYN_TCP_RPC_HOST").unwrap_or_else(|_| local_ip_for_advertise())
}

/// Resolve an explicit advertised host as a dialable IPv4 address.
///
/// The input can be an IPv4 literal or a network interface name. Interfaces
/// with multiple IPv4 addresses select the lowest address deterministically.
pub(crate) fn resolve_advertised_ipv4<R: IpResolver>(
    host_or_interface: &str,
    resolver: &R,
) -> Result<Ipv4Addr> {
    if let Ok(ip) = host_or_interface.parse::<IpAddr>() {
        return match ip {
            IpAddr::V4(ip) if ip.is_unspecified() => {
                bail!("unspecified IPv4 addresses cannot be advertised")
            }
            IpAddr::V4(ip) => Ok(ip),
            IpAddr::V6(_) => bail!("IPv6 addresses are not supported for advertised hosts"),
        };
    }

    let interfaces = resolver.list_afinet_netifas()?;
    let interface_found = interfaces.iter().any(|(name, _)| name == host_or_interface);
    let mut ipv4_addresses: Vec<_> = interfaces
        .into_iter()
        .filter_map(|(name, ip)| match ip {
            IpAddr::V4(ip) if name == host_or_interface && !ip.is_unspecified() => Some(ip),
            _ => None,
        })
        .collect();
    ipv4_addresses.sort_unstable();

    ipv4_addresses.into_iter().next().ok_or_else(|| {
        if interface_found {
            anyhow::anyhow!("Interface has no usable IPv4 address: {host_or_interface}")
        } else {
            anyhow::anyhow!("Interface not found: {host_or_interface}")
        }
    })
}

/// Resolve the IPv4 address used by IPv4-only advertised transports.
pub(crate) fn local_ipv4_for_advertise<R: IpResolver>(resolver: &R) -> Ipv4Addr {
    match resolver.local_ip() {
        Ok(IpAddr::V4(ip)) if !ip.is_unspecified() => ip,
        _ => Ipv4Addr::LOCALHOST,
    }
}

fn resolve<R: IpResolver>(resolver: R) -> String {
    let ip = resolver
        .local_ip()
        .or_else(|err| match err {
            Error::LocalIpAddressNotFound => resolver.local_ipv6(),
            _ => Err(err),
        })
        .unwrap_or(FALLBACK);

    match ip {
        IpAddr::V6(_) => format!("[{ip}]"),
        IpAddr::V4(_) => ip.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockIpResolver {
        v4: Result<IpAddr, Error>,
        v6: Result<IpAddr, Error>,
        interfaces: Vec<(String, IpAddr)>,
    }

    impl IpResolver for MockIpResolver {
        fn local_ip(&self) -> Result<IpAddr, Error> {
            self.v4
                .as_ref()
                .copied()
                .map_err(|_| Error::LocalIpAddressNotFound)
        }

        fn local_ipv6(&self) -> Result<IpAddr, Error> {
            self.v6
                .as_ref()
                .copied()
                .map_err(|_| Error::LocalIpAddressNotFound)
        }

        fn list_afinet_netifas(&self) -> Result<Vec<(String, IpAddr)>, Error> {
            Ok(self.interfaces.clone())
        }
    }

    #[test]
    fn ipv4_returned_unbracketed() {
        let r = MockIpResolver {
            v4: Ok(IpAddr::from([192, 168, 1, 100])),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: Vec::new(),
        };
        assert_eq!(resolve(r), "192.168.1.100");
    }

    #[test]
    fn ipv6_fallback_is_bracketed() {
        let r = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Ok(IpAddr::from([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1])),
            interfaces: Vec::new(),
        };
        assert_eq!(resolve(r), "[2001:db8::1]");
    }

    #[test]
    fn both_fail_uses_localhost() {
        let r = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: Vec::new(),
        };
        assert_eq!(resolve(r), "127.0.0.1");
    }

    #[test]
    fn explicit_ipv4_literal_is_used_directly() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: Vec::new(),
        };

        assert_eq!(
            resolve_advertised_ipv4("172.16.0.87", &resolver).unwrap(),
            Ipv4Addr::new(172, 16, 0, 87)
        );
    }

    #[test]
    fn explicit_host_rejects_ipv6_and_unspecified_addresses() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: Vec::new(),
        };

        assert_eq!(
            resolve_advertised_ipv4("2001:db8::1", &resolver)
                .unwrap_err()
                .to_string(),
            "IPv6 addresses are not supported for advertised hosts"
        );
        assert_eq!(
            resolve_advertised_ipv4("0.0.0.0", &resolver)
                .unwrap_err()
                .to_string(),
            "unspecified IPv4 addresses cannot be advertised"
        );
    }

    #[test]
    fn interface_selects_ipv4_deterministically() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: vec![
                ("ib0".to_string(), "fe80::1".parse().unwrap()),
                ("ib0".to_string(), "172.16.96.87".parse().unwrap()),
                ("eth0".to_string(), "10.52.1.2".parse().unwrap()),
                ("ib0".to_string(), "0.0.0.0".parse().unwrap()),
                ("ib0".to_string(), "172.16.0.87".parse().unwrap()),
            ],
        };

        assert_eq!(
            resolve_advertised_ipv4("ib0", &resolver).unwrap(),
            Ipv4Addr::new(172, 16, 0, 87)
        );
    }

    #[test]
    fn interface_requires_usable_ipv4_address() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: vec![
                ("ib0".to_string(), "fe80::1".parse().unwrap()),
                ("ib0".to_string(), "0.0.0.0".parse().unwrap()),
            ],
        };

        assert_eq!(
            resolve_advertised_ipv4("ib0", &resolver)
                .unwrap_err()
                .to_string(),
            "Interface has no usable IPv4 address: ib0"
        );
        assert_eq!(
            resolve_advertised_ipv4("missing0", &resolver)
                .unwrap_err()
                .to_string(),
            "Interface not found: missing0"
        );
    }

    #[test]
    fn ipv4_only_advertise_falls_back_from_ipv6() {
        let resolver = MockIpResolver {
            v4: Ok(IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
            v6: Ok(IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)),
            interfaces: Vec::new(),
        };

        assert_eq!(local_ipv4_for_advertise(&resolver), Ipv4Addr::LOCALHOST);
    }
}
