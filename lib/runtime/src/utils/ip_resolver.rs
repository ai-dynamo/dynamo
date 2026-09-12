// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local IP address resolution for advertising endpoints.

use anyhow::{Result, bail};
use local_ip_address::{Error, list_afinet_netifas, local_ip, local_ipv6};
use std::{ffi::OsString, net::IpAddr};

const FALLBACK: IpAddr = IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);

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

/// Read and normalize a host override from the environment.
pub(crate) fn host_override_from_env(name: &str) -> Result<Option<String>> {
    host_override_from_lookup(name, |key| std::env::var_os(key))
}

fn host_override_from_lookup(
    name: &str,
    mut get_env: impl FnMut(&str) -> Option<OsString>,
) -> Result<Option<String>> {
    let Some(value) = get_env(name) else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} must contain valid Unicode"))?;
    let value = value.trim();
    Ok((!value.is_empty()).then(|| value.to_string()))
}

/// Decide whether an interface address can be used to bind and advertise.
///
/// Unspecified, broadcast, multicast, and link-local addresses are skipped
/// because binding to them fails or advertises an unreachable endpoint.
/// Loopback is intentionally allowed on both families so interfaces like `lo`
/// keep working for local-only setups.
fn is_usable_unicast(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            !v4.is_unspecified() && !v4.is_broadcast() && !v4.is_multicast() && !v4.is_link_local()
        }
        IpAddr::V6(v6) => !v6.is_unspecified() && !v6.is_multicast() && !v6.is_unicast_link_local(),
    }
}

/// Resolve a configured host as an IP literal or exact network interface name.
///
/// IP literals are used as-is. Interface lookup skips link-local, multicast,
/// broadcast, and unspecified addresses, and among the remaining usable
/// addresses the last enumerated one wins (the existing TCP response-stream
/// tie-break).
pub(crate) fn resolve_host_or_interface<R: IpResolver>(
    host_or_interface: &str,
    resolver: &R,
) -> Result<IpAddr> {
    let host_or_interface = host_or_interface.trim();
    let ip = match host_or_interface.parse::<IpAddr>() {
        Ok(ip) => ip,
        Err(_) => {
            let mut seen = false;
            let mut selected = None;
            for (name, candidate) in resolver.list_afinet_netifas()? {
                if name != host_or_interface {
                    continue;
                }
                seen = true;
                if is_usable_unicast(&candidate) {
                    selected = Some(candidate);
                }
            }
            match selected {
                Some(ip) => ip,
                None if seen => bail!(
                    "interface '{host_or_interface}' has no usable unicast address (link-local, multicast, broadcast, and unspecified addresses are skipped)"
                ),
                None => bail!(
                    "'{host_or_interface}' is not a valid IP address and no network interface with that name was found"
                ),
            }
        }
    };
    if ip.to_canonical().is_unspecified() {
        bail!("unspecified IP addresses cannot be advertised");
    }
    Ok(ip)
}

pub(crate) fn resolve_local_host<R: IpResolver>(resolver: &R) -> IpAddr {
    resolver
        .local_ip()
        .or_else(|err| match err {
            Error::LocalIpAddressNotFound => resolver.local_ipv6(),
            _ => Err(err),
        })
        .unwrap_or(FALLBACK)
}

fn resolve<R: IpResolver>(resolver: R) -> String {
    let ip = resolve_local_host(&resolver);

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
    fn explicit_ip_literals_are_used_directly_and_trimmed() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: Vec::new(),
        };

        assert_eq!(
            resolve_host_or_interface(" 172.16.0.87 ", &resolver).unwrap(),
            "172.16.0.87".parse::<IpAddr>().unwrap()
        );
        assert_eq!(
            resolve_host_or_interface("2001:db8::1", &resolver).unwrap(),
            "2001:db8::1".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn explicit_host_rejects_unspecified_addresses() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: vec![("wildcard0".to_string(), "::ffff:0.0.0.0".parse().unwrap())],
        };

        for host in ["0.0.0.0", "::", "::ffff:0.0.0.0", "wildcard0"] {
            assert_eq!(
                resolve_host_or_interface(host, &resolver)
                    .unwrap_err()
                    .to_string(),
                "unspecified IP addresses cannot be advertised"
            );
        }
    }

    #[test]
    fn exact_interface_name_supports_ipv4_and_ipv6() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: vec![
                ("ib0".to_string(), "172.16.0.87".parse().unwrap()),
                ("eth0".to_string(), "2001:db8::20".parse().unwrap()),
            ],
        };

        assert_eq!(
            resolve_host_or_interface(" ib0 ", &resolver).unwrap(),
            "172.16.0.87".parse::<IpAddr>().unwrap()
        );
        assert_eq!(
            resolve_host_or_interface("eth0", &resolver).unwrap(),
            "2001:db8::20".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn interface_lookup_preserves_existing_last_address_behavior() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: vec![
                ("ib0".to_string(), "172.16.0.10".parse().unwrap()),
                ("ib0".to_string(), "172.16.0.20".parse().unwrap()),
            ],
        };

        assert_eq!(
            resolve_host_or_interface("ib0", &resolver).unwrap(),
            "172.16.0.20".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn missing_interface_reports_the_accepted_input_forms() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: Vec::new(),
        };

        assert_eq!(
            resolve_host_or_interface("missing0", &resolver)
                .unwrap_err()
                .to_string(),
            "'missing0' is not a valid IP address and no network interface with that name was found"
        );
    }

    #[test]
    fn interface_lookup_skips_link_local_addresses() {
        for interfaces in [
            vec![
                ("eth0".to_string(), "fe80::1".parse().unwrap()),
                ("eth0".to_string(), "169.254.10.10".parse().unwrap()),
                ("eth0".to_string(), "192.0.2.10".parse().unwrap()),
            ],
            vec![
                ("eth0".to_string(), "192.0.2.10".parse().unwrap()),
                ("eth0".to_string(), "169.254.10.10".parse().unwrap()),
                ("eth0".to_string(), "fe80::1".parse().unwrap()),
            ],
        ] {
            let resolver = MockIpResolver {
                v4: Err(Error::LocalIpAddressNotFound),
                v6: Err(Error::LocalIpAddressNotFound),
                interfaces,
            };
            assert_eq!(
                resolve_host_or_interface("eth0", &resolver).unwrap(),
                "192.0.2.10".parse::<IpAddr>().unwrap()
            );
        }
    }

    #[test]
    fn interface_lookup_skips_multicast_and_broadcast() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: vec![
                ("eth0".to_string(), "224.0.0.1".parse().unwrap()),
                ("eth0".to_string(), "255.255.255.255".parse().unwrap()),
                ("eth0".to_string(), "ff02::1".parse().unwrap()),
                ("eth0".to_string(), "192.0.2.20".parse().unwrap()),
            ],
        };

        assert_eq!(
            resolve_host_or_interface("eth0", &resolver).unwrap(),
            "192.0.2.20".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn interface_with_only_unusable_addresses_is_rejected() {
        let resolver = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
            interfaces: vec![("eth0".to_string(), "fe80::1".parse().unwrap())],
        };

        assert_eq!(
            resolve_host_or_interface("eth0", &resolver)
                .unwrap_err()
                .to_string(),
            "interface 'eth0' has no usable unicast address (link-local, multicast, broadcast, and unspecified addresses are skipped)"
        );
    }

    #[test]
    fn loopback_interface_addresses_remain_usable() {
        for (interfaces, expected) in [
            (
                vec![("lo".to_string(), "127.0.0.1".parse().unwrap())],
                "127.0.0.1",
            ),
            (
                vec![
                    ("lo".to_string(), "127.0.0.1".parse().unwrap()),
                    ("lo".to_string(), "::1".parse().unwrap()),
                ],
                "::1",
            ),
        ] {
            let resolver = MockIpResolver {
                v4: Err(Error::LocalIpAddressNotFound),
                v6: Err(Error::LocalIpAddressNotFound),
                interfaces,
            };

            assert_eq!(
                resolve_host_or_interface("lo", &resolver).unwrap(),
                expected.parse::<IpAddr>().unwrap()
            );
        }
    }

    #[test]
    fn host_override_is_trimmed_and_empty_is_unset() {
        assert_eq!(
            host_override_from_lookup("TEST_HOST", |_| Some(OsString::from(" ib0 "))).unwrap(),
            Some("ib0".to_string())
        );
        assert_eq!(
            host_override_from_lookup("TEST_HOST", |_| Some(OsString::from(" \t"))).unwrap(),
            None
        );
        assert_eq!(
            host_override_from_lookup("TEST_HOST", |_| None).unwrap(),
            None
        );
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_host_override_is_rejected() {
        use std::os::unix::ffi::OsStringExt;

        let error =
            host_override_from_lookup("TEST_HOST", |_| Some(OsString::from_vec(vec![0xff])))
                .unwrap_err();
        assert_eq!(error.to_string(), "TEST_HOST must contain valid Unicode");
    }
}
