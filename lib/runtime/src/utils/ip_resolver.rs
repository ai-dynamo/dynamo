// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local IP address resolution for advertising endpoints.

use local_ip_address::{Error, list_afinet_netifas, local_ip, local_ipv6};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

const DEFAULT_LOOPBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

/// IP address operations used by the runtime.
///
/// This trait allows address resolution and interface enumeration to be
/// controlled in tests.
pub trait IpResolver {
    fn local_ip(&self) -> Result<IpAddr, Error>;
    fn local_ipv6(&self) -> Result<IpAddr, Error>;

    fn list_afinet_netifas(&self) -> Result<Vec<(String, IpAddr)>, Error> {
        list_afinet_netifas()
    }
}

/// The system IP resolver.
pub struct DefaultIpResolver;

impl IpResolver for DefaultIpResolver {
    fn local_ip(&self) -> Result<IpAddr, Error> {
        local_ip()
    }

    fn local_ipv6(&self) -> Result<IpAddr, Error> {
        local_ipv6()
    }
}

/// Resolve a routable local address, trying IPv6 after any IPv4 resolver error.
///
/// A true "not found" result is returned only when both probes report it.
/// Other failures are combined so diagnostics include both probe results.
pub(crate) fn resolve_local_ip<R: IpResolver>(resolver: &R) -> Result<IpAddr, Error> {
    let ipv4_error = match resolver.local_ip() {
        Ok(IpAddr::V4(address)) => return Ok(IpAddr::V4(address)),
        Ok(address) => Error::StrategyError(format!(
            "IPv4 resolution returned an unexpected address family: {address}"
        )),
        Err(error) => error,
    };

    let ipv6_error = match resolver.local_ipv6() {
        Ok(IpAddr::V6(address)) if is_usable_ipv6(address) => return Ok(IpAddr::V6(address)),
        Ok(IpAddr::V6(address)) => unusable_ipv6_error(address),
        Ok(address) => Error::StrategyError(format!(
            "IPv6 resolution returned an unexpected address family: {address}"
        )),
        Err(error) => error,
    };

    if matches!(&ipv4_error, Error::LocalIpAddressNotFound)
        && matches!(&ipv6_error, Error::LocalIpAddressNotFound)
    {
        return Err(Error::LocalIpAddressNotFound);
    }

    Err(Error::StrategyError(format!(
        "IPv4 resolution failed: {ipv4_error}; IPv6 resolution failed: {ipv6_error}"
    )))
}

/// Resolve a configured host value as an IP literal or interface name.
///
/// Named dual-stack interfaces prefer IPv4. Addresses in each family are
/// sorted so selection does not depend on enumeration order.
pub(crate) fn resolve_host_or_interface<R: IpResolver>(
    host_or_interface: &str,
    resolver: &R,
) -> Result<IpAddr, Error> {
    if let Ok(address) = host_or_interface.parse::<IpAddr>() {
        return validate_configured_address(address);
    }

    let interfaces = resolver.list_afinet_netifas()?;
    let mut interface_found = false;
    let mut ipv4_addresses = Vec::new();
    let mut ipv6_addresses = Vec::new();

    for (name, address) in interfaces {
        if name != host_or_interface {
            continue;
        }

        interface_found = true;
        match address {
            IpAddr::V4(address) => ipv4_addresses.push(address),
            IpAddr::V6(address) if is_usable_ipv6(address) => ipv6_addresses.push(address),
            IpAddr::V6(_) => {}
        }
    }

    interface_found
        .then_some(())
        .ok_or_else(|| Error::StrategyError(format!("Interface not found: {host_or_interface}")))?;

    ipv4_addresses.sort_unstable();
    ipv6_addresses.sort_unstable();

    ipv4_addresses
        .first()
        .copied()
        .map(IpAddr::V4)
        .or_else(|| ipv6_addresses.first().copied().map(IpAddr::V6))
        .ok_or_else(|| {
            Error::StrategyError(format!(
                "Interface has no usable IP address: {host_or_interface}"
            ))
        })
}

/// Select the loopback address to use for compatibility fallback.
///
/// Prefer IPv4 when it is present. Use IPv6 loopback on an IPv6-only host.
/// If interface enumeration itself fails, preserve the historical IPv4 value.
pub(crate) fn fallback_loopback<R: IpResolver>(resolver: &R) -> IpAddr {
    let Ok(interfaces) = resolver.list_afinet_netifas() else {
        return DEFAULT_LOOPBACK;
    };

    let has_ipv4_loopback = interfaces
        .iter()
        .any(|(_, address)| *address == IpAddr::V4(Ipv4Addr::LOCALHOST));
    if has_ipv4_loopback {
        return IpAddr::V4(Ipv4Addr::LOCALHOST);
    }

    let has_ipv6_loopback = interfaces
        .iter()
        .any(|(_, address)| *address == IpAddr::V6(Ipv6Addr::LOCALHOST));
    if has_ipv6_loopback {
        return IpAddr::V6(Ipv6Addr::LOCALHOST);
    }

    DEFAULT_LOOPBACK
}

fn validate_configured_address(address: IpAddr) -> Result<IpAddr, Error> {
    match address {
        IpAddr::V4(_) => Ok(address),
        IpAddr::V6(address) if is_usable_ipv6(address) => Ok(IpAddr::V6(address)),
        IpAddr::V6(address) => Err(unusable_ipv6_error(address)),
    }
}

fn is_usable_ipv6(address: Ipv6Addr) -> bool {
    !address.is_unspecified() && !address.is_multicast() && !address.is_unicast_link_local()
}

fn unusable_ipv6_error(address: Ipv6Addr) -> Error {
    Error::StrategyError(format!(
        "IPv6 address is not usable without additional scope information: {address}"
    ))
}

/// Resolve the local IP for advertising endpoints, with loopback fallback.
///
/// IPv6 addresses are bracketed (for example, `[::1]`) so the result is safe
/// to interpolate into a `host:port` URL.
pub fn local_ip_for_advertise() -> String {
    resolve(DefaultIpResolver)
}

/// TCP RPC host: `DYN_TCP_RPC_HOST` if set, otherwise the resolved local IP.
pub fn tcp_rpc_host_from_env() -> String {
    std::env::var("DYN_TCP_RPC_HOST").unwrap_or_else(|_| local_ip_for_advertise())
}

fn resolve<R: IpResolver>(resolver: R) -> String {
    let ip = match resolve_local_ip(&resolver) {
        Ok(ip) => ip,
        Err(error) => {
            let loopback = fallback_loopback(&resolver);
            tracing::warn!(
                %error,
                %loopback,
                "Failed to resolve a routable local IP address; advertising loopback"
            );
            loopback
        }
    };

    match ip {
        IpAddr::V6(_) => format!("[{ip}]"),
        IpAddr::V4(_) => ip.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    #[derive(Clone, Copy)]
    pub(crate) enum ErrorOutcome {
        NotFound,
        Strategy(&'static str),
        Platform(&'static str),
    }

    impl ErrorOutcome {
        fn error(self) -> Error {
            match self {
                Self::NotFound => Error::LocalIpAddressNotFound,
                Self::Strategy(message) => Error::StrategyError(message.to_string()),
                Self::Platform(platform) => Error::PlatformNotSupported(platform.to_string()),
            }
        }
    }

    #[derive(Clone, Copy)]
    pub(crate) enum ProbeOutcome {
        Address(IpAddr),
        Error(ErrorOutcome),
    }

    impl ProbeOutcome {
        fn result(self) -> Result<IpAddr, Error> {
            match self {
                Self::Address(address) => Ok(address),
                Self::Error(error) => Err(error.error()),
            }
        }
    }

    pub(crate) struct StubResolver {
        pub(crate) ipv4: ProbeOutcome,
        pub(crate) ipv6: ProbeOutcome,
        pub(crate) interfaces: Vec<(&'static str, IpAddr)>,
        pub(crate) interface_error: Option<ErrorOutcome>,
    }

    impl StubResolver {
        pub(crate) fn new(ipv4: ProbeOutcome, ipv6: ProbeOutcome) -> Self {
            Self {
                ipv4,
                ipv6,
                interfaces: vec![
                    ("lo", IpAddr::V4(Ipv4Addr::LOCALHOST)),
                    ("lo", IpAddr::V6(Ipv6Addr::LOCALHOST)),
                ],
                interface_error: None,
            }
        }

        pub(crate) fn not_found() -> Self {
            Self::new(
                ProbeOutcome::Error(ErrorOutcome::NotFound),
                ProbeOutcome::Error(ErrorOutcome::NotFound),
            )
        }
    }

    impl IpResolver for StubResolver {
        fn local_ip(&self) -> Result<IpAddr, Error> {
            self.ipv4.result()
        }

        fn local_ipv6(&self) -> Result<IpAddr, Error> {
            self.ipv6.result()
        }

        fn list_afinet_netifas(&self) -> Result<Vec<(String, IpAddr)>, Error> {
            if let Some(error) = self.interface_error {
                return Err(error.error());
            }

            Ok(self
                .interfaces
                .iter()
                .map(|(name, address)| ((*name).to_string(), *address))
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{ErrorOutcome, ProbeOutcome, StubResolver};
    use super::*;

    fn address(address: IpAddr) -> ProbeOutcome {
        ProbeOutcome::Address(address)
    }

    fn error(error: ErrorOutcome) -> ProbeOutcome {
        ProbeOutcome::Error(error)
    }

    #[test]
    fn ipv4_returned_unbracketed() {
        let resolver = StubResolver::new(
            address(IpAddr::from([192, 168, 1, 100])),
            error(ErrorOutcome::NotFound),
        );

        assert_eq!(resolve(resolver), "192.168.1.100");
    }

    #[test]
    fn generic_ipv4_error_falls_back_to_bracketed_ipv6() {
        let resolver = StubResolver::new(
            error(ErrorOutcome::Strategy("IPv4 strategy failed")),
            address("2001:db8::1".parse().unwrap()),
        );

        assert_eq!(resolve(resolver), "[2001:db8::1]");
    }

    #[test]
    fn combined_diagnostic_retains_both_errors() {
        let resolver = StubResolver::new(
            error(ErrorOutcome::Strategy("IPv4 probe failed")),
            error(ErrorOutcome::Platform("test-platform")),
        );

        let result = resolve_local_ip(&resolver).unwrap_err().to_string();
        assert!(result.contains("IPv4 probe failed"), "{result}");
        assert!(result.contains("test-platform"), "{result}");
    }

    #[test]
    fn both_not_found_preserves_not_found_result() {
        assert_eq!(
            resolve_local_ip(&StubResolver::not_found()),
            Err(Error::LocalIpAddressNotFound)
        );
    }

    #[test]
    fn link_local_ipv6_is_rejected() {
        let resolver = StubResolver::new(
            error(ErrorOutcome::NotFound),
            address("fe80::1".parse().unwrap()),
        );

        let result = resolve_local_ip(&resolver).unwrap_err().to_string();
        assert!(result.contains("fe80::1"), "{result}");
        assert!(result.contains("scope information"), "{result}");
    }

    #[test]
    fn explicit_ipv6_literal_is_accepted() {
        let resolver = StubResolver::not_found();
        assert_eq!(
            resolve_host_or_interface("2001:db8::2", &resolver).unwrap(),
            "2001:db8::2".parse::<IpAddr>().unwrap()
        );
        assert_eq!(
            resolve_host_or_interface("fd00::2", &resolver).unwrap(),
            "fd00::2".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn unusable_ipv6_literals_are_rejected() {
        let resolver = StubResolver::not_found();
        for address in ["::", "ff02::1", "fe80::1"] {
            assert!(resolve_host_or_interface(address, &resolver).is_err());
        }
    }

    #[test]
    fn dual_stack_interface_prefers_lowest_ipv4_address() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = vec![
            ("eth0", "2001:db8::2".parse().unwrap()),
            ("eth0", "192.0.2.20".parse().unwrap()),
            ("eth0", "192.0.2.10".parse().unwrap()),
            ("eth0", "2001:db8::1".parse().unwrap()),
        ];

        assert_eq!(
            resolve_host_or_interface("eth0", &resolver).unwrap(),
            "192.0.2.10".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn ipv6_only_interface_selects_lowest_usable_address() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = vec![
            ("eth0", "fe80::1".parse().unwrap()),
            ("eth0", "2001:db8::20".parse().unwrap()),
            ("eth0", "2001:db8::10".parse().unwrap()),
        ];

        assert_eq!(
            resolve_host_or_interface("eth0", &resolver).unwrap(),
            "2001:db8::10".parse::<IpAddr>().unwrap()
        );
    }

    #[test]
    fn interface_errors_are_clear() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = vec![("eth0", "fe80::1".parse().unwrap())];

        let missing = resolve_host_or_interface("missing", &resolver)
            .unwrap_err()
            .to_string();
        assert!(
            missing.contains("Interface not found: missing"),
            "{missing}"
        );

        let unusable = resolve_host_or_interface("eth0", &resolver)
            .unwrap_err()
            .to_string();
        assert!(
            unusable.contains("Interface has no usable IP address: eth0"),
            "{unusable}"
        );
    }

    #[test]
    fn pure_ipv6_host_uses_ipv6_loopback_fallback() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = vec![("lo", IpAddr::V6(Ipv6Addr::LOCALHOST))];

        assert_eq!(
            fallback_loopback(&resolver),
            IpAddr::V6(Ipv6Addr::LOCALHOST)
        );
        assert_eq!(resolve(resolver), "[::1]");
    }

    #[test]
    fn ipv4_loopback_remains_preferred() {
        let resolver = StubResolver::not_found();

        assert_eq!(fallback_loopback(&resolver), DEFAULT_LOOPBACK);
        assert_eq!(resolve(resolver), "127.0.0.1");
    }
}
