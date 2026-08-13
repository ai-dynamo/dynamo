// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local IP address resolution for advertising endpoints.

use local_ip_address::{Error, list_afinet_netifas, local_ip, local_ipv6};
use std::{
    fmt,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    sync::OnceLock,
};

const DEFAULT_LOOPBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
static LOCAL_IP_FOR_ADVERTISE: OnceLock<String> = OnceLock::new();

/// IP address operations used by the runtime.
///
/// This trait allows address resolution and interface enumeration to be
/// controlled in tests.
pub trait IpResolver {
    fn local_ip(&self) -> Result<IpAddr, Error>;
    fn local_ipv6(&self) -> Result<IpAddr, Error>;
    fn list_afinet_netifas(&self) -> Result<Vec<(String, IpAddr)>, Error>;
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

    fn list_afinet_netifas(&self) -> Result<Vec<(String, IpAddr)>, Error> {
        list_afinet_netifas()
    }
}

#[derive(Debug)]
pub(crate) enum ProbeDiagnostic {
    NotFound,
    Unusable(IpAddr),
    Failed(Error),
    UnexpectedFamily {
        expected: &'static str,
        actual: IpAddr,
    },
}

impl ProbeDiagnostic {
    fn is_no_address(&self) -> bool {
        matches!(self, Self::NotFound | Self::Unusable(_))
    }
}

impl fmt::Display for ProbeDiagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(formatter, "no address found"),
            Self::Unusable(address) => write!(formatter, "unusable address {address}"),
            Self::Failed(error) => write!(formatter, "{error}"),
            Self::UnexpectedFamily { expected, actual } => {
                write!(formatter, "expected {expected}, got {actual}")
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum IpResolutionError {
    #[error("no usable local IP address found (IPv4: {ipv4}; IPv6: {ipv6})")]
    NoUsableAddress {
        ipv4: ProbeDiagnostic,
        ipv6: ProbeDiagnostic,
    },

    #[error("local IP address probes failed (IPv4: {ipv4}; IPv6: {ipv6})")]
    ProbeFailure {
        ipv4: ProbeDiagnostic,
        ipv6: ProbeDiagnostic,
    },

    #[error("{family} address probe failed: {diagnostic}")]
    FamilyProbeFailure {
        family: &'static str,
        diagnostic: ProbeDiagnostic,
    },

    #[error("failed to enumerate network interfaces: {0}")]
    InterfaceEnumeration(#[source] Error),

    #[error("interface not found: {0}")]
    InterfaceNotFound(String),

    #[error("interface has no usable IP address: {0}")]
    NoUsableInterfaceAddress(String),

    #[error("IP address is not usable: {0}")]
    UnusableAddress(IpAddr),

    #[error("invalid IP literal: {0}")]
    InvalidLiteral(String),
}

impl IpResolutionError {
    pub(crate) fn is_no_usable_address(&self) -> bool {
        matches!(self, Self::NoUsableAddress { .. })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedHost {
    bind_ip: IpAddr,
    advertise_ip: IpAddr,
    used_loopback_fallback: bool,
}

impl ResolvedHost {
    pub(crate) fn same_address(address: IpAddr) -> Self {
        Self {
            bind_ip: address,
            advertise_ip: address,
            used_loopback_fallback: false,
        }
    }

    pub(crate) fn loopback_fallback(address: IpAddr) -> Self {
        Self {
            bind_ip: address,
            advertise_ip: address,
            used_loopback_fallback: true,
        }
    }

    pub(crate) fn bind_ip(self) -> IpAddr {
        self.bind_ip
    }

    pub(crate) fn advertise_ip(self) -> IpAddr {
        self.advertise_ip
    }

    pub(crate) fn used_loopback_fallback(self) -> bool {
        self.used_loopback_fallback
    }
}

/// Resolve a usable local address, trying IPv6 after any IPv4 probe result.
///
/// A no-address result is returned only when both families are absent or
/// unusable. Other failures retain both probe diagnostics.
pub(crate) fn resolve_local_ip<R: IpResolver>(resolver: &R) -> Result<IpAddr, IpResolutionError> {
    let ipv4 = match probe_ipv4(resolver) {
        Ok(address) => return Ok(IpAddr::V4(address)),
        Err(diagnostic) => diagnostic,
    };

    let ipv6 = match probe_ipv6(resolver) {
        Ok(address) => return Ok(IpAddr::V6(address)),
        Err(diagnostic) => diagnostic,
    };

    if ipv4.is_no_address() && ipv6.is_no_address() {
        Err(IpResolutionError::NoUsableAddress { ipv4, ipv6 })
    } else {
        Err(IpResolutionError::ProbeFailure { ipv4, ipv6 })
    }
}

/// Resolve a configured host value as an IP literal or interface name.
///
/// Named dual-stack interfaces prefer the first usable IPv4 address in OS
/// enumeration order. IPv6 is used when no usable IPv4 address exists.
pub(crate) fn resolve_host_or_interface<R: IpResolver>(
    host_or_interface: &str,
    resolver: &R,
) -> Result<ResolvedHost, IpResolutionError> {
    if let Some(address) = parse_ip_literal(host_or_interface)? {
        if address.is_unspecified() {
            return resolve_wildcard(address, resolver);
        }

        validate_configured_address(address)?;
        return Ok(ResolvedHost::same_address(address));
    }

    let interfaces = resolver
        .list_afinet_netifas()
        .map_err(IpResolutionError::InterfaceEnumeration)?;
    let mut interface_found = false;
    let mut first_ipv4 = None;
    let mut first_ipv6 = None;

    for (name, address) in interfaces {
        if name != host_or_interface {
            continue;
        }

        interface_found = true;
        match address {
            IpAddr::V4(address) if is_usable_ipv4(address) && first_ipv4.is_none() => {
                first_ipv4 = Some(address);
            }
            IpAddr::V6(address) if is_usable_ipv6(address) && first_ipv6.is_none() => {
                first_ipv6 = Some(address);
            }
            _ => {}
        }
    }

    if !interface_found {
        return Err(IpResolutionError::InterfaceNotFound(
            host_or_interface.to_string(),
        ));
    }

    first_ipv4
        .map(IpAddr::V4)
        .or_else(|| first_ipv6.map(IpAddr::V6))
        .map(ResolvedHost::same_address)
        .ok_or_else(|| IpResolutionError::NoUsableInterfaceAddress(host_or_interface.to_string()))
}

/// Select the loopback address to use for compatibility fallback.
///
/// Prefer IPv4 when a non-loopback IPv4 address exists. Otherwise use IPv6
/// loopback on an IPv6-only host. Preserve the historical IPv4 default when
/// neither case can be established.
pub(crate) fn fallback_loopback<R: IpResolver>(resolver: &R) -> IpAddr {
    let Ok(interfaces) = resolver.list_afinet_netifas() else {
        return DEFAULT_LOOPBACK;
    };

    if interfaces
        .iter()
        .any(|(_, address)| matches!(address, IpAddr::V4(address) if !address.is_loopback()))
    {
        return DEFAULT_LOOPBACK;
    }

    if interfaces
        .iter()
        .any(|(_, address)| matches!(address, IpAddr::V6(address) if address.is_loopback()))
    {
        return IpAddr::V6(Ipv6Addr::LOCALHOST);
    }

    DEFAULT_LOOPBACK
}

fn probe_ipv4<R: IpResolver>(resolver: &R) -> Result<Ipv4Addr, ProbeDiagnostic> {
    match resolver.local_ip() {
        Ok(IpAddr::V4(address)) if is_usable_ipv4(address) => Ok(address),
        Ok(IpAddr::V4(address)) => Err(ProbeDiagnostic::Unusable(IpAddr::V4(address))),
        Ok(address) => Err(ProbeDiagnostic::UnexpectedFamily {
            expected: "IPv4",
            actual: address,
        }),
        Err(Error::LocalIpAddressNotFound) => Err(ProbeDiagnostic::NotFound),
        Err(error) => Err(ProbeDiagnostic::Failed(error)),
    }
}

fn probe_ipv6<R: IpResolver>(resolver: &R) -> Result<Ipv6Addr, ProbeDiagnostic> {
    match resolver.local_ipv6() {
        Ok(IpAddr::V6(address)) if is_usable_ipv6(address) => Ok(address),
        Ok(IpAddr::V6(address)) => Err(ProbeDiagnostic::Unusable(IpAddr::V6(address))),
        Ok(address) => Err(ProbeDiagnostic::UnexpectedFamily {
            expected: "IPv6",
            actual: address,
        }),
        Err(Error::LocalIpAddressNotFound) => Err(ProbeDiagnostic::NotFound),
        Err(error) => Err(ProbeDiagnostic::Failed(error)),
    }
}

fn resolve_wildcard<R: IpResolver>(
    wildcard: IpAddr,
    resolver: &R,
) -> Result<ResolvedHost, IpResolutionError> {
    let (advertise_ip, used_loopback_fallback) = match wildcard {
        IpAddr::V4(_) => match probe_ipv4(resolver) {
            Ok(address) => (IpAddr::V4(address), false),
            Err(diagnostic) if diagnostic.is_no_address() => (DEFAULT_LOOPBACK, true),
            Err(diagnostic) => {
                return Err(IpResolutionError::FamilyProbeFailure {
                    family: "IPv4",
                    diagnostic,
                });
            }
        },
        IpAddr::V6(_) => match probe_ipv6(resolver) {
            Ok(address) => (IpAddr::V6(address), false),
            Err(diagnostic) if diagnostic.is_no_address() => {
                (IpAddr::V6(Ipv6Addr::LOCALHOST), true)
            }
            Err(diagnostic) => {
                return Err(IpResolutionError::FamilyProbeFailure {
                    family: "IPv6",
                    diagnostic,
                });
            }
        },
    };

    Ok(ResolvedHost {
        bind_ip: wildcard,
        advertise_ip,
        used_loopback_fallback,
    })
}

fn parse_ip_literal(value: &str) -> Result<Option<IpAddr>, IpResolutionError> {
    let has_open_bracket = value.starts_with('[');
    let has_close_bracket = value.ends_with(']');

    if has_open_bracket || has_close_bracket {
        if !(has_open_bracket && has_close_bracket) {
            return Err(IpResolutionError::InvalidLiteral(value.to_string()));
        }

        let inner = &value[1..value.len() - 1];
        return inner
            .parse::<Ipv6Addr>()
            .map(|address| Some(IpAddr::V6(address)))
            .map_err(|_| IpResolutionError::InvalidLiteral(value.to_string()));
    }

    match value.parse::<IpAddr>() {
        Ok(address) => Ok(Some(address)),
        Err(_) if looks_like_ip_literal(value) => {
            Err(IpResolutionError::InvalidLiteral(value.to_string()))
        }
        Err(_) => Ok(None),
    }
}

fn looks_like_ip_literal(value: &str) -> bool {
    value.contains(':')
        || (value.contains('.')
            && value
                .chars()
                .all(|character| character.is_ascii_digit() || character == '.'))
}

fn validate_configured_address(address: IpAddr) -> Result<(), IpResolutionError> {
    let is_usable = match address {
        IpAddr::V4(address) => is_usable_ipv4(address),
        IpAddr::V6(address) => is_usable_ipv6(address),
    };

    if is_usable {
        Ok(())
    } else {
        Err(IpResolutionError::UnusableAddress(address))
    }
}

fn is_usable_ipv4(address: Ipv4Addr) -> bool {
    !address.is_unspecified()
        && !address.is_multicast()
        && !address.is_link_local()
        && !address.is_broadcast()
}

fn is_usable_ipv6(address: Ipv6Addr) -> bool {
    !address.is_unspecified() && !address.is_multicast() && !address.is_unicast_link_local()
}

/// Resolve the local IP for advertising endpoints, with loopback fallback.
///
/// IPv6 addresses are bracketed (for example, `[::1]`) so the result is safe
/// to interpolate into a `host:port` URL. Resolution is cached for the process
/// lifetime so system probes and fallback warnings occur once.
pub fn local_ip_for_advertise() -> String {
    cached_local_ip_for_advertise(&LOCAL_IP_FOR_ADVERTISE, &DefaultIpResolver)
}

/// TCP RPC host: `DYN_TCP_RPC_HOST` if set, otherwise the resolved local IP.
pub fn tcp_rpc_host_from_env() -> String {
    std::env::var("DYN_TCP_RPC_HOST").unwrap_or_else(|_| local_ip_for_advertise())
}

fn cached_local_ip_for_advertise<R: IpResolver>(cache: &OnceLock<String>, resolver: &R) -> String {
    cache.get_or_init(|| resolve(resolver)).clone()
}

fn resolve<R: IpResolver>(resolver: &R) -> String {
    let ip = match resolve_local_ip(resolver) {
        Ok(ip) => ip,
        Err(error) => {
            let loopback = fallback_loopback(resolver);
            tracing::warn!(
                %error,
                %loopback,
                "Failed to resolve a usable local IP address; advertising loopback"
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
    use std::cell::Cell;

    #[derive(Clone, Copy)]
    pub(crate) enum ProbeOutcome {
        Address(IpAddr),
        NotFound,
        Strategy(&'static str),
        Platform(&'static str),
    }

    impl ProbeOutcome {
        fn result(self) -> Result<IpAddr, Error> {
            match self {
                Self::Address(address) => Ok(address),
                Self::NotFound => Err(Error::LocalIpAddressNotFound),
                Self::Strategy(message) => Err(Error::StrategyError(message.to_string())),
                Self::Platform(platform) => Err(Error::PlatformNotSupported(platform.to_string())),
            }
        }

        fn error(self) -> Error {
            self.result().unwrap_err()
        }
    }

    pub(crate) struct StubResolver {
        pub(crate) ipv4: ProbeOutcome,
        pub(crate) ipv6: ProbeOutcome,
        pub(crate) interfaces: Vec<(&'static str, IpAddr)>,
        pub(crate) interface_error: Option<ProbeOutcome>,
        pub(crate) ipv4_calls: Cell<usize>,
        pub(crate) ipv6_calls: Cell<usize>,
    }

    impl StubResolver {
        pub(crate) fn new(ipv4: ProbeOutcome, ipv6: ProbeOutcome) -> Self {
            Self {
                ipv4,
                ipv6,
                interfaces: vec![
                    ("lo", IpAddr::V4(Ipv4Addr::LOCALHOST)),
                    ("lo", IpAddr::V6(Ipv6Addr::LOCALHOST)),
                    ("eth0", IpAddr::V4(Ipv4Addr::new(192, 0, 2, 10))),
                ],
                interface_error: None,
                ipv4_calls: Cell::new(0),
                ipv6_calls: Cell::new(0),
            }
        }

        pub(crate) fn not_found() -> Self {
            Self::new(ProbeOutcome::NotFound, ProbeOutcome::NotFound)
        }
    }

    impl IpResolver for StubResolver {
        fn local_ip(&self) -> Result<IpAddr, Error> {
            self.ipv4_calls.set(self.ipv4_calls.get() + 1);
            self.ipv4.result()
        }

        fn local_ipv6(&self) -> Result<IpAddr, Error> {
            self.ipv6_calls.set(self.ipv6_calls.get() + 1);
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
    use super::test_support::{ProbeOutcome, StubResolver};
    use super::*;
    use ProbeOutcome::{Address, NotFound, Platform, Strategy};

    fn ip(address: &str) -> IpAddr {
        address.parse().unwrap()
    }

    fn eth0(addresses: &[&str]) -> Vec<(&'static str, IpAddr)> {
        addresses
            .iter()
            .map(|address| ("eth0", ip(address)))
            .collect()
    }

    #[test]
    fn advertisement_formats_ipv4_and_ipv6() {
        let ipv4 = StubResolver::new(Address(ip("192.168.1.100")), NotFound);
        assert_eq!(resolve(&ipv4), "192.168.1.100");

        let ipv6 = StubResolver::new(NotFound, Address(ip("2001:db8::1")));
        assert_eq!(resolve(&ipv6), "[2001:db8::1]");
    }

    #[test]
    fn ipv4_not_found_or_failure_uses_usable_ipv6() {
        for ipv4 in [NotFound, Strategy("IPv4 strategy failed")] {
            let resolver = StubResolver::new(ipv4, Address(ip("2001:db8::1")));
            assert_eq!(resolve_local_ip(&resolver).unwrap(), ip("2001:db8::1"));
        }
    }

    #[test]
    fn probe_failures_retain_both_diagnostics() {
        let resolver = StubResolver::new(Strategy("IPv4 probe failed"), Platform("test-platform"));

        let result = resolve_local_ip(&resolver).unwrap_err().to_string();
        assert!(result.contains("IPv4 probe failed"), "{result}");
        assert!(result.contains("test-platform"), "{result}");
    }

    #[test]
    fn absent_and_unusable_results_are_no_address() {
        let assert_no_address = |resolver: StubResolver, expected_diagnostic: &str| {
            let error = resolve_local_ip(&resolver).unwrap_err();
            assert!(error.is_no_usable_address(), "{error}");
            assert!(error.to_string().contains(expected_diagnostic), "{error}");
        };

        assert_no_address(
            StubResolver::new(NotFound, Address(ip("fe80::1"))),
            "fe80::1",
        );
        assert_no_address(
            StubResolver::new(Address(ip("0.0.0.0")), NotFound),
            "0.0.0.0",
        );
    }

    #[test]
    fn configured_literals_are_parsed_and_validated() {
        let resolver = StubResolver::not_found();

        for literal in ["192.0.2.10", "2001:db8::2", "[2001:db8::2]", "fd00::2"] {
            let resolved = resolve_host_or_interface(literal, &resolver).unwrap();
            assert_eq!(
                resolved.bind_ip(),
                literal.trim_matches(['[', ']']).parse::<IpAddr>().unwrap()
            );
            assert_eq!(resolved.advertise_ip(), resolved.bind_ip());
        }

        for literal in ["[::1", "::1]", "[not-an-ip]", "192.0.2.999"] {
            let error = resolve_host_or_interface(literal, &resolver)
                .unwrap_err()
                .to_string();
            assert!(error.contains("invalid IP literal"), "{error}");
        }

        for literal in [
            "169.254.1.1",
            "224.0.0.1",
            "255.255.255.255",
            "fe80::1",
            "ff02::1",
        ] {
            let error = resolve_host_or_interface(literal, &resolver).unwrap_err();
            assert!(matches!(error, IpResolutionError::UnusableAddress(_)));
        }
    }

    #[test]
    fn wildcards_keep_bind_family_and_advertise_concrete_address() {
        let resolver = StubResolver::new(Address(ip("192.0.2.20")), Address(ip("2001:db8::20")));
        let cases = [
            ("0.0.0.0", ip("0.0.0.0"), ip("192.0.2.20")),
            ("::", ip("::"), ip("2001:db8::20")),
            ("[::]", ip("::"), ip("2001:db8::20")),
        ];

        for (literal, bind_ip, advertise_ip) in cases {
            let resolved = resolve_host_or_interface(literal, &resolver).unwrap();
            assert_eq!(resolved.bind_ip(), bind_ip);
            assert_eq!(resolved.advertise_ip(), advertise_ip);
        }
    }

    #[test]
    fn interface_selection_preserves_order_and_filters_unusable_addresses() {
        let mut resolver = StubResolver::not_found();
        let cases = [
            (
                &["2001:db8::2", "192.0.2.20", "192.0.2.10"][..],
                ip("192.0.2.20"),
            ),
            (&["169.254.10.5", "2001:db8::10"][..], ip("2001:db8::10")),
            (
                &["fe80::1", "2001:db8::20", "2001:db8::10"][..],
                ip("2001:db8::20"),
            ),
        ];

        for (addresses, expected) in cases {
            resolver.interfaces = eth0(addresses);
            let resolved = resolve_host_or_interface("eth0", &resolver).unwrap();
            assert_eq!(resolved.advertise_ip(), expected);
        }
    }

    #[test]
    fn interface_errors_retain_context() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = vec![("eth0", ip("fe80::1"))];

        for (name, expected) in [
            ("missing", "interface not found: missing"),
            ("eth0", "interface has no usable IP address: eth0"),
        ] {
            let error = resolve_host_or_interface(name, &resolver)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error}");
        }

        resolver.interface_error = Some(Platform("test-platform"));
        let error = resolve_host_or_interface("eth0", &resolver)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("failed to enumerate network interfaces"),
            "{error}"
        );
        assert!(error.contains("test-platform"), "{error}");
    }

    #[test]
    fn loopback_selection_matches_host_capability() {
        let mut resolver = StubResolver::not_found();
        assert_eq!(fallback_loopback(&resolver), DEFAULT_LOOPBACK);

        resolver.interfaces = vec![
            ("lo", IpAddr::V4(Ipv4Addr::LOCALHOST)),
            ("lo", IpAddr::V6(Ipv6Addr::LOCALHOST)),
            ("eth0", ip("2001:db8::10")),
        ];
        assert_eq!(
            fallback_loopback(&resolver),
            IpAddr::V6(Ipv6Addr::LOCALHOST)
        );
        assert_eq!(resolve(&resolver), "[::1]");

        resolver.interface_error = Some(Platform("test-platform"));
        assert_eq!(fallback_loopback(&resolver), DEFAULT_LOOPBACK);
    }

    #[test]
    fn advertised_ip_resolution_is_cached() {
        let cache = OnceLock::new();
        let resolver = StubResolver::new(NotFound, Address(ip("2001:db8::10")));

        for _ in 0..2 {
            assert_eq!(
                cached_local_ip_for_advertise(&cache, &resolver),
                "[2001:db8::10]"
            );
        }
        assert_eq!(resolver.ipv4_calls.get(), 1);
        assert_eq!(resolver.ipv6_calls.get(), 1);
    }
}
