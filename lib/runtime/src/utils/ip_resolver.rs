// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local IP address resolution for advertising endpoints.

use local_ip_address::{Error, list_afinet_netifas, local_ip, local_ipv6};
use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
    sync::OnceLock,
};

const DEFAULT_LOOPBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
static LOCAL_IP_FOR_ADVERTISE: OnceLock<IpAddr> = OnceLock::new();

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

#[derive(Debug, thiserror::Error)]
pub(crate) enum IpResolutionError {
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

#[derive(Default)]
struct AddressCandidates {
    ipv4: Option<Ipv4Addr>,
    ipv6: Option<Ipv6Addr>,
    ipv4_loopback: Option<Ipv4Addr>,
    ipv6_loopback: Option<Ipv6Addr>,
}

impl AddressCandidates {
    fn consider(&mut self, address: IpAddr) {
        if !is_usable(address) {
            return;
        }

        match address {
            IpAddr::V4(address) if address.is_loopback() => {
                self.ipv4_loopback.get_or_insert(address);
            }
            IpAddr::V4(address) => {
                self.ipv4.get_or_insert(address);
            }
            IpAddr::V6(address) if address.is_loopback() => {
                self.ipv6_loopback.get_or_insert(address);
            }
            IpAddr::V6(address) => {
                self.ipv6.get_or_insert(address);
            }
        }
    }

    fn preferred_non_loopback(&self) -> Option<IpAddr> {
        self.ipv4
            .map(IpAddr::V4)
            .or_else(|| self.ipv6.map(IpAddr::V6))
    }

    fn preferred_loopback(&self) -> Option<IpAddr> {
        self.ipv4_loopback
            .map(IpAddr::V4)
            .or_else(|| self.ipv6_loopback.map(IpAddr::V6))
    }

    fn non_loopback_for(&self, family: IpAddr) -> Option<IpAddr> {
        match family {
            IpAddr::V4(_) => self.ipv4.map(IpAddr::V4),
            IpAddr::V6(_) => self.ipv6.map(IpAddr::V6),
        }
    }

    fn loopback_for(&self, family: IpAddr) -> Option<IpAddr> {
        match family {
            IpAddr::V4(_) => self.ipv4_loopback.map(IpAddr::V4),
            IpAddr::V6(_) => self.ipv6_loopback.map(IpAddr::V6),
        }
    }

    fn other_non_loopback(&self, family: IpAddr) -> Option<IpAddr> {
        match family {
            IpAddr::V4(_) => self.ipv6.map(IpAddr::V6),
            IpAddr::V6(_) => self.ipv4.map(IpAddr::V4),
        }
    }
}

fn local_candidates<R: IpResolver>(resolver: &R) -> Result<AddressCandidates, IpResolutionError> {
    let interfaces = resolver
        .list_afinet_netifas()
        .map_err(IpResolutionError::InterfaceEnumeration)?;
    let mut candidates = AddressCandidates::default();
    for (_, address) in interfaces {
        candidates.consider(address);
    }
    Ok(candidates)
}

/// Resolve the preferred local bind and advertisement address from one
/// interface snapshot.
pub(crate) fn resolve_local_host<R: IpResolver>(
    resolver: &R,
) -> Result<ResolvedHost, IpResolutionError> {
    let candidates = local_candidates(resolver)?;
    if let Some(address) = candidates.preferred_non_loopback() {
        return Ok(ResolvedHost::same_address(address));
    }

    Ok(ResolvedHost::loopback_fallback(
        candidates.preferred_loopback().unwrap_or(DEFAULT_LOOPBACK),
    ))
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
            return resolve_wildcard(address, &local_candidates(resolver)?);
        }

        validate_configured_address(address)?;
        return Ok(ResolvedHost::same_address(address));
    }

    let interfaces = resolver
        .list_afinet_netifas()
        .map_err(IpResolutionError::InterfaceEnumeration)?;
    let mut interface_found = false;
    let mut candidates = AddressCandidates::default();

    for (name, address) in interfaces {
        if name != host_or_interface {
            continue;
        }

        interface_found = true;
        candidates.consider(address);
    }

    if !interface_found {
        return Err(IpResolutionError::InterfaceNotFound(
            host_or_interface.to_string(),
        ));
    }

    candidates
        .preferred_non_loopback()
        .or_else(|| candidates.preferred_loopback())
        .map(ResolvedHost::same_address)
        .ok_or_else(|| IpResolutionError::NoUsableInterfaceAddress(host_or_interface.to_string()))
}

fn resolve_wildcard(
    wildcard: IpAddr,
    candidates: &AddressCandidates,
) -> Result<ResolvedHost, IpResolutionError> {
    if let Some(advertise_ip) = candidates.non_loopback_for(wildcard) {
        return Ok(ResolvedHost {
            bind_ip: wildcard,
            advertise_ip,
            used_loopback_fallback: false,
        });
    }

    if let Some(advertise_ip) = candidates.other_non_loopback(wildcard) {
        return Ok(ResolvedHost {
            bind_ip: unspecified_for(advertise_ip),
            advertise_ip,
            used_loopback_fallback: false,
        });
    }

    let advertise_ip = candidates.preferred_loopback().unwrap_or(DEFAULT_LOOPBACK);
    Ok(ResolvedHost {
        bind_ip: unspecified_for(advertise_ip),
        advertise_ip,
        used_loopback_fallback: true,
    })
}

/// Resolve an advertisement address that is served by an existing listener.
/// Wildcard listeners stay in their bound family.
pub(crate) fn resolve_advertise_ip_for_bind<R: IpResolver>(
    bind_ip: IpAddr,
    resolver: &R,
) -> Result<IpAddr, IpResolutionError> {
    if !bind_ip.is_unspecified() {
        validate_configured_address(bind_ip)?;
        return Ok(bind_ip);
    }

    let candidates = local_candidates(resolver)?;
    Ok(candidates
        .non_loopback_for(bind_ip)
        .or_else(|| candidates.loopback_for(bind_ip))
        .unwrap_or_else(|| loopback_for(bind_ip)))
}

fn unspecified_for(address: IpAddr) -> IpAddr {
    match address {
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    }
}

fn loopback_for(address: IpAddr) -> IpAddr {
    match address {
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::LOCALHOST),
    }
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
    value.bytes().filter(|byte| *byte == b':').count() >= 2
        || (value.contains('.')
            && value
                .chars()
                .all(|character| character.is_ascii_digit() || character == '.'))
}

fn validate_configured_address(address: IpAddr) -> Result<(), IpResolutionError> {
    if is_usable(address) {
        Ok(())
    } else {
        Err(IpResolutionError::UnusableAddress(address))
    }
}

fn is_usable(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => {
            !address.is_unspecified() && !address.is_multicast() && !address.is_broadcast()
        }
        IpAddr::V6(address) => {
            !address.is_unspecified() && !address.is_multicast() && !address.is_unicast_link_local()
        }
    }
}

/// Resolve the local IP for advertising endpoints, with loopback fallback.
///
/// IPv6 addresses are bracketed (for example, `[::1]`) so the result is safe
/// to interpolate into a `host:port` URL. Resolution is cached for the process
/// lifetime only after a non-loopback address is found. Enumeration failures
/// and loopback fallbacks are retried on the next call.
pub fn local_ip_for_advertise() -> String {
    cached_local_ip_for_advertise(&LOCAL_IP_FOR_ADVERTISE, &DefaultIpResolver)
}

/// TCP RPC host: `DYN_TCP_RPC_HOST` if set, otherwise the resolved local IP.
pub fn tcp_rpc_host_from_env() -> String {
    std::env::var("DYN_TCP_RPC_HOST").unwrap_or_else(|_| local_ip_for_advertise())
}

fn cached_local_ip_for_advertise<R: IpResolver>(cache: &OnceLock<IpAddr>, resolver: &R) -> String {
    if let Some(address) = cache.get() {
        return format_host(*address);
    }

    match resolve_local_host(resolver) {
        Ok(resolved) if !resolved.used_loopback_fallback() => {
            let address = resolved.advertise_ip();
            if cache.set(address).is_err() {
                return format_host(*cache.get().expect("advertisement cache was initialized"));
            }
            format_host(address)
        }
        Ok(resolved) => {
            let loopback = resolved.advertise_ip();
            tracing::warn!(
                %loopback,
                "No usable non-loopback IP address found; advertising loopback"
            );
            format_host(loopback)
        }
        Err(error) => {
            tracing::warn!(
                %error,
                loopback = %DEFAULT_LOOPBACK,
                "Failed to resolve a usable local IP address; advertising loopback"
            );
            format_host(DEFAULT_LOOPBACK)
        }
    }
}

fn format_host(address: IpAddr) -> String {
    match address {
        IpAddr::V6(_) => format!("[{address}]"),
        IpAddr::V4(_) => address.to_string(),
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::cell::Cell;

    #[derive(Clone, Copy)]
    pub(crate) enum ProbeOutcome {
        NotFound,
        Strategy(&'static str),
        Platform(&'static str),
    }

    impl ProbeOutcome {
        fn result(self) -> Result<IpAddr, Error> {
            match self {
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
        pub(crate) interface_calls: Cell<usize>,
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
                ipv4_calls: Cell::new(0),
                ipv6_calls: Cell::new(0),
                interface_calls: Cell::new(0),
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
            self.interface_calls.set(self.interface_calls.get() + 1);
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
    use ProbeOutcome::{NotFound, Platform, Strategy};

    fn ip(address: &str) -> IpAddr {
        address.parse().unwrap()
    }

    fn interface(name: &'static str, addresses: &[&str]) -> Vec<(&'static str, IpAddr)> {
        addresses
            .iter()
            .map(|address| (name, ip(address)))
            .collect()
    }

    #[test]
    fn ipv6_only_inventory_uses_global_ipv6_without_family_probes() {
        let mut resolver = StubResolver::new(Strategy("IPv4 failed"), NotFound);
        resolver.interfaces = vec![
            ("lo", ip("127.0.0.1")),
            ("lo", ip("::1")),
            ("eth0", ip("2001:db8::5")),
        ];

        let resolved = resolve_local_host(&resolver).unwrap();
        assert_eq!(resolved.bind_ip(), ip("2001:db8::5"));
        assert!(!resolved.used_loopback_fallback());
        assert_eq!(resolver.ipv4_calls.get(), 0);
        assert_eq!(resolver.ipv6_calls.get(), 0);
        assert_eq!(resolver.interface_calls.get(), 1);
    }

    #[test]
    fn configured_literals_are_parsed_and_validated() {
        let resolver = StubResolver::not_found();

        for literal in [
            "192.0.2.10",
            "169.254.1.1",
            "2001:db8::2",
            "[2001:db8::2]",
            "fd00::2",
        ] {
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

        for literal in ["224.0.0.1", "255.255.255.255", "fe80::1", "ff02::1"] {
            let error = resolve_host_or_interface(literal, &resolver).unwrap_err();
            assert!(matches!(error, IpResolutionError::UnusableAddress(_)));
        }
    }

    #[test]
    fn wildcard_switches_bind_family_before_using_loopback() {
        let mut resolver = StubResolver::new(NotFound, NotFound);
        resolver.interfaces = interface("eth0", &["192.0.2.20"]);
        let cases = [
            ("0.0.0.0", ip("0.0.0.0"), ip("192.0.2.20")),
            ("::", ip("0.0.0.0"), ip("192.0.2.20")),
            ("[::]", ip("0.0.0.0"), ip("192.0.2.20")),
        ];

        for (literal, bind_ip, advertise_ip) in cases {
            let resolved = resolve_host_or_interface(literal, &resolver).unwrap();
            assert_eq!(resolved.bind_ip(), bind_ip);
            assert_eq!(resolved.advertise_ip(), advertise_ip);
        }
    }

    #[test]
    fn interface_alias_preserves_order_and_accepts_ipv4_link_local() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = interface("eth0:1", &["2001:db8::2", "169.254.10.5", "192.0.2.10"]);

        let resolved = resolve_host_or_interface("eth0:1", &resolver).unwrap();
        assert_eq!(resolved.advertise_ip(), ip("169.254.10.5"));
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
    fn loopback_only_host_preserves_ipv4_compatibility() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = vec![
            ("lo", IpAddr::V4(Ipv4Addr::LOCALHOST)),
            ("lo", IpAddr::V6(Ipv6Addr::LOCALHOST)),
        ];

        let resolved = resolve_local_host(&resolver).unwrap();
        assert_eq!(resolved.advertise_ip(), DEFAULT_LOOPBACK);
        assert!(resolved.used_loopback_fallback());
    }

    #[test]
    fn bound_wildcard_advertisement_stays_in_bound_family() {
        let mut resolver = StubResolver::not_found();
        resolver.interfaces = vec![
            ("lo", ip("127.0.0.1")),
            ("lo", ip("::1")),
            ("eth0", ip("2001:db8::20")),
        ];

        assert_eq!(
            resolve_advertise_ip_for_bind(ip("0.0.0.0"), &resolver).unwrap(),
            ip("127.0.0.1")
        );
        assert_eq!(
            resolve_advertise_ip_for_bind(ip("::"), &resolver).unwrap(),
            ip("2001:db8::20")
        );
    }

    #[test]
    fn transient_fallbacks_are_not_cached_but_success_is() {
        let cache = OnceLock::new();
        let mut enumeration_error = StubResolver::not_found();
        enumeration_error.interface_error = Some(Platform("transient"));
        assert_eq!(
            cached_local_ip_for_advertise(&cache, &enumeration_error),
            "127.0.0.1"
        );
        assert!(cache.get().is_none());

        let loopback = StubResolver::not_found();
        assert_eq!(
            cached_local_ip_for_advertise(&cache, &loopback),
            "127.0.0.1"
        );
        assert!(cache.get().is_none());

        let mut success = StubResolver::not_found();
        success.interfaces.push(("eth0", ip("2001:db8::10")));
        assert_eq!(
            cached_local_ip_for_advertise(&cache, &success),
            "[2001:db8::10]"
        );
        assert_eq!(cache.get(), Some(&ip("2001:db8::10")));

        assert_eq!(
            cached_local_ip_for_advertise(&cache, &loopback),
            "[2001:db8::10]"
        );
    }

    #[test]
    fn legacy_resolver_implementation_remains_source_compatible() {
        struct LegacyResolver;

        impl IpResolver for LegacyResolver {
            fn local_ip(&self) -> Result<IpAddr, Error> {
                Err(Error::LocalIpAddressNotFound)
            }

            fn local_ipv6(&self) -> Result<IpAddr, Error> {
                Err(Error::LocalIpAddressNotFound)
            }
        }

        fn assert_resolver<T: IpResolver>() {}
        assert_resolver::<LegacyResolver>();
    }
}
