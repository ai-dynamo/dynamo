// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Local IP address resolution for advertising endpoints.

use crate::pipeline::network::tcp::server::{DefaultIpResolver, IpResolver, resolve_local_ip};
use std::net::{IpAddr, Ipv4Addr};

const FALLBACK: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);

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

fn resolve<R: IpResolver>(resolver: R) -> String {
    let ip = resolve_local_ip(&resolver).unwrap_or(FALLBACK);

    match ip {
        IpAddr::V6(_) => format!("[{ip}]"),
        IpAddr::V4(_) => ip.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use local_ip_address::Error;

    struct MockIpResolver {
        v4: Result<IpAddr, Error>,
        v6: Result<IpAddr, Error>,
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
    }

    struct StrategyErrorThenIpv6Resolver;

    impl IpResolver for StrategyErrorThenIpv6Resolver {
        fn local_ip(&self) -> Result<IpAddr, Error> {
            Err(Error::StrategyError(
                "ifa_prefixlen must be initialized".to_string(),
            ))
        }

        fn local_ipv6(&self) -> Result<IpAddr, Error> {
            Ok(IpAddr::from([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1]))
        }
    }

    #[test]
    fn ipv4_returned_unbracketed() {
        let r = MockIpResolver {
            v4: Ok(IpAddr::from([192, 168, 1, 100])),
            v6: Err(Error::LocalIpAddressNotFound),
        };
        assert_eq!(resolve(r), "192.168.1.100");
    }

    #[test]
    fn ipv6_fallback_is_bracketed() {
        let r = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Ok(IpAddr::from([0x2001, 0xdb8, 0, 0, 0, 0, 0, 1])),
        };
        assert_eq!(resolve(r), "[2001:db8::1]");
    }

    #[test]
    fn strategy_error_falls_back_to_bracketed_ipv6() {
        assert_eq!(resolve(StrategyErrorThenIpv6Resolver), "[2001:db8::1]");
    }

    #[test]
    fn both_fail_uses_localhost() {
        let r = MockIpResolver {
            v4: Err(Error::LocalIpAddressNotFound),
            v6: Err(Error::LocalIpAddressNotFound),
        };
        assert_eq!(resolve(r), "127.0.0.1");
    }
}
